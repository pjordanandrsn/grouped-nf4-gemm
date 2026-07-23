#!/usr/bin/env python3
"""Canonical reducer for flagship A/B runs (Phase-A synthetic + Phase-B real).

Replaces the two ad-hoc reducers that were previously inlined in lane payload
scripts. Written after the 2026-07-22 gen5 bare-metal run exposed three
reducer defects (see RESULTS-flagship-gen5-metal.md, J1/J2):

  1. Key mismatch: the summary read `tok_per_s` (harness writes `toks_per_s`)
     with a `waterfall_toks` fallback that printed Phase-B's CEILING as if it
     were the achieved rate, and printed `None` for both Phase-A cells while
     the driver declared "evidence complete".
  2. RAPL wraparound: per-cell CPU energy was `last - first` on a cumulative
     counter. On the gen5 host (AMD Zen, 2^16 J range) the run wrapped 5
     times; one cell printed a negative joule figure and another was silently
     6.5x under-reported.
  3. Single energy figure: only window-mean GPU W was reported; idle was not
     subtracted and setup/download time inside a cell window diluted the mean.

Rules encoded here (the CELL-VOID rule = AB-VOID applied one layer down):
  - A cell that produces no parseable metric emits `CELL-VOID <file> reason=…`
    and the process exits 3. The summary never prints `None` as a value.
  - RAPL counters are wrap-corrected. The counter range comes from
    `rapl_meta.txt` (written by the sampler at run start) when present, else
    it is inferred from the wrap events themselves and labeled INFERRED.
  - A package column that was never readable is labeled ABSENT (a zero that
    means "no such socket" and a zero that means "read failed" are different
    facts; the sampler records which at run start).
  - Every decode cell reports BOTH energy figures, labeled: absolute
    (window-mean W / rate — an upper bound that includes idle draw) and
    marginal over idle ((meanW - idle_meanW) / rate). Neither alone.
  - Cells whose window includes setup/download also report a decode-window
    figure (flagA: trailing decode seconds; flagB: trailing contiguous
    above-idle-power window), labeled as a post-hoc window.

Usage: ab_reduce.py <ab-out dir>   (writes SUMMARY_reduced.txt and
ENERGY_reduced.txt into the dir, and prints both to stdout)
"""
import json
import math
import sys
from pathlib import Path

VOID = []


def _load(p: Path):
    try:
        d = json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001 - reason is reported, run is voided
        VOID.append((p.name, f"unparseable-json: {e}"))
        return None
    if not isinstance(d, dict):  # literal null/list/scalar parses fine but is no cell
        VOID.append((p.name, f"not-a-json-object: {type(d).__name__}"))
        return None
    return d


def summarize_cell(p: Path):
    d = _load(p)
    if d is None:
        reason = next((r for n, r in reversed(VOID) if n == p.name), "unparseable-json")
        return f"CELL-VOID {p.name} reason={reason}"
    if "results" in d:  # Phase-B real-checkpoint schema
        if d.get("c_box_probe"):  # no-stream floor probe: fraction is n/a by design
            cbs = sorted(r["c_box_ms"] for r in (d.get("results") or []) if r.get("c_box_ms"))
            if not cbs:
                VOID.append((p.name, "c_box_probe with no c_box_ms rows"))
                return f"CELL-VOID {p.name} reason=c_box_probe-no-rows"
            return (f"{p.name} -> C-BOX-PROBE c_box = {cbs[len(cbs) // 2]:.1f} ms/token "
                    f"(median across prompts, stream disabled)")
        offs = sorted(r["toks_per_s_off"] for r in (d.get("results") or [])
                      if r.get("toks_per_s_off"))
        if not offs:
            VOID.append((p.name, "no toks_per_s_off rows"))
            return f"CELL-VOID {p.name} reason=no-toks_per_s_off-rows"
        med = offs[len(offs) // 2]
        wf = d.get("waterfall_toks")
        frac = f" ({med / wf:.3f}x of waterfall {wf:.3f})" if wf else ""
        return f"{p.name} -> achieved {med:.3f} tok/s (median off-mode){frac}"
    if "toks_per_s" in d:  # Phase-A synthetic schema
        if "c_box_ms" in d:  # no-stream floor probe: fraction is n/a by design
            return (f"{p.name} -> C-BOX-PROBE c_box = {d['c_box_ms']:.1f} ms/token "
                    f"[moe={d.get('config', {}).get('moe', '?')}, no-stream]")
        wf = d.get("waterfall_ceiling_toks")
        frac = f" ({d['toks_per_s'] / wf:.3f}x of waterfall {wf:.3f})" if wf else ""
        return (f"{p.name} -> achieved {d['toks_per_s']:.3f} tok/s "
                f"[moe={d.get('config', {}).get('moe', '?')}]{frac}")
    VOID.append((p.name, "no recognized rate key"))
    return f"CELL-VOID {p.name} reason=no-recognized-rate-key (keys: {sorted(d)[:8]})"


def read_samples(out: Path):
    samp, width = [], None
    for ln in (out / "power.tsv").read_text().splitlines():
        parts = ln.split()
        if len(parts) < 3:
            continue
        try:
            t = float(parts[0])
            w = float(parts[1]) if parts[1].replace(".", "", 1).isdigit() else 0.0
            counters = [int(x) for x in parts[2:]]
        except ValueError:
            continue
        if width is None:
            width = len(counters)
        if len(counters) != width:  # ragged row (torn write) — drop, keep width stable
            continue
        samp.append((t, w, counters))
    return samp


def rapl_meta(out: Path, samp):
    """Return ({pkg_index: range_uj_or_None}, {pkg_index: 'meta'|'inferred'|'naive'}, notes)."""
    npkg = max((len(s[2]) for s in samp), default=0)
    ranges, source, notes = {}, {}, []
    meta = out / "rapl_meta.txt"
    if meta.exists():
        for ln in meta.read_text().splitlines():
            # format: "pkg<N> present=<0|1> max_energy_range_uj=<int|na>"
            parts = ln.split()
            if not parts or not parts[0].startswith("pkg"):
                continue  # blank/foreign line — meta is advisory, never crash on it
            try:
                i = int(parts[0].removeprefix("pkg"))
            except ValueError:
                continue
            f = dict(kv.split("=", 1) for kv in parts[1:] if "=" in kv)
            if f.get("present") == "0":
                ranges[i] = "ABSENT"
                source[i] = "meta"
            else:
                r = f.get("max_energy_range_uj", "na")
                ranges[i] = int(r) if r.isdigit() else None
                source[i] = "meta"
        for i in list(ranges):  # meta may declare packages the sampler never wrote
            if not isinstance(ranges[i], str) and i >= npkg:
                ranges[i] = "NOT-SAMPLED"
                source[i] = "meta"
    for i in range(npkg):
        if i in ranges:
            continue
        col = [s[2][i] for s in samp]
        if all(v == 0 for v in col):
            ranges[i] = "ABSENT?"  # constant zero + no meta: absent OR unreadable
            source[i] = "inferred"
            notes.append(f"pkg{i}: constant 0 with no rapl_meta.txt — cannot "
                         f"distinguish 'no such socket' from 'read failed' post-hoc; "
                         f"sampler now records which at run start")
            continue
        v_hi = [col[j - 1] for j in range(1, len(col)) if col[j] < col[j - 1]]
        if v_hi:
            # counter value just before a wrap approximates the range from below
            # (within ~1 s of package energy); round up to the nearest 2^k J.
            est = max(v_hi)
            k = math.ceil(math.log2(est / 1e6))
            ranges[i] = (1 << k) * 1_000_000
            source[i] = "inferred"
            notes.append(f"pkg{i}: range INFERRED from {len(v_hi)} wrap event(s): "
                         f"max pre-wrap {est} uJ -> 2^{k} J = {(1 << k)} J")
        else:
            ranges[i] = None  # no wraps observed; naive diff is exact
            source[i] = "naive"
    return ranges, source, notes


def cell_energy(samp, a, b, rng):
    w = [s for s in samp if a <= s[0] <= b]
    if len(w) < 3:
        return None
    mean_w = sum(x[1] for x in w) / len(w)
    pkg = []
    for i, r in sorted(rng.items()):
        if isinstance(r, str):  # ABSENT
            pkg.append((r, 0, 0.0))
            continue
        col = [x[2][i] for x in w]
        wraps = sum(1 for j in range(1, len(col)) if col[j] < col[j - 1])
        naive = (col[-1] - col[0]) / 1e6
        corr = naive + wraps * (r / 1e6 if r else 0.0)
        if wraps and not r:
            pkg.append(("WRAPPED-NO-RANGE", wraps, naive))
        else:
            pkg.append((corr, wraps, naive))
    return {"n": len(w), "mean_w": mean_w, "dur": b - a, "pkg": pkg, "w": w}


def main(out: Path):
    VOID.clear()  # module-level accumulator; reset per invocation
    lines_s, lines_e = [], []
    for p in sorted(out.glob("flag*.json")):
        lines_s.append(summarize_cell(p))

    cells = {}
    ct = (out / "cells.tsv")
    if ct.exists():
        for ln in ct.read_text().splitlines():
            parts = ln.split()
            if len(parts) != 3:  # blank/torn line — skip, don't abort the energy pass
                continue
            n, e, t = parts
            try:
                cells.setdefault(n, {})[e] = float(t)
            except ValueError:
                continue
    samp = read_samples(out) if (out / "power.tsv").exists() else []
    if samp and cells:
        rng, src, notes = rapl_meta(out, samp)
        hdr = ["cell          dur_s   gpu_meanW  gpu_J     " +
               "  ".join(f"pkg{i}_J(corr wraps naive)" for i in sorted(rng))]
        rows, idle = {}, None
        for name in cells:
            tt = cells[name]
            if "start" not in tt or "end" not in tt:
                continue
            r = cell_energy(samp, tt["start"], tt["end"], rng)
            if r:
                rows[name] = r
        idle = rows.get("idle")
        for name, r in rows.items():
            pkg_s = "  ".join(
                (f"{c:.1f} w={wr} naive={nv:.1f}" if not isinstance(c, str)
                 else f"{c}") for c, wr, nv in r["pkg"])
            hdr.append(f"{name:<13} {r['dur']:<7.1f} {r['mean_w']:<10.1f} "
                       f"{r['mean_w'] * r['dur']:<9.1f} {pkg_s}")
        lines_e.extend(hdr)
        for n in notes:
            lines_e.append(f"note: {n}")
        idle_w = idle["mean_w"] if idle else None
        idle_cpu = (idle["pkg"][0][0] / idle["dur"]
                    if idle and not isinstance(idle["pkg"][0][0], str) else None)
        if idle_w is not None:
            lines_e.append(f"idle baseline: GPU {idle_w:.1f} W"
                           + (f", CPU pkg0 {idle_cpu:.1f} W" if idle_cpu else ""))

        def jtok(name, rate, note=""):
            r = rows.get(name)
            if not r or not rate:
                return
            lines_e.append(
                f"{name}: {rate:.3f} tok/s | GPU J/token absolute {r['mean_w'] / rate:.1f}"
                + (f", marginal-over-idle {(r['mean_w'] - idle_w) / rate:.2f}"
                   if idle_w is not None else "")
                + (f" | CPU pkg0 absolute {r['pkg'][0][0] / r['dur'] / rate:.1f}"
                   + (f", marginal {((r['pkg'][0][0] / r['dur']) - idle_cpu) / rate:.2f}"
                      if idle_cpu else "")
                   if not isinstance(r["pkg"][0][0], str) else "")
                + (f"  [{note}]" if note else ""))

        for stem in ("flagA_none", "flagA_fused"):
            d = _load(out / f"{stem}.json") if (out / f"{stem}.json").exists() else None
            if d and "c_box_ms" in d:  # probe contract: timing-only, no throughput/energy rows
                lines_e.append(f"{stem}: c_box probe — J/token intentionally not derived")
                continue
            if d and d.get("toks_per_s"):
                jtok(stem, d["toks_per_s"],
                     "window-mean W over the whole cell incl. setup; decode is a "
                     "small tail of the window — absolute figure is diluted")
                # decode-window figure (docstring contract): the decode loop is the
                # last thing in the cell, so window the trailing token-count seconds.
                r = rows.get(stem)
                if r and idle_w is not None and d.get("median_s_per_tok"):
                    cfg = d.get("config") or {}
                    n_all = (cfg.get("tokens") or 0) + (cfg.get("warmup_tokens") or 0)
                    est = n_all * d["median_s_per_tok"] + 2.0
                    w = r["w"]
                    tail = [s for s in w if s[0] >= w[-1][0] - est]
                    if n_all and len(tail) >= 3:
                        mW = sum(x[1] for x in tail) / len(tail)
                        dur = tail[-1][0] - tail[0][0]
                        rate = d["toks_per_s"]
                        line = (f"{stem} decode-window ({dur:.0f}s, post-hoc trailing "
                                f"{n_all}-token estimate): GPU absolute {mW / rate:.1f} "
                                f"J/tok, marginal {(mW - idle_w) / rate:.2f}")
                        r0 = rng.get(0)
                        col = [x[2][0] for x in tail]
                        if isinstance(r0, int) or (r0 is None and idle_cpu):
                            wraps = sum(1 for j in range(1, len(col)) if col[j] < col[j - 1])
                            cJ = ((col[-1] - col[0]) + (wraps * r0 if isinstance(r0, int) else 0)) / 1e6
                            if dur > 0 and idle_cpu:
                                line += (f" | CPU pkg0 absolute {cJ / dur / rate:.1f}, "
                                         f"marginal {(cJ / dur - idle_cpu) / rate:.2f}")
                        lines_e.append(line)
        fb = _load(out / "flagB_real.json") if (out / "flagB_real.json").exists() else None
        if fb and fb.get("c_box_probe"):
            lines_e.append("flagB: c_box probe — J/token intentionally not derived")
            fb = None
        if fb:
            offs = sorted(r["toks_per_s_off"] for r in (fb.get("results") or [])
                          if r.get("toks_per_s_off"))
            if offs:
                med = offs[len(offs) // 2]
                jtok("flagB", med, "window includes checkpoint download")
                # decode-tail: contiguous trailing window with GPU W above idle+10
                r = rows.get("flagB")
                if r and idle_w is not None:
                    w = r["w"]
                    ti = 0
                    for j in range(len(w) - 1, -1, -1):
                        if w[j][1] <= idle_w + 10:
                            ti = j + 1
                            break
                    tail = w[ti:]
                    if len(tail) > 30:
                        mW = sum(x[1] for x in tail) / len(tail)
                        dur = tail[-1][0] - tail[0][0]
                        col = [x[2][0] for x in tail]
                        wraps = sum(1 for j in range(1, len(col)) if col[j] < col[j - 1])
                        r0 = rng.get(0)
                        cJ = ((col[-1] - col[0]) + (wraps * r0 if isinstance(r0, int) else 0)) / 1e6
                        lines_e.append(
                            f"flagB decode-tail ({dur:.0f}s, post-hoc window W>idle+10, "
                            f"all decode modes pooled, rate=median off): GPU absolute "
                            f"{mW / med:.1f} J/tok, marginal {(mW - idle_w) / med:.2f}"
                            + (f" | CPU pkg0 absolute {cJ / dur / med:.1f}, marginal "
                               f"{(cJ / dur - idle_cpu) / med:.2f}" if idle_cpu else ""))

    s_txt = "\n".join(lines_s) + "\n"
    e_txt = "\n".join(lines_e) + "\n" if lines_e else "(no power.tsv/cells.tsv)\n"
    (out / "SUMMARY_reduced.txt").write_text(s_txt)
    (out / "ENERGY_reduced.txt").write_text(e_txt)
    print(s_txt)
    print(e_txt, end="")
    if VOID:
        print(f"\nCELL-VOID: {len({n for n, _ in VOID})} cell(s) produced no "
              f"parseable row — run is NOT evidence-complete", file=sys.stderr)
        return 3
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
