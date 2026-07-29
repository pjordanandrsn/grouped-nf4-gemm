# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N1 — the arena bake: expert-major on-disk relocation of a checkpoint's
packed expert bytes, with a provenance manifest that makes the relocation
verifiable. Bake once, read forever.

Why this exists: safetensors shards are tensor-major, so one expert's
segments (blocks + scales for gate_up and down) are scattered across a
shard's data section; serving an expert from disk would cost several seeks
plus reads of unrelated bytes. The bake walks the checkpoint once and emits
a contiguous arena: **one aligned row per (layer, expert)**, the row's
internal layout byte-compatible with the pinned DRAM arena rows the engine
already gathers from (segments in fixed order, each 8-byte aligned — the
`Mxfp4PipelinedGptOss` layout), padded to the device block size so O_DIRECT
reads a whole expert in exactly one aligned request.

Provenance is per source slice — the moonshot_gather doctrine: relocation
reorders bytes, so ``sha256(arena) != sha256(file)`` and the honest receipt
is per-slice. For every (layer, expert, segment) the manifest records the
source file, the absolute byte range in that file, and the sha256 of those
bytes, hashed WHILE STREAMING the bake (one pass; the arena writer and the
hasher see the same buffer). ``verify`` re-reads the arena and re-hashes
every segment against the manifest — that is what a user runs to prove a
local copy is intact; ``--against-source`` additionally re-hashes the
checkpoint ranges, closing the chain shipped-file -> manifest -> arena.
Hashing reuses ``mxfp4_loader``'s header/byte-range primitives (the same
code path `verify_provenance` trusts); a provenance failure is never a
tolerance.

The claim this preserves: *the arena holds the same bytes the vendor
shipped, relocated.* This bake mode is a pure relocation and applies to
checkpoints that ship packed expert tensors (gpt-oss mxfp4 today; K3 when
its weights land). Checkpoints that need quantization first (bf16 -> NF4)
would bake QUANTIZED bytes whose provenance is source-hash + quantizer
version — a distinct mode, deliberately not implemented here.

  python3 nvme_arena.py bake --snapshot /path/to/hf/snapshot \
      --out /nvme/model.arena [--layers 0-35] [--align 4096]
  python3 nvme_arena.py verify --arena /nvme/model.arena [--against-source]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

from mxfp4_loader import EXPERT_SUFFIXES, _read_st_header

MAGIC = "gnf4-nvme-arena"
VERSION = 1
CHUNK = 4 << 20

# safetensors dtype tag -> bytes per element (packed formats ship as U8)
_DTYPE_BYTES = {"U8": 1, "I8": 1, "BOOL": 1, "F16": 2, "BF16": 2, "I16": 2,
                "U16": 2, "F32": 4, "I32": 4, "U32": 4, "F64": 8, "I64": 8,
                "U64": 8}


def _align8(n: int) -> int:
    return (n + 7) & ~7


def _align(n: int, a: int) -> int:
    return (n + a - 1) & ~(a - 1)


def resolve_weight_map(snapshot: str):
    """(weight_map, files) for a snapshot dir: sharded via the index, or a
    single model.safetensors serving every name."""
    idx = os.path.join(snapshot, "model.safetensors.index.json")
    if os.path.exists(idx):
        wm = json.load(open(idx))["weight_map"]
        return wm, sorted(set(wm.values()))
    single = os.path.join(snapshot, "model.safetensors")
    if not os.path.exists(single):
        raise FileNotFoundError(f"no index and no model.safetensors in {snapshot}")
    hdr, _ = _read_st_header(single)
    return {k: "model.safetensors" for k in hdr if k != "__metadata__"}, \
        ["model.safetensors"]


def discover_layers(weight_map: dict, prefix: str, suffixes) -> list[int]:
    layers = set()
    probe = "." + suffixes[0]
    for name in weight_map:
        if name.startswith(prefix + ".") and name.endswith(probe):
            layers.add(int(name[len(prefix) + 1:].split(".")[0]))
    return sorted(layers)


class _Src:
    """Per-shard header cache: name -> (path, absolute_lo, absolute_hi, shape, dtype)."""

    def __init__(self, snapshot: str, weight_map: dict):
        self.snapshot, self.weight_map = snapshot, weight_map
        self._hdrs = {}

    def locate(self, name: str):
        shard = self.weight_map.get(name)
        if shard is None:
            raise KeyError(f"{name} not in the checkpoint index")
        path = os.path.join(self.snapshot, shard)
        if shard not in self._hdrs:
            self._hdrs[shard] = _read_st_header(path)
        hdr, data_start = self._hdrs[shard]
        ent = hdr[name]
        lo, hi = ent["data_offsets"]
        return path, data_start + lo, data_start + hi, ent["shape"], ent["dtype"]


def _stream_copy_hash(src_path, src_lo, src_hi, dst_f, dst_off) -> str:
    """Copy [src_lo, src_hi) of src into dst at dst_off, sha256 on the way.
    The hash and the write consume the same buffer: what was hashed IS what
    landed in the arena."""
    h = hashlib.sha256()
    with open(src_path, "rb") as f:
        f.seek(src_lo)
        dst_f.seek(dst_off)
        remaining = src_hi - src_lo
        while remaining:
            buf = f.read(min(CHUNK, remaining))
            if not buf:
                raise EOFError(f"truncated read in {src_path} at {src_hi - remaining}")
            h.update(buf)
            dst_f.write(buf)
            remaining -= len(buf)
    return h.hexdigest()


def bake(snapshot: str, out: str, *, layers=None, prefix="model.layers",
         suffixes=EXPERT_SUFFIXES, align=4096, limit_experts=0,
         log=print) -> dict:
    """Walk the checkpoint expert-major; emit `<out>` (arena) plus
    `<out>.index.json` (geometry + row offsets, readable without the arena)
    and `<out>.manifest.json` (per-segment source provenance)."""
    weight_map, _files = resolve_weight_map(snapshot)
    src = _Src(snapshot, weight_map)
    if layers is None:
        layers = discover_layers(weight_map, prefix, suffixes)
    if not layers:
        raise ValueError(f"no MoE layers found under prefix {prefix!r}")

    # geometry from layer[0]; every layer is asserted identical during the walk
    seg_geo = []
    off = 0
    E = None
    for suf in suffixes:
        name = f"{prefix}.{layers[0]}.{suf}"
        _p, lo, hi, shape, dtype = src.locate(name)
        if dtype not in _DTYPE_BYTES:
            raise ValueError(f"{name}: unhandled safetensors dtype {dtype}")
        if E is None:
            E = shape[0]
        elif shape[0] != E:
            raise ValueError(f"{name}: expert dim {shape[0]} != {E}")
        nbytes = hi - lo
        if nbytes % E:
            raise ValueError(f"{name}: {nbytes} bytes not divisible by E={E}")
        per = nbytes // E
        seg_geo.append({"suffix": suf, "seg_off": off, "length": per,
                        "shape_per_expert": shape[1:], "dtype": dtype})
        off = _align8(off + per)
    row_bytes = off
    row_stride = _align(row_bytes, align)

    rows, manifest_rows = [], []
    arena_off = 0
    t0 = time.time()
    with open(out, "wb") as dst:
        for li, lay in enumerate(layers):
            locs = []
            for g in seg_geo:
                name = f"{prefix}.{lay}.{g['suffix']}"
                p, lo, hi, shape, dtype = src.locate(name)
                if shape[0] != E or dtype != g["dtype"] or \
                        (hi - lo) // E != g["length"]:
                    raise ValueError(f"{name}: geometry differs from layer {layers[0]}")
                locs.append((p, lo))
            n_e = min(E, limit_experts) if limit_experts else E
            for e in range(n_e):
                segs = []
                for g, (p, lo) in zip(seg_geo, locs):
                    s_lo = lo + e * g["length"]
                    s_hi = s_lo + g["length"]
                    sha = _stream_copy_hash(p, s_lo, s_hi, dst,
                                            arena_off + g["seg_off"])
                    segs.append({"suffix": g["suffix"],
                                 "source_file": os.path.basename(p),
                                 "source_range": [s_lo, s_hi], "sha256": sha})
                rows.append([lay, e, arena_off])
                manifest_rows.append({"layer": lay, "expert": e,
                                      "offset": arena_off, "segments": segs})
                arena_off += row_stride
            log(f"  baked layer {lay} ({li + 1}/{len(layers)}, "
                f"{arena_off / 1e9:.2f} GB, {time.time() - t0:.0f}s)")
        # Extend the file to the full striped size so a full-row O_DIRECT read
        # of the last expert never hits EOF. truncate() zero-fills any
        # inter-/intra-row pad slack as a hole and NEVER overwrites written
        # bytes — a per-row `seek(off-1); write(0)` clobbers the last data
        # byte whenever row_bytes == row_stride (a 4096-aligned row), which
        # the tiny-geometry unit tests never hit but a real checkpoint does.
        dst.flush()
        dst.truncate(arena_off)
        os.fsync(dst.fileno())

    index = {"magic": MAGIC, "version": VERSION, "snapshot": snapshot,
             "prefix": prefix, "align": align, "row_bytes": row_bytes,
             "row_stride": row_stride, "n_layers": len(layers),
             "n_experts_per_layer": E, "segments": seg_geo, "rows": rows,
             "arena_bytes": arena_off}
    manifest = {"magic": MAGIC, "version": VERSION, "algo": "sha256",
                "snapshot": snapshot, "baked_utc":
                time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "rows": manifest_rows}
    with open(out + ".index.json", "w") as f:
        json.dump(index, f)
    with open(out + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    log(f"bake complete: {len(rows)} rows x {row_stride} B "
        f"({arena_off / 1e9:.2f} GB) in {time.time() - t0:.0f}s")
    return index


def bake_expert_tensors(snapshot: str, out: str, *, name_template: str,
                        kinds, layers=None, align=4096, limit_experts=0,
                        workers: int = 8, log=print) -> dict:
    """Relocation bake for checkpoints that ship INDIVIDUAL per-expert
    tensors (DeepSeek/Mixtral lineage; Kimi-K3's
    `...block_sparse_moe.experts.{e}.{kind}`). Each row segment is one
    whole source tensor range — single-source, hash-preserving: the
    relocation claim at its purest. name_template must contain {layer},
    {expert} and {kind}; `kinds` fixes the segment order.

    Parallel: locations are resolved serially (header cache), then a
    thread pool streams+hashes experts concurrently — os.pwrite at
    absolute offsets, one reader fd per (thread, shard)."""
    import re
    import threading
    from concurrent.futures import ThreadPoolExecutor

    weight_map, _files = resolve_weight_map(snapshot)
    src = _Src(snapshot, weight_map)
    pat = re.compile(re.escape(name_template)
                     .replace(r"\{layer\}", r"(?P<layer>\d+)")
                     .replace(r"\{expert\}", r"(?P<expert>\d+)")
                     .replace(r"\{kind\}", "(?P<kind>" + "|".join(
                         re.escape(k) for k in kinds) + ")"))
    lays, exps = set(), set()
    for name in weight_map:
        m = pat.fullmatch(name)
        if m:
            lays.add(int(m["layer"]))
            exps.add(int(m["expert"]))
    if layers is None:
        layers = sorted(lays)
    if not layers:
        raise ValueError(f"no experts matched {name_template!r}")
    E = max(exps) + 1
    n_e = min(E, limit_experts) if limit_experts else E

    seg_geo, off = [], 0
    for kind in kinds:
        name = name_template.format(layer=layers[0], expert=0, kind=kind)
        _p, lo, hi, shape, dtype = src.locate(name)
        seg_geo.append({"suffix": kind, "seg_off": off, "length": hi - lo,
                        "shape_per_expert": shape, "dtype": dtype})
        off = _align8(off + (hi - lo))
    row_bytes = off
    row_stride = _align(row_bytes, align)
    total = len(layers) * n_e * row_stride
    log(f"expert-tensor bake: L={len(layers)} E={E} (baking {n_e}) "
        f"row={row_bytes} stride={row_stride} total={total/1e9:.1f} GB "
        f"workers={workers}")

    # resolve every source range serially (headers cached), then fan out IO
    jobs, rows = [], []
    arena_off = 0
    for lay in layers:
        for e in range(n_e):
            locs = []
            for g in seg_geo:
                name = name_template.format(layer=lay, expert=e,
                                            kind=g["suffix"])
                p, lo, hi, shape, dtype = src.locate(name)
                if hi - lo != g["length"] or dtype != g["dtype"]:
                    raise ValueError(f"{name}: geometry differs from "
                                     f"layer {layers[0]} expert 0")
                locs.append((p, lo, hi))
            jobs.append((lay, e, arena_off, locs))
            rows.append([lay, e, arena_off])
            arena_off += row_stride

    dst_fd = os.open(out, os.O_WRONLY | os.O_CREAT | getattr(os, "O_TRUNC", 0))
    os.truncate(out, arena_off)
    # Raise the fd ceiling as defense; the real fix is that `run` opens its
    # source fds per-job and closes them (finally), so concurrent fds are
    # bounded by workers x (distinct shards per expert) ~= workers x 1-2, NOT
    # workers x n_shards. The old per-thread fd cache leaked one fd per shard
    # per worker and blew Errno 24 at shard 87 of the 96-shard K3 bake.
    try:
        import resource
        _soft, _hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (min(_hard, 1 << 16), _hard))
    except (ImportError, ValueError, OSError):
        pass
    t0 = time.time()
    done = [0]
    lock = threading.Lock()

    def run(job):
        lay, e, base, locs = job
        segs = []
        local_fds = {}
        try:
            for g, (p, lo, hi) in zip(seg_geo, locs):
                fd = local_fds.get(p)
                if fd is None:
                    fd = local_fds[p] = os.open(p, os.O_RDONLY)
                h = hashlib.sha256()
                pos, woff = lo, base + g["seg_off"]
                while pos < hi:
                    buf = os.pread(fd, min(CHUNK, hi - pos), pos)
                    if not buf:
                        raise EOFError(f"truncated {p} at {pos}")
                    h.update(buf)
                    os.pwrite(dst_fd, buf, woff)
                    pos += len(buf)
                    woff += len(buf)
                segs.append({"suffix": g["suffix"],
                             "source_file": os.path.basename(p),
                             "source_range": [lo, hi], "sha256": h.hexdigest()})
        finally:
            for fd in local_fds.values():
                os.close(fd)
        with lock:
            done[0] += 1
            if done[0] % 2048 == 0:
                log(f"  {done[0]}/{len(jobs)} rows "
                    f"({done[0] * row_stride / 1e9:.0f} GB, "
                    f"{time.time() - t0:.0f}s)")
        return {"layer": lay, "expert": e, "offset": base, "segments": segs}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        man_rows = list(pool.map(run, jobs))
    os.fsync(dst_fd)
    os.close(dst_fd)

    index = {"magic": MAGIC, "version": VERSION, "snapshot": snapshot,
             "prefix": name_template, "align": align, "row_bytes": row_bytes,
             "row_stride": row_stride, "n_layers": len(layers),
             "n_experts_per_layer": n_e, "segments": seg_geo, "rows": rows,
             "arena_bytes": arena_off, "bake_mode": "relocate-expert-tensors",
             "moe_layers": layers}
    manifest = {"magic": MAGIC, "version": VERSION, "algo": "sha256",
                "snapshot": snapshot, "bake_mode": "relocate-expert-tensors",
                "baked_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ",
                                           time.gmtime()),
                "rows": man_rows}
    with open(out + ".index.json", "w") as f:
        json.dump(index, f)
    with open(out + ".manifest.json", "w") as f:
        json.dump(manifest, f)
    log(f"bake complete: {len(rows)} rows ({arena_off/1e9:.1f} GB) "
        f"in {time.time() - t0:.0f}s")
    return index


def load_index(arena: str) -> dict:
    idx = json.load(open(arena + ".index.json"))
    if idx.get("magic") != MAGIC:
        raise ValueError(f"{arena}.index.json: not a {MAGIC} index")
    return idx


def row_offset(index: dict, layer: int, expert: int) -> int:
    """O(1) lookup with an explicit-map fallback: rows are laid out in bake
    order, which for the uniform case is (layer_pos * E + expert) * stride."""
    key = getattr(row_offset, "_cache", None)
    if key is None or key[0] is not index:
        row_offset._cache = (index, {(l, e): o for l, e, o in index["rows"]})
        key = row_offset._cache
    return key[1][(layer, expert)]


def _hash_range(path, lo, hi):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(lo)
        remaining = hi - lo
        while remaining:
            buf = f.read(min(CHUNK, remaining))
            if not buf:
                raise EOFError(f"truncated {path}")
            h.update(buf)
            remaining -= len(buf)
    return h.hexdigest()


def verify(arena: str, *, against_source=None, limit=0, log=print) -> dict:
    """Re-read the arena and re-hash every segment of every row against the
    manifest. `against_source` (a snapshot dir) additionally re-hashes each
    recorded source range — the full chain shipped -> manifest -> arena."""
    index = load_index(arena)
    manifest = json.load(open(arena + ".manifest.json"))
    rows = manifest["rows"]
    if limit:
        rows = rows[:limit]
    bad = []
    t0 = time.time()
    with open(arena, "rb") as f:
        for i, row in enumerate(rows):
            for seg in row["segments"]:
                lo = row["offset"] + _seg_off(index, seg["suffix"])
                f.seek(lo)
                remaining = seg_len = _seg_len(index, seg["suffix"])
                h = hashlib.sha256()
                while remaining:
                    buf = f.read(min(CHUNK, remaining))
                    if not buf:
                        raise EOFError(f"arena truncated at row {row['layer']},{row['expert']}")
                    h.update(buf)
                    remaining -= len(buf)
                got = h.hexdigest()
                ok = got == seg["sha256"]
                if not ok:
                    bad.append((row["layer"], row["expert"], seg["suffix"],
                                "arena"))
                if against_source and ok:
                    # relocation mode: the segment IS the source bytes (one
                    # range, same hash). nf4-quantize mode: the segment
                    # records its consumed sources separately ("sources"),
                    # each with its own shipped-bytes hash — the two-hop
                    # chain: source unchanged AND arena self-consistent.
                    srcs = seg.get("sources") or [
                        {"source_file": seg["source_file"],
                         "source_range": seg["source_range"],
                         "sha256": seg["sha256"]}]
                    for s in srcs:
                        spath = os.path.join(against_source, s["source_file"])
                        s_lo, s_hi = s["source_range"]
                        if _hash_range(spath, s_lo, s_hi) != s["sha256"]:
                            bad.append((row["layer"], row["expert"],
                                        seg["suffix"], "source"))
                            break
            if (i + 1) % 512 == 0:
                log(f"  verified {i + 1}/{len(rows)} rows ({time.time() - t0:.0f}s)")
    report = {"rows_checked": len(rows), "failures": bad,
              "ok": not bad, "seconds": round(time.time() - t0, 1)}
    if bad:
        log(f"PROVENANCE FAIL: {len(bad)} segment(s) differ; first: "
            f"layer={bad[0][0]} expert={bad[0][1]} seg={bad[0][2]} ({bad[0][3]})")
    else:
        log(f"PROVENANCE OK: {len(rows)} rows, every segment matches the manifest"
            + (" and the source checkpoint" if against_source else ""))
    return report


def _seg_off(index, suffix):
    for g in index["segments"]:
        if g["suffix"] == suffix:
            return g["seg_off"]
    raise KeyError(suffix)


def _seg_len(index, suffix):
    for g in index["segments"]:
        if g["suffix"] == suffix:
            return g["length"]
    raise KeyError(suffix)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bake")
    b.add_argument("--snapshot", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--layers", default=None, help="e.g. 0-11 or 0,1,2")
    b.add_argument("--prefix", default="model.layers")
    b.add_argument("--align", type=int, default=4096)
    b.add_argument("--limit-experts", type=int, default=0)
    be = sub.add_parser("bake-experts",
                        help="relocation bake for per-expert-tensor "
                             "checkpoints (K3/DeepSeek lineage)")
    be.add_argument("--snapshot", required=True)
    be.add_argument("--out", required=True)
    be.add_argument("--name-template", required=True,
                    help="e.g. language_model.model.layers.{layer}."
                         "block_sparse_moe.experts.{expert}.{kind}")
    be.add_argument("--kinds", required=True, help="comma list, row order")
    be.add_argument("--layers", default=None)
    be.add_argument("--align", type=int, default=4096)
    be.add_argument("--limit-experts", type=int, default=0)
    be.add_argument("--workers", type=int, default=8)
    v = sub.add_parser("verify")
    v.add_argument("--arena", required=True)
    v.add_argument("--against-source", default=None)
    v.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    layers = None
    if getattr(args, "layers", None):
        if "-" in args.layers:
            a, b_ = args.layers.split("-")
            layers = list(range(int(a), int(b_) + 1))
        else:
            layers = [int(x) for x in args.layers.split(",")]
    if args.cmd == "bake":
        bake(args.snapshot, args.out, layers=layers, prefix=args.prefix,
             align=args.align, limit_experts=args.limit_experts)
        return 0
    if args.cmd == "bake-experts":
        bake_expert_tensors(args.snapshot, args.out,
                            name_template=args.name_template,
                            kinds=args.kinds.split(","), layers=layers,
                            align=args.align,
                            limit_experts=args.limit_experts,
                            workers=args.workers)
        return 0
    report = verify(args.arena, against_source=args.against_source,
                    limit=args.limit)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
