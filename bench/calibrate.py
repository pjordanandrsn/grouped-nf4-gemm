# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
#
# calibrate.py — Phase-0 calibration orchestrator for the hybrid CPU/GPU tier.
#
# Produces one machine-readable calibration blob per box (schema
# gnf4-hybrid-calib/1) holding *achieved* ceilings, never spec sheets:
#
#   B_dram  — STREAM triad + the G0 gate workload (grouped scattered
#             per-expert reads), via bench/hybrid_calib.c compiled here with
#             -march=native (the binary always matches the measured box)
#   B_vram  — device triad per GPU (torch, CUDA events)
#   B_link  — pinned H2D/D2H, both directions, 8 KB and 64 MB
#   B_nvme  — O_DIRECT seq/rand read (microbench; falls back to a page-cache
#             path where O_DIRECT is unsupported and says so)
#
# plus the hardware fingerprint (CPU flags incl. AVX-512/VBMI/VNNI, L3/CCD
# topology, THP + hugetlb state, governor, cgroup CPU quota, GPU inventory
# with power limits) and the G0 verdict:
#
#   scatter >= 70% of triad  -> proceed
#   50-70%                   -> proceed, re-solve expected speedups, report
#   < 50%                    -> STOP (CPU-tier economics need redesign)
#
# Benches run strictly serially. torch is imported lazily so the script runs
# (CPU-only) on boxes without CUDA or without torch.
#
# Usage:
#   python3 bench/calibrate.py --out receipts-hybrid-calib-<tag>.json \
#       [--nvme-dir /path/on/nvme] [--quick] [--skip-gpu] [--tag label]

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
C_SRC = HERE / "hybrid_calib.c"

GATE_PROCEED = 70.0
GATE_RESOLVE = 50.0


def _read(path, default=""):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return default


def _run(cmd, timeout=7200):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)


# --------------------------------------------------------------------------- #
# hardware fingerprint
# --------------------------------------------------------------------------- #

def host_fingerprint():
    cpuinfo = _read("/proc/cpuinfo")
    model = ""
    flags = set()
    for line in cpuinfo.splitlines():
        if line.startswith("model name") and not model:
            model = line.split(":", 1)[1].strip()
        elif line.startswith("flags") and not flags:
            flags = set(line.split(":", 1)[1].split())
    watch = ["avx2", "avx512f", "avx512bw", "avx512vl", "avx512_vbmi",
             "avx512vbmi", "avx512_vnni", "avx512vnni", "amx_tile"]
    nodes = sorted(p.name for p in Path("/sys/devices/system/node").glob("node[0-9]*")) \
        if Path("/sys/devices/system/node").exists() else []
    # cgroup v2 cpu quota — rented containers cap CPUs here, and a thread
    # ladder read without this is uninterpretable
    cpu_max = _read("/sys/fs/cgroup/cpu.max")
    quota = None
    if cpu_max and cpu_max.split()[0] != "max":
        q, period = cpu_max.split()[:2]
        quota = float(q) / float(period)
    return {
        "hostname": platform.node(),
        "kernel": platform.release(),
        "machine": platform.machine(),
        "cpu_model": model,
        "cpu_flags_watched": sorted(f for f in flags if f in watch),
        "online_cpus": os.cpu_count(),
        "cgroup_cpu_quota": quota,
        "numa_nodes": len(nodes),
        "mem_total_gib": round(
            int((_read("/proc/meminfo").split("MemTotal:")[1].split()[0]
                 if "MemTotal:" in _read("/proc/meminfo") else "0")) / 2**20, 1),
        "thp": _read("/sys/kernel/mm/transparent_hugepage/enabled"),
        "hugetlb_2m_total": _read("/sys/kernel/mm/hugepages/hugepages-2048kB/nr_hugepages"),
        "governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor",
                          default="(none)"),
        "loadavg_at_start": _read("/proc/loadavg"),
    }


def gpu_fingerprint():
    smi = shutil.which("nvidia-smi")
    if not smi:
        return []
    # capability = the MAX link the slot trained to; the CURRENT fields read
    # the idle state (typically Gen1 on a parked card) and once wrote a false
    # topology into a banked receipt — record both, labeled (Bugbot)
    r = _run([smi, "--query-gpu=index,name,memory.total,power.limit,"
                   "pcie.link.gen.max,pcie.link.width.max,"
                   "pcie.link.gen.current,pcie.link.width.current",
              "--format=csv,noheader"])
    gpus = []
    for line in r.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 4:
            gpus.append({"index": parts[0], "name": parts[1],
                         "vram": parts[2], "power_limit": parts[3],
                         "pcie_gen_max": parts[4] if len(parts) > 4 else "?",
                         "pcie_width_max": parts[5] if len(parts) > 5 else "?",
                         "pcie_gen_idle": parts[6] if len(parts) > 6 else "?",
                         "pcie_width_idle": parts[7] if len(parts) > 7 else "?"})
    return gpus


# --------------------------------------------------------------------------- #
# CPU microbench (compile + run)
# --------------------------------------------------------------------------- #

def build_microbench(workdir: Path):
    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if not cc:
        return None, "no C compiler on PATH"
    out = workdir / "hybrid_calib"
    cmd = [cc, "-O3", "-march=native", "-pthread", "-o", str(out), str(C_SRC)]
    r = _run(cmd, timeout=300)
    if r.returncode != 0:
        return None, f"compile failed: {r.stderr[-800:]}"
    return out, f"{cc} -O3 -march=native"


def run_microbench(binary: Path, args):
    r = _run([str(binary)] + args)
    if r.returncode != 0:
        return None, f"microbench failed: {r.stderr[-800:]}"
    try:
        return json.loads(r.stdout), None
    except json.JSONDecodeError as e:
        return None, f"microbench emitted bad JSON ({e}); stderr: {r.stderr[-400:]}"


# --------------------------------------------------------------------------- #
# GPU benches (torch, lazy)
# --------------------------------------------------------------------------- #

def gpu_benches(quick=False):
    try:
        import torch
    except ImportError:
        return {"skipped": "torch not importable"}
    if not torch.cuda.is_available():
        return {"skipped": "no CUDA device"}

    out = {"devices": []}
    n_elem = (1 << 28) if not quick else (1 << 26)      # 256M/64M fp32 elems per array
    for dev_i in range(torch.cuda.device_count()):
        dev = f"cuda:{dev_i}"
        torch.cuda.set_device(dev_i)
        rec = {"device": dev, "name": torch.cuda.get_device_name(dev_i)}

        # ---- device triad: a = b + 3*c, 24 B/elem STREAM convention
        try:
            b = torch.rand(n_elem, dtype=torch.float32, device=dev)
            c = torch.rand(n_elem, dtype=torch.float32, device=dev)
            a = torch.empty_like(b)
            torch.add(b, c, alpha=3.0, out=a)
            torch.cuda.synchronize()
            reps = 20
            times = []
            for _ in range(reps):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                torch.add(b, c, alpha=3.0, out=a)
                e.record()
                torch.cuda.synchronize()
                times.append(s.elapsed_time(e) / 1e3)
            times.sort()
            dt = times[len(times) // 2]
            # STREAM triad convention: 3 accesses/element at the ARRAY's
            # element size (fp32 here → 12 B/elem; the C bench uses doubles
            # → 24). Hard-coding 24 once inflated a 288 GB/s card to 511.
            rec["b_vram_triad_gbs"] = round(n_elem * 3 * a.element_size() / dt / 1e9, 1)
            del a, b, c
            torch.cuda.empty_cache()
        except RuntimeError as e:
            rec["b_vram_triad_gbs"] = None
            rec["vram_error"] = str(e)[:200]

        # ---- pinned link, both directions, 8 KB and 64 MB
        link = {}
        for label, nbytes, reps in (("8kb", 8 << 10, 500), ("64mb", 64 << 20, 40)):
            try:
                host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
                devt = torch.empty(nbytes, dtype=torch.uint8, device=dev)
                for name, src, dst in (("h2d", host, devt), ("d2h", devt, host)):
                    dst.copy_(src, non_blocking=True)
                    torch.cuda.synchronize()
                    s = torch.cuda.Event(enable_timing=True)
                    e = torch.cuda.Event(enable_timing=True)
                    s.record()
                    for _ in range(reps):
                        dst.copy_(src, non_blocking=True)
                    e.record()
                    torch.cuda.synchronize()
                    sec = s.elapsed_time(e) / 1e3 / reps
                    link[f"{name}_{label}"] = {
                        "gbs": round(nbytes / sec / 1e9, 2),
                        "usec": round(sec * 1e6, 1),
                    }
                del host, devt
            except RuntimeError as e:
                link[f"error_{label}"] = str(e)[:200]
        rec["b_link"] = link
        out["devices"].append(rec)
    return out


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="blob path (default: receipts-hybrid-calib-<tag>.json beside this script)")
    ap.add_argument("--tag", default=None, help="label for the receipt filename")
    ap.add_argument("--nvme-dir", default=None, help="directory on the drive to measure")
    ap.add_argument("--nvme-gib", type=int, default=8)
    ap.add_argument("--quick", action="store_true", help="smaller sizes, for smoke tests only — not a citable calibration")
    ap.add_argument("--skip-gpu", action="store_true")
    ap.add_argument("--skip-cpu", action="store_true")
    args = ap.parse_args()

    t_start = time.time()
    blob = {
        "schema": "gnf4-hybrid-calib/1",
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "quick_mode": bool(args.quick),
        "host": host_fingerprint(),
        "gpus": gpu_fingerprint(),
        "notes": [],
    }
    # Receipts are committed: replace the raw hostname with the neutral tag.
    # Rented-host identifiers (pod/container names) and lab hostnames are
    # exactly what the private-marker guard exists to keep out of the tree.
    blob["host"]["hostname"] = args.tag or "untagged"
    if args.quick:
        blob["notes"].append("quick mode: sizes reduced; NOT a citable calibration")

    if platform.system() != "Linux":
        blob["notes"].append(f"non-Linux host ({platform.system()}): CPU microbench skipped")
        args.skip_cpu = True

    # ---- CPU side (microbench)
    if not args.skip_cpu:
        workdir = Path(os.environ.get("TMPDIR", "/tmp")) / "gnf4-hybrid-calib"
        workdir.mkdir(parents=True, exist_ok=True)
        binary, how = build_microbench(workdir)
        blob["cpu_bench_build"] = how
        if binary is None:
            blob["cpu_bench"] = {"error": how}
        else:
            mb_args = []
            if args.quick:
                mb_args += ["--triad-gib", "1", "--arena-gib", "2",
                            "--fetches", "100", "--reps", "3"]
            if args.nvme_dir:
                mb_args += ["--nvme-dir", args.nvme_dir,
                            "--nvme-gib", str(args.nvme_gib if not args.quick else 2)]
            print(f"[calibrate] running CPU microbench ({how})...", file=sys.stderr)
            result, err = run_microbench(binary, mb_args)
            blob["cpu_bench"] = result if result else {"error": err}
    else:
        blob["cpu_bench"] = {"skipped": True}

    # ---- GPU side
    if not args.skip_gpu:
        print("[calibrate] running GPU benches...", file=sys.stderr)
        blob["gpu_bench"] = gpu_benches(quick=args.quick)
    else:
        blob["gpu_bench"] = {"skipped": True}

    # ---- gate verdict
    cb = blob.get("cpu_bench") or {}
    gate = None
    if isinstance(cb, dict) and "gate_g0" in cb:
        pct = cb["gate_g0"]["scatter_pct_of_triad"]
        verdict = ("proceed" if pct >= GATE_PROCEED
                   else "proceed-resolve-and-report" if pct >= GATE_RESOLVE
                   else "stop")
        gate = {"scatter_pct_of_triad": pct, "verdict": verdict,
                "thresholds": {"proceed": GATE_PROCEED, "resolve": GATE_RESOLVE},
                "triad_best": cb.get("triad_best"),
                "scatter_best": cb.get("scatter_best")}
    blob["gate_g0"] = gate
    blob["wall_seconds"] = round(time.time() - t_start, 1)

    # never let a hostname reach a committed filename — the blob already
    # substitutes --tag for hostname; the default path must not undo it
    tag = args.tag or "untagged"
    out = Path(args.out) if args.out else (HERE / "cold-engine" /
                                           f"receipts-hybrid-calib-{tag}.json")
    out.write_text(json.dumps(blob, indent=2) + "\n")

    # ---- human summary
    print(f"\ncalibration blob: {out}")
    h = blob["host"]
    print(f"host: {h['cpu_model']} | {h['online_cpus']} cpus | "
          f"quota {h['cgroup_cpu_quota']} | flags {h['cpu_flags_watched']}")
    if isinstance(cb, dict) and "triad_best" in cb:
        tb, sb = cb["triad_best"], cb["scatter_best"]
        print(f"B_dram triad best : {tb['gbs']:8.1f} GB/s  ({tb['threads']}t {tb['pin']} nt={tb['nt']})")
        print(f"grouped scatter   : {sb['gbs']:8.1f} GB/s  ({sb['threads']}t, {sb['block_mib']} MiB blocks, E={sb['experts']})")
    if gate:
        print(f"GATE G0           : {gate['scatter_pct_of_triad']:.1f}% of triad -> {gate['verdict'].upper()}")
    gb = blob.get("gpu_bench") or {}
    for d in (gb.get("devices") or []):
        l = d.get("b_link", {})
        h2d = l.get("h2d_64mb", {}).get("gbs", "?")
        d2h = l.get("d2h_64mb", {}).get("gbs", "?")
        print(f"{d['device']} {d['name']}: triad {d.get('b_vram_triad_gbs')} GB/s | "
              f"link h2d {h2d} / d2h {d2h} GB/s")
    if isinstance(cb, dict) and isinstance(cb.get("nvme"), dict):
        pts = cb["nvme"].get("points") or []
        best_seq = max((p["gbs"] for p in pts if p.get("ok") and p["mode"] == "seq"), default=None)
        print(f"B_nvme seq best   : {best_seq} GB/s (o_direct={cb['nvme'].get('o_direct')})")
    return 0 if (gate is None or gate["verdict"] != "stop") else 3


if __name__ == "__main__":
    sys.exit(main())
