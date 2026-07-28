#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N0 — the NVMe device microbench. Measures what the disk actually delivers
for the tier's access pattern: O_DIRECT reads at expert-sized granularity,
random offsets over the whole span, sustained, at several queue depths.

Two independent instruments, cross-checked in one receipt:

  * ``pread``  — threaded os.preadv, QD = thread count. This is the engine's
    own portable fallback path (io_uring is absent on some boxes — measured
    ENOSYS on the 6.6.32-qnap kernel — so this path must exist anyway, and
    benchmarking it here doubles as its validation).
  * ``fio``    — fio with libaio (true async QD) when a fio binary exists.
    Handles both ancient (2.2.10, clat in usec) and modern (3.x, clat_ns)
    JSON schemas.

Alignment discipline: O_DIRECT requires buffer, offset and length aligned to
the device logical block size or the syscall fails outright. Buffers come
from mmap (page-aligned); offsets are aligned to max(4096, lbs); request
sizes are MiB multiples (always 512-aligned). The probe records lbs/pbs and
the sweep enforces them.

If O_DIRECT is unavailable (macOS, some filesystems/containers) the harness
degrades loudly: macOS gets F_NOCACHE, elsewhere plain buffered reads, and
the receipt + stderr carry the warning that page cache makes the number an
upper bound fiction. Never silently.

Contention accounting: /proc/diskstats is snapshotted around every config;
the receipt reports non-harness device traffic during the window, so a
number taken on a shared box carries its own contamination evidence.

  python3 nvme_microbench.py probe --target /dev/nvme1n1
  python3 nvme_microbench.py sweep --target /dev/nvme1n1 \
      --engines pread,fio --sizes-mib 4,13,25,50 --qds 1,4,16,64 \
      --seconds 8 --out receipt.json
"""
from __future__ import annotations

import argparse
import ctypes
import json
import mmap
import os
import platform
import random
import shutil
import stat
import statistics
import subprocess
import sys
import threading
import time

MIB = 1 << 20


# ---------------------------------------------------------------- device info
def _read_sys(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except OSError:
        return None


def device_info(target: str) -> dict:
    """Model, size, block sizes and PCIe link for a block device (best effort;
    regular files report size + the filesystem's st_blksize)."""
    info = {"target": target}
    st = os.stat(target)
    if stat.S_ISBLK(st.st_mode):
        name = os.path.basename(os.path.realpath(target))
        base = name.rstrip("0123456789")
        if base.endswith("p") and base[:-1].rstrip("0123456789") != base[:-1]:
            name = base[:-1]  # nvme0n1p3 -> nvme0n1
        q = f"/sys/block/{name}/queue"
        info.update(
            kind="block",
            device=name,
            logical_block_size=int(_read_sys(f"{q}/logical_block_size") or 512),
            physical_block_size=int(_read_sys(f"{q}/physical_block_size") or 512),
            rotational=_read_sys(f"{q}/rotational"),
            scheduler=_read_sys(f"{q}/scheduler"),
        )
        ctrl = name.split("n")[0]  # nvme1n1 -> nvme1
        dev = f"/sys/class/nvme/{ctrl}"
        if os.path.isdir(dev):
            info.update(
                model=(_read_sys(f"{dev}/model") or "").strip(),
                pcie_speed=_read_sys(f"{dev}/device/current_link_speed"),
                pcie_width=_read_sys(f"{dev}/device/current_link_width"),
                pcie_max_speed=_read_sys(f"{dev}/device/max_link_speed"),
                pcie_max_width=_read_sys(f"{dev}/device/max_link_width"),
            )
        fd = os.open(target, os.O_RDONLY)
        try:
            info["size_bytes"] = os.lseek(fd, 0, os.SEEK_END)
        finally:
            os.close(fd)
    else:
        info.update(kind="file", size_bytes=st.st_size,
                    logical_block_size=st.st_blksize,
                    physical_block_size=st.st_blksize)
    return info


def io_uring_available() -> dict:
    """io_uring_setup syscall probe. ENOSYS = compiled out of the kernel
    (seen on 6.6.32-qnap); EPERM/EACCES = present but restricted."""
    if platform.system() != "Linux":
        return {"available": False, "reason": platform.system()}
    nr = {"x86_64": 425, "aarch64": 425}.get(platform.machine())
    if nr is None:
        return {"available": False, "reason": f"unknown arch {platform.machine()}"}
    libc = ctypes.CDLL(None, use_errno=True)
    params = ctypes.create_string_buffer(120)
    fd = libc.syscall(nr, 8, params)
    if fd >= 0:
        os.close(fd)
        return {"available": True}
    import errno as _errno
    e = ctypes.get_errno()
    return {"available": False, "errno": e,
            "reason": _errno.errorcode.get(e, str(e))}


def diskstats(device: str | None) -> dict | None:
    """Sectors read/written for one device from /proc/diskstats."""
    if not device or not os.path.exists("/proc/diskstats"):
        return None
    for line in open("/proc/diskstats"):
        f = line.split()
        if f[2] == device:
            return {"sectors_read": int(f[5]), "sectors_written": int(f[9]),
                    "t": time.time()}
    return None


# ------------------------------------------------------------- pread engine
_ODIRECT_MODE = None  # set once per process; the loud-warning latch


def open_direct(target: str) -> tuple[int, str]:
    """O_RDONLY + O_DIRECT, degrading loudly: macOS F_NOCACHE, else buffered
    with a one-time warning naming the consequence."""
    global _ODIRECT_MODE
    if hasattr(os, "O_DIRECT"):
        try:
            fd = os.open(target, os.O_RDONLY | os.O_DIRECT)
            _ODIRECT_MODE = "O_DIRECT"
            return fd, _ODIRECT_MODE
        except OSError as e:
            mode = f"buffered (O_DIRECT failed: {e.strerror})"
    elif platform.system() == "Darwin":
        import fcntl
        fd = os.open(target, os.O_RDONLY)
        fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
        mode = "F_NOCACHE (macOS; weaker than O_DIRECT)"
        if _ODIRECT_MODE != mode:
            print(f"WARNING: {mode} — treat throughput as approximate",
                  file=sys.stderr, flush=True)
        _ODIRECT_MODE = mode
        return fd, mode
    else:
        mode = "buffered (no O_DIRECT on this platform)"
    if "buffered" in mode:
        fd = os.open(target, os.O_RDONLY)
        if _ODIRECT_MODE != mode:
            print(f"WARNING: {mode} — page cache will serve rereads; the "
                  "number is NOT a device measurement", file=sys.stderr,
                  flush=True)
        _ODIRECT_MODE = mode
    return fd, mode


def pread_config(target: str, span: int, bs: int, qd: int, seconds: float,
                 align: int, warmup: float = 1.0, seq: bool = False) -> dict:
    """One (request size, queue depth) cell: qd threads, each looping aligned
    preads until the deadline. Returns achieved GB/s + latency percentiles
    (warmup-filtered)."""
    stop_at = time.perf_counter() + warmup + seconds
    warm_until = time.perf_counter() + warmup
    results = []          # (t_done, dt_ns) per request, per thread
    errors = []

    def worker(tid: int):
        fd, mode = open_direct(target)
        buf = mmap.mmap(-1, bs)
        mv = memoryview(buf)
        rng = random.Random(0xE4B0 + tid)
        local = []
        try:
            # sequential mode: threads own disjoint strides, walk forward
            n_slots = max(1, (span - bs) // align)
            pos = (tid * (span // max(qd, 1))) & ~(align - 1)
            while True:
                t0 = time.perf_counter()
                if t0 >= stop_at:
                    break
                if seq:
                    off = pos
                    pos += bs
                    if pos + bs > span:
                        pos = (tid * (span // max(qd, 1))) & ~(align - 1)
                else:
                    off = (rng.randrange(n_slots) * align) & ~(align - 1)
                    if off + bs > span:
                        off = span - bs
                        off &= ~(align - 1)
                n = os.preadv(fd, [mv], off)
                dt = time.perf_counter() - t0
                if n != bs:
                    errors.append(f"short read {n} at {off}")
                    break
                local.append((t0 + dt, dt))
        finally:
            mv.release()
            buf.close()
            os.close(fd)
        results.append(local)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(qd)]
    t_start = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t_start

    all_reqs = [r for lst in results for r in lst]
    reqs = [r for r in all_reqs if r[0] >= warm_until]
    if not reqs:
        return {"error": "no completed requests", "errors": errors}
    lats = sorted(dt for _, dt in reqs)
    measured_wall = max(t for t, _ in reqs) - warm_until
    nbytes = len(reqs) * bs
    def pct(p):
        return lats[min(len(lats) - 1, int(p / 100 * len(lats)))] * 1e3
    return {
        "requests": len(reqs), "bytes": nbytes,
        "bytes_all": len(all_reqs) * bs,   # incl. warmup — for diskstats accounting
        "wall_s": round(measured_wall, 3),
        "gbps": round(nbytes / measured_wall / 1e9, 3),
        "lat_ms": {"p50": round(pct(50), 2), "p90": round(pct(90), 2),
                   "p99": round(pct(99), 2), "max": round(lats[-1] * 1e3, 2)},
        "mode": _ODIRECT_MODE, "errors": errors[:3],
        "total_wall_s": round(wall, 3),
    }


# --------------------------------------------------------------- fio engine
def fio_config(fio_bin: str, target: str, bs: int, qd: int, seconds: float,
               seq: bool = False, size_limit: int | None = None,
               ioengine: str = "libaio") -> dict:
    """One cell via fio (true async QD; ioengine libaio or io_uring where the
    kernel + container policy allow it). --readonly is a hard safety
    interlock: fio refuses any write workload against the target."""
    cmd = [fio_bin, "--name=n0", f"--filename={target}", "--readonly",
           f"--rw={'read' if seq else 'randread'}", f"--bs={bs}",
           f"--iodepth={qd}", f"--ioengine={ioengine}", "--direct=1",
           "--time_based", f"--runtime={int(seconds)}", "--group_reporting",
           "--output-format=json"]
    if size_limit:
        cmd.append(f"--size={size_limit}")
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=seconds + 60)
    if out.returncode != 0:
        return {"error": f"fio rc={out.returncode}", "stderr": out.stderr[-400:],
                "stdout_tail": out.stdout[-300:]}
    try:
        j = json.loads(out.stdout[out.stdout.index("{"):])
    except ValueError:
        return {"error": "fio emitted no json", "stdout": out.stdout[-400:]}
    rd = j["jobs"][0]["read"]
    if "bw_bytes" in rd:            # fio 3.x: bytes, clat_ns
        gbps = rd["bw_bytes"] / 1e9
        cl = rd.get("clat_ns", {})
        scale = 1e-6
        io_bytes = rd.get("io_bytes", 0)
    else:                            # fio 2.x: bw KiB/s, io_bytes KiB, clat usec
        gbps = rd["bw"] * 1024 / 1e9
        cl = rd.get("clat", {})
        scale = 1e-3
        io_bytes = rd.get("io_bytes", 0) * 1024
    pcts = cl.get("percentile", {})
    def pick(key):
        for k, v in pcts.items():
            if abs(float(k) - key) < 0.01:
                return round(v * scale, 2)
        return None
    return {
        "gbps": round(gbps, 3), "iops": rd.get("iops"),
        "bytes_all": io_bytes,
        "lat_ms": {"p50": pick(50.0), "p90": pick(90.0), "p99": pick(99.0),
                   "max": round(cl.get("max", 0) * scale, 2)},
        "fio_version": j.get("fio version"),
    }


# --------------------------------------------------------------------- main
def cmd_probe(args) -> dict:
    info = {
        "host": platform.node(), "kernel": platform.release(),
        "machine": platform.machine(), "python": platform.python_version(),
        "device": device_info(args.target),
        "io_uring": io_uring_available(),
        "fio": None,
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    fio_bin = args.fio_bin or shutil.which("fio") or (
        "/sbin/fio" if os.path.exists("/sbin/fio") else None)
    if fio_bin:
        v = subprocess.run([fio_bin, "--version"], capture_output=True,
                           text=True)
        info["fio"] = {"bin": fio_bin, "version": v.stdout.strip()}
    return info


def cmd_sweep(args) -> dict:
    probe = cmd_probe(args)
    dev = probe["device"]
    lbs = dev.get("logical_block_size", 512)
    align = max(4096, lbs)
    span = dev["size_bytes"]
    if args.span_gib:
        span = min(span, args.span_gib << 30)
    span &= ~(align - 1)
    sizes = [int(s) * MIB for s in args.sizes_mib.split(",")]
    qds = [int(q) for q in args.qds.split(",")]
    engines = args.engines.split(",")
    fio_bin = (probe.get("fio") or {}).get("bin")
    for bsz in sizes:
        assert bsz % lbs == 0, f"request {bsz} not a multiple of lbs {lbs}"

    dsname = dev.get("device")
    cells = []
    for engine in engines:
        if engine.startswith("fio") and not fio_bin:
            print("NOTE: fio requested but not found; skipping", flush=True)
            continue
        for bsz in sizes:
            for qd in qds:
                pre = diskstats(dsname)
                if engine == "pread":
                    r = pread_config(args.target, span, bsz, qd, args.seconds,
                                     align)
                elif engine == "fio":
                    r = fio_config(fio_bin, args.target, bsz, qd, args.seconds)
                elif engine == "fio-uring":
                    r = fio_config(fio_bin, args.target, bsz, qd, args.seconds,
                                   ioengine="io_uring")
                else:
                    raise SystemExit(f"unknown engine {engine}")
                post = diskstats(dsname)
                if pre and post:
                    other = (post["sectors_read"] - pre["sectors_read"]) * 512 \
                        - r.get("bytes_all", r.get("bytes", 0))
                    r["other_read_mbps"] = round(
                        max(0, other) / (post["t"] - pre["t"]) / 1e6, 1)
                    r["other_write_mbps"] = round(
                        (post["sectors_written"] - pre["sectors_written"])
                        * 512 / (post["t"] - pre["t"]) / 1e6, 1)
                cell = {"engine": engine, "bs_mib": bsz // MIB, "qd": qd, **r}
                cells.append(cell)
                print(json.dumps(cell), flush=True)

    # self-pair on the best cell: an instrument that can't reproduce itself
    # to ~1.00x is reporting noise, not bandwidth
    scored = [c for c in cells if "gbps" in c]
    best = max(scored, key=lambda c: c["gbps"]) if scored else None
    if best and args.self_pair:
        r2 = (pread_config(args.target, span, best["bs_mib"] * MIB,
                           best["qd"], args.seconds, align)
              if best["engine"] == "pread" else
              fio_config(fio_bin, args.target, best["bs_mib"] * MIB,
                         best["qd"], args.seconds))
        pair = {"engine": best["engine"], "bs_mib": best["bs_mib"],
                "qd": best["qd"], **r2,
                "self_pair_ratio": round(r2.get("gbps", 0) / best["gbps"], 3)}
        cells.append({"self_pair": pair})
        print(json.dumps({"self_pair": pair}), flush=True)

    if args.seq_control and scored:
        b = max(scored, key=lambda c: c["gbps"])
        r = (fio_config(fio_bin, args.target, b["bs_mib"] * MIB, b["qd"],
                        args.seconds, seq=True) if fio_bin else
             pread_config(args.target, span, b["bs_mib"] * MIB, b["qd"],
                          args.seconds, align, seq=True))
        cells.append({"seq_control": {"bs_mib": b["bs_mib"], "qd": b["qd"], **r}})
        print(json.dumps(cells[-1]), flush=True)

    if best and args.sustain_seconds:
        r = (fio_config(fio_bin, args.target, best["bs_mib"] * MIB, best["qd"],
                        args.sustain_seconds) if best["engine"] == "fio" else
             pread_config(args.target, span, best["bs_mib"] * MIB, best["qd"],
                          args.sustain_seconds, align))
        cells.append({"sustained_best": {
            "seconds": args.sustain_seconds, "bs_mib": best["bs_mib"],
            "qd": best["qd"], **r}})
        print(json.dumps(cells[-1]), flush=True)

    return {"probe": probe, "params": {
        "sizes_mib": sizes and [s // MIB for s in sizes], "qds": qds,
        "seconds": args.seconds, "span_bytes": span, "align": align,
        "engines": engines, "access": "random-aligned"}, "cells": cells}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--target", required=True)
    p.add_argument("--fio-bin", default=None)
    s = sub.add_parser("sweep")
    s.add_argument("--target", required=True)
    s.add_argument("--engines", default="pread,fio")
    s.add_argument("--sizes-mib", default="4,13,25,50")
    s.add_argument("--qds", default="1,4,16,64")
    s.add_argument("--seconds", type=float, default=8.0)
    s.add_argument("--span-gib", type=int, default=0)
    s.add_argument("--self-pair", action="store_true")
    s.add_argument("--seq-control", action="store_true")
    s.add_argument("--sustain-seconds", type=float, default=0.0)
    s.add_argument("--fio-bin", default=None)
    s.add_argument("--out", default=None)
    args = ap.parse_args()
    out = cmd_probe(args) if args.cmd == "probe" else cmd_sweep(args)
    text = json.dumps(out, indent=1)
    if getattr(args, "out", None):
        with open(args.out, "w") as f:
            f.write(text)
        print(f"wrote {args.out}", flush=True)
    else:
        print(text)


if __name__ == "__main__":
    main()
