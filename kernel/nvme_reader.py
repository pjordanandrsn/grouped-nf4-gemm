# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""N2 — the arena read path: O_DIRECT reads with async submission into
aligned (optionally pinned) landing buffers.

Engine choice, measured not assumed: io_uring is the preferred submission
path where the kernel offers it, but it is *absent on real boxes* (ENOSYS
measured on 6.6.32-qnap — a 2026 kernel; see docs/nvme-ceilings.md), so the
portable core is a thread pool over `os.preadv`, one O_DIRECT fd per worker.
N0 measured that core at 99% of the device link at QD>=4 — the thread pool
is not a compromise at expert-row sizes, it is the ceiling. An io_uring
engine can drop in behind the same interface when a target box provides it;
nothing above the reader changes.

Zero-copy discipline: `read_row(layer, expert, dst)` DMA-lands bytes
directly in `dst` — the caller's landing buffer, which in the engine is a
pinned row the gather kernel then reads over UVA. No bounce, no assembly;
the reader never allocates data buffers of its own.

Alignment discipline (the syscall fails outright otherwise): the bake pads
every row offset and stride to the device block size; `dst` must be
page-aligned (torch pinned tensors and mmap both are — asserted, so a
misaligned buffer is a named error, not an EINVAL mystery).

Fallback: if O_DIRECT cannot be opened (platform, filesystem, container),
the reader degrades to buffered reads with a LOUD one-time warning naming
the consequence — the page cache would silently duplicate the DRAM tier and
hand eviction policy to the kernel, which is exactly the decision the
caller. Correctness is unaffected; the receipts say which
mode ran.
"""
from __future__ import annotations

import os
import platform
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

from nvme_arena import load_index, row_offset


def open_direct_ro(path: str):
    """(fd, mode): O_DIRECT where the platform has it, F_NOCACHE on macOS,
    buffered as the loud last resort."""
    if hasattr(os, "O_DIRECT"):
        try:
            return os.open(path, os.O_RDONLY | os.O_DIRECT), "O_DIRECT"
        except OSError as e:
            return os.open(path, os.O_RDONLY), \
                f"buffered (O_DIRECT failed: {e.strerror})"
    if platform.system() == "Darwin":
        import fcntl
        fd = os.open(path, os.O_RDONLY)
        fcntl.fcntl(fd, 48, 1)  # F_NOCACHE
        return fd, "F_NOCACHE"
    return os.open(path, os.O_RDONLY), "buffered (no O_DIRECT)"


def cpu_budget() -> int:
    """CPUs this process may actually use.

    Order matters, and the obvious calls are the wrong ones. Inside a container
    with a CPU quota but no cpuset, ``os.cpu_count()`` and
    ``os.sched_getaffinity`` both report the HOST's cores: on a RunPod L40S box
    they said **256** while ``cpu.max`` was ``2720000 100000`` — a real budget of
    **27.2**. So read the cgroup first and fall back only when there is no quota.
    """
    try:                                              # cgroup v2
        quota, period = open("/sys/fs/cgroup/cpu.max").read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:                                              # cgroup v1
        q = int(open("/sys/fs/cgroup/cpu/cpu.cfs_quota_us").read())
        p = int(open("/sys/fs/cgroup/cpu/cpu.cfs_period_us").read())
        if q > 0 and p > 0:
            return max(1, q // p)
    except (OSError, ValueError):
        pass
    try:
        return len(os.sched_getaffinity(0))           # respects taskset
    except AttributeError:                            # macOS, Windows
        return os.cpu_count() or 4


def default_qd(cpus: int | None = None) -> int:
    """Queue depth for this host: ``clamp(cpus // 4, 4, 16)``.

    A fixed ``qd=4`` was measured optimal on a 12-core box that sat at load ~9.8
    throughout, with 8 and 16 coming back *worse*. Re-measured on an idle 32-vCPU
    L40S against the same arena and the same scattered pattern, that inverts:

        qd=1  2.04 GB/s   qd=4  5.31   qd=8  5.95 (+12%)   qd=16  6.13 (+15%)

    So 4 was tuned to a CPU-starved regime rather than to the device. The floor
    stays at 4 and the divisor is deliberately coarse: this only departs from the
    old constant above ~20 CPUs, which is the only region where more depth was
    actually measured to help. Small and mid-size hosts get exactly what they got
    before, so the change cannot regress them.

    Pass ``qd=`` explicitly to override; this is only the default.
    """
    n = cpu_budget() if cpus is None else cpus
    return min(16, max(4, n // 4))


class ArenaReader:
    """Thread-pool async reader over a baked arena. qd = max in-flight reads.

    ``qd=None`` (the default) sizes it from the host's CPU budget — see
    :func:`default_qd`.
    """

    def __init__(self, arena_path: str, index: dict | None = None, *,
                 qd: int | None = None, warn=print):
        qd = default_qd() if qd is None else int(qd)
        if qd < 1:
            raise ValueError(f"qd must be >= 1, got {qd}")
        self.path = arena_path
        self.index = index or load_index(arena_path)
        self.row_bytes = self.index["row_bytes"]
        self.row_stride = self.index["row_stride"]
        self.align = self.index["align"]
        self.qd = qd
        self._tls = threading.local()
        self._fds: list[int] = []
        self._fds_lock = threading.Lock()
        self._warned = False
        self._warn = warn
        self.mode = None
        self._stats_lock = threading.Lock()
        self.reads = 0
        self.bytes_read = 0
        self._pool = ThreadPoolExecutor(max_workers=qd,
                                        thread_name_prefix="arena-read")
        # open one fd up front so `mode` is known (and warned) at init
        fd, mode = open_direct_ro(self.path)
        os.close(fd)
        self._note_mode(mode)

    def _note_mode(self, mode: str):
        if self.mode is None:
            self.mode = mode
        if "buffered" in mode and not self._warned:
            self._warned = True
            self._warn(
                f"WARNING: arena reads are {mode}: the page cache will "
                "shadow-copy the DRAM tier and the kernel — not the placement "
                "engine — will own eviction. Throughput numbers from this "
                "mode are not device measurements.", file=sys.stderr)

    def _fd(self) -> int:
        fd = getattr(self._tls, "fd", None)
        if fd is None:
            fd, mode = open_direct_ro(self.path)
            self._tls.fd = fd
            with self._fds_lock:
                self._fds.append(fd)
            self._note_mode(mode)
        return fd

    def _read(self, offset: int, dst: memoryview) -> int:
        n = len(dst)
        if n < self.row_stride:
            raise ValueError(f"dst {n} B < row_stride {self.row_stride}")
        dst = dst[: self.row_stride]
        fd = self._fd()
        done = 0
        while done < self.row_stride:
            got = os.preadv(fd, [dst[done:]], offset + done)
            if got <= 0:
                raise EOFError(f"arena short read at {offset + done}")
            done += got
        with self._stats_lock:
            self.reads += 1
            self.bytes_read += done
        return done

    @staticmethod
    def _advance(views, done):
        """The suffix of an iovec after ``done`` bytes have landed.

        `preadv` may return short, and the retry must resume INSIDE the buffer
        it stopped in — not at the start of the next one. Getting this wrong
        silently duplicates or drops a segment's bytes, which is the kind of
        corruption that reads as a plausible tensor.
        """
        out = []
        for v in views:
            if done >= len(v):
                done -= len(v)
                continue
            out.append(v[done:] if done else v)
            done = 0
        return out

    def _readv(self, offset: int, views) -> int:
        fd = self._fd()
        done = 0
        while done < self.row_stride:
            rest = self._advance(views, done)
            got = os.preadv(fd, rest, offset + done)
            if got <= 0:
                raise EOFError(f"arena short read at {offset + done}")
            done += got
        with self._stats_lock:
            self.reads += 1
            self.bytes_read += done
        return done

    def read_row_scatter(self, layer: int, expert: int, views):
        """Async: read one row, SCATTERING it across ``views`` in file order.

        The kernel writes each destination by DMA, so a caller that wants the
        row split by segment never copies it on the CPU. That matters more than
        it sounds: a CPU write to pinned memory makes the FOLLOWING H2D ~6x
        slower (measured 70.5 ms vs 11.65 ms for the same 281 MB), so a host
        copy is charged twice — once to make it, once as a penalty on the
        transfer (#73).

        ``views`` must cover exactly ``row_stride`` bytes; pad with scratch if
        the segments do not. Under O_DIRECT every base and length must be
        ``align``-aligned, which is checked here rather than left to EINVAL.
        """
        total = sum(len(v) for v in views)
        if total != self.row_stride:
            raise ValueError(
                f"scatter views cover {total} B but a row is {self.row_stride} B; "
                "include scratch for inter-segment gaps and trailing padding")
        if self.mode == "O_DIRECT":
            for i, v in enumerate(views):
                check_aligned(v, self.align)
                if len(v) % self.align:
                    raise ValueError(
                        f"scatter view {i} is {len(v)} B, not a multiple of "
                        f"{self.align}; O_DIRECT would EINVAL")
        off = row_offset(self.index, layer, expert)
        return self._pool.submit(self._readv, off, list(views))

    def read_row(self, layer: int, expert: int, dst: memoryview):
        """Async: returns a Future resolving to bytes read (== row_stride).
        dst must be page-aligned and >= row_stride long; bytes land in
        dst[:row_bytes] with the bake's segment geometry."""
        check_aligned(dst, self.align)
        off = row_offset(self.index, layer, expert)
        return self._pool.submit(self._read, off, dst)

    def read_row_sync(self, layer: int, expert: int, dst: memoryview) -> int:
        check_aligned(dst, self.align)
        return self._read(row_offset(self.index, layer, expert), dst)

    def traffic(self) -> dict:
        return {"reads": self.reads, "bytes_read": self.bytes_read,
                "mode": self.mode}

    def close(self):
        self._pool.shutdown(wait=True)
        with self._fds_lock:
            for fd in self._fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
            self._fds.clear()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def buffer_address(mv: memoryview) -> int:
    """Address of a memoryview's first byte (ctypes, no numpy needed)."""
    import ctypes
    return ctypes.addressof(ctypes.c_char.from_buffer(mv))


def check_aligned(mv: memoryview, align: int):
    addr = buffer_address(mv)
    if addr % align:
        raise ValueError(
            f"landing buffer address {addr:#x} not {align}-aligned — O_DIRECT "
            "would EINVAL. Allocate via mmap or a torch pinned tensor (both "
            "page-aligned), not bytes()/bytearray().")


def alloc_landing(n_bytes: int, *, pinned: bool = False, align: int = 4096):
    """Page-aligned landing buffer. pinned=True requires torch+CUDA (the
    engine path: gather reads it over UVA); tests use plain mmap. Returns
    (memoryview, keepalive) — hold keepalive as long as the view lives.

    A pinned tensor is NOT reliably page-aligned. PyTorch's caching host
    allocator SUBALLOCATES: a fresh `pin_memory()` lands on a page boundary, but
    once other pinned blocks exist it can return an interior offset — measured
    1024 B off on 2026-07-30, immediately after an engine pinned its cold expert
    stacks, which made O_DIRECT reads EINVAL far from the cause. So over-allocate
    by one alignment unit and hand back an aligned sub-view rather than trusting
    the allocator.
    """
    if pinned:
        import torch
        t = torch.empty(n_bytes + align, dtype=torch.uint8).pin_memory()
        mv = memoryview(t.numpy())
        pad = (-buffer_address(mv)) % align
        mv = mv[pad:pad + n_bytes]
        check_aligned(mv, align)
        return mv, t
    import mmap
    m = mmap.mmap(-1, n_bytes)
    return memoryview(m), m
