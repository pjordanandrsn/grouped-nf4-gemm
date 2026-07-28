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


class ArenaReader:
    """Thread-pool async reader over a baked arena. qd = max in-flight reads."""

    def __init__(self, arena_path: str, index: dict | None = None, *,
                 qd: int = 4, warn=print):
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


def alloc_landing(n_bytes: int, *, pinned: bool = False):
    """Page-aligned landing buffer. pinned=True requires torch+CUDA (the
    engine path: gather reads it over UVA); tests use plain mmap. Returns
    (memoryview, keepalive) — hold keepalive as long as the view lives."""
    if pinned:
        import torch
        t = torch.empty(n_bytes, dtype=torch.uint8).pin_memory()
        mv = memoryview(t.numpy())
        check_aligned(mv, 4096)
        return mv, t
    import mmap
    m = mmap.mmap(-1, n_bytes)
    return memoryview(m), m
