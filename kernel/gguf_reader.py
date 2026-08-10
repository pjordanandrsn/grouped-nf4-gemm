# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Minimal GGUF v2/v3 container reader — header, metadata, tensor table.

Purpose-built for the k-quant lane (kquant_ref.py): enumerate tensors with
their ggml types and ABSOLUTE byte extents so callers can read exactly one
tensor's bytes — from a local file, an NVMe arena, or an HTTP Range request
(the fixture fetcher parses remote headers from a ranged prefix; nothing here
does I/O beyond what you hand it).

Not a general GGUF toolkit on purpose: no mmap policy, no writing, no
alignment-rewriting. gguf-py exists for that; this reader exists so the
serving path has a dependency-free, auditable parse of exactly the fields it
uses. Jordan has fuzzed GGUF parsers professionally — every length here is
bounds-checked against the buffer before use, and a truncated header raises
`NeedMoreBytes(minimum_total)` rather than guessing.

Layout facts this encodes (GGUF spec, stable v2/v3):
  * little-endian throughout; magic "GGUF", u32 version, u64 tensor_count,
    u64 kv_count, then KVs, then tensor infos, then ALIGNED tensor data.
  * strings are u64-length-prefixed UTF-8; arrays are (u32 elem type, u64 n).
  * tensor info: name, u32 n_dims, u64 ne[n_dims] (ne0 FIRST = contiguous
    dim; logical torch shape is reversed(ne)), u32 ggml_type, u64 offset
    RELATIVE to the data section start.
  * data section starts at align_up(end_of_tensor_infos, alignment) where
    alignment = metadata["general.alignment"] (default 32).
"""
from __future__ import annotations

import struct
from dataclasses import dataclass

from kquant_ref import GGML_DEQUANT, GGML_TYPE_NAMES

GGUF_MAGIC = b"GGUF"
DEFAULT_ALIGNMENT = 32
# Value-type ids from the spec.
_T_U8, _T_I8, _T_U16, _T_I16, _T_U32, _T_I32, _T_F32, _T_BOOL, _T_STR, \
    _T_ARR, _T_U64, _T_I64, _T_F64 = range(13)
_SCALAR_FMT = {_T_U8: "<B", _T_I8: "<b", _T_U16: "<H", _T_I16: "<h",
               _T_U32: "<I", _T_I32: "<i", _T_F32: "<f", _T_BOOL: "<B",
               _T_U64: "<Q", _T_I64: "<q", _T_F64: "<d"}
# Sizes for SIZE-ONLY accounting of types the k-quant lane does not decode —
# so header math is right on any real file even when decode would refuse.
_EXTRA_TYPE_SIZES = {
    2: (32, 18), 3: (32, 20), 6: (32, 22), 7: (32, 24),   # Q4_0/Q4_1/Q5_0/Q5_1
    15: (256, 292),                                        # Q8_K
    16: (256, 66), 17: (256, 74), 18: (256, 98),           # IQ2_XXS/XS, IQ3_XXS
    19: (256, 50), 20: (32, 18), 21: (256, 110),           # IQ1_S, IQ4_NL, IQ3_S
    22: (256, 82), 23: (256, 136), 29: (256, 56),          # IQ2_S, IQ4_XS, IQ1_M
    24: (1, 1), 25: (1, 2), 26: (1, 4), 27: (1, 8), 28: (1, 8),  # I8..I64, F64
}


class NeedMoreBytes(Exception):
    """Header extends past the supplied buffer; `.minimum` is a safe re-fetch size."""

    def __init__(self, minimum: int):
        super().__init__(f"GGUF header needs at least {minimum} bytes")
        self.minimum = minimum


@dataclass(frozen=True)
class GGUFTensorInfo:
    name: str
    ne: tuple[int, ...]          # as stored: ne[0] = contiguous dim
    ggml_type: int
    offset_abs: int              # absolute file offset of this tensor's bytes
    nbytes: int

    @property
    def shape(self) -> tuple[int, ...]:
        """Logical torch shape (ne reversed)."""
        return tuple(reversed(self.ne))

    @property
    def type_name(self) -> str:
        return GGML_TYPE_NAMES.get(self.ggml_type, f"ggml#{self.ggml_type}")


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    alignment: int
    metadata: dict
    tensors: tuple[GGUFTensorInfo, ...]
    data_offset: int

    def tensor(self, name: str) -> GGUFTensorInfo:
        for t in self.tensors:
            if t.name == name:
                return t
        raise KeyError(name)


class _Cursor:
    def __init__(self, buf: bytes):
        self.buf, self.o = buf, 0

    def take(self, n: int) -> bytes:
        if n < 0 or self.o + n > len(self.buf):
            # 2x is a safe overshoot for the *next* fetch; callers loop.
            raise NeedMoreBytes(max(self.o + max(n, 0), len(self.buf) * 2, 4096))
        d = self.buf[self.o:self.o + n]
        self.o += n
        return d

    def scalar(self, fmt: str):
        return struct.unpack(fmt, self.take(struct.calcsize(fmt)))[0]

    def string(self) -> str:
        n = self.scalar("<Q")
        if n > 1 << 32:   # a corrupt length must not drive a huge take()
            raise ValueError(f"unreasonable GGUF string length {n}")
        return self.take(n).decode("utf-8", "replace")

    def value(self, t: int):
        if t == _T_STR:
            return self.string()
        if t == _T_ARR:
            et = self.scalar("<I")
            n = self.scalar("<Q")
            if n > 1 << 34:
                raise ValueError(f"unreasonable GGUF array length {n}")
            return [self.value(et) for _ in range(n)]
        if t not in _SCALAR_FMT:
            raise ValueError(f"unknown GGUF value type {t}")
        v = self.scalar(_SCALAR_FMT[t])
        return bool(v) if t == _T_BOOL else v


def type_size(ggml_type: int, n_elems: int) -> int:
    if ggml_type in GGML_DEQUANT:
        elems, bbytes, _ = GGML_DEQUANT[ggml_type]
    elif ggml_type in _EXTRA_TYPE_SIZES:
        elems, bbytes = _EXTRA_TYPE_SIZES[ggml_type]
    else:
        raise ValueError(f"no size traits for ggml type {ggml_type}")
    assert n_elems % elems == 0, (n_elems, elems, ggml_type)
    return (n_elems // elems) * bbytes


def parse_header(buf: bytes) -> GGUFHeader:
    """Parse a GGUF header from a byte prefix of the file.

    Raises NeedMoreBytes when `buf` is too short — re-fetch at least
    `.minimum` bytes and call again (the fixture fetcher's loop).
    """
    c = _Cursor(buf)
    if c.take(4) != GGUF_MAGIC:
        raise ValueError("not a GGUF file")
    version = c.scalar("<I")
    if version not in (2, 3):
        raise ValueError(f"unsupported GGUF version {version}")
    n_tensors = c.scalar("<Q")
    n_kv = c.scalar("<Q")
    if n_tensors > 1 << 24 or n_kv > 1 << 20:
        raise ValueError(f"unreasonable counts: {n_tensors} tensors, {n_kv} kv")
    md = {}
    for _ in range(n_kv):
        k = c.string()
        t = c.scalar("<I")
        md[k] = c.value(t)
    alignment = int(md.get("general.alignment", DEFAULT_ALIGNMENT) or DEFAULT_ALIGNMENT)
    infos = []
    for _ in range(n_tensors):
        name = c.string()
        nd = c.scalar("<I")
        if nd > 8:
            raise ValueError(f"unreasonable n_dims {nd} for {name!r}")
        ne = tuple(c.scalar("<Q") for _ in range(nd))
        gt = c.scalar("<I")
        rel = c.scalar("<Q")
        n_elems = 1
        for d in ne:
            n_elems *= d
        infos.append((name, ne, gt, rel, n_elems))
    data_offset = (c.o + alignment - 1) // alignment * alignment
    tensors = tuple(
        GGUFTensorInfo(name=name, ne=ne, ggml_type=gt,
                       offset_abs=data_offset + rel,
                       nbytes=type_size(gt, n_elems))
        for name, ne, gt, rel, n_elems in infos)
    return GGUFHeader(version=version, alignment=alignment, metadata=md,
                      tensors=tensors, data_offset=data_offset)


def read_header(path: str, initial: int = 64 << 20) -> GGUFHeader:
    """Parse a local file's header, growing the read if the KV section
    (tokenizer arrays…) outruns the initial guess."""
    want = initial
    while True:
        with open(path, "rb") as f:
            buf = f.read(want)
        try:
            return parse_header(buf)
        except NeedMoreBytes as e:
            if len(buf) < want:      # whole file read and still short: corrupt
                raise ValueError("truncated GGUF file") from e
            want = max(e.minimum, want * 2)


def read_tensor_bytes(path: str, info: GGUFTensorInfo) -> bytes:
    with open(path, "rb") as f:
        f.seek(info.offset_abs)
        data = f.read(info.nbytes)
    if len(data) != info.nbytes:
        raise ValueError(f"short read for {info.name}: {len(data)}/{info.nbytes}")
    return data
