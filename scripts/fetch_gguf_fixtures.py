#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Range-fetch small REAL tensors from released Glimmer GGUFs as test fixtures.

Parses each remote file's GGUF header over HTTP Range requests (no full
download — these files are 9-18 GB), picks the smallest tensor per ggml type,
downloads just those byte extents, and writes them plus a sha256-pinned
manifest to kernel/fixtures/gguf/ (gitignored — fixtures are re-fetchable,
provenance lives in the manifest, and the wheel must not carry model bytes).

Usage:
    HF_TOKEN=... python3 scripts/fetch_gguf_fixtures.py
    GNF4_GGUF_FIXTURES=kernel/fixtures/gguf python3 -m pytest kernel/test_kquant_ref.py

Apache-2.0 sources (byte extracts for test provenance, credited):
    meta-models/Muse-Glimmer-30B-GGUF   (kquant-dynamic)
    unsloth/Muse-Glimmer-30B-GGUF       (UD-Q2_K_XL)
"""
import datetime as dt
import hashlib
import json
import os
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "kernel"))
from gguf_reader import NeedMoreBytes, parse_header  # noqa: E402
from kquant_ref import GGML_TYPE_NAMES  # noqa: E402

SOURCES = [
    ("meta-dynamic",
     "https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/main/muse-glimmer-30B-kquant-dynamic.gguf"),
    ("unsloth-ud-q2kxl",
     "https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/main/Muse-Glimmer-30B-UD-Q2_K_XL.gguf"),
]
OUT = Path(__file__).resolve().parent.parent / "kernel" / "fixtures" / "gguf"
MAX_FIXTURE_BYTES = 6 << 20


def _fetch_range(url: str, start: int, length: int) -> bytes:
    req = urllib.request.Request(url, headers={
        "Range": f"bytes={start}-{start + length - 1}",
        "Authorization": f"Bearer {os.environ.get('HF_TOKEN', '')}",
    })
    with urllib.request.urlopen(req, timeout=120) as r:
        return r.read()


def _remote_header(url: str):
    want = 64 << 20
    while True:
        buf = _fetch_range(url, 0, want)
        try:
            return parse_header(buf)
        except NeedMoreBytes as e:
            want = max(e.minimum, want * 2)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, url in SOURCES:
        hdr = _remote_header(url)
        by_type = {}
        for t in hdr.tensors:
            if t.nbytes > MAX_FIXTURE_BYTES:
                continue
            cur = by_type.get(t.ggml_type)
            if cur is None or t.nbytes < cur.nbytes:
                by_type[t.ggml_type] = t
        print(f"{label}: {len(hdr.tensors)} tensors, "
              f"types {[GGML_TYPE_NAMES.get(g, g) for g in sorted(by_type)]}")
        # Types whose every whole tensor exceeds the cap (the XL file's Q2_K/
        # Q3_K live only in giant ffn matrices) still get REAL bytes: a prefix
        # of whole superblocks is independently decodable, so slice one. The
        # entry's shape is then [n_blocks, block_elems], not the tensor's.
        from kquant_ref import GGML_DEQUANT  # local import keeps header light
        present = {t.ggml_type for t in hdr.tensors}
        for gtype in sorted(present - set(by_type)):
            if gtype not in GGML_DEQUANT:
                continue
            elems, bbytes, _ = GGML_DEQUANT[gtype]
            t = min((x for x in hdr.tensors if x.ggml_type == gtype),
                    key=lambda x: x.nbytes)
            n_blocks = max(1, (1 << 20) // bbytes)
            by_type[gtype] = (t, n_blocks)
        for gtype, tv in sorted(by_type.items()):
            t, n_blocks = tv if isinstance(tv, tuple) else (tv, None)
            if n_blocks is None:
                length, shape, partial = t.nbytes, list(t.shape), False
            else:
                elems, bbytes, _ = GGML_DEQUANT[gtype]
                length, shape, partial = n_blocks * bbytes, [n_blocks, elems], True
            raw = _fetch_range(url, t.offset_abs, length)
            assert len(raw) == length, (t.name, len(raw), length)
            fname = f"{label}__{GGML_TYPE_NAMES.get(gtype, gtype)}__{t.name.replace('.', '_')}.bin"
            (OUT / fname).write_bytes(raw)
            entries.append({
                "file": fname, "source": label, "url": url, "tensor": t.name,
                "ggml_type": gtype,
                "type_name": GGML_TYPE_NAMES.get(gtype, str(gtype)),
                "shape": shape, "nbytes": length, "partial": partial,
                "sha256": hashlib.sha256(raw).hexdigest(),
            })
            print(f"  {t.name:40s} {GGML_TYPE_NAMES.get(gtype, gtype):5} "
                  f"{length / 1024:8.0f} KiB{' (partial)' if partial else ''}")
    manifest = {
        "fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "entries": entries,
    }
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(entries)} fixtures + manifest -> {OUT}")


if __name__ == "__main__":
    main()
