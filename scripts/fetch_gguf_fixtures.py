#!/usr/bin/env python3
# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Range-fetch small REAL tensors from released Glimmer GGUFs as test fixtures.

Parses each remote file's GGUF header over HTTP Range requests (no full
download — these files are 9-18 GB), picks the smallest tensor per ggml type,
downloads just those byte extents, and writes them plus a sha256-pinned
manifest to kernel/fixtures/gguf/ (gitignored — fixtures are re-fetchable,
provenance lives in the manifest, and the wheel must not carry model bytes).

Usage:
    HF_TOKEN=... python3 scripts/fetch_gguf_fixtures.py          # attention-era set -> kernel/fixtures/gguf/
    HF_TOKEN=... python3 scripts/fetch_gguf_fixtures.py --ffn    # expert / FFN set  -> kernel/fixtures/gguf-ffn/
    GNF4_GGUF_FIXTURES=kernel/fixtures/gguf python3 -m pytest kernel/test_kquant_ref.py
    GNF4_GGUF_FIXTURES=kernel/fixtures/gguf:kernel/fixtures/gguf-ffn python3 -m pytest kernel/test_kquant_backward.py

Apache-2.0 sources (byte extracts for test provenance, credited):
    meta-models/Muse-Glimmer-30B-GGUF   (kquant-dynamic)
    unsloth/Muse-Glimmer-30B-GGUF       (UD-Q2_K_XL; Q8_0 with --ffn)
    unsloth/Qwen3-30B-A3B-GGUF          (--ffn: Q3_K_M, Q4_K_M, Q5_K_M, Q6_K, Q8_0)
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
# A source renamed upstream after its fixtures were pinned. The original URL stays recorded; the bytes are pinned by
# sha256, and the renamed file was checked byte-identical for every pinned tensor on 2026-10-10.
RENAMED = {
    "meta-dynamic": {
        "renamed_to": "https://huggingface.co/meta-models/Muse-Glimmer-30B-GGUF/resolve/"
                      "70bf1b61ac09f91b24d39038091b41c582bc5d7a/Muse-Glimmer-30B-KQuant-Dynamic-Q4_K_XL.gguf",
        "note": "source renamed upstream; bytes pinned by sha256 (byte-identical at renamed_to, checked 2026-10-10)",
    },
}
# --ffn: REAL feed-forward bytes for the backward harness (kernel/test_kquant_backward.py). The MoE source gives whole
# expert matrices; the dense source a prefix of whole rows. Revisions are pinned in the URLs.
_QWEN = "https://huggingface.co/unsloth/Qwen3-30B-A3B-GGUF/resolve/d5b1d57bd0b504ac62ae6c725904e96ef228dc74/"
_GLIM = "https://huggingface.co/unsloth/Muse-Glimmer-30B-GGUF/resolve/faa5b025c584459c13febfa5c59883516710ae39/"
FFN_SOURCES = [
    ("qwen3-30b-a3b-Q4_K_M", _QWEN + "Qwen3-30B-A3B-Q4_K_M.gguf", "moe"),
    ("qwen3-30b-a3b-Q5_K_M", _QWEN + "Qwen3-30B-A3B-Q5_K_M.gguf", "moe"),
    ("qwen3-30b-a3b-Q6_K", _QWEN + "Qwen3-30B-A3B-Q6_K.gguf", "moe"),
    ("qwen3-30b-a3b-Q8_0", _QWEN + "Qwen3-30B-A3B-Q8_0.gguf", "moe"),
    ("qwen3-30b-a3b-Q3_K_M", _QWEN + "Qwen3-30B-A3B-Q3_K_M.gguf", "moe"),
    ("glimmer-UD-Q2_K_XL", _GLIM + "Muse-Glimmer-30B-UD-Q2_K_XL.gguf", "dense"),
    ("glimmer-Q8_0", _GLIM + "Muse-Glimmer-30B-Q8_0.gguf", "dense"),
]
FFN_EXPERTS = (0, 77)
FFN_PREFIX_ROWS = 256
OUT = Path(__file__).resolve().parent.parent / "kernel" / "fixtures" / "gguf"
OUT_FFN = Path(__file__).resolve().parent.parent / "kernel" / "fixtures" / "gguf-ffn"
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
            if len(buf) < want:
                # The server returned everything it has and the header still
                # does not parse: truncated/corrupt object. Growing the request
                # cannot help — the local read_header has this same guard.
                raise ValueError(f"truncated GGUF at {url} ({len(buf)} bytes)") from e
            want = max(e.minimum, want * 2)


def main_ffn():
    """--ffn: expert / FFN matrices per quant type, written to kernel/fixtures/gguf-ffn/ with a sha256 manifest."""
    from kquant_ref import GGML_DEQUANT
    OUT_FFN.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, url, kind in FFN_SOURCES:
        hdr = _remote_header(url)
        seen = set()
        for t in hdr.tensors:
            if (t.ggml_type not in GGML_DEQUANT or GGML_DEQUANT[t.ggml_type][2] is None or "ffn" not in t.name
                    or (kind == "moe" and "_exps" not in t.name)):
                continue
            role = t.name.split(".")[2]
            if (t.ggml_type, role) in seen:
                continue
            seen.add((t.ggml_type, role))
            elems, bbytes, _ = GGML_DEQUANT[t.ggml_type]
            row_bytes = t.shape[-1] // elems * bbytes
            if kind == "moe":
                _, n, k = t.shape
                picks = [(f"e{e}", t.offset_abs + e * n * row_bytes, n * row_bytes, [n, k]) for e in FFN_EXPERTS]
            else:
                picks = [("rows0", t.offset_abs, FFN_PREFIX_ROWS * row_bytes, [FFN_PREFIX_ROWS, t.shape[-1]])]
            for tag, off, length, shape in picks:
                raw = _fetch_range(url, off, length)
                assert len(raw) == length, (t.name, len(raw), length)
                fname = f"{label}__{GGML_TYPE_NAMES[t.ggml_type]}__{t.name.replace('.', '_')}__{tag}.bin"
                (OUT_FFN / fname).write_bytes(raw)
                entries.append({"file": fname, "source": label, "url": url, "tensor": t.name, "slice": tag,
                                "ggml_type": t.ggml_type, "type_name": GGML_TYPE_NAMES[t.ggml_type], "shape": shape,
                                "byte_offset": off, "nbytes": length, "partial": kind == "dense",
                                "sha256": hashlib.sha256(raw).hexdigest()})
                print(f"  {fname} {shape}")
    manifest = {"fetched": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "entries": entries}
    (OUT_FFN / "manifest.json").write_text(json.dumps(manifest, indent=1))
    print(f"wrote {len(entries)} fixtures + manifest -> {OUT_FFN}")


def main():
    if "--ffn" in sys.argv[1:]:
        return main_ffn()
    OUT.mkdir(parents=True, exist_ok=True)
    entries = []
    for label, orig_url in SOURCES:
        url = RENAMED.get(label, {}).get("renamed_to", orig_url)
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
                "file": fname, "source": label, "url": orig_url, **RENAMED.get(label, {}), "tensor": t.name,
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
