"""Real-bytes arena round-trip on a SLICE of the released Kimi-K3.

The toy fixtures cannot catch a wrong name_template or a wrong `kinds` order --
they were built from the same constants they test. This bakes a few experts of
the ACTUAL 1.56 TB release and checks the arena against the shipped bytes,
which is the cheap way to de-risk a 1446.5 GB bake.
"""
import json, os, struct, sys, time
sys.path.insert(0, "/share/ZFS2_DATA/k3slice/kernel")
from nvme_arena import bake_expert_tensors, load_index, row_offset, verify   # noqa: E402
from nvme_reader import ArenaReader, alloc_landing                            # noqa: E402

SNAP = "/share/ZFS532_DATA/hf-models/moonshotai_Kimi-K3"
OUT = "/share/ZFS2_DATA/k3slice/k3-slice.arena"
TPL = "language_model.model.layers.{layer}.block_sparse_moe.experts.{expert}.{kind}"
KINDS = ("w1.weight_packed", "w1.weight_scale", "w3.weight_packed",
         "w3.weight_scale", "w2.weight_packed", "w2.weight_scale")
LAYERS, NEXP = [1, 2], 4

t0 = time.time()
idx = bake_expert_tensors(SNAP, OUT, name_template=TPL, kinds=KINDS,
                          layers=LAYERS, limit_experts=NEXP, workers=4)
print(f"\nbake: {time.time()-t0:.1f}s  rows={len(idx['rows'])} "
      f"row_bytes={sum(g['length'] for g in idx['segments'])} "
      f"stride={idx['row_stride']}")
for g in idx["segments"]:
    print(f"  seg {g['suffix']:20s} off={g['seg_off']:>9} len={g['length']:>9} "
          f"shape={g['shape_per_expert']} {g['dtype']}")

print("\n=== verify(): re-hash every segment against its source ===")
v = verify(OUT)
print(" ", json.dumps(v)[:300] if isinstance(v, dict) else v)

# ---- independent check: arena bytes == safetensors byte ranges --------------
# verify() is the module checking itself. This re-derives the source ranges from
# the shard headers directly and compares raw bytes, so a bug shared by bake and
# verify cannot hide.
print("\n=== independent: arena row bytes vs shard byte ranges ===")
wm = json.load(open(os.path.join(SNAP, "model.safetensors.index.json")))["weight_map"]
hdr_cache = {}
def src_bytes(name):
    shard = wm[name]
    p = os.path.join(SNAP, shard)
    if p not in hdr_cache:
        with open(p, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            hdr_cache[p] = (json.loads(f.read(n)), 8 + n)
    hdr, base = hdr_cache[p]
    lo, hi = hdr[name]["data_offsets"]
    with open(p, "rb") as f:
        f.seek(base + lo)
        return f.read(hi - lo)

index = load_index(OUT)
rdr = ArenaReader(OUT, qd=4)
mv, keep = alloc_landing(index["row_stride"])
segs = {g["suffix"]: g for g in index["segments"]}
checked = mismatch = 0
for lay in LAYERS:
    for e in range(NEXP):
        rdr.read_row_sync(lay, e, mv)
        for suf, g in segs.items():
            got = bytes(mv[g["seg_off"]:g["seg_off"] + g["length"]])
            want = src_bytes(TPL.format(layer=lay, expert=e, kind=suf))
            checked += 1
            if got != want:
                mismatch += 1
                print(f"  MISMATCH layer={lay} expert={e} {suf}")
print(f"  {checked} segments compared, {mismatch} mismatches")

# ---- negative control -------------------------------------------------------
print("\n=== negative control: corrupt one arena byte, must be caught ===")
off = row_offset(index, LAYERS[0], 1)
with open(OUT, "r+b") as f:
    f.seek(off); b = f.read(1); f.seek(off); f.write(bytes([b[0] ^ 0xFF]))
rdr.read_row_sync(LAYERS[0], 1, mv)
g = segs["w1.weight_packed"]
got = bytes(mv[g["seg_off"]:g["seg_off"] + g["length"]])
want = src_bytes(TPL.format(layer=LAYERS[0], expert=1, kind="w1.weight_packed"))
print(f"  corrupted row differs from source: {got != want}  <- must be True")
with open(OUT, "r+b") as f:
    f.seek(off); f.write(b)
rdr.close()
print(f"\nRESULT: {'PASS' if mismatch == 0 else 'FAIL'} "
      f"({checked} real segments byte-identical to the release)")
