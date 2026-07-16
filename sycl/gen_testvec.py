# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
# Dump a small NF4 grouped-gemv test vector whose ORACLE is the parent kernel's
# canonical reference (dequant_ref + real bnb-quantized weights), so the SYCL
# port is checked against the identical numerics the Triton kernel is gated on.
import os
import struct
import sys

import numpy as np
import torch

sys.path.insert(0, "/work/v6/kernel")
from nf4_grouped import BLOCKSIZE, dequant_ref, repack_from_bnb  # noqa: E402
import bitsandbytes.functional as F  # noqa: E402

torch.manual_seed(0)
# Defaults = the small M1 correctness vector; env overrides let a perf-sized
# vector be dumped for the M2 bench (the activation-reuse win scales with N).
E = int(os.environ.get("TV_E", 4))
N = int(os.environ.get("TV_N", 128))
K = int(os.environ.get("TV_K", 128))
G = int(os.environ.get("TV_G", 3))
OUT = os.environ.get("TV_OUT", "/work/testvec.bin")
assert K % BLOCKSIZE == 0, "K must be a multiple of 64"
packed, states = [], []
for _ in range(E):
    w = torch.randn(N, K, dtype=torch.bfloat16)
    q, st = F.quantize_4bit(w, blocksize=BLOCKSIZE, quant_type="nf4")
    packed.append(q); states.append(st)
B, absmax = repack_from_bnb(packed, states, N, K)   # B[E,N,K/2] u8, absmax[E,N,K/64] f32
eids = [(g * 7 + 1) % E for g in range(G)]           # group -> expert (valid for any E,G)
acts = torch.randn(G, K, dtype=torch.float32)
exp = torch.empty(G, N, dtype=torch.float32)
for g in range(G):
    w = dequant_ref(B[eids[g]], absmax[eids[g]], N, K).float()   # [N,K], the canonical decode
    exp[g] = w @ acts[g]

with open(OUT, "wb") as f:
    f.write(struct.pack("<4i", E, N, K, G))
    f.write(np.asarray(eids, np.int32).tobytes())
    f.write(B.cpu().numpy().astype(np.uint8).tobytes())
    f.write(absmax.cpu().numpy().astype(np.float32).tobytes())
    f.write(acts.cpu().numpy().astype(np.float32).tobytes())
    f.write(exp.cpu().numpy().astype(np.float32).tobytes())
print(f"testvec written: E={E} N={N} K={K} G={G} eids={eids}")
