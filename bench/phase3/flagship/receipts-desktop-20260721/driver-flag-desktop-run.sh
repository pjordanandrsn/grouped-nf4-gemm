#!/bin/bash
# flag-desktop-run.sh — ON-POD: fill the tier table's one PROJECTED cell —
# "235B on a desktop (gen4 x16, ~150 GB system RAM): ≈3 tok/s by the waterfall
# arithmetic" — by measuring it. Report-tier replication of the stamped
# flagship on the desktop box class: link microbench -> property suite ->
# Phase-A synthetic (pure-stream ceiling + fused) -> Phase-B REAL checkpoint
# (438 GB download to DISK cache, stream-quantize to ~128 GB pinned, 3 greedy
# prompts). Evidence contract: /root/ab-out + /root/ab-run.log + AB-DONE/AB-FATAL.
set -u -o pipefail
mkdir -p /root/ab-out /root/hf
echo "== FLAG-DESKTOP start $(date -u +%FT%TZ) =="
export DEBIAN_FRONTEND=noninteractive HF_HOME=/root/hf HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.current,pcie.link.width.current --format=csv,noheader | tee /root/ab-out/gpu.txt
lscpu | grep -E '^(Model name|CPU\(s\))' | tee /root/ab-out/cpu.txt
free -g | tee /root/ab-out/ram.txt; df -h /root | tail -1 | tee /root/ab-out/disk.txt

PY=""
for p in python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import torch' 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "NO-TORCH-PYTHON"; echo "AB-FATAL"; exit 0; }
$PY -c "import torch; assert torch.cuda.is_available(); x=torch.zeros(1024,device='cuda')+1; torch.cuda.synchronize(); print('CUDA-OK', torch.cuda.get_device_name(0))" \
  || { echo "CUDA-GATE-FAIL"; echo "AB-FATAL"; exit 0; }

# RAM gate: ~128 GB pinned stacks + residents + OS
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
[ "$RAM_GB" -ge 145 ] || { echo "RAM-GATE-FAIL ${RAM_GB}GB < 145"; echo "AB-FATAL"; exit 0; }
# disk gate: 438 GB shards + workspace
FREE_GB=$(df -BG /root | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$FREE_GB" -ge 520 ] || { echo "DISK-GATE-FAIL ${FREE_GB}GB < 520"; echo "AB-FATAL"; exit 0; }

# LINK BAND GATE: the cell is the gen4-x16 class. Pinned H2D 1 GiB x 10 (the
# harness's own method). gen3-x16/gen4-x8 ~10-14, gen4-x16 ~20-27, gen5 40+.
LINK=$($PY - <<'PYL'
import torch
buf = torch.empty(1<<30, dtype=torch.uint8).pin_memory()
dst = torch.empty(1<<30, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize()
import time
t0=time.time()
for _ in range(10):
    dst.copy_(buf, non_blocking=True)
torch.cuda.synchronize()
print(round(10*(1<<30)/(time.time()-t0)/1e9, 2))
PYL
)
echo "measured pinned H2D: ${LINK} GB/s" | tee /root/ab-out/link.txt
$PY -c "l=float('$LINK'); import sys; sys.exit(0 if 18.0 <= l <= 32.0 else 1)" \
  || { echo "LINK-BAND-FAIL ${LINK} GB/s outside [18,32] (not the gen4-x16 class)"; echo "AB-FATAL"; exit 0; }

cd /root
git clone --depth 1 https://github.com/pjordanandrsn/grouped-nf4-gemm.git g || { echo "CLONE-FAIL"; echo "AB-FATAL"; exit 0; }
cd /root/g && git rev-parse HEAD | tee /root/ab-out/gnf4_sha.txt
$PY -m pip install -q bitsandbytes pytest transformers safetensors "huggingface_hub[hf_transfer]" 2>&1 | tail -1 \
  || { echo "PIP-FAIL"; echo "AB-FATAL"; exit 0; }
$PY -c "import triton, bitsandbytes as b; print('triton', triton.__version__, 'bnb', b.__version__)" | tee /root/ab-out/versions.txt \
  || { echo "DEPS-FAIL"; echo "AB-FATAL"; exit 0; }

echo "== property suite first =="
cd /root/g/kernel && timeout 900 $PY -m pytest test_nf4_grouped.py -q > /root/ab-out/suite.log 2>&1
echo "SUITE rc=$? $(grep -oE '[0-9]+ (passed|failed|error)' /root/ab-out/suite.log | tr '\n' ' ')" | tee -a /root/ab-out/suite.txt

cd /root/g
echo "== PHASE-A synthetic: pure-stream ceiling then fused $(date -u +%TZ) =="
timeout 1800 $PY bench/phase3/offload_decode_235b.py --moe none  --tokens 64 --out /root/ab-out/flagA_none.json  2>&1 | tail -4 || echo "PHASEA-none FAILED"
timeout 1800 $PY bench/phase3/offload_decode_235b.py --moe fused --tokens 64 --out /root/ab-out/flagA_fused.json 2>&1 | tail -4 || echo "PHASEA-fused FAILED"

echo "== PHASE-B real checkpoint (438 GB -> disk cache; stream-quantize; generate) $(date -u +%TZ) =="
timeout 7500 $PY bench/phase3/offload_generate_235b.py --cache /root/hf --out /root/ab-out/flagB_real.json 2>&1 | tail -12 || echo "PHASEB FAILED"

$PY - <<'PYS' | tee /root/ab-out/SUMMARY.txt
import json, glob, os
print("link:", open("/root/ab-out/link.txt").read().strip())
for f in sorted(glob.glob("/root/ab-out/flag*.json")):
    try:
        d = json.load(open(f))
        toks = d.get("tok_per_s") or d.get("tokens_per_s") or d.get("decode_toks")
        print(os.path.basename(f), "->", {k: d[k] for k in list(d)[:8]} if toks is None else f"{toks} tok/s")
    except Exception as e:
        print(os.path.basename(f), "parse-error", e)
PYS
echo "AB-DONE — evidence complete $(date -u +%FT%TZ)"
