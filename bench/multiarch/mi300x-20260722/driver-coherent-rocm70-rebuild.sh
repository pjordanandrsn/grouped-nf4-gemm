#!/bin/bash
# mi300_fix70b.sh — rev6: rev5 retry with the debian-dep landmine defused.
# torch swap must be --no-deps (pip aborts uninstalling debian-owned
# typing_extensions otherwise and SILENTLY leaves torch 6.4). Also swap
# pytorch-triton-rocm to torch 2.10's matching build. Loader then wants
# rocm70 — which the rev5 rebuild already produced.
set -u -o pipefail
S=/root/MI300_STATE
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1
echo "=== REV6 torch70 swap $(date -u +%FT%TZ) ===" >> $S
python3.12 -m pip install -q --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/rocm7.0 >> $S 2>&1
python3.12 -m pip install -q --force-reinstall --no-deps pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm7.0 >> $S 2>&1
python3.12 -c "import torch; print('torch', torch.__version__, '| hip', torch.version.hip, '| dev', torch.cuda.get_device_name(0))" >> $S 2>&1 \
  || { echo "TORCH70-FATAL" >> $S; echo "AB-FATAL"; exit 0; }
python3.12 -c "import triton; print('triton', triton.__version__)" >> $S 2>&1
cd /root/g
python3.12 - <<'PY' >> $S 2>&1 || { echo "BNB70-PROBE-FAIL" >> $S; echo "AB-FATAL"; exit 0; }
import torch
from bitsandbytes import functional as F
import bitsandbytes as b
w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
q, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4")
d = F.dequantize_4bit(q, st)
assert torch.isfinite(d).all()
print("bnb70 GPU bf16 quant+dequant OK |", b.__version__)
PY
echo "=== property suite (rev6) ===" >> $S
python3.12 -m pytest kernel/test_nf4_grouped.py -q > /root/mi300_suite.log 2>&1
echo "SUITE rc=$? $(grep -oE '[0-9]+ (passed|failed|error)' /root/mi300_suite.log | tr '\n' ' ')" >> $S
echo "=== census (rev6) ===" >> $S
python3.12 bench/phase1/harness.py --regimes prefill_s2048 decode_bs1 \
  --backends dequant_grouped fused_nf4 fused_v5loop --iters 20 --no-energy \
  --out /root/mi300_census.json > /root/mi300_census.log 2>&1
echo "CENSUS rc=$?" >> $S
echo ALLDONE >> $S
