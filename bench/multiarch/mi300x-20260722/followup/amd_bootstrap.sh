#!/bin/bash
# amd_bootstrap.sh — ON THE MI300X BOX. Distilled known-good stack (the 6-attempt
# census recipe): torch 2.10+rocm7.0 (--no-deps, dodges the debian typing_extensions
# abort) + matching pytorch-triton-rocm, bnb built from HIP source (ROCm PATH +
# cmake<4), GPU bf16 oracle verify. Writes /root/BOOTSTRAP with a final
# BOOTSTRAP-OK / BOOTSTRAP-FAIL marker. Idempotent-ish; safe to re-run.
set -u -o pipefail
S=/root/BOOTSTRAP
export DEBIAN_FRONTEND=noninteractive PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2 HIP_PLATFORM=amd
export PATH=/opt/rocm-7.0.2/bin:/opt/rocm-7.0.2/llvm/bin:/opt/rocm/bin:$PATH
echo "=== amd_bootstrap start $(date -u +%FT%TZ) ===" > $S
rocm-smi --showproductname 2>/dev/null | head -3 >> $S
echo "hipcc: $(command -v hipcc) | rocm trees: $(ls -d /opt/rocm* 2>/dev/null | tr '\n' ' ')" >> $S

# torch 2.10 + rocm7.0 (coherent with the box's ROCm 7.x driver) — --no-deps is
# MANDATORY (pip aborts uninstalling debian-owned typing_extensions otherwise)
python3.12 -m pip install -q --force-reinstall --no-deps torch --index-url https://download.pytorch.org/whl/rocm7.0 >> $S 2>&1
python3.12 -m pip install -q --force-reinstall --no-deps pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm7.0 >> $S 2>&1
python3.12 -m pip install -q numpy "cmake<4" ninja pytest safetensors >> $S 2>&1
python3.12 -c "import torch,triton; print('torch', torch.__version__, '| hip', torch.version.hip, '| triton', triton.__version__, '| dev', torch.cuda.get_device_name(0), '| cu', torch.cuda.get_device_properties(0).multi_processor_count)" >> $S 2>&1 \
  || { echo "TORCH-FAIL" >> $S; echo "BOOTSTRAP-FAIL" >> $S; exit 0; }

# bnb from HIP source (the rocm70 lib resolves against torch's loaded runtime)
if [ ! -d /root/bnb ]; then git clone -q --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes /root/bnb 2>>$S; fi
cd /root/bnb && rm -rf build
echo "cmake: $(cmake --version | head -1)" >> $S
# --no-deps on the editable install is MANDATORY: bnb's setup deps pull a
# default CUDA torch from PyPI (torch 2.13+cu130) which clobbers the rocm7.0
# torch -> oracle then dies "no NVIDIA driver". (First bootstrap hit exactly this.)
(cmake -DCOMPUTE_BACKEND=hip -DCMAKE_PREFIX_PATH=/opt/rocm-7.0.2 -S . -B build >/root/bnb_build.log 2>&1 \
  && cmake --build build -j16 >>/root/bnb_build.log 2>&1 \
  && python3.12 -m pip install -q -e . --no-deps >>/root/bnb_build.log 2>&1)
echo "BNBBUILD rc=$? lib=$(ls bitsandbytes/libbitsandbytes_rocm*.so 2>/dev/null | tr '\n' ' ')" >> $S
# belt-and-suspenders: re-assert rocm7.0 torch in case anything pulled cu-torch
python3.12 -c "import torch; assert 'rocm' in torch.__version__, torch.__version__" 2>/dev/null \
  || python3.12 -m pip install -q --force-reinstall --no-deps torch pytorch-triton-rocm --index-url https://download.pytorch.org/whl/rocm7.0 >> $S 2>&1
cd /root
python3.12 - <<'PY' >> $S 2>&1 || { echo "ORACLE-FAIL" >> $S; echo "BOOTSTRAP-FAIL" >> $S; exit 0; }
import torch
from bitsandbytes import functional as F
import bitsandbytes as b
w = torch.randn(64, 64, dtype=torch.bfloat16, device="cuda")
q, st = F.quantize_4bit(w, blocksize=64, quant_type="nf4")
d = F.dequantize_4bit(q, st)
assert torch.isfinite(d).all()
print("ORACLE OK — bnb", b.__version__, "GPU bf16 quant+dequant clean")
PY
# clone the repo once for all tasks
[ -d /root/g ] || git clone -q --depth 1 https://github.com/pjordanandrsn/grouped-nf4-gemm.git /root/g 2>>$S
( cd /root/g && git rev-parse HEAD ) >> $S 2>&1
echo "BOOTSTRAP-OK $(date -u +%FT%TZ)" >> $S
