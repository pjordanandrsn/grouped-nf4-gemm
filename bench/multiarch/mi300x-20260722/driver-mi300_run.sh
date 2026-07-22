#!/bin/bash
# MI300X port session (multiarch P2a): fingerprint -> hw_contract (bnb-free)
# -> bnb source build (HIP) -> suite -> census. Continue-on-fail; every rc recorded.
# (2026-07-21 rev2, after the 2-min rc=1 run): the runpod/pytorch:*-rocm* image
# has conda BASE (/opt/conda, py3.12) WITHOUT torch — torch lives in a conda
# ENV (or nowhere), and image cmake is 3.18 (bnb needs >=3.22). So: probe every
# python for torch (conda envs first), pip the rocm6.1 wheel if none, use
# "$PY" -m for pip/pytest everywhere, and pip a modern cmake before the bnb build.
S=/root/MI300_STATE
mkdir -p /root/g && tar -xzf /dev/shm/gnf4-mi300.tar.gz -C /root/g
cd /root/g
export PYTHONDONTWRITEBYTECODE=1
# (rev3, 2026-07-22) Ubuntu 24.04 system python is PEP-668 externally-managed:
# every bare pip install silently REFUSES (attempt 2: cmake/pytest/bnb all
# missing -> rc=127/1 with torch+contract green). The env var makes every pip
# call in this script (and the bnb build's `pip install -e .`) proceed.
export PIP_BREAK_SYSTEM_PACKAGES=1
# (rev4) AMD devcloud image keeps ROCm at /opt/rocm/core-7.14 OFF the default
# PATH -> cmake enable_language(HIP) fails (attempt 3, CMakeLists:273).
export ROCM_PATH=/opt/rocm HIP_PATH=/opt/rocm HIP_PLATFORM=amd
export PATH=/opt/rocm/bin:/opt/rocm/core-7.14/bin:/opt/rocm/llvm/bin:$PATH
echo "start $(date -u +%FT%TZ)" > $S
rocm-smi --showproductname 2>/dev/null | head -4 >> $S
echo "hipcc: $(command -v hipcc || ls /opt/rocm/bin/hipcc 2>/dev/null || echo MISSING) | rocm: $(ls -d /opt/rocm-* /opt/rocm 2>/dev/null | tr '\n' ' ')" >> $S

# --- python-with-torch probe: conda envs, conda base, then system pythons ---
PY=""
for p in /opt/conda/envs/*/bin/python /opt/conda/bin/python $(command -v python3.12 python3.11 python3.10 python3 2>/dev/null); do
  [ -x "$p" ] && "$p" -c 'import torch' 2>/dev/null && { PY="$p"; break; }
done
if [ -z "$PY" ]; then
  PY=$([ -x /opt/conda/bin/python ] && echo /opt/conda/bin/python || command -v python3)
  echo "no torch in any python — pip-installing rocm6.1 wheel into $PY" >> $S
  "$PY" -m pip install -q torch --index-url https://download.pytorch.org/whl/rocm6.1 >> $S 2>&1
fi
"$PY" -c "import torch; print('PY $PY | torch',torch.__version__,'| hip',torch.version.hip,'| dev',torch.cuda.get_device_name(0),'| cu',torch.cuda.get_device_properties(0).multi_processor_count)" >> $S 2>&1 \
  || { echo "TORCH-FATAL: no working torch even after pip" >> $S; echo ALLDONE >> $S; exit 0; }
"$PY" -c "import triton; print('triton', triton.__version__)" >> $S 2>&1
export PATH="$(dirname "$PY"):$PATH"; hash -r
"$PY" backends/detect.py >> $S 2>&1
echo "=== HW CONTRACT (bnb-free) ===" >> $S
"$PY" bench/hw_contract.py --device cuda > /root/mi300_contract.log 2>&1
echo "CONTRACT rc=$? $(tail -1 /root/mi300_contract.log)" >> $S
echo "=== bnb source build (HIP) ===" >> $S
"$PY" -m pip install -q --upgrade "cmake<4" ninja numpy >> $S 2>&1
hash -r
echo "cmake now: $(command -v cmake) $(cmake --version 2>/dev/null | head -1)" >> $S
git clone -q --depth 1 https://github.com/bitsandbytes-foundation/bitsandbytes /root/bnb 2>>$S
cd /root/bnb && (rm -rf build; cmake -DCOMPUTE_BACKEND=hip -S . -B build >/root/bnb_build.log 2>&1 && cmake --build build -j16 >>/root/bnb_build.log 2>&1 && "$PY" -m pip install -q -e . >>/root/bnb_build.log 2>&1)
echo "BNBBUILD rc=$? $(tail -1 /root/bnb_build.log | head -c 120)" >> $S
# import check MUST run from /root/g — from /root/bnb the SOURCE TREE imports
# as a cwd package (attempt 3: phantom "bnb 0.50.0.dev0" with no compiled lib,
# wheel fallback skipped, pytest then had no bitsandbytes at all)
cd /root/g
"$PY" -c "import bitsandbytes as b; print('bnb', b.__version__)" >> $S 2>&1 \
  || { echo "bnb source build failed — trying pip wheel" >> $S; "$PY" -m pip install -q bitsandbytes >> $S 2>&1; "$PY" -c "import bitsandbytes as b; print('bnb(pip)', b.__version__)" >> $S 2>&1; }
echo "=== property suite (needs bnb) ===" >> $S
"$PY" -m pip install -q pytest 2>&1 | tail -1
"$PY" -m pytest kernel/test_nf4_grouped.py -q > /root/mi300_suite.log 2>&1
echo "SUITE rc=$? $(grep -oE '[0-9]+ (passed|failed|error)' /root/mi300_suite.log | tr '\n' ' ')" >> $S
echo "=== census (needs bnb) ===" >> $S
"$PY" -m pip install -q pynvml 2>&1 | tail -1
"$PY" bench/phase1/harness.py --regimes prefill_s2048 decode_bs1 \
  --backends dequant_grouped fused_nf4 fused_v5loop --iters 20 --no-energy \
  --out /root/mi300_census.json > /root/mi300_census.log 2>&1
echo "CENSUS rc=$?" >> $S
echo ALLDONE >> $S
