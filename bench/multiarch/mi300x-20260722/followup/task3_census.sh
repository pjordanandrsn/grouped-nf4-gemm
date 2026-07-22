#!/bin/bash
# TASK3 verify — LDS fit-down: re-run the census with the patched kernel; the
# 6 fused_v5loop prefill_s2048 cells that skipped (Required 98304 > 65536) must
# now RUN and stay bit-accurate.
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2 HIP_PLATFORM=amd
export PATH=/opt/rocm-7.0.2/bin:/opt/rocm/bin:$PATH
S=/root/TASK3
cp /root/nf4_grouped_patched.py /root/g/kernel/nf4_grouped.py
echo "=== TASK3 census w/ LDS fit-down $(date -u +%FT%TZ) ===" > $S
cd /root/g
timeout 1800 python3.12 bench/phase1/harness.py --regimes prefill_s2048 decode_bs1 \
  --backends dequant_grouped fused_nf4 fused_v5loop --iters 20 --no-energy \
  --out /root/mi300_census_fixed.json > /root/task3_census.log 2>&1
echo "CENSUS rc=$?" >> $S
echo "--- v5loop prefill cells (were 6 skips):" >> $S
grep -aE "prefill_s2048 fused_v5loop" /root/task3_census.log >> $S
echo "--- remaining skips (want 0):" >> $S
grep -ac "skipped: out of resource" /root/task3_census.log >> $S
echo "TASK3-DONE $(date -u +%FT%TZ)" >> $S
