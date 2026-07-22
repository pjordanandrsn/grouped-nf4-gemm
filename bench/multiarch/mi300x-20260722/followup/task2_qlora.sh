#!/bin/bash
# TASK2 — 4-bit MoE QLoRA TRAIN-path correctness on CDNA3 (MI300X). The full
# test_mxfp4_qlora suite: grads-match-dense-autograd, bytes-bit-identical-after-
# -training-steps (frozen base), loss-descends, lora-zero-at-init, fused==loop.
# Synthetic — proves the train step works on AMD, no model download.
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2 HIP_PLATFORM=amd
export PATH=/opt/rocm-7.0.2/bin:/opt/rocm/bin:$PATH
S=/root/TASK2
echo "=== TASK2 qlora train-path $(date -u +%FT%TZ) ===" > $S
cd /root/g/kernel
timeout 1200 python3.12 -m pytest test_mxfp4_qlora.py -v -rs > /root/task2_qlora.log 2>&1
echo "QLORA rc=$? $(grep -oE '[0-9]+ (passed|failed|error|skipped)' /root/task2_qlora.log | tr '\n' ' ')" >> $S
echo "--- per-test:" >> $S
grep -aE "PASSED|FAILED|SKIPPED" /root/task2_qlora.log | sed 's#.*::##' >> $S
echo "TASK2-DONE $(date -u +%FT%TZ)" >> $S
