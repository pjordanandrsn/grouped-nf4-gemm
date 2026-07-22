#!/bin/bash
# TASK1 — MXFP4 native-byte kernel correctness on CDNA3 (MI300X). The NF4-suite
# analog for the MXFP4 lane: test_mxfp4_grouped (kernel) + _oracle (dequant-ref
# parity). Synthetic fixtures — no 60GB model download.
export PIP_BREAK_SYSTEM_PACKAGES=1 PYTHONDONTWRITEBYTECODE=1
export ROCM_PATH=/opt/rocm-7.0.2 HIP_PATH=/opt/rocm-7.0.2 HIP_PLATFORM=amd
export PATH=/opt/rocm-7.0.2/bin:/opt/rocm/bin:$PATH
S=/root/TASK1
echo "=== TASK1 mxfp4 correctness $(date -u +%FT%TZ) ===" > $S
cd /root/g/kernel
echo "--- test_mxfp4_grouped ---" >> $S
timeout 900 python3.12 -m pytest test_mxfp4_grouped.py -q > /root/task1_grouped.log 2>&1
echo "GROUPED rc=$? $(grep -oE '[0-9]+ (passed|failed|error|skipped)' /root/task1_grouped.log | tr '\n' ' ')" >> $S
echo "--- test_mxfp4_oracle ---" >> $S
timeout 900 python3.12 -m pytest test_mxfp4_oracle.py -q > /root/task1_oracle.log 2>&1
echo "ORACLE rc=$? $(grep -oE '[0-9]+ (passed|failed|error|skipped)' /root/task1_oracle.log | tr '\n' ' ')" >> $S
echo "TASK1-DONE $(date -u +%FT%TZ)" >> $S
