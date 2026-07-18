#!/bin/bash
mkdir /work/conf6.lock 2>/dev/null || exit 0
cd /work/v6
export CC=/usr/bin/cc
rm -f /work/CONF6_STATE
python -m pytest kernel/test_nf4_grouped.py -q > /work/conf6_suite.log 2>&1
echo "SUITE rc=$? $(grep -E 'passed|failed' /work/conf6_suite.log | tail -1)" >> /work/CONF6_STATE
for r in 1 2 3; do
  python bench/phase1/harness.py --regimes prefill_s2048 decode_bs1 \
    --backends dequant_grouped fused_nf4 fused_v5loop --iters 20 --no-energy \
    --out /work/conf6_a2000_d$r.json > /work/conf6_a2000_d$r.log 2>&1
  echo "REP$r rc=$?" >> /work/CONF6_STATE
done
echo ALLDONE >> /work/CONF6_STATE
