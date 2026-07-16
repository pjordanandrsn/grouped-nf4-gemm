#!/bin/bash
# Phase-1 cheap-census capture on ONE cloud GPU (A5000-class, ~24GB): OLMoE +
# Qwen3-30B, both bs1-decode contract captures + full ladder audit. Clean HF
# egress on a pod (the household-IP CF block does not apply here). Receipts +
# per-family reducer verdicts to /root/rp_out; STATE for the collector.
S=/root/RP_STATE
mkdir -p /root/g && tar -xzf /root/router_probe.tar.gz -C /root/g
cd /root/g/router_probe
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 HF_HOME=/root/hf-cache
echo "start $(date -u +%FT%TZ)" > $S
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 >> $S
pip install -q "transformers>=4.46" accelerate bitsandbytes 2>&1 | tail -1
python -c "import torch,transformers,bitsandbytes as b; print('torch',torch.__version__,'tfm',transformers.__version__,'bnb',b.__version__)" >> $S 2>&1
for fam in olmoe qwen3_moe; do
  echo "=== capture+audit $fam ===" >> $S
  timeout 3600 python capture/run_olmoe_capture.py --family $fam --device-label "cloud A5000 (pod)" \
    --out /root/rp_$fam --tokens 512 --prompts 12 > /root/rp_$fam.log 2>&1
  echo "$fam rc=$? $(grep -c 'captured' /root/rp_$fam.log)cap" >> $S
done
mkdir -p /root/rp_out
cp router_probe/receipts/*/EXPLORATORY_phase1_*.json /root/rp_out/ 2>/dev/null
cp receipts/*/EXPLORATORY_phase1_*.json /root/rp_out/ 2>/dev/null
cp /root/rp_*.log /root/rp_out/ 2>/dev/null
echo ALLDONE >> $S
