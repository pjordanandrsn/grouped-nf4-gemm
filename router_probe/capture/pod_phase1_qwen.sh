#!/bin/bash
# Phase-1 cheap-census capture, Qwen3-30B-A3B on ONE cloud GPU (>=24GB).
# Uses the Experts4bit streaming loader (bundled experts4bit_qlora) so the FUSED
# Qwen3MoeExperts stacks are NF4-quantized — stock load_in_4bit's nn.Linear walker
# skips them (bitsandbytes#1849), leaving experts in bf16 (~60GB) which OOMs any
# 24GB card. With Experts4bit the model loads at ~15GB and fits with capture room.
S=/root/RP_STATE
mkdir -p /root/g && tar -xzf /root/router_probe.tar.gz -C /root/g
cd /root/g/router_probe
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_DISABLE_XET=1 HF_HOME=/root/hf-cache
echo "start $(date -u +%FT%TZ)" > $S
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1 >> $S
pip install -q "transformers>=5.0" accelerate bitsandbytes safetensors 2>&1 | tail -1
export PYTHONPATH=/root/g/experts4bit-qlora  # pure-python package; PYTHONPATH avoids any pod-side build
python -c "import torch,transformers,bitsandbytes as b; from experts4bit_qlora.loader import load_moe_4bit_streaming; print('torch',torch.__version__,'tfm',transformers.__version__,'bnb',b.__version__,'e4b OK')" >> $S 2>&1
for fam in qwen3_moe; do
  echo "=== capture+audit $fam (Experts4bit) ===" >> $S
  timeout 5400 python capture/run_olmoe_capture.py --family $fam --device-label "cloud (pod, Experts4bit)" \
    --out /root/rp_$fam --tokens 512 --prompts 12 > /root/rp_$fam.log 2>&1
  echo "$fam rc=$? $(grep -c 'captured' /root/rp_$fam.log)cap" >> $S
done
mkdir -p /root/rp_out
cp receipts/*/EXPLORATORY_phase1_*.json /root/rp_out/ 2>/dev/null
cp /root/rp_*.log /root/rp_out/ 2>/dev/null
echo ALLDONE >> $S
