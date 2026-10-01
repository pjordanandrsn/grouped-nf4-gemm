#!/bin/bash
trap "" HUP
D=/share/ZFS530_DATA/.qpkg/container-station/bin/docker
C=k19-test; R=/share/Container/scripts/k19probe
$D rm -f $C >/dev/null 2>&1
$D run -d --name $C --runtime nvidia-runtime -e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=compute,utility pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel sleep 7200 >/dev/null
echo "== card before:"; $D exec $C nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
$D exec $C bash -c "pip install -q --no-input pytest numpy > /dev/null 2>&1; mkdir -p /root/k19"
$D cp $R/kernel $C:/root/k19/kernel
for f in replay_gemv.py k19_probe.py eids_b16.int16.bin; do $D cp $R/$f $C:/root/k19/$f; done
echo "== INTERP"; $D exec -w /root/k19/kernel $C bash -c "TRITON_INTERPRET=1 python -m pytest test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py -q 2>&1 | tail -4"
echo "== COMPILED A2000"; $D exec -w /root/k19/kernel $C bash -c "TRITON_INTERPRET=0 python -m pytest test_int4_grouped_smallm_interp.py test_int4_smallm_interp.py -q 2>&1 | tail -4"
echo "== PACKAGING"; $D exec -w /root/k19/kernel $C bash -c "python -m pytest test_packaging_covers_kernel.py -q 2>&1 | tail -3"
echo "== PROBE"; $D exec -w /root/k19 $C bash -c "PYTHONPATH=/root/k19/kernel python k19_probe.py --eids eids_b16.int16.bin --steps 8 --iters 20 --out probe2.json 2>&1 | tail -6"
$D cp $C:/root/k19/probe2.json $R/probe2.json 2>/dev/null
$D rm -f $C >/dev/null 2>&1
echo K19T_DONE
