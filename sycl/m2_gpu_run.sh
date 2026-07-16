#!/bin/bash
# M2 perf run on the P630. Same Gen9.5 NEO downgrade as m1_gpu_run.sh, then
# compile + run the bench (naive vs SLM-tiled, WG sweep) on BOTH the small
# correctness vector and a perf-sized vector.
export DEBIAN_FRONTEND=noninteractive
echo "=== force noble-archive NEO 23.43 (Gen9.5-capable) ==="
rm -f /etc/apt/sources.list.d/kobuk-team-ubuntu-intel-graphics-noble.sources
apt-get update -qq 2>/dev/null >/dev/null
apt-get install -y -qq --allow-downgrades intel-opencl-icd=23.43.27642.40-1ubuntu3 2>&1 | tail -1
echo "=== sycl-ls ==="
sycl-ls 2>&1 | grep -iE "opencl:gpu|level_zero:gpu" | head -3
echo "=== compile bench (exec from /tmp per exec=off trap) ==="
icpx -fsycl -O3 /work/nf4_gemv_bench.cpp -o /tmp/bench 2>&1 | tail -3
[ -x /tmp/bench ] || { echo COMPILE-FAILED-bench; exit 1; }
for TV in /work/testvec.bin /work/testvec_perf.bin; do
  [ -f "$TV" ] || continue
  echo "=== M2 bench on $(basename "$TV") (GPU) ==="
  ONEAPI_DEVICE_SELECTOR=opencl:gpu /tmp/bench "$TV" 40 2>&1
done
echo "M2-GPU-SCRIPT-DONE"
