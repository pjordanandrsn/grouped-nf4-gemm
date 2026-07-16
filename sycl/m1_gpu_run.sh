#!/bin/bash
export DEBIAN_FRONTEND=noninteractive
echo "=== force noble-archive NEO 23.43 (Gen9.5-capable) ==="
rm -f /etc/apt/sources.list.d/kobuk-team-ubuntu-intel-graphics-noble.sources
apt-get update -qq 2>/dev/null >/dev/null
apt-get install -y -qq --allow-downgrades intel-opencl-icd=23.43.27642.40-1ubuntu3 2>&1 | tail -2
echo "=== clinfo device ==="
clinfo 2>/dev/null | grep -iE "Device Name|Driver Version" | head -4
echo "=== sycl-ls ==="
sycl-ls 2>&1 | head -5
echo "=== compile both (exec from /tmp per exec=off trap) ==="
icpx -fsycl -O3 /work/hello_sycl.cpp -o /tmp/hello 2>&1 | tail -3
[ -x /tmp/hello ] || { echo COMPILE-FAILED-hello; exit 1; }
icpx -fsycl -O3 /work/nf4_gemv_sycl.cpp -o /tmp/nf4 2>&1 | tail -3
[ -x /tmp/nf4 ] || { echo COMPILE-FAILED-nf4; exit 1; }
echo "=== M0 on GPU backend ==="
ONEAPI_DEVICE_SELECTOR=opencl:gpu /tmp/hello 2>&1 | tail -6
echo "=== M1 (NF4 gemv vs canonical reference) on GPU ==="
ONEAPI_DEVICE_SELECTOR=opencl:gpu /tmp/nf4 /work/testvec.bin 2>&1
echo "M1-GPU-SCRIPT-DONE"
