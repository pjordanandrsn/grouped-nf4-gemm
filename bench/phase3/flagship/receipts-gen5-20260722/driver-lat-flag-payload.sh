#!/bin/bash
# lat_flag_payload.sh — ON THE METAL BOX (Latitude g3-h100-small, ml-in-a-box).
# Bare-metal 235B fill WITH ENERGY: link microbench -> suite -> Phase-A
# (none/fused) -> Phase-B real checkpoint (438 GB to disk, stream-quantize,
# generate) — every cell wrapped in 1 Hz GPU-power + RAPL sampling; J/token
# computed per cell. The measured link decides which projected row this
# grades (gen4-x16 band [18,32] GB/s vs gen5 ~45-55) — recorded, not fatal.
# Markers: /root/ab-out + /root/ab-run.log + AB-DONE / AB-FATAL.
set -u -o pipefail
mkdir -p /root/ab-out /root/hf
echo "== FLAG-METAL start $(date -u +%FT%TZ) =="
export DEBIAN_FRONTEND=noninteractive HF_HOME=/root/hf HF_HUB_DISABLE_XET=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# Ubuntu-24 ml-in-a-box is PEP-668 externally-managed -> bare pip refuses
# ("hint: See PEP 668") -> PIP-FAIL -> AB-FATAL. Cost 5 fire/fail/delete loops
# on 2026-07-22 before this line. (Same trap as the AMD box.)
export PIP_BREAK_SYSTEM_PACKAGES=1

nvidia-smi --query-gpu=name,memory.total,driver_version,pcie.link.gen.max,pcie.link.width.max --format=csv,noheader | tee /root/ab-out/gpu.txt
lscpu | grep -E '^(Model name|CPU\(s\))' | tee /root/ab-out/cpu.txt
free -g | tee /root/ab-out/ram.txt; df -h /root | tail -1 | tee /root/ab-out/disk.txt
dmidecode -s system-product-name 2>/dev/null | tee /root/ab-out/baremetal.txt || true

PY=""
for p in python3.12 python3.11 python3.10 python3; do
  if command -v "$p" >/dev/null && "$p" -c 'import torch' 2>/dev/null; then PY="$p"; break; fi
done
[ -n "$PY" ] || { echo "NO-TORCH-PYTHON"; echo "AB-FATAL"; exit 0; }
$PY -c "import torch; assert torch.cuda.is_available(); x=torch.zeros(1024,device='cuda')+1; torch.cuda.synchronize(); print('CUDA-OK', torch.cuda.get_device_name(0))" \
  || { echo "CUDA-GATE-FAIL"; echo "AB-FATAL"; exit 0; }
RAM_GB=$(free -g | awk '/^Mem:/{print $2}')
[ "$RAM_GB" -ge 145 ] || { echo "RAM-GATE-FAIL ${RAM_GB}GB"; echo "AB-FATAL"; exit 0; }
FREE_GB=$(df -BG /root | awk 'NR==2{gsub("G","",$4); print $4}')
[ "$FREE_GB" -ge 520 ] || { echo "DISK-GATE-FAIL ${FREE_GB}GB"; echo "AB-FATAL"; exit 0; }

# ---- ENERGY SAMPLER: 1 Hz, whole run: epoch, GPU W, RAPL uJ (pkg 0 [+1]) ----
RAPL0=/sys/class/powercap/intel-rapl:0/energy_uj
RAPL1=/sys/class/powercap/intel-rapl:1/energy_uj
( while true; do
    W=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -1)
    E0=$(cat $RAPL0 2>/dev/null || echo 0); E1=$(cat $RAPL1 2>/dev/null || echo 0)
    echo "$(date +%s.%N) $W $E0 $E1"
    sleep 1
  done ) > /root/ab-out/power.tsv 2>/dev/null &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
cell(){ echo "$1 $2 $(date +%s.%N)" >> /root/ab-out/cells.tsv; }

# idle baseline
cell idle start; sleep 60; cell idle end

# link microbench (records the box class; NOT fatal on metal)
LINK=$($PY - <<'PYL'
import torch, time
buf = torch.empty(1<<30, dtype=torch.uint8).pin_memory()
dst = torch.empty(1<<30, dtype=torch.uint8, device="cuda")
torch.cuda.synchronize(); t0=time.time()
for _ in range(10): dst.copy_(buf, non_blocking=True)
torch.cuda.synchronize()
print(round(10*(1<<30)/(time.time()-t0)/1e9, 2))
PYL
)
echo "measured pinned H2D: ${LINK} GB/s" | tee /root/ab-out/link.txt
$PY -c "l=float('$LINK'); print('link class:', 'gen4-x16 band (desktop row)' if 18<=l<=32 else ('gen5 band' if l>38 else 'BELOW gen4-x16 band'))" | tee -a /root/ab-out/link.txt

cd /root
git clone --depth 1 https://github.com/pjordanandrsn/grouped-nf4-gemm.git g || { echo "CLONE-FAIL"; echo "AB-FATAL"; exit 0; }
cd /root/g && git rev-parse HEAD | tee /root/ab-out/gnf4_sha.txt
$PY -m pip install -q -U bitsandbytes pytest transformers safetensors huggingface_hub 2>&1 | tail -1 \
  || { echo "PIP-FAIL"; echo "AB-FATAL"; exit 0; }
$PY -c "import triton, bitsandbytes as b; print('triton', triton.__version__, 'bnb', b.__version__)" | tee /root/ab-out/versions.txt \
  || { echo "DEPS-FAIL"; echo "AB-FATAL"; exit 0; }
# bnb native-lib CUDA match: this box's CUDA (H100 + driver 580 -> cuda132) is
# too new for the pip wheel's bundled libs -> "libbitsandbytes_cuda132.so not
# found" killed Phase-B on the 2026-07-22 run. Force BNB_CUDA_VERSION to the
# HIGHEST lib bnb actually bundles (a lower-CUDA lib runs fine on a newer
# driver — CUDA is backward-compatible) and verify a real GPU quantize.
BNB_DIR=$($PY -c "import bitsandbytes,os; print(os.path.dirname(bitsandbytes.__file__))" 2>/dev/null)
if ! $PY -c "import torch;from bitsandbytes import functional as F;F.quantize_4bit(torch.zeros(64,64,device='cuda',dtype=torch.bfloat16),blocksize=64,quant_type='nf4')" 2>/dev/null; then
  AVAIL=$(ls "$BNB_DIR"/libbitsandbytes_cuda*.so 2>/dev/null | grep -oE 'cuda[0-9]+' | grep -oE '[0-9]+' | sort -n | tail -1)
  [ -n "$AVAIL" ] && export BNB_CUDA_VERSION=$AVAIL
  echo "bnb GPU lib mismatch -> BNB_CUDA_VERSION=$BNB_CUDA_VERSION (bundled: $(ls "$BNB_DIR"/libbitsandbytes_cuda*.so 2>/dev/null | xargs -n1 basename 2>/dev/null | tr '\n' ' '))" | tee -a /root/ab-out/versions.txt
fi
$PY -c "import torch;from bitsandbytes import functional as F;q,st=F.quantize_4bit(torch.zeros(64,64,device='cuda',dtype=torch.bfloat16),blocksize=64,quant_type='nf4');F.dequantize_4bit(q,st);print('BNB-GPU-OK (BNB_CUDA_VERSION=${BNB_CUDA_VERSION:-auto})')" 2>&1 | tail -1 | tee -a /root/ab-out/versions.txt

echo "== property suite =="
cell suite start
cd /root/g/kernel && timeout 900 $PY -m pytest test_nf4_grouped.py -q > /root/ab-out/suite.log 2>&1
echo "SUITE rc=$? $(grep -oE '[0-9]+ (passed|failed|error)' /root/ab-out/suite.log | tr '\n' ' ')" | tee /root/ab-out/suite.txt
cell suite end

cd /root/g
echo "== PHASE-A: none then fused $(date -u +%TZ) =="
cell flagA_none start
timeout 1800 $PY bench/phase3/offload_decode_235b.py --moe none  --tokens 64 --out /root/ab-out/flagA_none.json  2>&1 | tail -3 || echo "PHASEA-none FAILED"
cell flagA_none end
cell flagA_fused start
timeout 1800 $PY bench/phase3/offload_decode_235b.py --moe fused --tokens 64 --out /root/ab-out/flagA_fused.json 2>&1 | tail -3 || echo "PHASEA-fused FAILED"
cell flagA_fused end

echo "== PHASE-B real checkpoint $(date -u +%TZ) =="
cell flagB start
timeout 9000 $PY bench/phase3/offload_generate_235b.py --cache /root/hf --out /root/ab-out/flagB_real.json 2>&1 | tail -8 || echo "PHASEB FAILED"
cell flagB end
kill $SAMPLER 2>/dev/null; trap - EXIT

# ---- energy summary: per-cell mean GPU W, GPU J/token, RAPL pkg J ----
$PY - <<'PYS' | tee /root/ab-out/ENERGY.txt
import json
cells = {}
for ln in open("/root/ab-out/cells.tsv"):
    name, edge, ts = ln.split()
    cells.setdefault(name, {})[edge] = float(ts)
rows = []
samp = [ln.split() for ln in open("/root/ab-out/power.tsv") if len(ln.split()) == 4]
samp = [(float(t), float(w) if w.replace('.','',1).isdigit() else 0.0, int(e0), int(e1)) for t, w, e0, e1 in samp]
def window(a, b):
    return [s for s in samp if a <= s[0] <= b]
for name, tt in cells.items():
    if "start" not in tt or "end" not in tt: continue
    w = window(tt["start"], tt["end"])
    if len(w) < 3: continue
    meanW = sum(x[1] for x in w) / len(w)
    dur = tt["end"] - tt["start"]
    d0 = (w[-1][2] - w[0][2]) / 1e6  # J (uJ->J), pkg0; wraps unhandled (report-tier)
    d1 = (w[-1][3] - w[0][3]) / 1e6
    rows.append((name, round(dur,1), round(meanW,1), round(meanW*dur,1), round(d0,1), round(d1,1)))
print("cell          dur_s   gpu_meanW  gpu_J    rapl0_J  rapl1_J")
for r in rows: print("%-13s %-7s %-10s %-8s %-8s %s" % r)
# J/token for the decode cells using harness token counts
try:
    for f, n in (("flagA_none", 64), ("flagA_fused", 64)):
        d = json.load(open(f"/root/ab-out/{f}.json"))
        row = [r for r in rows if r[0] == f]
        if row and d.get("tok_per_s"):
            dec_s = n / d["tok_per_s"]
            print(f"{f}: GPU J/token ~= {round(row[0][2]*dec_s/n, 1)} (mean-W x decode_s / tok; cell window includes setup)")
    b = json.load(open("/root/ab-out/flagB_real.json"))
    row = [r for r in rows if r[0] == "flagB"]
    offs = [r2["toks_per_s_off"] for r2 in b["results"] if r2.get("toks_per_s_off")]
    if row and offs:
        med = sorted(offs)[len(offs)//2]
        print(f"flagB: decode ~{round(med,3)} tok/s; cell meanW {row[0][2]} -> upper-bound GPU J/token ~= {round(row[0][2]/med,1)}")
except Exception as e:
    print("jton calc:", e)
PYS

$PY -c "
import json,glob
for f in sorted(glob.glob('/root/ab-out/flag*.json')):
    d=json.load(open(f)); print(f.split('/')[-1], '->', d.get('tok_per_s') or d.get('waterfall_toks'))
" | tee /root/ab-out/SUMMARY.txt
echo "AB-DONE — evidence complete $(date -u +%FT%TZ)"
