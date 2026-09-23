#!/bin/bash
# #386 calibration runner, run inside pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime (CPU only; no GPU needed):
#   docker run --rm -v <dir>:/w pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime bash /w/run.sh
# where <dir> holds this script and kernel.tgz = `git ls-files 'kernel/*.py' | tar -czf kernel.tgz -T -` of the
# commit under test. Pass 1 runs the whole interpreter boundary file against the shipped kernels; pass 2 runs
# the #386 cases against a copy with exactly the five straddling promotions removed (every one must FAIL).
set -u
cd /w && rm -rf shipped unpromoted && mkdir -p shipped unpromoted
tar -C shipped -xzf kernel.tgz && tar -C unpromoted -xzf kernel.tgz
python - <<'PY'
import re
subs = {  # module: [(exact line fragment, replacement)] -- ONLY the operand that straddles in each case
    "host_gather.py":      [("want.to(tl.int64) * row_words", "want * row_words")],
    "mxfp4_pipelined.py":  [("slot.to(tl.int64) * row_words", "slot * row_words")],
    "mxfp4_residency.py":  [("slot.to(tl.int64) * row_words", "slot * row_words")],
    "fp8_kv.py":           [("row = tl.load(tbl_ptr + blk).to(tl.int64)", "row = tl.load(tbl_ptr + blk)"),
                            ("row = tl.load(tbl_ptr + slot * bps + blk).to(tl.int64)", "row = tl.load(tbl_ptr + slot * bps + blk)")],
}
n = 0
for mod, pairs in subs.items():
    p = f"/w/unpromoted/kernel/{mod}"; t = open(p).read()
    for old, new in pairs:
        c = t.count(old); assert c == 1, (mod, old, c); t = t.replace(old, new); n += 1
    open(p, "w").write(t)
print(f"unpromoted copy: {n} promotions removed")
PY
python -c 'import pytest' 2>/dev/null || pip install -q --no-input pytest >/dev/null 2>&1
python -c 'import pytest; print("pytest", pytest.__version__)'
export TRITON_INTERPRET=1
rm -rf /w/receipts && mkdir -p /w/receipts
( cd /w && diff -u shipped/kernel unpromoted/kernel ) > /w/receipts/unpromote.diff
python -c "import torch, triton, numpy, sys; print('torch', torch.__version__); print('triton', triton.__version__); print('numpy', numpy.__version__); print('python', sys.version.split()[0])" > /w/receipts/versions.txt
echo "===== SHIPPED (whole file)"
cd /w/shipped/kernel && python -m pytest test_offset_boundary_interp.py -q -rA -p no:cacheprovider 2>&1 | grep -v DeprecationWarning | tee /w/receipts/shipped.log | grep -E "^(PASSED|FAILED|SKIPPED|ERROR)|passed|failed|rror" | tail -30
echo "===== UNPROMOTED (the #386 cases)"
cd /w/unpromoted/kernel && python -m pytest test_offset_boundary_interp.py -q -rA -p no:cacheprovider -k "host_gather or slot_gather or perm_gather or fp8_append" 2>&1 | tee /w/receipts/unpromoted.log | grep -E "^(PASSED|FAILED|SKIPPED|ERROR)|^E +AssertionError|passed|failed|rror" | cut -c1-260 | tail -30
