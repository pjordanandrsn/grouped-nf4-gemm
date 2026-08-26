# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Parse a replay kernel-census table into k12_verdict's census dict.

This is instrument, not glue: its output IS the attribution gate's
input, so a bug here changes the verdict rather than the logging.
Committed and tested beside the calculator instead of living in a
scratchpad harness ([[commit-the-instrument-not-just-receipts]]).

The table is `torch.profiler`'s `key_averages().table(...)`, written
by `step_decomp --replay-profile-out`. Two properties of that format
decide the whole parser, and both were confirmed against a real SV2
census rather than assumed:

1. **Names are truncated with an ellipsis**, but the truncation falls
   AFTER the kernel-family name, so `unrolled_elementwise_kernel`,
   `elementwise_kernel`, `indexSelectS`, `reduce_kernel` all survive.
   That is what makes `k12_verdict.TRACKED`'s substring matchers work
   at all -- including the `::elementwise_kernel` matcher, which the
   real rows confirm is disjoint from the decorated families:
   `at::native::elementwise_kernel` contains it,
   `at::native::unrolled_elementwise_kernel` does not.

2. **Truncation can make two DISTINCT rows collide.** A real census
   holds two different `elementwise_kernel<128, 4, ...>` rows, 384
   calls each, whose displayed names are byte-identical. Assigning
   into a dict would silently keep one and drop 384 calls -- half a
   family -- which is exactly the kind of quiet halving that makes an
   attribution gate pass or fail on a number nobody chose. Counts are
   ACCUMULATED on collision.

Counts are normalised to calls-per-step using the replay count in the
header, because that is the unit `k12_verdict` renders ("calls/step")
and the unit PREREG-k12's census numbers are quoted in.
"""

import re

#: `profiled replay steps: 8 (active window: 8/8)`
_HDR = re.compile(r"profiled replay steps:\s*(\d+)")
#: a data row ends with the "# of Calls" column
_CALLS = re.compile(r"\s(\d+)\s*$")
#: the Name column is everything before the first run of 2+ spaces
#: that precedes the numeric columns
_NAME = re.compile(r"^\s*(\S.*?)\s{2,}\d")
#: Self CUDA is the SIXTH numeric field after the name, not the eighth
#: (CUDA total). That distinction is not pedantry: summing CUDA total
#: read 23.6 ms/step against a 12.6 ms truth in the F1 budget parser,
#: because dispatcher rows inherit a CUDA-total from their children
#: with zero self. Credit to bench/hybrid-g9/f1/step_budget.py, which
#: paid for that lesson.
_ROW = re.compile(
    r"^\s*(.+?)\s{2,}([\d.]+)%\s+([\d.]+\w*s)\s+([\d.]+)%\s+"
    r"([\d.]+\w*s)\s+([\d.]+\w*s)\s+([\d.]+\w*s)\s+"
    r"([\d.]+)%\s+([\d.]+\w*s)\s+([\d.]+\w*s)\s+(\d+)\s*$")
_TOTAL = re.compile(r"Self CUDA time total:\s*([\d.]+)\s*(\w+)")
#: Host-runtime bookkeeping, not device kernels. They carry call
#: counts with zero self-CUDA, so they inflate the row set without
#: contributing work. None of them match TRACKED, but a parser that
#: returns them invites a future caller to sum them.
_NON_KERNEL = ("cudaLaunch", "cuLaunchKernel", "Activity Buffer",
               "cudaMemcpy", "cudaStreamSynchronize", "cudaFree",
               "cudaDeviceSynchronize", "cudaMalloc",
               "Runtime Trigger", "aten::", "e4b::", "ProfilerStep")
#: Below this the table was row-limited and a family's small rows may
#: have fallen off the bottom. See `parse`.
MIN_COVERAGE = 0.90
#: ...and above THIS the sum exceeds what the device ran, which a
#: correct kernel-view sum cannot do. Two causes, both real: summing
#: CUDA total instead of Self CUDA (dispatcher rows inherit a total
#: from their children with zero self), or summing the op view and the
#: kernel view together, which double-counts every kernel. Guarding
#: only the LOW side let a wrong-column mutation pass this module's
#: own tests -- coverage of 9900% read as "plenty".
MAX_COVERAGE = 1.05
_SCALE = {"us": 1.0, "ms": 1e3, "s": 1e6, "ns": 1e-3}


def _us(v: str, unit: str) -> float:
    if unit not in _SCALE:
        raise ValueError(f"unknown time unit {unit!r}")
    return float(v) * _SCALE[unit]


def _row_us(field: str) -> float:
    m = re.fullmatch(r"([\d.]+)(\w+)", field)
    if not m:
        raise ValueError(f"unparseable time field {field!r}")
    return _us(m.group(1), m.group(2))


def parse(text: str, per_step: bool = True,
          check_coverage: bool = True) -> dict:
    """{row_name: calls} from one census table.

    `per_step` divides by the header's replay count. Pass False only
    if you mean raw totals; the verdict wants per-step.
    """
    m = _HDR.search(text)
    steps = int(m.group(1)) if m else 1
    if steps <= 0:
        raise ValueError(f"replay step count {steps} is not positive")
    out: dict[str, float] = {}
    seen_us = 0.0
    for line in text.splitlines():
        if line.startswith("-") or "# of Calls" in line:
            continue
        nm = _NAME.match(line)
        cl = _CALLS.search(line.rstrip())
        if not (nm and cl):
            continue
        name = nm.group(1).strip()
        if not name or name.startswith("Name"):
            continue
        if any(k in name for k in _NON_KERNEL):
            continue
        row = _ROW.match(line)
        if row:
            seen_us += _row_us(row.group(7))     # Self CUDA
        # ACCUMULATE: two distinct kernels can print the same
        # truncated name, and dropping one loses half a family
        out[name] = out.get(name, 0) + int(cl.group(1))
    if check_coverage:
        _check_coverage(text, seen_us)
    if per_step:
        out = {k: v / steps for k, v in out.items()}
    return out


def _check_coverage(text: str, seen_us: float) -> None:
    """Refuse a row-limited table.

    `--replay-profile-out` writes `table(row_limit=120)` sorted by CUDA
    time, so a long census silently drops its SMALLEST rows -- exactly
    where the tracked raw-ATen families live. K12's attribution gate
    asks "did these rows fall in count?", and a row that fell off the
    bottom of the table is indistinguishable from a row that fused
    away. That would let the gate confirm the mechanism using rows the
    profiler simply stopped printing.

    The footer's `Self CUDA time total` is ground truth for what the
    device ran; if the printed rows do not cover it, the table is
    partial and this refuses instead of returning a census that looks
    complete.
    """
    m = _TOTAL.search(text)
    if not m:
        return                     # no footer: nothing to check against
    total = _us(m.group(1), m.group(2))
    if total <= 0:
        return
    cov = seen_us / total
    if cov < MIN_COVERAGE:
        raise ValueError(
            f"census covers {cov:.1%} of the footer's Self CUDA total "
            f"({seen_us:.0f}us of {total:.0f}us); the table was "
            "row-limited, so a tracked row missing from it cannot be "
            "told apart from one that fused away")
    if cov > MAX_COVERAGE:
        raise ValueError(
            f"census sums to {cov:.1%} of the footer's Self CUDA total "
            f"({seen_us:.0f}us of {total:.0f}us) -- more than the "
            "device ran. Either the wrong column was summed (CUDA "
            "total, not Self CUDA) or two profiler views were added "
            "together")


def census(before_text: str, after_text: str) -> dict:
    """The `census` block k12_verdict consumes."""
    return {"before": parse(before_text), "after": parse(after_text)}
