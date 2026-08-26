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


def parse(text: str, per_step: bool = True) -> dict:
    """{row_name: calls} from one census table.

    `per_step` divides by the header's replay count. Pass False only
    if you mean raw totals; the verdict wants per-step.
    """
    m = _HDR.search(text)
    steps = int(m.group(1)) if m else 1
    if steps <= 0:
        raise ValueError(f"replay step count {steps} is not positive")
    out: dict[str, float] = {}
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
        # ACCUMULATE: two distinct kernels can print the same
        # truncated name, and dropping one loses half a family
        out[name] = out.get(name, 0) + int(cl.group(1))
    if per_step:
        out = {k: v / steps for k, v in out.items()}
    return out


def census(before_text: str, after_text: str) -> dict:
    """The `census` block k12_verdict consumes."""
    return {"before": parse(before_text), "after": parse(after_text)}
