#!/usr/bin/env python3
"""Which small-int-tensor construct is LEGAL inside a CUDA graph capture?

The pinned rewrite still fails capture, and "operation failed due to a previous
error during capture" does not say why. So test each construct on its own, in
its own process, against a trivial kernel -- no gnf4 code involved at all.
"""
import subprocess
import sys

import torch

CASES = [
    "pageable_torch_tensor",
    "pin_inside_then_to",
    "prepinned_to",
    "prepinned_copy_into_preallocated",
    "prepinned_copy_blocking",
    "device_arange_only",
    "pin_memory_call_only",
]


def body(case, buf):
    x = torch.ones(64, device="cuda")
    if case == "pageable_torch_tensor":
        i = torch.tensor([1, 2, 3], dtype=torch.int32, device="cuda")
    elif case == "pin_inside_then_to":
        i = torch.tensor([1, 2, 3], dtype=torch.int32).pin_memory().to(
            "cuda", non_blocking=True)
    elif case == "prepinned_to":
        i = buf["host"].to("cuda", non_blocking=True)
    elif case == "prepinned_copy_into_preallocated":
        buf["dev"].copy_(buf["host"], non_blocking=True)
        i = buf["dev"]
    elif case == "prepinned_copy_blocking":
        buf["dev"].copy_(buf["host"], non_blocking=False)
        i = buf["dev"]
    elif case == "device_arange_only":
        i = torch.arange(3, dtype=torch.int32, device="cuda")
    elif case == "pin_memory_call_only":
        torch.tensor([1, 2, 3], dtype=torch.int32).pin_memory()
        i = buf["dev"]
    return x[: i.numel()] + i.float()


def child(case):
    buf = {"host": torch.tensor([1, 2, 3], dtype=torch.int32).pin_memory(),
           "dev": torch.zeros(3, dtype=torch.int32, device="cuda")}
    step = lambda: body(case, buf)  # noqa: E731
    try:
        step()
        torch.cuda.synchronize()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(5):
                step()
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            step()
        torch.cuda.synchronize()
        g.replay()
        torch.cuda.synchronize()
        print(f"RESULT {case} OK")
    except Exception as e:
        print(f"RESULT {case} FAIL {type(e).__name__}: {str(e)[:110]}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(sys.argv[1])
    else:
        print(torch.cuda.get_device_name(0), torch.__version__)
        for c in CASES:
            p = subprocess.run([sys.executable, __file__, c],
                               capture_output=True, text=True, timeout=600)
            line = [x for x in p.stdout.splitlines() if x.startswith("RESULT ")]
            print(line[0][7:] if line else
                  f"{c} CHILD-DIED {(p.stderr or '').strip().splitlines()[-1:]}")
