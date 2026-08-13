# DRAFT — HELD FOR REVIEW. NOT FILED.

**This is a draft of a possible upstream issue. It has not been submitted
anywhere and must not be submitted as-is.**

PyTorch's [`AI_POLICY.md`](https://github.com/pytorch/pytorch/blob/main/AI_POLICY.md)
**prohibits contributions created by "fully autonomous agents"** and requires
that AI-generated content be disclosed, contained in code/quote blocks, and
accompanied by human commentary explaining its relevance, with a human
understanding and taking responsibility for the work. It also forbids using AI
to shift verification burden onto reviewers.

So this draft is written to be **filed by Jordan, with his own analysis added**,
or not filed at all. The measurements below are real and reproducible; the
judgement about whether they are worth an upstream maintainer's time is his.

---

## Severity, stated honestly before anything else

This is **not a correctness bug**. Reduced-precision reduction in BF16 GEMMs is
documented, intentional, on by default, and controlled by a public flag. The
[numerical accuracy note](https://docs.pytorch.org/docs/stable/notes/numerical_accuracy.html)
states the justification plainly — *"For performance, certain GPU architectures,
especially more recent ones, allow a few truncations of the intermediate
accumulation results to the reduced precision"* — and describes it as *"often
benign from the perspective of model convergence."*

The bug template is explicit that a numerical-accuracy report must justify *why
one result is wrong* when the difference is already documented. **A plain
"accuracy differs" report here would deserve to be closed.**

The only claim worth making is narrower: **there are shapes where the default
setting pays the documented accuracy cost and does not collect the performance
it is documented to buy.** That is a heuristic-quality observation, and it is
the entire content of this draft.

**Prior art searched:** no existing report of this phenomenon found. The nearest
is [#108621](https://github.com/pytorch/pytorch/issues/108621) (closed) —
inductor's Triton matmul templates not honouring the reduced-precision flags —
which is a different subsystem.

---

## The claim

On an RTX 4090 (sm_89), torch 2.8.0+cu128 / CUDA 12.8, for bf16
`F.linear`: of 7 shapes where the default engages reduced-precision reduction,
**3 gain no measurable speed from it while paying 23–42% more relative error.**

The remaining 4 collect a genuine 11–22% speedup, i.e. the trade works as
documented on most shapes. Three controls never engage it at all.

## Measurements

Error is relative Frobenius against an fp64 GEMM **on the same bf16 values**, so
it isolates the GEMM's reduction and rounding rather than any input difference.
Timings are medians of 200 CUDA-event-timed calls after 30 warm-up calls.

```
gpu   NVIDIA GeForce RTX 4090  sm_89        torch 2.8.0+cu128  cuda 12.8

shape             M     K     N | default err   ms  | redprec_off err  ms   | err↑    ms
qwen3-gate_up   128  2048  1536 | 2.3552e-03  0.035 | 1.6564e-03  0.035     | 1.422  1.000  <-- no speed
qwen3-gate_up   738  2048  1536 | 2.3631e-03  0.062 | 1.6561e-03  0.075     | 1.427  0.834
gemma4-gate_up  128  2816  1408 | 2.3601e-03  0.036 | 1.6621e-03  0.035     | 1.420  1.029  <-- no speed
gemma4-gate_up  738  2816  1408 | 2.6228e-03  0.074 | 1.6626e-03  0.095     | 1.578  0.776
gptoss-down      64  2880  2880 |      (---)  (---) |      (---)  (---)     | 1.471  0.893
gptoss-down     369  2880  2880 |      (---)  (---) |      (---)  (---)     | 1.412  0.802
gptoss-gate_up  369  2880  5760 |      (---)  (---) |      (---)  (---)     | 1.228  0.992  <-- no speed
gptoss-gate_up   64  2880  5760 |                   |            (control)  | 1.000  1.005
olmoe-gate_up   256  2048  2048 |                   |            (control)  | 1.000  0.979
qwen3-down      128   768  2048 |                   |            (control)  | 1.000  1.009
```

**Noise floor:** the three controls, where the flag provably changes nothing
(error ratio exactly 1.000), read timing ratios of 1.005, 0.979 and 1.009 —
about ±2%. The three flagged rows sit at 1.000, 1.029 and 0.992, i.e. **within
noise of no change**, against error increases of 1.228–1.422×.

The claim is therefore "no *measurable* speed," not "slower." One row (1.029)
is marginally the wrong side of the control spread and that is not enough to
call it a regression.

**Cross-architecture, same script, same shapes:**

| device | behaviour |
|---|---|
| H100 80GB (sm_90) | flag **never engages** — error ratio exactly 1.000 on all 10 shapes, timing 0.983–1.019 |
| RTX 4090 (sm_89) | engages on 7 of 10 |
| RTX A2000 (sm_86) | engages on a **different** subset, including one 4090 control |

So the triggering set is per-architecture heuristic, which is expected, and is
why this is a heuristic-quality report rather than a numerics report.

## Reproducer

Self-contained, torch only, no downloaded artifacts:
[`bench/phase1/repro_bf16_linear.py`](../../repro_bf16_linear.py).

```
python repro_bf16_linear.py --iters 200
```

It prints error and time per shape with the flag on and off, plus a summary
line flagging rows that cost accuracy without buying speed.

## What a maintainer might reasonably say back, and the honest answer

- *"Working as intended; set the flag."* — Largely fair. The counter is only
  that a default which costs accuracy for nothing on some shapes is a heuristic
  worth looking at, not that the behaviour is wrong.
- *"n=1, one card, one driver."* — Correct, and a real limit. Two runs on two
  different 4090 instances reproduced the **error** figures bit-identically; the
  **timing** halves of this claim are single-run.
- *"This is cuBLASLt's heuristic, not PyTorch's."* — Probably true. If so the
  useful outcome is a note in the numerical-accuracy docs, or a forward to
  NVIDIA, rather than a PyTorch code change.

## Provenance of these numbers

They fell out of an unrelated kernel benchmark, where the same seven shapes
showed anomalously high error for a bf16 `F.linear` baseline on Ada and not on
Hopper, reproducing bit-identically across two runs on two separate 4090
instances. The standalone reproducer above was written afterwards to isolate
the cause, and it does: with reduced-precision reduction off, every shape
collapses to a uniform 1.656e-3–1.664e-3.

## Before filing — checklist for Jordan

- [ ] Decide whether this clears the bar for a maintainer's time. It is a small
      finding; "not worth filing" is a legitimate outcome.
- [ ] Re-run the reproducer yourself and confirm the numbers on your own
      hardware. Do not file numbers you have not seen.
- [ ] Add `python collect_env.py` output — the template requires it.
- [ ] Add your own commentary in your own words. Per their AI policy, AI output
      must be fenced, disclosed, and accompanied by human analysis; an issue
      that is mostly unreviewed AI text may be closed and repeat offences can
      lead to a ban.
- [ ] Consider whether NVIDIA/cuBLAS is the better venue.
- [ ] Consider re-running the timing halves n≥3 on a second Ada card first,
      since that is the weakest part of the claim.
