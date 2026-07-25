# PREREG — making the streamed KV tier faster: what moves the bound, what moves the constant

**Tier: CONFIRMATORY. Status: STAMPED before any of it was built.** The
predictions below are derived from the model validated in finding #15
(`t = bytes / link`) and from measurements already in
`receipts-stream-20260725/`. **No implementation existed when this was written**,
which is the point — writing predictions after building is how the last five
post-hoc explanations happened.

Code under test: e4b `claude/e4b-gemma-inflight-d41f93` @ `02f29ea`,
gnf4 `kernel/nf4-kv-cache` @ `ebe6edc`. Both local, unpushed.

## The frame

Attention reads the **whole** cache every step, so a streamed tier is bounded by

```
ceiling = link / KV(ctx)
```

and only three things move that bound: **move fewer bytes**, **keep some
resident**, or **get a faster link**. Everything else — prefetch, kernel fusion,
overhead removal — improves the *constant* and lets you approach the bound. Both
kinds are worth having; conflating them is how a 1.3× constant-factor win gets
sold as lifting a ceiling. Each hypothesis below is labelled which it is.

## Methodology fix, carried over from #15

#15's harness scored on `t_host − t_gpu`: a difference of two separately-timed
loops carrying **~±15 ms** of noise. That was fine against a 286 ms overhead and
useless against a 36 ms one — S3 confirmed at 3.961 on one run and would have
falsified at 4.970 on the next **from identical code**.

So nothing here is scored on a difference of two point measurements. Every
transfer claim is scored on a **fit across ≥ 4 points**, `t = c + bytes/B`,
reporting both the effective bandwidth `B` and the per-step constant `c` — the
same shape `pcie_probe.py` uses. A fit over four points tolerates the noise floor
that a subtraction of two does not.

## Baselines (measured, not assumed)

94L × 4kv × 128d (Qwen3-235B geometry), A2000, `receipts-stream-20260725`:

| quantity | value |
|---|---:|
| packed KV @32K | 1.774 GB (576 B/layer/token) |
| link, measured asymptote | 6.20 GB/s |
| resident load, 94 layers @32K (this is the **dequant**) | 967.5 / 968.1 ms |
| streamed load @32K | 1256.1 / 1254.5 ms |
| ⇒ transfer @32K | 288.6 / 286.4 ms (predicted 286.2) |
| ⇒ per layer @32K | dequant **10.3 ms**, transfer **3.05 ms** |

---

## A1 — split residency (**moves the bound**) — RUNS THIS CYCLE

Today residence is binary, so free VRAM sits unused. Keeping the oldest `f` of
the cache resident and streaming the rest should give `ceiling = link/((1−f)·KV)`.
Oldest, not newest: the head is positionally stable, so nothing reshuffles as the
cache grows.

- **A1a.** Fit overhead against **streamed** bytes across f ∈ {0, 0.25, 0.5,
  0.75} at ctx 32768. Fitted `B` ∈ **[5.0, 7.5] GB/s** — i.e. the same law, with
  fewer bytes. *Falsified outside.*
- **A1b.** Fitted per-step constant `c` < **25 ms**. *Falsified at ≥ 25 ms* —
  that would mean splitting introduces a fixed cost that eats the saving at
  small f.
- **A1c.** Correctness: the split cache returns **byte-identical** K/V to a
  fully-resident cache at every f. `torch.equal`, not a tolerance. *Falsified by
  any mismatch* — and a failure here voids A1a/A1b rather than merely failing.
- **A1d.** GPU peak scales with f: peak(f=0.5) / peak(f=0) ∈ **[0.4, 0.65]**.
  *Falsified outside.*

**Pre-committed decision.** If A1a and A1c hold, split residency becomes the
default shape of the streamed tier and the binary switch is retired — it is
strictly dominated, since f=0 and f=1 are its endpoints.

## B1 — prefetch (**moves the constant**) — RUNS THIS CYCLE

Overlap layer L+1's copy with layer L's compute on a side stream, converting
`compute + transfer` into `max(compute, transfer)`. The machinery exists in
`offload.py`. Per layer at 32K the dequant is **10.3 ms** against a **3.05 ms**
transfer, so the transfer should hide **completely**.

- **B1a.** Prefetched streamed load / resident load at 32K ∈ **[1.00, 1.15]** —
  the transfer essentially free. *Falsified above 1.25.*
- **B1b.** Prefetched / non-prefetched streamed load at 32K ∈ **[1.20, 1.35]**
  (predicted 1.295 = 1254/968). *Falsified outside.*
- **B1c.** Double-buffering costs one extra layer resident: GPU peak rises by
  **< 100 MB** against non-prefetched. *Falsified at ≥ 100 MB.*

**Stated in advance, not scored:** B1's win is bounded by the compute it hides
behind. If D1 later removes the dequant, B1's benefit shrinks toward zero — the
two are **substitutes, not complements**, and a future D1 result must not be
combined with B1's as though they add.

**Confound.** This harness's "compute" is dequantization, not real attention. So
B1 tests hiding transfer behind dequant, which is real work on the real path, but
it is not the same as hiding it behind an attention kernel. Stated now so a
later real-model number is not silently scored against B1.

## A2 — eviction, repriced (**moves the bound**) — REFRAMED, NO NEW RUN

#13 closed token-axis sparsity: quantization dominates eviction ~9× at matched
bytes. That ratio is a quality-per-byte figure and is **unchanged** by
streaming — so #13's verdict is not reopened.

What changes is what eviction *competes with*. Resident bytes cost VRAM once;
streamed bytes cost PCIe **every step**. Once you are at NF4 and cannot compress
further without a fidelity cliff, eviction is the only remaining byte-reduction
axis — and in the streamed regime its competitor is not quantization, it is
**"cannot meet the latency target at all"**. Against that alternative the
relevant number is not the 9× ratio but the absolute quality cost at the budget
streaming imposes, which #11 and #13 already measured (+3.33 ppl at 260 tokens,
+1.09 with H2O selection).

No new experiment is registered because none is needed: the curve exists, and
what was wrong was the *pricing*, not the measurement. Recorded so this is not
mistaken for #13 being quietly reopened.

## Registered but DEFERRED, with the gate stated

Predictions committed now so they cannot be written after the fact; neither runs
this cycle.

- **C1 — fp16 absmax.** 0.5 + 4/64 = 0.5625 B/elem today; fp16 gives 0.53125.
  **C1a:** streamed bytes fall by exactly **5.56%**. **C1b:** wikitext ppl
  changes by **< 0.01** (the absmax is a scale, and fp16 carries ~3 decimal
  digits of it). *Gate:* the packed layout is read by every kernel in
  `kernel/`, so this is a format change with a blast radius far larger than its
  5.6%, and it does not go first.
- **D1 — fused attend on the streamed packed bytes.** `attend_nf4_kv_gqa`
  (#12) reads nibbles in the mainloop and never materializes a bf16 layer.
  **D1a:** GPU peak per layer drops by the bf16 materialization (**67 MB** at
  32K). **D1b:** at GQA 16:1 the streamed path's per-layer cost drops toward the
  kernel's 0.82× fp16 rather than dequant + SDPA. *Gate:* needs a real attention
  integration, which is per-architecture, and #12 measured the same kernel
  **4.59× slower** at GQA 4:1 — so this is a regime-dependent build, not a
  drop-in.
- **B2 — per-call overhead.** Not registered with an interval, deliberately:
  the ~570 µs/append measured in #15 is dominated by a Python layer loop that
  belongs to `transformers`, not to this code, so there is no lever here worth
  predicting.

## Scoring

Results land in `receipts-faster-20260725/`. Every prediction is marked
**confirmed / falsified / void** with the measured value beside the predicted
interval. Falsified entries stay in this document. A1c is a **gate**: if the
split cache is not byte-exact, A1a/A1b are void rather than scored, because a
wrong cache can be arbitrarily fast.
