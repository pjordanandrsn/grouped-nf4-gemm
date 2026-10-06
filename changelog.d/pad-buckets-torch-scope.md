### Docs: the bucketed-padding default is measured under torch 2.12 and torch 2.8

The 0.42.0 `auto` default's (#492) first speed evidence was taken under torch 2.12.1 / triton 3.7.1 only.
experts4bit-qlora's TC1 amendment 51 then read e4b slower than expected under torch 2.8 with the default on (environment
ratio 0.739, against a registered [0.84, 0.98]). For a while `docs/STATUS.md` said the default's torch-2.8 effect was
unmeasured (#494).

Amendment 52 (`e4b.train.pad-buckets.torch28.qwen3.5090.2026-10-06`) has since measured it on packed rows under torch
2.8.0 / triton 3.4.0, and all four of its predictions HELD:
- 0.983 [0.974, 0.993] (matched) and 0.939 (shipped) of the single block's step;
- the matched peak 4.24 GB lower;
- held-out within 0.0002.

The default stands in both measured environments, and amendment 51's slower torch-2.8 e4b is not the buckets' cost.
Reported, not scored: under torch 2.8 the bucketed arms left the GPU idle more of the step (median utilisation 87 %
against 97 %), so host-side time in the bucketed delta is the open lead.

`NF4_QLORA_PAD_BUCKETS=0` restores the single block exactly. Docs only.
