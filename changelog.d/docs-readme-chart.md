### Docs: a first-screen chart generated from the claims register

The README opens with one chart: decode against Unsloth's MoE kernel with weights stored in 4-bit (RTX 4090 and H100),
Unsloth's win at H100 prefill with weights resident in bf16 on the same chart, and OLMoE training against this project's
per-expert loop in its own panel. `scripts/build_readme_chart.py` reads every figure from `docs/claims.json`, and a CI
step fails when the committed chart is stale. A collapsed block under the results states what the receipts recorded
(GPU, torch, driver) and what they did not (clock locking, ECC, Triton version). No code changes.
