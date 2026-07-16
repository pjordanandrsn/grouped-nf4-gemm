Following up on the grouped-GEMM note above, since it's now concrete: I've built and published the compute half — [**grouped-nf4-gemm**](https://github.com/pjordanandrsn/grouped-nf4-gemm) (MIT), a fused single-launch grouped GEMM that decodes NF4 in-register inside the mainloop (fp32 accumulation; no dequantized tensor is ever materialized). It consumes the plain-`nn.Parameter` + absmax storage from your note above, as implemented in #1965, so it drops onto the `Experts4bit` surface unchanged.

Headline numbers, each from a preregistered protocol with blind confirmatories on fresh instances:

- **decode bs=1:** 1.25–2.97× blind-confirmed medians vs the dequantize→GEMM path across census MoE shapes (OLMoE, Qwen3-30B, Gemma-4-26B, gpt-oss-120B geometries), never slower;
- **energy:** J/token strictly below the dequantize path in 32/32 blind-confirmed cells — the claim that survived every protocol untouched;
- **fidelity:** numerical error ≤0.755× the dequantize path's in all measured cells — the fp32-accumulation design goal, held under blind;
- **at scale:** as the sole MoE path it serves the real Qwen3-235B-A22B checkpoint (bnb NF4, experts streamed from pinned host RAM) at 4.3 tok/s on ~15 GB VRAM, replicated on five hosts.

The full confirmatory record is in-repo, including the protocols that failed. Sharing it now so it's on the table for your MoE-inference-kernel work post-v0.50.0 — take it, adapt it, or treat it as prior art; happy to align with whichever direction you choose. No change to #1965's scope, and still no rush on my end.
