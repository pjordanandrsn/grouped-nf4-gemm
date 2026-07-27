# PREREG — routed-union vs verify-width (the speculative-decoding byte economics)

**Tier: CONFIRMATORY on U1–U2 (Qwen3-235B); EXPLORATORY-REGISTERED on U3 (K3).
Status: DRAFT until OTS-stamped; stamp must precede any qualifying capture.**

## Question

A draft-then-verify decode step streams the **union** of w tokens' routed
experts in one forward instead of w separate top-k sets. Whether token-level
speculative decoding pays on a streaming-MoE stack is, on the byte side,
exactly the ratio r(w) = union(w) / (w·k).

## Exploratory basis (computed 2026-07-27 from EXISTING data — $0, pre-dating
this registration and labelled accordingly)

Qwen3-30B-A3B decode capture (294,912 records = 48 layers × 6,144 decode
tokens, E=128, k=8, `router_probe/union-exploratory-qwen30b.json`, script
`router_probe/union_vs_k.py` — windows never span non-consecutive
`record_token` indices):

  r(2)=0.775 · r(4)=0.598 · r(8)=0.446 · r(16)=0.309
  (uniform-null ratios: 0.969 · 0.910 · 0.807 · 0.644 — realized ≈ 0.66× null)

## Registered predictions

- **U1 (235B, same family/E/k):** a fresh decode capture of Qwen3-235B-A22B
  (natural prompts, decode-only, ctx 512, ≥4 prompts × ≥256 tokens, same
  capture harness + boundary masking) yields
  **r(4) ∈ [0.50, 0.70]** and **r(8) ∈ [0.37, 0.53]**.
  *Falsified below/above those bands; r(4) > 0.85 additionally kills the
  spec-dec lane's byte premise outright (that is the decision this number
  gates).*
- **U2 (concentration, not chance):** realized union(w) < uniform-null
  union(w) at every w ≥ 2 with margin ≥ 10% of null. *Falsified if routing is
  union-indistinguishable from uniform — the access-pattern law would need
  revisiting, not just spec-dec.*
- **U3 (K3, out-of-sample scale-up; WIDE band, exploratory-registered):**
  K3 (E=896, k=16), same computation when a K3-capable box exists:
  **r(4) ∈ [0.50, 0.85]**. Registered now so the scale-up direction is a
  prediction rather than a postdiction; adjudicated only when run.

## Decision rule (binding)

- U1 holds → the spec-dec build (roadmap S1) proceeds, with its end-to-end
  prereg deriving its tok/s band from the MEASURED r(w) + the additive law —
  never from this document's exploratory numbers.
- U1 falsified low (r better than band) → same, happily.
- **r(4) > 0.85 → the spec-dec lane is CLOSED on byte economics** and the
  roadmap says so; no reinterpretation.

## Fixture & cost

One pod, ≤ $5, teardown discipline + continuous evidence pull; the capture
doubles with the arena middle-anchor measurement (separate, unregistered
telemetry: VRAM/host-RSS/never-resident% on 235B). A2000/QNAP used only for
the $0 offline union computation, never timing.

## Not claimed

No tok/s numbers here (bytes only — acceptance rate and overhead division are
S1's prereg). Nothing transfers across host classes; nothing about chat.
