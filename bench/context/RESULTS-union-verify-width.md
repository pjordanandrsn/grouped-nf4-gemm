# RESULTS — routed-union vs verify-width (PREREG `0b4ec29c`)

**U1 HOLDS · U2 HOLDS · the spec-dec lane is OPEN, as a registered result.**

Fresh Qwen3-235B-A22B decode capture, 2026-07-27: 4 prompts × 256 decode
tokens (the registered minimum), 94 layers, 96,256 records, L40S pod
(`zprcaxjch4mper`), capture harness with `--incremental --skip-audit`,
windows never spanning non-consecutive `record_token` indices.

| w | union | ratio r(w) | uniform null | n windows |
|---|---|---|---|---|
| 2 | 12.63 | 0.7892 | 15.50 | 95,880 |
| 4 | 19.60 | **0.6125** | 29.12 | 95,128 |
| 8 | 29.21 | **0.4564** | 51.62 | 93,624 |
| 16 | 40.41 | 0.3157 | 82.42 | 90,616 |

- **U1**: r(4)=0.6125 ∈ [0.50, 0.70] ✓ ; r(8)=0.4564 ∈ [0.37, 0.53] ✓.
- **U2**: realized union ≤ 0.9×null at every w ≥ 2 (union/null 0.82 → 0.49);
  routing concentration is structural, not chance.
- **Lane rule**: r(4) ≤ 0.85 ⇒ the token-speculation lane proceeds; its
  end-to-end prereg derives tok/s from THESE measured ratios, never from the
  exploratory ones.
- **Cross-model agreement** (the ladder working as designed): the 30B
  exploratory anchor predicted 0.598 / 0.446; the 235B registered result
  landed 0.6125 / 0.4564 — within ~2.5% on a model 8× larger, same E/k.

Disclosures: capture ran with inference prefetch ON (copy scheduling only —
routing labels are staging-invariant per the bit-identity results; #16/#22).
The MEM_ANCHOR middle-anchor telemetry was LOST: it was placed after the
audit section that `--skip-audit` skips, and the process exited before a
/proc salvage — placement bug, disclosed; the 235B arena anchor waits for the
next load. Two SECURE 4090s wedged before this pod (empty runtime ≥10 min;
deleted, 404-verified, $0.34); the capture ran on the L40S instead.

Evidence: `union-235b-registered.json` (this dir);
`~/e4b-evidence/m1-union/m1-evidence.tgz` sha256 `a6cb37a6…` (topk/meta/join
arrays + logs + both result JSONs).
