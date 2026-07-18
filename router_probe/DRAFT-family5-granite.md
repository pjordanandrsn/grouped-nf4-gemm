# DRAFT — Family 5: Granite-3.0-3B-A800M (E=40, k=8, L=32)

> **DRAFT / addendum-hold.** Receipt banked at
> `receipts/20260718/EXPLORATORY_phase1_granitemoe.json`; not appended to
> RESULTS.md, not stamped, until reviewed.

**Why this family.** A low-E anchor at k=8: OLMoE (E=64) and Qwen3-30B (E=128)
share k=8, and the gpt-oss pair showed H falling E=32→E=128 at fixed k=4.
Granite (E=40, k=8, Apache) tests whether the k=8 axis shows the same
E-dependence. Capture: 98,304 records (12 prompts × 256 tok × 32 layers),
resident NF4 via the experts4bit streaming loader (`granitemoe` adapter —
router at `block_sparse_moe.router`, logits at tuple position 2, expert count
from `num_local_experts`), DO H100.

| Δ | linear | MLP-d | MLP-4d | attn2 | attn4 | attn4_w512 | attn6_w512 | ceiling | verdict |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.569 | 0.895 | 0.898 | 0.888 | 0.898 | 0.900 | **0.901** | 0.901 | **model-limited** |
| 2 | 0.555 | 0.887 | 0.891 | 0.882 | 0.890 | 0.892 | 0.888 | 0.892 | **model-limited** |
| 4 | 0.563 | 0.881 | 0.885 | 0.876 | 0.884 | 0.884 | 0.881 | 0.885 | **model-limited** |

Verdicts from the committed reducer (`reduce/reduce_ceiling.py`):
**model-limited at all three leads** — the second family (after OLMoE) to get
the clean verdict, at the first attempt and the smallest data volume of any
family.

**The shape is the finding.** Granite's ladder is flat from the *MLP-d rung
onward* — a one-hidden-layer MLP on the local features already reads 0.895,
and five further rungs of capacity (through attn6_w512) buy +0.006. Granite's
future routing is almost entirely predictable from cheap, local,
current-position features; there is no cross-stream structure for the
attention probes to find (contrast Qwen, where attn2 sat +0.20 above MLP-4d
and the ladder never flattened). Same clean-plateau class as OLMoE, reached
one rung earlier.

**E-axis reading (5 families).** At k=8: E=40 → 0.90, E=64 → 0.91, E=128 →
≈0.845-and-unpinned. At k=4: E=32 → 0.83, E=128 → 0.745. Both k-slices are
consistent with H declining in E with the decline concentrated at high E
(≈flat 40→64, down at 128), and k=8 sitting above k=4 at comparable E. Granite
strengthens the headline: **the wire-law H is a family property dominated by
expert count, not top-k alone.**

**Ops note.** Three fires to land: (1) config crash — `GraniteMoeConfig` names
the expert count `num_local_experts` (loader had already quantized 32/32
layers cleanly, validating the granitemoe path); (2) H100 boot race — CUDA
init returned error 802 (fabric manager not yet up); the runner now retries
CUDA init for up to 6 min before failing loud.
