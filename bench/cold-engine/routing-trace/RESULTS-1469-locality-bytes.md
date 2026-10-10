# Nearby-token expert reuse, in bytes (experts4bit-qlora#1469, items 2 and 3)

Receipt: [`locality-1469.json`](locality-1469.json). Harness: [`locality_1469.py`](locality_1469.py). Traces: the 16
committed `{olmoe,granite,qwen,qwen3}_{prose,code,math,dialogue}.jsonl`, 512 autoregressive decode steps each in
[`capture_routing.py`](capture_routing.py)'s format (`qwen` is Qwen1.5-MoE-A2.7B, `qwen3` is Qwen3-30B-A3B). Policy
transfers: this campaign's [`oracle-headroom.json`](oracle-headroom.json); its Qwen3-30B-A3B rows were added with
`oracle_headroom.py --models qwen3 --steps-held 1.0,1.5,2.0`, the other rows are unchanged. No box, no GPU for this
read.

| traces | captured by | prompts × decode steps | versions |
|---|---|---|---|
| olmoe, granite, qwen | this campaign, `capture_routing.py` | 4 × 512, 64-token prompts | as recorded in each trace |
| qwen3 | experts4bit-qlora#1469 item 1 census, box `loc-a4000-1` (RTX A4000), e4b NF4 host residency path; read in experts4bit-qlora#1555 | 4 × 512, 64-token prompts | Qwen3-30B-A3B `ad44e777`, transformers 5.16.1, torch 2.8.0 (in each trace's `meta`) |

The `qwen3_*.jsonl` files are the census's `decode_<kind>.jsonl` (gzipped in experts4bit-qlora#1555's receipts), byte for byte (sha256 `1293f390…` prose, `64647ca1…`
code, `b432a869…` math, `1dc7aea0…` dialogue).

## The question

Stepped MoE (arXiv:2610.07348) routes once per segment of S tokens, so a segment's experts load once. Existing
checkpoints route every token, and nothing here changes that. experts4bit-qlora#1469 asks the transferable question: how much reuse does
**exact** per-token routing already have among nearby tokens, and what is it worth in bytes moved per decoded token?

## Prior work this builds on

This campaign already measured temporal locality on these traces. [R4](RESULTS-r4.md) found that at 4–8-token windows
long-run expert **frequency** predicts reuse better than recency at every capacity with signal, 38 of 40 cells across
two models ([`RESULTS-generalization.md`](RESULTS-generalization.md)). [`oracle_headroom.py`](oracle_headroom.py) scored
the shipped device cache, LRU and Belady's optimum. Decode wall-clock is transfer-bound on real routing, r = 0.987
([`../RESULTS-wall-real-routing.md`](../RESULTS-wall-real-routing.md)).

## What nearby tokens share (window W, non-overlapping, within the sequence)

| model (layers × experts, top-k) | churn | union per layer, W = 16 / 64 | loads per token, W = 1 / 16 / 64 |
|---|---|---|---|
| OLMoE-1B-7B (16 × 64, top-8) | 0.45–0.66 | 36.0 / 44.8 | 128 / 36.0 / 11.2 |
| Granite-3.0-3B-A800M (32 × 40, top-8) | 0.54–0.60 | 29.6 / 35.2 | 256 / 59.2 / 17.6 |
| Qwen1.5-MoE-A2.7B (24 × 60, top-4) | 0.84–0.90 | 28.2 / 40.1 | 96 / 42.3 / 15.0 |
| Qwen3-30B-A3B (48 × 128, top-8) | 0.60–0.64 | 42.4 / 63.9 | 384 / 127.2 / 47.9 |

- **Churn is high.** 45–90% of a layer's top-k changes from one token to the next.
- **Unions grow slowly.** A 16-token window touches 36 of OLMoE's 64 experts per layer, against 128 routed slots.
  A cache holding exactly the current window's experts loads 3.6× fewer rows per token at W = 16, and 11× fewer at
  W = 64.

## That cache needs most of the arena, and a persistent cache does far better there

The window cache's capacity is every layer's largest window union. At that same capacity, Belady's optimum (the floor
no replacement policy goes below) is:

| model | W = 16: capacity (share of arena) | window cache, rows/token | Belady at that capacity, rows/token |
|---|---|---|---|
| OLMoE | 729 rows (71%) | 36.0 | 2.4 |
| Granite | 1,085 rows (85%) | 59.2 | 2.9 |
| Qwen1.5-MoE | 910 rows (63%) | 42.3 | 5.0 |
| Qwen3-30B-A3B | 2,479 rows (40%) | 127.2 | 15.3 |

- **Flushing at a segment boundary throws the reuse away.** Most of a window's experts recur in the next window, so a
  cache that persists across windows moves 8–20× fewer rows at the same capacity (Qwen3-30B-A3B: 8.3×).
- **Nearby-token reuse under exact routing is a capacity question, not a segment question.** It is what a persistent,
  frequency-led cache already captures. That is R4's finding, now in bytes.
- **Belady is an oracle.** The online policies sit above it: the shipped cache and LRU are 1.5–2.5× Belady at
  1–2 steps' worth of rows, pooled over the four prompts ([`oracle-headroom.json`](oracle-headroom.json); [`RESULTS-policies.md`](RESULTS-policies.md)).

## Bytes and link floors per decoded token

A row is one expert of one layer, gate, up and down in NF4 with an fp32 absmax per 64 weights: 3.54 MB (OLMoE),
1.33 MB (Granite), 4.87 MB (Qwen1.5-MoE), 2.65 MB (Qwen3-30B-A3B). Floors divide bytes per token by measured throughput:

- host-to-device PCIe: RTX A2000 6.24 GB/s, the median of 10 loggetta receipts;
- host-to-device PCIe: RTX 5090 20.75 GB/s, the median of 3;
- NVMe read on the owned NAS: 3.3 GB/s ([`../../nvme/receipts`](../../nvme/receipts)).

A floor bounds transfer time; it does not predict a step.

| model | source | rows/token | MB/token | A2000 PCIe ms | 5090 PCIe ms | NVMe ms |
|---|---|---|---|---|---|---|
| OLMoE | no reuse (W = 1) | 128.0 | 453.0 | 72.6 | 21.8 | 137.3 |
| OLMoE | shipped cache, 256 rows | 54.6 | 193.2 | 31.0 | 9.3 | 58.6 |
| OLMoE | Belady, 256 rows | 26.4 | 93.6 | 15.0 | 4.5 | 28.4 |
| OLMoE | Belady, ~729 rows | 2.4 | 8.6 | 1.4 | 0.4 | 2.6 |
| Granite | no reuse | 256.0 | 339.7 | 54.5 | 16.4 | 103.0 |
| Granite | shipped cache, 512 rows | 94.8 | 125.8 | 20.2 | 6.1 | 38.1 |
| Granite | Belady, 512 rows | 38.7 | 51.4 | 8.2 | 2.5 | 15.6 |
| Qwen1.5-MoE | no reuse | 96.0 | 467.1 | 74.9 | 22.5 | 141.6 |
| Qwen1.5-MoE | shipped cache, 192 rows | 59.0 | 287.3 | 46.0 | 13.9 | 87.1 |
| Qwen1.5-MoE | Belady, 192 rows | 37.2 | 181.1 | 29.0 | 8.7 | 54.9 |
| Qwen3-30B-A3B | no reuse | 384.0 | 1,019.2 | 163.3 | 49.1 | 308.9 |
| Qwen3-30B-A3B | shipped cache, 768 rows | 184.3 | 489.2 | 78.4 | 23.6 | 148.2 |
| Qwen3-30B-A3B | Belady, 768 rows | 96.4 | 256.0 | 41.0 | 12.3 | 77.6 |
| Qwen3-30B-A3B | Belady, ~2,479 rows | 15.3 | 40.6 | 6.5 | 2.0 | 12.3 |

Every row of the receipt, including LRU and the 1 and 1.5-step capacities, is in `locality-1469.json`.

- **Host-bound vs GPU-bound.** With experts off the GPU and caches of 1–2 steps' worth of rows, the PCIe floor alone is
  tens of milliseconds per token. That is the regime the wall-time result found transfer-bound. A cache near the window
  capacities brings the floor to about a millisecond (Qwen3-30B-A3B: 2.0 ms on an RTX 5090, 6.5 ms on an A2000), where
  the GPU's own decode step decides.
- **Not measured here:** a clean single-token decode step time with all experts resident. The committed serve receipts
  record throughput with prefill and batching mixed in. The crossover is therefore stated in transfer floors, not
  located.

## Slice-granular residency

Splitting each expert's gate, up and down along intermediate-channel slices and caching slices exactly does not move
fewer bytes. Exact routing executes every slice of every routed expert, so a slice cache holds the same bytes per
routed expert as a row cache. It changes only how a fixed capacity packs: partial rows can use the space a whole row
cannot. Executing a subset of slices changes the numerics and is out of scope without a registered quality gate (experts4bit-qlora#1469).

## Item 3: loggetta's offload cost model

**Decision: no locality input now.**

- **Training offload moves the whole slab.** loggetta's host-residency training bound prices 2 × the slab per
  micro-batch, because experts4bit-qlora's offload engine stages each layer's whole stack for its forward and its
  recompute, whatever the router picks. Routing locality does not change those bytes.
- **Serving plans predict no throughput.** loggetta prices serving memory, not transfer time. There is no transfer-time
  term for a locality input to feed.
- **What would change it:**
  - a census that departs from these models. Qwen3-30B-A3B's (item 1) does not: its churn sits inside the other three
    models' range, and its persistent-cache ratio (8.3×) is at the low end of theirs (8.5–20×). Its window unions are
    larger in absolute terms (128 experts per layer), but a smaller share of the arena (40% at W = 16);
  - a measured serve-path fit of decode time against transfers per step;
  - a loggetta serve transfer bound that would consume the result.

  Until then, these numbers are evidence for experts4bit-qlora's and grouped-nf4-gemm's residency work, not a planner
  input.
