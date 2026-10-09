### Nearby-token expert reuse in bytes, on the committed routing traces (experts4bit-qlora#1469); no code change

`bench/cold-engine/routing-trace/RESULTS-1469-locality-bytes.md`; harness `locality_1469.py`; receipt
`locality-1469.json`. It is in sample, on the 12 committed decode traces of OLMoE-1B-7B, Granite-3.0-3B-A800M and
Qwen1.5-MoE-A2.7B. No box and no GPU.
- **Nearby tokens share experts, but only a cache of most of the arena keeps them.** Churn between consecutive tokens
  is 45–90% of a layer's top-k, and a 16-token window touches 28–36 experts per layer. A cache holding exactly a
  window's experts needs 63–85% of the arena. At that capacity, Belady's optimum moves 8–20× fewer rows per token than
  flushing at the window boundary.
- **So under exact routing, nearby-token reuse is a capacity question.** A persistent, frequency-led cache already
  captures it, which is R4's finding, now in bytes.
- **Bytes and link floors per decoded token,** against measured PCIe (RTX A2000, RTX 5090) and NAS NVMe throughput,
  for no reuse, the shipped cache, LRU and Belady.
- **Slice-granular exact caching moves the same bytes** and changes only how capacity packs.
- **loggetta's offload cost model gets no locality input now.** Training offload moves the whole slab, and serving
  predicts no transfer time.
