# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Experiment A — attention-based (H2O-style) selection vs recency eviction.

Registered in ``bench/context/PREREG-kv-context.md`` (OTS-stamped before this
ran). Finding #11 measured *recency* eviction and concluded token-axis sparsity
costs 28x more quality than quantization at matched bytes. Two confounds were
flagged in advance of that conclusion, and this tests both:

  (a) sink+recent is the weakest possible selection rule, and
  (b) wikitext next-token prediction is the least favourable task, because it
      depends on exactly the dense recent context the policy deletes.

So: two selection rules (recency, H2O) x two fixtures (wikitext, induction)
x {fp16, nf4}, all at matched bytes.

The induction fixture is a MECHANISM TEST, labelled as such per the prereg's
standing rule 4: it is built to contain a sparse long-range dependency, so a
good result on it is evidence about the mechanism, not about real workloads.
Its point is that a 128-token recency window makes induction *impossible* --
the matching token sits 256 positions back -- which is the regime eviction is
supposed to serve and the regime wikitext cannot probe.

Two interpretations were needed where the prereg underspecified, both fixed
before the run and recorded in the receipts:

  * K (the top-K budget) is chosen so H2O keeps exactly as many tokens as the
    recency arm it is compared against -- standing rule 1 is matched BYTES, and
    equal token counts at equal quantization is exactly that.
  * The prereg accumulates attention per (layer, kv head, key position) but does
    not say at what granularity the keep-set is chosen. It cannot be per-head:
    the packed store is ``[T, H, D]`` with ONE token axis shared by every head,
    so a per-head keep-set is not representable without ragged storage. Scores
    are therefore summed over kv heads and the keep-set is per LAYER. This is
    coarser than published H2O and, like the prereg's confound (iii), should if
    anything understate H2O.
"""
import json
import os
import sys
import time

import torch

sys.path.insert(0, "/root/g/kernel")
from experts4bit_qlora import NF4KVCache, load_moe_4bit_streaming  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

MODEL = os.environ.get("A_MODEL", "allenai/OLMoE-1B-7B-0924")
CH = int(os.environ.get("A_CHUNK", "128"))
WIKI_N = int(os.environ.get("A_WIKI_TOKENS", "1024"))
IND_HALF = int(os.environ.get("A_IND_HALF", "256"))
IND_SEEDS = [int(s) for s in os.environ.get("A_IND_SEEDS", "0,1,2").split(",")]
OUT = os.environ.get("A_OUT", "/root/g/bench/context/attn_select.json")

SINK, REC_RECENCY = 4, 128            # the registered recency arm: 132 tokens held
H2O_RECENT, H2O_TOPK = 64, 64         # sink4 + rec64 + top64 = the same 132


# --------------------------------------------------------------------------
# policies
# --------------------------------------------------------------------------
class Recency:
    """sink + recent. What #11 measured; the control for selection."""

    name = "recency"
    wants_attention = False

    def __init__(self, sink=SINK, recent=REC_RECENCY):
        self.sink, self.recent = sink, recent
        self.budget = sink + recent

    def cache_kwargs(self):
        return dict(keep_sink=self.sink, keep_recent=self.recent)

    def observe(self, attentions):
        pass

    def step(self, cache):
        cache.evict()


class H2O:
    """sink + top-K by accumulated attention + recent.

    Attention mass is accumulated per (layer, kv head, key position), summed
    over query positions and over the query heads mapping to each kv head, then
    summed over kv heads to give the per-layer score the shared token axis
    requires. Only attention observed SO FAR is used -- selection at chunk
    boundaries sees no future queries, which is the whole point of testing
    whether a retrospective importance signal can serve a prospective need.
    """

    name = "h2o"
    wants_attention = True

    def __init__(self, sink=SINK, recent=H2O_RECENT, topk=H2O_TOPK, n_kv_heads=None):
        self.sink, self.recent, self.topk = sink, recent, topk
        self.budget = sink + recent + topk
        self.n_kv_heads = n_kv_heads
        self.acc: dict[int, torch.Tensor] = {}

    def cache_kwargs(self):
        return {}                       # eviction is driven by evict_index, not evict()

    def observe(self, attentions):
        """``attentions[l]`` is ``[B, H_q, q_len, kv_len]`` from eager attention."""
        for li, a in enumerate(attentions):
            hq, q, kv = a.shape[1], a.shape[2], a.shape[3]
            s = a[0].float().sum(dim=1)                      # over query positions -> [H_q, kv]
            hkv = self.n_kv_heads or hq
            s = s.view(hkv, hq // hkv, kv).sum(1)            # query heads -> their kv head
            s = s.sum(0)                                     # -> [kv], the shared token axis
            prev = self.acc.get(li)
            if prev is not None:
                s[: prev.shape[0]] += prev                   # older slots keep their history
            self.acc[li] = s

    def step(self, cache):
        keep = {}
        if not self.acc or next(iter(self.acc.values())).shape[0] <= self.budget:
            return                                           # same guard as evict()
        for li, acc in self.acc.items():
            n = acc.shape[0]
            head = torch.arange(self.sink, device=acc.device)
            tail = torch.arange(n - self.recent, n, device=acc.device)
            mid = torch.arange(self.sink, n - self.recent, device=acc.device)
            k = min(self.topk, mid.shape[0])
            top = mid[torch.topk(acc[mid], k).indices]
            keep[li] = torch.sort(torch.cat([head, top, tail])).values
        cache.evict_index(keep)
        for li, idx in keep.items():
            self.acc[li] = self.acc[li][idx]                 # scores follow the tokens


class Full:
    name = "full"
    wants_attention = False
    budget = None

    def cache_kwargs(self):
        return {}

    def observe(self, attentions):
        pass

    def step(self, cache):
        pass


# --------------------------------------------------------------------------
# scoring
# --------------------------------------------------------------------------
def run(model, ids, policy, quantize, split_at=None):
    """Teacher-forced in chunks so eviction lands BETWEEN forwards.

    Returns per-region perplexity plus both byte figures: ``resident`` is the
    post-eviction steady state (#11's convention, so the numbers compare) and
    ``peak`` is post-update/pre-eviction, which is what the allocator actually
    has to hold. #11 reported only the former.
    """
    n = ids.shape[1]
    cache = NF4KVCache(quantize_keys=quantize, quantize_values=quantize,
                       **policy.cache_kwargs())
    nll = {"all": 0.0, "first": 0.0, "second": 0.0}
    cnt = {"all": 0, "first": 0, "second": 0}
    resident = peak = 0
    for lo in range(0, n, CH):
        length = min(CH, n - lo)
        with torch.no_grad():
            out = model(ids[:, lo:lo + length], past_key_values=cache, use_cache=True,
                        output_attentions=policy.wants_attention)
        if policy.wants_attention:
            assert out.attentions is not None and out.attentions[0] is not None, (
                "output_attentions returned nothing -- the model is not on the eager "
                "attention path, so H2O has no scores to select with")
            policy.observe(out.attentions)
        last = length - 1 if lo + length >= n else length     # final position has no target
        if last > 0:
            logp = torch.log_softmax(out.logits[0, :last].float(), -1)
            tgt = ids[0, lo + 1:lo + 1 + last]
            tok_nll = -logp.gather(1, tgt[:, None])[:, 0]
            pos = torch.arange(lo + 1, lo + 1 + last, device=ids.device)
            nll["all"] += float(tok_nll.sum())
            cnt["all"] += last
            if split_at is not None:
                m = pos >= split_at
                nll["second"] += float(tok_nll[m].sum())
                cnt["second"] += int(m.sum())
                nll["first"] += float(tok_nll[~m].sum())
                cnt["first"] += int((~m).sum())
        peak = max(peak, cache.memory_bytes())
        policy.step(cache)
        resident = max(resident, cache.memory_bytes())
        del out
        torch.cuda.empty_cache()
    ppl = {k: (float(torch.exp(torch.tensor(nll[k] / cnt[k]))) if cnt[k] else None)
           for k in nll}
    held = cache.held_length(0)
    del cache
    torch.cuda.empty_cache()
    return dict(ppl=ppl, counts=cnt, resident_bytes=resident, peak_bytes=peak,
                held_tokens=held)


def main():
    torch.manual_seed(0)
    tok = AutoTokenizer.from_pretrained(MODEL)
    model, _ = load_moe_4bit_streaming(MODEL, device="cuda:0", dtype=torch.bfloat16,
                                       r=8, alpha=16, offload=True, pin=True,
                                       quant_type="nf4")
    model.eval()
    # EVERY arm runs eager, not just the H2O ones. H2O needs output_attentions
    # (prereg confound (ii)); running the controls on sdpa would leave a kernel
    # difference sitting between the arms being compared, for no benefit -- the
    # cost is speed, and speed is not measured here.
    model.set_attn_implementation("eager")
    cfg = model.config
    n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
    print(f"model={MODEL} layers={cfg.num_hidden_layers} "
          f"H_q={cfg.num_attention_heads} H_kv={n_kv} "
          f"attn={cfg._attn_implementation}", flush=True)

    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    wiki = tok("\n\n".join(x for x in ds["text"] if x.strip()),
               return_tensors="pt").input_ids[:, :WIKI_N].cuda()

    def induction(seed):
        g = torch.Generator().manual_seed(seed)
        base = torch.randint(1000, 20000, (IND_HALF,), generator=g)
        return torch.cat([base, base]).unsqueeze(0).cuda()

    # Factories, not instances: H2O carries accumulated scores, so every arm
    # must start from a clean one or the second dtype inherits the first's.
    #
    # Registered: full, recency(sink4+rec128), H2O(sink4+rec64+top64) -- the last
    # two hold exactly 132 tokens, so they are compared at equal bytes. The
    # 260-token pair is UNREGISTERED and exploratory: A3 quotes recency's
    # +3.316, which is #11's sink4+rec256 arm, so scoring A3 against its own
    # cited reference needs a matched H2O at that budget too. Reported as
    # exploratory, never swapped in for a registered arm.
    ARMS = [
        ("full", lambda: Full(), True),
        ("recency-132", lambda: Recency(SINK, REC_RECENCY), True),
        ("h2o-132", lambda: H2O(SINK, H2O_RECENT, H2O_TOPK, n_kv), True),
        ("recency-260", lambda: Recency(SINK, 256), False),
        ("h2o-260", lambda: H2O(SINK, H2O_RECENT, 192, n_kv), False),
    ]

    rows = []
    jobs = [("wikitext", wiki, None, None)]
    jobs += [(f"induction-s{s}", induction(s), IND_HALF, s) for s in IND_SEEDS]
    for fixture, ids, split_at, seed in jobs:
        for arm, make, registered in ARMS:
            for quant in (False, True):
                pol = make()
                t0 = time.time()
                r = run(model, ids, pol, quant, split_at)
                r.update(fixture=fixture, arm=arm, dtype="nf4" if quant else "fp16",
                         registered=registered, seed=seed, secs=round(time.time() - t0, 1),
                         budget=pol.budget, tokens=ids.shape[1])
                rows.append(r)
                tag = f"{fixture:<14} {arm:<12} {'nf4' if quant else 'fp16'}"
                sec = r["ppl"]["second"]
                print(f"{tag}  ppl={r['ppl']['all']:8.3f}"
                      f"  2nd={'   n/a' if sec is None else f'{sec:8.3f}'}"
                      f"  held={r['held_tokens']:4d}"
                      f"  res={r['resident_bytes'] / 2**20:7.2f}MB"
                      f"  peak={r['peak_bytes'] / 2**20:7.2f}MB  {r['secs']}s", flush=True)
                json.dump(rows, open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
