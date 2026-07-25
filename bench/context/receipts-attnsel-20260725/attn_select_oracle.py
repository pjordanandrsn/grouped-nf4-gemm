# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Post-hoc diagnostic for Experiment A. NOT a registered test.

A2 was falsified: H2O at a 132-token budget recovers only ~23% of the log gap
that recency destroyed on the induction fixture. Two very different causes give
that same number, and the pre-committed decision ("token-axis sparsity is closed
out") reads differently under each:

  * **scoring** — accumulated attention is the wrong importance signal, and a
    better one would do better at the same budget; or
  * **timing** — the signal is fine but unobservable when it is needed. The
    first eviction fires after 256 tokens, and *nothing has yet attended* to the
    tokens the second copy will need. No causal selector can know.

An oracle separates them. It scores with attention accumulated over the WHOLE
sequence — including the second-copy queries that have not run yet at eviction
time — and then evicts to the same 132-token budget. It is not a policy anyone
could deploy; it is an upper bound on what selection at this budget can buy.

  oracle ~= H2O   -> scoring is at its ceiling; the budget is the binding limit
  oracle >> H2O   -> the signal exists and only arrives too late

Deliberately does not modify ``attn_select.py``: the registered arms have run,
and the harness that produced them should stay exactly as it was stamped. Both
policies here recover the chunk geometry from the attention tensor's own shape
rather than needing a new hook.
"""
import json
import os
import sys

import torch

sys.path.insert(0, "/root/g/kernel")
sys.path.insert(0, "/root/g/bench/context")
import attn_select as A  # noqa: E402
from experts4bit_qlora import NF4KVCache, load_moe_4bit_streaming  # noqa: E402

OUT = os.environ.get("O_OUT", "/root/g/bench/context/attn_select_oracle.json")
SEEDS = [int(s) for s in os.environ.get("O_SEEDS", "0,1,2").split(",")]


class _Tracked:
    """Mixin: carry the ABSOLUTE position of every held slot alongside the cache.

    The score table is indexed by position; the cache is indexed by slot.
    Eviction is exactly what makes those disagree, so a policy that scores by
    position has to keep the map itself.
    """

    def _note(self, attentions):
        q = attentions[0].shape[2]
        new = torch.arange(self.seen, self.seen + q, device=attentions[0].device)
        self.seen += q
        for li in range(len(attentions)):
            prev = self.pos.get(li)
            self.pos[li] = new if prev is None else torch.cat([prev, new])

    def _select(self, n, score):
        head = torch.arange(self.sink, device=score.device)
        tail = torch.arange(n - self.recent, n, device=score.device)
        mid = torch.arange(self.sink, n - self.recent, device=score.device)
        k = min(self.topk, mid.shape[0])
        top = mid[torch.topk(score[mid], k).indices]
        return torch.sort(torch.cat([head, top, tail])).values


class Oracle(_Tracked):
    """sink + top-K by attention accumulated over the ENTIRE sequence + recent.

    Cheating by construction: the scores come from a prior pass that has already
    seen the whole fixture.
    """

    name = "oracle"
    wants_attention = True                 # only to recover the chunk length

    def __init__(self, scores, sink, recent, topk):
        self.scores = scores
        self.sink, self.recent, self.topk = sink, recent, topk
        self.budget = sink + recent + topk
        self.pos: dict[int, torch.Tensor] = {}
        self.seen = 0

    def cache_kwargs(self):
        return {}

    def observe(self, attentions):
        self._note(attentions)

    def step(self, cache):
        if not self.pos or next(iter(self.pos.values())).shape[0] <= self.budget:
            return
        keep = {li: self._select(p.shape[0], self.scores[li][p])
                for li, p in self.pos.items()}
        cache.evict_index(keep)
        for li, idx in keep.items():
            self.pos[li] = self.pos[li][idx]


class H2OTracked(A.H2O, _Tracked):
    """The registered H2O policy, instrumented to report WHICH tokens it keeps.

    Selection is the parent's, unchanged — this only records the position map so
    the keep-set can be inspected. Verified against the registered arm by
    perplexity, which must match to the last digit.
    """

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.pos: dict[int, torch.Tensor] = {}
        self.seen = 0

    def observe(self, attentions):
        super().observe(attentions)
        self._note(attentions)

    def step(self, cache):
        if not self.acc or next(iter(self.acc.values())).shape[0] <= self.budget:
            return
        keep = {li: self._select(a.shape[0], a) for li, a in self.acc.items()}
        cache.evict_index(keep)
        for li in keep:
            self.acc[li] = self.acc[li][keep[li]]
            self.pos[li] = self.pos[li][keep[li]]


def oracle_scores(model, ids, n_kv):
    """Pass 1: full cache, no eviction, accumulate attention over everything."""
    probe = A.H2O(0, 0, 10 ** 9, n_kv)          # budget unreachable -> never evicts
    A.run(model, ids, probe, quantize=False, split_at=None)
    return probe.acc


def fixture(seed):
    g = torch.Generator().manual_seed(seed)
    base = torch.randint(1000, 20000, (A.IND_HALF,), generator=g)
    return torch.cat([base, base]).unsqueeze(0).cuda()


def main():
    model, _ = load_moe_4bit_streaming(A.MODEL, device="cuda:0", dtype=torch.bfloat16,
                                       r=8, alpha=16, offload=True, pin=True,
                                       quant_type="nf4")
    model.eval()
    model.set_attn_implementation("eager")
    n_kv = getattr(model.config, "num_key_value_heads", model.config.num_attention_heads)

    rows, keepsets = [], {}
    for seed in SEEDS:
        ids = fixture(seed)
        sc = oracle_scores(model, ids, n_kv)
        for quant in (False, True):
            pol = Oracle(sc, A.SINK, A.H2O_RECENT, A.H2O_TOPK)
            r = A.run(model, ids, pol, quant, split_at=A.IND_HALF)
            r.update(fixture=f"induction-s{seed}", arm="oracle-132",
                     dtype="nf4" if quant else "fp16", registered=False, seed=seed)
            rows.append(r)
            print(f"induction-s{seed} oracle-132 {'nf4 ' if quant else 'fp16'} "
                  f"2nd={r['ppl']['second']:11.3f} held={r['held_tokens']}", flush=True)
            if not quant:
                keepsets.setdefault(seed, {})["oracle"] = sorted(pol.pos[0].tolist())
        pol = H2OTracked(A.SINK, A.H2O_RECENT, A.H2O_TOPK, n_kv)
        r = A.run(model, ids, pol, False, split_at=A.IND_HALF)
        print(f"induction-s{seed} h2o-132(tracked) fp16 "
              f"2nd={r['ppl']['second']:11.3f}  <- must match the registered arm",
              flush=True)
        rows.append(dict(r, fixture=f"induction-s{seed}", arm="h2o-132-tracked",
                         dtype="fp16", registered=False, seed=seed))
        keepsets[seed]["h2o"] = sorted(pol.pos[0].tolist())
        json.dump(rows, open(OUT, "w"), indent=2)

    # WHICH tokens each rule ends up holding. The fixture needs first-copy
    # positions: predicting token p (>= 256) requires holding p-256 and p-257.
    print("\n--- layer-0 keep-set after the final eviction ---")
    summary = {}
    for seed, ks in keepsets.items():
        for name, kept in ks.items():
            first = [p for p in kept if p < A.IND_HALF]
            summary[f"s{seed}-{name}"] = dict(n=len(kept), n_first_copy=len(first),
                                              first_copy=first)
            print(f"seed {seed} {name:<7}: {len(kept):3d} slots, "
                  f"{len(first):3d} from the first copy  {first[:20]}")
    json.dump(dict(rows=rows, keepsets=summary), open(OUT, "w"), indent=2)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
