# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Real-model capture hooks (Phase-1 instrument; Phase-0 deliverable).

NOT RUN under Phase 0 — the charter forbids touching any real checkpoint until
the fixture gate reads 4/4. Written now so Phase 1 is a run command, not a
build; smoke path is the fixtures, which share the serialization format.

Captures, per decode step and per MoE layer l (bs1 only, prefill excluded):
  hidden_post_block_l  — the layer-l block output a runtime predictor would hold
  router_logits_l      — layer-l router scores (contract stream 2)
  token_embedding      — the input embedding for the current token (stream 3)
  topk_set             — the REALIZED top-k at layer l (label stream; the
                         loader's Δ join turns these into l+Δ labels)

Adapters map architecture-specific module paths; add entries, never edit the
capture math. Records are appended layer-major within a token so the loader's
"Δ records ahead" == "Δ layers ahead" invariant holds (fixtures emulate this).
"""
from __future__ import annotations

import numpy as np

ADAPTERS = {
    # module path templates per family; {i} = layer index
    "olmoe": {
        "block": "model.layers.{i}",
        "gate": "model.layers.{i}.mlp.gate",
        "embed": "model.embed_tokens",
        "k_attr": "num_experts_per_tok",
    },
    "qwen3_moe": {
        "block": "model.layers.{i}",
        "gate": "model.layers.{i}.mlp.gate",
        "embed": "model.embed_tokens",
        "k_attr": "num_experts_per_tok",
    },
}


def _resolve(model, path):
    m = model
    for part in path.split("."):
        m = m[int(part)] if part.isdigit() else getattr(m, part)
    return m


class DecodeCapture:
    """Attach to a HF MoE model; collects contract streams during bs1 decode.

    Usage (Phase 1):
        cap = DecodeCapture(model, family="qwen3_moe", layers=range(0, L))
        cap.arm()
        ... generate(...) one token at a time (decode only; skip prefill) ...
        cap.disarm(); cap.save(out_dir, E=..., k=...)
    """

    def __init__(self, model, family: str, layers):
        self.model = model
        self.ad = ADAPTERS[family]
        self.layers = list(layers)
        self.family = family
        self.buf = {"hidden": [], "logits": [], "embed": [], "topk": []}
        self._handles = []
        self._embed_now = None
        self._decode_mode = False
        self._k = None

    def set_k(self, k: int):
        self._k = int(k)

    def arm(self):
        emb = _resolve(self.model, self.ad["embed"])
        self._handles.append(emb.register_forward_hook(self._embed_hook))
        for i in self.layers:
            blk = _resolve(self.model, self.ad["block"].format(i=i))
            gate = _resolve(self.model, self.ad["gate"].format(i=i))
            self._handles.append(gate.register_forward_hook(self._gate_hook))
            self._handles.append(blk.register_forward_hook(self._block_hook))

    def begin_decode(self):
        """Call after prefill, before the first generated token (bs1)."""
        self._decode_mode = True

    def _is_decode(self, t):
        return self._decode_mode and t.shape[-2] == 1  # seq-len-1 forward = decode step

    def _embed_hook(self, mod, args, out):
        if self._is_decode(out):
            self._embed_now = out.detach()[0, -1].float().cpu().numpy()

    def _gate_hook(self, mod, args, out):
        logits = out[0] if isinstance(out, tuple) else out
        if not self._decode_mode or self._embed_now is None:
            return
        v = logits.detach().reshape(-1, logits.shape[-1])[-1].float().cpu().numpy()
        self.buf["logits"].append(v)
        k = self._k
        self.buf["topk"].append(np.argsort(-v)[:k].astype(np.int32))
        self.buf["embed"].append(self._embed_now)

    def _block_hook(self, mod, args, out):
        h = out[0] if isinstance(out, tuple) else out
        if self._is_decode(h):
            self.buf["hidden"].append(h.detach()[0, -1].float().cpu().numpy())

    def disarm(self):
        for h in self._handles:
            h.remove()
        self._handles = []

    def save(self, out_dir, E: int, k: int, extra_meta=None):
        from capture.streams import write_capture
        n = min(len(self.buf["hidden"]), len(self.buf["logits"]),
                len(self.buf["embed"]), len(self.buf["topk"]))
        meta = {"E": E, "k": k, "family": self.family, "decode_only": True,
                "layers": self.layers, "records": n,
                "tier": "EXPLORATORY"}
        meta.update(extra_meta or {})
        write_capture(
            out_dir,
            np.stack(self.buf["hidden"][:n]),
            np.stack(self.buf["logits"][:n]),
            np.stack(self.buf["embed"][:n]),
            np.stack(self.buf["topk"][:n]),
            meta,
        )
        return n
