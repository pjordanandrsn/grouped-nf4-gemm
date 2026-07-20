# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""Per-expert -> fused-stack GATHER for DeepSeek-V3-lineage MoE checkpoints
(Kimi K2 today, Kimi K3 when its weights land) — the Moonshot analogue of
gpt-oss's `mxfp4_loader.to_kernel_shapes`.

Why this exists (shape recon, 2026-07-19, PLAN-kimi-k3-port.md): gpt-oss ships
its experts PRE-FUSED as `[E, N, ...]` blocks. DeepSeek-V3-lineage models
(Kimi K2 = `DeepseekV3ForCausalLM`, and K3 as its evolution) ship experts as
INDIVIDUAL per-expert tensors —
    model.layers.{L}.mlp.experts.{i}.{gate,up,down}_proj.weight   (+ a scale)
with gate and up as SEPARATE tensors. Our kernel/engine want the fused arena
`gate_up [E, 2N, ...]` / `down [E, hidden, N]`. This module gathers the former
into the latter.

**The gather is format-agnostic on purpose.** It only stacks per-expert
tensors along a new E axis and concatenates gate+up; it never inspects the
dtype. So the SAME code path serves:
  * Kimi K2 = fp8 e4m3 with `[128,128]` block scales (`weight_scale_inv`) —
    testable TODAY; feeds an fp8 consumer, NOT our mxfp4 decode kernel.
  * Kimi K3 = MXFP4 (e2m1/e8m0) when released — feeds `ExpertsMxfp4` /
    `Mxfp4PipelinedGptOss` verbatim (same fused shapes as gpt-oss).
The format-specific decode is the CONSUMER's job, never the gather's.

**gate+up is a CLEAN CONCAT, not interleaved** (recon point 2): gate and up
arrive as separate tensors, so the fused block is `[gate ; up]` contiguous
(first N rows gate, next N up) and the epilogue is `chunk(2)` — the gpt-oss
`[...::2]` interleave gotcha does NOT recur here. `moonshot_apply_gate` below
encodes that; the nonlinearity is swappable (K2/DeepSeek = SwiGLU; K3 = SiTU,
read from source at seam time).

**Provenance for a gathered layout is PER-SOURCE-TENSOR.** Concatenation
reorders bytes, so `sha256(arena) != sha256(any single file range)`. The
honest receipt is: for each individual expert tensor, `sha256(file byte
range) == sha256(the exact bytes we placed into its arena slice)`. That is
what `verify_gather_provenance` asserts — the same "bit-identical to the
release" claim, at per-expert granularity.
"""
from __future__ import annotations

import hashlib
import re
from typing import Callable, Optional

import torch

# DeepSeek-V3 / Moonshot routed-expert tensor naming. The scale suffix is
# parametric: K2 uses `weight_scale_inv` (fp8 block-inv); K3's mxfp4 scale
# suffix is confirmed at seam time (pass via `scale_suffix`).
PROJ = ("gate_proj", "up_proj", "down_proj")
_EXPERT_RE = re.compile(
    r"^(?P<prefix>.*)\.layers\.(?P<layer>\d+)\.mlp\.experts\.(?P<idx>\d+)\."
    r"(?P<proj>gate_proj|up_proj|down_proj)\.(?P<kind>weight(?:_scale_inv|_scale|_scales)?)$"
)


def expert_tensor_name(layer, idx, proj, kind="weight", prefix="model"):
    return f"{prefix}.layers.{layer}.mlp.experts.{idx}.{proj}.{kind}"


def discover_layer(weight_map: dict, layer: int, prefix: str = "model") -> dict:
    """From a safetensors `weight_map` (name->shard), return
    {n_experts, has_scale, scale_suffix, weight_names?} for a routed-MoE layer.
    A layer whose experts.* namespace is empty (e.g. the dense
    `first_k_dense_replace` layer 0) reports n_experts=0."""
    idxs, scale_suffixes = set(), set()
    pat = f".layers.{layer}.mlp.experts."
    for name in weight_map:
        if pat not in name:
            continue
        m = _EXPERT_RE.match(name)
        if not m or int(m["layer"]) != layer:
            continue
        idxs.add(int(m["idx"]))
        if m["kind"] != "weight":
            scale_suffixes.add(m["kind"])
    n = (max(idxs) + 1) if idxs else 0
    if idxs and sorted(idxs) != list(range(n)):
        raise ValueError(f"layer {layer}: non-contiguous expert indices "
                         f"(got {len(idxs)}, max {max(idxs)})")
    scale_suffix = None
    if len(scale_suffixes) > 1:
        raise ValueError(f"layer {layer}: multiple scale suffixes {scale_suffixes}")
    if scale_suffixes:
        scale_suffix = scale_suffixes.pop()
    return {"n_experts": n, "has_scale": scale_suffix is not None,
            "scale_suffix": scale_suffix}


def gather_layer(get_tensor: Callable[[str], torch.Tensor], layer: int,
                 n_experts: int, *, scale_suffix: Optional[str] = "weight_scale_inv",
                 prefix: str = "model", concat_gate_up: bool = True) -> dict:
    """Gather one layer's per-expert tensors into fused stacks.

    ``get_tensor(name)`` returns the tensor for a full tensor name (wrap a
    safetensors `safe_open`, a shard-map, or a plain dict). Returns a dict of
    fused tensors, dtype untouched:

      concat_gate_up=True (kernel-ready, gpt-oss-shaped):
        gate_up [E, 2N, *]  (contiguous [gate; up]),  down [E, H, N]
        gate_up_scale [E, 2Ns, *] (if scales),         down_scale [E, ...]
      concat_gate_up=False (keep separate; for an fp8 consumer that wants them):
        gate [E,N,*], up [E,N,*], down [E,H,N] (+ per-proj scales)

    Stacking is `torch.stack` along a new leading E axis — format-agnostic,
    zero decode.
    """
    def stack(proj, kind):
        return torch.stack([
            get_tensor(expert_tensor_name(layer, e, proj, kind, prefix))
            for e in range(n_experts)])

    out = {}
    gate_w = stack("gate_proj", "weight")
    up_w = stack("up_proj", "weight")
    down_w = stack("down_proj", "weight")
    if concat_gate_up:
        out["gate_up"] = torch.cat([gate_w, up_w], dim=1).contiguous()  # [E, 2N, *]
    else:
        out["gate"], out["up"] = gate_w, up_w
    out["down"] = down_w

    if scale_suffix:
        gate_s = stack("gate_proj", scale_suffix)
        up_s = stack("up_proj", scale_suffix)
        down_s = stack("down_proj", scale_suffix)
        if concat_gate_up:
            out["gate_up_scale"] = torch.cat([gate_s, up_s], dim=1).contiguous()
        else:
            out["gate_scale"], out["up_scale"] = gate_s, up_s
        out["down_scale"] = down_s
    return out


def moonshot_apply_gate(gate_up: torch.Tensor, nonlinearity=torch.nn.functional.silu):
    """Clean-split GLU for concatenated [gate; up] (NOT interleaved): the GEMM
    over the fused `[2N, K]` weight yields `[T, 2N]`; the first N columns are
    gate, the next N up. K2/DeepSeek use SwiGLU (silu); pass K3's SiTU here
    once read from source (see GLU_VARIANTS / apply_glu)."""
    n = gate_up.shape[-1] // 2
    gate, up = gate_up[..., :n], gate_up[..., n:]
    return nonlinearity(gate) * up


# ---- swappable epilogue registry (K2 verified; K3 SiTU guarded) -------------
#
# The epilogue is the ONE genuinely model-specific piece of the port. The
# gather is format-agnostic; the decode is the consumer's; only the GLU
# nonlinearity differs by model. This registry makes activation a one-line
# swap AND refuses to run a guessed formula silently (R6 / do-not-overclaim):
# a variant is usable only when its formula is sourced.

def _swiglu(gate, up):
    """SwiGLU — VERIFIED. K2 is `DeepseekV3ForCausalLM`; the DeepSeek-V3 /
    Llama MoE expert epilogue is silu(gate) * up. This is the working default
    and the correctness baseline the SiTU variant will be diffed against."""
    return torch.nn.functional.silu(gate) * up


class _UnverifiedEpilogue:
    """A named epilogue whose formula is NOT yet sourced. Calling it raises —
    it never silently substitutes a guess. `candidates` records the plausible
    readings to CHECK against the model source / tech report, so activation is
    a one-line edit (drop the confirmed lambda in, delete this guard) — not a
    reverse-engineering task done under time pressure at seam time."""

    def __init__(self, name, why, candidates):
        self.name, self.why, self.candidates = name, why, candidates

    def __call__(self, *_a, **_k):
        raise NotImplementedError(
            f"{self.name}: formula UNVERIFIED — {self.why}. Do not guess (R6). "
            f"Confirm against the K3 model source, then register the real fn. "
            f"Candidate readings to disambiguate: {self.candidates}")


# SiTU = "Sigmoid Tanh Unit" (Kimi K3, launch coverage 2026-07-16). The tech
# report / model card is not out; the name alone underdetermines the formula.
# Guarded until a shard's modeling_*.py or the report pins it.
SITU_UNVERIFIED = _UnverifiedEpilogue(
    "SiTU",
    "K3 tech report unreleased; 'Sigmoid Tanh Unit' has several plausible forms",
    candidates=(
        "gate * sigmoid(gate) * tanh(up)      # sigmoid-gate, tanh on the up branch",
        "sigmoid(gate) * tanh(up)             # both branches nonlinear",
        "(gate * tanh(softplus(gate))) * up   # a Mish-like gate x linear up",
        "gate * tanh(sigmoid(gate)) * up      # composed gate x linear up",
    ),
)

# name -> callable(gate, up) -> hidden. Add K3's confirmed SiTU here at seam.
GLU_VARIANTS = {
    "swiglu": _swiglu,          # K2 / DeepSeek-V3 — VERIFIED
    "silu": _swiglu,            # alias
    "situ": SITU_UNVERIFIED,    # K3 — GUARDED until sourced
}


def apply_glu(gate_up: torch.Tensor, variant: str = "swiglu") -> torch.Tensor:
    """Clean-split GLU by registry name. `variant='swiglu'` works today;
    `'situ'` raises until its formula is confirmed and registered. The split
    is always clean-concat ([gate; up]), never interleaved."""
    fn = GLU_VARIANTS.get(variant)
    if fn is None:
        raise KeyError(f"unknown GLU variant {variant!r}; have {sorted(GLU_VARIANTS)}")
    n = gate_up.shape[-1] // 2
    return fn(gate_up[..., :n], gate_up[..., n:])


def register_glu_variant(name: str, fn) -> None:
    """Register a confirmed epilogue (e.g. K3's real SiTU once sourced):
    `register_glu_variant('situ', lambda g, u: <confirmed formula>)`."""
    GLU_VARIANTS[name] = fn


# ---- provenance (per-source-tensor; concat reorders bytes) ------------------
def file_sha256_map(path: str, layer: int, n_experts: int, *,
                    scale_suffix: Optional[str] = "weight_scale_inv",
                    prefix: str = "model") -> dict:
    """Per-expert-tensor sha256 of the file data-section byte ranges (the
    release bytes). Reuses `mxfp4_loader.file_tensor_sha256`."""
    from mxfp4_loader import file_tensor_sha256
    kinds = ("weight",) + ((scale_suffix,) if scale_suffix else ())
    table = {}
    for e in range(n_experts):
        for proj in PROJ:
            for kind in kinds:
                name = expert_tensor_name(layer, e, proj, kind, prefix)
                table[name] = file_tensor_sha256(path, name)
    return table


def _t_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_gather_provenance(get_tensor: Callable[[str], torch.Tensor],
                             file_hashes: dict, layer: int, n_experts: int, *,
                             scale_suffix: Optional[str] = "weight_scale_inv",
                             prefix: str = "model") -> dict:
    """Assert every per-expert tensor the gather READS is byte-identical to its
    release file range (`file_hashes`). This is the gathered-layout provenance
    receipt: the fused arena is built from exactly these verified bytes, so
    each arena slice inherits the identity. Raises on any mismatch."""
    report = {}
    kinds = ("weight",) + ((scale_suffix,) if scale_suffix else ())
    for e in range(n_experts):
        for proj in PROJ:
            for kind in kinds:
                name = expert_tensor_name(layer, e, proj, kind, prefix)
                got = _t_sha(get_tensor(name))
                want = file_hashes[name]
                report[name] = want == got
                if want != got:
                    raise ValueError(
                        f"PROVENANCE FAIL {name}: file {want[:16]} != loaded {got[:16]}")
    return report
