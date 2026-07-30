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
import os
import re
from typing import Callable, Optional

import torch

# DeepSeek-V3 / Moonshot routed-expert tensor naming. Only `prefix` used to be
# parametric, which was not enough: released K3 renamed the CONTAINER and the
# per-projection spellings too (see K3_SCHEME), so a hardcoded regex silently
# discovered zero experts. Naming is now a pluggable scheme; the gather logic
# it feeds is unchanged.
PROJ = ("gate_proj", "up_proj", "down_proj")


class NameScheme:
    """How one checkpoint family spells its routed-expert tensors.

    ``proj`` maps the canonical gate/up/down names this module speaks onto the
    on-disk spelling. ``weight_kind`` is what the family calls the weight
    itself — "weight" for a bare release, "weight_packed" for a pack-quantized
    one. A caller asking for ``kind="weight"`` gets ``weight_kind``; any other
    kind (i.e. a scale suffix) passes through verbatim, so callers never need
    to know which family they are on.
    """

    _KINDS = ("weight", "weight_scale_inv", "weight_scale", "weight_scales")

    def __init__(self, name, container, proj, weight_kind="weight",
                 default_scale_suffix="weight_scale_inv"):
        self.name = name
        self.container = container
        self.proj = dict(proj)
        self.weight_kind = weight_kind
        self.default_scale_suffix = default_scale_suffix
        alts = "|".join(re.escape(p) for p in dict.fromkeys(self.proj.values()))
        kinds = "|".join(re.escape(k) for k in
                         dict.fromkeys((weight_kind,) + self._KINDS))
        self.regex = re.compile(
            rf"^(?P<prefix>.*)\.layers\.(?P<layer>\d+)\.{re.escape(container)}\."
            rf"(?P<idx>\d+)\.(?P<proj>{alts})\.(?P<kind>{kinds})$")

    def spell(self, kind):
        """On-disk spelling of a requested tensor kind."""
        return self.weight_kind if kind == "weight" else kind

    def __repr__(self):
        return f"NameScheme({self.name!r}, container={self.container!r})"


# Kimi K2 (`DeepseekV3ForCausalLM`): bare fp8 weights under `mlp.experts`, with
# `weight_scale_inv` block-inv scales. The historical default — unchanged.
K2_SCHEME = NameScheme(
    "k2", "mlp.experts",
    {"gate_proj": "gate_proj", "up_proj": "up_proj", "down_proj": "down_proj"},
    weight_kind="weight", default_scale_suffix="weight_scale_inv")
# Kimi K3 AS RELEASED — measured 2026-07-30 against the real checkpoint
# (moonshotai/Kimi-K3, 93 layers, 896 experts/layer, top-16). Three renames vs
# K2: container `mlp.experts` -> `block_sparse_moe.experts`; projections
# gate/up/down_proj -> w1/w3/w2 (w1=gate, w3=up, w2=down, confirmed by shapes,
# not by convention); weights `weight` -> `weight_packed` with a `weight_scale`
# companion. This CONFIRMS the mxfp4 prediction in the module docstring:
# config declares format "mxfp4-pack-quantized", num_bits 4, group_size 32,
# e8m0 uint8 scales — so the fused arena feeds the mxfp4 decode path verbatim.
K3_SCHEME = NameScheme(
    "k3", "block_sparse_moe.experts",
    {"gate_proj": "w1", "up_proj": "w3", "down_proj": "w2"},
    weight_kind="weight_packed", default_scale_suffix="weight_scale")
SCHEMES = {s.name: s for s in (K2_SCHEME, K3_SCHEME)}
_EXPERT_RE = K2_SCHEME.regex  # back-compat alias for the historical default
_UNSET = object()


def resolve_scheme(scheme) -> NameScheme:
    """``None`` -> the K2 default; a key from ``SCHEMES``; or a NameScheme."""
    if scheme is None:
        return K2_SCHEME
    if isinstance(scheme, NameScheme):
        return scheme
    try:
        return SCHEMES[scheme]
    except KeyError:
        raise ValueError(
            f"unknown scheme {scheme!r}; known: {sorted(SCHEMES)} "
            "(or pass a NameScheme)") from None


def expert_tensor_name(layer, idx, proj, kind="weight", prefix="model",
                       scheme=None):
    sc = resolve_scheme(scheme)
    return (f"{prefix}.layers.{layer}.{sc.container}.{idx}."
            f"{sc.proj.get(proj, proj)}.{sc.spell(kind)}")


def discover_layer(weight_map: dict, layer: int, prefix: str = "model",
                   scheme=None) -> dict:
    """From a safetensors `weight_map` (name->shard), return
    {n_experts, has_scale, scale_suffix, weight_names?} for a routed-MoE layer.
    A layer whose experts.* namespace is empty (e.g. the dense
    `first_k_dense_replace` layer 0) reports n_experts=0.

    ``scheme`` selects the naming family (default K2). Pass ``"k3"`` for a
    released-K3 checkpoint — with the default scheme a K3 index legitimately
    reports n_experts=0, which is why the caller must say which family it has
    (or use :func:`detect_scheme`)."""
    sc = resolve_scheme(scheme)
    idxs, scale_suffixes = set(), set()
    pat = f".layers.{layer}.{sc.container}."
    for name in weight_map:
        if pat not in name:
            continue
        m = sc.regex.match(name)
        if not m or int(m["layer"]) != layer:
            continue
        idxs.add(int(m["idx"]))
        if m["kind"] != sc.weight_kind:
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


def detect_scheme(weight_map: dict, layer: int = 1):
    """Pick the scheme whose container/spelling this checkpoint actually uses.

    Returns the matching :class:`NameScheme`, or ``None`` if no known scheme
    finds experts (a genuinely unknown family — better to say so than to
    silently gather nothing)."""
    for sc in SCHEMES.values():
        if discover_layer(weight_map, layer, scheme=sc)["n_experts"]:
            return sc
    return None


def gather_layer(get_tensor: Callable[[str], torch.Tensor], layer: int,
                 n_experts: int, *, scale_suffix=_UNSET,
                 prefix: str = "model", concat_gate_up: bool = True,
                 scheme=None) -> dict:
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
    sc = resolve_scheme(scheme)
    if scale_suffix is _UNSET:
        scale_suffix = sc.default_scale_suffix

    def stack(proj, kind):
        return torch.stack([
            get_tensor(expert_tensor_name(layer, e, proj, kind, prefix, sc))
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

def make_situ(beta: float, linear_beta: float | None = None):
    """SiTU epilogue factory — SOURCED, not guessed.

    Transcribed from Kimi-K3's own `modeling_kimi_linear.py::SituAndMul`
    (fetched from the release 2026-07-30)::

        situ_a = beta * tanh(gate / beta) * sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * tanh(up / linear_beta)
        return situ_a * up

    Both branches are BOUNDED by a tanh whose scale is a config parameter — and
    note that **none of the four candidate readings guarded below is this**. The
    nearest (``gate * sigmoid(gate) * tanh(up)``) has the tanh on the wrong
    branch and no beta scaling, so guessing would have produced a model that ran
    and was quietly wrong. `beta`/`linear_beta` are per-checkpoint
    (`activation_situ_beta`, `activation_situ_linear_beta`), so a different
    checkpoint must re-register with its own values.
    """
    def _situ(gate, up):
        situ_a = beta * torch.tanh(gate / beta) * torch.sigmoid(gate)
        if linear_beta is not None:
            up = linear_beta * torch.tanh(up / linear_beta)
        return (situ_a * up).to(gate.dtype)
    _situ.__name__ = f"situ_beta{beta:g}" + (
        f"_lin{linear_beta:g}" if linear_beta is not None else "")
    return _situ


# Kimi-K3 as released: activation_situ_beta 4.0, activation_situ_linear_beta 25.0
# (config.json text_config). Registered under the bare name so `apply_glu(x,
# "situ")` matches the shipped checkpoint; re-register for any other values.
SITU_K3_BETA, SITU_K3_LINEAR_BETA = 4.0, 25.0
_situ_k3 = make_situ(SITU_K3_BETA, SITU_K3_LINEAR_BETA)

# name -> callable(gate, up) -> hidden.
GLU_VARIANTS = {
    "swiglu": _swiglu,          # K2 / DeepSeek-V3 — VERIFIED
    "silu": _swiglu,            # alias
    "situ": _situ_k3,           # K3 — SOURCED from the release's own modeling code
    "situ_unverified": SITU_UNVERIFIED,   # the guard, kept for future unknowns
}


def apply_glu(gate_up: torch.Tensor, variant: str = "swiglu") -> torch.Tensor:
    """Clean-split GLU by registry name. `'swiglu'` (K2/DeepSeek) and `'situ'`
    (K3, transcribed from the release's own modeling code) are both live;
    `'situ_unverified'` remains the guard for a future unsourced epilogue. The
    split is always clean-concat ([gate; up]), never interleaved."""
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
                    scale_suffix=_UNSET,
                    prefix: str = "model",
                    weight_map: Optional[dict] = None,
                    snapshot: Optional[str] = None,
                    scheme=None) -> dict:
    """Per-expert-tensor sha256 of the file data-section byte ranges (the
    release bytes). Reuses `mxfp4_loader.file_tensor_sha256`.

    Sharded checkpoints (gpt-oss, K2/K3) keep each tensor in its own shard:
    pass the index's ``weight_map`` (+ ``snapshot`` dir) so every name hashes
    against its own file. ``path`` alone serves single-file checkpoints."""
    from mxfp4_loader import file_tensor_sha256
    sc = resolve_scheme(scheme)
    if scale_suffix is _UNSET:
        scale_suffix = sc.default_scale_suffix
    kinds = ("weight",) + ((scale_suffix,) if scale_suffix else ())
    base = snapshot if snapshot is not None else os.path.dirname(path)
    table = {}
    for e in range(n_experts):
        for proj in PROJ:
            for kind in kinds:
                name = expert_tensor_name(layer, e, proj, kind, prefix, sc)
                if weight_map is not None:
                    if name not in weight_map:
                        raise KeyError(f"{name} not in the checkpoint index")
                    tpath = os.path.join(base, weight_map[name])
                else:
                    tpath = path
                table[name] = file_tensor_sha256(tpath, name)
    return table


def _t_sha(t: torch.Tensor) -> str:
    return hashlib.sha256(
        t.detach().contiguous().view(torch.uint8).numpy().tobytes()).hexdigest()


def verify_gather_provenance(get_tensor: Callable[[str], torch.Tensor],
                             file_hashes: dict, layer: int, n_experts: int, *,
                             scale_suffix=_UNSET,
                             prefix: str = "model", scheme=None) -> dict:
    """Assert every per-expert tensor the gather READS is byte-identical to its
    release file range (`file_hashes`). This is the gathered-layout provenance
    receipt: the fused arena is built from exactly these verified bytes, so
    each arena slice inherits the identity. Raises on any mismatch."""
    report = {}
    sc = resolve_scheme(scheme)
    if scale_suffix is _UNSET:
        scale_suffix = sc.default_scale_suffix
    kinds = ("weight",) + ((scale_suffix,) if scale_suffix else ())
    for e in range(n_experts):
        for proj in PROJ:
            for kind in kinds:
                name = expert_tensor_name(layer, e, proj, kind, prefix, sc)
                got = _t_sha(get_tensor(name))
                want = file_hashes[name]
                report[name] = want == got
                if want != got:
                    raise ValueError(
                        f"PROVENANCE FAIL {name}: file {want[:16]} != loaded {got[:16]}")
    return report
