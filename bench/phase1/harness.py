#!/usr/bin/env python3
"""Phase-1 baseline harness: census shapes x regimes x backends, with J/token receipts.

Measures the baselines the fused kernel must beat (gemm_predictions.json):
the dequant->bf16 grouped path (the e4b product path), bnb gemv_4bit at bs1
(the existing NF4-aware reference), and — import-guarded, recorded as skipped
when absent — Unsloth MoE backends and Marlin. The Phase-2 kernel drops into
the same registry, so its receipts land in the same JSON schema the thresholds
were registered against.

Fidelity per TOLERANCE_CONTRACT.md: the fp64 reference is exact math on the
SAME dequantized values (A_fp64 @ dequant_fp64(W).T), so per-cell error
measures GEMM reduction/rounding — not quantization loss, which every path
shares. The dequant path's B-rel per cell is the comparator the fused kernel's
2x bound is registered against.

Energy: NVML power sampling (pynvml if present, nvidia-smi polling otherwise)
over a >=1 s timed window; J/token = mean watts x window / tokens. Receipts
carry the sampling method and rate — a 50 Hz poll cannot resolve per-launch
spikes, only sustained draw, which is what the J/token claim is about.
"""

from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[2]
BLOCKSIZE = 64  # quantize_moe_experts default; KERNEL_CONTRACT convention pin


# ---------------------------------------------------------------- fixtures
@dataclass
class GemmSpec:
    model: str
    proj: str  # gate_up | down
    N: int
    K: int
    E: int
    top_k: int


def census_specs(census_path: Path, models: list[str] | None) -> list[GemmSpec]:
    d = json.loads(census_path.read_text())
    specs = []
    for m in d["models"]:
        if models and not any(s in m["model"] for s in models):
            continue
        for proj, nk in m["per_expert_gemms"].items():
            specs.append(
                GemmSpec(m["model"], proj, nk["N"], nk["K"], m["experts"], m["top_k"])
            )
    return specs


class QuantStack:
    """One fused expert stack, quantized per expert exactly as quantize_moe_experts
    does (per-expert quantize_4bit over the [N,K] slice, canonical #1949 layout).

    The fp64 reference is computed per expert on demand (chunked) rather than held
    resident — a stacked w_ref64 is ~17 GB fp64 on GPT-OSS-120B and blocked the
    big-census cells on 24 GB cards. Marlin state (GPTQ repack + its own dequant
    reference) is built lazily on first use so the vLLM dependency stays optional."""

    def __init__(self, spec: GemmSpec, device: str, seed: int = 42):
        self.spec = spec
        self.device = device
        g = torch.Generator(device="cpu").manual_seed(seed)
        from bitsandbytes import functional as F

        # Per-expert generation: one [N,K] fp32 draw at a time (~120 MB peak
        # host) instead of a monolithic [E,N,K] — DeepSeek-V3-class stacks
        # (E=256, 4096x7168) need ~30 GB host RAM the old way, which the
        # OOM-killer answered with SIGKILL on 24-GB pod containers. Sequential
        # draws consume the same generator stream (N*K is even for every
        # census/held-out shape, so the normal-pair stream stays aligned).
        self.packed, self.states = [], []
        for _ in range(spec.E):
            w_e = torch.randn(spec.N, spec.K, generator=g, dtype=torch.float32)
            w_e = (w_e * 0.02).to(device=device, dtype=torch.bfloat16)
            q, st = F.quantize_4bit(w_e, blocksize=BLOCKSIZE, quant_type="nf4")
            self.packed.append(q)
            self.states.append(st)
            del w_e
        self._marlin = None

    def dequant_bf16(self, e: int) -> torch.Tensor:
        from bitsandbytes import functional as F

        return F.dequantize_4bit(self.packed[e], self.states[e])

    def ref64(self, e: int) -> torch.Tensor:
        """fp64 of the exact decode values (fp32 LUT x absmax — the contract's
        reference, TOLERANCE_CONTRACT.md), computed on demand. NOT the
        bf16-materialized copy: a bf16 reference makes the materialize-then-GEMM
        path's rounding invisible (its own error becomes the ground truth) and
        mis-scores fp32-input kernels. This aligns the harness fidelity with the
        property suite's; the dequant path's b_rel rises accordingly — that is
        its real error, now visible."""
        import sys

        sys.path.insert(0, str(REPO / "kernel"))
        from nf4_grouped import dequant_ref

        B, A = self.fusedpack()
        return dequant_ref(B[e], A[e], self.spec.N, self.spec.K).to(torch.float64)

    def fusedpack(self):
        """Lazy expert-major repack for the Phase-2 fused kernel (same NF4 bytes;
        [E,N,K//2] + fp32 absmax [E,N,K//64] per KERNEL_CONTRACT)."""
        if getattr(self, "_fusedpack", None) is None:
            import sys

            sys.path.insert(0, str(REPO / "kernel"))
            from nf4_grouped import repack_from_bnb

            self._fusedpack = repack_from_bnb(
                self.packed, self.states, self.spec.N, self.spec.K
            )
        return self._fusedpack

    def marlin(self):
        """Lazy GPTQ-int4 repack of the SAME bf16 source values via vLLM's marlin
        utilities. Marlin quantizes to a different format, so its fidelity reference
        is its OWN dequantized values (w_ref), not the NF4 ones — recorded per cell."""
        if self._marlin is None:
            from vllm.model_executor.layers.quantization.utils import marlin_utils as mu
            from vllm.model_executor.layers.quantization.utils.marlin_utils_test import (
                marlin_quantize,
            )
            from vllm.scalar_type import scalar_types

            qs, refs = [], []
            for e in range(self.spec.E):
                w = self.dequant_bf16(e).t().contiguous().to(torch.float16)  # [K,N]
                w_ref, q_w, s, g_idx, sort_idx, _ = marlin_quantize(
                    w, scalar_types.uint4b8, group_size=128, act_order=False
                )
                qs.append((q_w, s, g_idx, sort_idx))
                refs.append(w_ref)  # [K,N] fp16, marlin's own dequant
            dev = torch.device(self.device)
            self._marlin = {
                "q": qs,
                "ref": refs,
                "qtype": scalar_types.uint4b8,
                # setup state real deployments cache: workspace + the empty zp of the
                # symmetric-gptq path (vllm 0.25 surface: marlin_make_workspace_new)
                "ws": mu.marlin_make_workspace_new(dev),
                "zp": mu.marlin_make_empty_g_idx(dev),
            }
        return self._marlin


def make_activations(
    spec: GemmSpec, regime: str, device: str, seed: int = 7, routing=None, layer=None
):
    """Per-regime grouped problem: list of (expert_id, A[M,K] bf16).

    ``prefill_measured`` replays a MEASURED group-size vector (per-expert token
    counts) instead of the uniform 2048*k/E — so empty/cold experts and hot
    experts appear as they really do. Empty groups are dropped (a grouped GEMM
    never launches a 0-row tile). ``layer`` selects the histogram layer: None =
    the representative (median-occupancy) layer, an int = that layer index."""
    g = torch.Generator(device="cpu").manual_seed(seed)

    def act(m):
        return (torch.randn(m, spec.K, generator=g, dtype=torch.float32) * 0.5).to(
            device=device, dtype=torch.bfloat16
        )

    if regime == "decode_bs1":
        experts = list(range(spec.top_k))  # k experts, one token each
        return [(e, act(1)) for e in experts]
    if regime.startswith("decode_m"):
        # batched decode: k active experts x M tokens each — the continuous-
        # batching band between bs1 (M=1, gemv path) and full prefill. The
        # kernel dispatches to the M-tile path the moment M > 1.
        m = int(regime[len("decode_m"):])
        experts = list(range(spec.top_k))
        return [(e, act(m)) for e in experts]
    if regime == "prefill_s2048":
        m = max(1, round(2048 * spec.top_k / spec.E))  # uniform routing, census note
        return [(e, act(m)) for e in range(spec.E)]
    if regime.startswith("tokbudget_"):
        # A step's TOKEN budget, uniformly routed: T tokens x top_k experts
        # spread over E, so the cell holds ~T*top_k rows. `tokbudget_2048` is
        # `prefill_s2048` by construction and is kept distinct only so an
        # M-axis sweep reads as one family. The axis exists because the
        # dequant-on-forward pattern pays a per-step tax that is roughly
        # independent of T (nf4moe's writeup measures ~2.5 s/step at 743B),
        # so any comparison against it is a function of T and must state T.
        m = max(1, round(int(regime[len("tokbudget_"):]) * spec.top_k / spec.E))
        return [(e, act(m)) for e in range(spec.E)]
    if regime == "prefill_measured":
        if routing is None:
            raise RuntimeError("prefill_measured needs --routing <histogram.json>")
        counts = (
            routing["representative_counts"]
            if layer is None
            else routing["per_layer_counts"][layer]
        )
        if len(counts) != spec.E:
            raise RuntimeError(f"routing E={len(counts)} != census E={spec.E}")
        return [(e, act(int(c))) for e, c in enumerate(counts) if c > 0]
    raise ValueError(regime)


# ---------------------------------------------------------------- backends
def bk_dequant_grouped(stack: QuantStack, groups):
    """The e4b product path: dequantize the active experts to bf16 in global
    memory, then per-expert bf16 mm (the sparse loop the integration runs)."""
    outs = []
    for e, a in groups:
        w = stack.dequant_bf16(e)
        outs.append(a @ w.t())
    return outs


def bk_gemv4bit(stack: QuantStack, groups):
    """bnb's NF4-aware gemv at M=1 — dequantizes inside the kernel; the closest
    existing point to the fused claim. bs1 only (gemv semantics).

    Strongest-self treatment, symmetric with the fused op's cached assembly:
    the per-call ``.t()`` views (pure metadata, but python-per-expert-per-call)
    are built once and cached, so the baseline pays only its genuine kernel
    launches inside the timed region."""
    from bitsandbytes import functional as F

    if getattr(stack, "_gemv_t", None) is None:
        stack._gemv_t = [p.t() for p in stack.packed]
    outs = []
    for e, a in groups:
        if a.shape[0] != 1:
            raise RuntimeError("gemv_4bit is M=1 only")
        outs.append(F.gemv_4bit(a, stack._gemv_t[e], state=stack.states[e]))
    return outs


def _grouped_inputs(stack, groups):
    """Concatenated-token form shared by the grouped backends: a_cat [T,K],
    dequantized b [G,K,N] bf16, group sizes."""
    a_cat = torch.cat([a for _, a in groups])
    b = torch.stack([stack.dequant_bf16(e).t().contiguous() for e, _ in groups])
    sizes = [a.shape[0] for _, a in groups]
    return a_cat, b, sizes


def _split(out_cat, sizes):
    outs, i = [], 0
    for m in sizes:
        outs.append(out_cat[i : i + m])
        i += m
    return outs


def bk_dequant_grouped_mm(stack: QuantStack, groups):
    """Dequant + ONE native grouped GEMM (torch._grouped_mm) — the execution class
    unsloth's grouped_mm backend rides. Alignment rejections (jagged M=1 groups on
    some torch versions) surface as skips, which is itself a measured fact."""
    if not hasattr(torch, "_grouped_mm"):
        raise ImportError(f"torch {torch.__version__} has no _grouped_mm")
    a_cat, b, sizes = _grouped_inputs(stack, groups)
    offs = torch.cumsum(
        torch.tensor(sizes, device=a_cat.device, dtype=torch.int32),
        0,
        dtype=torch.int32,
    )
    out = torch._grouped_mm(a_cat, b, offs=offs)
    IMPL_NOTE["dequant_grouped_mm"] = "torch._grouped_mm"
    return _split(out, sizes)


def bk_unsloth(stack, groups):  # pragma: no cover - optional dependency
    """Dequant + a grouped-GEMM of the CLASS unsloth's MoE backend rides.

    ⚠️ This backend has never executed unsloth's own kernel, and cannot. Two
    independent reasons, either one sufficient — both verified on real silicon
    by ``validate_unsloth_native.py`` G1, not argued from source:

      1. It tries tgale96's ``grouped_gemm.ops.gmm`` FIRST and returns on
         success. Wherever that package is installed — as it was in the env
         that banked the v6 comparator receipts — the unsloth probes below are
         never reached, and ``IMPL_NOTE`` records ``grouped_gemm.ops.gmm``.
      2. Where it IS reached, ``getattr(unsloth.kernels.moe, "grouped_gemm")``
         yields the SUBMODULE, not a callable, so the call raises
         ``TypeError: 'module' object is not callable``. Observed verbatim.

    (An earlier note here blamed the empty ``unsloth/kernels/moe/__init__.py``
    for making that getattr return None. The file really is 0 bytes, but the
    inference was wrong: python binds a submodule onto its parent package on
    import regardless of __init__ contents. Conclusion unchanged, mechanism
    corrected in place.)

    So this arm is a fair proxy for the execution CLASS, and was always
    reported as one — but it is not a head-to-head against unsloth.

    For unsloth's actual kernel use ``unsloth_native`` / ``unsloth_native_bf16``
    below. Kept unchanged so the v6 comparator receipts stay reproducible."""
    import importlib

    a_cat, b, sizes = _grouped_inputs(stack, groups)
    probes = []
    # the standalone grouped_gemm package (tgale96) — unsloth's non-native path
    if importlib.util.find_spec("grouped_gemm") is not None:
        try:
            from grouped_gemm import ops as gg_ops

            batch_sizes = torch.tensor(sizes, dtype=torch.int64)  # cpu by API
            out = gg_ops.gmm(a_cat, b, batch_sizes, trans_b=False)
            IMPL_NOTE["unsloth"] = "grouped_gemm.ops.gmm"
            return _split(out, sizes)
        except Exception as e:
            probes.append(f"grouped_gemm.ops.gmm: {type(e).__name__}: {e}")
    for mod, attrs in (
        ("unsloth_zoo.moe_utils", ("grouped_gemm", "gmm")),
        ("unsloth.kernels.moe", ("grouped_gemm", "gmm")),
    ):
        if importlib.util.find_spec(mod.split(".")[0]) is None:
            probes.append(f"{mod}: package absent")
            continue
        try:
            m = importlib.import_module(mod)
        except Exception as e:
            probes.append(f"{mod}: import {type(e).__name__}")
            continue
        for attr in attrs:
            fn = getattr(m, attr, None)
            if fn is None:
                continue
            try:
                batch_sizes = torch.tensor(sizes, dtype=torch.int64)
                out = fn(a_cat, b, batch_sizes)
                IMPL_NOTE["unsloth"] = f"{mod}.{attr}"
                return _split(out, sizes)
            except Exception as e:
                probes.append(f"{mod}.{attr}: {type(e).__name__}: {e}")
    raise ImportError(
        "no unsloth grouped path ran; probed: " + " | ".join(probes)
        if probes
        else "unsloth/grouped_gemm not installed"
    )


_UNSLOTH_NATIVE_ENTRY = "unsloth.kernels.moe.grouped_gemm.interface.grouped_gemm"


def _unsloth_native_fn():
    """Unsloth's OWN Triton grouped GEMM (fwd + dX/dW autograd Function).

    Imported by explicit module path, not getattr off the package: both
    ``unsloth/kernels/moe/__init__.py`` and ``.../grouped_gemm/__init__.py`` are
    empty upstream, which is exactly what made the older ``unsloth`` backend's
    probe dead."""
    import importlib

    if importlib.util.find_spec("unsloth") is None:
        raise ImportError("unsloth not installed")
    mod = importlib.import_module("unsloth.kernels.moe.grouped_gemm.interface")
    return mod.grouped_gemm, mod


def _unsloth_native_call(a_cat, w_stack, sizes):
    """One call into unsloth's kernel.

    ``w_stack`` is [G, N, K] — unsloth's documented expert-stack layout, which
    is also the layout gnf4's fused stack already holds, so NEITHER side pays a
    transpose to meet the other. (``_grouped_inputs`` builds [G, K, N] for the
    torch/tgale96 grouped backends; that transpose is their API's requirement,
    not unsloth's.)

    Two upstream contract details, both asserted/executed unconditionally in
    ``interface.py`` and neither documented in the signature:
      * ``m_sizes`` must be a CUDA tensor (``assert m_sizes.device.type ==
        "cuda"``), not the CPU tensor tgale96's ``gmm`` takes.
      * ``gather_indices`` is ``.view(-1)``'d unconditionally, so it must be a
        real tensor even when both permutes are off, where it is otherwise
        unused. Passing the documented default of None is an AttributeError.

    ``autotune=True`` hands unsloth's OWN autotuner the choice of config per
    shape — deliberately their strongest setting, against gnf4's shipped
    default. It also makes ``kernel_config_fwd`` unnecessary, which the
    non-autotune path asserts on. ``topk`` is inert here: with both permutes
    off, upstream sets ``total_tokens = X.shape[0]`` and uses topk only for a
    ``num_tokens`` it does not consult."""
    gg, _ = _unsloth_native_fn()
    m_sizes = torch.tensor(sizes, device=a_cat.device, dtype=torch.int32)
    gather_indices = torch.arange(
        a_cat.shape[0], device=a_cat.device, dtype=torch.int32
    )
    return gg(
        X=a_cat,
        W=w_stack,
        m_sizes=m_sizes,
        topk=1,
        gather_indices=gather_indices,
        permute_x=False,
        permute_y=False,
        autotune=True,
    )


def bk_unsloth_native(stack, groups):  # pragma: no cover - optional dependency
    """Unsloth's own kernel in the 4-BIT-STORAGE regime — the apples-to-apples
    cell against the fused kernel.

    The dequant sits INSIDE the timed region because 4-bit storage genuinely
    requires it once per forward per layer: unsloth's GEMM consumes bf16
    weights, so a 4-bit checkpoint must be materialized before it can run. That
    is a real cost of serving this regime with a bf16 kernel, not a handicap
    imposed by the harness — and it is the same cost ``bk_dequant_grouped`` and
    ``bk_unsloth`` already pay."""
    a_cat = torch.cat([a for _, a in groups])
    w_stack = torch.stack([stack.dequant_bf16(e) for e, _ in groups])
    sizes = [a.shape[0] for _, a in groups]
    out = _unsloth_native_call(a_cat, w_stack, sizes)
    IMPL_NOTE["unsloth_native"] = f"{_UNSLOTH_NATIVE_ENTRY} (4-bit storage, dequant timed in)"
    return _split(out, sizes)


def bk_unsloth_native_bf16(stack, groups):  # pragma: no cover - optional dependency
    """Unsloth's own kernel in ITS OWN regime: bf16-RESIDENT weights, nothing to
    dequantize, nothing in the timed path but their kernel.

    gnf4 does not compete here at all — with no 4-bit storage there is no
    round-trip to skip, and this is the job unsloth's kernel is built for. The
    arm exists so the 4-bit cell above cannot be misread as a claim about the
    regime they actually target: this is their CEILING, and it belongs in the
    same table.

    Strongest-self treatment, symmetric with the fused op's cached assembly
    (see ``bk_fused_nf4``): the resident stack is materialized once per groups
    object and cached, so the arm is charged only its genuine kernel launch."""
    cache = getattr(stack, "_unsloth_resident", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        w_stack = torch.stack([stack.dequant_bf16(e) for e, _ in groups])
        sizes = [a.shape[0] for _, a in groups]
        stack._unsloth_resident = (id(groups), a_cat, w_stack, sizes)
    _, a_cat, w_stack, sizes = stack._unsloth_resident
    out = _unsloth_native_call(a_cat, w_stack, sizes)
    IMPL_NOTE["unsloth_native_bf16"] = f"{_UNSLOTH_NATIVE_ENTRY} (bf16-resident, cached stack)"
    return _split(out, sizes)


def unsloth_native_fingerprint():
    """Provenance for the receipt: which unsloth is installed, and whether its
    TMA path is actually live on this device.

    TMA gates on ``torch.cuda.get_device_capability()[0] >= 9`` AND a triton
    carrying the TMA API. On sm_86/sm_89 it is OFF, so a comparison run only on
    Ampere/Ada would benchmark unsloth's kernel with its fast path compiled
    out. Recorded per-run so no reader has to take that on trust."""
    import importlib

    _, mod = _unsloth_native_fn()
    unsloth = importlib.import_module("unsloth")
    return {
        "entry": _UNSLOTH_NATIVE_ENTRY,
        "unsloth_version": getattr(unsloth, "__version__", "unknown"),
        "supports_tma": bool(mod.supports_tma()),
        "device_capability": list(torch.cuda.get_device_capability()),
    }


def bk_marlin(stack, groups):  # pragma: no cover - optional dependency
    """vLLM's GPTQ-Marlin W4A16 GEMM per active expert — the best existing
    4-bit-in-kernel comparator. fp16 activations (marlin's dtype); repack is
    lazy + cached on the stack; fidelity for these cells uses marlin's OWN
    dequant reference (different quant format than NF4)."""
    import importlib

    if importlib.util.find_spec("vllm") is None:
        raise ImportError("vllm (marlin) not installed")
    from vllm.model_executor.layers.quantization.utils import marlin_utils as mu

    m = stack.marlin()
    outs = []
    for e, a in groups:
        q_w, s, g_idx, sort_idx = m["q"][e]
        a16 = a.to(torch.float16)
        out = mu.apply_gptq_marlin_linear(
            a16,
            q_w,
            s,
            m["zp"],
            g_idx,
            sort_idx,
            m["ws"],
            m["qtype"],
            stack.spec.N,
            stack.spec.K,
            is_k_full=True,
        )
        outs.append(out)
    IMPL_NOTE["marlin"] = (
        "vllm mu.apply_gptq_marlin_linear (fp16, group=128, fp32_reduce)"
    )
    return outs


def bk_fused_nf4(stack: QuantStack, groups):
    """The Phase-2 kernel: ONE launch, dequant inside the mainloop, no bf16
    weight materialization.

    The contract's op boundary takes PRE-ASSEMBLED inputs (A_cat [T,K] +
    group sizes + a device expert_ids tensor) — sort/concat live upstream,
    like every grouped GEMM. The harness fixture hands each backend a LIST of
    per-expert tensors, which the loop backends (dequant/gemv) consume
    natively but the fused op must first assemble; doing that per timed call
    charged the kernel ~0.07–0.09 ms of pure fixture conversion (torch.cat of
    k tiny tensors + an eids H2D) that no real integration pays per step
    (post-router tokens are already one contiguous buffer). Assembly is
    therefore cached per groups object; the kernel launch + its own
    descriptor build remain inside the timed region."""
    import sys

    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    out = gemm_4bit_grouped(a_cat, B, A, sizes, ids)
    IMPL_NOTE["fused_nf4"] = (
        "gemm_4bit_grouped (triton, single launch; op-boundary inputs pre-assembled)"
    )
    return _split(out, sizes)


def bk_fused_nf4_v1cfg(stack: QuantStack, groups):
    """Ablation backend: the fused kernel FORCED onto the retired v1 decode
    config (the Gate-2 census dict: 256/8 on its three keyed shapes, 128/4
    default). Exists so a confirmatory can state the new-config-vs-old-config
    claim PAIRED — same process, same stack, same thermal context — which the
    v1 blind confirmatory showed is the only instance-robust comparison."""
    import sys

    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    v1 = {(1536, 2048): (256, 8), (1408, 2816): (256, 8), (2816, 704): (256, 8)}
    out = gemm_4bit_grouped(
        a_cat, B, A, sizes, ids,
        decode_config=v1.get((stack.spec.N, stack.spec.K), (128, 4)),
        split_k=1,  # the retired config predates split-K; keep it faithful
    )
    IMPL_NOTE["fused_nf4_v1cfg"] = "gemm_4bit_grouped @ retired v1 census-dict config"
    return _split(out, sizes)


def bk_fused_nf4_v2cfg(stack: QuantStack, groups):
    """Ablation backend: the exact v2 semantics — constant (64, 2), no
    split-K — the paired comparator for the v3 SM-conditional-constant claim
    (the A2000 cells where v2 measured consistent paired losses)."""
    import sys

    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    out = gemm_4bit_grouped(
        a_cat, B, A, sizes, ids, decode_config=(64, 2), split_k=1
    )
    IMPL_NOTE["fused_nf4_v2cfg"] = "gemm_4bit_grouped @ v2 constant (64/2), no split"
    return _split(out, sizes)


def bk_fused_nf4_nosplit(stack: QuantStack, groups):
    """Ablation backend: the CURRENT plan's config with split-K forced OFF —
    the paired comparator for the v3 split-K claim on starved cells (same
    process, same stack, same thermal context)."""
    import sys

    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    out = gemm_4bit_grouped(a_cat, B, A, sizes, ids, split_k=1)
    IMPL_NOTE["fused_nf4_nosplit"] = "gemm_4bit_grouped, split_k forced 1"
    return _split(out, sizes)


def bk_fused_routed(stack: QuantStack, groups):
    """The PRODUCT path: decode_dispatch chooses dequant-vs-fused ONCE PER
    STACK and the branch is cached — MoE expert shapes are static at model
    load, so a real integration never pays the dispatch per call. (v4 timed
    a per-call implementation of this backend and its ~40-100us python floor
    correctly failed the registered identity bands — the measurement-boundary
    lesson now encoded here.) At prefill (any group > 1 token) this is just
    the fused kernel."""
    sizes = [a.shape[0] for _, a in groups]
    if max(sizes) == 1:
        route = getattr(stack, "_decode_route", None)
        if route is None:
            import sys

            sys.path.insert(0, str(REPO / "kernel"))
            from nf4_grouped import decode_dispatch

            sm = torch.cuda.get_device_properties(0).multi_processor_count
            route = decode_dispatch(stack.spec.N, stack.spec.K, len(groups), sm)
            stack._decode_route = route
        if route[0] == "dequant":
            out = bk_dequant_grouped(stack, groups)
            IMPL_NOTE["fused_routed"] = "below-floor: dequant path (load-time dispatch)"
            return out
    out = bk_fused_nf4(stack, groups)
    IMPL_NOTE["fused_routed"] = "fused kernel (load-time dispatch: eligible)"
    return out


def bk_fused_nf4_v3prefill(stack: QuantStack, groups):
    """Ablation backend: the M-tile path forced onto the RETIRED pre-v4
    prefill config (block_m 16/64 by group size, BLOCK_N=64, w4/s3) — the
    paired comparator for the v4 prefill-config claim."""
    import sys

    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    out = gemm_4bit_grouped(
        a_cat, B, A, sizes, ids,
        block_m=16 if max(sizes) <= 16 else 64,
        prefill_config=(64, 4, 3),
    )
    IMPL_NOTE["fused_nf4_v3prefill"] = "M-tile @ retired pre-v4 config (64-row/BN64/w4/s3)"
    return _split(out, sizes)


IMPL_NOTE: dict = {}

def bk_fused_v5loop(stack: QuantStack, groups):
    """Ablation backend for the v6 confirmatory: the fused kernel FORCED onto
    the v5 M-tile mainloop (prefill_variant=0, per-element L1 codebook gather)
    so the register-LUT rewrite can be adjudicated as a same-instance paired
    ratio — instance-robust, unlike dequant-relative ratios (the dequant
    baseline swung ~25% between two A5000 hosts in the v6 exploratory while
    the fused kernel held within 0.2 ms). Same assembly caching as
    bk_fused_nf4; decode cells take the identical decode path (variant is a
    prefill-only knob), so this backend is only meaningful at prefill."""
    import sys
    sys.path.insert(0, str(REPO / "kernel"))
    from nf4_grouped import gemm_4bit_grouped

    B, A = stack.fusedpack()
    cache = getattr(stack, "_fused_asm", None)
    if cache is None or cache[0] != id(groups):
        a_cat = torch.cat([a for _, a in groups])
        sizes = [a.shape[0] for _, a in groups]
        ids = torch.tensor(
            [e for e, _ in groups], dtype=torch.int32, device=a_cat.device
        )
        stack._fused_asm = (id(groups), a_cat, sizes, ids)
    _, a_cat, sizes, ids = stack._fused_asm
    out = gemm_4bit_grouped(a_cat, B, A, sizes, ids, prefill_variant=0)
    IMPL_NOTE["fused_v5loop"] = (
        "gemm_4bit_grouped with prefill_variant=0 (v5 M-tile mainloop forced)"
    )
    return _split(out, sizes)


BACKENDS = {
    "dequant_grouped": bk_dequant_grouped,
    "gemv_4bit": bk_gemv4bit,
    "dequant_grouped_mm": bk_dequant_grouped_mm,
    "unsloth": bk_unsloth,
    "unsloth_native": bk_unsloth_native,
    "unsloth_native_bf16": bk_unsloth_native_bf16,
    # Q2's self-pair validator: the SAME function under a second key, so the
    # backend loop times the fused kernel twice within one cell. Its ratio
    # against `fused_nf4` must read ~1.000x; anything outside [0.97, 1.03] means
    # the box drifted mid-cell and every ratio measured beside it is noise. An
    # unpaired sweep on a shared box once reported the default config as 1.283x
    # faster than itself, which is what this key exists to catch.
    "fused_nf4_selfpair": bk_fused_nf4,
    "marlin": bk_marlin,
    "fused_nf4": bk_fused_nf4,
    "fused_nf4_v1cfg": bk_fused_nf4_v1cfg,
    "fused_nf4_v2cfg": bk_fused_nf4_v2cfg,
    "fused_nf4_nosplit": bk_fused_nf4_nosplit,
    "fused_routed": bk_fused_routed,
    "fused_nf4_v3prefill": bk_fused_nf4_v3prefill,
    "fused_v5loop": bk_fused_v5loop,
}


# ---------------------------------------------------------------- measurement
class PowerSampler:
    """Mean GPU watts over start()..stop(). pynvml preferred; nvidia-smi poll
    fallback. Records its own method + achieved rate into the receipt."""

    def __init__(self, device_index: int = 0):
        self.samples: list[float] = []
        self._stop = threading.Event()
        self.method = "none"
        try:
            import pynvml

            pynvml.nvmlInit()
            self._h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
            self._read = lambda: pynvml.nvmlDeviceGetPowerUsage(self._h) / 1000.0
            self.method = "pynvml"
        except Exception:
            self._read = lambda: float(
                subprocess.run(
                    [
                        "nvidia-smi",
                        "--query-gpu=power.draw",
                        "--format=csv,noheader,nounits",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=2,
                )
                .stdout.strip()
                .splitlines()[0]
            )
            self.method = "nvidia-smi"

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except Exception:
                pass
            time.sleep(0.02)

    def start(self):
        self.samples = []
        self._stop.clear()
        self._t = threading.Thread(target=self._run, daemon=True)
        self._t.start()

    def stop(self) -> tuple[float | None, int]:
        self._stop.set()
        self._t.join(timeout=2)
        return (
            statistics.mean(self.samples) if self.samples else None,
            len(self.samples),
        )


def time_backend(fn, stack, groups, iters: int, device: str):
    for _ in range(min(10, iters)):  # warmup
        fn(stack, groups)
    torch.cuda.synchronize()
    ev0, ev1 = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    times = []
    for _ in range(iters):
        ev0.record()
        fn(stack, groups)
        ev1.record()
        torch.cuda.synchronize()
        times.append(ev0.elapsed_time(ev1))
    return statistics.median(times)


def energy_window(fn, stack, groups, device: str, min_s: float = 1.2):
    """Repeat the call for >= min_s under the power sampler; J/call from mean W."""
    sampler = PowerSampler()
    torch.cuda.synchronize()
    sampler.start()
    t0 = time.monotonic()
    calls = 0
    while time.monotonic() - t0 < min_s:
        fn(stack, groups)
        calls += 1
    torch.cuda.synchronize()
    wall = time.monotonic() - t0
    watts, n = sampler.stop()
    if watts is None or calls == 0:
        return None, None, sampler.method, n
    return watts, watts * wall / calls, sampler.method, n


def fidelity(stack: QuantStack, groups, outs, ref: str = "nf4") -> float:
    """Relative Frobenius error vs the fp64 exact GEMM on identical dequantized
    values, computed per expert on demand (no resident reference stack). Marlin
    cells use marlin's OWN dequant (ref="marlin") — a different quant format,
    so comparing it to the NF4 values would measure format distance, not GEMM
    arithmetic."""
    num = den = 0.0
    for (e, a), out in zip(groups, outs):
        if ref == "marlin":
            w64 = stack.marlin()["ref"][e].to(torch.float64)  # [K,N]
            r = a.to(torch.float64) @ w64
        else:
            r = a.to(torch.float64) @ stack.ref64(e).t()
        num += (out.to(torch.float64) - r).norm().item() ** 2
        den += r.norm().item() ** 2
    return (num**0.5) / (den**0.5)


# ---------------------------------------------------------------- driver
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--models", nargs="*", default=None, help="substring filters over census models"
    )
    ap.add_argument("--regimes", nargs="*", default=["decode_bs1", "prefill_s2048"])
    ap.add_argument(
        "--backends",
        nargs="*",
        default=[
            "dequant_grouped",
            "gemv_4bit",
            "dequant_grouped_mm",
            "unsloth",
            "marlin",
        ],
    )
    ap.add_argument("--iters", type=int, default=100)
    ap.add_argument(
        "--paired-base",
        default=None,
        help="backend re-timed immediately before EVERY backend (including "
        "itself, giving an adjacent self-pair). Writes base_ms_paired + "
        "paired_ratio per cell. Roughly doubles timing cost; without it, "
        "ratios are not paired in any meaningful sense.",
    )
    ap.add_argument("--no-energy", action="store_true")
    ap.add_argument("--out", default=None)
    ap.add_argument(
        "--routing",
        nargs="*",
        default=None,
        help="routing_hist.py JSON(s); enables prefill_measured, matched to a spec by E+k",
    )
    ap.add_argument(
        "--extra-shapes",
        default=None,
        help="JSON file of additional GemmSpec dicts (held-out shapes)",
    )
    ap.add_argument(
        "--energy-window",
        type=float,
        default=1.2,
        help="power-sampling window seconds per cell",
    )
    ap.add_argument(
        "--routing-layer",
        default="rep",
        help="prefill_measured histogram layer: rep (median-occupancy) | all | <int>",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny E/N/K, iters=3 (still needs CUDA)"
    )
    args = ap.parse_args()

    assert torch.cuda.is_available(), "Phase-1 baselines are GPU measurements"
    device = "cuda"
    specs = census_specs(REPO / "census" / "shape_census.json", args.models)
    if args.extra_shapes:
        for s in json.loads(Path(args.extra_shapes).read_text()):
            specs.append(GemmSpec(**s))
    if args.smoke:
        specs = [GemmSpec("smoke", "gate_up", 256, 128, 8, 2)]
        args.iters = 3

    # routing histograms keyed by (E,k); a spec picks the one that matches its shape
    routings = {}
    for p in args.routing or []:
        r = json.loads(Path(p).read_text())
        routings[(r["E"], r["k"])] = r
    if routings and "prefill_measured" not in args.regimes:
        args.regimes = list(args.regimes) + ["prefill_measured"]

    def _driver_version():
        # portable across vendors: nvidia-smi on NVIDIA, rocm-smi on AMD, and
        # a missing tool is metadata-only — never fatal to the census.
        for cmd in (
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
            ["rocm-smi", "--showdriverversion", "--csv"],
        ):
            try:
                out = subprocess.run(
                    cmd, capture_output=True, text=True
                ).stdout.strip()
                if out:
                    return out.splitlines()[-1].strip()
            except FileNotFoundError:
                continue
        return ""

    env = {
        "gpu": torch.cuda.get_device_name(0),
        "capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
        "torch": torch.__version__,
        "driver": _driver_version(),
    }
    try:
        import bitsandbytes

        env["bitsandbytes"] = bitsandbytes.__version__
    except Exception as e:  # pragma: no cover
        env["bitsandbytes"] = f"unavailable: {e}"

    # Recorded unconditionally, not only when an unsloth backend is selected:
    # whether their TMA path was live is a property of the RUN, and a reader
    # comparing two receipts needs it present in both to see that it differed.
    try:
        env["unsloth_native"] = unsloth_native_fingerprint()
    except Exception as e:  # pragma: no cover - optional dependency
        env["unsloth_native"] = f"unavailable: {e}"

    cells = []
    for spec in specs:
        print(
            f"== {spec.model} {spec.proj} N={spec.N} K={spec.K} E={spec.E} k={spec.top_k}"
        )
        try:
            stack = QuantStack(spec, device)
        except Exception as e:  # stack won't fit this device: cells -> skipped,
            for regime in args.regimes:  # the rest of the run proceeds (NOT-RUN
                for name in args.backends:  # exclusion, never a dead process)
                    cells.append({
                        "model": spec.model, "proj": spec.proj, "regime": regime,
                        "backend": name,
                        **{k: getattr(spec, k) for k in ("N", "K", "E", "top_k")},
                        "status": "skipped",
                        "reason": f"stack build: {str(e)[:160]}",
                    })
            print(f"   stack build failed -> all cells skipped: {str(e)[:100]}")
            torch.cuda.empty_cache()
            continue
        routing = routings.get((spec.E, spec.top_k))
        # (regime, layer) work items; prefill_measured expands to per-layer when
        # --routing-layer all, else the representative (None) or a fixed int.
        variants = []
        for regime in args.regimes:
            if regime != "prefill_measured":
                variants.append((regime, None))
                continue
            if routing is None:
                continue  # no matching histogram for this shape; skip quietly
            if args.routing_layer == "all":
                variants += [
                    (regime, L) for L in range(len(routing["per_layer_counts"]))
                ]
            elif args.routing_layer == "rep":
                variants.append((regime, None))
            else:
                variants.append((regime, int(args.routing_layer)))
        for regime, layer in variants:
            groups = make_activations(
                spec, regime, device, routing=routing, layer=layer
            )
            tokens = sum(a.shape[0] for _, a in groups)
            for name in args.backends:
                fn = BACKENDS[name]
                cell = {
                    "model": spec.model,
                    "proj": spec.proj,
                    "regime": regime,
                    "backend": name,
                    **{k: getattr(spec, k) for k in ("N", "K", "E", "top_k")},
                    "tokens_per_call": tokens,
                    "n_groups": len(groups),
                }
                if regime == "prefill_measured":
                    L = routing["representative_layer"] if layer is None else layer
                    cell["routing_src"] = routing["model"]
                    cell["routing_layer"] = L
                    cell["routing_occupancy"] = routing["layer_summary"][L]["occupancy"]
                try:
                    if name == "gemv_4bit" and regime != "decode_bs1":
                        raise RuntimeError("gemv_4bit is bs1-only by definition")
                    outs = fn(stack, groups)
                    ref = "marlin" if name == "marlin" else "nf4"
                    cell["b_rel_vs_fp64"] = fidelity(stack, groups, outs, ref=ref)
                    cell["fidelity_ref"] = ref
                    if name in IMPL_NOTE:
                        cell["impl"] = IMPL_NOTE[name]
                    # TRUE PAIRING: re-time the base immediately before every
                    # comparator and take the ratio per pair, so box drift
                    # cancels inside each pair instead of accumulating across a
                    # cell. Timing all backends once and dividing against one
                    # shared base timing is NOT pairing — it silently charges
                    # every comparator whatever the box did since the base ran.
                    #
                    # Applied uniformly, including when `name` IS the base: that
                    # row becomes a genuine ADJACENT base-vs-base self-pair,
                    # which is what a [0.97,1.03] validity band assumes. The
                    # previous bracketing self-pair spanned a whole cell and
                    # systematically read ~4% on sub-0.2 ms shapes.
                    if args.paired_base:
                        cell["base_ms_paired"] = time_backend(
                            BACKENDS[args.paired_base], stack, groups,
                            args.iters, device,
                        )
                    cell["ms_median"] = time_backend(
                        fn, stack, groups, args.iters, device
                    )
                    if cell.get("base_ms_paired"):
                        # >1 == the base (fused) is faster than this backend.
                        cell["paired_ratio"] = (
                            cell["ms_median"] / cell["base_ms_paired"]
                        )
                    cell["tok_per_s"] = tokens / (cell["ms_median"] / 1e3)
                    if not args.no_energy:
                        watts, j_call, method, n = energy_window(
                            fn, stack, groups, device, min_s=args.energy_window
                        )
                        cell.update(
                            {
                                "watts_mean": watts,
                                "j_per_token": (j_call / tokens) if j_call else None,
                                "power_method": method,
                                "power_samples": n,
                            }
                        )
                    cell["status"] = "ok"
                    print(
                        f"   {regime:>14} {name:<16} {cell['ms_median']:8.3f} ms "
                        f"{cell['tok_per_s']:10.1f} tok/s  err {cell['b_rel_vs_fp64']:.2e}"
                    )
                except Exception as e:
                    cell.update({"status": "skipped", "reason": str(e)[:200]})
                    print(f"   {regime:>14} {name:<16} skipped: {str(e)[:80]}")
                cells.append(cell)
        del stack
        torch.cuda.empty_cache()

    out = {
        "phase": 1,
        "spec": "gemm_predictions.json",
        "env": env,
        "blocksize": BLOCKSIZE,
        "cells": cells,
    }
    path = Path(args.out or f"phase1_{env['gpu'].replace(' ', '_')}.json")
    path.write_text(json.dumps(out, indent=1))
    print(f"receipts -> {path} ({len(cells)} cells)")


if __name__ == "__main__":
    main()
