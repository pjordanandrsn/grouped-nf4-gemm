# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).

"""Does a real KV cache have exploitable low-rank structure? — held-out probe.

The synthetic tests in ``kernel/test_nf4_kv_lowrank.py`` prove the absorption
algebra is exact. They cannot prove the method is *useful*, because iid data has
no low-rank structure and data built inside a rank-r subspace has it by
construction. Only real weights on real text answer that.

Two disciplines make this a measurement rather than a demo:

1. **Held-out calibration.** The basis is fit on the first half of the tokens
   and evaluated on the second. A basis scored on the tokens it was fit to is
   an SVD reconstruction error, not a prediction, and will flatter any rank.
2. **An iid control** at the same shape. Without it, "rank 64 keeps 97%" is
   unreadable — you cannot tell structure from the fact that 64 of 128
   directions is simply a lot of directions.

Loading uses ``experts4bit-qlora``'s streaming loader rather than a plain
``from_pretrained``. That is not incidental: stock loading materializes the
checkpoint in bf16 before quantizing, so a 7B MoE needs ~14 GB transiently and
OOMs a 12 GB card that holds the quantized model with room to spare. The
streaming path quantizes fused experts on the way in, never materializing the
dense copy — the same reason the engine exists for serving.

Writes a JSON receipt. Run:  python bench/context/lowrank_probe.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "kernel"))

MODEL = os.environ.get("LOWRANK_PROBE_MODEL", "allenai/OLMoE-1B-7B-0924")
N_TOKENS = int(os.environ.get("LOWRANK_PROBE_TOKENS", "1024"))
RANKS = [int(r) for r in os.environ.get("LOWRANK_PROBE_RANKS", "64").split(",")]
# Full rank sweep for the dominance question: at what rank does low-rank reach
# NF4's fidelity, and what ratio is left there? Ranks that the 64-element
# blocksize cannot pack are measured anyway — otherwise a reader cannot tell
# whether the packer is the binding constraint or the data is.
SWEEP = [8, 16, 32, 48, 64, 80, 96, 112, 120, 124, 128]
# Offload the frozen 4-bit expert bases to pinned CPU RAM. The A2000 is shared
# with home-lab services, so the probe must not assume the whole card.
OFFLOAD = os.environ.get("LOWRANK_PROBE_OFFLOAD", "1") == "1"


def _unpack(ret):
    """The streaming loader returns (model, ...) across versions; take the model."""
    return (ret[0], ret[1:]) if isinstance(ret, tuple) else (ret, ())


def _heldout_err(x: torch.Tensor, rank: int, half: int) -> float:
    """Basis from the first `half` tokens, error measured on the rest."""
    fit, held = x[:half], x[half:]
    H, D = x.shape[1], x.shape[2]
    B = torch.empty(H, rank, D, device=x.device, dtype=torch.float32)
    for h in range(H):
        B[h] = torch.linalg.svd(fit[:, h, :].float(), full_matrices=False)[2][:rank]
    codes = torch.einsum("thd,hrd->thr", held.float(), B)
    back = torch.einsum("thr,hrd->thd", codes, B)
    return float((back - held.float()).norm() / held.float().norm())


def _energy_curve(x: torch.Tensor, ranks) -> dict:
    """Fraction of squared energy in the top-r directions, mean over heads."""
    out = {}
    T, H, D = x.shape
    sv = torch.stack([torch.linalg.svdvals(x[:, h, :].float()) for h in range(H)])
    e = sv ** 2
    tot = e.sum(dim=1)
    for r in ranks:
        out[r] = float((e[:, :r].sum(dim=1) / tot).mean())
    # rank needed for 90% energy, the shape-independent summary
    cum = torch.cumsum(e, dim=1) / tot[:, None]
    out["rank_for_90pct"] = float((cum < 0.90).sum(dim=1).float().mean() + 1)
    return out


def main() -> int:
    from experts4bit_qlora import load_moe_4bit_streaming
    from transformers import AutoTokenizer

    from nf4_kv_lowrank import calibrate_basis, project_to_codes, reconstruct_ref

    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(MODEL)
    # r/alpha are the LoRA shapes the loader attaches; irrelevant here (we only
    # read the KV cache from a forward pass) but required by the signature.
    model, _ = _unpack(load_moe_4bit_streaming(
        MODEL, device="cuda:0", dtype=torch.bfloat16, r=8, alpha=16,
        offload=OFFLOAD, pin=True, quant_type="nf4"))
    model.eval()

    # real text, not random ids: token identity drives the attention geometry
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tok(text, return_tensors="pt").input_ids[:, :N_TOKENS].cuda()

    # Hook the projections to capture keys BEFORE rotary embedding. RoPE rotates
    # identical content into different directions by position, which inflates
    # apparent rank; measuring only post-RoPE keys would confound "keys are not
    # low-rank" with "rotation spread them out". v_proj is captured too, purely
    # as a control — V has no RoPE, so it must match the cache exactly.
    grabbed: dict[str, torch.Tensor] = {}

    def _hook(name):
        def fn(_m, _i, o):
            grabbed.setdefault(name, o.detach())
        return fn

    handles = []
    for li, layer in enumerate(model.model.layers):
        handles.append(layer.self_attn.k_proj.register_forward_hook(_hook(f"k{li}")))
        handles.append(layer.self_attn.v_proj.register_forward_hook(_hook(f"v{li}")))
    with torch.no_grad():
        out = model(ids, use_cache=True)
    for h in handles:
        h.remove()
    cache = out.past_key_values
    n_layers = len(cache)
    half = ids.shape[1] // 2

    layers, iid_layers = {}, {}
    for li in range(n_layers):
        k = cache.layers[li].keys if hasattr(cache, "layers") else cache[li][0]
        v = cache.layers[li].values if hasattr(cache, "layers") else cache[li][1]
        k = k[0].transpose(0, 1).contiguous()          # [T, H, D]
        v = v[0].transpose(0, 1).contiguous()
        T, H, D = k.shape
        rec = {"kv_heads": H, "head_dim": D, "tokens": T}
        for name, x in (("K", k), ("V", v)):
            fit, held = x[:half], x[half:]
            rec[name] = {"energy_in_fit": _energy_curve(fit, RANKS)}
            # the honest number: basis from `fit`, error measured on `held`
            for r in RANKS:
                if r > D or r % 64:
                    continue
                B = calibrate_basis(fit, r)
                back = reconstruct_ref(project_to_codes(held, B), B)
                rel = ((back - held.float()).norm() / held.float().norm()).item()
                rec[name][f"heldout_rel_err_r{r}"] = rel
                same = reconstruct_ref(project_to_codes(fit, B), B)
                rec[name][f"insample_rel_err_r{r}"] = float(
                    ((same - fit.float()).norm() / fit.float().norm()).item())
        # pre-RoPE keys, and the V control that proves the hook reads the
        # same tensor the cache holds
        k_pre = grabbed[f"k{li}"][0].view(-1, H, D)
        v_pre = grabbed[f"v{li}"][0].view(-1, H, D)
        rec["K_preRoPE_heldout_rel_err_r64"] = _heldout_err(k_pre, 64, half)
        rec["V_proj_control_heldout_rel_err_r64"] = _heldout_err(v_pre, 64, half)
        rec["sweep"] = {f"K_r{r}": _heldout_err(k, r, half) for r in SWEEP}
        rec["sweep"].update({f"V_r{r}": _heldout_err(v, r, half) for r in SWEEP})
        layers[li] = rec
        # iid control at the identical shape
        g = torch.Generator(device="cpu").manual_seed(li)
        ctrl = (torch.randn(T, H, D, generator=g) * 0.5).cuda().bfloat16()
        iid_layers[li] = _energy_curve(ctrl[:half], RANKS)

    receipt = {
        "model": MODEL, "tokens": int(ids.shape[1]), "n_layers": n_layers,
        "ranks": RANKS, "calibration": "first half / evaluated on second half",
        "layers": layers, "iid_control": iid_layers,
        "elapsed_s": round(time.time() - t0, 1),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
        "loader": "experts4bit_qlora.load_moe_4bit_streaming", "offload": OFFLOAD,
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    dst = os.path.join(os.path.dirname(__file__), "lowrank_probe.json")
    with open(dst, "w") as f:
        json.dump(receipt, f, indent=2)

    r = RANKS[0]
    print(f"{MODEL}  {ids.shape[1]} tokens, {n_layers} layers, rank {r}")
    print(f"{'layer':>5} {'K held':>8} {'K in-samp':>10} {'V held':>8} "
          f"{'V in-samp':>10} {'r90 real':>9} {'r90 iid':>8}")
    for li in range(n_layers):
        L = layers[li]
        print(f"{li:>5} {L['K'][f'heldout_rel_err_r{r}']:>8.4f} "
              f"{L['K'][f'insample_rel_err_r{r}']:>10.4f} "
              f"{L['V'][f'heldout_rel_err_r{r}']:>8.4f} "
              f"{L['V'][f'insample_rel_err_r{r}']:>10.4f} "
              f"{L['K']['energy_in_fit']['rank_for_90pct']:>9.1f} "
              f"{iid_layers[li]['rank_for_90pct']:>8.1f}")
    def _mean(key):
        return sum(layers[i][key] if isinstance(layers[i].get(key), float)
                   else layers[i]["sweep"][key] for i in range(n_layers)) / n_layers

    kh = [layers[i]["K"][f"heldout_rel_err_r{r}"] for i in range(n_layers)]
    vh = [layers[i]["V"][f"heldout_rel_err_r{r}"] for i in range(n_layers)]
    print(f"\nheld-out rel err @ rank {r}: K mean {sum(kh)/len(kh):.4f} "
          f"(max {max(kh):.4f}), V mean {sum(vh)/len(vh):.4f} (max {max(vh):.4f})")

    kpre = sum(layers[i]["K_preRoPE_heldout_rel_err_r64"] for i in range(n_layers)) / n_layers
    vctl = sum(layers[i]["V_proj_control_heldout_rel_err_r64"] for i in range(n_layers)) / n_layers
    print(f"K pre-RoPE @ rank 64: {kpre:.4f}  (post-RoPE {sum(kh)/len(kh):.4f}) "
          f"| V control {vctl:.4f} must equal V {sum(vh)/len(vh):.4f}")

    print(f"\n{'rank':>5} {'ratio':>6} {'K held':>8} {'V held':>8}  packable")
    for rr in SWEEP:
        print(f"{rr:>5} {128/rr:>5.2f}x {_mean(f'K_r{rr}'):>8.3f} "
              f"{_mean(f'V_r{rr}'):>8.3f}  {'yes' if rr % 64 == 0 else 'no'}")
    print(f"receipt -> {dst}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
