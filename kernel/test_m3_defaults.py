# Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
"""The M3 default flip, and the applicability guard PASS did not cover.

RESULTS-m3-default-on is a PASS on QUALITY AND SPEED: 8192 scored
tokens, both knobs and their composition inside a +-0.05 perplexity
bar, 7.84 -> 6.80 ms. It licensed the defaults to move.

It did NOT establish that fp8-COMPUTE is APPLICABLE everywhere. That
path asserts sm_89+, ``v_groups == 1``, ``k_groups in (1, 2, 4)`` and
more -- none of which M3 varied, because M3 ran one config on one
5090. Flipping the default unconditionally would turn a working f32
install into an AssertionError on every pre-Ada GPU.

So these tests pin the three properties the flip has to have:
  1. unset env -> the fast path WHERE IT CAN RUN
  2. unset env -> the certified path where it cannot, silently
  3. an EXPLICIT request is never downgraded -- it fails loudly
"""
import pathlib

import pytest
import torch

import fp8_paged_attn as fpa
import nf4_grouped


class _Q:
    """Stand-in for q: the predicate reads only dtype and device."""

    def __init__(self, dtype=torch.bfloat16, device="cpu"):
        self.dtype = dtype
        self.device = device


def _cap(monkeypatch, major, minor):
    monkeypatch.setattr(torch.cuda, "get_device_capability",
                        lambda dev=None: (major, minor))


# ---- the predicate ---------------------------------------------------

def test_supported_config_returns_no_reason(monkeypatch):
    _cap(monkeypatch, 8, 9)
    assert fpa.fp8_compute_unsupported(_Q(), 128, 1, 1) is None


@pytest.mark.parametrize("cc", [(8, 0), (8, 6), (7, 5)])
def test_pre_ada_parts_are_refused_by_capability(monkeypatch, cc):
    """A100 is sm_80, 3090 is sm_86, T4 is sm_75 -- all real."""
    _cap(monkeypatch, *cc)
    why = fpa.fp8_compute_unsupported(_Q(), 128, 1, 1)
    assert why and "sm_89+" in why, why


def test_v_groups_and_k_groups_are_refused(monkeypatch):
    _cap(monkeypatch, 9, 0)
    assert "per-row only" in fpa.fp8_compute_unsupported(_Q(), 128, 1, 2)
    assert "k_groups" in fpa.fp8_compute_unsupported(_Q(), 128, 3, 1)
    assert ">=32-wide" in fpa.fp8_compute_unsupported(_Q(), 64, 4, 1)


def test_fp32_query_is_refused(monkeypatch):
    _cap(monkeypatch, 9, 0)
    why = fpa.fp8_compute_unsupported(_Q(torch.float32), 128, 1, 1)
    assert why and "bf16/fp16" in why


# ---- the default -----------------------------------------------------

def test_unset_env_selects_fp8_where_it_can_run(monkeypatch):
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    _cap(monkeypatch, 8, 9)
    assert fpa._compute_default(_Q(), 128, 1, 1) == "fp8"


def test_unset_env_falls_back_to_f32_where_it_cannot(monkeypatch):
    """The regression this guard exists to prevent.

    Without it, an A100 install that worked yesterday raises today.
    """
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    _cap(monkeypatch, 8, 0)
    assert fpa._compute_default(_Q(), 128, 1, 1) == "f32"


def test_an_EXPLICIT_fp8_request_is_never_downgraded(monkeypatch):
    """Silently handing back f32 under the name the user asked for is
    how an arm gets mislabelled. The path's asserts tell them."""
    monkeypatch.setenv("GNF4_ATTN_COMPUTE", "fp8")
    _cap(monkeypatch, 8, 0)              # cannot actually run it
    assert fpa._compute_default(_Q(), 128, 1, 1) == "fp8"


def test_explicit_f32_is_honoured(monkeypatch):
    monkeypatch.setenv("GNF4_ATTN_COMPUTE", "f32")
    _cap(monkeypatch, 9, 0)
    assert fpa._compute_default(_Q(), 128, 1, 1) == "f32"


def test_no_call_context_yields_the_certified_path(monkeypatch):
    """Nothing to judge applicability against -> do not guess."""
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    assert fpa._compute_default() == "f32"


def test_a_typo_still_raises(monkeypatch):
    monkeypatch.setenv("GNF4_ATTN_COMPUTE", "FP8")
    with pytest.raises(ValueError, match="not a compute mode"):
        fpa._compute_default(_Q(), 128, 1, 1)


# ---- dot-pad ---------------------------------------------------------

def test_dotpad_defaults_ON(monkeypatch):
    monkeypatch.delenv("GNF4_GEMV_DOTPAD", raising=False)
    assert nf4_grouped._dotpad() is True


def test_dotpad_can_be_forced_off(monkeypatch):
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "0")
    assert nf4_grouped._dotpad() is False


def test_dotpad_typo_raises_rather_than_reading_as_off(monkeypatch):
    """Previously ANY value but '1' meant OFF, so 'true' silently ran
    the scalar path. Now the default is ON, that same typo would read
    as a deliberate disable -- worse, because it looks intentional."""
    monkeypatch.setenv("GNF4_GEMV_DOTPAD", "true")
    with pytest.raises(ValueError, match="not 0 or 1"):
        nf4_grouped._dotpad()


def test_the_guard_and_the_asserts_share_one_predicate():
    """Two copies of this predicate is how a default starts choosing a
    path its own asserts reject.

    Counted over the WHOLE module rather than a computed window -- an
    offset-based slice of source is brittle and has already produced
    two tests in this repo that passed for the wrong reason.
    """
    src = pathlib.Path(fpa.__file__).read_text()
    # the predicate is DEFINED once and USED by the fp8 path
    assert src.count("def fp8_compute_unsupported(") == 1
    # NOT an exact call-string match: adding a kwarg to the predicate
    # broke that once already, which is the third brittle
    # source-spelling check in this repo. Count the call sites instead.
    # ...and the preconditions are not ALSO inlined next to it
    assert 'assert v_groups == 1, "fp8 compute folds' not in src, \
        "v_groups precondition re-inlined; it belongs in the predicate"
    # once inside the predicate itself, nowhere else. There are TWO
    # fp8 branches (split and packed) and both must gate on it.
    assert src.count("get_device_capability(q.device) < (8, 9)") == 1
    assert src.count("get_device_capability(q.device) >= (8, 9)") == 0, \
        "sm_89 precondition re-inlined; it belongs in the predicate"
    assert src.count("_why = fp8_compute_unsupported(") == 2, \
        "both fp8 branches -- split AND packed -- must gate on it"


def test_the_PACKED_branch_precondition_gates_the_default(monkeypatch):
    """The gap a split-path-only guard would have left.

    The packed fp8 branch reduces over BT*H_kv rather than ktile, so
    it refuses small block_tokens*n_kv_heads -- a config the split
    branch accepts. A default that consulted only the split path's
    preconditions would select fp8 straight into that assert. M3 never
    varied pack_heads either.
    """
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    _cap(monkeypatch, 9, 0)
    # identical config: fine unpacked, refused packed
    assert fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, pack_heads=False,
                                       block_tokens=8, n_kv_heads=2) is None
    why = fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, pack_heads=True,
                                      block_tokens=8, n_kv_heads=2)
    assert why and "BT*H_kv" in why, why
    # and the DEFAULT honours the difference
    assert fpa._compute_default(_Q(), 128, 1, 1, pack_heads=True,
                                block_tokens=8, n_kv_heads=2) == "f32"
    assert fpa._compute_default(_Q(), 128, 1, 1, pack_heads=True,
                                block_tokens=16, n_kv_heads=2) == "fp8"


def test_a_caller_supplied_ktile_gates_the_default(monkeypatch):
    """The regression review caught (gnf4#291).

    ktile is a KWARG, derived only when the caller passes None. I had
    skipped it in the predicate on a circularity argument that does
    not apply to a supplied value -- so `ktile=16` would have had the
    default select fp8 and then hit the leftover `ktile >= 32` assert,
    on a call that previously ran f32 without complaint.
    """
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    _cap(monkeypatch, 9, 0)
    # unsupplied -> vacuous, fp8 derives ktile=64
    assert fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, ktile=None) is None
    assert fpa._compute_default(_Q(), 128, 1, 1, ktile=None) == "fp8"
    # supplied and too small -> f32, NOT fp8-then-assert
    why = fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, ktile=16)
    assert why and "ktile" in why, why
    assert fpa._compute_default(_Q(), 128, 1, 1, ktile=16) == "f32"
    # supplied and adequate -> fp8
    assert fpa._compute_default(_Q(), 128, 1, 1, ktile=64) == "fp8"


def test_ktile_does_not_gate_the_PACKED_branch(monkeypatch):
    """Over-restriction is a cost too, not just a crash.

    The packed kernel never takes KTILE -- it reduces over
    block_tokens * n_kv_heads. Applying the split branch's ktile
    constraint to a packed call downgraded it to f32 when packed fp8
    would have run, losing the speedup for a constraint that branch
    does not have (review, gnf4#291). A guard that is too strict
    fails quietly, which is why it needs its own test.
    """
    monkeypatch.delenv("GNF4_ATTN_COMPUTE", raising=False)
    _cap(monkeypatch, 9, 0)
    # split: small ktile refuses
    assert fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, ktile=16) is not None
    # packed with the SAME small ktile: fp8 is fine, BT*H_kv decides
    assert fpa.fp8_compute_unsupported(
        _Q(), 128, 1, 1, ktile=16, pack_heads=True,
        block_tokens=16, n_kv_heads=2) is None
    assert fpa._compute_default(
        _Q(), 128, 1, 1, ktile=16, pack_heads=True,
        block_tokens=16, n_kv_heads=2) == "fp8"
    # ...and BT*H_kv still gates it
    assert fpa._compute_default(
        _Q(), 128, 1, 1, ktile=16, pack_heads=True,
        block_tokens=8, n_kv_heads=2) == "f32"


def test_every_reason_string_is_reachable(monkeypatch):
    """A predicate branch nothing can trigger is not a guard.

    Each refusal above is driven by a config that reaches it, so the
    set of distinct reasons must equal the number of branches.
    """
    _cap(monkeypatch, 9, 0)
    reasons = {
        fpa.fp8_compute_unsupported(_Q(), 128, 1, 2),          # v_groups
        fpa.fp8_compute_unsupported(_Q(), 128, 3, 1),          # k_groups
        fpa.fp8_compute_unsupported(_Q(), 64, 4, 1),           # width
        fpa.fp8_compute_unsupported(_Q(torch.float32), 128, 1, 1),  # dtype
    }
    reasons.add(fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, ktile=16))
    reasons.add(fpa.fp8_compute_unsupported(_Q(), 128, 1, 1, pack_heads=True,
                                            block_tokens=8, n_kv_heads=2))
    assert len(reasons) == 6 and None not in reasons, reasons
    _cap(monkeypatch, 8, 0)
    assert fpa.fp8_compute_unsupported(_Q(), 128, 1, 1) not in reasons
