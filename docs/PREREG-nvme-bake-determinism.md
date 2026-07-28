# PREREG — cross-box bake determinism (pre-data)

Registered before the consumer-box bake starts. Replaces the split-site *file
transfer* registered in amendment 1, which was abandoned on measurement:
the pod→mini link sustained **~6 MB/s**, i.e. ~6 h and ~$6 of idle GPU to
move 128 GB. The distribution claim is tested better, cheaper, and more
strongly by reproducing the bake independently than by copying it.

## The question

Amendment 1's split-site story was "bake once, verify anywhere, serve on
the consumer box." A file copy demonstrates transport. **Independent
reproduction demonstrates something stronger: that the arena is a
*function of the checkpoint*, not an artifact of the machine that baked
it.** If two different GPUs (L40S sm_89 vs A2000 sm_86), two different
bitsandbytes runtimes and two different hosts turn the same shipped bytes
into the same arena bytes, then a user can bake locally and check against
a published manifest — no 128 GB download, and provenance survives.

## Fixture

The consumer box (A2000 12 GB, bnb 0.50.0) bakes Qwen3-235B-A22B-Instruct-2507
from **its own local 438 GB source download** (independent of the pod's
copy) with `nvme_bake_nf4.py`, identical flags. The comparator is the
L40S bake's manifest, already banked
(`bench/nvme/receipts/n5b/…`, 12,032 rows × 4 segments of sha256, mini
`~/n5c-evidence/q235b.arena.manifest.json`). The L40S arena file itself no
longer exists; the manifest is the whole point — that is what a user
would be handed.

## Registered outcomes

- **D1 (primary):** for every (layer, expert, segment), consumer-box `sha256` ==
  L40S `sha256`. Scored as the **fraction of the 48,128 segments that
  match**. Registered prediction: **≥ 0.999**. NF4 quantization is
  deterministic arithmetic over identical inputs; block absmax is a max
  and a divide.
- **D2 (interpretation, pre-committed both ways):**
  - D1 ≥ 0.999 → *"the arena is reproducible across GPU architecture and
    host; publish the manifest, let users bake locally."*
  - D1 < 0.999 → **the more interesting outcome**: quantization is
    device-dependent, and any provenance claim over a *quantized* arena
    must name the box that produced it. This would immediately narrow the
    `nf4-quantize` bake mode's two-hop claim, and I would report it as a
    limitation of that mode, not a bug to be tuned away. The relocation
    mode (K3, gpt-oss) is unaffected either way, since it copies bytes.
- **D3 (source control):** the consumer box's own source shards must first match
  the recorded source hashes in the L40S manifest (`sources[].sha256`) on
  a ≥ 64-segment spot check. If the two downloads differ, D1 is void and
  the finding is about the download, not the bake.

## Rules

Bands do not move after data. Whichever way D1 lands, it ships at full
volume: a pass strengthens the distribution story, a fail narrows the
quantize-bake provenance claim. No re-runs.

---

*Publication note (2026-07-28, post-data): the owning host's appliance
vendor and LAN octet were replaced with "consumer box" for publication,
the same redaction already applied to `nvme-ceilings.md`. Disclosed here
rather than done silently. **No registered content changed** — question,
fixture, outcomes D1/D2/D3, the 0.999 band and the no-re-runs rule are
byte-for-byte as registered pre-data; the pre-data text is in the git
history of this file.*
