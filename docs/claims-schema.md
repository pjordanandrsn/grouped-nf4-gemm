# Claims register — schema (docs/claims.json in both repos)

One JSON object per claim. A claim is one sentence a reader could act on,
with a number where there is one. Every README/docs number must map to an
entry. The bookkeeping fields below are enforced by
`scripts/check_claims_register.py` (CI); the prose that quotes ids by
`scripts/check_readme_claims.py`.

```json
{
  "id": "e4b.serve.b1.qwen3-30b.nf4.5090",         // stable slug, never reused
  "package": "experts4bit-qlora" | "grouped-nf4-gemm",
  "area": "train" | "offload" | "serve" | "kernel" | "parity" | "provenance" | "portability" | "roadmap",
  "claim": "one sentence, present tense, the thing a user gets",
  "value": 98.3, "unit": "tok/s",                    // omit for qualitative claims
  "model": "Qwen/Qwen3-30B-A3B", "hardware": "RTX 5090 (sm_120), rented Vast host",
  "conditions": "B=1, NF4 experts, fp8 paged KV, 512-token prompt, --no-fuse-qkv",
  "measured_on": "2026-09-03",                       // ISO date of the run; required on measured rows
  "status": "verified" | "confirmed" | "measured" | "measured-private" | "projected" | "retired" | "superseded" | "open",
  "evidence": ["bench/hybrid-g9/b1/RESULTS-b1-decomposition.md"],  // PUBLIC, resolvable at HEAD (forms below)
  "evidence_private": ["INT4B16/P25-PARITY.md"],    // exists but not in this repo -- reader cannot check it
  "supersedes": ["<id>"], "superseded_by": "<id>",
  "retired_reason": "why, in one sentence, with the measurement that retired it",
  "quoted_in": ["README.md#What is measured", "docs/METHODOLOGY.md#L13"],
  "notes": "caveats, refused arms, corrections -- never the headline"
}
```

Status meanings (the file's own `status_vocabulary` is authoritative; the
system manifest's `evidence_vocabulary` covers both registers):
- **verified** — reproduced under stated conditions, receipt public in this repo.
- **confirmed** — pre-registered, OpenTimestamps-stamped blind confirmatory
  run (the kernel register's top tier).
- **measured** — one run, receipt public. (The repos' own tier language.)
- **measured-private** — the receipt exists only in the private audit tree; the
  number is real but a reader of this repo cannot check it. MUST be flagged in
  prose, or the receipt published.
- **projected** — arithmetic, not a run.
- **retired** — was published, now known wrong; keep the entry so the retraction
  is findable, never delete.
- **superseded** — still true as measured, but a later entry replaces it as the
  number to quote (e.g. v0 offload tok/s superseded by the paged engine).
- **open** — a claim the docs make that has no evidence either way yet.

## Field rules (enforced)

**`evidence[]`** — every entry resolves at HEAD of this repository, in one of
these forms; annotated strings (`"receipts-m3/"`, `"kernel/RESULTS-*.md
(0.7.0)"`, `"CHANGELOG.md 0.24.0"`) are refused:

| form | meaning | resolved how |
|---|---|---|
| `"kernel/RESULTS-x.md"`, `"kernel/receipts-m3/"` | a bare repository path (file or directory) | exists at HEAD |
| `{"path": "CHANGELOG.md", "section": "0.24.0"}` | a section of a Markdown file | the file has a heading whose text starts with `section` (`## 0.24.0 — 2026-09-03`) |
| `{"glob": "bench/x/receipts/*.json"}` | a family of files | at least one file matches at HEAD |
| `{"repository": "experts4bit-qlora", "path": "bench/dgrad-gate/RESULTS-dgrad-gate.md"}` | a receipt in the related repository | `repository` is a related project (capabilities.json `related_projects` / the system manifest); the path is resolved when `--sibling DIR` names a checkout, else listed as SKIP |
| `"https://github.com/<owner>/<repo>/issues/319"` | the issue tracker (open claims) | must be under this repository's own GitHub URL; never fetched |

Any object form may carry a `note` (free text). `measured`, `confirmed` and
`verified` rows need a non-empty `evidence`; `measured-private` rows a
non-empty `evidence_private` (paths outside the repository, listed and never
resolved).

**`measured_on`** — required on `measured`, `measured-private`, `confirmed`
and `verified` rows; an ISO calendar date `YYYY-MM-DD` (a month is not a
date). It is the date of the run the numbers come from — the receipt's own
stated date, or, where the receipt states none, the receipt's first commit,
which the row's `notes` then say.

**`superseded`** — `superseded_by` names a claim that exists, is active
(verified / confirmed / measured / measured-private) and lists this id in its
own `supersedes`: "replaced by a later claim that names it". Every
`supersedes` entry exists and has status `superseded`.

**`retired`** — `retired_reason` is a non-empty sentence carrying the
measurement that retired the claim.

**`quoted_in[]`** — where the claim is quoted; each entry resolves as a
location: `path` (exists), `path#L<n>` or `path:<n>` (a line the file has),
or `path#<heading>` (a heading whose text starts with the anchor:
`CHANGELOG.md#0.18.0`, `README.md#What was retired`). A location, not a
containment check — the file need not spell the id on that line.

**`notes`** on an active row carry no placeholder word (`pending`, `TBD`,
`TODO`): an active row is presented as current, so it states the fact or
becomes `open`. `notes` are never read for numbers by any check; the
headline numbers are `value`, `unit` and the `claim` sentence.

Human-readable companion: docs/STATUS.md renders this as three lists —
"what you get today" (verified/confirmed/measured), "what changed"
(superseded/retired with the one-line reason), "what is open".
