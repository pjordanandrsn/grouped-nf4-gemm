# What the stamps on the 2026-08-15 protocols actually prove

Three protocols were registered on 2026-08-15 (the HBM3 half of the graphed
race, the R-scan, and the R-scan's amendment 1). All three are now
Bitcoin-anchored. Upgrading them exposed a precision problem in how this repo
describes its own evidence, so the timeline is written out per protocol rather
than summarised as "stamped pre-data".

**An OpenTimestamps anchor is an UPPER bound.** It proves the exact bytes
existed *before* its block's time. It cannot prove they existed before
anything earlier. Bitcoin's granularity — ~10-minute blocks plus calendar
aggregation latency — is 10× to 50× coarser than these runs, which finish in
2–4 minutes on a rented H100. **In all three cases below the anchoring block
was mined after the run had already produced its data.** The anchor therefore
establishes content integrity (the registered bands cannot have been rewritten
afterwards) but does **not**, on its own, establish pre-data registration.

**The load-bearing pre-data evidence is the public push receipt.** GitHub
stamps `PushEvent`/`CreateEvent` server-side; the committer cannot set those
times, unlike `git` author/committer dates. For every protocol here the push
receipt precedes the creation of the first pod that could produce data.

| protocol | claimed `registered_utc` | GitHub receipt (server-side) | first data-capable pod created | data landed | Bitcoin anchor |
|---|---|---|---|---|---|
| `prereg_graphed_buckets_hbm3` | 18:05:00Z | **19:02:41Z** (branch create, `707e83e`) | 19:05:58Z | 19:07:48Z | block 962618, **19:14:58Z** |
| `prereg_graphed_rscan` | 19:27:00Z | **19:30:28Z** (branch create, `be1a44f`) | 19:34:04Z | 19:42:45Z | block 962625, **21:00:27Z** |
| `prereg_graphed_rscan_amendment1` | 19:55:00Z | **19:47:32Z** (push, `7176046`) | 19:47:49Z | 19:55:10Z | block 962625, **21:00:27Z** |

Margins from public receipt to first data-capable pod: **3m17s**, **3m36s**,
**17s**. The practice held in all three; only the description of what proves
it was loose.

## Erratum: `prereg_graphed_rscan_amendment1.json`'s own `registered_utc` is wrong

The file states `19:55:00Z`. Its true public receipt is **19:47:32Z**, and the
data it governs landed at 19:55:10Z — so the file, read literally, claims a
registration 10 seconds ahead of its own experiment when the real margin was
7m38s. The value is a hand-written field, not a measurement, and it was
written optimistically while the amendment was being drafted.

**It is not corrected in place.** The `.ots` attests the exact bytes; editing
the file to make the record look better would destroy the attestation that
makes the record worth anything. The error is recorded here instead, which is
the same rule this repo applies to falsified bands: amend and record, never
rewrite. See also the sibling rule against repairing a distributed artifact.

## The rule going forward

1. `.ots` proves **integrity** and an upper time bound. Cite it for "these are
   the bands that were registered", not for "they were registered first".
2. For any run that turns around faster than Bitcoin confirmation, the
   **public push receipt** is the pre-data evidence — and it must precede the
   creation of the first pod, not merely the first timed cell.
3. Write `registered_utc` from the clock at stamping time, and treat any
   hand-written timestamp as unverified narrative; the receipts are the
   evidence.
