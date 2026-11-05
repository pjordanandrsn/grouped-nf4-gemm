### ColdTier: a request's resident rows resolve before any of its misses picks a victim

`ColdTier.ensure` resolved a request's keys in request order, so a miss could choose as its victim a slot holding a row
the same request reached later. That row was then read back from disk: two page-ins for one miss, and an unrelated row
evicted to hold the second. Every already-resident row of the request is now resolved first (the hit counted, a
reclaimable row resurrected, the slot protected), then the misses claim slots. Page-ins now equal misses on every
request. Served bytes, the speculative path, reservations and in-flight fills, the lock discipline and the
`hot_rows too small` refusal are unchanged. One accounting difference: a speculative ensure that stops early for lack of
an evictable slot now counts hits for the resident rows after the point where it stopped.
