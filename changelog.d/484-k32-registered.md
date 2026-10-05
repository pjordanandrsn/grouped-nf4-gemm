### K32 registered: K30's split-K R-term sweep on one rented RTX A4000, the second ≤64-SM architecture (bench only)

- `kernel/PREREG-k32-splitk-r-term-a4000.md`: K30's instrument and rule, frozen at its measured cut `aa562718`, on one
  rented RTX A4000 (48 SMs, sm_86) via e4b's `bench/k30/` runner (`K30_CARD=A4000`).
- The lane cannot change the default. K30's OFF stands, since KEEP needs both cards. An A4000 OFF makes that OFF
  two-card; an A4000 KEEP is reported as an sm_86-specific observation.
