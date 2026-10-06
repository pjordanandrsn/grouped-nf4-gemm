### Docs: the bucketed-padding default's speed evidence is torch 2.12 only

`docs/STATUS.md` now says that every speed reading behind the 0.42.0 `auto` default (#492) was taken under torch
2.12.1 / triton 3.7.1, so under torch 2.8 / triton 3.4 the effect is unmeasured. experts4bit-qlora's TC1
amendment 51 found e4b slower than expected under torch 2.8 with the default on (environment ratio 0.739, against
a registered [0.84, 0.98]). That is a signal, not a measurement of bucketing, and a torch-2.8 A/B is registered
next. The default does not change. `NF4_QLORA_PAD_BUCKETS=0` restores the single block exactly. Docs only.
