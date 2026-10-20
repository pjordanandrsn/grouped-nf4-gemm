### Docs: the sm_120 census's byte ratio is 3.56x, not 2x

`gnf4.kernel.sm120-census-vs-grouped-mm`'s text, STATUS and the census RESULTS said the bf16 comparator ran on "half the
bytes" / "2x the bytes". The receipt's own `nf4_gb` counts 4.5 bits a weight (4-bit codes plus one fp32 absmax per 64)
against bf16's 16, so the NF4 arm reads 3.56x fewer weight bytes. Text only: the 2.1-6.0x speed value is unchanged.
