/* Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
 *
 * cpu_kernels.c — the hybrid tier's CPU compute backend (Phase 2, gate G2).
 *
 * Grouped expert GEMV / small-batch GEMM directly on packed NF4 and MXFP4
 * bytes (the same arena bytes the GPU kernels consume — one artifact, no
 * repack, no materialized dequant), fp32 accumulate, LOCKED summation
 * order. Plus the single-call router epilogue that removes the G1 p50
 * interpreter-dispatch gap.
 *
 * Built at first use by gnf4_native.build with `cc -O3 -march=native
 * -fopenmp -shared -fPIC` on the box it runs on; ISA selection is a
 * compile-time fact of that box, with a portable path always present.
 *
 * ## The locked summation order (the bit-exactness contract)
 *
 * For output element (row r, col n), K elements are consumed in 16-lane
 * groups g = 0,1,2,... ascending. Lane-group g accumulates into vector
 * accumulator g % 4 (four independent chains for ILP). Per lane:
 *
 *     w    = LUT[code] * scale        (one fp32 multiply, one rounding)
 *     prod = w * a                    (fp32 multiply)
 *     acc += prod                     (fp32 add — MUL+ADD, deliberately
 *                                      NOT FMA: two roundings, so the
 *                                      reference is expressible in plain
 *                                      numpy; the kernel is bandwidth-
 *                                      bound and pays nothing for it)
 *
 * After the K loop: comb = (acc0 + acc1) + (acc2 + acc3) elementwise,
 * then a SEQUENTIAL scalar sum of comb's 16 lanes, lane 0 first. The
 * python mirror is `ordered_gemv_ref` in kernel/cpu_grouped.py; the test
 * suite requires exact equality. The hand-vector paths implement the
 * identical tree (vectorizing ACROSS lanes never reorders WITHIN a lane's
 * chain).
 *
 * Nibble order is format-specific and oracle-pinned upstream:
 *   NF4   element 2j = HIGH nibble (bnb convention)
 *   MXFP4 element 2j = LOW  nibble (transformers convention)
 *
 * MXFP4 scale = 2^(e8m0 - 127). For 1 <= e <= 254 this is exact via bit
 * assembly ((e) << 23). e = 0 gives the subnormal 2^-127 (exactly
 * representable; the multiply stays exact). e = 0xFF means 2^128 = +inf
 * under the upstream oracle's ldexp semantics, where ldexp(0, 128) == 0
 * but 0 * inf == NaN — so 0xFF blocks take a scalar ldexpf path to
 * preserve the oracle's exact behavior.
 *
 * ## Threading
 *
 * OpenMP parallel-for over (group, column-tile) work items, static
 * schedule. Pinning is the caller's contract: run under OMP_PLACES=cores
 * OMP_PROC_BIND=spread (the bench records both). A custom spin-pool for
 * decode latency is Phase-3 executor work, not Phase 2.
 */

#include <stdint.h>
#include <string.h>
#include <math.h>

#if defined(__x86_64__)
#include <immintrin.h>
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define EXPORT __attribute__((visibility("default")))

/* ------------------------------------------------------------ constants -- */

/* bnb NF4 codebook — must match kernel/nf4_grouped.py NF4_LUT exactly */
static const float NF4_LUT[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f,
};

/* e2m1 codebook — must match kernel/mxfp4_pack_ref.py FP4_VALUES exactly */
static const float FP4_LUT[16] = {
    0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

/* ------------------------------------------------------------- features -- */

EXPORT int gnf4_cpu_features(void) {
    int f = 0;
#if defined(__x86_64__)
    unsigned a, b, c, d;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        if ((b >> 5) & 1) f |= 1;          /* avx2 */
        if ((b >> 16) & 1) f |= 2;         /* avx512f */
        if ((c >> 1) & 1) f |= 4;          /* avx512vbmi */
        if ((c >> 11) & 1) f |= 8;         /* avx512vnni */
    }
#endif
#if defined(__AVX512F__)
    f |= 16;                               /* compiled with the hand path */
#endif
#ifdef _OPENMP
    f |= 32;
#endif
    return f;
}

/* --------------------------------------------- portable exact inner core -- */

/* 16-lane group step, portable: compilers vectorize across lanes without
 * reassociating any lane's chain, so this is both the fallback and the
 * executable spec. codes[16] are LUT indices, scale is the block scale. */
static inline void lane_step(const uint8_t *codes, const float *lut,
                             float scale, const float *a, float *acc16) {
    for (int i = 0; i < 16; i++) {
        float w = lut[codes[i]] * scale;
        acc16[i] += w * a[i];
    }
}

static inline float combine_acc(const float acc[4][16]) {
    float comb[16];
    for (int i = 0; i < 16; i++)
        comb[i] = (acc[0][i] + acc[1][i]) + (acc[2][i] + acc[3][i]);
    float s = 0.0f;
    for (int i = 0; i < 16; i++)
        s += comb[i];
    return s;
}

/* unpack 16 packed bytes -> 32 ordered codes. hi_first: element 2j = high
 * nibble (NF4); else element 2j = low nibble (MXFP4). */
static inline void unpack32(const uint8_t *p, int hi_first, uint8_t *codes) {
    for (int j = 0; j < 16; j++) {
        uint8_t hi = p[j] >> 4, lo = p[j] & 0x0F;
        codes[2 * j] = hi_first ? hi : lo;
        codes[2 * j + 1] = hi_first ? lo : hi;
    }
}

/* mxfp4 block scale, oracle-exact (see header) */
static inline float e8m0_scale(uint8_t e) {
    if (e >= 1 && e <= 254) {
        union { uint32_t u; float f; } v;
        v.u = (uint32_t)e << 23;
        return v.f;
    }
    return ldexpf(1.0f, (int)e - 127);     /* e==0 subnormal, e==255 inf */
}

/* ------------------------------------------------- AVX-512 hand inner core */

#if defined(__AVX512F__)

/* Same tree as lane_step, 16 lanes at once. The 16-entry fp32 LUT lives in
 * one zmm and is indexed with vpermps (_mm512_permutexvar_ps) — 16 value
 * lookups per instruction. (vpermb would give 64 BYTE lookups, but the
 * values are fp32 and the exactness contract forbids the int8/VNNI detour;
 * 16/instruction already sits past the memory roofline.) */
static inline __m512 lane_step_avx512(__m512i codes32, __m512 lutv,
                                      __m512 scale, const float *a,
                                      __m512 acc) {
    __m512 w = _mm512_mul_ps(_mm512_permutexvar_ps(codes32, lutv), scale);
    __m512 prod = _mm512_mul_ps(w, _mm512_loadu_ps(a));
    return _mm512_add_ps(acc, prod);       /* mul+add, NOT fma — locked */
}

/* unpack 16 packed bytes into two zmm of 16 u32 codes (ordered elements
 * 0..15 and 16..31) */
static inline void unpack32_avx512(const uint8_t *p, int hi_first,
                                   __m512i *c0, __m512i *c1) {
    __m128i bytes = _mm_loadu_si128((const __m128i *)p);
    __m128i hi = _mm_and_si128(_mm_srli_epi16(bytes, 4), _mm_set1_epi8(0x0F));
    __m128i lo = _mm_and_si128(bytes, _mm_set1_epi8(0x0F));
    __m128i a = hi_first ? _mm_unpacklo_epi8(hi, lo) : _mm_unpacklo_epi8(lo, hi);
    __m128i b = hi_first ? _mm_unpackhi_epi8(hi, lo) : _mm_unpackhi_epi8(lo, hi);
    *c0 = _mm512_cvtepu8_epi32(a);
    *c1 = _mm512_cvtepu8_epi32(b);
}

#endif /* __AVX512F__ */

/* ------------------------------------------------------------- NF4 GEMV -- */

#if defined(__AVX512F__)
/* T=1 specialization — the decode gate shape. The generic cell's
 * accv[8][4] array is runtime-indexed, so the compiler spills every
 * accumulator to the stack and the loop degrades to load-op-store (~5x:
 * measured 264 stack vmovaps + 126 scalar adds in the generic build).
 * Four NAMED locals are register-resident by construction. Same locked
 * tree, exactly. */
static float nf4_cell_t1_avx512(const float *a, int64_t K,
                                const uint8_t *w_row, const float *amax_row) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    const __m512 lutv = _mm512_loadu_ps(NF4_LUT);
    int64_t nblk = K / 64;
    for (int64_t b = 0; b < nblk; b++) {
        /* fold the block scale into the LUT once: w = (LUT*sc)[code] is the
         * SAME fp32 value as LUT[code]*sc (identical mul, per distinct
         * code instead of per element) — one zmm mul per block replaces
         * four. Exactness preserved by construction. */
        __m512 slut = _mm512_mul_ps(lutv, _mm512_set1_ps(amax_row[b]));
        const uint8_t *p = w_row + b * 32;
        __m512i c0, c1, c2, c3;
        unpack32_avx512(p, 1, &c0, &c1);
        unpack32_avx512(p + 16, 1, &c2, &c3);
        const float *ar = a + b * 64;
        acc0 = _mm512_add_ps(acc0, _mm512_mul_ps(
            _mm512_permutexvar_ps(c0, slut), _mm512_loadu_ps(ar)));
        acc1 = _mm512_add_ps(acc1, _mm512_mul_ps(
            _mm512_permutexvar_ps(c1, slut), _mm512_loadu_ps(ar + 16)));
        acc2 = _mm512_add_ps(acc2, _mm512_mul_ps(
            _mm512_permutexvar_ps(c2, slut), _mm512_loadu_ps(ar + 32)));
        acc3 = _mm512_add_ps(acc3, _mm512_mul_ps(
            _mm512_permutexvar_ps(c3, slut), _mm512_loadu_ps(ar + 48)));
    }
    float acc[4][16];
    _mm512_storeu_ps(acc[0], acc0);
    _mm512_storeu_ps(acc[1], acc1);
    _mm512_storeu_ps(acc[2], acc2);
    _mm512_storeu_ps(acc[3], acc3);
    return combine_acc(acc);
}

/* returns 0 and writes *out, or 1 when a 0xFF block demands the exact
 * ldexp slow path (a sentinel VALUE cannot work — inf/NaN are legitimate
 * outputs of 0xFF blocks) */
static int mx_cell_t1_avx512(const float *a, int64_t K, const uint8_t *w_row,
                             const uint8_t *sc_row, float *out) {
    __m512 acc0 = _mm512_setzero_ps(), acc1 = _mm512_setzero_ps();
    __m512 acc2 = _mm512_setzero_ps(), acc3 = _mm512_setzero_ps();
    const __m512 lutv = _mm512_loadu_ps(FP4_LUT);
    int64_t nblk = K / 32;
    for (int64_t b = 0; b < nblk; b++) {
        uint8_t e = sc_row[b];
        if (e == 0xFF) return 1;
        __m512 slut = _mm512_mul_ps(lutv, _mm512_set1_ps(e8m0_scale(e)));
        __m512i c0, c1;
        unpack32_avx512(w_row + b * 16, 0, &c0, &c1);
        const float *ar = a + b * 32;
        __m512 p0 = _mm512_mul_ps(_mm512_permutexvar_ps(c0, slut),
                                  _mm512_loadu_ps(ar));
        __m512 p1 = _mm512_mul_ps(_mm512_permutexvar_ps(c1, slut),
                                  _mm512_loadu_ps(ar + 16));
        if ((b & 1) == 0) {
            acc0 = _mm512_add_ps(acc0, p0);
            acc1 = _mm512_add_ps(acc1, p1);
        } else {
            acc2 = _mm512_add_ps(acc2, p0);
            acc3 = _mm512_add_ps(acc3, p1);
        }
    }
    float acc[4][16];
    _mm512_storeu_ps(acc[0], acc0);
    _mm512_storeu_ps(acc[1], acc1);
    _mm512_storeu_ps(acc[2], acc2);
    _mm512_storeu_ps(acc[3], acc3);
    *out = combine_acc(acc);
    return 0;
}
#endif /* __AVX512F__ */

/* one (rows-of-group, one output column) cell.
 * a: T rows of K fp32 (contiguous, the group's rows)
 * w_row: K/2 packed bytes;  amax_row: K/64 fp32
 * out: T strided by N */
static void nf4_cell(const float *a, int T, int64_t K, const uint8_t *w_row,
                     const float *amax_row, float *out, int64_t N,
                     int use512) {
    /* T accumulator sets — T <= 8 by contract (decode shapes). Zeroed in
     * the branch that uses them: an unconditional memset costs ~25% of the
     * cell's streamed bytes on the vector path. */
    float acc[8][4][16];
#if defined(__AVX512F__)
    if (use512) {
        __m512 accv[8][4];
        for (int r = 0; r < T; r++)
            for (int g = 0; g < 4; g++) accv[r][g] = _mm512_setzero_ps();
        __m512 lutv = _mm512_loadu_ps(NF4_LUT);
        int64_t nblk = K / 64;
        for (int64_t b = 0; b < nblk; b++) {
            __m512 sc = _mm512_set1_ps(amax_row[b]);
            const uint8_t *p = w_row + b * 32;
            __m512i c0, c1, c2, c3;
            unpack32_avx512(p, 1, &c0, &c1);
            unpack32_avx512(p + 16, 1, &c2, &c3);
            int64_t base = b * 64;
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                accv[r][0] = lane_step_avx512(c0, lutv, sc, ar, accv[r][0]);
                accv[r][1] = lane_step_avx512(c1, lutv, sc, ar + 16, accv[r][1]);
                accv[r][2] = lane_step_avx512(c2, lutv, sc, ar + 32, accv[r][2]);
                accv[r][3] = lane_step_avx512(c3, lutv, sc, ar + 48, accv[r][3]);
            }
        }
        for (int r = 0; r < T; r++) {
            for (int g = 0; g < 4; g++)
                _mm512_storeu_ps(acc[r][g], accv[r][g]);
            out[(int64_t)r * N] = combine_acc(acc[r]);
        }
        return;
    }
#else
    (void)use512;
#endif
    {
        memset(acc, 0, sizeof(float) * (size_t)T * 64);
        int64_t nblk = K / 64;
        uint8_t codes[64];
        for (int64_t b = 0; b < nblk; b++) {
            float sc = amax_row[b];
            unpack32(w_row + b * 32, 1, codes);
            unpack32(w_row + b * 32 + 16, 1, codes + 32);
            int64_t base = b * 64;
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                for (int g = 0; g < 4; g++)
                    lane_step(codes + g * 16, NF4_LUT, sc, ar + g * 16,
                              acc[r][g & 3]);
            }
        }
        for (int r = 0; r < T; r++)
            out[(int64_t)r * N] = combine_acc(acc[r]);
    }
}

/* a         [R_total, K] fp32, rows sorted by group
 * B         [E, N, K/2] u8
 * absmax    [E, N, K/64] fp32
 * eids      [G] i64, sizes [G] i32 (all > 0, sum = R_total)
 * out       [R_total, N] fp32
 * returns 0, or -1 on bad shape */
EXPORT int gnf4_gemv_nf4_grouped(const float *a, const uint8_t *B,
                                 const float *absmax, const int64_t *eids,
                                 const int32_t *sizes, int G, int64_t N,
                                 int64_t K, float *out, int threads) {
    if (K % 64 || N <= 0 || G <= 0) return -1;
    int use512 = (gnf4_cpu_features() & 16) != 0;
    int64_t row_off[512];
    if (G > 512) return -1;
    int64_t r0 = 0;
    for (int g = 0; g < G; g++) {
        if (sizes[g] <= 0 || sizes[g] > 8) return -1;
        row_off[g] = r0;
        r0 += sizes[g];
    }
    const int64_t TILE = 32;               /* output columns per work item */
    int64_t tiles_n = (N + TILE - 1) / TILE;
    int64_t total = (int64_t)G * tiles_n;
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < total; u++) {
        int g = (int)(u / tiles_n);
        int64_t tn = (u % tiles_n) * TILE;
        int64_t tn_end = tn + TILE > N ? N : tn + TILE;
        int64_t e = eids[g];
        const uint8_t *w_e = B + e * N * (K / 2);
        const float *am_e = absmax + e * N * (K / 64);
        const float *a_g = a + row_off[g] * K;
        float *out_g = out + row_off[g] * N;
#if defined(__AVX512F__)
        if (use512 && sizes[g] == 1) {
            for (int64_t n = tn; n < tn_end; n++)
                out_g[n] = nf4_cell_t1_avx512(a_g, K, w_e + n * (K / 2),
                                              am_e + n * (K / 64));
            continue;
        }
#endif
        for (int64_t n = tn; n < tn_end; n++)
            nf4_cell(a_g, sizes[g], K, w_e + n * (K / 2), am_e + n * (K / 64),
                     out_g + n, N, use512);
    }
    return 0;
}

/* ----------------------------------------------------------- MXFP4 GEMV -- */

static void mx_cell(const float *a, int T, int64_t K, const uint8_t *w_row,
                    const uint8_t *sc_row, float *out, int64_t N, int use512) {
    float acc[8][4][16];
    int64_t nblk = K / 32;                 /* 32-element e8m0 blocks */
#if defined(__AVX512F__)
    if (use512) {
        __m512 accv[8][4];
        for (int r = 0; r < T; r++)
            for (int g = 0; g < 4; g++) accv[r][g] = _mm512_setzero_ps();
        __m512 lutv = _mm512_loadu_ps(FP4_LUT);
        for (int64_t b = 0; b < nblk; b++) {
            uint8_t e = sc_row[b];
            if (e == 0xFF) goto scalar_tail;   /* oracle ldexp semantics */
            __m512 sc = _mm512_set1_ps(e8m0_scale(e));
            __m512i c0, c1;
            unpack32_avx512(w_row + b * 16, 0, &c0, &c1);
            int64_t base = b * 32;
            /* two lane-groups per block; group parity alternates 0,1,2,3
             * across consecutive blocks: lane-group index = 2b, 2b+1 */
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                int g0 = (int)((2 * b) & 3), g1 = (int)((2 * b + 1) & 3);
                accv[r][g0] = lane_step_avx512(c0, lutv, sc, ar, accv[r][g0]);
                accv[r][g1] = lane_step_avx512(c1, lutv, sc, ar + 16, accv[r][g1]);
            }
        }
        for (int r = 0; r < T; r++) {
            for (int g = 0; g < 4; g++)
                _mm512_storeu_ps(acc[r][g], accv[r][g]);
            out[(int64_t)r * N] = combine_acc(acc[r]);
        }
        return;
    }
scalar_tail:;
#else
    (void)use512;
#endif
    {
        memset(acc, 0, sizeof(float) * (size_t)T * 64);
        uint8_t codes[32];
        for (int64_t b = 0; b < nblk; b++) {
            float sc = e8m0_scale(sc_row[b]);
            unpack32(w_row + b * 16, 0, codes);
            int64_t base = b * 32;
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                int g0 = (int)((2 * b) & 3), g1 = (int)((2 * b + 1) & 3);
                if (sc_row[b] == 0xFF) {
                    /* ldexp path: w = ldexpf(LUT[c], 128); 0 stays 0 */
                    for (int i = 0; i < 16; i++) {
                        float w = ldexpf(FP4_LUT[codes[i]], 128);
                        acc[r][g0][i] += w * ar[i];
                    }
                    for (int i = 0; i < 16; i++) {
                        float w = ldexpf(FP4_LUT[codes[16 + i]], 128);
                        acc[r][g1][i] += w * ar[16 + i];
                    }
                } else {
                    lane_step(codes, FP4_LUT, sc, ar, acc[r][g0]);
                    lane_step(codes + 16, FP4_LUT, sc, ar + 16, acc[r][g1]);
                }
            }
        }
        for (int r = 0; r < T; r++)
            out[(int64_t)r * N] = combine_acc(acc[r]);
    }
}

EXPORT int gnf4_gemv_mxfp4_grouped(const float *a, const uint8_t *B,
                                   const uint8_t *scales, const int64_t *eids,
                                   const int32_t *sizes, int G, int64_t N,
                                   int64_t K, float *out, int threads) {
    if (K % 32 || N <= 0 || G <= 0 || G > 512) return -1;
    int use512 = (gnf4_cpu_features() & 16) != 0;
    int64_t row_off[512];
    int64_t r0 = 0;
    for (int g = 0; g < G; g++) {
        if (sizes[g] <= 0 || sizes[g] > 8) return -1;
        row_off[g] = r0;
        r0 += sizes[g];
    }
    const int64_t TILE = 32;
    int64_t tiles_n = (N + TILE - 1) / TILE;
    int64_t total = (int64_t)G * tiles_n;
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < total; u++) {
        int g = (int)(u / tiles_n);
        int64_t tn = (u % tiles_n) * TILE;
        int64_t tn_end = tn + TILE > N ? N : tn + TILE;
        int64_t e = eids[g];
        const uint8_t *w_e = B + e * N * (K / 2);
        const uint8_t *sc_e = scales + e * N * (K / 32);
        const float *a_g = a + row_off[g] * K;
        float *out_g = out + row_off[g] * N;
#if defined(__AVX512F__)
        if (use512 && sizes[g] == 1) {
            for (int64_t n = tn; n < tn_end; n++)
                if (mx_cell_t1_avx512(a_g, K, w_e + n * (K / 2),
                                      sc_e + n * (K / 32), &out_g[n]))
                    mx_cell(a_g, 1, K, w_e + n * (K / 2), sc_e + n * (K / 32),
                            out_g + n, N, 0);
            continue;
        }
#endif
        for (int64_t n = tn; n < tn_end; n++)
            mx_cell(a_g, sizes[g], K, w_e + n * (K / 2), sc_e + n * (K / 32),
                    out_g + n, N, use512);
    }
    return 0;
}

/* -------------------------------------------- router epilogue (gate G1) -- */

/* fp32 -> bf16, round-to-nearest-even (torch semantics) */
static inline uint16_t f32_to_bf16(float x) {
    union { float f; uint32_t u; } v = { x };
    uint32_t r = (v.u + 0x7FFF + ((v.u >> 16) & 1)) >> 16;
    return (uint16_t)r;
}

/* Deterministic top-k + softmax weights in ONE call, replacing the numpy
 * dispatch chain in e4b's cpu_router (the measured G1 p50 gap).
 *
 * Selection rule: descending by value, ties to the LOWER index — identical
 * to a stable descending sort. Output is in that order.
 * mode 0: weights = softmax over ALL logits, gathered at the selected ids,
 *         optionally renormalized (olmoe/qwen3). exp via expf.
 * mode 1: weights = softmax over the selected k logits (gpt_oss).
 * dense=1: first run logits = W @ x + bias (fp32, locked ascending-j
 *          scalar chain per row — E rows of K, tiny). */
EXPORT void gnf4_route_epilogue_bf16(const float *logits_in, int64_t T,
                                     int64_t E, int64_t k, int mode, int norm,
                                     int64_t *idx_out, int64_t idx_stride,
                                     uint16_t *wts_out, int64_t wts_stride) {
    for (int64_t t = 0; t < T; t++) {
        const float *lg = logits_in + t * E;
        int64_t *idx = idx_out + t * idx_stride;
        uint16_t *wts = wts_out + t * wts_stride;
        /* insertion top-k: scan ascending; strict > replaces, ties keep the
         * earlier index — equals stable-descending order */
        float bv[64];
        int64_t bi[64];
        int64_t kk = k > 64 ? 64 : k;
        int64_t used = 0;
        for (int64_t e = 0; e < E; e++) {
            float v = lg[e];
            if (used < kk) {
                int64_t p = used++;
                while (p > 0 && bv[p - 1] < v) {
                    bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--;
                }
                bv[p] = v; bi[p] = e;
            } else if (v > bv[kk - 1]) {
                int64_t p = kk - 1;
                while (p > 0 && bv[p - 1] < v) {
                    bv[p] = bv[p - 1]; bi[p] = bi[p - 1]; p--;
                }
                bv[p] = v; bi[p] = e;
            }
        }
        float w[64];
        if (mode == 0) {
            float m = lg[0];
            for (int64_t e = 1; e < E; e++) if (lg[e] > m) m = lg[e];
            float z = 0.0f;
            for (int64_t e = 0; e < E; e++) z += expf(lg[e] - m);
            for (int64_t j = 0; j < kk; j++) w[j] = expf(bv[j] - m) / z;
            if (norm) {
                float s = 0.0f;
                for (int64_t j = 0; j < kk; j++) s += w[j];
                for (int64_t j = 0; j < kk; j++) w[j] /= s;
            }
        } else {
            float m = bv[0];
            float z = 0.0f;
            for (int64_t j = 0; j < kk; j++) z += expf(bv[j] - m);
            for (int64_t j = 0; j < kk; j++) w[j] = expf(bv[j] - m) / z;
        }
        for (int64_t j = 0; j < kk; j++) {
            idx[j] = bi[j];
            wts[j] = f32_to_bf16(w[j]);
        }
    }
}

/* dense fp32 gemv for the router linear itself: logits[e] = sum_j W[e,j]*x[j]
 * (+bias). Locked ascending-j scalar chain — E*K is ~128K mults, DRAM-bound
 * on the weight read. Single-threaded on purpose (latency path). */
EXPORT void gnf4_dense_gemv_f32(const float *W, const float *x,
                                const float *bias, float *out, int64_t E,
                                int64_t K) {
    for (int64_t e = 0; e < E; e++) {
        const float *w = W + e * K;
        float acc[16] = {0};
        int64_t j = 0;
        for (; j + 16 <= K; j += 16)
            for (int i = 0; i < 16; i++)
                acc[i] += w[j + i] * x[j + i];
        float s = 0.0f;
        for (int i = 0; i < 16; i++) s += acc[i];
        for (; j < K; j++) s += w[j] * x[j];
        out[e] = bias ? s + bias[e] : s;
    }
}
