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

#define _GNU_SOURCE   /* cpu_set_t / CPU_SET / pthread_setaffinity_np / syscall */
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <pthread.h>
#include <sched.h>
#include <unistd.h>

#if defined(__linux__)
#include <linux/futex.h>
#include <sys/syscall.h>
#endif

#if defined(__x86_64__)
#include <immintrin.h>
#include <cpuid.h>
#endif

#ifdef _OPENMP
#include <omp.h>
#endif

#define EXPORT __attribute__((visibility("default")))

static inline void cpu_relax(void) {
#if defined(__x86_64__)
    _mm_pause();
#else
    sched_yield();
#endif
}

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

/* -------------------------------------------------- persistent pool ------ */
/* Executor-owned worker pool (Phase 3): pinned workers, spin-then-futex
 * idle, one job at a time, static contiguous partition (deterministic —
 * work items are independent, so partitioning cannot change bits). This
 * replaces the per-call OpenMP region whose fork/join and unpinned
 * placement were part of the G2 scaling wall. OpenMP remains the fallback
 * when the pool is not started. */

#define POOL_MAX 128

typedef void (*pool_fn)(int64_t lo, int64_t hi, void *arg);

static struct {
    pthread_t th[POOL_MAX];
    int cpu_of[POOL_MAX];
    int n;
    _Atomic uint32_t gen;
    _Atomic int done;
    _Atomic int stop;
    pool_fn fn;
    void *arg;
    int64_t total;
} P;

static void futex_wait_u32(_Atomic uint32_t *addr, uint32_t val) {
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAIT_PRIVATE, val, NULL, NULL, 0);
#else
    (void)addr; (void)val;
    sched_yield();
#endif
}

static void futex_wake_all_u32(_Atomic uint32_t *addr) {
#if defined(__linux__)
    syscall(SYS_futex, addr, FUTEX_WAKE_PRIVATE, 0x7fffffff, NULL, NULL, 0);
#else
    (void)addr;
#endif
}

/* minimal L3-domain topology (same sysfs source as the calibration bench):
 * spread worker w across domains for memory-channel coverage */
static int topo_spread_cpu(int w) {
    static int cpus[POOL_MAX];
    static int ncpu = -1;
    static int dom_of[POOL_MAX];
    static int ndom = 1;
    if (ncpu < 0) {
        char sigs[16][256];
        int nsig = 0;
        ncpu = 0;
        for (int c = 0; c < POOL_MAX; c++) {
            char path[128];
            snprintf(path, sizeof path,
                     "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", c);
            FILE *f = fopen(path, "r");
            if (!f) {
                if (c >= (int)sysconf(_SC_NPROCESSORS_ONLN)) break;
                cpus[ncpu] = c; dom_of[ncpu++] = 0;
                continue;
            }
            char sig[256] = {0};
            if (!fgets(sig, sizeof sig, f)) sig[0] = 0;
            fclose(f);
            int d = -1;
            for (int i = 0; i < nsig; i++)
                if (!strcmp(sigs[i], sig)) { d = i; break; }
            if (d < 0 && nsig < 16) { d = nsig; snprintf(sigs[nsig++], 256, "%s", sig); }
            cpus[ncpu] = c; dom_of[ncpu++] = d < 0 ? 0 : d;
        }
        ndom = nsig > 0 ? nsig : 1;
        if (ncpu == 0) { cpus[0] = 0; ncpu = 1; }
    }
    int dom = w % ndom, idx = w / ndom, seen = 0;
    for (int i = 0; i < ncpu; i++)
        if (dom_of[i] == dom && seen++ == idx) return cpus[i];
    return cpus[w % ncpu];
}

static void *pool_worker(void *idp) {
    int wid = (int)(intptr_t)idp;
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(P.cpu_of[wid], &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
    uint32_t seen = 0;
    for (;;) {
        uint32_t g = atomic_load_explicit(&P.gen, memory_order_acquire);
        if (atomic_load(&P.stop)) break;
        if (g == seen) {
            int spins = 0;
            while ((g = atomic_load_explicit(&P.gen, memory_order_acquire)) == seen
                   && !atomic_load(&P.stop)) {
                if (++spins > 20000) {          /* ~100 µs, then sleep */
                    futex_wait_u32(&P.gen, seen);
                    spins = 0;
                }
                cpu_relax();
            }
            if (atomic_load(&P.stop)) break;
        }
        seen = g;
        int64_t per = (P.total + P.n - 1) / P.n;
        int64_t lo = (int64_t)wid * per;
        int64_t hi = lo + per > P.total ? P.total : lo + per;
        if (lo < hi) P.fn(lo, hi, P.arg);
        atomic_fetch_add_explicit(&P.done, 1, memory_order_release);
    }
    return NULL;
}

EXPORT int gnf4_pool_start(int nthreads) {
    if (P.n) return P.n;                   /* already running */
    if (nthreads <= 0 || nthreads > POOL_MAX)
        nthreads = (int)sysconf(_SC_NPROCESSORS_ONLN);
    if (nthreads > POOL_MAX) nthreads = POOL_MAX;
    atomic_store(&P.stop, 0);
    atomic_store(&P.gen, 0);
    P.n = nthreads;                        /* workers read P.n per job */
    int created = 0;
    for (int w = 0; w < nthreads; w++) {
        P.cpu_of[w] = topo_spread_cpu(w);
        if (pthread_create(&P.th[w], NULL, pool_worker,
                           (void *)(intptr_t)w) != 0)
            break;                         /* out of threads/limits */
        created++;
    }
    if (created != nthreads) {
        /* a partial pool must not leave pool_run waiting on ghosts: shrink
         * to what exists, or tear down entirely when nothing started */
        P.n = created;
        if (created == 0) return 0;
    }
    return P.n;
}

EXPORT void gnf4_pool_stop(void) {
    if (!P.n) return;
    atomic_store(&P.stop, 1);
    atomic_fetch_add(&P.gen, 1);
    futex_wake_all_u32(&P.gen);
    for (int w = 0; w < P.n; w++) pthread_join(P.th[w], NULL);
    P.n = 0;
    atomic_store(&P.stop, 0);
}

EXPORT int gnf4_pool_size(void) { return P.n; }

static int pool_run(pool_fn fn, void *arg, int64_t total) {
    if (!P.n) return -1;
    P.fn = fn; P.arg = arg; P.total = total;
    atomic_store_explicit(&P.done, 0, memory_order_release);
    atomic_fetch_add_explicit(&P.gen, 1, memory_order_release);
    futex_wake_all_u32(&P.gen);
    /* The caller must not displace a worker: at nthreads == ncores a
     * hard-spinning caller costs one whole core and the job time goes
     * ~10x (measured 190 -> 17 GB/s at 48/48 on metal). Yield
     * periodically so the scheduler can run the worker underneath. */
    int s = 0;
    while (atomic_load_explicit(&P.done, memory_order_acquire) != P.n) {
        if (++s > 2000) { sched_yield(); s = 0; }
        cpu_relax();
    }
    return 0;
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

/* NOTE: a portable (scalar-loop) pre-dequant variant was tried and
 * REVERTED — on the AVX2 fallback the fused lane_step vectorizes better
 * than a wbuf store/reload split (T=32 regressed 175 -> 238 ms on the
 * dev Xeon). The hoist lives only in the AVX-512 paths, where the
 * dequantized vectors stay in zmm registers across the row loop. */

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

/* Row-batched step over a PRE-DEQUANTIZED weight vector. The dequant
 * (permute + scale multiply) is invariant across the T rows of a group,
 * so hoisting it halves the per-row vector work — the 632 us/row term
 * the rows-curve fit exposed (Phase 8 diag). w is the SAME value the
 * per-row path computed, so every lane's mul+add sees identical
 * operands in identical order: bit-exactness is preserved by
 * construction, and the ordered-ref tests hold it. */
static inline __m512 lane_step_w_avx512(__m512 w, const float *a,
                                        __m512 acc) {
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

/* Two output columns per pass: the activation loads (the only operand both
 * columns share) are issued once, and the two independent accumulator sets
 * widen the ILP window. Same locked tree per column, exactly. */
static void nf4_cell_t1_pair_avx512(const float *a, int64_t K,
                                    const uint8_t *w0, const float *am0,
                                    const uint8_t *w1, const float *am1,
                                    float *o0, float *o1) {
    __m512 a0 = _mm512_setzero_ps(), a1 = _mm512_setzero_ps();
    __m512 a2 = _mm512_setzero_ps(), a3 = _mm512_setzero_ps();
    __m512 b0 = _mm512_setzero_ps(), b1 = _mm512_setzero_ps();
    __m512 b2 = _mm512_setzero_ps(), b3 = _mm512_setzero_ps();
    const __m512 lutv = _mm512_loadu_ps(NF4_LUT);
    int64_t nblk = K / 64;
    for (int64_t b = 0; b < nblk; b++) {
        __m512 s0 = _mm512_mul_ps(lutv, _mm512_set1_ps(am0[b]));
        __m512 s1 = _mm512_mul_ps(lutv, _mm512_set1_ps(am1[b]));
        __m512i c0, c1, c2, c3, d0, d1, d2, d3;
        unpack32_avx512(w0 + b * 32, 1, &c0, &c1);
        unpack32_avx512(w0 + b * 32 + 16, 1, &c2, &c3);
        unpack32_avx512(w1 + b * 32, 1, &d0, &d1);
        unpack32_avx512(w1 + b * 32 + 16, 1, &d2, &d3);
        const float *ar = a + b * 64;
        __m512 av;
        av = _mm512_loadu_ps(ar);
        a0 = _mm512_add_ps(a0, _mm512_mul_ps(_mm512_permutexvar_ps(c0, s0), av));
        b0 = _mm512_add_ps(b0, _mm512_mul_ps(_mm512_permutexvar_ps(d0, s1), av));
        av = _mm512_loadu_ps(ar + 16);
        a1 = _mm512_add_ps(a1, _mm512_mul_ps(_mm512_permutexvar_ps(c1, s0), av));
        b1 = _mm512_add_ps(b1, _mm512_mul_ps(_mm512_permutexvar_ps(d1, s1), av));
        av = _mm512_loadu_ps(ar + 32);
        a2 = _mm512_add_ps(a2, _mm512_mul_ps(_mm512_permutexvar_ps(c2, s0), av));
        b2 = _mm512_add_ps(b2, _mm512_mul_ps(_mm512_permutexvar_ps(d2, s1), av));
        av = _mm512_loadu_ps(ar + 48);
        a3 = _mm512_add_ps(a3, _mm512_mul_ps(_mm512_permutexvar_ps(c3, s0), av));
        b3 = _mm512_add_ps(b3, _mm512_mul_ps(_mm512_permutexvar_ps(d3, s1), av));
    }
    float acc[4][16];
    _mm512_storeu_ps(acc[0], a0);
    _mm512_storeu_ps(acc[1], a1);
    _mm512_storeu_ps(acc[2], a2);
    _mm512_storeu_ps(acc[3], a3);
    *o0 = combine_acc(acc);
    _mm512_storeu_ps(acc[0], b0);
    _mm512_storeu_ps(acc[1], b1);
    _mm512_storeu_ps(acc[2], b2);
    _mm512_storeu_ps(acc[3], b3);
    *o1 = combine_acc(acc);
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
            /* dequant ONCE per block, reuse across the T rows */
            __m512 w0 = _mm512_mul_ps(_mm512_permutexvar_ps(c0, lutv), sc);
            __m512 w1 = _mm512_mul_ps(_mm512_permutexvar_ps(c1, lutv), sc);
            __m512 w2 = _mm512_mul_ps(_mm512_permutexvar_ps(c2, lutv), sc);
            __m512 w3 = _mm512_mul_ps(_mm512_permutexvar_ps(c3, lutv), sc);
            int64_t base = b * 64;
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                accv[r][0] = lane_step_w_avx512(w0, ar, accv[r][0]);
                accv[r][1] = lane_step_w_avx512(w1, ar + 16, accv[r][1]);
                accv[r][2] = lane_step_w_avx512(w2, ar + 32, accv[r][2]);
                accv[r][3] = lane_step_w_avx512(w3, ar + 48, accv[r][3]);
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

/* shared driver context: one work item = (group, 32-column tile) */
typedef struct {
    const float *a;
    const uint8_t *B;
    const float *absmax;
    const uint8_t *scales;
    const int64_t *eids;
    const int32_t *sizes;
    const int64_t *row_off;
    int64_t N, K, tiles_n;
    float *out;
    int use512;
} GemvCtx;

#define GEMV_TILE 32
/* accumulator rows per cell: 8 rows x 4 lane groups = 32 zmm registers,
 * i.e. the entire AVX-512 register file. Raising it spills. */
#define NF4_CELL_ROWS 8

static void nf4_range(int64_t lo, int64_t hi, void *argp) {
    GemvCtx *c = (GemvCtx *)argp;
    for (int64_t u = lo; u < hi; u++) {
        int g = (int)(u / c->tiles_n);
        int64_t tn = (u % c->tiles_n) * GEMV_TILE;
        int64_t tn_end = tn + GEMV_TILE > c->N ? c->N : tn + GEMV_TILE;
        int64_t e = c->eids[g];
        const uint8_t *w_e = c->B + e * c->N * (c->K / 2);
        const float *am_e = c->absmax + e * c->N * (c->K / 64);
        const float *a_g = c->a + c->row_off[g] * c->K;
        float *out_g = c->out + c->row_off[g] * c->N;
#if defined(__AVX512F__)
        if (c->use512 && c->sizes[g] == 1) {
            int64_t n = tn;
            for (; n + 1 < tn_end; n += 2)
                nf4_cell_t1_pair_avx512(
                    a_g, c->K,
                    w_e + n * (c->K / 2), am_e + n * (c->K / 64),
                    w_e + (n + 1) * (c->K / 2), am_e + (n + 1) * (c->K / 64),
                    &out_g[n], &out_g[n + 1]);
            for (; n < tn_end; n++)
                out_g[n] = nf4_cell_t1_avx512(a_g, c->K, w_e + n * (c->K / 2),
                                              am_e + n * (c->K / 64));
            continue;
        }
#endif
        /* Rows chunk INSIDE the work item (Phase 8). The register
         * blocking caps a cell at NF4_CELL_ROWS (8 accumulator rows x 4
         * lane groups = 32 zmm, the whole register file), but a group at
         * batch may hold more rows than that. Chunking here instead of
         * splitting the group in the caller is what makes the batch pay
         * off: the weight row is ~K/2 bytes, so after chunk 0 it is L1-hot
         * and chunks 1..n cost no DRAM traffic. Splitting into separate
         * GROUPS (the pre-Phase-8 behaviour) sent the chunks to different
         * work items — possibly different threads — and re-read the
         * weights from DRAM every time, which is exactly the amortization
         * G8 measures. */
        for (int64_t n = tn; n < tn_end; n++) {
            int32_t rem = c->sizes[g];
            int64_t r0 = 0;
            while (rem > 0) {
                int T = rem > NF4_CELL_ROWS ? NF4_CELL_ROWS : rem;
                nf4_cell(a_g + r0 * c->K, T, c->K, w_e + n * (c->K / 2),
                         am_e + n * (c->K / 64), out_g + r0 * c->N + n,
                         c->N, c->use512);
                r0 += T;
                rem -= T;
            }
        }
    }
}

static int gemv_common(GemvCtx *c, int G, int threads, pool_fn range,
                       int64_t *row_off_buf) {
    int64_t r0 = 0;
    for (int g = 0; g < G; g++) {
        /* group size is no longer capped at the cell's row blocking —
         * nf4_range/mx_range chunk internally (Phase 8) */
        if (c->sizes[g] <= 0) return -1;
        row_off_buf[g] = r0;
        r0 += c->sizes[g];
    }
    c->row_off = row_off_buf;
    c->tiles_n = (c->N + GEMV_TILE - 1) / GEMV_TILE;
    int64_t total = (int64_t)G * c->tiles_n;
    if (P.n) {                             /* executor pool, when started */
        pool_run(range, c, total);
        return 0;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < total; u++)
        range(u, u + 1, c);
    return 0;
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
    if (K % 64 || N <= 0 || G <= 0 || G > 512) return -1;
    int64_t row_off[512];
    GemvCtx c = { .a = a, .B = B, .absmax = absmax, .scales = NULL,
                  .eids = eids, .sizes = sizes, .N = N, .K = K, .out = out,
                  .use512 = (gnf4_cpu_features() & 16) != 0 };
    return gemv_common(&c, G, threads, nf4_range, row_off);
}

/* ------------------------------------------------------- fused NF4 FFN -- */
/* One pool job per MoE layer instead of two: gu GEMV -> silu(gate)*up ->
 * dn GEMV, chained inside a work item. The serving measurement that
 * motivates it: the pool's wake/join floor (~441 us cold at 32 workers)
 * is paid PER CALL, and the DRAM tier's decode cost is call-count-bound,
 * not bandwidth-bound (fixbox receipts, e4b bench/hybrid-g9/fixbox/).
 *
 * Partition changes from (group, column tile) to (group, row chunk):
 * the dn stage needs a row's FULL gu output, so a column-tiled item
 * cannot chain without cross-item waits. Each item owns <= NF4_CELL_ROWS
 * rows of one group and runs all three stages for them; the intermediate
 * never leaves the worker's thread-local scratch.
 *
 * Activation contract ("gnf4-silu-horner6/1"): silu computed as a LOCKED
 * f32 op sequence — clamp to [-87, 87], t = xc * -log2(e), n = floor
 * (t + 0.5), f = t - n, degree-6 Horner for 2^f, scale by the exact
 * power 2^n via exponent bits, sig = 1/(1+e), silu = x * sig. The numpy
 * spec mirrors the sequence op for op (same f32 rounding, no FMA — the
 * -ffp-contract=off this file already requires), so fused-vs-spec stays
 * EXACT equality, same bar as the GEMV tree. Scalar expf is not used:
 * it is ~10 ns/element, which at decode shapes would cost more than the
 * pool wake the fusion removes; and libm expf is not lockable across
 * hosts anyway. Accuracy of the polynomial is ~2e-8 relative on the
 * clamped domain — far below NF4 quantization noise. */

#define SILU_C6 1.535336188319500e-4f
#define SILU_C5 1.339887440266574e-3f
#define SILU_C4 9.618437357674640e-3f
#define SILU_C3 5.550332471162809e-2f
#define SILU_C2 2.402264791363012e-1f
#define SILU_C1 6.931472028550421e-1f
#define SILU_LOG2E_NEG -1.442695040888963e0f

static inline float silu_locked(float x) {
    /* piecewise tails: beyond the clamp true silu is x (resp. < 1.5e-36),
     * and computing x * sig(clamped) would return x * 1.7e-38 — garbage
     * for astronomically negative x. Exact x / exact 0 instead; the zero
     * tail also removes the subnormal-vs-FTZ corner from the contract. */
    if (x > 87.0f) return x;
    if (x < -87.0f) return 0.0f;
    float xc = x;
    float t = xc * SILU_LOG2E_NEG;
    float n = floorf(t + 0.5f);
    float f = t - n;
    float p = SILU_C6;
    p = p * f + SILU_C5;
    p = p * f + SILU_C4;
    p = p * f + SILU_C3;
    p = p * f + SILU_C2;
    p = p * f + SILU_C1;
    p = p * f + 1.0f;
    union { uint32_t u; float f32; } sc;
    sc.u = (uint32_t)((int32_t)n + 127) << 23;   /* n in [-126, 126] */
    float e = p * sc.f32;
    return x * (1.0f / (1.0f + e));
}

#if defined(__AVX512F__)
/* identical op sequence, 16 lanes at a time — lane l computes exactly
 * silu_locked(x[l]) (same clamp, same Horner order, mul+add not fma) */
static inline __m512 silu_locked_avx512(__m512 x) {
    __m512 xc = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-87.0f)),
                              _mm512_set1_ps(87.0f));
    __m512 t = _mm512_mul_ps(xc, _mm512_set1_ps(SILU_LOG2E_NEG));
    __m512 n = _mm512_roundscale_ps(
        _mm512_add_ps(t, _mm512_set1_ps(0.5f)),
        _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    __m512 f = _mm512_sub_ps(t, n);
    __m512 p = _mm512_set1_ps(SILU_C6);
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(SILU_C5));
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(SILU_C4));
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(SILU_C3));
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(SILU_C2));
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(SILU_C1));
    p = _mm512_add_ps(_mm512_mul_ps(p, f), _mm512_set1_ps(1.0f));
    __m512i sc = _mm512_slli_epi32(
        _mm512_add_epi32(_mm512_cvtps_epi32(n), _mm512_set1_epi32(127)), 23);
    __m512 e = _mm512_mul_ps(p, _mm512_castsi512_ps(sc));
    __m512 sig = _mm512_div_ps(
        _mm512_set1_ps(1.0f), _mm512_add_ps(_mm512_set1_ps(1.0f), e));
    __m512 r = _mm512_mul_ps(x, sig);
    /* same piecewise tails as the scalar path, mask-selected */
    r = _mm512_mask_mov_ps(r, _mm512_cmp_ps_mask(
            x, _mm512_set1_ps(87.0f), _CMP_GT_OQ), x);
    return _mm512_maskz_mov_ps(_mm512_cmp_ps_mask(
            x, _mm512_set1_ps(-87.0f), _CMP_GE_OQ), r);
}
#endif

/* per-thread scratch for the fused chain: NF4_CELL_ROWS rows of gu output
 * plus the activated half. Grows on demand, lives for the thread — pool
 * workers are persistent, so this is a one-time cost per worker, not a
 * leak that accumulates (and small: rows * (N_gu + H) floats). */
static __thread float *ffn_tls = NULL;
static __thread int64_t ffn_tls_cap = 0;

static float *ffn_scratch(int64_t floats) {
    if (ffn_tls_cap < floats) {
        free(ffn_tls);
        ffn_tls = (float *)malloc((size_t)floats * sizeof(float));
        ffn_tls_cap = ffn_tls ? floats : 0;
    }
    return ffn_tls;
}

typedef struct {
    const float *a;
    const uint8_t *B_gu; const float *am_gu;
    const uint8_t *B_dn; const float *am_dn;
    const int64_t *eids;
    const int32_t *sizes;
    const int64_t *row_off;      /* [G] first row of each group */
    const int64_t *item_off;     /* [G+1] first item of each group */
    int G;
    int64_t N_gu, K_gu, N_dn;    /* H = N_gu/2 is dn's K */
    float *out;
    int use512;
    volatile int err;            /* scratch OOM: refuse loudly, never
                                  * hand back torch.empty garbage */
} FfnCtx;

static void ffn_range(int64_t lo, int64_t hi, void *argp) {
    FfnCtx *c = (FfnCtx *)argp;
    int64_t H = c->N_gu / 2;
    int g = 0;
    for (int64_t u = lo; u < hi; u++) {
        while (c->item_off[g + 1] <= u) g++;     /* u ascends; g follows */
        int64_t chunk = u - c->item_off[g];
        int64_t r0 = c->row_off[g] + chunk * NF4_CELL_ROWS;
        int32_t left = c->sizes[g] - (int32_t)(chunk * NF4_CELL_ROWS);
        int T = left > NF4_CELL_ROWS ? NF4_CELL_ROWS : left;
        int64_t e = c->eids[g];
        const uint8_t *wgu = c->B_gu + e * c->N_gu * (c->K_gu / 2);
        const float *agu = c->am_gu + e * c->N_gu * (c->K_gu / 64);
        const uint8_t *wdn = c->B_dn + e * c->N_dn * (H / 2);
        const float *adn = c->am_dn + e * c->N_dn * (H / 64);
        const float *a_r = c->a + r0 * c->K_gu;
        float *gu = ffn_scratch(NF4_CELL_ROWS * (c->N_gu + H));
        if (!gu) { c->err = 1; return; }
        float *h = gu + NF4_CELL_ROWS * c->N_gu;
        int64_t n = 0;
#if defined(__AVX512F__)
        if (c->use512 && T == 1) {
            for (; n + 1 < c->N_gu; n += 2)
                nf4_cell_t1_pair_avx512(
                    a_r, c->K_gu,
                    wgu + n * (c->K_gu / 2), agu + n * (c->K_gu / 64),
                    wgu + (n + 1) * (c->K_gu / 2),
                    agu + (n + 1) * (c->K_gu / 64), &gu[n], &gu[n + 1]);
        }
#endif
        for (; n < c->N_gu; n++)
            nf4_cell(a_r, T, c->K_gu, wgu + n * (c->K_gu / 2),
                     agu + n * (c->K_gu / 64), gu + n, c->N_gu, c->use512);
        for (int r = 0; r < T; r++) {
            const float *row = gu + (int64_t)r * c->N_gu;
            float *hr = h + (int64_t)r * H;
            int64_t j = 0;
#if defined(__AVX512F__)
            if (c->use512)
                for (; j + 16 <= H; j += 16)
                    _mm512_storeu_ps(hr + j, _mm512_mul_ps(
                        silu_locked_avx512(_mm512_loadu_ps(row + j)),
                        _mm512_loadu_ps(row + H + j)));
#endif
            for (; j < H; j++)
                hr[j] = silu_locked(row[j]) * row[H + j];
        }
        n = 0;
#if defined(__AVX512F__)
        if (c->use512 && T == 1) {
            for (; n + 1 < c->N_dn; n += 2)
                nf4_cell_t1_pair_avx512(
                    h, H, wdn + n * (H / 2), adn + n * (H / 64),
                    wdn + (n + 1) * (H / 2), adn + (n + 1) * (H / 64),
                    &c->out[r0 * c->N_dn + n], &c->out[r0 * c->N_dn + n + 1]);
        }
#endif
        for (; n < c->N_dn; n++)
            nf4_cell(h, T, H, wdn + n * (H / 2), adn + n * (H / 64),
                     c->out + r0 * c->N_dn + n, c->N_dn, c->use512);
    }
}

/* a       [R_total, K_gu] fp32, rows sorted by group
 * B_gu    [E, N_gu, K_gu/2] u8, am_gu [E, N_gu, K_gu/64] f32
 * B_dn    [E, N_dn, H/2] u8,   am_dn [E, N_dn, H/64] f32, H = N_gu/2
 * out     [R_total, N_dn] fp32; gate = gu[:, :H], up = gu[:, H:]
 * returns 0, or -1 on bad shape */
EXPORT int gnf4_gemv_nf4_ffn_grouped(
        const float *a, const uint8_t *B_gu, const float *am_gu,
        const uint8_t *B_dn, const float *am_dn, const int64_t *eids,
        const int32_t *sizes, int G, int64_t N_gu, int64_t K_gu,
        int64_t N_dn, float *out, int threads) {
    if (K_gu % 64 || N_gu <= 0 || N_gu % 2 || (N_gu / 2) % 64 || N_dn <= 0)
        return -1;
    if (G <= 0 || G > 512) return -1;
    int64_t row_off[512], item_off[513];
    int64_t r0 = 0, items = 0;
    for (int g = 0; g < G; g++) {
        if (sizes[g] <= 0) return -1;
        row_off[g] = r0;
        item_off[g] = items;
        r0 += sizes[g];
        items += (sizes[g] + NF4_CELL_ROWS - 1) / NF4_CELL_ROWS;
    }
    item_off[G] = items;
    FfnCtx c = { .a = a, .B_gu = B_gu, .am_gu = am_gu, .B_dn = B_dn,
                 .am_dn = am_dn, .eids = eids, .sizes = sizes,
                 .row_off = row_off, .item_off = item_off, .G = G,
                 .N_gu = N_gu, .K_gu = K_gu, .N_dn = N_dn, .out = out,
                 .use512 = (gnf4_cpu_features() & 16) != 0, .err = 0 };
    if (P.n) {
        pool_run(ffn_range, &c, items);
        return c.err ? -2 : 0;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < items; u++)
        ffn_range(u, u + 1, &c);
    return c.err ? -2 : 0;
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
            /* dequant once per block, reuse across rows (see nf4_cell) */
            __m512 w0 = _mm512_mul_ps(_mm512_permutexvar_ps(c0, lutv), sc);
            __m512 w1 = _mm512_mul_ps(_mm512_permutexvar_ps(c1, lutv), sc);
            int64_t base = b * 32;
            /* two lane-groups per block; group parity alternates 0,1,2,3
             * across consecutive blocks: lane-group index = 2b, 2b+1 */
            for (int r = 0; r < T; r++) {
                const float *ar = a + (int64_t)r * K + base;
                int g0 = (int)((2 * b) & 3), g1 = (int)((2 * b + 1) & 3);
                accv[r][g0] = lane_step_w_avx512(w0, ar, accv[r][g0]);
                accv[r][g1] = lane_step_w_avx512(w1, ar + 16, accv[r][g1]);
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

static void mx_range(int64_t lo, int64_t hi, void *argp) {
    GemvCtx *c = (GemvCtx *)argp;
    for (int64_t u = lo; u < hi; u++) {
        int g = (int)(u / c->tiles_n);
        int64_t tn = (u % c->tiles_n) * GEMV_TILE;
        int64_t tn_end = tn + GEMV_TILE > c->N ? c->N : tn + GEMV_TILE;
        int64_t e = c->eids[g];
        const uint8_t *w_e = c->B + e * c->N * (c->K / 2);
        const uint8_t *sc_e = c->scales + e * c->N * (c->K / 32);
        const float *a_g = c->a + c->row_off[g] * c->K;
        float *out_g = c->out + c->row_off[g] * c->N;
#if defined(__AVX512F__)
        if (c->use512 && c->sizes[g] == 1) {
            for (int64_t n = tn; n < tn_end; n++)
                if (mx_cell_t1_avx512(a_g, c->K, w_e + n * (c->K / 2),
                                      sc_e + n * (c->K / 32), &out_g[n]))
                    mx_cell(a_g, 1, c->K, w_e + n * (c->K / 2),
                            sc_e + n * (c->K / 32), out_g + n, c->N, 0);
            continue;
        }
#endif
        for (int64_t n = tn; n < tn_end; n++) {   /* row chunking: see nf4_range */
            int32_t rem = c->sizes[g];
            int64_t r0 = 0;
            while (rem > 0) {
                int T = rem > NF4_CELL_ROWS ? NF4_CELL_ROWS : rem;
                mx_cell(a_g + r0 * c->K, T, c->K, w_e + n * (c->K / 2),
                        sc_e + n * (c->K / 32), out_g + r0 * c->N + n,
                        c->N, c->use512);
                r0 += T;
                rem -= T;
            }
        }
    }
}

EXPORT int gnf4_gemv_mxfp4_grouped(const float *a, const uint8_t *B,
                                   const uint8_t *scales, const int64_t *eids,
                                   const int32_t *sizes, int G, int64_t N,
                                   int64_t K, float *out, int threads) {
    if (K % 32 || N <= 0 || G <= 0 || G > 512) return -1;
    int64_t row_off[512];
    GemvCtx c = { .a = a, .B = B, .absmax = NULL, .scales = scales,
                  .eids = eids, .sizes = sizes, .N = N, .K = K, .out = out,
                  .use512 = (gnf4_cpu_features() & 16) != 0 };
    return gemv_common(&c, G, threads, mx_range, row_off);
}

/* ------------------------------------------------- grouped dgrad (CPU) -- */

/* Backward of the grouped GEMV for a FROZEN quantized base (hybrid Phase 5):
 *
 *     gi[t, k] = sum_n g[t, n] * w[n, k]
 *
 * — the same packed rows the forward read, consumed with TRANSPOSED access.
 * Each work item owns one (group, 128-column K-tile): a row's 128-element
 * slice is 64 packed NF4 bytes = exactly one cache line, so walking rows
 * within a K-tile reads whole lines with zero waste. Rows are dequantized
 * once into an L1-resident scratch tile ([DG_NTILE][DG_KTILE] fp32, 8 KB)
 * and every token in the group accumulates from that scratch — the packed
 * bytes are read ONCE however many tokens the group carries, and no
 * transposed copy of the weights ever exists (one-artifact invariant).
 *
 * Locked reduction order, mirrored by the executable spec
 * (`cpu_grouped.ordered_dgrad_ref`): for each (t, k) the contributions are
 * folded over n STRICTLY ASCENDING — n-tiles ascend, rows ascend inside a
 * tile, K-tiles are disjoint — one mul+add per n, never FMA:
 *     w = LUT[code] * scale;  p = w * g[t, n];  acc += p
 * (the scratch holds w, so reuse across tokens repeats the same fp32
 * values and the chain is unchanged).
 *
 * Group sizes are NOT capped at 8 here: a training microbatch parks many
 * rows on one expert, and the scratch reuse is precisely what makes large
 * T cost the same packed-byte traffic as T=1. */

#define DG_KTILE 128
#define DG_NTILE 16

typedef struct {
    const float *g;                        /* [R_total, N] grouped rows */
    const uint8_t *B;
    const float *absmax;                   /* nf4: [E, N, K/64] */
    const uint8_t *scales;                 /* mxfp4: [E, N, K/32] u8 e8m0 */
    const int64_t *eids;
    const int32_t *sizes;
    const int64_t *row_off;
    int64_t N, K, tiles_k;
    float *out;                            /* [R_total, K], caller-zeroed */
    int use512;
} DgradCtx;

/* one row's K-tile slice -> scratch (kw fp32 values), NF4 */
static inline void dg_row_nf4(const uint8_t *w_row, const float *am_row,
                              int64_t k0, int kw, float *dst, int use512) {
#if defined(__AVX512F__)
    if (use512) {
        __m512 lutv = _mm512_loadu_ps(NF4_LUT);
        for (int b = 0; b < kw / 64; b++) {
            int64_t blk = k0 / 64 + b;
            __m512 slut = _mm512_mul_ps(lutv, _mm512_set1_ps(am_row[blk]));
            const uint8_t *p = w_row + blk * 32;
            __m512i c0, c1, c2, c3;
            unpack32_avx512(p, 1, &c0, &c1);
            unpack32_avx512(p + 16, 1, &c2, &c3);
            _mm512_storeu_ps(dst + b * 64, _mm512_permutexvar_ps(c0, slut));
            _mm512_storeu_ps(dst + b * 64 + 16,
                             _mm512_permutexvar_ps(c1, slut));
            _mm512_storeu_ps(dst + b * 64 + 32,
                             _mm512_permutexvar_ps(c2, slut));
            _mm512_storeu_ps(dst + b * 64 + 48,
                             _mm512_permutexvar_ps(c3, slut));
        }
        return;
    }
#else
    (void)use512;
#endif
    {
        uint8_t codes[64];
        for (int b = 0; b < kw / 64; b++) {
            int64_t blk = k0 / 64 + b;
            float sc = am_row[blk];
            unpack32(w_row + blk * 32, 1, codes);
            unpack32(w_row + blk * 32 + 16, 1, codes + 32);
            for (int i = 0; i < 64; i++)
                dst[b * 64 + i] = NF4_LUT[codes[i]] * sc;
        }
    }
}

/* MXFP4 variant: 32-element e8m0 blocks; e==0xFF keeps the oracle's ldexp
 * semantics (w = ldexpf(LUT[c], 128); zeros stay zero) exactly like the
 * forward cell's scalar path. */
static inline void dg_row_mx(const uint8_t *w_row, const uint8_t *sc_row,
                             int64_t k0, int kw, float *dst, int use512) {
    for (int b = 0; b < kw / 32; b++) {
        int64_t blk = k0 / 32 + b;
        uint8_t e = sc_row[blk];
        const uint8_t *p = w_row + blk * 16;
        if (e == 0xFF) {
            uint8_t codes[32];
            unpack32(p, 0, codes);
            for (int i = 0; i < 32; i++)
                dst[b * 32 + i] = ldexpf(FP4_LUT[codes[i]], 128);
            continue;
        }
#if defined(__AVX512F__)
        if (use512) {
            __m512 slut = _mm512_mul_ps(_mm512_loadu_ps(FP4_LUT),
                                        _mm512_set1_ps(e8m0_scale(e)));
            __m512i c0, c1;
            unpack32_avx512(p, 0, &c0, &c1);
            _mm512_storeu_ps(dst + b * 32, _mm512_permutexvar_ps(c0, slut));
            _mm512_storeu_ps(dst + b * 32 + 16,
                             _mm512_permutexvar_ps(c1, slut));
            continue;
        }
#else
        (void)use512;
#endif
        {
            uint8_t codes[32];
            float sc = e8m0_scale(e);
            unpack32(p, 0, codes);
            for (int i = 0; i < 32; i++)
                dst[b * 32 + i] = FP4_LUT[codes[i]] * sc;
        }
    }
}

/* out_rows[t][k0..k0+kw) += sum over this n-tile's rows of
 * scratch[n] * g[t][n0+n]; ascending n, mul+add (contract=off holds). */
static void dg_accum(const float *scratch, int rows, int64_t n0,
                     const float *g_rows, int64_t T, int64_t N,
                     float *out_rows, int64_t K, int64_t k0, int kw,
                     int use512) {
#if defined(__AVX512F__)
    if (use512 && kw == DG_KTILE) {
        for (int64_t t = 0; t < T; t++) {
            float *o = out_rows + t * K + k0;
            const float *gr = g_rows + t * N + n0;
            __m512 a0 = _mm512_loadu_ps(o);
            __m512 a1 = _mm512_loadu_ps(o + 16);
            __m512 a2 = _mm512_loadu_ps(o + 32);
            __m512 a3 = _mm512_loadu_ps(o + 48);
            __m512 a4 = _mm512_loadu_ps(o + 64);
            __m512 a5 = _mm512_loadu_ps(o + 80);
            __m512 a6 = _mm512_loadu_ps(o + 96);
            __m512 a7 = _mm512_loadu_ps(o + 112);
            for (int n = 0; n < rows; n++) {
                __m512 gb = _mm512_set1_ps(gr[n]);
                const float *s = scratch + (int64_t)n * DG_KTILE;
                a0 = _mm512_add_ps(a0, _mm512_mul_ps(_mm512_loadu_ps(s), gb));
                a1 = _mm512_add_ps(a1,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 16), gb));
                a2 = _mm512_add_ps(a2,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 32), gb));
                a3 = _mm512_add_ps(a3,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 48), gb));
                a4 = _mm512_add_ps(a4,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 64), gb));
                a5 = _mm512_add_ps(a5,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 80), gb));
                a6 = _mm512_add_ps(a6,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 96), gb));
                a7 = _mm512_add_ps(a7,
                        _mm512_mul_ps(_mm512_loadu_ps(s + 112), gb));
            }
            _mm512_storeu_ps(o, a0);
            _mm512_storeu_ps(o + 16, a1);
            _mm512_storeu_ps(o + 32, a2);
            _mm512_storeu_ps(o + 48, a3);
            _mm512_storeu_ps(o + 64, a4);
            _mm512_storeu_ps(o + 80, a5);
            _mm512_storeu_ps(o + 96, a6);
            _mm512_storeu_ps(o + 112, a7);
        }
        return;
    }
#else
    (void)use512;
#endif
    for (int64_t t = 0; t < T; t++) {
        float *o = out_rows + t * K + k0;
        const float *gr = g_rows + t * N + n0;
        for (int n = 0; n < rows; n++) {
            float gv = gr[n];
            const float *s = scratch + (int64_t)n * DG_KTILE;
            for (int i = 0; i < kw; i++)
                o[i] += s[i] * gv;
        }
    }
}

static void dg_range_nf4(int64_t lo, int64_t hi, void *argp) {
    DgradCtx *c = (DgradCtx *)argp;
    float scratch[DG_NTILE * DG_KTILE];
    for (int64_t u = lo; u < hi; u++) {
        int gidx = (int)(u / c->tiles_k);
        int64_t k0 = (u % c->tiles_k) * DG_KTILE;
        int kw = (int)(k0 + DG_KTILE > c->K ? c->K - k0 : DG_KTILE);
        int64_t e = c->eids[gidx];
        int64_t T = c->sizes[gidx];
        const uint8_t *w_e = c->B + e * c->N * (c->K / 2);
        const float *am_e = c->absmax + e * c->N * (c->K / 64);
        const float *g_rows = c->g + c->row_off[gidx] * c->N;
        float *out_rows = c->out + c->row_off[gidx] * c->K;
        for (int64_t n0 = 0; n0 < c->N; n0 += DG_NTILE) {
            int rows = (int)(n0 + DG_NTILE > c->N ? c->N - n0 : DG_NTILE);
            for (int r = 0; r < rows; r++)
                dg_row_nf4(w_e + (n0 + r) * (c->K / 2),
                           am_e + (n0 + r) * (c->K / 64), k0, kw,
                           scratch + (int64_t)r * DG_KTILE, c->use512);
            dg_accum(scratch, rows, n0, g_rows, T, c->N, out_rows, c->K,
                     k0, kw, c->use512);
        }
    }
}

static void dg_range_mx(int64_t lo, int64_t hi, void *argp) {
    DgradCtx *c = (DgradCtx *)argp;
    float scratch[DG_NTILE * DG_KTILE];
    for (int64_t u = lo; u < hi; u++) {
        int gidx = (int)(u / c->tiles_k);
        int64_t k0 = (u % c->tiles_k) * DG_KTILE;
        int kw = (int)(k0 + DG_KTILE > c->K ? c->K - k0 : DG_KTILE);
        int64_t e = c->eids[gidx];
        int64_t T = c->sizes[gidx];
        const uint8_t *w_e = c->B + e * c->N * (c->K / 2);
        const uint8_t *sc_e = c->scales + e * c->N * (c->K / 32);
        const float *g_rows = c->g + c->row_off[gidx] * c->N;
        float *out_rows = c->out + c->row_off[gidx] * c->K;
        for (int64_t n0 = 0; n0 < c->N; n0 += DG_NTILE) {
            int rows = (int)(n0 + DG_NTILE > c->N ? c->N - n0 : DG_NTILE);
            for (int r = 0; r < rows; r++)
                dg_row_mx(w_e + (n0 + r) * (c->K / 2),
                          sc_e + (n0 + r) * (c->K / 32), k0, kw,
                          scratch + (int64_t)r * DG_KTILE, c->use512);
            dg_accum(scratch, rows, n0, g_rows, T, c->N, out_rows, c->K,
                     k0, kw, c->use512);
        }
    }
}

/* sizes have no 1..8 cap here — see the section comment */
static int dgrad_common(DgradCtx *c, int G, int threads, pool_fn range,
                        int64_t *row_off_buf) {
    int64_t r0 = 0;
    for (int g = 0; g < G; g++) {
        if (c->sizes[g] <= 0) return -1;
        row_off_buf[g] = r0;
        r0 += c->sizes[g];
    }
    c->row_off = row_off_buf;
    c->tiles_k = (c->K + DG_KTILE - 1) / DG_KTILE;
    int64_t total = (int64_t)G * c->tiles_k;
    if (P.n) {
        pool_run(range, c, total);
        return 0;
    }
#ifdef _OPENMP
    omp_set_num_threads(threads > 0 ? threads : omp_get_max_threads());
#endif
#pragma omp parallel for schedule(static)
    for (int64_t u = 0; u < total; u++)
        range(u, u + 1, c);
    return 0;
}

/* g        [R_total, N] fp32, rows sorted by group
 * B        [E, N, K/2] u8 · absmax [E, N, K/64] fp32
 * out      [R_total, K] fp32, ZERO-INITIALIZED by the caller
 * returns 0, or -1 on bad shape */
EXPORT int gnf4_dgrad_nf4_grouped(const float *g, const uint8_t *B,
                                  const float *absmax, const int64_t *eids,
                                  const int32_t *sizes, int G, int64_t N,
                                  int64_t K, float *out, int threads) {
    if (K % 64 || N <= 0 || G <= 0 || G > 512) return -1;
    int64_t row_off[512];
    DgradCtx c = { .g = g, .B = B, .absmax = absmax, .scales = NULL,
                   .eids = eids, .sizes = sizes, .N = N, .K = K, .out = out,
                   .use512 = (gnf4_cpu_features() & 16) != 0 };
    return dgrad_common(&c, G, threads, dg_range_nf4, row_off);
}

EXPORT int gnf4_dgrad_mxfp4_grouped(const float *g, const uint8_t *B,
                                    const uint8_t *scales,
                                    const int64_t *eids, const int32_t *sizes,
                                    int G, int64_t N, int64_t K, float *out,
                                    int threads) {
    if (K % 32 || N <= 0 || G <= 0 || G > 512) return -1;
    int64_t row_off[512];
    DgradCtx c = { .g = g, .B = B, .absmax = NULL, .scales = scales,
                   .eids = eids, .sizes = sizes, .N = N, .K = K, .out = out,
                   .use512 = (gnf4_cpu_features() & 16) != 0 };
    return dgrad_common(&c, G, threads, dg_range_mx, row_off);
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
