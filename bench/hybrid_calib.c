/* Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
 *
 * hybrid_calib.c — CPU-side calibration microbench for the hybrid CPU/GPU
 * tier (Phase 0, gate G0). Measures on the box it runs on, never spec sheets:
 *
 *   1. STREAM triad DRAM bandwidth — regular AND non-temporal stores, thread
 *      ladder, compact-vs-spread L3(CCD) pinning, first-touch by the same
 *      partition that later runs the timed loop.
 *   2. The G0 gate workload: grouped scattered per-expert reads — k distinct
 *      MB-scale contiguous blocks per fetch, chosen by a fixed-seed routing
 *      trace over E experts, read-reduced. Thread and block-size sweep.
 *      Gate = best scatter GB/s as a fraction of best triad GB/s.
 *   3. O_DIRECT NVMe sequential + random reads (falls back to a page-cache
 *      path with fadvise(DONTNEED) where O_DIRECT is unsupported, e.g.
 *      overlayfs in rented containers — the receipt names which path ran).
 *
 * Plain C11 + pthreads, Linux-only (pthread barriers, sched affinity,
 * MADV_HUGEPAGE, O_DIRECT). No external deps. JSON on stdout, notes on
 * stderr. Built by bench/calibrate.py with `cc -O3 -march=native` on the
 * target, so ISA selection is a compile-time fact of the measured box; the
 * SIMD paths compile out cleanly elsewhere (scalar fallback).
 *
 * Method notes (each earned its place):
 *   - First-touch init runs under the same thread/pin partition as the timed
 *     loop, so pages land NUMA/CCD-local to their reader. The scatter arena
 *     is first-touched parallel-spread — the placement a NUMA-aware loader
 *     would produce; single-thread touch would pin 100% of pages to one node
 *     and understate multi-CCD read bandwidth.
 *   - Read loops accumulate an xor sink published per thread — defeats DCE.
 *   - Timing: barrier -> t0 -> work -> barrier -> t1, per rep; median of reps.
 *   - Byte counting is STREAM-convention: triad counts 24 B/element (2 reads
 *     + 1 write); regular-store RFO traffic is NOT added, so the NT-store
 *     variant may legitimately report higher. Scatter counts pure read bytes
 *     (k * block_bytes per fetch).
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <stdatomic.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <pthread.h>
#include <time.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sched.h>

#if defined(__x86_64__)
#include <immintrin.h>
#include <cpuid.h>
#endif

/* ---------------------------------------------------------------- utils -- */

static double now_s(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC_RAW, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static void die(const char *msg) {
    fprintf(stderr, "hybrid_calib: %s (errno=%s)\n", msg, strerror(errno));
    exit(1);
}

static uint64_t xs(uint64_t *s) {           /* xorshift64* */
    uint64_t x = *s;
    x ^= x >> 12; x ^= x << 25; x ^= x >> 27;
    *s = x;
    return x * 0x2545F4914F6CDD1DULL;
}

static int cmp_d(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return (x > y) - (x < y);
}
static double median_d(double *v, int n) {
    qsort(v, (size_t)n, sizeof *v, cmp_d);
    return (n & 1) ? v[n / 2] : 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

/* ------------------------------------------------------- cpu detection -- */

typedef struct { int avx2, avx512f, avx512bw, avx512vl, avx512vbmi, avx512vnni; } CpuFeat;

static CpuFeat detect_features(void) {
    CpuFeat f = {0};
#if defined(__x86_64__)
    unsigned a, b, c, d;
    if (__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        f.avx2       = (b >> 5) & 1;
        f.avx512f    = (b >> 16) & 1;
        f.avx512bw   = (b >> 30) & 1;
        f.avx512vl   = (unsigned)(b >> 31) & 1;
        f.avx512vbmi = (c >> 1) & 1;
        f.avx512vnni = (c >> 11) & 1;
    }
#endif
    return f;
}

/* L3 topology: group online CPUs by cache/index3/shared_cpu_list; each
 * distinct list is one L3 domain (a CCD on AMD). */
#define MAX_DOMAINS 64
#define MAX_CPUS 1024

typedef struct {
    int ncpus, ndomains;
    int cpu_ids[MAX_CPUS];
    int domain_of[MAX_CPUS];
    char sig[MAX_DOMAINS][256];
} Topo;

static Topo topo_read(void) {
    Topo t; memset(&t, 0, sizeof t);
    for (int cpu = 0; cpu < MAX_CPUS; cpu++) {
        char p[288];
        struct stat st;
        snprintf(p, sizeof p, "/sys/devices/system/cpu/cpu%d", cpu);
        if (stat(p, &st) != 0) break;                 /* no such cpu: done */
        snprintf(p, sizeof p,
                 "/sys/devices/system/cpu/cpu%d/cache/index3/shared_cpu_list", cpu);
        FILE *fp = fopen(p, "r");
        char sig[256] = "one-domain";
        if (fp) {
            if (!fgets(sig, sizeof sig, fp)) snprintf(sig, sizeof sig, "one-domain");
            fclose(fp);
            sig[strcspn(sig, "\n")] = 0;
        }
        int dom = -1;
        for (int i = 0; i < t.ndomains; i++)
            if (strcmp(t.sig[i], sig) == 0) { dom = i; break; }
        if (dom < 0 && t.ndomains < MAX_DOMAINS) {
            dom = t.ndomains++;
            snprintf(t.sig[dom], 256, "%s", sig);
        }
        if (t.ncpus < MAX_CPUS) {
            t.cpu_ids[t.ncpus] = cpu;
            t.domain_of[t.ncpus] = dom < 0 ? 0 : dom;
            t.ncpus++;
        }
    }
    if (t.ncpus == 0) {
        long n = sysconf(_SC_NPROCESSORS_ONLN);
        if (n < 1) n = 1;
        if (n > MAX_CPUS) n = MAX_CPUS;
        for (int i = 0; i < (int)n; i++) { t.cpu_ids[i] = i; t.domain_of[i] = 0; }
        t.ncpus = (int)n;
    }
    if (t.ndomains == 0) t.ndomains = 1;
    return t;
}

/* worker w of n. mode 0 = compact (cpu-number order: fills one CCD, then the
 * next, on the usual EPYC/TR enumeration), mode 1 = spread (round-robin
 * across L3 domains). */
static int pick_cpu(const Topo *t, int w, int mode) {
    if (mode == 0 || t->ndomains <= 1) return t->cpu_ids[w % t->ncpus];
    int dom = w % t->ndomains, idx = w / t->ndomains, seen = 0;
    for (int i = 0; i < t->ncpus; i++)
        if (t->domain_of[i] == dom && seen++ == idx) return t->cpu_ids[i];
    return t->cpu_ids[w % t->ncpus];
}

static void pin_self(const Topo *t, int w, int mode) {
    cpu_set_t set;
    CPU_ZERO(&set);
    CPU_SET(pick_cpu(t, w, mode), &set);
    (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

static void *big_alloc(size_t bytes) {
    void *p = mmap(NULL, bytes, PROT_READ | PROT_WRITE,
                   MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (p == MAP_FAILED) return NULL;
    (void)madvise(p, bytes, MADV_HUGEPAGE);
    return p;
}

/* -------------------------------------------------------------- shared -- */

typedef struct {
    pthread_barrier_t *bar;
    const Topo *topo;
    int nthreads, pin_mode, wid, reps;
    double *rep_secs;                 /* shared; wid 0 writes rep r */
    double t0;                        /* wid 0 scratch */
    /* triad */
    double *A, *B, *C;
    size_t n_elem;
    int nt_stores;
    /* scatter */
    uint8_t *arena;
    size_t block_bytes;
    int topk, n_fetches, tiles_per_block;
    const int *trace;
    _Atomic long *cursor;
    long total_units;
    uint64_t sink;
} Job;

/* --- triad ------------------------------------------------------------- */

static void triad_range(double *A, const double *B, const double *C,
                        size_t lo, size_t hi, int nt) {
    const double s = 3.0;
    size_t i = lo;
#if defined(__AVX512F__)
    __m512d vs = _mm512_set1_pd(s);
    if (nt) for (; i + 8 <= hi; i += 8)
        _mm512_stream_pd(A + i, _mm512_fmadd_pd(vs, _mm512_loadu_pd(C + i),
                                                _mm512_loadu_pd(B + i)));
    else for (; i + 8 <= hi; i += 8)
        _mm512_storeu_pd(A + i, _mm512_fmadd_pd(vs, _mm512_loadu_pd(C + i),
                                                _mm512_loadu_pd(B + i)));
#elif defined(__AVX2__)
    __m256d vs = _mm256_set1_pd(s);
    if (nt) for (; i + 4 <= hi; i += 4)
        _mm256_stream_pd(A + i, _mm256_fmadd_pd(vs, _mm256_loadu_pd(C + i),
                                                _mm256_loadu_pd(B + i)));
    else for (; i + 4 <= hi; i += 4)
        _mm256_storeu_pd(A + i, _mm256_fmadd_pd(vs, _mm256_loadu_pd(C + i),
                                                _mm256_loadu_pd(B + i)));
#endif
    for (; i < hi; i++) A[i] = B[i] + s * C[i];
#if defined(__x86_64__)
    if (nt) _mm_sfence();
#else
    (void)nt;
#endif
}

static void *triad_worker(void *arg) {
    Job *j = (Job *)arg;
    pin_self(j->topo, j->wid, j->pin_mode);
    /* Chunk boundaries stay 8-element (64 B) aligned: NT stores fault on
     * unaligned addresses (_mm256/_mm512_stream_pd), and mmap'd arrays are
     * page-aligned, so aligning lo aligns every vector store. */
    size_t chunk = (j->n_elem / (size_t)j->nthreads) & ~(size_t)7;
    size_t lo = (size_t)j->wid * chunk;
    size_t hi = (j->wid == j->nthreads - 1) ? j->n_elem : lo + chunk;
    for (size_t i = lo; i < hi; i++) { j->A[i] = 1.0; j->B[i] = 2.0; j->C[i] = 0.5; }
    pthread_barrier_wait(j->bar);                       /* init done */
    triad_range(j->A, j->B, j->C, lo, hi, j->nt_stores);/* warmup rep */
    pthread_barrier_wait(j->bar);
    for (int r = 0; r < j->reps; r++) {
        pthread_barrier_wait(j->bar);
        if (j->wid == 0) j->t0 = now_s();
        pthread_barrier_wait(j->bar);
        triad_range(j->A, j->B, j->C, lo, hi, j->nt_stores);
        pthread_barrier_wait(j->bar);
        if (j->wid == 0) j->rep_secs[r] = now_s() - j->t0;
    }
    return NULL;
}

/* Arrays are allocated fresh per config so first-touch places pages under
 * the SAME thread partition that then times them. Reusing one allocation
 * across configs would let the first config (1 thread) own every page's
 * NUMA placement and understate all later multi-thread rows. */
static double run_triad(const Topo *topo, size_t n_elem, int nthreads,
                        int pin_mode, int nt, int reps) {
    double *A = big_alloc(n_elem * 8), *B = big_alloc(n_elem * 8),
           *C = big_alloc(n_elem * 8);
    if (!A || !B || !C) die("triad alloc failed");
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)nthreads);
    double rep_secs[64];
    if (reps > 64) reps = 64;
    Job *jobs = calloc((size_t)nthreads, sizeof(Job));
    pthread_t *th = calloc((size_t)nthreads, sizeof(pthread_t));
    for (int w = 0; w < nthreads; w++)
        jobs[w] = (Job){ .bar = &bar, .topo = topo, .nthreads = nthreads,
                         .pin_mode = pin_mode, .wid = w, .reps = reps,
                         .rep_secs = rep_secs,
                         .A = A, .B = B, .C = C, .n_elem = n_elem,
                         .nt_stores = nt };
    for (int w = 0; w < nthreads; w++)
        pthread_create(&th[w], NULL, triad_worker, &jobs[w]);
    for (int w = 0; w < nthreads; w++) pthread_join(th[w], NULL);
    pthread_barrier_destroy(&bar);
    free(jobs); free(th);
    munmap(A, n_elem * 8); munmap(B, n_elem * 8); munmap(C, n_elem * 8);
    double dt = median_d(rep_secs, reps);
    return (double)n_elem * 24.0 / dt / 1e9;
}

/* --- scatter ----------------------------------------------------------- */

static uint64_t read_block(const uint8_t *p, size_t bytes) {
    uint64_t acc = 0;
    size_t i = 0;
#if defined(__AVX512F__)
    __m512i v = _mm512_setzero_si512();
    for (; i + 256 <= bytes; i += 256) {
        v = _mm512_xor_si512(v, _mm512_loadu_si512((const void *)(p + i)));
        v = _mm512_xor_si512(v, _mm512_loadu_si512((const void *)(p + i + 64)));
        v = _mm512_xor_si512(v, _mm512_loadu_si512((const void *)(p + i + 128)));
        v = _mm512_xor_si512(v, _mm512_loadu_si512((const void *)(p + i + 192)));
    }
    acc ^= (uint64_t)_mm512_reduce_add_epi64(v);
#elif defined(__AVX2__)
    __m256i v = _mm256_setzero_si256();
    for (; i + 128 <= bytes; i += 128) {
        v = _mm256_xor_si256(v, _mm256_loadu_si256((const void *)(p + i)));
        v = _mm256_xor_si256(v, _mm256_loadu_si256((const void *)(p + i + 32)));
        v = _mm256_xor_si256(v, _mm256_loadu_si256((const void *)(p + i + 64)));
        v = _mm256_xor_si256(v, _mm256_loadu_si256((const void *)(p + i + 96)));
    }
    uint64_t tmp[4];
    _mm256_storeu_si256((__m256i *)tmp, v);
    acc ^= tmp[0] ^ tmp[1] ^ tmp[2] ^ tmp[3];
#endif
    const uint64_t *q = (const uint64_t *)(p + i);
    for (; i + 8 <= bytes; i += 8) acc ^= *q++;
    return acc;
}

/* Work units are (fetch, expert-slot, tile) triples handed out by an atomic
 * cursor in trace order: any thread count balances, the access pattern stays
 * the trace's. Tiles cut blocks into 1 MiB chunks so threads cooperate
 * within a block too. */
#define TILE_BYTES ((size_t)1 << 20)

static void *scatter_worker(void *arg) {
    Job *j = (Job *)arg;
    pin_self(j->topo, j->wid, j->pin_mode);
    pthread_barrier_wait(j->bar);
    if (j->wid == 0) j->rep_secs[0] = now_s();          /* t_start slot */
    pthread_barrier_wait(j->bar);
    uint64_t sink = 0;
    long per_fetch = (long)j->topk * j->tiles_per_block;
    for (;;) {
        long u = atomic_fetch_add_explicit(j->cursor, 1, memory_order_relaxed);
        if (u >= j->total_units) break;
        long fetch = u / per_fetch, rem = u % per_fetch;
        int slot = (int)(rem / j->tiles_per_block);
        int tile = (int)(rem % j->tiles_per_block);
        int eid  = j->trace[fetch * j->topk + slot];
        size_t off = (size_t)tile * TILE_BYTES;
        size_t n = off + TILE_BYTES > j->block_bytes ? j->block_bytes - off
                                                     : TILE_BYTES;
        sink ^= read_block(j->arena + (size_t)eid * j->block_bytes + off, n);
    }
    pthread_barrier_wait(j->bar);
    if (j->wid == 0) j->rep_secs[1] = now_s();          /* t_stop slot */
    j->sink = sink;
    pthread_barrier_wait(j->bar);
    return NULL;
}

static double run_scatter(const Topo *topo, uint8_t *arena, size_t block_bytes,
                          int topk, int n_fetches, const int *trace,
                          int nthreads, int pin_mode, uint64_t *sink_out) {
    pthread_barrier_t bar;
    pthread_barrier_init(&bar, NULL, (unsigned)nthreads);
    double marks[2] = {0, 0};
    _Atomic long cursor = 0;
    int tiles = (int)((block_bytes + TILE_BYTES - 1) / TILE_BYTES);
    long total_units = (long)n_fetches * topk * tiles;
    Job *jobs = calloc((size_t)nthreads, sizeof(Job));
    pthread_t *th = calloc((size_t)nthreads, sizeof(pthread_t));
    for (int w = 0; w < nthreads; w++)
        jobs[w] = (Job){ .bar = &bar, .topo = topo, .nthreads = nthreads,
                         .pin_mode = pin_mode, .wid = w, .rep_secs = marks,
                         .arena = arena, .block_bytes = block_bytes,
                         .topk = topk, .n_fetches = n_fetches,
                         .tiles_per_block = tiles, .trace = trace,
                         .cursor = &cursor, .total_units = total_units };
    for (int w = 0; w < nthreads; w++)
        pthread_create(&th[w], NULL, scatter_worker, &jobs[w]);
    uint64_t sink = 0;
    for (int w = 0; w < nthreads; w++) {
        pthread_join(th[w], NULL);
        sink ^= jobs[w].sink;
    }
    pthread_barrier_destroy(&bar);
    free(jobs); free(th);
    *sink_out ^= sink;
    double bytes = (double)n_fetches * (double)topk * (double)block_bytes;
    return bytes / (marks[1] - marks[0]) / 1e9;
}

/* parallel-spread first-touch + fill of the arena */
typedef struct {
    const Topo *topo;
    uint8_t *arena;
    size_t lo, hi;
    int wid;
} FillJob;

static void *fill_worker(void *arg) {
    FillJob *f = (FillJob *)arg;
    pin_self(f->topo, f->wid, 1 /* spread */);
    uint64_t s = 0x9E3779B97F4A7C15ULL ^ (uint64_t)f->wid;
    uint64_t *p = (uint64_t *)(f->arena + f->lo);
    size_t n = (f->hi - f->lo) / 8;
    for (size_t i = 0; i < n; i++) p[i] = xs(&s);
    return NULL;
}

/* --- NVMe -------------------------------------------------------------- */

typedef struct {
    const char *path;
    size_t file_bytes, io_bytes;
    int rand_mode, nthreads, wid, direct;
    double gbs;
    uint64_t seed;
} NvmeJob;

static void *nvme_worker(void *arg) {
    NvmeJob *nj = (NvmeJob *)arg;
    int fd = open(nj->path, O_RDONLY | (nj->direct ? O_DIRECT : 0));
    if (fd < 0) { nj->gbs = -1; return NULL; }
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, nj->io_bytes)) { close(fd); nj->gbs = -1; return NULL; }
    size_t span = nj->file_bytes / (size_t)nj->nthreads;
    size_t base = (size_t)nj->wid * span;
    size_t nio = span / nj->io_bytes;
    if (nio == 0) nio = 1;
    uint64_t s = nj->seed + (uint64_t)nj->wid * 0x9E3779B97F4A7C15ULL;
    uint64_t sink = 0;
    double t0 = now_s();
    for (size_t i = 0; i < nio; i++) {
        off_t off = (off_t)(nj->rand_mode
                            ? base + (xs(&s) % nio) * nj->io_bytes
                            : base + i * nj->io_bytes);
        ssize_t got = pread(fd, buf, nj->io_bytes, off);
        if (got <= 0) break;
        sink ^= ((uint64_t *)buf)[0];
    }
    double t1 = now_s();
    if (sink == 0x1ULL) fprintf(stderr, ".");           /* keep sink live */
    nj->gbs = (double)nio * (double)nj->io_bytes / (t1 - t0) / 1e9;
    free(buf); close(fd);
    return NULL;
}

static int nvme_prepare(const char *file, size_t file_bytes) {
    struct stat st;
    if (stat(file, &st) == 0 && (size_t)st.st_size >= file_bytes) return 0;
    int fd = open(file, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) return -1;
    size_t chunk = (size_t)16 << 20;
    void *buf = NULL;
    if (posix_memalign(&buf, 4096, chunk)) { close(fd); return -1; }
    uint64_t s = 0xC0FFEE;
    for (size_t i = 0; i < chunk / 8; i++) ((uint64_t *)buf)[i] = xs(&s);
    int rc = 0;
    for (size_t off = 0; off < file_bytes && rc == 0; off += chunk)
        if (write(fd, buf, chunk) != (ssize_t)chunk) rc = -1;
    fsync(fd); close(fd); free(buf);
    return rc;
}

static void drop_cache(const char *file) {
    int fd = open(file, O_RDONLY);
    if (fd >= 0) { (void)posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED); close(fd); }
}

/* ---------------------------------------------------------------- main -- */

int main(int argc, char **argv) {
    size_t triad_gib = 2, arena_gib = 8, nvme_gib = 8, io_mib = 1;
    int topk = 8, n_fetches = 400, reps = 5;
    uint64_t seed = 20260816;
    const char *nvme_dir = NULL;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--triad-gib") && i + 1 < argc) triad_gib = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--arena-gib") && i + 1 < argc) arena_gib = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--topk") && i + 1 < argc) topk = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fetches") && i + 1 < argc) n_fetches = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--reps") && i + 1 < argc) reps = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) seed = (uint64_t)atoll(argv[++i]);
        else if (!strcmp(argv[i], "--nvme-dir") && i + 1 < argc) nvme_dir = argv[++i];
        else if (!strcmp(argv[i], "--nvme-gib") && i + 1 < argc) nvme_gib = (size_t)atol(argv[++i]);
        else if (!strcmp(argv[i], "--io-mib") && i + 1 < argc) io_mib = (size_t)atol(argv[++i]);
        else { fprintf(stderr, "unknown arg %s\n", argv[i]); return 2; }
    }

    CpuFeat feat = detect_features();
    Topo topo = topo_read();

    printf("{\n");
    printf("  \"features\": {\"avx2\": %d, \"avx512f\": %d, \"avx512bw\": %d, "
           "\"avx512vl\": %d, \"avx512vbmi\": %d, \"avx512vnni\": %d},\n",
           feat.avx2, feat.avx512f, feat.avx512bw, feat.avx512vl,
           feat.avx512vbmi, feat.avx512vnni);
    printf("  \"topology\": {\"online_cpus\": %d, \"l3_domains\": %d},\n",
           topo.ncpus, topo.ndomains);
    printf("  \"compiled_simd\": \"%s\",\n",
#if defined(__AVX512F__)
           "avx512"
#elif defined(__AVX2__)
           "avx2"
#else
           "scalar"
#endif
    );
    fflush(stdout);

    /* thread ladder: powers of two + all cores + one-full-CCD */
    int tset[20]; int ntset = 0;
    for (int t = 1; t <= topo.ncpus && ntset < 16; t *= 2) tset[ntset++] = t;
    if (tset[ntset - 1] != topo.ncpus) tset[ntset++] = topo.ncpus;
    int per_dom = topo.ncpus / topo.ndomains;
    if (per_dom > 1) {
        int have = 0;
        for (int i = 0; i < ntset; i++) if (tset[i] == per_dom) have = 1;
        if (!have && ntset < 20) tset[ntset++] = per_dom;
    }

    /* ------------------------------------------------ STREAM triad sweep */
    size_t n_elem = triad_gib * ((size_t)1 << 30) / 8 / 3;
    n_elem -= n_elem % 64;

    double best_triad = 0; int bt_thr = 0, bt_pin = 0, bt_nt = 0;
    printf("  \"triad\": [\n");
    int first = 1;
    for (int nt = 0; nt < 2; nt++)
        for (int pin = 0; pin < 2; pin++) {
            if (topo.ndomains == 1 && pin == 1) continue;
            for (int ti = 0; ti < ntset; ti++) {
                double g = run_triad(&topo, n_elem, tset[ti], pin, nt, reps);
                printf("%s    {\"threads\": %d, \"pin\": \"%s\", \"nt\": %d, \"gbs\": %.2f}",
                       first ? "" : ",\n", tset[ti], pin ? "spread" : "compact", nt, g);
                first = 0;
                fflush(stdout);
                if (g > best_triad) { best_triad = g; bt_thr = tset[ti]; bt_pin = pin; bt_nt = nt; }
            }
        }
    printf("\n  ],\n");
    printf("  \"triad_best\": {\"gbs\": %.2f, \"threads\": %d, \"pin\": \"%s\", \"nt\": %d},\n",
           best_triad, bt_thr, bt_pin ? "spread" : "compact", bt_nt);
    fflush(stdout);

    /* ------------------------------------------- grouped scatter (gate) */
    size_t arena_bytes = arena_gib << 30;
    uint8_t *arena = big_alloc(arena_bytes);
    if (!arena) die("arena alloc failed");
    {
        int nf = topo.ncpus < 16 ? topo.ncpus : 16;
        FillJob *fj = calloc((size_t)nf, sizeof(FillJob));
        pthread_t *th = calloc((size_t)nf, sizeof(pthread_t));
        size_t chunk = arena_bytes / (size_t)nf;
        chunk -= chunk % 4096;
        for (int w = 0; w < nf; w++) {
            fj[w] = (FillJob){ .topo = &topo, .arena = arena,
                               .lo = (size_t)w * chunk,
                               .hi = w == nf - 1 ? arena_bytes : (size_t)(w + 1) * chunk,
                               .wid = w };
            pthread_create(&th[w], NULL, fill_worker, &fj[w]);
        }
        for (int w = 0; w < nf; w++) pthread_join(th[w], NULL);
        free(fj); free(th);
    }

    size_t blocks[5] = { (size_t)2 << 20, (size_t)4 << 20, (size_t)8 << 20,
                         (size_t)16 << 20, (size_t)32 << 20 };
    double best_scatter = 0; size_t bs_block = 0; int bs_thr = 0, bs_E = 0;
    uint64_t sink = 0;
    printf("  \"scatter\": [\n");
    first = 1;
    for (int bi = 0; bi < 5; bi++) {
        size_t bb = blocks[bi];
        long e_fit = (long)(arena_bytes / bb);
        int E = e_fit > 4096 ? 4096 : (int)e_fit;
        if (E < 128) continue;   /* gate calls for 128+ experts */
        uint64_t ts = seed;
        int *trace = malloc((size_t)n_fetches * (size_t)topk * sizeof(int));
        for (int f = 0; f < n_fetches; f++) {
            int *slot = trace + (size_t)f * topk;
            for (int k = 0; k < topk; k++) {
                int cand, dup;
                do {
                    cand = (int)(xs(&ts) % (uint64_t)E);
                    dup = 0;
                    for (int q = 0; q < k; q++) if (slot[q] == cand) dup = 1;
                } while (dup);
                slot[k] = cand;
            }
        }
        for (int ti = 0; ti < ntset; ti++) {
            double g = run_scatter(&topo, arena, bb, topk, n_fetches, trace,
                                   tset[ti], 1, &sink);
            printf("%s    {\"block_mib\": %zu, \"experts\": %d, \"threads\": %d, \"gbs\": %.2f}",
                   first ? "" : ",\n", bb >> 20, E, tset[ti], g);
            first = 0;
            fflush(stdout);
            if (g > best_scatter) { best_scatter = g; bs_block = bb; bs_thr = tset[ti]; bs_E = E; }
        }
        free(trace);
    }
    printf("\n  ],\n");
    if (sink == 0x1ULL) fprintf(stderr, "(sink)\n");
    double gate_pct = best_triad > 0 ? 100.0 * best_scatter / best_triad : 0;
    printf("  \"scatter_best\": {\"gbs\": %.2f, \"block_mib\": %zu, \"threads\": %d, \"experts\": %d},\n",
           best_scatter, bs_block >> 20, bs_thr, bs_E);
    printf("  \"gate_g0\": {\"scatter_pct_of_triad\": %.1f},\n", gate_pct);
    munmap(arena, arena_bytes);
    fflush(stdout);

    /* --------------------------------------------------------- NVMe read */
    if (nvme_dir) {
        char file[512];
        snprintf(file, sizeof file, "%s/hybrid_calib_nvme.dat", nvme_dir);
        size_t fbytes = nvme_gib << 30;
        if (nvme_prepare(file, fbytes) == 0) {
            /* probe O_DIRECT support once */
            int probe = open(file, O_RDONLY | O_DIRECT);
            int direct = probe >= 0;
            if (probe >= 0) close(probe);
            printf("  \"nvme\": {\"o_direct\": %s, \"points\": [\n", direct ? "true" : "false");
            int qds[4] = { 1, 4, 8, 16 };
            first = 1;
            for (int m = 0; m < 2; m++)
                for (int qi = 0; qi < 4; qi++) {
                    int q = qds[qi];
                    if (!direct) drop_cache(file);
                    NvmeJob *njs = calloc((size_t)q, sizeof(NvmeJob));
                    pthread_t *th = calloc((size_t)q, sizeof(pthread_t));
                    for (int w = 0; w < q; w++) {
                        njs[w] = (NvmeJob){ .path = file, .file_bytes = fbytes,
                                            .io_bytes = io_mib << 20,
                                            .rand_mode = m, .nthreads = q,
                                            .wid = w, .direct = direct,
                                            .seed = seed };
                        pthread_create(&th[w], NULL, nvme_worker, &njs[w]);
                    }
                    double sum = 0; int ok = 1;
                    for (int w = 0; w < q; w++) {
                        pthread_join(th[w], NULL);
                        if (njs[w].gbs < 0) ok = 0; else sum += njs[w].gbs;
                    }
                    printf("%s    {\"mode\": \"%s\", \"qd\": %d, \"gbs\": %.2f, \"ok\": %s}",
                           first ? "" : ",\n", m ? "rand" : "seq", q,
                           ok ? sum : 0.0, ok ? "true" : "false");
                    first = 0;
                    fflush(stdout);
                    free(njs); free(th);
                }
            printf("\n  ]},\n");
            unlink(file);
        } else {
            printf("  \"nvme\": {\"error\": \"prepare-failed\"},\n");
        }
    }

    printf("  \"config\": {\"triad_gib\": %zu, \"arena_gib\": %zu, \"topk\": %d, "
           "\"fetches\": %d, \"seed\": %llu, \"reps\": %d, \"tile_bytes\": %zu}\n",
           triad_gib, arena_gib, topk, n_fetches,
           (unsigned long long)seed, reps, TILE_BYTES);
    printf("}\n");
    return 0;
}
