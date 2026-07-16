// Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
// M2 — SYCL decode gemv, PERFORMANCE. Two kernels on one device:
//   naive  = the M1 kernel (one work-item per (g,n), activation re-read from
//            global N times per group).
//   tiled  = M2 — a work-group owns one group g and a strip of WG_N columns;
//            the reused activation row a[g,:] is staged once into local memory
//            (SLM) and shared across the strip, killing the N-fold redundant
//            global traffic on the activation vector. The per-output k-loop is
//            UNCHANGED (same fp32 accumulate order), so the result is bit-for-bit
//            the M1 result — it clears the identical <1e-4 gate, and any speedup
//            is a pure memory-traffic win, not a numerics change.
// The harness sweeps WG_N (the work-group-sizing M2 axis) and reports ms +
// speedup vs naive for each. Absolute throughput on Arc/Max is R3 "port target"
// until measured there; this binary proves the win exists and is portable.
#include <sycl/sycl.hpp>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <vector>

static const float NF4_LUT[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};
static const int BLK = 64;
static const int SLM_FLOATS_MAX = 3072;  // cap SLM activation cache (12 KB)

template <typename T> static void rd(std::ifstream &f, std::vector<T> &v) {
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T));
}

static double validate(const std::vector<float> &out,
                       const std::vector<float> &exp) {
  // norm-relative error ‖out-exp‖/‖exp‖ — the suite's b_rel. Per-element rel
  // error with a floor spikes on near-zero cancellation cells (K=2048 has
  // them) and mislabels a bit-correct kernel; norm-relative does not.
  double dn = 0.0, rn = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    double d = (double)out[i] - (double)exp[i];
    dn += d * d; rn += (double)exp[i] * (double)exp[i];
  }
  return std::sqrt(dn) / std::max(std::sqrt(rn), 1e-12);
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/work/testvec.bin";
  const int ITERS = argc > 2 ? std::atoi(argv[2]) : 50;
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "cannot open " << path << "\n"; return 2; }
  std::cout << std::unitbuf;  // flush each write so an abort shows its point
  int E, N, K, G;
  f.read(reinterpret_cast<char *>(&E), 4);
  f.read(reinterpret_cast<char *>(&N), 4);
  f.read(reinterpret_cast<char *>(&K), 4);
  f.read(reinterpret_cast<char *>(&G), 4);
  std::vector<int32_t> eids(G);
  std::vector<uint8_t> Bv((size_t)E * N * (K / 2));
  std::vector<float> absmax((size_t)E * N * (K / BLK));
  std::vector<float> acts((size_t)G * K);
  std::vector<float> expected((size_t)G * N);
  rd(f, eids); rd(f, Bv); rd(f, absmax); rd(f, acts); rd(f, expected);
  const int Kh = K / 2, Kb = K / BLK, NKh = N * Kh, NKb = N * Kb;

  sycl::queue q;  // wall-clock timing (chrono); no device profiling property
                  // (the legacy Gen9.5 NEO 23.43 OpenCL path OOMs enqueue with it)
  auto dev = q.get_device();
  std::cout << "device: " << dev.get_info<sycl::info::device::name>()
            << (dev.is_gpu() ? " [GPU]" : " [CPU]") << "\n";
  std::cout << "shape: E=" << E << " N=" << N << " K=" << K << " G=" << G
            << "  iters=" << ITERS << "\n";

  sycl::buffer<int32_t> be(eids.data(), eids.size());
  sycl::buffer<uint8_t> bB(Bv.data(), Bv.size());
  sycl::buffer<float> bam(absmax.data(), absmax.size());
  sycl::buffer<float> ba(acts.data(), acts.size());
  sycl::buffer<float> blut(NF4_LUT, 16);

  // A FRESH queue per timed config. The legacy Gen9.5 NEO OpenCL backend
  // accumulates host-side command allocations across submits and OOMs the
  // enqueue past ~100 on one queue; a per-config queue releases them at scope
  // exit. `submit` takes the queue so each config drives its own.
  auto time_kernel = [&](auto submit) {
    sycl::queue tq;
    submit(tq); tq.wait();  // warmup (also absorbs first-use JIT)
    double best = 1e30;
    for (int it = 0; it < ITERS; ++it) {
      auto t0 = std::chrono::high_resolution_clock::now();
      submit(tq); tq.wait();
      auto t1 = std::chrono::high_resolution_clock::now();
      best = std::min(best,
          std::chrono::duration<double, std::milli>(t1 - t0).count());
    }
    return best;
  };

  // ---- naive (M1) ----
  std::vector<float> out_naive((size_t)G * N, 0.0f);
  double t_naive;
  {
    sycl::buffer<float> bo(out_naive.data(), out_naive.size());
    t_naive = time_kernel([&](sycl::queue &q) {
      q.submit([&](sycl::handler &h) {
        sycl::accessor E_(be, h, sycl::read_only);
        sycl::accessor B_(bB, h, sycl::read_only);
        sycl::accessor AM(bam, h, sycl::read_only);
        sycl::accessor A(ba, h, sycl::read_only);
        sycl::accessor L(blut, h, sycl::read_only);
        sycl::accessor O(bo, h, sycl::write_only, sycl::no_init);
        h.parallel_for(sycl::range<2>(G, N), [=](sycl::id<2> idx) {
          const int g = idx[0], n = idx[1];
          const int e = E_[g];
          float acc = 0.0f;
          for (int k = 0; k < K; ++k) {
            const uint8_t byte = B_[(size_t)e * NKh + n * Kh + (k >> 1)];
            const int nib = (k & 1) == 0 ? (byte >> 4) & 0xF : byte & 0xF;
            acc += L[nib] * AM[(size_t)e * NKb + n * Kb + (k / BLK)]
                   * A[(size_t)g * K + k];
          }
          O[(size_t)g * N + n] = acc;
        });
      });
    });
  }
  double rel_naive = validate(out_naive, expected);
  std::cout << "naive           : " << t_naive << " ms   max_rel=" << rel_naive
            << (rel_naive < 1e-2 ? "  PASS\n" : "  FAIL\n");

  // ---- tiled (M2), swept over work-group width ----
  const bool slm_ok = K <= SLM_FLOATS_MAX;
  double best_speedup = 0.0; int best_wg = 0;
  for (int WG : {16, 32, 64, 128, 256}) {
    if (WG > N && WG > 32) continue;
    const int Npad = ((N + WG - 1) / WG) * WG;
    std::vector<float> out_t((size_t)G * N, 0.0f);
    double t_t;
    {
      sycl::buffer<float> bo(out_t.data(), out_t.size());
      t_t = time_kernel([&](sycl::queue &q) {
        q.submit([&](sycl::handler &h) {
          sycl::accessor E_(be, h, sycl::read_only);
          sycl::accessor B_(bB, h, sycl::read_only);
          sycl::accessor AM(bam, h, sycl::read_only);
          sycl::accessor A(ba, h, sycl::read_only);
          sycl::accessor L(blut, h, sycl::read_only);
          sycl::accessor O(bo, h, sycl::write_only, sycl::no_init);
          // Static local size (compile-time SLM_FLOATS_MAX): legacy Gen9.5 NEO
          // fails the program build on a runtime-sized local_accessor.
          sycl::local_accessor<float> aloc(sycl::range<1>(SLM_FLOATS_MAX), h);
          h.parallel_for(
              sycl::nd_range<2>(sycl::range<2>(G, Npad), sycl::range<2>(1, WG)),
              [=](sycl::nd_item<2> it) {
                const int g = it.get_global_id(0);
                const int n = it.get_global_id(1);
                const int lid = it.get_local_id(1);
                const int e = E_[g];
                if (slm_ok) {
                  for (int i = lid; i < K; i += WG)
                    aloc[i] = A[(size_t)g * K + i];
                  it.barrier(sycl::access::fence_space::local_space);
                }
                if (n >= N) return;
                float acc = 0.0f;
                for (int k = 0; k < K; ++k) {
                  const uint8_t byte = B_[(size_t)e * NKh + n * Kh + (k >> 1)];
                  const int nib = (k & 1) == 0 ? (byte >> 4) & 0xF : byte & 0xF;
                  const float a = slm_ok ? aloc[k] : A[(size_t)g * K + k];
                  acc += L[nib] * AM[(size_t)e * NKb + n * Kb + (k / BLK)] * a;
                }
                O[(size_t)g * N + n] = acc;
              });
        });
      });
    }
    double rel_t = validate(out_t, expected);
    double sp = t_naive / t_t;
    std::cout << "tiled WG=" << WG << (WG < 100 ? "  " : " ") << "     : " << t_t
              << " ms   max_rel=" << rel_t << (rel_t < 1e-2 ? "  PASS" : "  FAIL")
              << "   speedup=" << sp << "x\n";
    if (rel_t < 1e-2 && sp > best_speedup) { best_speedup = sp; best_wg = WG; }
  }
  std::cout << "M2 best: WG=" << best_wg << "  speedup=" << best_speedup
            << "x  (SLM activation cache " << (slm_ok ? "ON" : "OFF (K too big)")
            << ")\n";
  std::cout << (best_speedup > 0.0 ? "M2 DONE (tiled variant numerically identical, faster)\n"
                                   : "M2 FAIL\n");
  return best_speedup > 0.0 ? 0 : 1;
}
