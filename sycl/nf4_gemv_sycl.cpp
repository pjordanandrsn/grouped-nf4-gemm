// Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
// M1 — SYCL decode gemv, correctness. Mirrors kernel/nf4_grouped.py's
// _gemv_nf4_grouped bit-for-bit: high-nibble-first NF4 unpack, 64-element
// blockwise fp32 absmax, fp32 accumulate. Naive one-work-item-per-(g,n) tiling
// (speed is M2). Validates against a test vector whose oracle is the parent's
// canonical dequant_ref (see gen_testvec.py). Device-independent numerics —
// runs on CPU-SYCL or GPU-SYCL identically.
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

// Exact bitsandbytes NF4 codebook (copied from kernel/nf4_grouped.py NF4_LUT).
static const float NF4_LUT[16] = {
    -1.0f, -0.6961928009986877f, -0.5250730514526367f, -0.39491748809814453f,
    -0.28444138169288635f, -0.18477343022823334f, -0.09105003625154495f, 0.0f,
    0.07958029955625534f, 0.16093020141124725f, 0.24611230194568634f,
    0.33791524171829224f, 0.44070982933044434f, 0.5626170039176941f,
    0.7229568362236023f, 1.0f};
static const int BLK = 64;

template <typename T> static void rd(std::ifstream &f, std::vector<T> &v) {
  f.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(T));
}

int main(int argc, char **argv) {
  const char *path = argc > 1 ? argv[1] : "/work/testvec.bin";
  std::ifstream f(path, std::ios::binary);
  if (!f) { std::cerr << "cannot open " << path << "\n"; return 2; }
  int E, N, K, G;
  f.read(reinterpret_cast<char *>(&E), 4);
  f.read(reinterpret_cast<char *>(&N), 4);
  f.read(reinterpret_cast<char *>(&K), 4);
  f.read(reinterpret_cast<char *>(&G), 4);
  std::vector<int32_t> eids(G);
  std::vector<uint8_t> B((size_t)E * N * (K / 2));
  std::vector<float> absmax((size_t)E * N * (K / BLK));
  std::vector<float> acts((size_t)G * K);
  std::vector<float> expected((size_t)G * N);
  rd(f, eids); rd(f, B); rd(f, absmax); rd(f, acts); rd(f, expected);
  std::vector<float> out((size_t)G * N, 0.0f);

  sycl::queue q;
  std::cout << "device: " << q.get_device().get_info<sycl::info::device::name>()
            << (q.get_device().is_gpu() ? " [GPU]\n" : " [CPU]\n");
  const int Kh = K / 2, Kb = K / BLK, NKh = N * Kh, NKb = N * Kb;
  {
    sycl::buffer<int32_t> be(eids.data(), eids.size());
    sycl::buffer<uint8_t> bB(B.data(), B.size());
    sycl::buffer<float> bam(absmax.data(), absmax.size());
    sycl::buffer<float> ba(acts.data(), acts.size());
    sycl::buffer<float> blut(NF4_LUT, 16);
    sycl::buffer<float> bo(out.data(), out.size());
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
          const float am = AM[(size_t)e * NKb + n * Kb + (k / BLK)];
          acc += L[nib] * am * A[(size_t)g * K + k];
        }
        O[(size_t)g * N + n] = acc;
      });
    });
    q.wait();
  }

  double max_rel = 0.0, max_abs = 0.0;
  for (size_t i = 0; i < out.size(); ++i) {
    double a = out[i], b = expected[i], d = std::fabs(a - b);
    max_abs = std::max(max_abs, d);
    double denom = std::max(std::fabs(b), 1e-4);
    max_rel = std::max(max_rel, d / denom);
  }
  std::cout << "cells=" << out.size() << " max_abs=" << max_abs
            << " max_rel=" << max_rel << "\n";
  // fp32 kernel vs fp32 reference: identical arithmetic up to FP reassociation.
  bool ok = max_rel < 1e-4;
  std::cout << (ok ? "M1 PASS (numerics match canonical dequant_ref)\n"
                   : "M1 FAIL\n");
  return ok ? 0 : 1;
}
