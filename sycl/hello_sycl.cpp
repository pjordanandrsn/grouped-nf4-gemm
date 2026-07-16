// Copyright (c) 2026 Cerin Amroth LLC. MIT license (see LICENSE).
// M0 — toolchain gate: enumerate SYCL devices, confirm the Intel GPU (P630 /
// Arc / Max) is visible, and run a trivial kernel ON it. If the GPU does not
// enumerate (Gen9.5 dropped by the installed compute-runtime — the same
// NEO-version regime the OpenVINO work hit), this prints it plainly so we
// know the hardware path is the blocker before any NF4 kernel is written.
#include <sycl/sycl.hpp>
#include <iostream>
#include <vector>

int main() {
  std::cout << "=== SYCL devices ===\n";
  for (auto &p : sycl::platform::get_platforms()) {
    std::cout << "Platform: " << p.get_info<sycl::info::platform::name>() << "\n";
    for (auto &d : p.get_devices()) {
      const char *k = d.is_gpu() ? "GPU" : d.is_cpu() ? "CPU" : "other";
      std::cout << "  [" << k << "] " << d.get_info<sycl::info::device::name>()
                << "  CU=" << d.get_info<sycl::info::device::max_compute_units>()
                << "  gmem=" << (d.get_info<sycl::info::device::global_mem_size>() >> 20) << "MB\n";
    }
  }

  bool have_gpu = false;
  sycl::device dev;
  try { dev = sycl::device(sycl::gpu_selector_v); have_gpu = true; }
  catch (...) { std::cout << "\n*** NO SYCL GPU DEVICE — falling back to default ***\n"; dev = sycl::device(sycl::default_selector_v); }

  sycl::queue q(dev);
  std::cout << "\nRunning kernel on: " << q.get_device().get_info<sycl::info::device::name>()
            << " (" << (q.get_device().is_gpu() ? "GPU" : "not-GPU") << ")\n";

  const int n = 4096;
  std::vector<float> a(n, 1.5f), b(n, 2.25f), c(n, 0.f);
  {
    sycl::buffer<float> ba(a.data(), n), bb(b.data(), n), bc(c.data(), n);
    q.submit([&](sycl::handler &h) {
      sycl::accessor A(ba, h, sycl::read_only);
      sycl::accessor B(bb, h, sycl::read_only);
      sycl::accessor C(bc, h, sycl::write_only, sycl::no_init);
      h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) { C[i] = A[i] + B[i]; });
    });
    q.wait();
  }
  bool ok = (c[0] == 3.75f && c[n - 1] == 3.75f);
  std::cout << "vectoradd c[0]=" << c[0] << " c[n-1]=" << c[n-1] << " (expect 3.75)\n";
  std::cout << (ok ? (have_gpu ? "M0 PASS (GPU)\n" : "M0 PASS (CPU-only — GPU path unavailable)\n")
                   : "M0 FAIL\n");
  return ok ? 0 : 1;
}
