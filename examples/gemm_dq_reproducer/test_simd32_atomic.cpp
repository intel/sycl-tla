#include <sycl/sycl.hpp>
#include <cstdio>
#include <vector>

struct Params { float *dst; };

// Use intel's vector type to get contiguous register allocation
namespace intel {
template<typename T, int N>
using vector_t = T __attribute__((ext_vector_type(N)));
}

void kernel_func(Params p, sycl::nd_item<1> item) {
#ifdef __SYCL_DEVICE_ONLY__
    auto sg = item.get_sub_group();
    int lid = sg.get_local_linear_id();

    long addr_lo = (long)&p.dst[lid];
    long addr_hi = (long)&p.dst[lid + 16];
    float val_lo = 1.0f;
    float val_hi = 2.0f;

    // Create vector types that span the right number of GRFs:
    // For SIMD32 scatter atomic:
    //   Address: 32 × 64-bit = 4 GRFs → need a "2 × long" per lane → 4 GRFs total
    //   Data: 32 × 32-bit = 2 GRFs → need a "2 × float" per lane → 2 GRFs total
    //
    // In SPMD: each lane has 2 longs and 2 floats.
    // The compiler should allocate these in contiguous GRFs.

    intel::vector_t<long, 2> addrs = {addr_lo, addr_hi};
    intel::vector_t<float, 2> vals = {val_lo, val_hi};

    asm volatile(
        "lsc_atomic_fadd.ugm (M1_NM, 32)  %%null:d32  flat[%0]:a64  %1  %%null"
        :: "rw"(addrs), "rw"(vals)
    );
#endif
}

template<class...> class TestKernel;

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    float *dst = sycl::malloc_device<float>(32, q);
    q.memset(dst, 0, 32 * sizeof(float)).wait();

    Params p{dst};
    q.submit([&](sycl::handler &h) {
        h.parallel_for<TestKernel<>>(
            sycl::nd_range<1>(16, 16),
            [=](sycl::nd_item<1> item) [[intel::reqd_sub_group_size(16)]] {
                kernel_func(p, item);
            });
    }).wait();

    std::vector<float> result(32);
    q.memcpy(result.data(), dst, 32 * sizeof(float)).wait();

    printf("SIMD32 atomic via ext_vector_type:\n");
    bool pass = true;
    for (int i = 0; i < 32; i++) {
        float expected = (i < 16) ? 1.0f : 2.0f;
        if (result[i] != expected) {
            printf("  [%2d] got %.1f expected %.1f FAIL\n", i, result[i], expected);
            pass = false;
        }
    }
    if (pass) printf("  All 32 elements correct: PASS\n");

    sycl::free(dst, q);
    return pass ? 0 : 1;
}
