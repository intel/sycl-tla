// Reproducer: Intel SYCL/IGC inline-vISA "+rw" constraint in loop drops pre-load mov
//
// Problem description:
//   In a templated loop over M iterations, an inline vISA asm block uses "+rw"
//   to express a read-write operand (scalar float accessed via data[mi*stride]).
//   With "+rw" the compiler MUST emit a "mov" to load data[mi*stride] into the
//   asm virtual register before executing "mad / exp".
//   Instead, the compiler silently drops the load mov, leaving the asm register
//   with undefined / stale content.
//
//   Changing "+rw" to "=rw" (write-only) makes the compiler emit the mov
//   (accidentally correct-looking), but "=rw" is semantically wrong because
//   "mad" reads %0 as an input operand.
//
// Build:
//   icpx -fsycl -O2 -o reproducer_rw_mov reproducer_rw_mov.cpp
//
// Inspect ISA:
//   SYCL_DUMP_IMAGES=1 ./reproducer_rw_mov
//   ocloc disasm -file <kernel>.bin -device <device> -dump dumps/
//
// Expected ISA per loop iteration for kernel_rw ("+rw"):
//   mov  rN  data[mi*16+ni]       <- pre-load (the read half of "+rw")
//   mad  rN  rN  scale  neg_max
//   exp  rN  rN
//   [store rN -> data[mi*16+ni]]
//
// Actual ISA per loop iteration for kernel_rw ("+rw"):
//   mad  rN  rN  scale  neg_max   <- rN is UNINITIALIZED -- mov pre-load MISSING
//   exp  rN  rN
//
// For kernel_eq ("=rw"), the mov IS present (accidentally, wrong semantics):
//   mov  rN  data[mi*16+ni]
//   mad  rN  rN  scale  neg_max
//   exp  rN  rN

#include <sycl/sycl.hpp>
#include <cstdio>

// kernel A: "+rw" (read-write) -- correct semantics, but compiler drops pre-load mov
struct KernelRW {
    float* data;
    int ni;
    float scale;
    float neg_max;
    void operator()(sycl::nd_item<1>) const {
#ifdef __SYCL_DEVICE_ONLY__
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            __asm__ volatile (
                "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0> %2(0,0)<1;1,0>\n\t"
                "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
                : "+rw"(data[mi * 16 + ni])   // READ-WRITE: compiler must emit mov
                : "rw"(scale), "rw"(neg_max)
            );
        }
#else
        (void)data; (void)ni; (void)scale; (void)neg_max;
#endif
    }
};

// kernel B: "=rw" (write-only) -- wrong semantics for mad, but does emit mov accidentally
struct KernelEQ {
    float* data;
    int ni;
    float scale;
    float neg_max;
    void operator()(sycl::nd_item<1>) const {
#ifdef __SYCL_DEVICE_ONLY__
        #pragma unroll
        for (int mi = 0; mi < 4; ++mi) {
            __asm__ volatile (
                "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0> %2(0,0)<1;1,0>\n\t"
                "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
                : "=rw"(data[mi * 16 + ni])   // WRITE-ONLY (wrong semantics, emits mov by accident)
                : "rw"(scale), "rw"(neg_max)
            );
        }
#else
        (void)data; (void)ni; (void)scale; (void)neg_max;
#endif
    }
};

int main() {
    sycl::queue q{sycl::gpu_selector_v};
    std::printf("Device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

    int N = 16 * 4;
    float* a = sycl::malloc_shared<float>(N, q);
    float* b = sycl::malloc_shared<float>(N, q);
    for (int i = 0; i < N; ++i) a[i] = b[i] = static_cast<float>(i + 1) * 0.1f;

    float scale = 0.125f, neg_max = -1.0f;

    // kernel A: "+rw" -- results wrong (garbage) if pre-load mov is missing
    q.parallel_for(sycl::nd_range<1>{{16}, {16}}, KernelRW{a, 0, scale, neg_max});
    q.wait();

    // kernel B: "=rw" -- results look correct by accident
    q.parallel_for(sycl::nd_range<1>{{16}, {16}}, KernelEQ{b, 0, scale, neg_max});
    q.wait();

    std::printf("\n  i |    +rw result |    =rw result\n");
    std::printf("----+--------------+--------------\n");
    for (int i = 0; i < 4; ++i)
        std::printf("  %d |  %12.6f |  %12.6f\n", i, a[i * 16], b[i * 16]);
    std::printf("\n(Results should match. Mismatch confirms +rw pre-load mov is dropped.)\n");

    sycl::free(a, q);
    sycl::free(b, q);
    return 0;
}
