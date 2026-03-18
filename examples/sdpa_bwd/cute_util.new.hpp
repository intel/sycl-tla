#pragma once

// Default LscAtomicFadd: per-element loop fallback (works for any InnerSize, no inline asm)
template <typename T>
struct LscAtomicFadd {
    template <class TensorCoord, class TensorReg, class MatrixC>
    CUTLASS_DEVICE static void apply(
        TensorCoord const& tCgC,
        TensorReg const& tCrC,
        MatrixC& C,
        int local_id,
        int offset)
    {
        constexpr int N = T::value;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            auto [m, n] = tCgC(offset + i);
            cutlass::atomicAdd(&C(m, n + local_id), tCrC(offset + i));
        }
    }
};

// Helper struct for template specialization of scale_exp operations
#ifdef __SYCL_DEVICE_ONLY__
template<typename T>
struct ScaleExpHelper {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        // Generic implementation using loop
        static constexpr int M = T::value;
        CUTLASS_PRAGMA_UNROLL
        for (int mi = 0; mi < M; ++mi) {
            __asm__ volatile (
                "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0> %2(0,0)<1;1,0>\n\t"
                "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
                : "+rw"(tensor(mi, ni))
                : "rw"(scale), "rw"(neg_max_scaled)
            );
        }
    }
};

template<>
struct ScaleExpHelper<Int<1>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0> %2(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
            : "+rw"(tensor(0, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

// Multi-row specialisations: all M rows go into ONE asm block so that scale and
// neg_max_scaled are materialised only once (no per-row mov).
// Each tensor element is a separate "+rw" scalar operand — the compiler maps it
// directly to the existing virtual register, so no pack/unpack movs are needed.
// Operand layout: %0..%{M-1} = per-row data, %M = scale, %{M+1} = neg_max_scaled.

template<>
struct ScaleExpHelper<Int<2>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %2(0,0)<0;1,0> %3(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<0;1,0> %3(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>"
            : "+rw"(tensor(0, ni)), "+rw"(tensor(1, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

template<>
struct ScaleExpHelper<Int<4>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>"
            : "+rw"(tensor(0, ni)), "+rw"(tensor(1, ni)),
              "+rw"(tensor(2, ni)), "+rw"(tensor(3, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

template<>
struct ScaleExpHelper<Int<8>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
            : "+rw"(tensor(0, ni)), "+rw"(tensor(1, ni)),
              "+rw"(tensor(2, ni)), "+rw"(tensor(3, ni)),
              "+rw"(tensor(4, ni)), "+rw"(tensor(5, ni)),
              "+rw"(tensor(6, ni)), "+rw"(tensor(7, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

// scale=%32, neg_max_scaled=%33 (indices follow 32 output operands %0..%31)
template<>
struct ScaleExpHelper<Int<32>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1>  %0(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1>  %1(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %2(0,0)<1>  %2(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %3(0,0)<1>  %3(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %4(0,0)<1>  %4(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %5(0,0)<1>  %5(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %6(0,0)<1>  %6(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %7(0,0)<1>  %7(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %8(0,0)<1>  %8(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %9(0,0)<1>  %9(0,0)<1;1,0>  %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %10(0,0)<1> %10(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %11(0,0)<1> %11(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %12(0,0)<1> %12(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %13(0,0)<1> %13(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %14(0,0)<1> %14(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %15(0,0)<1> %15(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %16(0,0)<1> %16(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %17(0,0)<1> %17(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %18(0,0)<1> %18(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %19(0,0)<1> %19(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %20(0,0)<1> %20(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %21(0,0)<1> %21(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %22(0,0)<1> %22(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %23(0,0)<1> %23(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %24(0,0)<1> %24(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %25(0,0)<1> %25(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %26(0,0)<1> %26(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %27(0,0)<1> %27(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %28(0,0)<1> %28(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %29(0,0)<1> %29(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %30(0,0)<1> %30(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %31(0,0)<1> %31(0,0)<1;1,0> %32(0,0)<0;1,0> %33(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1>  %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1>  %1(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %2(0,0)<1>  %2(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %3(0,0)<1>  %3(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %4(0,0)<1>  %4(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %5(0,0)<1>  %5(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %6(0,0)<1>  %6(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %7(0,0)<1>  %7(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %8(0,0)<1>  %8(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %9(0,0)<1>  %9(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %10(0,0)<1> %10(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %11(0,0)<1> %11(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %12(0,0)<1> %12(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %13(0,0)<1> %13(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %14(0,0)<1> %14(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %15(0,0)<1> %15(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %16(0,0)<1> %16(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %17(0,0)<1> %17(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %18(0,0)<1> %18(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %19(0,0)<1> %19(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %20(0,0)<1> %20(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %21(0,0)<1> %21(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %22(0,0)<1> %22(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %23(0,0)<1> %23(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %24(0,0)<1> %24(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %25(0,0)<1> %25(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %26(0,0)<1> %26(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %27(0,0)<1> %27(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %28(0,0)<1> %28(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %29(0,0)<1> %29(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %30(0,0)<1> %30(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %31(0,0)<1> %31(0,0)<1;1,0>"
            : "+rw"(tensor(0,  ni)), "+rw"(tensor(1,  ni)),
              "+rw"(tensor(2,  ni)), "+rw"(tensor(3,  ni)),
              "+rw"(tensor(4,  ni)), "+rw"(tensor(5,  ni)),
              "+rw"(tensor(6,  ni)), "+rw"(tensor(7,  ni)),
              "+rw"(tensor(8,  ni)), "+rw"(tensor(9,  ni)),
              "+rw"(tensor(10, ni)), "+rw"(tensor(11, ni)),
              "+rw"(tensor(12, ni)), "+rw"(tensor(13, ni)),
              "+rw"(tensor(14, ni)), "+rw"(tensor(15, ni)),
              "+rw"(tensor(16, ni)), "+rw"(tensor(17, ni)),
              "+rw"(tensor(18, ni)), "+rw"(tensor(19, ni)),
              "+rw"(tensor(20, ni)), "+rw"(tensor(21, ni)),
              "+rw"(tensor(22, ni)), "+rw"(tensor(23, ni)),
              "+rw"(tensor(24, ni)), "+rw"(tensor(25, ni)),
              "+rw"(tensor(26, ni)), "+rw"(tensor(27, ni)),
              "+rw"(tensor(28, ni)), "+rw"(tensor(29, ni)),
              "+rw"(tensor(30, ni)), "+rw"(tensor(31, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};

// scale=%16, neg_max_scaled=%17 (indices follow 16 output operands %0..%15)
template<>
struct ScaleExpHelper<Int<16>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1>  %0(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1>  %1(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %2(0,0)<1>  %2(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %3(0,0)<1>  %3(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %4(0,0)<1>  %4(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %5(0,0)<1>  %5(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %6(0,0)<1>  %6(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %7(0,0)<1>  %7(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %8(0,0)<1>  %8(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %9(0,0)<1>  %9(0,0)<1;1,0>  %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %10(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %11(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %12(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %13(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %14(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %15(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1>  %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1>  %1(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %2(0,0)<1>  %2(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %3(0,0)<1>  %3(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %4(0,0)<1>  %4(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %5(0,0)<1>  %5(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %6(0,0)<1>  %6(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %7(0,0)<1>  %7(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %8(0,0)<1>  %8(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %9(0,0)<1>  %9(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %10(0,0)<1> %10(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %11(0,0)<1> %11(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %12(0,0)<1> %12(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %13(0,0)<1> %13(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %14(0,0)<1> %14(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %15(0,0)<1> %15(0,0)<1;1,0>"
            : "+rw"(tensor(0,  ni)), "+rw"(tensor(1,  ni)),
              "+rw"(tensor(2,  ni)), "+rw"(tensor(3,  ni)),
              "+rw"(tensor(4,  ni)), "+rw"(tensor(5,  ni)),
              "+rw"(tensor(6,  ni)), "+rw"(tensor(7,  ni)),
              "+rw"(tensor(8,  ni)), "+rw"(tensor(9,  ni)),
              "+rw"(tensor(10, ni)), "+rw"(tensor(11, ni)),
              "+rw"(tensor(12, ni)), "+rw"(tensor(13, ni)),
              "+rw"(tensor(14, ni)), "+rw"(tensor(15, ni))
            : "rw"(scale), "rw"(neg_max_scaled)
        );
    }
};


// Batched LSC atomic fadd with vISA inline assembly.
// Specialization for InnerSize=8: single asm block of 8 atomic fadds.
// The 8 inner elements have equally-spaced addresses (incrementing by a constant
// row stride), allowing in-place address advancement via vISA add instructions.
template <>
struct LscAtomicFadd<Int<1>> {
    template <class TensorCoord, class TensorReg, class MatrixC>
    CUTLASS_DEVICE static void apply(
        TensorCoord const& tCgC,
        TensorReg const& tCrC,
        MatrixC& C,
        int local_id,
        int offset)
    {
        auto [m0, n0] = tCgC(offset);
        uint64_t addr = reinterpret_cast<uint64_t>(&C(m0, n0 + local_id));

        float v0 = tCrC(offset);

        __asm__ volatile (
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %1 %%null\n"
            : "+rw"(addr)
            : "rw"(v0)
            : "memory"
        );
    }
};

template <>
struct LscAtomicFadd<Int<2>> {
    template <class TensorCoord, class TensorReg, class MatrixC>
    CUTLASS_DEVICE static void apply(
        TensorCoord const& tCgC,
        TensorReg const& tCrC,
        MatrixC& C,
        int local_id,
        int offset)
    {
        auto [m0, n0] = tCgC(offset);
        uint64_t addr = reinterpret_cast<uint64_t>(&C(m0, n0 + local_id));
        auto [m1, n1] = tCgC(offset + 1);
        uint64_t stride = reinterpret_cast<uint64_t>(&C(m1, n1 + local_id)) - addr;

        float v0 = tCrC(offset+0), v1 = tCrC(offset+1);

        __asm__ volatile (
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %2 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %3 %%null\n"
            : "+rw"(addr)
            : "rw"(stride), "rw"(v0), "rw"(v1)
            : "memory"
        );
    }
};

template <>
struct LscAtomicFadd<Int<4>> {
    template <class TensorCoord, class TensorReg, class MatrixC>
    CUTLASS_DEVICE static void apply(
        TensorCoord const& tCgC,
        TensorReg const& tCrC,
        MatrixC& C,
        int local_id,
        int offset)
    {
        auto [m0, n0] = tCgC(offset);
        uint64_t addr = reinterpret_cast<uint64_t>(&C(m0, n0 + local_id));
        auto [m1, n1] = tCgC(offset + 1);
        uint64_t stride = reinterpret_cast<uint64_t>(&C(m1, n1 + local_id)) - addr;

        float v0 = tCrC(offset+0), v1 = tCrC(offset+1);
        float v2 = tCrC(offset+2), v3 = tCrC(offset+3);

        __asm__ volatile (
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %2 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %3 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %4 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %5 %%null\n"
            : "+rw"(addr)
            : "rw"(stride), "rw"(v0), "rw"(v1), "rw"(v2), "rw"(v3)
            : "memory"
        );
    }
};

template <>
struct LscAtomicFadd<Int<8>> {
    template <class TensorCoord, class TensorReg, class MatrixC>
    CUTLASS_DEVICE static void apply(
        TensorCoord const& tCgC,
        TensorReg const& tCrC,
        MatrixC& C,
        int local_id,
        int offset)
    {
        auto [m0, n0] = tCgC(offset);
        uint64_t addr = reinterpret_cast<uint64_t>(&C(m0, n0 + local_id));
        auto [m1, n1] = tCgC(offset + 1);
        uint64_t stride = reinterpret_cast<uint64_t>(&C(m1, n1 + local_id)) - addr;

        float v0 = tCrC(offset+0), v1 = tCrC(offset+1);
        float v2 = tCrC(offset+2), v3 = tCrC(offset+3);
        float v4 = tCrC(offset+4), v5 = tCrC(offset+5);
        float v6 = tCrC(offset+6), v7 = tCrC(offset+7);

        __asm__ volatile (
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %2 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %3 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %4 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %5 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %6 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %7 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %8 %%null\n"
            "add (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0> %1(0,0)<0;1,0>\n"
            "lsc_atomic_fadd.ugm (M1, 16) %%null:d32 flat[%0]:a64 %9 %%null\n"
            : "+rw"(addr)
            : "rw"(stride), "rw"(v0), "rw"(v1), "rw"(v2), "rw"(v3),
              "rw"(v4), "rw"(v5), "rw"(v6), "rw"(v7)
            : "memory"
        );
    }
};

#endif // end of __SYCL_DEVICE_ONLY__
