#pragma once
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
            float val = tensor(mi, ni);
            __asm__ volatile (
                "mad (M1, 16) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<0;1,0> %3(0,0)<1;1,0>\n\t"
                "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
                : "=rw"(val)
                : "0"(val), "rw"(scale), "rw"(neg_max_scaled)
            );
            tensor(mi, ni) = val;
        }
    }
};

template<>
struct ScaleExpHelper<Int<1>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        float val = tensor(0, ni);
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %1(0,0)<1;1,0> %2(0,0)<0;1,0> %3(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>"
            : "=rw"(val)
            : "0"(val), "rw"(scale), "rw"(neg_max_scaled)
        );
        tensor(0, ni) = val;
    }
};

template<>
struct ScaleExpHelper<Int<2>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
        float val0 = tensor(0, ni);
        float val1 = tensor(1, ni);
        __asm__ volatile (
            "mad (M1, 16) %0(0,0)<1> %2(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "mad (M1, 16) %1(0,0)<1> %3(0,0)<1;1,0> %4(0,0)<0;1,0> %5(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
            "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>"
            : "=rw"(val0), "=rw"(val1)
            : "0"(val0), "1"(val1), "rw"(scale), "rw"(neg_max_scaled)
        );
        tensor(0, ni) = val0;
        tensor(1, ni) = val1;
    }
};

template<>
struct ScaleExpHelper<Int<4>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
    float val0 = tensor(0, ni);
    float val1 = tensor(1, ni);
    float val2 = tensor(2, ni);
    float val3 = tensor(3, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %4(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %5(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %6(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %7(0,0)<1;1,0> %8(0,0)<0;1,0> %9(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3), "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(0, ni) = val0;
    tensor(1, ni) = val1;
    tensor(2, ni) = val2;
    tensor(3, ni) = val3;
    }
};

template<>
struct ScaleExpHelper<Int<8>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
    float val0 = tensor(0, ni);
    float val1 = tensor(1, ni);
    float val2 = tensor(2, ni);
    float val3 = tensor(3, ni);
    float val4 = tensor(4, ni);
    float val5 = tensor(5, ni);
    float val6 = tensor(6, ni);
    float val7 = tensor(7, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %8(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %9(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %4(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %5(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %6(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %7(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3),
          "=rw"(val4), "=rw"(val5), "=rw"(val6), "=rw"(val7)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3),
          "4"(val4), "5"(val5), "6"(val6), "7"(val7),
          "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(0, ni) = val0;
    tensor(1, ni) = val1;
    tensor(2, ni) = val2;
    tensor(3, ni) = val3;
    tensor(4, ni) = val4;
    tensor(5, ni) = val5;
    tensor(6, ni) = val6;
    tensor(7, ni) = val7;
    }
};

template<>
struct ScaleExpHelper<Int<32>> {
    template<class Engine0, class Layout0>
    static CUTLASS_DEVICE void apply(
        Tensor<Engine0, Layout0> &tensor, int ni, float scale, float neg_max_scaled) {
    // Unrolled 4x8 processing - group 0
    float val0 = tensor(0, ni);
    float val1 = tensor(1, ni);
    float val2 = tensor(2, ni);
    float val3 = tensor(3, ni);
    float val4 = tensor(4, ni);
    float val5 = tensor(5, ni);
    float val6 = tensor(6, ni);
    float val7 = tensor(7, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %8(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %9(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %4(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %5(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %6(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %7(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3),
          "=rw"(val4), "=rw"(val5), "=rw"(val6), "=rw"(val7)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3),
          "4"(val4), "5"(val5), "6"(val6), "7"(val7),
          "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(0, ni) = val0;
    tensor(1, ni) = val1;
    tensor(2, ni) = val2;
    tensor(3, ni) = val3;
    tensor(4, ni) = val4;
    tensor(5, ni) = val5;
    tensor(6, ni) = val6;
    tensor(7, ni) = val7;

    // Unrolled 4x8 processing - group 1
    val0 = tensor(8, ni);
    val1 = tensor(9, ni);
    val2 = tensor(10, ni);
    val3 = tensor(11, ni);
    val4 = tensor(12, ni);
    val5 = tensor(13, ni);
    val6 = tensor(14, ni);
    val7 = tensor(15, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %8(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %9(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %4(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %5(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %6(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %7(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3),
          "=rw"(val4), "=rw"(val5), "=rw"(val6), "=rw"(val7)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3),
          "4"(val4), "5"(val5), "6"(val6), "7"(val7),
          "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(8, ni) = val0;
    tensor(9, ni) = val1;
    tensor(10, ni) = val2;
    tensor(11, ni) = val3;
    tensor(12, ni) = val4;
    tensor(13, ni) = val5;
    tensor(14, ni) = val6;
    tensor(15, ni) = val7;

    // Unrolled 4x8 processing - group 2
    val0 = tensor(16, ni);
    val1 = tensor(17, ni);
    val2 = tensor(18, ni);
    val3 = tensor(19, ni);
    val4 = tensor(20, ni);
    val5 = tensor(21, ni);
    val6 = tensor(22, ni);
    val7 = tensor(23, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %8(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %9(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %4(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %5(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %6(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %7(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3),
          "=rw"(val4), "=rw"(val5), "=rw"(val6), "=rw"(val7)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3),
          "4"(val4), "5"(val5), "6"(val6), "7"(val7),
          "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(16, ni) = val0;
    tensor(17, ni) = val1;
    tensor(18, ni) = val2;
    tensor(19, ni) = val3;
    tensor(20, ni) = val4;
    tensor(21, ni) = val5;
    tensor(22, ni) = val6;
    tensor(23, ni) = val7;

    // Unrolled 4x8 processing - group 3
    val0 = tensor(24, ni);
    val1 = tensor(25, ni);
    val2 = tensor(26, ni);
    val3 = tensor(27, ni);
    val4 = tensor(28, ni);
    val5 = tensor(29, ni);
    val6 = tensor(30, ni);
    val7 = tensor(31, ni);
    __asm__ volatile (
        "mad (M1, 16) %0(0,0)<1> %8(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %1(0,0)<1> %9(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %2(0,0)<1> %10(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %3(0,0)<1> %11(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %4(0,0)<1> %12(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %5(0,0)<1> %13(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %6(0,0)<1> %14(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "mad (M1, 16) %7(0,0)<1> %15(0,0)<1;1,0> %16(0,0)<0;1,0> %17(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %0(0,0)<1> %0(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %1(0,0)<1> %1(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %2(0,0)<1> %2(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %3(0,0)<1> %3(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %4(0,0)<1> %4(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %5(0,0)<1> %5(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %6(0,0)<1> %6(0,0)<1;1,0>\n\t"
        "exp (M1, 16) %7(0,0)<1> %7(0,0)<1;1,0>"
        : "=rw"(val0), "=rw"(val1), "=rw"(val2), "=rw"(val3),
          "=rw"(val4), "=rw"(val5), "=rw"(val6), "=rw"(val7)
        : "0"(val0), "1"(val1), "2"(val2), "3"(val3),
          "4"(val4), "5"(val5), "6"(val6), "7"(val7),
          "rw"(scale), "rw"(neg_max_scaled)
    );
    tensor(24, ni) = val0;
    tensor(25, ni) = val1;
    tensor(26, ni) = val2;
    tensor(27, ni) = val3;
    tensor(28, ni) = val4;
    tensor(29, ni) = val5;
    tensor(30, ni) = val6;
    tensor(31, ni) = val7;
    }
};
#endif // end of __SYCL_DEVICE_ONLY__
