/*
// Copyright (c) 2019-2024 Ben Ashbaugh
//
// SPDX-License-Identifier: MIT
*/

#include <sycl.hpp>

#include <algorithm>
#include <chrono>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include "../gemm_validation.hpp"
// #include "pvc_prefetch.hpp"
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

using test_clock = std::chrono::high_resolution_clock;

using namespace cute;

#define tM (8)
#define tK (16)
#define tN (16)

#define get_sub_group_id() \
    (sycl::ext::oneapi::experimental::this_nd_item<3>() \
                    .get_sub_group() \
                    .get_group_id()[0])
#define get_sub_group_local_id() \
    (sycl::ext::oneapi::experimental::this_nd_item<3>() \
                    .get_sub_group() \
                    .get_local_id()[0])

using dtype_a = bfloat16_t;
using dtype_b = bfloat16_t;
using dtype_c = float;
using dtype_acc = float;

size_t testIterations = 10;

dtype_acc threshold = 0.01f;

#define WARMUP_ITERATIONS 10

#define PREFETCH_DISTANCE 3

#define CACHE_FLUSH 1

#define random_float() (generate_real_random<double>())

#define split_barrier_arrive() __builtin_IB_work_group_barrier_arrive(0)
#define split_barrier_wait() __builtin_IB_work_group_barrier_wait(0)

template <typename result_type>
inline result_type generate_real_random(
        result_type a = 0.0, result_type b = 1.0) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine engine(seed);
    std::uniform_real_distribution<result_type> distribution(a, b);

    return distribution(engine);
}

template <typename T>
static void init_matrix(
        T *M, size_t numRows, size_t numCols, size_t batch = 1) {
    // std::random_device dev;
    // std::mt19937 rng(dev());
    // std::uniform_real_distribution<float> dist(0, 1.0);
    for (size_t b = 0; b < batch; b++) {
        for (size_t r = 0; r < numRows; r++) {
            for (size_t c = 0; c < numCols; c++) {
                M[b * numRows * numCols + r * numCols + c]
                        = bfloat16_t(random_float());
            }
        }
    }
}

template <typename T>
static void vnni_matrix(T *dst, const T *src, size_t numRows, size_t numCols,
        size_t factor, size_t batch = 1) {
    for (size_t b = 0; b < batch; b++) {
        for (size_t r = 0; r < numRows / factor; r++) {
            for (size_t c = 0; c < numCols; c++) {
                for (size_t k = 0; k < factor; k++) {
                    dst[b * numRows * numCols + r * numCols * factor
                            + c * factor + k]
                            = src[b * numRows * numCols
                                    + (r * factor + k) * numCols + c];
                }
            }
        }
    }
}

template <typename T>
float check_results(size_t M, size_t N, const T *C, const T *C_ref) {
    float err = 0.f;
    size_t error_cnt = 0;
    for (size_t m = 0; m < M; m++) {
        for (size_t n = 0; n < N; n++) {
            auto index = m * N + n;
            auto localErr = std::fabs(C[index] - C_ref[index])
                    / std::max(std::fabs(C[index]), std::fabs(C_ref[index]));
            err = std::max(localErr, err);
            if (localErr >= threshold) {
                error_cnt++;
#if 0
         std::cerr << "Error at m = " << m << ", n = " << n << ": (local error"
                   << localErr << "): Wanted " << C_ref[index] << ", got "
                  << C[index] << std::endl;
#endif
            }
        }
    }

    auto pass_rate = (1.f - ((float)error_cnt / (M * N)));

    // std::cout << "\n\n==== Pass rate is: " << pass_rate << "% !!!\n"
    //        << std::endl;

    return pass_rate;
}

inline size_t time_event(sycl::event &e) {
    // get start and end times
    cl_ulong start_time = e.template get_profiling_info<
            sycl::info::event_profiling::command_start>();

    cl_ulong end_time = e.template get_profiling_info<
            sycl::info::event_profiling::command_end>();

    // return the delta
    return static_cast<size_t>(end_time - start_time);
}

template <int wg_tile_m, int wg_tile_n, int sg_tile_m, int sg_tile_n,
        int sg_tile_k, bool wg_order_m_first = false, uint32_t snake_n = 0>
void cute_gemm(size_t M, size_t K, size_t N) {

    auto queue = sycl::queue {{sycl::property::queue::enable_profiling()}};
    auto context = queue.get_info<sycl::info::queue::context>();
    auto device = queue.get_info<sycl::info::queue::device>();

    const uint32_t total_iterations = WARMUP_ITERATIONS + testIterations;
#ifdef CACHE_FLUSH
    const uint32_t mem_iter = total_iterations;
#else
    const uint32_t mem_iter = 1;
#endif

    dtype_a *A_host = (dtype_a *)std::malloc(sizeof(dtype_a) * M * K);
    dtype_b *B_host = (dtype_b *)std::malloc(sizeof(dtype_b) * N * K);
    dtype_c *C_host = (dtype_c *)std::malloc(sizeof(dtype_c) * M * N);

    dtype_a *A_dev = (dtype_a *)sycl::malloc_device(
            sizeof(dtype_a) * M * K * mem_iter, device, context);
    dtype_b *B_dev = (dtype_b *)sycl::malloc_device(
            sizeof(dtype_b) * N * K * mem_iter, device, context);
    dtype_acc *C_dev = (dtype_acc *)sycl::malloc_device(
            sizeof(dtype_c) * M * N * mem_iter, device, context);

    dtype_acc *C_ref_host = (dtype_acc *)std::malloc(sizeof(dtype_acc) * M * N);
    printf("\n\nInitializing source matrices, MKN(%d, %d, %d), "
           "Config(%d,%d,%d,%d,%d), Nd_range: %s, Snake_n(%d)...\n",
            M, K, N, wg_tile_m, wg_tile_n, sg_tile_m, sg_tile_n, sg_tile_k,
            wg_order_m_first ? "N order" : "M order", snake_n);
    init_matrix(A_host, M, K);
    init_matrix(B_host, K, N);
    for (int it = 0; it < mem_iter; it++) {
        queue.memcpy(A_dev + it * M * K, A_host, sizeof(dtype_a) * M * K)
                .wait();
        queue.memcpy(B_dev + it * N * K, B_host, sizeof(dtype_b) * N * K)
                .wait();
        queue.memcpy(C_dev + it * M * N, C_host, sizeof(dtype_c) * M * N)
                .wait();
    }

    printf("Running gemm tests...\n");

    std::vector<float> event_times(total_iterations);

    static constexpr auto subgroup_size = 16;

    static_assert(sg_tile_k % subgroup_size == 0 && sg_tile_k >= subgroup_size);

    static constexpr auto item_tile_m = sg_tile_m;
    static constexpr auto item_tile_n = sg_tile_n / subgroup_size;

    static constexpr auto sg_per_wg_m = wg_tile_m / sg_tile_m;
    static constexpr auto sg_per_wg_n = wg_tile_n / sg_tile_n;

    static constexpr auto MM = (sg_tile_m + tM - 1) / tM;
    static constexpr auto KK = sg_tile_k / subgroup_size;
    static constexpr auto NN = (sg_tile_n + tN - 1) / tN;

    sycl::range<2> group_range_m_first {
            (M + wg_tile_m - 1) / wg_tile_m, (N + wg_tile_n - 1) / wg_tile_n};
    sycl::range<2> local_range_m_first {
            (wg_tile_m + item_tile_m - 1) / item_tile_m,
            (wg_tile_n + item_tile_n - 1) / item_tile_n};

    sycl::range<2> group_range_n_first = {
            (N + wg_tile_n - 1) / wg_tile_n, (M + wg_tile_m - 1) / wg_tile_m};
    sycl::range<2> local_range_n_first
            = {(wg_tile_n + item_tile_n - 1) / item_tile_n,
                    (wg_tile_m + item_tile_m - 1) / item_tile_m};

    sycl::nd_range<2> nd_range(wg_order_m_first
                    ? group_range_m_first * local_range_m_first
                    : group_range_n_first * local_range_n_first,
            wg_order_m_first ? local_range_m_first : local_range_n_first);

    dtype_a *A_dev_ptr = nullptr;
    dtype_b *B_dev_ptr = nullptr;
    dtype_c *C_dev_ptr = nullptr;

    for (uint32_t test = 0; test < WARMUP_ITERATIONS + testIterations; test++) {
        sycl::event ev;
#ifdef CACHE_FLUSH
        A_dev_ptr = A_dev + test * M * K;
        B_dev_ptr = B_dev + test * N * K;
        C_dev_ptr = C_dev + test * M * N;
#else
        auto A_dev_ptr = A_dev;
        auto B_dev_ptr = B_dev;
        auto C_dev_ptr = C_dev;
#endif

        static constexpr auto wg_per_wave = 64;

        ev = queue.submit([&](sycl::handler &cgh) {
            cgh.parallel_for(nd_range,
                    [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(
                            subgroup_size)]] {
                        static constexpr auto dim_m = wg_order_m_first ? 0 : 1;
                        static constexpr auto dim_n = wg_order_m_first ? 1 : 0;
                        int m, n;

                        if constexpr (snake_n != 0) {
                            static constexpr auto snake_m
                                    = wg_per_wave / snake_n;
                            auto repeat_m
                                    = wg_per_wave * (N / (snake_n * wg_tile_n));
                            m = ((id.get_group_linear_id() / repeat_m) * snake_m
                                        + (id.get_group_linear_id() % repeat_m
                                                  % wg_per_wave)
                                                / snake_n)
                                            * wg_tile_m
                                    + (get_sub_group_id() / sg_per_wg_n)
                                            * sg_tile_m;

                            n = (((id.get_group_linear_id() % repeat_m)
                                         / wg_per_wave)
                                                * snake_n
                                        + (id.get_group_linear_id() % repeat_m
                                                % wg_per_wave % snake_n))
                                            * wg_tile_n
                                    + (get_sub_group_id() % sg_per_wg_n)
                                            * sg_tile_n;
                        } else {
                            m = id.get_group(dim_m) * wg_tile_m
                                    + (get_sub_group_id() / sg_per_wg_n)
                                            * sg_tile_m;
                            n = id.get_group(dim_n) * wg_tile_n
                                    + (get_sub_group_id() % sg_per_wg_n)
                                            * sg_tile_n;
                        }
                        Tensor tAr = make_tensor<ushort>(
                                Shape<Int<sg_tile_m * KK>, Int<1>> {});
                        Tensor tBr = make_tensor<ushort>(
                                Shape<Int<sg_tile_k * 2>, Int<NN / 2>> {});
                        Tensor tCr = make_tensor<dtype_acc>(
                                Shape<Int<tM>, Int<MM>, Int<NN>> {});

                        auto A_copy = make_xe_2d_copy<XE_2D_U16x8x16x4x2_LD_N>(
                                make_tensor(make_gmem_ptr(A_dev_ptr),
                                        make_shape(M, K)));
                        auto B_copy = make_xe_2d_copy<XE_2D_U16x16x16x2x2_V>(
                                make_tensor(make_gmem_ptr(B_dev_ptr),
                                        make_shape(K, N)));
                        auto C_copy = make_xe_2d_copy<XE_2D_U32x8x16x1x1_ST_N>(
                                make_tensor(make_gmem_ptr(C_dev_ptr),
                                        make_shape(M, N)));
                        // TODO: - decide on how to deal with vector types
                        //       - create layouts with tiling/partitioning

                        Tensor tAi = make_tensor(make_inttuple_iter(m, 0),
                                make_layout(make_shape(_1 {}, _1 {}, K),
                                        make_stride(_1 {}, sg_tile_m * E<0> {},
                                                E<1> {})));
                        Tensor tBi = make_tensor(make_inttuple_iter(0, n),
                                make_layout(
                                        make_shape(_1 {}, K, Int<NN / 2> {}),
                                        make_stride(_1 {}, E<0> {},
                                                2 * tN * E<1> {})));
                        Tensor tCi = make_tensor(make_inttuple_iter(m, n),
                                make_layout(Shape<_1, Int<MM>, Int<NN>> {},
                                        make_stride(_1 {}, tM * E<0> {},
                                                tN * E<1> {})));
                        TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
                                Layout<Shape<_1, _1, _1>>>
                                tiled_mma;

                        uint32_t prefetch_k = 0;
#ifdef PREFETCH_DEFAULT
                        for (uint8_t p = 0; p < PREFETCH_DISTANCE; p++) {
                            prefetch(A_copy, tAi(_, _, prefetch_k));
                            prefetch(B_copy, tBi(_, prefetch_k, _));
                            prefetch_k += sg_tile_k;
                        }
#endif

                        uint32_t k_end = K + sg_tile_k - 1;
                        for (uint32_t k = 0; k < k_end; k += sg_tile_k) {
                            copy(A_copy, tAi(_, _, k), tAr);
                            copy(B_copy, tBi(_, k, _), tBr);

#ifdef PREFETCH_DEFAULT
                            prefetch(A_copy, tAi(_, _, prefetch_k));
                            prefetch(B_copy, tBi(_, prefetch_k, _));
                            prefetch_k += sg_tile_k;
#endif
                            auto tAr_view = make_tensor(
                                    static_cast<decltype(tAr) &&>(tAr).data(),
                                    Shape<Int<tM>, Int<MM>, Int<KK>> {});
                            auto tBr_view = make_tensor(
                                    static_cast<decltype(tBr) &&>(tBr).data(),
                                    Shape<Int<tK>, Int<KK>, Int<NN>> {});
#pragma unroll
                            for (uint8_t kl = 0; kl < KK; kl++) {
                                gemm(tiled_mma, tAr_view(_, _, kl),
                                        tBr_view(_, kl, _), tCr);
                            }
                        }

                        copy(C_copy, tCr, tCi);
                    });
        });

        ev.wait_and_throw();
        event_times[test] = time_event(ev) / 1e9; // seconds
    }

    fflush(stdout);

    get_gemm_gold<dtype_a, dtype_b, dtype_acc>(M, N, K, mem_layout::row_major,
            mem_layout::row_major, (dtype_a *)A_host, (dtype_b *)B_host,
            (dtype_c *)C_ref_host);

    auto C_host_validate
            = (dtype_c *)std::malloc(M * N * sizeof(dtype_c) * mem_iter);
    queue.memcpy(C_host_validate, C_dev, M * N * sizeof(dtype_c) * mem_iter)
            .wait();

    int i = 0;
    for (i = 0; i < mem_iter; i++) {
        auto pass_rate
                = check_results(M, N, C_host_validate + i * M * N, C_ref_host);
        if (pass_rate <= 0.99f) {
            printf("Validation failed at iter %d, pass rate: %f\n", i,
                    pass_rate);
            break;
        }
    }
    if (i == mem_iter) { printf("Validation passed\n"); }

    auto best = 999.f;
    auto worst = 0.f;
    double average = 0.f;

    auto best_iter = 0;
    auto worst_iter = 0;

    for (uint32_t i = WARMUP_ITERATIONS; i < total_iterations; i++) {
#if 1
        printf("GPU time is %f ms, Tflops is: %f, HBM (GBs) is %f\n",
                event_times[i] / 1e3, 2.0 * M * N * K / 1e12 / event_times[i],
                (M * K * sizeof(dtype_a) + K * N * sizeof(dtype_b)
                        + M * N * sizeof(dtype_c))
                        / event_times[i] / 1e9);
#endif
        average += event_times[i];
        best = min(best, event_times[i]);
        worst = max(worst, event_times[i]);
        if (best == event_times[i]) { best_iter = i; }
        if (worst == event_times[i]) { worst_iter = i; }
    }
    average = average - best - worst;
    average /= (testIterations - 2);
    auto tflo = 2.0 * M * N * K / 1e12;
    auto hbm = (M * K * sizeof(dtype_a) + K * N * sizeof(dtype_b)
                       + M * N * sizeof(dtype_c))
            / 1e9;
    printf("Performance result:\n");
    printf("    Best at iter %d, %f ms; Worst at iter %d, %f ms\n", best_iter,
            best, worst_iter, worst);
    printf("    Tflops  [min: %f, max: %f, average: %f]\n", tflo / worst,
            tflo / best, tflo / average);
    printf("    HBM(GBs)[min: %f, max: %f, average: %f]\n", hbm / worst,
            hbm / best, hbm / average);

    std::free(A_host);
    std::free(B_host);
    std::free(C_host);
    std::free(C_host_validate);
    std::free(C_ref_host);
    free(A_dev, queue);
    free(B_dev, queue);
    free(C_dev, queue);

    queue.wait_and_throw();

    printf("Done!!!\n");
}

int main(int argc, char **argv) {
    // M, K, N
    cute_gemm<256, 256, 32, 64, 32, false>(4096, 4096, 4096);
    cute_gemm<256, 256, 32, 64, 32, true>(4096, 4096, 4096);

    cute_gemm<256, 256, 32, 64, 32, false>(8192, 8192, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(8192, 8192, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(1, 5120, 13824);
    cute_gemm<256, 256, 32, 64, 32, true>(1, 5120, 13824);

    cute_gemm<256, 256, 32, 64, 32, false>(1024, 28672, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(1024, 28672, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(3072, 4096, 3072);
    cute_gemm<256, 256, 32, 64, 32, true>(3072, 4096, 3072);

    cute_gemm<256, 256, 32, 64, 32, false>(4, 4096, 12288);
    cute_gemm<256, 256, 32, 64, 32, true>(4, 4096, 12288);

    //shape from habana
    cute_gemm<256, 256, 32, 64, 32, false>(512, 8192, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(512, 8192, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(512, 8192, 32768);
    cute_gemm<256, 256, 32, 64, 32, true>(512, 8192, 32768);

    cute_gemm<256, 256, 32, 64, 32, false>(512, 32768, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(512, 32768, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(16384, 8192, 1024);
    cute_gemm<256, 256, 32, 64, 32, true>(16384, 8192, 1024);

    cute_gemm<256, 256, 32, 64, 32, false>(16384, 1024, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(16384, 1024, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(16384, 8192, 4096);
    cute_gemm<256, 256, 32, 64, 32, true>(16384, 8192, 4096);

    cute_gemm<256, 256, 32, 64, 32, false>(16384, 4096, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(16384, 4096, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(4096, 16384, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(4096, 16384, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(8192, 16384, 4096);
    cute_gemm<256, 256, 32, 64, 32, true>(8192, 16384, 4096);

    cute_gemm<256, 256, 32, 64, 32, false>(1024, 16384, 8192);
    cute_gemm<256, 256, 32, 64, 32, true>(1024, 16384, 8192);

    cute_gemm<256, 256, 32, 64, 32, false>(8192, 16384, 1024);
    cute_gemm<256, 256, 32, 64, 32, true>(8192, 16384, 1024);

    cute_gemm<256, 256, 32, 64, 32, false>(8, 128, 16384);
    cute_gemm<256, 256, 32, 64, 32, true>(8, 128, 16384);

    cute_gemm<256, 256, 32, 64, 32, false>(8, 16384, 128);
    cute_gemm<256, 256, 32, 64, 32, true>(8, 16384, 128);

    cute_gemm<256, 256, 32, 64, 32, false>(32768, 128, 4096);
    cute_gemm<256, 256, 32, 64, 32, true>(32768, 128, 4096);

    cute_gemm<256, 256, 32, 64, 32, false>(32768, 4096, 128);
    cute_gemm<256, 256, 32, 64, 32, true>(32768, 4096, 128);

    cute_gemm<256, 256, 32, 64, 32, false>(4096, 4096, 128);
    cute_gemm<256, 256, 32, 64, 32, true>(4096, 4096, 128);

    // large N
    cute_gemm<256, 256, 32, 64, 32, true, 0>(4096, 4096, 4096 * 16);
    cute_gemm<256, 256, 32, 64, 32, true, 32>(4096, 4096, 4096 * 16);
    cute_gemm<256, 256, 32, 64, 32, true, 64>(4096, 4096, 4096 * 16);

    return 0;
}
