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
#include "pvc_prefetch.hpp"
#include <cute/numeric/arithmetic_tuple.hpp>
#include <cute/tensor.hpp>

using test_clock = std::chrono::high_resolution_clock;

using namespace cute;

using dtype_a = bfloat16_t;
using dtype_b = bfloat16_t;
using dtype_c = float;
using dtype_acc = float;

size_t testIterations = 50;
dtype_acc threshold = 0.01f;

#define B_VNNI 0

#define WARMUP_ITERATIONS 0

#define PREFETCH_DISTANCE 1

#define split_barrier_arrive() __builtin_IB_work_group_barrier_arrive(0)
#define split_barrier_wait() __builtin_IB_work_group_barrier_wait(0)

template <typename T>
static void init_matrix(T *M, size_t numRows, size_t numCols) {
  std::random_device dev;
  std::mt19937 rng(dev());
  std::uniform_real_distribution<float> dist(-1.0, 1.0);
  for (size_t r = 0; r < numRows; r++) {
    for (size_t c = 0; c < numCols; c++) {
      M[r * numCols + c] = bfloat16_t(dist(rng));
    }
  }
}

template <typename T>
static void vnni_matrix(T *dst, const T *src, size_t numRows, size_t numCols,
                        size_t factor) {
  for (size_t r = 0; r < numRows / factor; r++) {
    for (size_t c = 0; c < numCols; c++) {
      for (size_t k = 0; k < factor; k++) {
        dst[r * numCols * factor + c * factor + k] =
            src[(r * factor + k) * numCols + c];
      }
    }
  }
}

template <typename T>
void check_results(size_t M, size_t N, const T *C, const T *C_ref) {
  float err = 0.f;
  size_t error_cnt = 0;
  for (size_t m = 0; m < M; m++) {
    for (size_t n = 0; n < N; n++) {
      auto index = m * N + n;
      auto localErr = std::fabs(C[index] - C_ref[index]) /
                      std::max(std::fabs(C[index]), std::fabs(C_ref[index]));
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

  auto pass_rate = (1.f - ((float)error_cnt / (M * N))) * 100; // %

  std::cout << "\n\n==== Pass rate is: " << pass_rate << "% !!!\n" << std::endl;
}

inline size_t time_event(sycl::event &e) {
  // get start and end times
  cl_ulong start_time = e.template get_profiling_info<
      sycl::info::event_profiling::command_start>();

  cl_ulong end_time =
      e.template get_profiling_info<sycl::info::event_profiling::command_end>();

  // return the delta
  return static_cast<size_t>(end_time - start_time);
}

template <int wg_tile_m, int wg_tile_n, int sg_tile_m, int sg_tile_n,
          int sg_tile_k, int item_tile_m>
void cute_gemm(size_t M, size_t K, size_t N) {

  auto queue = sycl::queue{{sycl::property::queue::enable_profiling()}};
  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();

  dtype_a *A_host = (dtype_a *)syclcompat::malloc_host(sizeof(dtype_a) * M * K);
  dtype_b *B_host = (dtype_b *)syclcompat::malloc_host(sizeof(dtype_b) * N * K);
  dtype_c *C_host = (dtype_c *)syclcompat::malloc_host(sizeof(dtype_c) * M * N);

  dtype_a *A_dev =
      (dtype_a *)sycl::malloc_device(sizeof(dtype_a) * M * K, device, context);
  dtype_b *B_dev =
      (dtype_b *)sycl::malloc_device(sizeof(dtype_b) * N * K, device, context);
  dtype_acc *C_dev = (dtype_acc *)sycl::malloc_device(sizeof(dtype_c) * M * N,
                                                      device, context);

  printf("Initializing source matrices...\n");
  init_matrix(A_host, M, K);
  init_matrix(B_host, K, N);

  dtype_b *Bvnni_host =
      (dtype_b *)syclcompat::malloc_host(sizeof(dtype_b) * N * K);
  vnni_matrix(Bvnni_host, B_host, K, N, 2);

  queue.memcpy(A_dev, A_host, sizeof(dtype_a) * M * K).wait();
  queue.memcpy(B_dev, Bvnni_host, sizeof(dtype_b) * N * K).wait();
  queue.memcpy(C_dev, C_host, sizeof(dtype_c) * M * N).wait();

  printf("Computing reference...\n");
  dtype_acc *C_ref_host =
      (dtype_acc *)syclcompat::malloc_host(sizeof(dtype_acc) * M * N);

  get_gemm_gold<dtype_a, dtype_b, dtype_acc>(
      M, N, K, mem_layout::row_major, mem_layout::row_major, (dtype_a *)A_host,
      (dtype_b *)B_host, (dtype_c *)C_ref_host);

  printf("Running gemm tests, MKN: (%d, %d, %d)...\n", M, K, N);

  const uint32_t total_iterations = WARMUP_ITERATIONS + testIterations;

  std::vector<float> event_times(total_iterations);

  static constexpr auto subgroup_size = 16;

  static_assert(sg_tile_k % subgroup_size == 0 && sg_tile_k >= subgroup_size);

  static constexpr auto item_tile_n = sg_tile_n / subgroup_size;

  static constexpr auto sg_per_wg_m = wg_tile_m / sg_tile_m;
  static constexpr auto sg_per_wg_n = wg_tile_n / sg_tile_n;

  static constexpr auto tM = 8;
  static constexpr auto tK = 16;
  static constexpr auto tN = 16;

  static constexpr auto MM = (sg_tile_m + tM - 1) / tM;
  static constexpr auto KK = sg_tile_k / subgroup_size;
  static constexpr auto NN = (sg_tile_n + tN - 1) / tN;

  sycl::range<2> group_range{(N + wg_tile_n - 1) / wg_tile_n,
                             (M + wg_tile_m - 1) / wg_tile_m};
  sycl::range<2> local_range{(wg_tile_n + item_tile_n - 1) / item_tile_n,
                             (wg_tile_m + item_tile_m - 1) / item_tile_m};
  sycl::nd_range<2> nd_range(group_range * local_range, local_range);

  for (uint32_t test = 0; test < total_iterations; test++) {
    sycl::event ev;
    ev = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(
          nd_range, [=](sycl::nd_item<2> id) [[sycl::reqd_sub_group_size(
                        subgroup_size)]] {
            const int m = id.get_group(1) * wg_tile_m +
                          (get_sub_group_id() / sg_per_wg_n) * sg_tile_m;
            const int n = id.get_group(0) * wg_tile_n +
                          (get_sub_group_id() % sg_per_wg_n) * sg_tile_n;

            Tensor tAr =
                make_tensor<ushort>(Shape<Int<sg_tile_m * KK>, Int<1>>{});
            Tensor tBr =
                make_tensor<ushort>(Shape<Int<KK * sg_tile_n / NN>, Int<NN>>{});
            Tensor tCr =
                make_tensor<dtype_acc>(Shape<Int<tM>, Int<MM>, Int<NN>>{});

            auto A_copy = make_xe_2d_copy<XE_2D_U16x8x16x4x2_LD_N>(
                make_tensor(make_gmem_ptr(A_dev), make_shape(M, K)));
            auto B_copy = make_xe_2d_copy<XE_2D_U16x16x16x2x1_LD_N>(
                make_tensor(make_gmem_ptr(B_dev), make_shape(K, N)));
            auto C_copy = make_xe_2d_copy<XE_2D_U32x8x16x1x1_ST_N>(
                make_tensor(make_gmem_ptr(C_dev), make_shape(M, N)));
            // TODO: - decide on how to deal with vector types
            //       - create layouts with tiling/partitioning

            Tensor tAi = make_tensor(
                make_inttuple_iter(m, 0),
                make_layout(make_shape(_1{}, _1{}, K),
                            make_stride(_1{}, MM * tM * E<0>{}, E<1>{})));
            Tensor tBi = make_tensor(
                make_inttuple_iter(0, n),
                make_layout(make_shape(_1{}, K, Int<NN>{}),
                            make_stride(_1{}, E<0>{}, tN * E<1>{})));
            Tensor tCi = make_tensor(
                make_inttuple_iter(m, n),
                make_layout(Shape<_1, Int<MM>, Int<NN>>{},
                            make_stride(_1{}, tM * E<0>{}, tN * E<1>{})));
            TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
                     Layout<Shape<_1, _1, _1>>>
                tiled_mma;

            uint32_t prefetch_k = 0;
#ifdef PREFETCH_DEFAULT
            for (uint32_t p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
              HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
              ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#else
                HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#endif
              HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
              ((ushort *)A_dev, tM, M, K, m, prefetch_k);
              prefetch_k += tK * KK;
            }
#endif

            for (int k = 0; k < K + tK * KK - 1; k += tK * KK) {
              copy(A_copy, tAi(_, _, k), tAr);
              copy(B_copy, tBi(_, k / KK, _), tBr);

#ifdef PREFETCH_DEFAULT
              for (uint32_t p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
                HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
                ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#else
                  HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                  ((ushort *)B_dev, tN, K, N, prefetch_k, n);
#endif
                HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)A_dev, tM, M, K, m, prefetch_k);
                prefetch_k += tK * KK;
              }
#endif
              auto tAr_view =
                  make_tensor(static_cast<decltype(tAr) &&>(tAr).data(),
                              Shape<Int<tM>, Int<MM>, Int<KK>>{});
              auto tBr_view =
                  make_tensor(static_cast<decltype(tBr) &&>(tBr).data(),
                              Shape<Int<tK>, Int<KK>, Int<NN>>{});
              for (uint32_t kl = 0; kl < KK; kl++) {
                gemm(tiled_mma, tAr_view(_, _, kl), tBr_view(_, kl, _), tCr);
              }
            }

            copy(C_copy, tCr, tCi);
          });
    });

    ev.wait_and_throw();
    event_times[test] = time_event(ev) / 1e9; // seconds
  }

  double average_event_time = 0.f;
  auto best = 999.f;
  for (uint32_t i = WARMUP_ITERATIONS; i < total_iterations; i++) {
#if 0
    printf("GPU time is %f ms, Tflops is: %f, HBM (GBs) is %f\n",
           event_times[i] / 1e3, 2.0 * M * N * K / 1e12 / event_times[i],
           (M * K * sizeof(dtype_a) + K * N * sizeof(dtype_b) +
            M * N * sizeof(dtype_c)) /
               event_times[i] / 1e9);
#endif
    average_event_time += event_times[i];
    best = min(best, event_times[i]);
  }
  average_event_time /= testIterations;
  printf("MKN (%d, %d, %d), Best is %f ms, %f Tflops, %f HBM (GBs)\n", M, K, N,
         best * 1e3, 2.0 * M * N * K / 1e12 / best,
         (M * K * sizeof(dtype_a) + K * N * sizeof(dtype_b) +
          M * N * sizeof(dtype_c)) /
             best / 1e9);

  printf("Checking results... ");
  fflush(stdout);

  auto C_host_validate =
      (dtype_c *)sycl::malloc_host(M * N * sizeof(dtype_c), queue);
  queue.memcpy(C_host_validate, C_dev, M * N * sizeof(dtype_c)).wait();
  check_results(M, N, C_host_validate, C_ref_host);

  free(A_host, queue);
  free(B_host, queue);
  free(C_host, queue);
  free(Bvnni_host, queue);
  free(C_host_validate, queue);
  free(C_ref_host, queue);
  free(A_dev, queue);
  free(B_dev, queue);
  free(C_dev, queue);

  printf(" done!\n");
}

int main(int argc, char **argv) {
  // M, K, N
  cute_gemm<256, 256, 32, 64, 32, 32>(2048, 2048, 2048);
  cute_gemm<256, 256, 32, 64, 32, 32>(4096, 4096, 4096);
  cute_gemm<256, 256, 32, 64, 32, 32>(8192, 8192, 8192);

  cute_gemm<32, 512, 32, 32, 32, 32>(1, 5120, 5120);
  cute_gemm<32, 512, 32, 32, 32, 32>(1, 13824, 5120);
  cute_gemm<32, 512, 32, 32, 32, 32>(1, 5120, 13824);

  cute_gemm<32, 512, 32, 32, 32, 32>(4, 8192, 2048);
  cute_gemm<32, 512, 32, 32, 32, 32>(4, 4096, 250880);
  cute_gemm<32, 512, 32, 32, 32, 32>(4, 16384, 4096);
  cute_gemm<32, 512, 32, 32, 32, 32>(4, 4096, 12288);
  cute_gemm<32, 512, 32, 32, 32, 32>(4, 14336, 5376);

  cute_gemm<32, 512, 32, 32, 32, 32>(256, 4096, 4096);
  cute_gemm<32, 512, 32, 32, 32, 32>(512, 379, 2043);
  cute_gemm<32, 512, 32, 32, 32, 32>(1024, 28672, 8192);
  cute_gemm<32, 512, 32, 32, 32, 32>(8192, 1024, 4096);

  return 0;
}
