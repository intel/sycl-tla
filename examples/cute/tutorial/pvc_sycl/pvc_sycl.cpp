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

int testIterations = 10;
dtype_acc threshold = 0.01f;
size_t matrixSize = 4096;

#define B_VNNI

#define WARMUP_ITERATIONS 100

#define PREFETCH_DISTANCE 1

#define split_barrier_arrive() __builtin_IB_work_group_barrier_arrive(0)
#define split_barrier_wait() __builtin_IB_work_group_barrier_wait(0)

template <typename T>
static void fill_matrix(T *M, size_t numRows, size_t numCols) {
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
static void fill_matrix_B(T *M, size_t numRows, size_t numCols) {
  for (size_t r = 0; r < numRows; r++) {
    for (size_t c = 0; c < numCols; c++) {
      M[r * numCols + c] = bfloat16_t(0.0f);
    }
  };

  for (size_t r = 0; r < numRows; r++) {
    M[r * numCols + r] = bfloat16_t(1.0f);
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
        // std::cerr << "Error at m = " << m << ", n = " << n << ": (local
        // error"
        //           << localErr << "): Wanted " << C_ref[index] << ", got "
        //          << C[index] << std::endl;
        // return;
      }
    }
  }

  auto fail_rate = (float)error_cnt * 100 / (M * N);

  std::cout << "\n\n==== fail points %d  is: " << fail_rate << "% !!!\n"
            << std::endl;
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

template <int tM, int tN, int tK, int MM, int NN>
static void go_dpas_blockread_vnni_tiled(sycl::queue queue, dtype_acc *C,
                                         dtype_a *A, dtype_b *B, size_t M,
                                         size_t N, size_t K, dtype_acc *C_ref) {
  int total_iterations = WARMUP_ITERATIONS + testIterations;

  std::vector<float> event_times(total_iterations);

  sycl::range<2> group_range{(M + WG_SIZE_Y - 1) / WG_SIZE_Y,
                             (N + WG_SIZE_X - 1) / WG_SIZE_X};
  sycl::range<2> local_range{(WG_SIZE_Y + ITEM_SIZE_Y - 1) / ITEM_SIZE_Y,
                             (WG_SIZE_X + ITEM_SIZE_X - 1) / ITEM_SIZE_X};
  sycl::nd_range<2> nd_range(group_range * local_range, local_range);

  for (int test = 0; test < total_iterations; test++) {
    sycl::event ev;
    ev = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(nd_range, [=](sycl::nd_item<2>
                                         id) [[sycl::reqd_sub_group_size(16)]] {
        const int M = id.get_global_range(0) * ITEM_SIZE_Y;
        const int N = id.get_global_range(1) * ITEM_SIZE_X;
        const int m = id.get_group(0) * WG_SIZE_Y +
                      (get_sub_group_id() / SGS_PER_WG_X) * SG_SIZE_Y;
        const int n = id.get_group(1) * WG_SIZE_X +
                      (get_sub_group_id() % SGS_PER_WG_X) * SG_SIZE_X;

        Tensor tAr = make_tensor<ushort>(Shape<Int<8 * KK * MM>, Int<1>>{});
        Tensor tBr = make_tensor<ushort>(Shape<Int<16 * KK>, Int<NN>>{});
        Tensor tCr = make_tensor<dtype_acc>(Shape<_8, Int<MM>, Int<NN>>{});

        auto A_copy =
            make_xe_2d_A_copy(make_tensor(make_gmem_ptr(A), make_shape(M, K)));
        auto B_copy =
            make_xe_2d_B_copy(make_tensor(make_gmem_ptr(B), make_shape(K, N)));
        auto C_copy =
            make_xe_2d_copy(make_tensor(make_gmem_ptr(C), make_shape(M, N)));
        // TODO: - decide on how to deal with vector types
        //       - create layouts with tiling/partitioning

        Tensor tAi = make_tensor(
            make_inttuple_iter(m, 0),
            make_layout(make_shape(_1{}, _1{}, K),
                        make_stride(_1{}, MM * tM * E<0>{}, E<1>{})));
        Tensor tBi =
            make_tensor(make_inttuple_iter(0, n),
                        make_layout(make_shape(_1{}, K, Int<NN>{}),
                                    make_stride(_1{}, E<0>{}, tN * E<1>{})));
        Tensor tCi = make_tensor(
            make_inttuple_iter(m, n),
            make_layout(Shape<_1, Int<MM>, Int<NN>>{},
                        make_stride(_1{}, tM * E<0>{}, tN * E<1>{})));
        TiledMMA<MMA_Atom<XE_8x16x16_BF16BF16F32F32_NN>,
                 Layout<Shape<_1, _1, _1>>>
            tiled_mma;

        int prefetch_k = 0;
#ifdef PREFETCH_DEFAULT
        for (int p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
          HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
          ((ushort *)B, tN, K, N, prefetch_k, n);
#else
                HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                ((ushort *)B, tN, K, N, prefetch_k, n);
#endif
          HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
          ((ushort *)A, tM, M, K, m, prefetch_k);
          prefetch_k += tK * KK;
        }
#endif

        for (int k = 0; k < K; k += tK * KK) {
          copy(A_copy, tAi(_, _, k), tAr);
          copy(B_copy, tBi(_, k / 2, _), tBr);

#ifdef PREFETCH_DEFAULT
          for (int p = 0; p < PREFETCH_DISTANCE; p++) {
#ifdef B_VNNI
            HELPER_NAME(btile_block_prefetch_vnni, 4, 4)
            ((ushort *)B, tN, K, N, prefetch_k, n);
#else
                  HELPER_NAME(btile_block_prefetch_rowmajor, 4, 4)
                  ((ushort *)B, tN, K, N, prefetch_k, n);
#endif
            HELPER_NAME(atile_block_prefetch_rowmajor, 4, 4)
            ((ushort *)A, tM, M, K, m, prefetch_k);
            prefetch_k += tK * KK;
          }
#endif
          auto tAr_view = make_tensor(static_cast<decltype(tAr) &&>(tAr).data(),
                                      Shape<_8, Int<MM>, Int<KK>>{});
          auto tBr_view = make_tensor(static_cast<decltype(tBr) &&>(tBr).data(),
                                      Shape<_16, Int<KK>, Int<NN>>{});
          for (int kl = 0; kl < KK; kl++) {
            gemm(tiled_mma, tAr_view(_, _, kl), tBr_view(_, kl, _), tCr);
          }
        }

        copy(C_copy, tCr, tCi);
      });
    });

    ev.wait_and_throw();
    event_times[test] = time_event(ev) / 1e6;
  }

  double average_event_time = 0.f;
  auto best = 999.f;
  for (int i = WARMUP_ITERATIONS; i < total_iterations; i++) {
    printf("GPU time is %f ms, Gflops is: %f\n", event_times[i],
           2.0 * M * N * K / 1e9 / (event_times[i] / 1e3));
    average_event_time += event_times[i];
    best = min(best, event_times[i]);
  }
  average_event_time /= testIterations;
  printf("Average is %f gflops, best is %f gflops\n",
         2.0 * M * N * K / 1e9 / (average_event_time / 1e3),
         2.0 * M * N * K / 1e9 / (best / 1e3));

  printf("Checking results... ");
  fflush(stdout);
  check_results(M, N, C, C_ref);
  printf(" done!\n");
}

int main(int argc, char **argv) {
  sycl::queue queue{{sycl::property::queue::enable_profiling()}};
  auto context = queue.get_info<sycl::info::queue::context>();
  auto device = queue.get_info<sycl::info::queue::device>();

  const auto M = matrixSize;
  const auto N = matrixSize;
  const auto K = matrixSize;

  dtype_a *A_vec =
      (dtype_a *)syclcompat::malloc_shared(sizeof(dtype_a) * M * K);
  dtype_b *B_vec =
      (dtype_b *)syclcompat::malloc_shared(sizeof(dtype_b) * N * K);
  dtype_b *Bvnni_vec =
      (dtype_b *)syclcompat::malloc_shared(sizeof(dtype_b) * N * K);
  dtype_acc *C_vec =
      (dtype_acc *)syclcompat::malloc_shared(sizeof(dtype_acc) * M * N);
  dtype_acc *C_ref =
      (dtype_acc *)syclcompat::malloc_shared(sizeof(dtype_acc) * M * N);

  printf("Initializing source matrices...\n");
  fill_matrix(A_vec, M, K);
  fill_matrix(B_vec, K, N);

  vnni_matrix(Bvnni_vec, B_vec, K, N, 2);

  printf("Computing reference...\n");
  get_gemm_gold<dtype_a, dtype_b, dtype_acc>(
      M, N, K, mem_layout::row_major, mem_layout::row_major, (dtype_a *)A_vec,
      (dtype_b *)B_vec, (dtype_acc *)C_ref);

  printf("Creating source buffers...\n");
  auto A = A_vec;
  auto B = B_vec;
  auto Bvnni = Bvnni_vec;

  printf("Running gemm tests, MKN: (%d, %d, %d)...\n", M, K, N);

#ifdef B_VNNI
  go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 4>(queue, C_vec, A, Bvnni, M, N, K,
                                                C_ref);
#else
  go_dpas_blockread_vnni_tiled<8, 16, 16, 4, 4>(queue, C_vec, A, B, M, N, K,
                                                C_ref);
#endif

  printf("Done.\n");

  free(A_vec, queue);
  free(B_vec, queue);
  free(C_vec, queue);
  free(Bvnni_vec, queue);
  free(C_ref, queue);

  return 0;
}
