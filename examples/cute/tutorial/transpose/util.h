#pragma once

#include "../../../common/sycl_cute_common.hpp"
#include <iomanip>

template <typename T> struct TransposeParams {
  T *__restrict__ input;
  T *__restrict__ output;

  const int M;
  const int N;

  TransposeParams(T *__restrict__ input_, T *__restrict__ output_, int M_,
                  int N_)
      : input(input_), output(output_), M(M_), N(N_) {}
};

template <typename T, bool isTranspose = true, bool isFMA = false,
          bool is_random = true>
int benchmark(void (*transpose)(TransposeParams<T> params), int M, int N,
              int iterations = 10, bool verify = true) {
  using namespace cute;

  auto tensor_shape_S = make_shape(M, N);
  auto tensor_shape_D = (isTranspose) ? make_shape(N, M) : make_shape(M, N);

  //
  // Allocate and initialize
  //
  std::vector<T> h_S(size(tensor_shape_S));
  std::vector<T> h_D(size(tensor_shape_D));

  auto d_S = compat::malloc<T>(size(tensor_shape_S));
  auto d_D = compat::malloc<T>(size(tensor_shape_D));

  if (not is_random) {
    for (size_t i = 0; i < h_S.size(); ++i) {
      h_S[i] = static_cast<T>(i);
    }
  } else {
    random_fill<T>(h_S);
  }

  compat::memcpy<T>(d_S, h_S.data(), size(tensor_shape_S));

  TransposeParams<T> params(d_S, d_D, M, N);

  for (int i = 0; i < iterations; i++) {
    auto t1 = std::chrono::high_resolution_clock::now();
    transpose(params);
    compat::wait_and_throw();
    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> tDiff = t2 - t1;
    double time_ms = tDiff.count();
    double M_ = double(M);
    double N_ = double(N);
    double bytes = 2 * M_ * N_ * sizeof(T);

    std::cout << "Trial " << i << " Completed in " << time_ms << "ms ("
              << std::fixed << std::setprecision(2) << 1e-6 * bytes / time_ms
              << " GB/s)" << std::endl;
  }

  if (verify) {
    compat::memcpy<T>(h_D.data(), d_D, size(tensor_shape_D));

    int bad = 0;
    if constexpr (isTranspose) {
      auto transpose_function = make_layout(tensor_shape_S, LayoutRight{});
      for (size_t i = 0; i < h_D.size(); ++i)
        if (h_D[i] != h_S[transpose_function(i)])
          bad++;
    } else {
      for (size_t i = 0; i < h_D.size(); ++i)
        if (h_D[i] != h_S[i])
          bad++;
    }
#if 0
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        std::cout << (int)h_S[i * N + j] << "\t";
      }
      std::cout << std::endl;
    }
    std::cout << std::endl;
    for (size_t i = 0; i < M; ++i) {
      for (size_t j = 0; j < N; ++j) {
        std::cout << (int)h_D[i * N + j] << "\t";
      }
      std::cout << std::endl;
    }

#endif

    if (bad > 0) {
      std::cout << "Validation failed. Correct values: " << h_D.size() - bad
                << ". Incorrect values: " << bad << std::endl;
    } else {
      std::cout << "Validation success." << std::endl;
    }
  }
  return 0;
}
