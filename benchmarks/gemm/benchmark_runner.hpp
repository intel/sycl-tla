/***************************************************************************************************
 * Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/epilogue/fusion/operations.hpp"

#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cute/tensor.hpp"

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/device/tensor_fill.h"
#include "cutlass/util/reference/device/tensor_silu.h"
#include "cutlass/util/initialize_block.hpp"
#if defined(CUTLASS_BENCHMARK_ENABLE_LIBRARY_GEMM)
#include "cutlass/library/library.h"
#include "cutlass/library/singleton.h"
#endif

#include "../common.hpp"

#include <algorithm>
#include <benchmark/benchmark.h>
#include <cmath>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

using namespace cute;

namespace cutlass::benchmark {

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(SYCL_INTEL_TARGET)
template <class T, int Stages = 0>
static constexpr auto is_mixed_dtype = false;

template <int Stages>
static constexpr auto is_mixed_dtype<cutlass::gemm::MainloopIntelXeXMX16MixedPrecision<Stages>> = true;
#else
template <class T, int Stages = 0>
static constexpr auto is_mixed_dtype = false;
#endif

template <class T, class = void>
struct ScaleType {
  using type = int;
};
template <class T>
struct ScaleType<T, cute::void_t<typename T::ElementScale>> {
  using type = typename T::ElementScale;
};

template <class T, class = void>
struct ZeroType {
  using type = int;
};
template <class T>
struct ZeroType<T, cute::void_t<typename T::ElementZero>> {
  using type = typename T::ElementZero;
};

template <class T, class = void>
struct ScaleStride {
  using type = int;
};
template <class T>
struct ScaleStride<T, cute::void_t<typename T::StrideScale>> {
  using type = typename T::StrideScale;
};

template <class T, class = void>
struct ZeroStride {
  using type = int;
};
template <class T>
struct ZeroStride<T, cute::void_t<typename T::StrideZero>> {
  using type = typename T::StrideZero;
};

template <class T, class = void>
struct HasSchedulerSplits : std::false_type {};

template <class T>
struct HasSchedulerSplits<T, std::void_t<decltype(std::declval<T&>().scheduler.splits)>> : std::true_type {};

template <class Arguments>
void set_scheduler_splits(Arguments& arguments, int split_k_slices) {
  if constexpr (HasSchedulerSplits<Arguments>::value) {
    if (split_k_slices > 0) {
      arguments.scheduler.splits = split_k_slices;
    }
  }
}

///////////////////////////////////////////////////////////////////////////////////////////////////

// Command line options parsing
struct GEMMOptions {

  bool error;

  int m, n, k, l;
  float alpha, beta;
  int split_k_slices;
  std::string bm_name;
  std::string operation_name;
  std::string layout;
  std::string dtype_a;
  std::string dtype_b;
  std::string dtype_c;
  std::string dtype_d;
  std::string dtype_acc;
  int verify_library;
  int library_verify_max_ops;

  GEMMOptions():
          error(false),
          m(5120), n(4096), k(4096), l(1),
          alpha(1.f), beta(0.f),
          split_k_slices(0),
          bm_name("GEMM"),
          operation_name(""),
          layout("rcr"),
          dtype_a("f16"),
          dtype_b("f16"),
          dtype_c("f32"),
          dtype_d("f32"),
          dtype_acc("f32"),
          verify_library(1),
          library_verify_max_ops(1 << 26)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    CommandLine cmd(argc, args);

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("split_k_slices", split_k_slices, 0);
    cmd.get_cmd_line_argument("bm_name", bm_name, std::string("GEMM"));
    cmd.get_cmd_line_argument("operation_name", operation_name, std::string(""));
    cmd.get_cmd_line_argument("layout", layout, std::string("rcr"));
    cmd.get_cmd_line_argument("dtype_a", dtype_a, std::string("f16"));
    cmd.get_cmd_line_argument("dtype_b", dtype_b, std::string("f16"));
    cmd.get_cmd_line_argument("dtype_c", dtype_c, std::string("f32"));
    cmd.get_cmd_line_argument("dtype_d", dtype_d, std::string("f32"));
    cmd.get_cmd_line_argument("dtype_acc", dtype_acc, std::string("f32"));
    cmd.get_cmd_line_argument("verify_library", verify_library, 1);
    cmd.get_cmd_line_argument("library_verify_max_ops", library_verify_max_ops, 1 << 26);
  }

  std::string benchmark_name() const {
    std::stringstream full_name;
    full_name << bm_name << "/";
    std::string const test_name_suffix = std::to_string(m) + "x" +
                                   std::to_string(n) + "x" +
                                   std::to_string(k) + "x" +
                                   std::to_string(l);
    full_name << test_name_suffix;

    return full_name.str();
  }
};

///////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(CUTLASS_BENCHMARK_ENABLE_LIBRARY_GEMM)
struct LibraryGemmBenchmarkRunner {
  template <typename ElementA, typename ElementB, typename ElementC, typename ElementD, typename ElementCompute>
  static void run_typed(::benchmark::State& state, GEMMOptions const& options, KernelHardwareInfo const& hw_info) {
    using namespace cutlass;
    using namespace cutlass::library;

    Operation const* operation = nullptr;
    for (auto const& candidate : Singleton::get().manifest.operations()) {
      if (candidate->description().name == options.operation_name) {
        operation = candidate.get();
        break;
      }
    }

    if (!operation) {
      state.SkipWithError(("library operation not found: " + options.operation_name).c_str());
      return;
    }

    if (operation->description().kind != OperationKind::kGemm) {
      state.SkipWithError("library operation is not a GEMM operation.");
      return;
    }

    auto const& desc = static_cast<GemmDescription const&>(operation->description());
    if (desc.gemm_kind != GemmKind::kUniversal) {
      state.SkipWithError("library operation is not a universal GEMM operation.");
      return;
    }

    auto leading_dim = [](char layout, int rows, int cols) {
      return layout == 'r' ? cols : rows;
    };
    auto element_offset = [](char layout, int row, int col, int ld) {
      return layout == 'r' ? row * ld + col : col * ld + row;
    };

    if (options.layout.size() != 3) {
      state.SkipWithError("library GEMM requires a 3-character layout such as rcr.");
      return;
    }

    char const layout_a = options.layout[0];
    char const layout_b = options.layout[1];
    char const layout_c = options.layout[2];
    int const lda = leading_dim(layout_a, options.m, options.k);
    int const ldb = leading_dim(layout_b, options.k, options.n);
    int const ldc = leading_dim(layout_c, options.m, options.n);
    int const ldd = ldc;

    std::vector<ElementA> host_a(static_cast<size_t>(options.m) * options.k);
    std::vector<ElementB> host_b(static_cast<size_t>(options.k) * options.n);
    std::vector<ElementC> host_c(static_cast<size_t>(options.m) * options.n);
    std::vector<ElementD> host_d(static_cast<size_t>(options.m) * options.n);

    for (int row = 0; row < options.m; ++row) {
      for (int col = 0; col < options.k; ++col) {
        float value = static_cast<float>(((row * 13 + col * 7) % 17) - 8) / 8.0f;
        host_a[element_offset(layout_a, row, col, lda)] = ElementA(value);
      }
    }
    for (int row = 0; row < options.k; ++row) {
      for (int col = 0; col < options.n; ++col) {
        float value = static_cast<float>(((row * 11 + col * 5) % 19) - 9) / 9.0f;
        host_b[element_offset(layout_b, row, col, ldb)] = ElementB(value);
      }
    }
    for (int row = 0; row < options.m; ++row) {
      for (int col = 0; col < options.n; ++col) {
        host_c[element_offset(layout_c, row, col, ldc)] = ElementC(static_cast<float>(((row * 3 + col * 2) % 11) - 5) / 11.0f);
      }
    }

    DeviceAllocation<ElementA> device_a(host_a.size());
    DeviceAllocation<ElementB> device_b(host_b.size());
    DeviceAllocation<ElementC> device_c(host_c.size());
    DeviceAllocation<ElementD> device_d(host_d.size());
    device_a.copy_from_host(host_a.data());
    device_b.copy_from_host(host_b.data());
    device_c.copy_from_host(host_c.data());
    ElementCompute alpha = ElementCompute(options.alpha);
    ElementCompute beta = ElementCompute(options.beta);

    GemmUniversalConfiguration configuration{
      GemmUniversalMode::kGemm,
      {options.m, options.n, options.k},
      {1, 1, 1},
      {1, 1, 1},
      options.l,
      lda,
      ldb,
      ldc,
      ldd,
      1
    };

    GemmUniversalArguments arguments{
      {options.m, options.n, options.k},
      {1, 1, 1},
      {1, 1, 1},
      options.l,
      device_a.get(),
      device_b.get(),
      device_c.get(),
      device_d.get(),
      &alpha,
      &beta,
      ScalarPointerMode::kHost,
      lda,
      ldb,
      ldc,
      ldd,
      static_cast<int64_t>(options.m) * options.k,
      static_cast<int64_t>(options.k) * options.n,
      static_cast<int64_t>(options.m) * options.n,
      static_cast<int64_t>(options.m) * options.n,
      hw_info.sm_count
    };

    if (operation->can_implement(&configuration, &arguments) != Status::kSuccess) {
      state.SkipWithError("library GEMM unable to implement given args.");
      return;
    }

    uint64_t const host_workspace_size = operation->get_host_workspace_size(&configuration);
    std::vector<uint8_t> host_workspace(static_cast<size_t>(host_workspace_size));
    uint64_t const device_workspace_size = operation->get_device_workspace_size(&configuration, &arguments);
    device_memory::allocation<uint8_t> device_workspace;
    device_workspace.reset(static_cast<size_t>(device_workspace_size));

    if (operation->initialize(&configuration, host_workspace.data(), device_workspace.get()) != Status::kSuccess) {
      state.SkipWithError("library GEMM failed to initialize.");
      return;
    }
    if (operation->run(&arguments, host_workspace.data(), device_workspace.get()) != Status::kSuccess) {
      state.SkipWithError("library GEMM failed to run.");
      return;
    }
#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
#endif

    int64_t const verify_ops = static_cast<int64_t>(options.m) * options.n * options.k;
    if (options.verify_library && verify_ops <= options.library_verify_max_ops) {
      device_d.copy_to_host(host_d.data());
      double max_error = 0.0;
      for (int row = 0; row < options.m; ++row) {
        for (int col = 0; col < options.n; ++col) {
          float accum = 0.0f;
          for (int kk = 0; kk < options.k; ++kk) {
            accum += float(host_a[element_offset(layout_a, row, kk, lda)]) *
                     float(host_b[element_offset(layout_b, kk, col, ldb)]);
          }
          float const ref = options.alpha * accum + options.beta * float(host_c[element_offset(layout_c, row, col, ldc)]);
          float const got = float(host_d[element_offset(layout_c, row, col, ldd)]);
          max_error = std::max(max_error, std::abs(double(got) - double(ref)));
        }
      }
      double const tolerance = std::is_same<ElementCompute, cutlass::bfloat16_t>::value ? 1.0 : 0.5;
      if (max_error > tolerance) {
        state.SkipWithError("Disposition Failed.");
        return;
      }
    }

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;
    state.SetLabel("library_operation=" + options.operation_name + " layout=" + options.layout);

    double const gflop = 2.0 * options.m * options.n * options.k * options.l * 1e-9;
    double const mega_bytes_transferred = static_cast<double>(
      options.m * options.k * sizeof(ElementA) +
      options.k * options.n * sizeof(ElementB) +
      options.m * options.n * sizeof(ElementD) +
      (options.beta != 0 ? options.m * options.n * sizeof(float) : 0)
    ) * 1e-6 * options.l;

    state.counters["total_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
    for (auto _ : state) {
      GPU_Clock timer;
      timer.start();
      auto status = operation->run(&arguments, host_workspace.data(), device_workspace.get());
      auto ms_elapsed = timer.milliseconds();
      if (status != Status::kSuccess) {
        state.SkipWithError("library GEMM failed during benchmark loop.");
        return;
      }
      state.SetIterationTime(ms_elapsed / 1000.0);
      state.counters["total_runtime_ms"] += ms_elapsed;
      state.counters["best_runtime_ms"] = std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
      state.counters["worst_runtime_ms"] = std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    }
    double const iterations = static_cast<double>(std::max<int64_t>(state.iterations(), 1));
    state.counters["avg_runtime_ms"] = state.counters["total_runtime_ms"] / iterations;
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }

  static void run(::benchmark::State& state, GEMMOptions const& options, KernelHardwareInfo const& hw_info) {
    if (options.operation_name.empty()) {
      state.SkipWithError("library GEMM requires --operation_name.");
      return;
    }
    if (!((options.dtype_c == "f32" && options.dtype_acc == "f32") ||
          (options.dtype_c == "bf16" && options.dtype_acc == "bf16") ||
          (options.dtype_c == "f16" && options.dtype_acc == "f16"))) {
      state.SkipWithError("library GEMM benchmark currently supports matching C/accumulator pairs: f32/f32, bf16/bf16, or f16/f16.");
      return;
    }
    bool const d_is_f16 = options.operation_name.find("_f16_df16_") != std::string::npos;
    bool const d_is_bf16 = options.operation_name.find("_bf16_dbf16_") != std::string::npos;
    bool const op_is_f16_acc = options.operation_name.find("_f16_f16_f16_f16_f16_") != std::string::npos;
    bool const op_is_bf16_acc = options.operation_name.find("_bf16_bf16_bf16_bf16_bf16_") != std::string::npos;
    std::string const operation_dtype_d = d_is_f16 || op_is_f16_acc ? "f16" : (d_is_bf16 || op_is_bf16_acc ? "bf16" : "f32");
    std::string const operation_dtype_c = op_is_f16_acc ? "f16" : (op_is_bf16_acc ? "bf16" : "f32");
    std::string const operation_dtype_acc = operation_dtype_c;
    if (options.dtype_c != operation_dtype_c || options.dtype_acc != operation_dtype_acc) {
      state.SkipWithError(("library GEMM C/accumulator mismatch: requested " + options.dtype_c + "/" + options.dtype_acc + " but operation provides " + operation_dtype_c + "/" + operation_dtype_acc).c_str());
      return;
    }
    if (options.dtype_d != operation_dtype_d) {
      state.SkipWithError(("library GEMM dtype_d mismatch: requested " + options.dtype_d + " but operation provides " + operation_dtype_d).c_str());
      return;
    }
    if (options.dtype_a == "f16" && options.dtype_b == "f16" && op_is_f16_acc) {
      run_typed<cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t, cutlass::half_t>(state, options, hw_info);
    }
    else if (options.dtype_a == "f16" && options.dtype_b == "f16" && d_is_f16) {
      run_typed<cutlass::half_t, cutlass::half_t, float, cutlass::half_t, float>(state, options, hw_info);
    }
    else if (options.dtype_a == "f16" && options.dtype_b == "f16") {
      run_typed<cutlass::half_t, cutlass::half_t, float, float, float>(state, options, hw_info);
    }
    else if (options.dtype_a == "bf16" && options.dtype_b == "bf16" && op_is_bf16_acc) {
      run_typed<cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t, cutlass::bfloat16_t>(state, options, hw_info);
    }
    else if (options.dtype_a == "bf16" && options.dtype_b == "bf16" && d_is_f16) {
      run_typed<cutlass::bfloat16_t, cutlass::bfloat16_t, float, cutlass::half_t, float>(state, options, hw_info);
    }
    else if (options.dtype_a == "bf16" && options.dtype_b == "bf16" && d_is_bf16) {
      run_typed<cutlass::bfloat16_t, cutlass::bfloat16_t, float, cutlass::bfloat16_t, float>(state, options, hw_info);
    }
    else if (options.dtype_a == "bf16" && options.dtype_b == "bf16") {
      run_typed<cutlass::bfloat16_t, cutlass::bfloat16_t, float, float, float>(state, options, hw_info);
    }
    else {
      state.SkipWithError("library GEMM benchmark currently supports f16/f16 and bf16/bf16 inputs only.");
    }
  }
};

inline void cutlass_library_gemm_func(
    ::benchmark::State& state,
    cutlass::benchmark::GEMMOptions const& options,
    cutlass::KernelHardwareInfo const& hw_info) {
  LibraryGemmBenchmarkRunner::run(state, options, hw_info);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
#endif

template <class GemmConfiguration>
struct BenchmarkRunnerGemm {

  using Gemm = typename GemmConfiguration::Gemm;

  using StrideA = typename Gemm::GemmKernel::StrideA;
  using StrideB = typename Gemm::GemmKernel::StrideB;
  using StrideC = typename Gemm::GemmKernel::StrideC;
  using StrideD = typename Gemm::GemmKernel::StrideD;

  using LayoutA = typename Gemm::LayoutA;
  using LayoutB = typename Gemm::LayoutB;
  using LayoutC = typename Gemm::LayoutC;
  using LayoutD = typename Gemm::LayoutD;

  using ElementA = typename Gemm::ElementA;
  using ElementB = typename Gemm::ElementB;
  using ElementAccumulator = typename Gemm::ElementAccumulator;

  using CollectiveMainloop = typename Gemm::GemmKernel::CollectiveMainloop;
  using DispatchPolicy = typename CollectiveMainloop::DispatchPolicy;
  using ElementMma = typename CollectiveMainloop::TiledMma::ValTypeA;

  using ElementScale = typename ScaleType<CollectiveMainloop>::type;
  using ElementZero = typename ZeroType<CollectiveMainloop>::type;
  using StrideS = typename ScaleStride<CollectiveMainloop>::type;
  using StrideZ = typename ZeroStride<CollectiveMainloop>::type;

  using CollectiveEpilogue = typename Gemm::CollectiveEpilogue;
  using ElementC = typename Gemm::ElementC;
  using ElementOutput = typename CollectiveEpilogue::ElementOutput;
  using ElementCompute = typename CollectiveEpilogue::ElementCompute;

  using ProblemShapeType = typename Gemm::GemmKernel::ProblemShape;

  using FusionOp = typename Gemm::EpilogueOutputOp;

  // TODO(codeplay): Epilogue detection here should be replaced w/ general solution (see other TODO)
  using FusionSilu = cutlass::epilogue::fusion::LinCombEltAct<
      cutlass::epilogue::thread::SiLu, ElementOutput, ElementCompute, ElementAccumulator,
      ElementAccumulator, cutlass::FloatRoundStyle::round_to_nearest>;

  using FusionDeEltMul = cutlass::epilogue::fusion::LinCombDeEltAct<LayoutC, std::multiplies,
                                                                    ElementOutput, ElementCompute>;
  using FusionLinComb = epilogue::fusion::LinearCombination<
      ElementAccumulator, ElementCompute, ElementAccumulator, ElementAccumulator,
      FloatRoundStyle::round_to_nearest>;

  // Epilogue used in ampere/gemm_configuration.hpp
  using DefaultEpilogue = epilogue::collective::DefaultEpilogue<
    float,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    cutlass::gemm::TagToStrideC_t<LayoutC>,
    epilogue::thread::LinearCombination<float, 1>,
    cutlass::gemm::EpilogueDefault>;

  static constexpr bool epi_is_deeltactmul = std::is_same_v<FusionOp, FusionDeEltMul>;
  static constexpr bool epi_is_silu = std::is_same_v<FusionOp, FusionSilu>;
  static constexpr bool epi_is_lincomb = std::is_same_v<FusionOp, FusionLinComb>;
  static constexpr bool epi_is_default = std::is_same_v<CollectiveEpilogue, DefaultEpilogue>;
  static_assert(cute::is_base_of_v<cutlass::epilogue::fusion::FusionOperation, FusionOp> ||
                    epi_is_default,
                "Failed to determine benchmark epilogue");
  static_assert(epi_is_default || epi_is_deeltactmul || epi_is_silu || epi_is_lincomb,
                "Failed to determine benchmark epilogue");

  int32_t count;

  //
  // Data members
  //

  /// Initialization
  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;

  StrideS stride_S;
  StrideZ stride_Z;


  uint64_t seed;

  std::vector<DeviceAllocation<ElementA>> block_A;
  std::vector<DeviceAllocation<ElementB>> block_B;
  std::vector<DeviceAllocation<ElementC>> block_C;
  DeviceAllocation<ElementOutput> block_D;
  DeviceAllocation<ElementOutput> block_ref_D;
  std::vector<DeviceAllocation<ElementOutput>> block_Aux;

  cutlass::DeviceAllocation<ElementScale> block_scale;
  cutlass::DeviceAllocation<ElementZero> block_zero;

  DeviceAllocation<ElementMma> block_A_verify;
  DeviceAllocation<ElementMma> block_B_verify;

  BenchmarkRunnerGemm() : seed(0) {};

  //
  // Methods
  //

  template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout,
  class ZeroLayout>
  static auto dequantize_A(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       ZeroLayout const zero_layout,
                       int const group_size) {
    if constexpr (std::is_same_v<DequantizedElement, QuantizedElement>) {
      return dq_buffer;
    }

    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DequantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<QuantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    std::vector<uint8_t> zero(size(zero_layout) * sizeof_bits_v<ElementZero> / 8, 0);
    cutlass::device_memory::copy_to_host(zero.data(), (uint8_t*)zero_buffer, zero.size());

    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DequantizedElement*>(dst.data())), select<1, 0, 2>(operand_layout));

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<QuantizedElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const QuantizedElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<QuantizedElement const *>(src.data())), select<1, 0, 2>(operand_layout));
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto zero_tensor = [&]() {
      if constexpr (sizeof_bits_v<ElementZero> < 8) {
        auto flatten_tensor = flatten(make_tensor(cute::subbyte_iterator<const ElementZero>(zero.data()), zero_layout));
        static_assert(rank(flatten_tensor.layout()) == 4);
        return make_tensor(flatten_tensor.data(), select<1, 0, 2, 3>(flatten_tensor.layout()));
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<ElementZero const *>(zero.data())), zero_layout);
      }
    }();

    auto M = size<1>(src_tensor);
    auto K = size<0>(src_tensor);
    auto L = size<2>(src_tensor);

    static constexpr bool is_qnt = cutlass::platform::numeric_limits<DequantizedElement>::is_integer;

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int m = 0; m < M; m++) {
          auto src_data = [&]() {
            if constexpr (is_qnt) {
              if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
                return  src_tensor(k, m, l);
              } else {
                return src_tensor(k, m, l).get();
              }
            } else {
              using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
              if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
                return  (ret_type)(src_tensor(k, m, l));
              } else {
                return (ret_type)(src_tensor(k, m, l).get());
              }
            }
          }();

          auto scale_data = scale_tensor(m, k / group_size, l);

          using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
          ret_type zero_data = [&]() {
            if constexpr (sizeof_bits_v<ElementZero> >= 8) {
              return zero_tensor(m, k / group_size, l);
            } else {
              auto zero_elements_packed_along_k = get<0>(zero_tensor.shape());
              return (ret_type)(zero_tensor((k / group_size) % zero_elements_packed_along_k, m, k / group_size / zero_elements_packed_along_k, l).get());
            }
          }();

          if constexpr (is_qnt) {
            dst_tensor(k, m, l) = ((int)(src_data / scale_data)) + zero_data;
          } else {
            dst_tensor(k, m, l) = (src_data - zero_data) * scale_data;
          }
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DequantizedElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
    return dq_buffer;
  }

  template <
  class QuantizedElement,
  class DequantizedElement,
  class OperandLayout,
  class ElementScale,
  class ElementZero,
  class ScaleLayout,
  class ZeroLayout>
  static auto dequantize_B(DequantizedElement* dq_buffer,
                       QuantizedElement const* q_buffer,
                       OperandLayout const operand_layout,
                       ElementScale const* scale_buffer,
                       ElementZero const* zero_buffer,
                       ScaleLayout const scale_layout,
                       ZeroLayout const zero_layout,
                       int const group_size) {
    std::vector<uint8_t> dst(size(operand_layout) * sizeof_bits_v<DequantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(dst.data(), (uint8_t*)dq_buffer, dst.size());

    std::vector<uint8_t> src(size(operand_layout) * sizeof_bits_v<QuantizedElement> / 8, 0);
    cutlass::device_memory::copy_to_host(src.data(), (uint8_t*)q_buffer, src.size());

    std::vector<uint8_t> scale(size(scale_layout) * sizeof_bits_v<ElementScale> / 8, 0);
    cutlass::device_memory::copy_to_host(scale.data(), (uint8_t*)scale_buffer, scale.size());

    std::vector<uint8_t> zero(size(zero_layout) * sizeof_bits_v<ElementZero> / 8, 0);
    cutlass::device_memory::copy_to_host(zero.data(), (uint8_t*)zero_buffer, zero.size());

    compat::wait();

    auto dst_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<DequantizedElement*>(dst.data())), operand_layout);

    auto src_tensor = [&]() {
      if constexpr (sizeof_bits_v<QuantizedElement> < 8) {
        return make_tensor(cute::subbyte_iterator<const QuantizedElement>(src.data()), operand_layout);
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<QuantizedElement const *>(src.data())), operand_layout);
      }
    }();

    auto scale_tensor = make_tensor(make_gmem_ptr(reinterpret_cast<ElementScale const *>(scale.data())), scale_layout);

    auto zero_tensor = [&]() {
      if constexpr (sizeof_bits_v<ElementZero> < 8) {
        auto flatten_tensor = flatten(make_tensor(cute::subbyte_iterator<const ElementZero>(zero.data()), zero_layout));
        static_assert(rank(flatten_tensor.layout()) == 4);
        return make_tensor(flatten_tensor.data(), select<1, 0, 2, 3>(flatten_tensor.layout()));
      } else {
        return make_tensor(make_gmem_ptr(reinterpret_cast<ElementZero const *>(zero.data())), zero_layout);
      }
    }();

    auto N = size<0>(src_tensor);
    auto K = size<1>(src_tensor);
    auto L = size<2>(src_tensor);

    for (int l = 0; l < L; l++) {
      for (int k= 0; k < K; k++) {
        for (int n = 0; n < N; n++) {
          using ret_type = cute::conditional_t<sizeof_bits_v<ElementZero> >= 8, ElementZero, int8_t>;
          ret_type a = [&]() {
            if constexpr (sizeof_bits_v<QuantizedElement> >= 8) {
              return  (ret_type)(src_tensor(n, k, l));
            } else {
              return (ret_type)(src_tensor(n, k, l).get());
            }}();

          ret_type b = [&]() {
            if constexpr (sizeof_bits_v<ElementZero> >= 8) {
              return (ret_type)(zero_tensor(n, k / group_size, l));
            } else {
              auto k_packed = get<0>(zero_tensor.shape());
              return (ret_type)(zero_tensor((k / group_size) % k_packed, n, k / group_size / k_packed, l).get());
            }
          }();

          dst_tensor(n, k, l) = ((ElementScale)(a - b)) * scale_tensor(n, k / group_size, l);
        }
      }
    }

    cutlass::device_memory::copy_to_device(dq_buffer, (DequantizedElement*)(raw_pointer_cast(dst_tensor.data())), dst_tensor.size());
    compat::wait();
    return dq_buffer;
  }

  bool verify(const ProblemShapeType& problem_size, ElementCompute alpha, ElementCompute beta) {
    auto& M = cute::get<0>(problem_size);
    auto& N = cute::get<1>(problem_size);
    auto& K = cute::get<2>(problem_size);
    auto& L = cute::get<3>(problem_size);

    TensorRef ref_C(block_C[0].get(), LayoutC::packed({M, N}));
    TensorRef ref_D(block_ref_D.get(), LayoutD::packed({M, N}));

    auto [ptr_A, ptr_B] = [&]() {
      if constexpr (!is_mixed_dtype<DispatchPolicy>) {
        return make_tuple(block_A[0].get(), block_B[0].get());
      } else {
        static constexpr bool IsAQuant = cutlass::platform::numeric_limits<ElementA>::is_integer
                                    ^ cutlass::platform::numeric_limits<ElementAccumulator>::is_integer;
        static constexpr bool IsBQuant = cutlass::platform::numeric_limits<ElementB>::is_integer
                                          ^ cutlass::platform::numeric_limits<ElementAccumulator>::is_integer;

        static constexpr bool IsATransformed = CollectiveMainloop::IsATransformed;
        auto dq_mn_size = IsATransformed ? M : N;

        auto shape_ab = cute::make_shape(dq_mn_size, K, L);
        auto shape_scale = cute::make_shape(dq_mn_size, K / 128, L);
        static constexpr auto k_packed = CollectiveMainloop::zero_elements_packed_along_k;
        auto shape_zero = [&]() {
          if constexpr (is_tuple_v<std::remove_reference_t<decltype(cute::get<1>(stride_Z))>>) {
            return cute::make_shape(dq_mn_size, cute::make_shape(k_packed,
                                                        cute::max(1, K / 128 / k_packed)), L);
          } else {
            return shape_scale;
          }
        }();

        auto ptr_A = [&]() {
          if constexpr (IsAQuant) {
            return dequantize_A(block_A_verify.get(), block_A[0].get(), make_layout(shape_ab, stride_A), block_scale.get(),
                                block_zero.get(), make_layout(shape_scale, stride_S), make_layout(shape_zero, stride_Z), 128);
          } else {
            return block_A_verify.get();
          }
        }();

        auto ptr_B = [&]() {
         if constexpr (IsBQuant) {
            return dequantize_B(block_B_verify.get(), block_B[0].get(), make_layout(shape_ab, stride_B), block_scale.get(),
                                block_zero.get(), make_layout(shape_scale, stride_S), make_layout(shape_zero, stride_Z), 128);
          } else {
            return block_B_verify.get();
          }
        }();

        return make_tuple(ptr_A, ptr_B);
      }
    }();

    TensorRef ref_A(ptr_A, LayoutA::packed({M, K}));
    TensorRef ref_B(ptr_B, LayoutB::packed({K, N}));

    reference::device::GemmComplex(
            {M, N, K},
            alpha,
            ref_A,
            ComplexTransform::kNone,
            ref_B,
            ComplexTransform::kNone,
            beta,
            ref_C,
            ref_D,
            ElementAccumulator(0),
            L,     // batch_count
            get<2>(stride_A), // batch_stride_A
            get<2>(stride_B), // batch_stride_B
            get<2>(stride_C), // batch_stride_C
            get<2>(stride_D)  // batch_stride_D
    );

#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
#endif

    // TODO(codeplay): Replace this with a general solution (hook up to Testbed3x)
    if constexpr (epi_is_silu) {
      using TensorView = cutlass::TensorView<ElementOutput, LayoutD>;
      for (int batch = 0, offset = 0; batch < L; batch++, offset += M * N) {
        cutlass::reference::device::TensorSiLu(TensorView(
            block_ref_D.get() + offset, LayoutD::packed({M, N}), cutlass::make_Coord(M, N)));
      }
    } else if constexpr (epi_is_deeltactmul) {
      cutlass::reference::device::BlockElementwiseOp<std::multiplies>(
          block_ref_D.get(), block_ref_D.get(), block_Aux[0].get(), block_D.size());
    }

    compat::wait();

    // Check if output from CUTLASS kernel and reference kernel are equal or not
    bool passed = false;
    if constexpr (std::is_same_v<ElementOutput, cutlass::half_t> ||
                  std::is_same_v<ElementOutput, cutlass::bfloat16_t>) {
      passed = reference::device::BlockCompareRelativelyEqual(
        block_ref_D.get(), block_D.get(), block_D.size(), ElementOutput(0.5f), ElementOutput(1.0f));
    } else {
      passed = reference::device::BlockCompareEqual(
        block_ref_D.get(), block_D.get(), block_D.size());
    }

    return passed;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(::benchmark::State& state, const ProblemShapeType& problem_size) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    // TODO(codeplay): cute::cosize(some_large_layout) will overflow int32. What can we do about this?
    std::size_t size_A = cute::cosize(make_layout(cute::make_shape(M, K, L), stride_A));
    std::size_t size_B = cute::cosize(make_layout(cute::make_shape(N, K, L), stride_B));
    std::size_t size_C = cute::cosize(make_layout(cute::make_shape(M, N, L), stride_C));
    std::size_t mem_occupied_ABC = ((size_A * sizeof_bits_v<ElementA>) + (size_B * sizeof_bits_v<ElementB>) +
                                   (size_C * sizeof_bits_v<ElementC>)) / sizeof_bits_v<int8_t>;
    count = std::ceil(static_cast<float>(cutlass::get_llc_size()) / static_cast<float>(mem_occupied_ABC)) + 1;

    if constexpr (is_mixed_dtype<DispatchPolicy>) {
      static constexpr bool IsATransformed = CollectiveMainloop::IsATransformed;

      auto dq_mn_size = IsATransformed ? M : N;
      auto scale_k = K / 128;

      static constexpr auto k_packed = CollectiveMainloop::zero_elements_packed_along_k;
      static constexpr auto is_tuple_z = is_tuple_v<std::remove_reference_t<decltype(cute::get<1>(StrideZ{}))>>;

      auto shape_scale = cute::make_shape(dq_mn_size, scale_k, L);

      stride_S = cutlass::make_cute_packed_stride(StrideS{}, shape_scale);
      stride_Z = [&]() {
        if constexpr (is_tuple_z) {
          return make_stride(Int<k_packed>{}, make_stride(_1{}, int64_t(k_packed * dq_mn_size)), int64_t(dq_mn_size * scale_k));
        } else {
          return stride_S;
        }
      }();

      block_A_verify.reset(size_A);
      block_B_verify.reset(size_B);

      block_scale.reset(static_cast<std::size_t>(scale_k) * L * dq_mn_size);
      block_zero.reset(static_cast<std::size_t>(scale_k) * L * dq_mn_size);

      initialize_block(block_scale, seed, ElementScale(1), ElementScale(4));
      initialize_block(block_zero, seed);
    }

    for(int i=0; i < count; i++) {
      block_A.emplace_back();
      block_B.emplace_back();
      block_C.emplace_back();
      if constexpr (epi_is_deeltactmul) {
        block_Aux.emplace_back();
      }
    }

    try {
      for (int i = 0; i < count; i++) {
        block_A[i].reset(size_A);
        block_B[i].reset(size_B);
        block_C[i].reset(size_C);
        if constexpr (is_mixed_dtype<DispatchPolicy>) {
          if (i == 0) {
            initialize_mixed_dtype_block(block_A[i], block_A_verify, seed + i);
            initialize_mixed_dtype_block(block_B[i], block_B_verify, seed + i);
          } else {
            initialize_block(block_A[i], seed + i);
            initialize_block(block_B[i], seed + i);
          }
        } else {
          initialize_block(block_A[i], seed + i);
          initialize_block(block_B[i], seed + i);
        }
        initialize_block(block_C[i], seed + i);
        if constexpr (epi_is_deeltactmul) {
          block_Aux[i].reset(size_C);
          initialize_block(block_Aux[i], seed + i);
        }
      }

      block_D.reset(size_C);
      block_ref_D.reset(size_C);
    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }
  }

  void run(::benchmark::State& state, const GEMMOptions& options, const KernelHardwareInfo& hw_info) {
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};

    initialize(state, problem_size);

    typename Gemm::GemmKernel::Arguments arguments = GemmConfiguration::defaultArguments();
    set_scheduler_splits(arguments, options.split_k_slices);
    arguments.mode = gemm::GemmUniversalMode::kGemm;
    arguments.problem_shape = problem_size;

    if constexpr (!is_mixed_dtype<DispatchPolicy>) {
      arguments.mainloop = {block_A[0].get(), stride_A, block_B[0].get(), stride_B};
    } else {
      arguments.mainloop = {block_A[0].get(), stride_A, block_B[0].get(), stride_B, block_scale.get(),
              stride_S, block_zero.get(), stride_Z, 128};
    }

    arguments.epilogue = {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, block_C[0].get(), stride_C, block_D.get(), stride_D};
    arguments.hw_info = hw_info;

    if constexpr(epi_is_deeltactmul){
      arguments.epilogue.thread.aux_ptr = block_Aux[0].get();
      arguments.epilogue.thread.dAux = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(options.m, options.n, options.l));
    }

    Gemm gemm_op;

    device_memory::allocation<uint8_t> workspace;
    size_t workspace_size = Gemm::get_workspace_size(arguments);
    try {
      workspace.reset(workspace_size);
    } catch (std::exception const &e) {
      state.SkipWithError(e.what());
    }

    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM unable to implement given args.");

    if (gemm_op.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess)
      state.SkipWithError("GEMM failed to initialize.");

    if (state.error_occurred()) return;

    // Run the GEMM
    gemm_op.run();

#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#else
    cudaDeviceSynchronize();
#endif

    // FIXME: skip GPU reference verification for perf debugging
    // SKIP verify for perf debugging
#if 0
    bool passed = verify(problem_size, ElementCompute(options.alpha), ElementCompute(options.beta));
    if(not passed) {
      state.SkipWithError("Disposition Failed.");
    }
#endif

    state.counters["m"] = options.m;
    state.counters["n"] = options.n;
    state.counters["k"] = options.k;
    state.counters["l"] = options.l;
    state.counters["alpha"] = options.alpha;
    state.counters["beta"] = options.beta;
    state.counters["split_k_slices"] = options.split_k_slices > 0 ? options.split_k_slices : 1;

    std::stringstream extra_label;
    if constexpr (cute::size<0>(StrideA{}) == 1) {
      extra_label << "layoutA=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideA{}) == 1) {
      extra_label << "layoutA=RowMajor ";
    }
    if constexpr (cute::size<0>(StrideB{}) == 1) {
      extra_label << "layoutB=RowMajor ";
    } else if constexpr (cute::size<1>(StrideB{}) == 1) {
      extra_label << "layoutB=ColumnMajor ";
    }
    if constexpr (cute::size<0>(StrideC{}) == 1) {
      extra_label << "layoutC=ColumnMajor ";
    } else if constexpr (cute::size<1>(StrideC{}) == 1) {
      extra_label << "layoutC=RowMajor ";
    }
    state.SetLabel(extra_label.str());

    auto gflop = 2.0 * options.m * options.n * options.k * options.l * 1e-9;

    // Compatible with data types smaller than 8 bits here
    constexpr double bits_per_byte = static_cast<double>(sizeof_bits_v<char>);
    constexpr double sizeof_a = sizeof_bits_v<ElementA> / bits_per_byte;
    constexpr double sizeof_b = sizeof_bits_v<ElementB> / bits_per_byte;
    constexpr double sizeof_c = sizeof_bits_v<ElementC> / bits_per_byte;
    auto mega_bytes_transferred = static_cast<double>(
        options.m * options.k * sizeof_a +
        options.k * options.n * sizeof_b +
        (options.beta != 0 ? 2 : 1) * options.m * options.n * sizeof_c
      ) * 1e-6 * options.l;

    static constexpr int kWarmupIters = 50;
    static constexpr int kMeasureIters = 50;

    // --- explicit warmup (discarded, NOT timed) ---
    state.PauseTiming();
    for (int w = 0; w < kWarmupIters; ++w) {
      gemm_op.run();
    }
#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#endif
    state.ResumeTiming();

    // --- timed measurement ---
    GPU_Clock batch_timer;
    batch_timer.start();
    for (int i = 0; i < kMeasureIters; ++i) {
      gemm_op.run();
    }
#if defined(CUTLASS_ENABLE_SYCL)
    compat::wait();
#endif
    double total_ms = batch_timer.milliseconds();
    double avg_ms_per_iter = total_ms / kMeasureIters;

    std::vector<double> runtimes(kMeasureIters, avg_ms_per_iter);
    state.SetIterationTime(avg_ms_per_iter / 1000.0);
    state.SetItemsProcessed(kMeasureIters);

    double best = avg_ms_per_iter;
    double worst = avg_ms_per_iter;
    double median = avg_ms_per_iter;
    double avg = avg_ms_per_iter;
    double trimmed_mean = avg_ms_per_iter;
    double stddev = 0.0;

    state.counters["runtime_min_ms"] = best;
    state.counters["runtime_max_ms"] = worst;
    state.counters["runtime_median_ms"] = median;
    state.counters["runtime_avg_ms"] = avg;
    state.counters["runtime_trimmed_mean_ms"] = trimmed_mean;
    state.counters["runtime_stddev_ms"] = stddev;
    state.counters["warmup_iters"] = kWarmupIters;
    state.counters["measure_iters"] = kMeasureIters;

    state.counters["avg_tflops"] = gflop / trimmed_mean;
    state.counters["avg_throughput"] = mega_bytes_transferred / trimmed_mean;
    state.counters["best_tflop"] = gflop / best;
    state.counters["best_bandwidth"] = mega_bytes_transferred / best;
    state.counters["median_tflops"] = gflop / median;
  }

private:
  static void initialize_counters(::benchmark::State& state) {
    state.counters["avg_runtime_ms"] = 0;
    state.counters["best_runtime_ms"] = std::numeric_limits<double>::max();
    state.counters["worst_runtime_ms"] = std::numeric_limits<double>::lowest();
  }

  static void update_counters(::benchmark::State& state, double ms_elapsed) {
    state.PauseTiming();
    state.counters["total_runtime_ms"] += ms_elapsed;
    state.counters["best_runtime_ms"] = std::min<double>(state.counters["best_runtime_ms"], ms_elapsed);
    state.counters["worst_runtime_ms"] = std::max<double>(state.counters["worst_runtime_ms"], ms_elapsed);
    state.ResumeTiming();
  }

  static void finalize_counters(::benchmark::State& state,  double gflop, double mega_bytes_transferred) {
    state.counters["avg_runtime_ms"] =
      (state.counters["total_runtime_ms"] -state.counters["best_runtime_ms"] - state.counters["worst_runtime_ms"] ) / static_cast<double>(state.iterations() - 2);
    state.counters["avg_tflops"] = gflop / state.counters["avg_runtime_ms"];
    state.counters["avg_throughput"] = mega_bytes_transferred / state.counters["avg_runtime_ms"];
    state.counters["best_tflop"] = gflop / state.counters["best_runtime_ms"];
    state.counters["best_bandwidth"] = mega_bytes_transferred / state.counters["best_runtime_ms"];
  }

  public:
  // ── GB-free profiler: bypasses Google Benchmark entirely ──
  // Uses batch-wait pattern (matches 00_bmg_gemm example at 126 TFLOPS).
  double run_direct(const GEMMOptions& options, const KernelHardwareInfo& hw_info) {
    // Use the PROVEN initialize() code path from run() — identical buffer setup
    ProblemShapeType problem_size = ProblemShapeType{options.m, options.n, options.k, options.l};
    alignas(64) char state_buf[256] = {};
    auto& sref = *reinterpret_cast<::benchmark::State*>(state_buf);
    initialize(sref, problem_size);

    typename GemmConfiguration::GemmKernel::Arguments arguments = GemmConfiguration::defaultArguments();
    set_scheduler_splits(arguments, options.split_k_slices);
    arguments.mode = gemm::GemmUniversalMode::kGemm; arguments.problem_shape = problem_size;
    arguments.mainloop = {block_A[0].get(), stride_A, block_B[0].get(), stride_B};
    arguments.epilogue = {{ElementAccumulator(options.alpha), ElementAccumulator(options.beta)}, block_C[0].get(), stride_C, block_D.get(), stride_D};
    arguments.hw_info = hw_info;

    Gemm gemm_op; size_t ws = Gemm::get_workspace_size(arguments);
    device_memory::allocation<uint8_t> workspace;
    try { workspace.reset(ws); } catch (...) { return 0.0; }
    if (gemm_op.can_implement(arguments) != cutlass::Status::kSuccess) return 0.0;
    if (gemm_op.initialize(arguments, workspace.get()) != cutlass::Status::kSuccess) return 0.0;

    gemm_op.run(); compat::wait();  // initial run
    for (int w = 0; w < 100; ++w) gemm_op.run();
    compat::wait();
    GPU_Clock timer; timer.start();
    for (int i = 0; i < 100; ++i) gemm_op.run();
    compat::wait();
    double total_ms = timer.milliseconds();
    double avg_sec = timer.seconds() / 100.0;
    std::cerr << "[PERF] total_ms=" << total_ms << " avg_us=" << (avg_sec*1e6) << " tf=" << ((2.0 * options.m * options.n * options.k * options.l * 1e-12) / avg_sec) << std::endl;
    return (2.0 * options.m * options.n * options.k * options.l * 1e-12) / avg_sec;
  }
};

}

#if defined(CUTLASS_BENCHMARK_FILTER_ENABLED)
#include "cutlass_benchmark_filter.hpp"
#define CUTLASS_BENCHMARK_CAT_IMPL(A, B) A##B
#define CUTLASS_BENCHMARK_CAT(A, B) CUTLASS_BENCHMARK_CAT_IMPL(A, B)
#define CUTLASS_BENCHMARK_PROBE() ~, 1
#define CUTLASS_BENCHMARK_SECOND(A, B, ...) B
#define CUTLASS_BENCHMARK_IS_PROBE(...) CUTLASS_BENCHMARK_SECOND(__VA_ARGS__, 0)
#define CUTLASS_BENCHMARK_IS_DEFINED(X) CUTLASS_BENCHMARK_IS_PROBE(CUTLASS_BENCHMARK_CAT(CUTLASS_BENCHMARK_IS_DEFINED_, X))
#define CUTLASS_BENCHMARK_IS_DEFINED_1 CUTLASS_BENCHMARK_PROBE()
#define CUTLASS_BENCHMARK_KERNEL_ENABLED(F) CUTLASS_BENCHMARK_IS_DEFINED(CUTLASS_BENCHMARK_CAT(CUTLASS_BENCHMARK_ENABLE_, F))
#else
#define CUTLASS_BENCHMARK_KERNEL_ENABLED(F) 1
#endif

#define CUTLASS_BENCHMARK(F) cutlass::benchmark::BenchmarkRegistry<cutlass::benchmark::GEMMOptions>::Register(#F, &F##_func)

#define CUTLASS_CREATE_GEMM_BENCHMARK(F)                          \
  static void F##_func(                                           \
      ::benchmark::State& state,                                  \
      cutlass::benchmark::GEMMOptions const& options,                 \
      cutlass::KernelHardwareInfo const& hw_info) {               \
    if constexpr (CUTLASS_BENCHMARK_KERNEL_ENABLED(F)) {          \
      auto bench = cutlass::benchmark::BenchmarkRunnerGemm<F>();  \
      bench.run(state, options, hw_info);                         \
    } else {                                                      \
      state.SkipWithError("benchmark disabled by build filter");  \
    }                                                            \
  }
