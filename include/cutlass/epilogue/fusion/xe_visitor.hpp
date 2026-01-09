/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

/*! \file
  \brief Visitor tree operations for the Xe epilogue
*/

#pragma once

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"


namespace cutlass::epilogue::fusion {

template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_0,_1,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct XeRowBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrRowType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Row broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<1>(StrideMNL{}))>, bool>; // row vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast, "XeRowBroadcast requires static MN stride for non-dynamic broadcast case."); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_0,_1>{} || IsDynamicBroadcast, "XeRowBroadcast requires MN stride=(0,1) for non-dynamic broadcast case.");

  struct SharedStorage { };

  struct Arguments {
    PtrRowType ptr_row = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dRow = {};
  };

  struct Params {
    PtrRowType ptr_row = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dRow = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_row, ElementCompute(args.null_default), args.dRow};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeRowBroadcast() { }

  CUTLASS_HOST_DEVICE
  XeRowBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false) {
    auto const& [stride_M, stride_N, stride_L] = params.dRow;
    // Nullptr default
    if (EnableNullptr && params.ptr_row == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_N == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_row[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class MRow, class CTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(MRow mRow_, CTensor tCcRow_, Params const& params_)
      : mRow(mRow_),
        tCcRow(tCcRow_),
        params(params_),
        tCrRow(make_fragment_like<ElementCompute>(tCcRow)) {  // Allocate register storage matching coordinate layout
      if (EnableNullptr && params.ptr_row == nullptr) {
        fill(tCrRow, params.null_default);
      }
    }

    MRow mRow;                                                                         // Global bias tensor (M, N) - pointer already offset to correct batch
    CTensor tCcRow;                                                                    // Global output coordinates per thread
    decltype(make_fragment_like<ElementCompute>(tCcRow)) tCrRow;                     // Register cache: bias values pre-loaded in begin(), indexed same as tCcRow
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if (EnableNullptr && params.ptr_row == nullptr) {
        return;
      }
      
      // Load all bias values from global memory into registers once per epilogue tile.
      // Subsequent visit() calls read from tCrRow
      // Uses scalar loads indexed by global N coordinates from tCcRow
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCcRow); ++i) {
        auto coord = tCcRow(i);
        int n_coord = get<1>(coord);  // Extract global column index
        if (n_coord < get<1>(shape(mRow))) {
          tCrRow(i) = ElementCompute(mRow(_0{}, n_coord));  // mRow: stride_M=0 (broadcast), stride_N=1
        } else {
          tCrRow(i) = ElementCompute(0);  // Zero-pad out-of-bounds (for non-tile-aligned N)
        }
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_row;

      if (EnableNullptr && params.ptr_row == nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          frg_row[i] = params.null_default;
        }
        return frg_row;
      }

      // Read from pre-loaded register cache
      // Slice tCrRow by epilogue iteration (epi_m, epi_n)
      Tensor tCrRow_mn = tCrRow(_,_,_,epi_m,epi_n);
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        int idx = epi_v * FragmentSize + i;
        frg_row[i] = tCrRow_mn(idx);
      }

      return frg_row;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto layout_N = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_N = get<1>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_N = repeat_like(shape_N, int(0));
        if (get<1>(params.dRow) == bool(1)) {
          stride_N = transform_leaf(compact_major<LayoutLeft>(shape_N),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_N, stride_N);
      }
      else {
        return make_layout(shape_N);
      }
    }();

    auto layout_M = make_layout(M, repeat_like(M, _0{}));
    ElementInput const* ptr_row;
    if constexpr(IsArrayOfPointers) {
      ptr_row = params.ptr_row[l];  // Array-of-pointers: each batch has separate allocation
    } else {
      // When single contiguous allocation with shape (L, 1, N) or (L, N),
      // compute byte offset: batch_stride elements per batch
      auto batch_stride = size_t(get<2>(params.dRow));
      ptr_row = params.ptr_row + l * batch_stride;  // Advance pointer to batch l's data
    }
    // Create 2D tensor (M, N) with pre-offset pointer
    // Pointer already points to correct batch, so layout should NOT include batch dimension.
    Tensor mRow = make_tensor(make_gmem_ptr(ptr_row), make_layout(layout_M, layout_N));

    // Use global output coordinates from epilogue - these are already partitioned per thread
    // and tiled by (epi_m, epi_n). Row broadcast uses these to index bias vector.
    auto tCcRow = args.tCcD;

    return ConsumerStoreCallbacks(mRow, tCcRow, params);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template<
  int Stages,
  class CtaTileShapeMNK,
  class ElementInput_,
  class ElementCompute = cute::remove_pointer_t<ElementInput_>,
  class StrideMNL_ = Stride<_1,_0,_0>,
  int Alignment = 128 / sizeof_bits_v<cute::remove_pointer_t<ElementInput_>>,
  bool EnableNullptr = true // Fallback scalar broadcast for nullptr params
>
struct XeColBroadcast {
  using StrideMNL = StrideMNL_;
  // Get base element input type.
  using ElementInput = cute::remove_pointer_t<ElementInput_>;
  // Check if input is an array of pointers.
  static constexpr bool IsArrayOfPointers = is_same_v<ElementInput*, ElementInput_>;
  using PtrColType = cute::conditional_t<IsArrayOfPointers, ElementInput const* const*, ElementInput const*>;

  static_assert(Stages == 0, "Column broadcast doesn't support smem pipelining");

  static constexpr bool IsDynamicBroadcast = is_same_v<remove_cvref_t<decltype(get<0>(StrideMNL{}))>, bool>; // column vector or scalar broadcast
  static_assert(is_static_v<decltype(take<0,2>(StrideMNL{}))> || IsDynamicBroadcast, "XeColBroadcast requires static MN stride for non-dynamic broadcast case."); // batch stride can be dynamic or static
  static_assert(take<0,2>(StrideMNL{}) == Stride<_1,_0>{} || IsDynamicBroadcast, "XeColBroadcast requires MN stride=(1,0) for non-dynamic broadcast case.");

  struct SharedStorage { };

  struct Arguments {
    PtrColType ptr_col = nullptr;
    ElementInput null_default = ElementInput(0);
    StrideMNL dCol = {};
  };

  struct Params {
    PtrColType ptr_col = nullptr;
    ElementCompute null_default = ElementCompute(0);
    StrideMNL dCol = {};
  };

  template <class ProblemShape>
  static constexpr Params
  to_underlying_arguments(ProblemShape const& problem_shape, Arguments const& args, void* workspace) {
    return {args.ptr_col, ElementCompute(args.null_default), args.dCol};
  }

  template <class ProblemShape>
  static bool
  can_implement(ProblemShape const& problem_shape, Arguments const& args) {
    return true;
  }

  template <class ProblemShape>
  static size_t
  get_workspace_size(ProblemShape const& problem_shape, Arguments const& args) {
    return 0;
  }

  template <class ProblemShape>
  static cutlass::Status
  initialize_workspace(ProblemShape const& problem_shape, Arguments const& args, void* workspace, cudaStream_t stream,
    CudaHostAdapter* cuda_adapter = nullptr) {
    return cutlass::Status::kSuccess;
  }

  CUTLASS_HOST_DEVICE
  XeColBroadcast() { }

  CUTLASS_HOST_DEVICE
  XeColBroadcast(Params const& params, SharedStorage const& shared_storage)
      : params(params), is_zero_(false) {
    auto const& [stride_M, stride_N, stride_L] = params.dCol;
    // Nullptr default
    if (EnableNullptr && params.ptr_col == nullptr) {
      is_zero_ = params.null_default == ElementCompute(0);
    }
    // Dynamic non-batched scalar broadcast
    else if (IsDynamicBroadcast && stride_M == bool(0) && stride_L == repeat_like(stride_L, 0)) {
       if constexpr (!IsArrayOfPointers) {
         is_zero_ = params.ptr_col[0] == ElementInput(0);
       }
    }
  }

  Params params;
  bool is_zero_ = false;
  ElementInput *smem = nullptr;

  CUTLASS_DEVICE bool
  is_producer_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_C_load_needed() const {
    return false;
  }

  CUTLASS_DEVICE bool
  is_zero() const {
    return is_zero_;
  }

  template <class... Args>
  CUTLASS_DEVICE auto
  get_producer_load_callbacks(ProducerLoadArgs<Args...> const& args) {
    return EmptyProducerLoadCallbacks{};
  }

  template<class MCol, class CTensor>
  struct ConsumerStoreCallbacks : EmptyConsumerStoreCallbacks {
    CUTLASS_DEVICE
    ConsumerStoreCallbacks(MCol mCol_, CTensor tCcCol_, Params const& params_)
      : mCol(mCol_),
        tCcCol(tCcCol_),
        params(params_),
        tCrCol(make_fragment_like<ElementCompute>(tCcCol)) {  // Allocate register storage matching coordinate layout
      if (EnableNullptr && params.ptr_col == nullptr) {
        fill(tCrCol, params.null_default);
      }
    }

    MCol mCol;                                                                         // Global bias tensor (M, N) - pointer already offset to correct batch
    CTensor tCcCol;                                                                    // Global output coordinates per thread
    decltype(make_fragment_like<ElementCompute>(tCcCol)) tCrCol;                     // Register cache: bias values pre-loaded in begin(), indexed same as tCcCol
    Params const& params;

    CUTLASS_DEVICE void
    begin() {
      if (EnableNullptr && params.ptr_col == nullptr) {
        return;
      }
      
      // Load all bias values from global memory into registers once per epilogue tile.
      // Subsequent visit() calls read from tCrCol
      // Uses scalar loads indexed by global M coordinates from tCcCol
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < size(tCcCol); ++i) {
        auto coord = tCcCol(i);
        int m_coord = get<0>(coord);  // Extract global row index
        if (m_coord < get<0>(shape(mCol))) {
          tCrCol(i) = ElementCompute(mCol(m_coord, _0{}));  // mCol: stride_M=1, stride_N=0 (broadcast)
        } else {
          tCrCol(i) = ElementCompute(0);  // Zero-pad out-of-bounds (for non-tile-aligned M)
        }
      }
    }

    template <typename ElementAccumulator, int FragmentSize>
    CUTLASS_DEVICE Array<ElementCompute, FragmentSize>
    visit(Array<ElementAccumulator, FragmentSize> const& frg_acc, int epi_v, int epi_m, int epi_n) {
      Array<ElementCompute, FragmentSize> frg_col;

      if (EnableNullptr && params.ptr_col == nullptr) {
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < FragmentSize; ++i) {
          frg_col[i] = params.null_default;
        }
        return frg_col;
      }

      // Read from pre-loaded register cache
      // Slice tCrCol by epilogue iteration (epi_m, epi_n)
      Tensor tCrCol_mn = tCrCol(_,_,_,epi_m,epi_n);
      
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < FragmentSize; ++i) {
        int idx = epi_v * FragmentSize + i;
        frg_col[i] = tCrCol_mn(idx);
      }

      return frg_col;
    }
  };

  template <
    bool ReferenceSrc, // do register tensors reference the src or dst layout of the tiled copy
    class... Args
  >
  CUTLASS_DEVICE auto
  get_consumer_store_callbacks(ConsumerStoreArgs<Args...> const& args) {
    auto [M, N, K, L] = args.problem_shape_mnkl;
    auto [m, n, k, l] = args.tile_coord_mnkl;

    auto layout_M = [&] () CUTLASS_LAMBDA_FUNC_INLINE {
      auto shape_M = get<0>(args.problem_shape_mnkl);
      if constexpr (IsDynamicBroadcast) {
        auto stride_M = repeat_like(shape_M, int(0));
        if (get<0>(params.dCol) == bool(1)) {
          stride_M = transform_leaf(compact_major<LayoutLeft>(shape_M),
            [] (auto const& stride) { return static_cast<int>(stride); }
          );
        }
        return make_layout(shape_M, stride_M);
      }
      else {
        return make_layout(shape_M);
      }
    }();

    auto layout_N = make_layout(N, repeat_like(N, _0{}));
    ElementInput const* ptr_col;
    if constexpr(IsArrayOfPointers) {
      ptr_col = params.ptr_col[l];  // Array-of-pointers: each batch has separate allocation
    } else {
      // BATCHED BIAS: Single contiguous allocation with shape (L, M, 1) or (L, M)
      // Compute byte offset: batch_stride elements per batch
      auto batch_stride = size_t(get<2>(params.dCol));
      ptr_col = params.ptr_col + l * batch_stride;  // Advance pointer to batch l's data
    }
    // Create 2D tensor (M, N) with pre-offset pointer - avoids double-offset bug.
    // Key: Pointer already points to correct batch, so layout should NOT include batch dimension.
    Tensor mCol = make_tensor(make_gmem_ptr(ptr_col), make_layout(layout_M, layout_N));

    // Use global output coordinates from epilogue - these are already partitioned per thread
    // and tiled by (epi_m, epi_n). Column broadcast uses these to index bias vector.
    auto tCcCol = args.tCcD;

    return ConsumerStoreCallbacks(mCol, tCcCol, params);
  }
};

} // namespace cutlass::epilogue::fusion
