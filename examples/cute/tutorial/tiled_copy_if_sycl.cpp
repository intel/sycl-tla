/***************************************************************************************************
 * Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <sycl/sycl.hpp>
#include <syclcompat.hpp>

#include <cute/tensor.hpp>

#include "cutlass/util/print_error.hpp"

// This example extends `tiled_copy` using predicate tensors to guard memory accesses performed
// by `cute::copy_if()`. This enables tensors to have shapes that are not integer multiples of
// block sizes.
//
// This is accomplished by instantiating a tensor of coordinates which correspond to tensor elements
// to be accessed and then computing a predicate tensor which masks accesses. The example demonstrates
// how constructing of an identity tensor containing coordinates and a predicate tensor containing
// mask bits can be implemented using the same CuTe operations used to tile the tensors in
// Global Memory.
//
// This example implements two variants:
//  - copy_if_kernel() uses `cute::local_partition()` to construct each thread's slice
//  - copy_if_kernel_vectorized() uses `make_tiled_copy() to implement vectorized memory accesses.
//
// The tensor shapes and strides must be divisible by the shape of the vector access.
//

/// Simple copy kernel.
//
// Uses local_partition() to partition a tile among threads arranged as (THR_M, THR_N).
template <class TensorS, class TensorD, class BlockShape, class ThreadLayout>
void copy_if_kernel(TensorS S, TensorD D, BlockShape block_shape, ThreadLayout)
{
    using namespace cute;

    // Construct a coordinate tensor whose elements are the coordinates used to access tensors S and D.
    auto shape_S = shape(S);
    Tensor C = make_identity_tensor(shape_S); // (m, n) coordinates
    // Construct a predicate tensor which compares the coordinates with the original shape
    Tensor P = cute::lazy::transform(C, [&](auto c) { return elem_less(c, shape_S); });

    // Tile the input tensor into blocks
    auto block_coord = make_coord(syclcompat::work_group_id::x(), syclcompat::work_group_id::y());
    Tensor tile_S = local_tile(S, block_shape, block_coord); // (BlockShape_M, BlockShape_N)
    Tensor tile_P = local_tile(P, block_shape, block_coord); // (BlockShape_M, BlockShape_N)
    Tensor tile_D = local_tile(D, block_shape, block_coord); // (BlockShape_M, BlockShape_N)

    // Construct a partitioning of the tile among threads with the given thread arrangement.

    // Concept:                         Tensor  ThrLayout       ThrIndex
    Tensor thr_tile_S = local_partition(tile_S, ThreadLayout{}, syclcompat::local_id::x());
    Tensor thr_tile_P = local_partition(tile_P, ThreadLayout{}, syclcompat::local_id::x());
    Tensor thr_tile_D = local_partition(tile_D, ThreadLayout{}, syclcompat::local_id::x());

    // Copy from GMEM to GMEM using `thr_tile_P` to guard accesses.
    copy_if(thr_tile_P, thr_tile_S, thr_tile_D);
}

/// Main function
int main(int argc, char** argv)
{
    //
    // Given a 2D shape, perform an efficient copy using predicates to guard accesses
    //

    using namespace cute;
    using Element = float;

    // Define a tensor shape with dynamic extents (m, n)
    auto tensor_shape = make_shape(528, 300); // Not divisible by block shape

    //
    // Allocate and initialize
    //
    std::vector<Element> h_S(size(tensor_shape));
    std::vector<Element> h_D(size(tensor_shape));

    auto d_S = syclcompat::malloc<Element>(size(tensor_shape));
    auto d_D = syclcompat::malloc<Element>(size(tensor_shape));
    auto d_Zero = syclcompat::malloc<Element>(size(tensor_shape));

    for (size_t i = 0; i < h_S.size(); ++i) {
        h_S[i] = static_cast<Element>(i);
    }

    syclcompat::memcpy<Element>(d_S, h_S.data(), size(tensor_shape));
    syclcompat::memcpy<Element>(d_D, h_D.data(), size(tensor_shape));
    syclcompat::memcpy<Element>(d_Zero, h_D.data(), size(tensor_shape));

    //
    // Make tensors
    //
    Tensor tensor_S = make_tensor(d_S, make_layout(tensor_shape));
    Tensor tensor_D = make_tensor(d_D, make_layout(tensor_shape));

    //
    // Partition
    //

    // Define a statically sized block (M, N).
    //
    // Note, by convention, capital letters are used to represent static modes.
    auto block_shape = make_shape(Int<128>{}, Int<64>{});

    // Tile the tensor (m, n) ==> ((M, N), m', n') where (M, N) is the static tile
    // shape, and modes (m', n') correspond to the number of tiles.
    //
    // These will be used to determine the CUDA kernel grid dimensinos.
    Tensor tiled_tensor_D = tiled_divide(tensor_D, block_shape);        // ((M, N), m', n')

    // Describes the layout of threads which is then replicated to tile 'block_shape.'
    Layout thr_layout = make_layout(make_shape(Int<32>{}, Int< 8>{}));  // (ThrM, ThrN)

    auto gridDim  = syclcompat::dim3(size<1>(tiled_tensor_D), size<2>(tiled_tensor_D));
    auto blockDim = syclcompat::dim3(size(thr_layout));

    //
    // Launch the kernel
    //
    syclcompat::launch<copy_if_kernel<decltype(tensor_S), decltype(tensor_D), 
        decltype(block_shape), decltype(thr_layout)>>(
            gridDim, blockDim, tensor_S, tensor_D, block_shape, thr_layout
        );
    syclcompat::wait_and_throw();

    //
    // Verify
    //
    syclcompat::memcpy<Element>(h_D.data(), d_D, size(tensor_shape));

    int32_t errors = 0;
    int32_t const kErrorLimit = 10;

    if (h_S.size() != h_D.size()) {
      return 1;
    }

    for (size_t i = 0; i < h_D.size(); ++i) {
        if (h_S[i] != h_D[i]) {
        std::cerr << "Error. S[" << i << "]: " << h_S[i] << ",   D[" << i << "]: " << h_D[i] << std::endl;

        if (++errors >= kErrorLimit) {
            std::cerr << "Aborting on " << kErrorLimit << "nth error." << std::endl;
            return -1;
        }
        }
    }

    std::cout << "Success." << std::endl;

    return 0;
}