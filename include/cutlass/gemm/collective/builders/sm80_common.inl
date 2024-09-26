/***************************************************************************************************
 * Copyright (c) 2024 - 2024 Codeplay Software Ltd. All rights reserved.
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

#include "cute/arch/mma_sm80.hpp"
#include "cute/atom/mma_traits_sm80.hpp"
#include "cute/arch/copy_sm80.hpp"


namespace cutlass::gemm::collective::detail {
  //================== Floating Point MMA ==================//
  template<typename ElementA, typename ElementB, typename ElementMMA>
  struct getMMAType;

  template <class LayoutA, typename ElementA, typename ElementAccumulator>
  struct getMemoryAtomsOperandA;

  template <class LayoutB, typename ElementB, typename ElementAccumulator>
  struct getMemoryAtomsOperandB;

  template<>
  struct getMMAType<cutlass::half_t, cutlass::half_t, float> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_32, _32, _8>
                        >;
  };

  template<>
  struct getMMAType<cutlass::bfloat16_t, cutlass::bfloat16_t, float> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F32BF16BF16F32_TN>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_32, _32, _8>
                        >;
  };

  template<>
  struct getMMAType<cutlass::half_t,  cutlass::half_t, cutlass::half_t> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_F16F16F16F16_TN>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_32, _32, _16>
                        >;
  };
  
  template<>
  struct getMMAType<cutlass::tfloat32_t, cutlass::tfloat32_t, float> {
    using MMA_Atom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_32, _32, _8>
                        >;
  };

  template<>
  struct getMMAType<float, float, float> {
    using MMA_Atom = MMA_Atom<SM80_16x8x8_F32TF32TF32F32_TN>; // Use TF32 MMA when both operands are F32
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_32, _32, _8>
                        >;
  };

  template<>
  struct getMMAType<double, double, double> {
    using MMA_Atom = MMA_Atom<SM80_8x8x4_F64F64F64F64_TN>;
    using TiledMma = TiledMMA<
                      MMA_Atom,
                      Layout<Shape<_2,_2,_1>>,
                      Tile<Layout<Shape<_16,_2>,Stride<_2,_1>>,
                           Layout<Shape<_16,_2>,Stride<_2,_1>>,
                           Underscore>>;
  };
  
  //================== Integer MMA (with saturation only) ==================//
  template<>
  struct getMMAType<int8_t, int8_t, int32_t> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_S32S8S8S32_TN_SATURATE>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_64, _64, _16>
                        >;
  };

  template<>
  struct getMMAType<int8_t, uint8_t, int32_t> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_S32S8U8S32_TN_SATURATE>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_64, _64, _16>
                        >;
  };

  template <>
  struct getMMAType<uint8_t, int8_t, int32_t> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_S32U8S8S32_TN_SATURATE>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_64, _64, _16>
                        >;
  };

  template<>
  struct getMMAType<uint8_t, uint8_t, int32_t> {
    using MMA_Atom = MMA_Atom<SM80_16x8x16_S32U8U8S32_TN_SATURATE>;
    using TiledMMA = TiledMMA<MMA_Atom,
                        Layout<Shape<_2,_2,_1>, Stride<_2,_1,_1>>,
                        Tile<_64, _64, _16>
                        >;
  };

  //================== Memory Layouts ==================//

  template<>
  struct getMemoryAtomsOperandA<cutlass::layout::RowMajor, cutlass::half_t, float> {
    using SmemLayoutAtom = decltype(
                            composition(Swizzle<3,3,3>{},
                            Layout<Shape < _8,_64>,
                            Stride<_64, _1>>{})
                          );
    
    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_8>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, half_t>;
  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, cutlass::half_t, float> {
    using SmemLayoutAtom = decltype(
      composition(Swizzle<3,3,3>{},
                Layout<Shape <_64, _8>,
                       Stride< _1,_64>>{}));

    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, half_t>{},
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},
                    Layout<Shape < _8, _1>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U16x8_LDSM_T, half_t>;
  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, double, double> {
    using SmemLayoutAtom = decltype(
      composition(Swizzle<2,0,4>{},
                  Layout<Shape <_4,_16>,
                         Stride<_1, _4>>{}));
    
    using GmemTiledCopy = decltype(
          make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<double>, double>{},
                    Layout<Shape < _8,_16>,
                           Stride<_16, _1>>{},
                    Layout<Shape<_1,_1>>{}));
                  
    using SmemCopyAtom = Copy_Atom<DefaultCopy, double>;

  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, double, double> {
    using SmemLayoutAtom = decltype(
      composition(Swizzle<2,2,2>{},
                  Layout<Shape <_16, _4>,
                         Stride< _1,_16>>{}));
  
    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, double>{}, 
                    Layout<Shape <_16, _8>,
                           Stride< _1,_16>>{},                           
                    Layout<Shape<_2,_1>>{}));          

    using SmemCopyAtom = Copy_Atom<DefaultCopy, double>;      

  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, int8_t, int32_t> {
    using SmemLayoutAtom = decltype(
      composition(
        Swizzle<2,4,3>{},
        Layout<Shape <_16,_64>,
              Stride<_64, _1>>{}));

    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                      Layout<Shape <_32,_4>,
                            Stride< _4,_1>>{},
                      Layout<Shape<_1,Int<16>>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;
  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, int8_t, int32_t> {
    using SmemLayoutAtom = decltype(
      composition(
        Swizzle<2,0,8>{},
        Layout<Shape <_64, _16>,
              Stride<_1, _64>>{}));

    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, int8_t>{},
                      Layout<Shape <_4, _32>,
                            Stride<_1, _4>>{},
                      Layout<Shape<_1,Int<16>>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, uint8_t>;
  };

  template <>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, float, float> {
    using SmemLayoutAtom = decltype(
      composition(Swizzle<3,2,3>{},
                Layout<Shape < _8,_32>,
                       Stride<_32, _1>>{}));

    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>{},
                    Layout<Shape <_16,_8>,
                           Stride< _8,_1>>{},
                    Layout<Shape < _1,_4>>{}));
                  
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, float>;
  };

  template <>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, float, float> {
    using SmemLayoutAtom = decltype(
      composition(Swizzle<2,2,3>{},
                Layout<Shape < _8,_16>,
                       Stride<_16, _1>>{}));

    using GmemTiledCopy = decltype(
      make_tiled_copy(Copy_Atom<SM80_CP_ASYNC_CACHEALWAYS<cute::uint128_t>, float>{},
                    Layout<Shape <_32,_4>,
                           Stride< _4,_1>>{},
                    Layout<Shape < _1,_4>>{}));

    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, float>;
  };

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, cutlass::bfloat16_t, float> : 
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, cutlass::half_t, float> {};

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, cutlass::bfloat16_t, float> : 
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, cutlass::half_t,float> {};

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, uint8_t, int32_t> :
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, int8_t, int32_t>  {};

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, uint8_t, int32_t> :
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, int8_t, int32_t> {};

  template<>
  struct getMemoryAtomsOperandA <cutlass::layout::RowMajor, cutlass::tfloat32_t, float> : 
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, float, float> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, cutlass::half_t, float> :
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, cutlass::half_t, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, cutlass::half_t, float> :
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, cutlass::half_t, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, cutlass::bfloat16_t, float> :
         getMemoryAtomsOperandB <cutlass::layout::RowMajor, cutlass::half_t, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, cutlass::bfloat16_t, float> :
         getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, cutlass::half_t, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, float, float> :
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, float, float> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, float, float> :
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, float, float> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, cutlass::tfloat32_t, float> :
         getMemoryAtomsOperandB <cutlass::layout::RowMajor, float, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, cutlass::tfloat32_t, float> :
         getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, float, float>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, double, double> :
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, double, double>  {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, double, double> :
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, double, double> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, int8_t, int32_t> :
         getMemoryAtomsOperandA <cutlass::layout::ColumnMajor, int8_t, int32_t> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, int8_t, int32_t> :
         getMemoryAtomsOperandA <cutlass::layout::RowMajor, int8_t, int32_t> {};

  template<>
  struct getMemoryAtomsOperandB <cutlass::layout::RowMajor, uint8_t, int32_t> : 
         getMemoryAtomsOperandB <cutlass::layout::RowMajor, int8_t, int32_t>  {};

  template <>
  struct  getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, uint8_t, int32_t> :
          getMemoryAtomsOperandB <cutlass::layout::ColumnMajor, int8_t, int32_t> {};

}
