# Copyright (c) 2024 - 2025 Codeplay Software Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

foreach(INPUT_TYPE bfloat16_t half_t)
  set(input_name "cutlass_test_unit_flash_attention_decode_bf16")
  set(test_name_input "XE_Flash_Attention_Decode_bf16")
  set(is_bf16 true)
  set(MMAOP MMAOperationBF16)
  set(OUT_TYPE_LIST float)
  if(${INPUT_TYPE} STREQUAL "half_t")
    set(input_name "cutlass_test_unit_flash_attention_decode_fp16")
    set(is_bf16 false)
    list(APPEND OUT_TYPE_LIST half_t)
    set(test_name_input "XE_Flash_Attention_Decode_fp16")
    set(MMAOP MMAOperationFP16)
  else()
    list(APPEND OUT_TYPE_LIST bfloat16_t)
  endif()

  foreach(OUT_TYPE IN LISTS OUT_TYPE_LIST)
    set(out_name "fp32")
    set(TILEDCOPYSTORE GmemTiledCopyStoreU32)
    if(${OUT_TYPE} STREQUAL "bfloat16_t")
      set(out_name "bf16")
      set(TILEDCOPYSTORE GmemTiledCopyStoreU16)
    elseif(${OUT_TYPE} STREQUAL "half_t")
      set(out_name "fp16")
      set(TILEDCOPYSTORE GmemTiledCopyStoreU16)
    endif()
    set(input_name_out "${input_name}_fp32_${out_name}")
    set(test_name_input_out "${test_name_input}_fp32_${out_name}")
    foreach(HEAD_DIM 64 96 128 192)
      foreach(KV_TILE 512 1024)
        set(NUM_SG 8)
        if(${KV_TILE} EQUAL 1024)
          set(NUM_SG 16)
        endif()
        set(out_exe "${input_name_out}_h${HEAD_DIM}_${KV_TILE}_xe")
        set(TEST_NAME "${test_name_input_out}_h${HEAD_DIM}_KVTile${KV_TILE}")
        cutlass_test_unit_add_executable(
          ${out_exe} xe_flash_decode.cpp
        )
        target_compile_definitions(${out_exe} PRIVATE
                                    INPUT_TYPE=${INPUT_TYPE}
                                    OUT_TYPE=${OUT_TYPE}
                                    TEST_NAME=${TEST_NAME}
                                    HEAD_DIM=${HEAD_DIM}
                                    KV_TILE=${KV_TILE}
                                    NUM_SG=${NUM_SG}
                                    MMAOP=${MMAOP}
                                    TILEDCOPYSTORE=${TILEDCOPYSTORE})
      endforeach()
    endforeach()
  endforeach()
endforeach()

add_custom_target(
  cutlass_test_unit_flash_attention_decode
  DEPENDS
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h64_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h64_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h64_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h64_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h64_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h64_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h64_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h64_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h96_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h96_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h96_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h96_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h96_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h96_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h96_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h96_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h128_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h128_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h128_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h128_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h128_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h128_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h128_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h128_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h192_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_fp32_h192_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h192_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp32_h192_1024_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h192_512_xe
  cutlass_test_unit_flash_attention_decode_bf16_fp32_bf16_h192_1024_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h192_512_xe
  cutlass_test_unit_flash_attention_decode_fp16_fp32_fp16_h192_1024_xe
)

add_custom_target(
  test_unit_flash_attention_decode
  DEPENDS
  test_unit_flash_attention_decode_bf16_fp32_fp32_h64_512_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h64_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h64_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h64_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h64_512_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h64_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h64_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h64_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h96_512_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h96_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h96_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h96_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h96_512_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h96_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h96_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h96_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h128_512_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h128_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h128_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h128_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h128_512_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h128_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h128_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h128_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h192_512_xe
  test_unit_flash_attention_decode_bf16_fp32_fp32_h192_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h192_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp32_h192_1024_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h192_512_xe
  test_unit_flash_attention_decode_bf16_fp32_bf16_h192_1024_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h192_512_xe
  test_unit_flash_attention_decode_fp16_fp32_fp16_h192_1024_xe
)
