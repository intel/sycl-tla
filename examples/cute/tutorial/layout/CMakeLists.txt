# Copyright (c) 2023 - 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright (C) 2025 Intel Corporation, All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

if (CUTLASS_ENABLE_SYCL)
  message(STATUS "Building CuTe Layout tutorial examples with SYCL")

  cutlass_example_add_executable(
    cute_layout_01_rank_depth_size
    01_rank_depth_size.cpp
  )

  cutlass_example_add_executable(
    cute_layout_02_hierarchical_access
    02_hierarchical_access.cpp
  )

  cutlass_example_add_executable(
    cute_layout_03_construction
    03_layout_construction.cpp
  )

  cutlass_example_add_executable(
    cute_layout_04_using_print
    04_using_layout_print.cpp
  )

  cutlass_example_add_executable(
    cute_layout_05_compatibility
    05_layout_compatibility.cpp
  )

  cutlass_example_add_executable(
    cute_layout_06_coordinate_mapping
    06_coordinate_mapping.cpp
  )

  cutlass_example_add_executable(
    cute_layout_07_index_mapping
    07_index_mapping.cpp
  )

  cutlass_example_add_executable(
    cute_layout_08_manipulation
    08_layout_manipulation.cpp
  )

  cutlass_example_add_executable(
    cute_layout_hierarchical_vs_flat
    hierarchical_vs_flat_demo.cpp
  )

   cutlass_example_add_executable(
    cute_layout_static_vs_dynamic
    static_vs_dynamic_demo.cpp
  )

else()
  message(STATUS "Skipping CuTe Layout tutorial examples (SYCL not enabled)")
endif()
