cutlass_example_add_executable(
  gemm_dq_reproducer
  gemm_dq_reproducer.cpp
)
target_include_directories(gemm_dq_reproducer PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../sdpa_bwd)
