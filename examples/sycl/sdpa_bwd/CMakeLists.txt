set(TEST_GROUPS --groups=128)

cutlass_example_add_executable(
  sdpa_backward
  sdpa_backward.cpp
  cnpy.cpp
)
target_link_options(sdpa_backward PUBLIC "-lz")
set_target_properties(sdpa_backward PROPERTIES CXX_COMPILER_LAUNCHER "IGC_VectorAliasBBThreshold=10000" )
set_target_properties(sdpa_backward PROPERTIES CXX_LINKER_LAUNCHER "IGC_VectorAliasBBThreshold=10000" )
