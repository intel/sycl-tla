# Google Benchmark STUB — headers only, no library build
# run_direct() bypasses ::benchmark::RunSpecifiedBenchmarks() entirely
set(GOOGLEBENCHMARK_DIR "" CACHE STRING "Location of GoogleBenchmark headers")
if(NOT TARGET benchmark)
  add_library(benchmark INTERFACE)
  add_library(benchmark_main INTERFACE)
  target_include_directories(benchmark INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-src/include)
endif()
