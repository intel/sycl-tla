# Google Benchmark — use pre-built static library (header + .a)
# run_direct() bypasses ::benchmark::RunSpecifiedBenchmarks() entirely,
# but cmake needs a real library target (not INTERFACE) to generate correct
# link flags. INTERFACE targets produce -lbenchmark::benchmark which ld fails on.
#
# Headers: CMAKE_CURRENT_SOURCE_DIR/_deps (FetchContent clones into source tree)
# Library: CMAKE_BINARY_DIR/_deps (FetchContent builds into binary tree)
set(GOOGLEBENCHMARK_DIR "" CACHE STRING "Location of GoogleBenchmark headers")
if(NOT TARGET benchmark)
  add_library(benchmark STATIC IMPORTED)
  set_target_properties(benchmark PROPERTIES
    IMPORTED_LOCATION "${CMAKE_BINARY_DIR}/_deps/googlebenchmark-build/src/libbenchmark.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-src/include"
  )
  add_library(benchmark::benchmark ALIAS benchmark)
endif()
