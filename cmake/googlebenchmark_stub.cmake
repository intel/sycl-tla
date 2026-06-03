# Google Benchmark — use pre-built static library (header + .a)
# run_direct() bypasses ::benchmark::RunSpecifiedBenchmarks() entirely,
# but cmake needs a real library target (not INTERFACE) to generate correct
# link flags. INTERFACE targets produce -lbenchmark::benchmark which ld fails on.
set(GOOGLEBENCHMARK_DIR "" CACHE STRING "Location of GoogleBenchmark headers")
if(NOT TARGET benchmark)
  add_library(benchmark STATIC IMPORTED)
  set_target_properties(benchmark PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-build/src/libbenchmark.a"
    INTERFACE_INCLUDE_DIRECTORIES "${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-src/include"
  )
  add_library(benchmark::benchmark ALIAS benchmark)
  add_library(benchmark_main STATIC IMPORTED)
  set_target_properties(benchmark_main PROPERTIES
    IMPORTED_LOCATION "${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-build/src/libbenchmark_main.a"
  )
endif()
