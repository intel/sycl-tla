# Google Benchmark — use pre-built static library (header + .a)
# run_direct() bypasses ::benchmark::RunSpecifiedBenchmarks() entirely,
# but cmake needs a real library target (not INTERFACE) to generate correct
# link flags. INTERFACE targets produce -lbenchmark::benchmark which ld fails on.
#
# Headers: CMAKE_CURRENT_SOURCE_DIR/_deps (FetchContent clones into source tree)
# Library: CMAKE_BINARY_DIR/_deps (FetchContent builds into binary tree)
set(GOOGLEBENCHMARK_DIR "" CACHE STRING "Location of GoogleBenchmark headers")
set(GOOGLEBENCHMARK_BUILD_DIR "" CACHE STRING "Location of prebuilt GoogleBenchmark static library tree")
set(_googlebenchmark_include_dir "${CMAKE_CURRENT_SOURCE_DIR}/_deps/googlebenchmark-src/include")
set(_googlebenchmark_library "${CMAKE_BINARY_DIR}/_deps/googlebenchmark-build/src/libbenchmark.a")
if(GOOGLEBENCHMARK_DIR)
  set(_googlebenchmark_include_dir "${GOOGLEBENCHMARK_DIR}/include")
endif()
if(GOOGLEBENCHMARK_BUILD_DIR)
  set(_googlebenchmark_library "${GOOGLEBENCHMARK_BUILD_DIR}/src/libbenchmark.a")
endif()
if(NOT TARGET benchmark)
  add_library(benchmark STATIC IMPORTED)
  set_target_properties(benchmark PROPERTIES
    IMPORTED_LOCATION "${_googlebenchmark_library}"
    INTERFACE_INCLUDE_DIRECTORIES "${_googlebenchmark_include_dir}"
  )
  add_library(benchmark::benchmark ALIAS benchmark)
endif()
