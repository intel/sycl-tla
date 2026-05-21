# Benchmarks

```
cd cutlass-fork/build/
```

## Compiling GEMM benchmarks with CUDA backend
```
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=OFF -DDPCPP_SYCL_TARGET=nvptx64_nvidia_cuda -DDPCPP_SYCL_ARCH=sm_80 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gemm_cuda
./benchmarks/gemm/cutlass_benchmarks_gemm_cuda --config_file=../benchmarks/device/ampere/input_files/input_gemm.in
```

## Compiling and Running GEMM benchmarks with default configurations with CUDA backend
```
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=OFF -DDPCPP_SYCL_TARGET=nvptx64_nvidia_cuda -DDPCPP_SYCL_ARCH=sm_80 -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gemm_cuda
```

## Compiling GEMM benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_gemm_sycl
./benchmarks/gemm/cutlass_benchmarks_gemm --config_file=../benchmarks/device/pvc/input_files/input_gemm.in
```

## Compiling and Running GEMM benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_gemm_sycl
```

## Intel GEMM profiler default and custom configurations

The Intel GEMM profiler workflow under `test/benchmarks/` now splits configuration into:

- `test/benchmarks/build_config_bmg_perf.json`: build-time CMake and compiler environment
- `test/benchmarks/runtime_config_bmg_perf.json`: runtime environment used by the search workflow

`default_compiler_profiles()` loads both files and emits them into the workspace `compiler_profiles.json`.
When `build_config.cmake_vars.DPCPP_SYCL_TARGET` is set to `auto`, the workflow detects the visible Intel GPU with `xpu-smi discovery`, honors `ZE_AFFINITY_MASK` when selecting a device, and writes the resolved target to `reports/device_target_detection.json` plus the generated CMake build plan. Explicit non-`auto` targets still take precedence.

### Default configuration

The checked-in default is the current **best-known validated BMG performance baseline** used by the profiler:

- build config defaults to `selected_compile_variant = perf_default`
- runtime config defaults to `selected_runtime_variant = default`
- `CUTLASS_SYCL_PROFILING_ENABLED=OFF` avoids queue profiling overhead in benchmark runs
- `CUTLASS_ENABLE_EXAMPLES=OFF` and `CUTLASS_ENABLE_TESTS=OFF` keep profiler-oriented builds lean
- compile env keeps the validated 256-GRF + large-register-file settings
- runtime env only injects the active execution settings such as `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`

This means the out-of-box config is already tuned for the current BMG search flow. Experimental build variants remain available, but they are not treated as validated replacements for the default baseline.

### Expanded BMG GEMM/StreamK tile search

The default persisted catalog remains the validated level-0 set. For larger B70/BMG exploration, the profiler can opt into an expanded benchmark-backed Gemm/StreamK/DataParallel/SplitK catalog:

```
python3 -m intel_gemm_profiler.workflow \
  --kernel-catalog-source expanded_bmg \
  --build-candidate-benchmark \
  ...
```

This catalog enumerates StreamK/DataParallel/SplitK `TileShape=(M,N,K)` with `M={64,128,256,512}`, `N={64,128,256}`, and `K={32,64}` for the fixed BMG subgroup layout `(sg_m,sg_n)=(8,4)`. Ordinary GEMM uses the same Cartesian set plus source-observed SG8x4 shapes such as `128x256x16`, `128x512x32`, `256x192x64`, and `256x256x16`, and also registers source-observed valid tile/subgroup pairs such as `256x128x32/SG8x2` and `32x128x32/SG4x8`. It covers the current benchmark-backed ordinary GEMM dtype/layout families plus the StreamK/DataParallel/SplitK families. The generated candidate build plan automatically sets:

```
-DCUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON
```

That CMake option registers the matching C++ benchmark entries in `benchmarks/gemm/benchmarks_sycl.hpp`. The split count remains a runtime sweep (`--split_k_slices`) for the single registered SplitK kernel per tile, so only tile/scheduler variants require compile-time registration.

`expanded_streamk` remains accepted as a compatibility alias for the same expanded BMG catalog.

### Custom configuration for experiments

Custom testing is still supported in two ways:

1. **Runtime-only experiments**: create a custom `compiler_profiles.json`, change `runtime_config.selected_runtime_variant` and/or `profiles[*].runtime_env_override`, then pass it to:

   ```
   python3 test/benchmarks/run_phase_a.py --compiler-profiles-json /path/to/compiler_profiles.json ...
   python3 test/benchmarks/run_phase_b.py --compiler-profiles-json /path/to/compiler_profiles.json ...
   ```

2. **Build-time experiments**: change `build_config.selected_compile_variant` or `build_config.compile_env_variants`, rebuild the benchmark/example binaries with that config, then point the workflow to the rebuilt executables with `--benchmark-exe` and `--streamk-example-exe`.

At the moment, the workflow consumes `runtime_config` directly during execution; `build_config` is the recorded source of truth for how the benchmark binaries should be built for each experiment.

For example, `perf_128grf_experiment` is intentionally marked as a **needs-validation** experiment. The validated production baseline remains `perf_default` until B60 A/B measurements prove otherwise.

## Compiling Flash Attention v2 benchmarks with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja cutlass_benchmarks_flash_attention
./benchmarks/flash_attention/flash_attention_prefill/cutlass_benchmarks_flash_attention_prefill_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_prefill_extend_nokvcache.in
./benchmarks/flash_attention/flash_attention_prefill_cachedKV/cutlass_benchmarks_flash_attention_prefill_cachedkv_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_prefill_extend_kvcache.in
./benchmarks/flash_attention/flash_attention_decode/cutlass_benchmarks_flash_attention_decode_xe --config_file=../benchmarks/device/bmg/input_files/input_sglang_flash_attention_decode_kvcache.in
```

## Compiling and Running Flash Attention v2 benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks_flash_attention
```

## Compiling and Running all benchmarks with default configurations with Intel Xe backend
```
# Choose DPCPP_SYCL_TARGET from 
# target = intel_gpu_pvc | intel_gpu_bmg_g21
cmake .. -GNinja -DCUTLASS_ENABLE_SYCL=ON -DDPCPP_SYCL_TARGET=$target -DCUTLASS_ENABLE_BENCHMARKS=ON -DCUTLASS_ENABLE_TESTS=ON

ninja benchmarks
```
