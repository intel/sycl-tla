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

### Default configuration

The checked-in default is the current **best-known validated BMG performance baseline** used by the profiler:

- build config defaults to `selected_compile_variant = perf_default`
- runtime config defaults to `selected_runtime_variant = default`
- `CUTLASS_SYCL_PROFILING_ENABLED=OFF` avoids queue profiling overhead in benchmark runs
- `CUTLASS_ENABLE_EXAMPLES=OFF` and `CUTLASS_ENABLE_TESTS=OFF` keep profiler-oriented builds lean
- compile env keeps the validated 256-GRF + large-register-file settings
- runtime env only injects the active execution settings such as `ONEAPI_DEVICE_SELECTOR=level_zero:gpu`

This means the out-of-box config is already tuned for the current BMG search flow, while keeping experimental variants available.

### Custom configuration for experiments

Custom testing is still supported in two ways:

1. **Runtime-only experiments**: create a custom `compiler_profiles.json`, change `runtime_config.selected_runtime_variant` and/or `profiles[*].runtime_env_override`, then pass it to:

   ```
   python3 test/benchmarks/run_phase_a.py --compiler-profiles-json /path/to/compiler_profiles.json ...
   python3 test/benchmarks/run_phase_b.py --compiler-profiles-json /path/to/compiler_profiles.json ...
   ```

2. **Build-time experiments**: change `build_config.selected_compile_variant` or `build_config.compile_env_variants`, rebuild the benchmark/example binaries with that config, then point the workflow to the rebuilt executables with `--benchmark-exe` and `--streamk-example-exe`.

At the moment, the workflow consumes `runtime_config` directly during execution; `build_config` is the recorded source of truth for how the benchmark binaries should be built for each experiment.

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
