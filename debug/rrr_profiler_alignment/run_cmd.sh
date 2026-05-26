#!/bin/bash
# Run profiler RRR_6 kernel
./cutlass_benchmarks_gemm_sycl --kernel=BmgGemmBF16BF16FP32_RRR_6 --m=8192 --n=8192 --k=4096
