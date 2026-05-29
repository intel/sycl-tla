# Manual Test: Top1 Kernel RRR_Gemm_128x128x32_SG2x4

## Files
- `kernel_list.txt` — kernel name
- `benchmarks_sycl.hpp` — generated mini HPP (41570 bytes)
- `main.cpp` — test harness
- `flags.make` — cmake compile flags
- `link.txt` — cmake link command
- `make.log` — build output (compile failed due to cmake dep issue)

## Kernel Configuration
| Parameter | Value |
|-----------|-------|
| Name | BmgGemmBF16BF16FP32_RRR_Gemm_128x128x32_SG2x4 |
| Data type | BF16 × BF16 → FP32 |
| Layout | RRR (A: RowMajor, B: RowMajor) |
| Tile | 128 × 128 × 32 |
| Subgroup | 2 × 4 |
| Atom | XE_DPAS_TT<8, float, cute::bfloat16_t> |
| Pipeline | 2 (default) |
| Problem | M=8192, N=4096, K=1536 |

## Expected Result
154.9 TFLOPS (from full screening run batch_0507_gpu6.csv)
