# B70 GEMM Profiler — Presentation Outline (12 pages)

---

## Slide 1: Title
**B70 GEMM Auto-Profiling Pipeline on Intel Xe GPU**

- 1772 kernel configurations screened
- 154.9 TFLOPS peak (RRR layout)
- Fully automated catalog → compile → screen pipeline
- 886 batches, GPU 0-3 round-robin execution

---

## Slide 2: Profiler Architecture Overview

```
┌─────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────┐
│ catalog.py  │───→│ gen_mini_hpp │───→│  cmake make  │───→│   GPU    │
│ kernel      │    │ .py          │    │  + icpx link │    │ screen   │
│ enumeration │    │ per-batch    │    │  33s/batch   │    │ 30s/2k   │
└─────────────┘    └──────────────┘    └──────────────┘    └──────────┘
      ↓                   ↓                   ↓                 ↓
  constraints    macro strip +       perf flags baked    CSV output
  + ali shapes   declare generator   DELETE_ON_ERROR     unified results
```

**Key numbers:**
- 1772 kernels → 886 batches (2/batch)
- 33s compile + 30s screen ≈ 1.5 min/batch
- Full run: ~22 hours (sequential)

---

## Slide 3: Kernel Search Space

**Search dimensions:**

| Dimension | Values | Count |
|-----------|--------|-------|
| tile_m | {8,16,32,64,128,256,512} | 7 |
| tile_n | {32,64,96,128,192,256,512} | 7 |
| tile_k | {16,32,64} | 3 |
| sg_m | {1,2,4,8} | 4 |
| sg_n | {2,4,8} | 3 |
| stages | {1,2,3} | 3 |
| scheduler | {Gemm,StreamK,DataParallel,SplitK} | 4 |
| layout | {RCR, RRR} | 2 |

**Constraints:**
- B70 HW max: subgroup ∈ {16,32} → sg_m × sg_n ∈ {16,32}
- SG 8×8 invalid (64 > 32)
- StreamK/DP/SplitK: SG fixed 8×4

**Coverage (bf16 only):**

| Type | RCR | RRR | Total |
|------|-----|-----|-------|
| Gemm_ SG | 76 | 76 | 152 |
| GemmExhaustive | 684 | 684 | 1368 |
| StreamK | 49 | 32 | 81 |
| DataParallel | 49 | 32 | 81 |
| SplitK | 49 | 32 | 81 |
| Hand-written | 9 | 2 | 11 |
| **Total** | | | **1772** |

---

## Slide 4: Two-Phase Search Strategy

**Phase 1: Compute Shape (Gemm scheduler only)**
- Search: tile_m × tile_n × tile_k × sg_m × sg_n × stages
- Goal: find most compute-efficient configs
- output: top-10 per occupancy bucket

**Phase 2: Execution Strategy (StreamK/DP/SplitK sweep)**
- Input: Phase 1 top-10 compute configs
- Search: scheduler × splits on fixed tile+sg
- Goal: find best scheduler for each compute config

**Key insight from data:**
- On 8192×4096×1536: Gemm/DP/StreamK/SplitK within 1% for same tile
- Scheduler matters at K-skinny or M/N-small problems
- Phase 2 useful for different problem shapes, not same shape

**Future: occupancy-bucket diversity**
- Large-tile (low occupancy): favors DP
- Small-tile (high occupancy): favors StreamK/SplitK
- Retain top-N from MULTIPLE occupancy classes, not just top GFLOPS

---

## Slide 5: Kernel Generation — gen_mini_hpp.py

**Problem:** Full benchmarks_sycl.hpp has 835 lines, 300+ macro-generated kernel types. Compile takes 30+ min. Each batch needs only 2 kernel types.

**Solution:** gen_mini_hpp.py generates per-batch mini HPP.

**Strategy (after multiple iterations):**
1. Copy full HPP preamble (macros + templates)
2. **Strip** ALL BMG_DECLARE_*/BMG_REGISTER_*/BMG_SOURCE_* macro calls and .def includes
3. Generate batch-specific using declarations + CUTLASS_CREATE_GEMM_BENCHMARK
4. Override `register_gemm_benchmarks()` with batch-specific registrations

**Three kernel generation patterns:**

| Pattern | How gen_mini generates |
|---------|----------------------|
| Gemm_ SG | `BMG_DECLARE_GEMM_TILE_SG(PREFIX, CONFIG, ATOM, M,N,K,SGM,SGN)` |
| GemmExhaustive | `BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(...)` |
| StreamK/DP/SplitK | Direct `GemmConfiguration<..., Scheduler::X, ...>` + `CUTLASS_CREATE` |

**Key pitfalls fixed:**
- Preamble strip: prevents redefinition with macro-generated types
- `covered()` check: prevents redefinition with hand-written `using` declarations
- `EXPANDED_STREAMK_TILES` set: must match all full HPP individual calls
- `.DELETE_ON_ERROR`: cmake deletes .o on link failure → `sed -i '/.DELETE_ON_ERROR/d'`

---

## Slide 6: Compile System

**Toolchain:**
- icpx (Intel oneAPI 2025.3)
- cmake 3.28
- SYCL target: spir64_gen, device bmg-g31

**Critical compile flags (baked at build time):**
```bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
```

**Build flow:**
```
gen_mini_hpp.py → benchmark_sycl.hpp (~700 lines)
    ↓
cmake make cutlass_benchmarks_gemm_sycl -j128  → main.cpp.o (33s)
    ↓  (.DELETE_ON_ERROR disabled)
icpx manual link → cutlass_benchmarks_gemm_sycl (1s)
    ↓  (GB stub link fails, manual link with pre-built libbenchmark.a)
GPU screen (ZE_AFFINITY_MASK=N)
```

**Performance impact of preamble stripping:**
- Before: 835-line mini HPP, 300+ extra kernel types → IGC conservative → 148 TFLOPS
- After: 500-line mini HPP, only batch types → IGC aggressive → **155 TFLOPS (+4-6%)**

**Google Benchmark handling:**
- GB headers needed (benchmark::State type)
- GB library NOT needed (run_direct bypasses)
- `cmake/googlebenchmark_stub.cmake`: INTERFACE targets with ALIAS
- Manual link with pre-built `libbenchmark.a`

---

## Slide 7: Runtime & Perf Optimization

**Scheduler Types (all share same CollectiveMma):**

| Scheduler | Kernel | Behavior | Best For |
|-----------|--------|----------|----------|
| Gemm | KernelXe | Static tile assignment | General |
| StreamK | KernelXeCooperative | Dynamic tile dispatch | K-skinny, small batch |
| DataParallel | KernelXeCooperative | Splits M×N grid | Large problems |
| SplitK | KernelXeCooperative | Splits K, partial sums + reduce | K-large, M/N-small |

**SplitK at B70:**
- `PersistentTileSchedulerXeStreamKParams` exposes: `DecompositionMode`, `splits`, `ReductionMode`
- B70 hardware: only `splits=1` works reliably
- `splits=2` causes runtime hang (hardware limitation)
- Fixed: `GemmConfiguration::defaultArguments()` changed splits from 2→1

**Key fixes to `run_direct()`:**
- Added `set_scheduler_splits()` call (was in `run()` but missing in `run_direct()`)
- Passes `options.split_k_slices` → `arguments.scheduler.splits`

---

## Slide 8: Test Results — Top 20

| # | TFLOPS | Kernel | Layout |
|---|--------|--------|--------|
| 1 | 154.9 | Gemm_128x128x32_SG2x4 | RRR |
| 2 | 149.6 | Gemm_128x256x32_SG4x4 | RRR |
| 3 | 149.2 | Gemm_256x128x32_SG4x4 | RRR |
| 4 | 147.5 | Gemm_128x128x32_SG2x4 | RCR |
| 5 | 147.5 | Gemm_256x128x32_SG8x2 | RRR |
| 6 | 146.6 | Gemm_256x128x32_SG4x4 | RCR |
| 7 | 145.6 | GemmExhaustive_128x256x32_SG2x8_ST2 | RCR |
| 8 | 144.7 | GemmExhaustive_256x128x32_SG4x4_ST2 | RCR |
| 9 | 144.4 | GemmExhaustive_128x256x32_SG2x8_ST3 | RCR |
| 10 | 144.3 | DataParallel_256x256x32 | RCR |

---

## Slide 9: Scheduler Comparison (Same Tile)

**Tile: 256×256×32, Problem: 8192×4096×1536**

| Scheduler | TFLOPS | vs Gemm |
|-----------|--------|---------|
| Gemm (SG8x4) | 136.5 | baseline |
| DataParallel | **144.3** | +5.7% |
| SplitK (splits=1) | **144.2** | +5.6% |
| StreamK | **142.3** | +4.2% |

**Key finding:** On large GEMM, all schedulers perform within 1% of each other. Scheduler choice matters more at K-skinny or small-batch problems.

---

## Slide 10: RRR vs RCR Layout

**Top Config Comparison:**

| Config | RCR TFLOPS | RRR TFLOPS | RRR Win |
|--------|-----------|-----------|---------|
| 128×128×32 SG2x4 | 147.5 | **154.9** | +5.0% |
| 128×256×32 SG4x4 | 131.1 | **149.6** | +14.1% |
| 256×128×32 SG4x4 | 146.6 | **149.2** | +1.8% |

**Statistical summary (992 kernels):**
- RCR: 692 kernels, mean 75.4, max 147.5
- RRR: 75 kernels, mean 84.7, max 154.9
- RRR mean +12.3%, RRR max +5.0%

**Why RRR wins:** RowMajor B → contiguous memory access in TiledMMA → better cache locality and register reuse.

---

## Slide 11: Build & Screen SOP

**One-command screening:**
```bash
RESULTS_DIR=$WS/results_final_full BATCHES=all bash tools/run_seq.sh
```

**Must-check before run:**
- [ ] `source /opt/intel/oneapi/compiler/2025.3/env/vars.sh`
- [ ] Export SYCL_PROGRAM_COMPILE_OPTIONS, IGC_VectorAliasBBThreshold, IGC_ExtraOCLOptions
- [ ] CPU governor = performance
- [ ] `_deps/` restored (googlebenchmark + googletest headers)
- [ ] `touch compiler_depend.ts` in cmake build dir

**Result directory convention:**
| Dir | Content |
|-----|---------|
| `results_original/` | First screening |
| `results_dp_only/` | DataParallel-only test |
| `results_full_fixed/` | 496-batch full run |
| `results_final_full/` | Current 886-batch run |

---

## Slide 12: Lessons Learned & Next Steps

**Major pitfalls overcome:**
1. Perf flags not baked → 4 TFLOPS vs 155 TFLOPS
2. Google Benchmark link failure → stub cmake + manual icpx link
3. .DELETE_ON_ERROR → .o deleted on every link fail
4. gen_mini macro staleness → preamble strip + direct generate
5. covered() macro resolution → Python can't see C++ preprocessor output
6. Remote git sync → `git fetch && git reset --hard` instead of `git pull`

**Next steps:**
- [ ] RRR StreamK/DP/SplitK data collection (886-batch running)
- [ ] Occupancy-bucket top-N selection for Phase 2
- [ ] SplitK splits=1 heuristic
- [ ] Autotune cache implementation
- [ ] RRR + perf flags baked screening for full coverage

**Code location:** `github.com/tinafengfun/sycl-tla`
