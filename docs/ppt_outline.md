# B70 GEMM Profiler — Technical Deep Dive (16 slides)

---

## Slide 1: Title — B70 GEMM Auto-Profiling Pipeline

**Profiling Intel Xe GPU (BMG G31) with Cutlass 3.x**

- **6,007 lines** of profiling infrastructure across 14 Python modules
- **1,772 unique BF16 kernel configurations** screened
- **154.9 TFLOPS** peak (RRR layout, 128×128×32 SG2×4)
- **886 batches**, 4-GPU round-robin, fully automated

```
Problem: M=8192, N=4096, K=1536, BF16
Hardware: BMG G31, Intel Xe driver
Compiler: Intel oneAPI 2025.3 (icpx)
Framework: Cutlass 3.x with SYCL backend
```

---

## Slide 2: Profiler Module Architecture

```
┌──────────────────────────────────────────────────────────┐
│                 Intel GEMM Profiler                       │
│                  (6,007 lines Python)                     │
├──────────┬──────────┬──────────┬──────────┬─────────────┤
│ catalog  │constraint│ device   │ selector │  runner     │
│  945 LOC │  453 LOC │ target   │ 345 LOC  │  498 LOC    │
│ kernel   │ HW limits│  285 LOC │ filter   │  benchmark  │
│ enum     │ filtering│ GPU caps │ matching │  execution  │
├──────────┼──────────┼──────────┼──────────┼─────────────┤
│candidates│prefilter │schemas   │workflow  │ali_dataset  │
│  734 LOC │ 193 LOC  │ 260 LOC  │1384 LOC  │  190 LOC    │
│ tile×sg  │tuning    │JSON      │CLI+pipe  │Ali shapes   │
│ collision│heuristics│contract  │orchestra │dataset      │
├──────────┴──────────┴──────────┴──────────┴─────────────┤
│            Tools Layer (compile + screen)                 │
│  gen_mini_hpp.py (140 LOC)  run_seq.sh (86 LOC)          │
│  gen_main.py (36 LOC)       cmake stub (9 LOC)           │
└──────────────────────────────────────────────────────────┘
```

**Key modules:**
- `catalog.py`: kernel enumeration (945 lines, 4 generation functions)
- `constraints.py`: HW limits, tile validity gating (453 lines)
- `runner.py`: benchmark execution, perf counter parsing (498 lines)
- `workflow.py`: CLI, variant switching, JSON config (1,384 lines)

---

## Slide 3: Kernel Coverage — Quantified

**Kernel generation functions and their yield:**

| Source Function | BF16-RCR | BF16-RRR | Total | Lines |
|---|---|---|---|
| `generated_expanded_streamk_kernel_catalog()` | 458 | 0 | **458** | L463 |
| `exhaustive_regular_gemm_tile_candidates()` | 684 | 684 | **1,368** | L265 |
| `exhaustive_streamk_tile_candidates()` | 192 | 192 | **384** | L124 |
| Hard-coded seed + hand-written | 216 | 2 | **218** | L327 |
| **Total** | **1,550** | **878** | **2,428** | — |
| After dedup + SG8×8 filter | **1,100** | **672** | **1,772** | — |

**By kernel type (BF16 only, deduplicated):**

| Pattern | RCR | RRR | How defined |
|---------|-----|-----|-------------|
| `GemmExhaustive_{tile}_SG{sg}_{ST}` | 684 | 684 | `catalog.py` L265–317 |
| `Gemm_{tile}_SG{sg}` (.def sourced) | 76 | 76 | `source_templates.py` L75 |
| `StreamK_{tile}` | 49 | 32 | `catalog.py` L124–191 |
| `DataParallel_{tile}` | 49 | 32 | `catalog.py` L124–191 |
| `SplitK_{tile}` | 49 | 32 | `catalog.py` L124–191 |
| Hand-written (`RCR_N`, `RRR_N`) | 9 | 2 | `benchmarks_sycl.hpp` L200–280 |

---

## Slide 4: Search Space Dimensions

**Compute shape (Phase 1):**

| Dimension | Range | Count | Constraint |
|-----------|-------|-------|------------|
| tile_m | {8, 16, 32, 64, 128, 256, 512} | 7 | × tile_n × tile_k ≤ LDS budget |
| tile_n | {32, 64, 96, 128, 192, 256, 512} | 7 | tile_n % (sg_n × 16) = 0 |
| tile_k | {16, 32, 64} | 3 | tile_k % 16 = 0 |
| sg_m | {1, 2, 4, 8} | 4 | sg_m × sg_n ∈ {16, 32} |
| sg_n | {2, 4, 8} | 3 | B70 HW max subgroup = 32 |
| stages | {1, 2, 3} | 3 | `EXHAUSTIVE_REGULAR_GEMM_STAGES` |
| scheduler | {Gemm, StreamK, DP, SplitK} | 4 | StreamK/DP/SplitK: SG fixed 8×4 |
| layout | {RCR, RRR} | 2 | A: RowMajor, B: ColumnMajor or RowMajor |

**Validation: `is_valid_xe2_tile_sg()`** (`source_templates.py` L75):
```python
tile_m % (sg_m * 8)  == 0   # M alignment to atom
tile_n % (sg_n * 16) == 0   # N alignment to atom
tile_k % (1  * 16) == 0     # K alignment to atom
sg_m * sg_n ∈ {16, 32}       # B70 subgroup product
reg_m * reg_n * 16 ≤ 256     # GRF register pressure
```

---

## Slide 5: Catalog Generation Layers

**`generated_layered_bmg_kernel_catalog()` (L631–730, 100 lines):**

```
Layer 0: generated_expanded_streamk_kernel_catalog()   →  458 kernels
    │   (hand-crafted + expanded + StreamK seeds)
    │
Layer 1: exhaustive_regular_gemm_tile_candidates()      → +684 RCR
    │   (full tile×sg×stage enumeration, layout=rcr)
    │
Layer 2: exhaustive_regular_gemm_tile_candidates(       → +684 RRR
    │        layout=rrr)  ← added in later patch
    │
Layer 3: exhaustive_streamk_tile_candidates()           → +192 RCR
    │   (StreamK/DP/SplitK for all legal tiles, K≥32)
    │
Layer 4: exhaustive_streamk_tile_candidates(            → +192 RRR
    │        layout=rrr)  ← added in later patch
    │
    ▼
    Dedup by kernel_name + filter SG8×8 + exclude example kernels
    = 1,772 kernels → 886 batches (2/batch)
```

**Why layered?** Allows incremental addition of search dimensions without regenerating everything. Each layer can be toggled independently via constraints.

---

## Slide 6: Template Generation — gen_mini_hpp.py Deep Dive

**Problem:** Full `benchmarks_sycl.hpp` = 835 lines, 300+ macro-generated types. IGC faces 500+ kernel types even for a 2-kernel batch.

**Design (140 lines, 6 functions):**

```python
def gen_mini(kernels, output):
    text = full_text()                    # Read full HPP
    
    # 1. STRIP preamble: remove ALL BMG_DECLARE_*/BMG_REGISTER_*/BMG_SOURCE_*
    #    macro calls and .def includes (but keep #define definitions)
    preamble_lines = []
    for line in text[:register_pos].split('\n'):
        if re.match(r'^(BMG_DECLARE_|BMG_REGISTER_|BMG_SOURCE_)\(', stripped):
            continue
        if '.def' in stripped: continue
        preamble_lines.append(line)
    
    # 2. GENERATE batch-specific types
    for k in kernels:
        t, pfx, p = classify(k)          # regex: 'ge', 'gs', 'sk', 'hw'
        if t == 'ge':
            declares.append(BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(...))
        elif t == 'gs':
            declares.append(BMG_DECLARE_GEMM_TILE_SG(...))
        else:  # sk — direct C++ type generation
            declares.append(using TileShape = Shape<M,N,K>)
            declares.append(using Tiler = TiledMMAHelper<...>::TiledMMA)
            declares.append(using K = GemmConfiguration<..., Scheduler::X, ...>)
            declares.append(CUTLASS_CREATE_GEMM_BENCHMARK(K))
    
    # 3. OVERRIDE register_gemm_benchmarks()
    new_text = preamble + declares + register_block
```

**Key pitfall:** Python regex cannot see C++ preprocessor output. `covered()` must check both literal `using` declarations and known expanded tile sets.

---

## Slide 7: Build System — cmake + icpx Pipeline

**Compile flags baked at make time (critical!):**

```bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
```

**Without these: 4 TFLOPS. With them: 155 TFLOPS.**

**Per-batch build flow (86-line `run_seq.sh`):**

```
Step 1: git checkout -- benchmarks_sycl.hpp             # restore clean state
Step 2: python3 gen_mini_hpp.py --manifest batch.txt    # 0.1s
Step 3: python3 gen_main.py batch.txt main.cpp          # 0.1s
Step 4: touch compiler_depend.ts                         # prevent cmake reconfigure
Step 5: sed -i "/^\.DELETE_ON_ERROR/d" build.make        # keep .o on link fail
Step 6: make cutlass_benchmarks_gemm_sycl -j128          # 33s compile
Step 7: icpx -fsycl -device bmg-g31 manual link          # 1s (GB stub)
Step 8: ZE_AFFINITY_MASK=$gpu ./binary --kernel=NAME     # 15s/kernel
```

**Google Benchmark stub** (`cmake/googlebenchmark_stub.cmake`, 9 lines):
```cmake
add_library(benchmark INTERFACE)
add_library(benchmark_main INTERFACE)
add_library(benchmark::benchmark ALIAS benchmark)
```
Headers needed (`benchmark::State`), library NOT needed (`run_direct` bypasses GB).

---

## Slide 8: Timing & GPU Hardware Interface

**`run_direct()` in `benchmark_runner.hpp` (L1159–1189):**

```cpp
GPU_Clock timer;
timer.start();
for (int i = 0; i < 100; ++i) 
    gemm_op.run();
compat::wait();    // SYCL queue synchronization
double avg_sec = timer.seconds() / 100.0;
return (2.0 * m * n * k * 1e-12) / avg_sec;  // TFLOPS
```

**Key design decisions:**
1. **Warmup**: 1 explicit run + 100 iterations before timing (L1176–1178)
2. **Batch submit**: All 100 runs submitted before `compat::wait()` (L1181–1182)
3. **No GB overhead**: Bypasses Google Benchmark's `State` iteration loop
4. **Scheduler splits**: `set_scheduler_splits()` via SFINAE `HasSchedulerSplits<T>` (L126–134)

**GPU affinity:**
```bash
ZE_AFFINITY_MASK=$gpu    # 0,1,2,3 → round-robin across 4 GPUs
```

**Hardware info collected:**
```cpp
hw.sm_count = KernelHardwareInfo::query_device_multiprocessor_count(hw.device_id);
arguments.hw_info = hw_info;   // passed to kernel for tile scheduling
```

---

## Slide 9: Major Bugs and Fixes (Part 1 — Compile & Link)

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 1 | `.DELETE_ON_ERROR` deletes `.o` on every batch | cmake default behavior: if link fails (GB stub), `.o` is deleted | `sed -i '/.DELETE_ON_ERROR/d' build.make` | `510b9815` |
| 2 | Perf flags NOT baked → 4.4 TFLOPS | `SYCL_PROGRAM_COMPILE_OPTIONS` only at runtime, IGC sees it too late | Export before `make`, verified "Compilation from IR" absent in log | `fa3585a5` |
| 3 | `run_direct` ignored `split_k_slices` | `set_scheduler_splits()` was only called in `run()` path | Added call: `set_scheduler_splits(arguments, options.split_k_slices)` | `16ba4c72` |
| 4 | SplitK `defaultArguments()` used `splits=2` | B70 hardware hangs with splits=2 (kernel never returns) | Changed to `{1, StreamKMode::SplitK}` in `gemm_configuration_sycl.hpp` | `7fc16df3` |
| 5 | Remote git sync: `git pull` fails with unstaged files | Screening modifies source → `git pull` aborts silently | `git fetch && git reset --hard origin/main` | SOP documented |

---

## Slide 10: Major Bugs and Fixes (Part 2 — Kernel Generation)

| # | Bug | Root Cause | Fix | Commit |
|---|-----|-----------|-----|--------|
| 6 | `gen_mini` redefinition: DataParallel_512x128x32 | `covered()` checked `using` declarations but tile (512,128,32) missing from `EXPANDED_STREAMK_TILES` | Added K=32 variants to tile set: 64×128×32, 128×128×32, 256×128×32, 512×128×32 | `fd3baad2` |
| 7 | `gen_mini` redefinition: Gemm_128x128x32_SG2x4 | Preamble contained `BMG_DECLARE_GEMM_TILE_SG` calls via `.def` file → 300+ types generated BEFORE batch | Strip ALL BMG_DECLARE_*/BMG_REGISTER_*/BMG_SOURCE_* calls + `.def` includes from preamble | `b73b4a45` |
| 8 | `gen_mini` `KeyError: ST` for `gs` type | `gs` block accidentally included `BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE` (which needs `ST` param not present in `gs`) | Removed spurious line; `gs` only uses `BMG_DECLARE_GEMM_TILE_SG` | `e88751cc` |
| 9 | SG8×8 kernels generated but invalid on B70 | `valid_subgroup_sizes` was `[16, 32, 64]` → allowed sg product 64 | Changed to `[16, 32]` (B70 HW max subgroup = 32) | `f98ed755` |
| 10 | Performance regression: 148→155 TFLOPS after preamble strip | With 300+ extra types, IGC optimization was conservative. Stripping → only batch types → IGC aggressive | Strip preamble before generating batch declares | `b73b4a45` |

---

## Slide 11: B70 Results — Key Findings

**Top 10 kernels (8192×4096×1536, BF16):**

| # | TFLOPS | Kernel | Layout | Tile | SG | ST |
|---|--------|--------|--------|------|-----|-----|
| 1 | **154.9** | Gemm_128×128×32_SG2×4 | **RRR** | 128×128×32 | 2×4 | 2 |
| 2 | 149.6 | Gemm_128×256×32_SG4×4 | **RRR** | 128×256×32 | 4×4 | 2 |
| 3 | 149.2 | Gemm_256×128×32_SG4×4 | **RRR** | 256×128×32 | 4×4 | 2 |
| 4 | 147.5 | Gemm_128×128×32_SG2×4 | RCR | 128×128×32 | 2×4 | 2 |
| 5 | 147.5 | Gemm_256×128×32_SG8×2 | **RRR** | 256×128×32 | 8×2 | 2 |
| 6 | 146.6 | Gemm_256×128×32_SG4×4 | RCR | 256×128×32 | 4×4 | 2 |
| 7 | 145.6 | GemmExhaustive_128×256×32_SG2×8_ST2 | RCR | 128×256×32 | 2×8 | 2 |
| 8 | 144.3 | DataParallel_256×256×32 | RCR | 256×256×32 | 8×4 | 2 |
| 9 | 144.2 | SplitK_256×256×32 | RCR | 256×256×32 | 8×4 | 2 |
| 10 | 142.3 | StreamK_256×256×32 | RCR | 256×256×32 | 8×4 | 2 |

**Scheduler comparison at 256×256×32 SG8×4 (same compute):**
- Gemm: 136.5 → DP: **144.3** (+5.7%) → SplitK: **144.2** (+5.6%) → StreamK: **142.3** (+4.2%)
- All within 1.4% of each other — scheduler choice matters less at large GEMM

**RRR vs RCR (same 128×128×32 SG2×4):** RRR 154.9 > RCR 147.5 (+5.0%)

---

## Slide 12: ALI Performance Comparison

**B70 (BMG G31, this work) vs Previous ALI Generation:**

| Metric | ALI Gen (est.) | B70 (BMG G31) | Improvement |
|--------|---------------|---------------|-------------|
| Peak TFLOPS | ~130 | **154.9** | +19% |
| Mean TFLOPS | ~65 | **76.3** | +17% |
| Kernels screened | ~800 | **1,772** | +122% |
| Batch compile time | 30+ min | **33 sec** | **54×** |
| Full run time | ~48h | **~22h (1 GPU)** | 2.2× |
| Batches | ~400 | **886** | +122% |
| RRR coverage | partial | **full** (684 GE) | — |

**Key enablers for B70 improvement:**
1. **Perf flags baked at compile**: `IGC_VectorAliasBBThreshold=10000` + `-cl-intel-256-GRF-per-thread`
2. **Modern DPAS atom**: `XE_DPAS_TT<8, float, cute::bfloat16_t>` replacing legacy `XE_8x16x16_F32BF16BF16F32_TT`
3. **gen_mini preamble strip**: IGC optimization quality improved 4-6 TFLOPS
4. **RRR layout fully explored**: previously only RCR was exhaustive
5. **StreamK/SplitK/DP**: previously untested, now 81+81 variants each

---

## Slide 13: Incremental Screening & Checkpoint Recovery

**Problem:** 886-batch screening takes ~22 hours. Hardware crash or SSH disconnect loses all progress.

**Solution implemented in `run_seq.sh`:**
- Results written to `$RESULTS_DIR/batch_XXXX_gpuX.csv` immediately after each batch
- No global state — each batch is independent
- Resume by checking which `batch_*.csv` files exist:

```bash
# Resume from last completed batch:
LAST_DONE=$(ls $RESULTS_DIR/*.csv 2>/dev/null | tail -1 | grep -o "batch_[0-9]*")
START_FROM=$((10#${LAST_DONE#batch_} + 1))
for i in $(seq $START_FROM $((BATCHES-1))); do ... done
```

**File-level isolation:**
```
results_final_full/
  batch_0000_gpu0.csv    # 2 kernels, completed
  batch_0001_gpu1.csv    # 2 kernels, completed
  ...
  batch_0885_gpu1.csv    # 1 kernel, completed
```

**Recovery procedure:**
1. `git fetch && git reset --hard origin/main` (clean source)
2. Restore `_deps/` headers
3. Run with `BATCHES=all` — existing CSVs not overwritten (but binary regenerated each time)

---

## Slide 14: Automatic Fault Recovery

**Fault types and auto-handling:**

| Fault | Detection | Recovery | Implemented? |
|-------|-----------|----------|-------------|
| Compile error | `[ ! -s "$OBJ" ]` | Restore source, skip batch | ✅ L49–53 |
| Link error | `[ ! -x "$BIN" ]` | Restore source, skip batch | ✅ L64–68 |
| Kernel timeout | `timeout 120` wrapper | Record `TIMEOUT` in CSV, continue | ✅ L72 |
| GPU hang | SYCL runtime exception | `|| true` capture, record 0 TFLOPS | ✅ L72 |
| Source corruption | `git checkout` at start | Restore from git every batch | ✅ L22 |

**`.DELETE_ON_ERROR` mitigation:**
```bash
# cmake deletes .o when link target fails
# Fix: remove the directive from generated build.make
sed -i '/^\.DELETE_ON_ERROR/d' $BDIR/.../build.make
```

**Per-batch source isolation:**
```bash
cp benchmarks_sycl.hpp /tmp/bak_hpp    # backup before modify
# ... generate + compile + screen ...
cp /tmp/bak_hpp benchmarks_sycl.hpp    # restore after
rm -f benchmarks_sycl.hpp.cache        # force cache invalidation
```

---

## Slide 15: Future Work

**1. Compilation efficiency (current: 33s/batch)**
- **Pre-compiled template cache**: compile `GemmConfiguration` template once, link per-batch
- **Parallel batch compilation**: 4 GPUs → 4 simultaneous `make -j32` instances
- **Incremental make**: only recompile when `benchmarks_sycl.hpp` content hash changes

**2. Multi-GPU scaling**
- Current: GPU 0-3 round-robin, sequential
- Target: 4 parallel workers, each with own cmake build dir
- Script: `PARALLEL=4 BATCHES=all bash tools/run_seq.sh`
- Expected: 22h → 5.5h (4× speedup)

**3. Data analysis & hardware characterization**
- **Occupancy-bucket analysis**: compute TFLOPS vs (tile_area / sg_product)
- **Register pressure sweet spot**: TFLOPS vs GRF usage
- **RRR vs RCR sensitivity**: per-tile win/loss heatmap
- **Scheduler preference by tile size**: when does StreamK beat DP?

**4. Hardware-specific tuning**
- **SplitK splits search**: test splits=2 on updated driver (currently hangs)
- **StreamK chunk/waves**: expose `PersistentTileSchedulerXeStreamKParams` knobs
- **Prefetch toggle**: `GmemTiledCopyA/B` variants
- **GRF mode**: test 128 vs 256 register modes

**5. Production autotune cache**
- Schema: `(M,N,K,dtype,arch) → {tile, sg, stages, scheduler, splits, tflops}`
- Phase 1: exhaustive compute shape → top-10 per occupancy bucket
- Phase 2: scheduler sweep on Phase 1 top-10

---

## Slide 16: Summary & Key Takeaways

**Profiler at a glance:**
- **6,007 lines** Python + **140 lines** HPP generator + **86 lines** shell
- **1,772 BF16 kernels** across 2 layouts × 4 schedulers × 3 stages
- **154.9 TFLOPS** peak on B70 (8192×4096×1536)

**Key innovations:**
1. **Hierarchical catalog**: layered generation with dedup → 6 independent kernel sources
2. **gen_mini preamble strip**: 300→0 extra types → +4-6 TFLOPS from IGC
3. **Baked perf flags**: `SYCL_PROGRAM_COMPILE_OPTIONS` at make time → 38× improvement
4. **Fault-tolerant batch pipeline**: per-batch source isolation, auto-skip on failure
5. **Google Benchmark bypass**: `run_direct()` with 100-iteration timed loop

**Hardware discoveries:**
- RRR layout consistently outperforms RCR (+5% to +14%)
- SG 2×4 beats SG 8×4 on optimal tile (128×128×32)
- SplitK splits=2 hangs on B70 (driver limitation)
- All schedulers within 1.4% on large problem — scheduler matters at skinny shapes

**Code:** github.com/tinafengfun/sycl-tla  
**Data:** docs/screening_final.csv (992 kernels)
