# Intel GPU GEMM Profiler — DEC Review

**Date:** 2026-05-20  
**Author:** SYCL-TLA Profiler Team  
**Status:** `layered_bmg` search running on B70 (Maginfra2 device 7 @ 2500 MHz), 348/3424 batches compiled, 0 errors, 0 timeouts

---

## 1. Architecture Overview

### 1.1 High-Level Pipeline

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         PROFILER WORKFLOW                                 │
│                                                                         │
│  Phase A: Probe                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ compiler  │───▶│ DPAS     │───▶│ anomaly  │───▶│ constraints +    │   │
│  │ profiles  │    │ probe    │    │ detection│    │ blocked rules    │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘   │
│                                                                         │
│  Phase B: Search                                                         │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────────────┐   │
│  │ candidate │───▶│ build    │───▶│ screening│───▶│ confirmation     │   │
│  │ space gen│    │ (compile)│    │ (1 iter) │    │ (N iterations)   │   │
│  └──────────┘    └──────────┘    └──────────┘    └──────────────────┘   │
│                                                                         │
│  Dispatch                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐                           │
│  │ dispatch │───▶│ reference│───▶│ artifact │                           │
│  │ table    │    │ compare  │    │ bundle   │                           │
│  └──────────┘    └──────────┘    └──────────┘                           │
└─────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Core Python Components

| Component | File | Responsibility |
|---|---|---|
| **catalog.py** | `intel_gemm_profiler/catalog.py` | Kernel catalog generation: seed, expanded_bmg, layered_bmg, generator_manifest |
| **candidates.py** | `intel_gemm_profiler/candidates.py` | Candidate space gen, build manifest, shape filtering |
| **constraints.py** | `intel_gemm_profiler/constraints.py` | Allowed values, blocked rules, compile/runtime profiles |
| **workflow.py** | `intel_gemm_profiler/workflow.py` | Orchestration: Phase A probe → Phase B search → dispatch |
| **runner.py** | `intel_gemm_profiler/runner.py` | Subprocess manager: compile, benchmark, timeout, log parsing |
| **selector.py** | `intel_gemm_profiler/selector.py` | Results post-processing: dispatch table, reference comparison |
| **schemas.py** | `intel_gemm_profiler/schemas.py` | Data schemas, epilogue/scheduler metadata inference |
| **source_templates.py** | `intel_gemm_profiler/source_templates.py` | Source scan for observed tile/SG pairs, legality validation |
| **dispatch.py** | `intel_gemm_profiler/dispatch.py` | Runtime dispatch lookup from optimal table |
| **device_target.py** | `intel_gemm_profiler/device_target.py` | Device auto-detection (xpu-smi → CMake target mapping) |
| **hw_specs.py** | `intel_gemm_profiler/hw_specs.py` | Hardware reference specs, efficiency bounds, anomaly detection |

### 1.3 Python ↔ CUTLASS C++ Integration

```
  profiler (Python)                    CUTLASS (C++)
  ┌────────────────┐     ┌───────────────────────────────────┐
  │ candidate      │     │ benchmarks_sycl.hpp                │
  │ space (JSON)   │────▶│  ├─ seed kernel registrations      │
  └────────┬───────┘     │  ├─ expanded_bmg registrations     │
           │             │  ├─ source_template registrations  │
  ┌────────▼───────┐     │  ├─ exhaustive GEMM (layered)      │
  │ build manifest │     │  └─ StreamK/DP/SplitK types        │
  │ (CMake vars +  │     └───────────────────────────────────┘
  │  filter list)  │     ┌───────────────────────────────────┐
  └────────────────┘────▶│ CMakeLists.txt                     │
                         │  ├─ generates filter header        │
                         │  ├─ generates batch-local headers  │
                         │  └─ builds cutlass_benchmarks_gemm │
                         └──────────────┬────────────────────┘
                                        │
                         ┌──────────────▼────────────────────┐
                         │ benchmark_runner.hpp               │
                         │  ├─ BenchmarkRunnerGemm<F>         │
                         │  ├─ filter-gated instantiation     │
                         │  ├─ correctness verification        │
                         │  └─ timing (Google Benchmark)      │
                         └────────────────────────────────────┘
```

### 1.4 Workflow Orchestration (Phase B Detail)

```
  Phase B Search Flow:
  ┌──────────────────────────────────────────────────────────────┐
  │ 1. CANDIDATE SPACE GENERATION                                 │
  │    generate_candidate_space(shapes, constraints, profiles)     │
  │    → catalog lookup → layout/dtype/signature filter           │
  │    → blocked rules → candidate dedup                         │
  │    → output: candidate_space.json (3568 entries)              │
  │                                                               │
  │ 2. BUILD MANIFEST                                              │
  │    build_candidate_build_manifest(candidate_space, bs=1)      │
  │    → unique kernel list (3424 kernels)                        │
  │    → batch partition (3424 batches)                           │
  │    → per-batch filter file + CMake variables                  │
  │    → output: candidate_build_manifest.json                    │
  │                                                               │
  │ 3. PREFLIGHT BUILD (one batch at a time)                      │
  │    execute_candidate_build_preflight_plans(manifest)          │
  │    → for each batch:                                          │
  │        cmake configure → cmake --build → executable           │
  │        log saved → status tracked (built/error/timeout)      │
  │                                                               │
  │ 4. SCREENING (1 iteration, all shapes × candidates)           │
  │    run_entries_with_batch_benchmarks(screening_entries)      │
  │    → routes each (shape, kernel) to correct batch executable  │
  │    → chunked by --benchmark-entry-chunk-size=32               │
  │    → chunk timeout → retry missing entries recursively        │
  │                                                               │
  │ 5. CONFIRMATION (top-k candidates, N iterations)              │
  │    generate_confirmation_entries(screening_results, top-k=8) │
  │    → re-runs top performers with more iterations              │
  │    → median-based dispatch ranking                            │
  │                                                               │
  │ 6. DISPATCH + REPORT                                          │
  │    build_dispatch_table(confirmation_results)                 │
  │    → optimal kernel per shape                                 │
  │    build_reference_comparison(dispatch, reference)           │
  │    → TFLOPS comparison vs reference baseline                  │
  └──────────────────────────────────────────────────────────────┘
```

### 1.5 Phase A: Probe (Pre-Search Validation Pipeline)

Before Phase B begins, Phase A smoke-tests the environment to prevent wasting compute on a broken setup.

```
  Phase A Probe Flow:
  ┌──────────────────────────────────────────────────────────────┐
  │ STEP 1: ENVIRONMENT COLLECTION                                 │
  │   collect_environment_metadata(shell_init, benchmark_exe,     │
  │                                 streamk_example_exe)          │
  │   → checks: oneAPI env vars, xpu-smi, benchmark binary exists │
  │   → output: env_caps = { benchmark_available: true,           │
  │                          streamk_example_available: true }    │
  │                                                               │
  │ STEP 2: DPAS BASELINE PROBE ★                                  │
  │   build_dpas_probe_entry(shapes, static_candidate_space)      │
  │   → selects the smallest candidate + smallest shape            │
  │   → runs 1 benchmark iteration                                 │
  │   → purpose: verify the entire compile→run→verify chain works  │
  │   → output: dpas_baseline_probe = { status, avg_tflops }      │
  │                                                               │
  │ STEP 3: COMPILER PROFILE PROBES                                 │
  │   build_phase_a_probe_entries() → 4 entries:                   │
  │     small_tile:  smallest tile_m candidate  on M≤8 shape       │
  │     medium_tile: mid tile_m candidate         on M=64 shape     │
  │     large_tile:  largest tile candidate       on M≥128 shape   │
  │     splitk:      first SplitK candidate       on narrow shape   │
  │   → each runs with its designated compiler flags variant       │
  │   → output: probe_rows (status + avg_tflops per entry)         │
  │                                                               │
  │ STEP 4: ANOMALY DETECTION                                      │
  │   detect_probe_anomalies(probe_rows, hw_spec)                  │
  │   → compares probe TFLOPS against:                             │
  │       - DPAS baseline (minimum expected performance)           │
  │       - HW reference specs (B60 calibrated data)               │
  │   → flags: "severely_below_spec", "above_expected"            │
  │   → adds auto-blocked rules for failing candidates             │
  │                                                               │
  │ STEP 5: CONSTRAINT UPDATE                                       │
  │   apply_run_probe_constraints(static_constraints, probe_rows)  │
  │   → failed/timed-out candidates → persistent blocked_rules     │
  │   → these rules carry into Phase B, avoiding re-running        │
  │     known-bad candidates in the main search                    │
  └──────────────────────────────────────────────────────────────┘
```

**What is DPAS?**

```
  DPAS = Dot Product Accumulate Systolic
  → Intel Xe GPU's matrix multiply-accumulate hardware unit
  → Analogous to NVIDIA Tensor Cores
  → CUTLASS accesses it via the XE_DPAS_TT MMA atom
  → Every GEMM kernel in our search uses DPAS

  DPAS Probe is NOT a DPAS instruction correctness test.
  It is the simplest possible GEMM run to smoke-test the whole pipeline.
```

| Aspect | Detail |
|---|---|
| **What it picks** | Smallest `tile_m`, smallest `sg_count`, smallest `tile_k` candidate |
| **Shape it uses** | Smallest (K, M) from input shapes |
| **Why the minimum** | Most likely to succeed if anything works; fastest compile & run |
| **What it proves** | oneAPI env → SYCL runtime → benchmark binary → GEMM kernel → DPAS hardware → correctness check — all functional |
| **What it provides** | A TFLOPS value used as **minimum performance reference** for anomaly detection |
| **If it fails** | Phase A reports `dpas_baseline_probe.status = "fail"`; Phase B should not proceed |
| **NVIDIA analogy** | CUDA CUTLASS profiler has a similar warmup / sanity-check kernel, but not explicitly called "DPAS probe" |

**How Probe Results Feed into Phase B:**

```
  probe results → three feedback mechanisms:

  1. blocked_rules: timed-out/failed candidates get persistent rules
     → Phase B skips that candidate class entirely
     Example: {match: {tile_m: 512, sg_m: 8}, reason: "severely_below_spec"}

  2. compiler_profiles: each profile (small/medium/large_tile) gets status
     → failed profiles are excluded from candidate generation

  3. anomaly_report: candidates significantly below baseline are flagged
     → logged for human review; surfaces warnings
```

---

 & Design Highlights

### 2.1 Multi-Level Catalog Architecture (Extensible)

```
  catalog_source options:
  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐   ┌──────────────┐
  │ persisted     │   │ generator     │   │ expanded_bmg  │   │ layered_bmg  │
  │ (level0)      │   │ (manifest)    │   │ (L0 opt-in)   │   │ (L0+L1)      │
  └──────────────┘   └──────────────┘   └──────────────┘   └──────────────┘
      28 kernels         dynamic           330 kernels         3424 kernels
      (default)         (cutlass lib)      (benchmark)        (exhaustive)

  Each level is additive. New levels are opt-in via --kernel-catalog-source.
  Old defaults remain unchanged. layered_bmg strictly contains expanded_bmg.
```

### 2.2 Legality Filter for Regular GEMM

SYCL-TLA GEMM kernels have strong coupling constraints between tile shape and subgroup layout. Not all parameter combinations produce valid kernels. We apply a deterministic legality check:

```
  is_valid_xe2_tile_sg(tile=(m,n,k), sg=(sg_m,sg_n,sg_k), atom=(8,16,16)):

    1. tile_m % (sg_m × 8) == 0     M must be divisible by subgroup M coverage
    2. tile_n % (sg_n × 16) == 0    N must be divisible by subgroup N coverage
    3. tile_k % (1 × 16) == 0       K must be divisible by DPAS atom K
    4. reg_m × reg_n × 16 ≤ 256     Accumulator registers ≤ GRF (256)
       where reg_m = tile_m / sg_m / 8, reg_n = tile_n / sg_n / 16
```

This filter eliminates combinations that would fail at template static_assert time, reducing search space by 67%:

```
  naive Cartesian: 7×7×3×4×3 = 1764 tile/SG combos
  after legality filter: 579 legal combos (−67.2%)
```

### 2.3 Batch Compilation with Kernel Filter Gating

Previous attempts at compiling all kernels in one executable failed (4+ hours with no progress on 330 kernels). The solution:

**Per-batch filter files** → **generated C++ headers** → **conditional template instantiation**

```
  Mechanism per batch:
  ┌────────────────────────────────────────────────────────────────┐
  │ selected_kernel_filter_part000.list                            │
  │   ^BmgGemmBF16BF16FP32_RCR_5$                                  │
  └──────────────────────┬─────────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────────┐
  │ CMake generates: cutlass_benchmark_filter.hpp                  │
  │   #define CUTLASS_BENCHMARK_ENABLE_BmgGemmBF16BF16FP32_RCR_5 1 │
  │                                                                │
  │ CMake generates: cutlass_benchmark_exhaustive_gemm_declare.hpp │
  │   BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(PREFIX, CONFIG, ...)  │
  └──────────────────────┬─────────────────────────────────────────┘
                         │
  ┌──────────────────────▼─────────────────────────────────────────┐
  │ benchmark_runner.hpp: CUTLASS_CREATE_GEMM_BENCHMARK(F)         │
  │   if constexpr (CUTLASS_BENCHMARK_KERNEL_ENABLED(F)):          │
  │       BenchmarkRunnerGemm<F>()  ← heavy template instantiation │
  │   else:                                                         │
  │       state.SkipWithError(...)   ← lightweight, no instantiate  │
  └────────────────────────────────────────────────────────────────┘

  Only the enabled kernel produces device code. All others are no-ops.
```

### 2.4 Timeout with Process-Group Cleanup

```
  Before (broken):
    subprocess.run(command, timeout=1800)
    → kills only bash wrapper
    → orphan cmake/gmake/icpx/clang consume CPU indefinitely

  After (fixed):
    subprocess.Popen(command, start_new_session=True)
    → timeout → os.killpg(pid, SIGTERM) → os.killpg(pid, SIGKILL)
    → entire process group killed atomically
    → no orphan accumulation
```

### 2.5 Chunk Timeout with Automatic Retry

Screening entries are partitioned into config chunks (32 entries per chunk). A single pathological entry can cause the entire chunk to time out, making later entries false negatives:

```
  Automatic retry:
    1. After chunk timeout, parse all successfully completed rows
    2. Compute missing entries (entries with no result row)
    3. Recursively split into smaller _retryXX_YYY chunks
    4. Single-entry retry groups that still emit no rows → marked timeout
    5. Recursion stops when all entries have results or are definitively timed out
```

---

## 2.B End-to-End Walkthrough: One Kernel From Catalog to TFLOPs

Using `BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64` (BF16 input, FP32 output, StreamK scheduler, 512×256×64 tile) as a concrete running example through the entire pipeline.

### Step 1: Catalog — "Which kernels exist?"

```
  # catalog.py — "catalog" = a menu listing all searchable kernels

  benchmark_streamk_tile_candidates(
      "BmgGemmBF16BF16FP32",
      "bf16", "bf16", "f32", "f32",
      tile_shapes=[..., (512, 256, 64), ...]    # ← this tile
  )

  → produces:
  {
      "kernel_name": "BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64",
      "layout": "rcr",
      "tile_m": 512, "tile_n": 256, "tile_k": 64,
      "sg_m": 8, "sg_n": 4,
      "streamk_mode": "streamk",
  }
```

**Catalog source names explained in plain language:**

| Catalog | Meaning | Size |
|---|---|---|
| `persisted` | Pre-saved menu — hand-validated kernels from level0 | 28 entries |
| `generator` | Auto-generated menu — CUTLASS Python generator output | dynamic |
| `expanded_bmg` | Extended menu — seed + tile variants + StreamK family | 330 kernels |
| `layered_bmg` | Full buffet — expanded_bmg + exhaustive regular GEMM enumeration | 3424 kernels |

### Step 2: Candidate — "Which (kernel, shape) pairs to test?"

```
  # candidates.py — "candidate" = one (kernel, shape) pair

  Input shape: M=8192, N=4096, K=1536, dtype=bf16, layout=rcr
  Matched kernel: BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64

  → candidate:
  {
      "candidate_id": "rcr_bf16bf16f32_tm512_tn256_tk64_sg8x4_st2_sk1_streamk",
      "kernel_id":    "BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64",
      "shape_id":     "rcr_bf16_8192_4096_1536",
  }
```

### Step 3: Batch Filter — "Which kernel does this batch compile?"

```
  kernel_id → selected_kernel_list → filter file:
    selected_kernel_filter_part042.list:
      ^BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64$

  → CMake generates C++ header:
    #define CUTLASS_BENCHMARK_ENABLE_BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64 1
```

### Step 4: C++ Template — "How the kernel becomes code"

```cpp
  // Hand-written in benchmarks_sycl.hpp:
  using BmgGemm_BF16BF16FP32_StreamK_TileShape_512_256_64 = Shape<_512, _256, _64>;

  using BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64 =
      Gemm_Bench_BF16BF16FP32_RCR_StreamK<
          Shape<_512, _256, _64>,                    // tile
          Scheduler::GemmStreamK>;                   // StreamK scheduler

  // Macro expands to:
  CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64);

  // → static void Bmg..._func(State& state, ...) {
  //     if constexpr (ENABLED) {
  //       BenchmarkRunnerGemm<Bmg...>().run(state, ...);
  //     }
  //   }
```

### Step 5: Compilation — on CPU, ~2 minutes

```
  icpx -fsycl main.cpp
    → template chain: GemmConfiguration<...> → CollectiveMma → CollectiveEpilogue
    → LLVM IR → SPIR-V device code → IGC → BMG G31 GPU ISA
    → link → cutlass_benchmarks_gemm_sycl (~100MB executable)
```

### Step 6: Screening — on GPU, ~2 milliseconds

```
  ./cutlass_benchmarks_gemm_sycl     --benchmark_filter=BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64     --m=8192 --n=4096 --k=1536

  GPU execution:
    load A(bf16, 8192×1536) + B(bf16, 1536×4096)
    → 512×256 tile, StreamK decomposition
    → each work-item: (512/8/8)×(256/4/16) = 32 DPAS instructions
    → output C(f32, 8192×4096)
    → correctness: BlockCompareRelativelyEqual(epsilon=0.5)

  stdout:
    avg_runtime_ms: 2.134
    avg_tflops: 131.8          ← ★ THE PERFORMANCE NUMBER
```

### Step 7: Parsing + Dispatch — "Which kernel wins?"

```
  parse_benchmark_log() → row: {status:"pass", avg_tflops:131.8}
  → screening: all 3568 candidates, 1 iteration each
  → top-8 per shape → confirmation: 2 iterations, median
  → dispatch_table.json:
  {
    "8192x4096x1536_bf16": {
      "kernel": "BmgGemmBF16BF16FP32_RCR_StreamK_512x256x64",
      "tflops": 131.8
    }
  }
```

### Complete Data Flow

```
  catalog (Python) → candidate → filter file → C++ header → icpx (CPU, 2min)
    → executable → GPU run (2ms) → TFLOPs = 2×M×N×K÷runtime÷1e12
      → parse → dispatch table
```

---



---

## 2.C Timing & TFLOPS Measurement

### Dual Timer Architecture

Each kernel execution uses two independent timers:

```
  for (auto _ : state) {                    // Google Benchmark controls iteration count
      state.PauseTiming();                  // ← pause GB clock
      // ... argument setup, SYCL init ...  //    (NOT timed)
      gemm_op.initialize(arguments);        //    (NOT timed)
      state.ResumeTiming();                 // ← resume GB clock

      GPU_Clock timer;                      // ← independent GPU timer
      timer.start();
      gemm_op.run();                        // ★ ONLY kernel execution is timed
      double ms = timer.milliseconds();

      state.SetIterationTime(ms / 1000);    // report to GB
      update_counters(state, ms);
  }
```

| Timer | Purpose |
|---|---|
| `GPU_Clock` (custom) | Precise kernel execution measurement |
| Google Benchmark `state` | Warmup control, total-iterations gating, statistical reporting |

### Iteration Count & Warmup

The profiler does NOT set explicit `--benchmark_repetitions` or `--benchmark_min_time`.  Google Benchmark defaults apply:

- **Warmup:** GB runs one untimed warmup iteration before any timed iterations.
- **MinTime:** default `0.5` seconds cumulative kernel time per benchmark invocation.
- **Iteration count:** auto-determined by GB.  For a ~2ms kernel: `0.5s / 0.002s ≈ 250` iterations.  For a ~0.8ms kernel: `0.5s / 0.0008s ≈ 625` iterations.
- **Screening:** 1 GB invocation per (candidate, shape) pair.
- **Confirmation:** `--confirm-runs 2` → 2 independent GB invocations, taking the **median** TFLOPS.

### TFLOPS Formula

```cpp
  // benchmark_runner.hpp:389
  double gflop = 2.0 × M × N × K × batch_count / 1e9;

  // finalize_counters():1126-1131
  avg_runtime_ms = (total - best - worst) / (iterations - 2);
  // ↑ removes fastest and slowest iterations, uses trimmed mean

  avg_tflops = gflop / avg_runtime_ms;
  best_tflop = gflop / best_runtime_ms;
```

**Example (8192×4096×1536, StreamK 512×256×64):**

```
  gflop = 2 × 8192 × 4096 × 1536 × 1 / 1e9 = 102.9 GFLOP
  avg_runtime_ms ≈ 0.78 ms
  avg_tflops = 102.9 / 0.78 ≈ 131.8 TFLOPS
```

### Per-Kernel Screening Time

```
  ┌─────────────────────────────────────────────┐
  │ GB warmup:   1 iteration  (untimed)         │
  │ GB timed:    ~600 iterations × 0.8ms        │
  │              stops when cumulative ≥ 0.5s   │
  │ Total:       ~0.5 seconds per benchmark     │
  └─────────────────────────────────────────────┘

  3568 candidates × 0.5s = ~1784s ≈ 30 min (pure GPU time)
  + SYCL launch overhead: ~0.1s per entry = ~357s
  + shape switching: ~0.01s
  Total screening: ~35-40 min for all 3568 candidates
```



### Timer Implementation: GPU_Clock → SYCLTimer → std::chrono::steady_clock

```
  GPU_Clock (GPU_Clock.hpp)
    └─ #if SYCL → SYCLTimer (sycl_timer.hpp)
         ├─ default (CUTLASS_SYCL_PROFILING_ENABLED=OFF):
         │   start(): queue.wait() → std::chrono::steady_clock::now()
         │   stop():  queue.wait() → std::chrono::steady_clock::now()
         │   ms():    std::chrono::duration<float, std::milli>(stop - start)
         │
         └─ profiling (not currently enabled):
             start/stop via SYCL events → GPU hardware timestamps
```

**Measured comparison (B70, 128MB vector add, 200 iterations):**

| Metric | chrono::steady_clock | SYCL event (GPU timestamp) |
|---|---|---|
| Mean | 375.8 μs | 365.6 μs |
| Median | 375.5 μs | 365.9 μs |
| StdDev | 3.4 μs | 2.0 μs |
| Min/Max | 359-413 μs | 350-367 μs |

```
  Ratio:  1.03x (chrono / event)
  Difference: 10.2 μs (SYCL kernel launch + queue.wait overhead)
  For a 800 μs GEMM kernel: overhead ≈ 1.3% — negligible
```

**Verdict:** `std::chrono::steady_clock` with queue synchronization is sufficient
for GEMM profiling.  SYCL events offer ~2× lower jitter but require
`enable_profiling` on the queue and are not needed for our use case.

The test code is at `tools/util/test/sycl_timer_compare.cpp`.

### What Is (and Isn't) Timed

Verification runs **once before** the timing loop — never inside timed iterations:

```cpp
  // ★ ONE-TIME (before loop, NOT timed):
  gemm_op.run();                              // warmup run
  compat::wait();
  bool passed = verify(problem_size, alpha, beta);
  if (!passed) state.SkipWithError("Disposition Failed.");

  // ★ TIMING LOOP (only kernel execution):
  for (auto _ : state) {
      state.PauseTiming();
      // ... argument setup ...              // NOT timed
      gemm_op.initialize(arguments, ...);    // NOT timed
      state.ResumeTiming();

      GPU_Clock timer;                       // ← timing starts
      timer.start();
      gemm_op.run();                         // ★ ONLY THIS
      double ms = timer.milliseconds();      // ← timing ends
  }
```

| Component | Timed? | Note |
|---|---|---|
| SYCL kernel submit | ✅ Yes | `gemm_op.run()` submits to queue |
| GPU execution | ✅ Yes | Xe cores compute GEMM |
| SYCL queue wait | ✅ Yes | Implicit inside `gemm_op.run()` |
| Argument setup | ❌ No | During `PauseTiming()` |
| Memory allocation | ❌ No | Done before timing loop |
| Correctness verify | ❌ No | Runs once before loop starts |
| Launch overhead | ✅ Yes | ~0.05-0.1ms per launch |

For a ~0.8ms kernel: launch overhead ≈ 6-12%. For a ~2ms kernel: ≈ 2-5%.


### Three-Stage Timing Pipeline

| Stage | Iterations | Statistical Method | Purpose |
|---|---|---|---|
| **Screening** | 1 GB invocation, ~0.5s | Trimmed mean (drop best+worst) | Fast ranking of all candidates |
| **Confirmation** | 2 GB invocations, ~0.5s each | Median of 2 runs | Eliminate outlier noise |
| **Dispatch** | confirmation median | Single value | Final optimal kernel selection |


## 3. Search Space Coverage

### 3.1 Why 3000+ Combinations Per Shape

The SYCL-TLA GEMM search space spans **multiple orthogonal compile-time and runtime dimensions**:

| Dimension | Values | Count |
|---|---|---|
| **dtype A/B** | `bf16`, `f16` | 2 |
| **dtype C/D/Acc** | `f32` | 1 |
| **layout** | `rcr` | 1 |
| **tile_m** | `8, 16, 32, 64, 128, 256, 512` | 7 |
| **tile_n** | `32, 64, 96, 128, 192, 256, 512` | 7 |
| **tile_k** | `16, 32, 64` | 3 |
| **sg_m** | `1, 2, 4, 8` | 4 |
| **sg_n** | `2, 4, 8` | 3 |
| **stages** | `1, 2, 3` | 3 |
| **scheduler** | `Gemm, StreamK, DataParallel, SplitK` | 4 |
| **split_k** | `1, 2, 3, 4, 6` (SplitK only) | 4 |
| **grf_mode** | `256` (fixed for BMG perf) | 1 |

Each candidate is a point in this 11-dimensional space.

### 3.2 Complete Candidate Breakdown (one-case input: BF16+F16 RCR)

```
┌──────────────────────────────────────────────────────────────────────┐
│              LAYERED_BMG = 3568 CANDIDATES (2 dtypes)                 │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ L0: expanded_bmg .................. 474 candidates  (13.3%)     │ │
│  │  ├─ seed plain GEMM                8/dtype   (hand-tuned)      │ │
│  │  ├─ expanded SG8x4 GEMM           27/dtype   (source-observed) │ │
│  │  ├─ source template GEMM          58/dtype   (multi-SG scan)   │ │
│  │  ├─ StreamK                       24/dtype   (sk=1)            │ │
│  │  ├─ DataParallel                  24/dtype   (sk=1)            │ │
│  │  └─ SplitK                        96/dtype   (24 tiles × 4 sk) │ │
│  │                                            unique kernels: 330 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐ │
│  │ L1: exhaustive regular GEMM ........ 3094 candidates  (86.7%)   │ │
│  │  579 legal (tile, sg) pairs × 3 stages × 2 dtypes               │ │
│  │  = 3474 kernel entries                                          │ │
│  │  after dedup with L0: 3094 net new candidates                    │ │
│  │                                            unique kernels: 3094 │ │
│  └─────────────────────────────────────────────────────────────────┘ │
│                                                                       │
│  TOTAL: 3568 accepted / 3424 unique benchmark kernels                 │
└──────────────────────────────────────────────────────────────────────┘
```

### 3.3 Legal Tile/Subgroup Distributions

**tile_m × legal combinations:**

| tile_m | count | note |
|--------|-------|------|
| 8 | 45 | smallest tiles, fewest SG options viable |
| 16 | 87 | |
| 32 | 120 | |
| 64 | 138 | peak: widest range of (N, K, SG) options |
| 128 | 102 | |
| 256 | 60 | |
| 512 | 27 | largest tiles, most restrictive SG requirements |

**tile_n × legal combinations:**

| tile_n | count | note |
|--------|-------|------|
| 32 | 57 | |
| 64 | 105 | |
| 96 | 36 | not a power-of-2, fewer SG options |
| 128 | 141 | peak coverage |
| 192 | 60 | |
| 256 | 108 | |
| 512 | 72 | |

**tile_k × legal combinations:**

| tile_k | count | note |
|--------|-------|------|
| 16 | 193 | |
| 32 | 193 | surprisingly uniform across all K values |
| 64 | 193 | |

**subgroup layout × legal combinations:**

| SG layout | count | note |
|-----------|-------|------|
| 1x2, 2x2, 4x2 | 60 each | 2-wide N, more options |
| 1x4, 2x4, 4x4 | 51 each | 4-wide N |
| 1x8, 2x8, 4x8 | 36 each | 8-wide N, fewer legal tiles |
| 8x2 | 57 | largest M coverage |
| 8x4 | 48 | productized default |
| 8x8 | 33 | largest SG, most restrictive |

### 3.4 Coverage vs. Theoretical Maximum

```
  Theoretical Cartesian (no constraints):
    7 × 7 × 3 × 4 × 3 = 1764 tile/SG combos
    × 3 stages = 5292 per dtype
    × 2 dtypes = 10584 candidates

  With legality filter:
    579 tile/SG combos
    × 3 stages = 1737 per dtype
    × 2 dtypes = 3474 candidates (plus non-GEMM modes)

  Current layered_bmg:
    3568 accepted candidates — covers 100% of legal regular GEMM +
    all StreamK/DataParallel/SplitK benchmark-backed combinations

  What is NOT covered (future levels):
    ┌──────────────────────────┬────────────────────────────────────┐
    │ copy atom variants       │ auto-detected; not yet enumerated  │
    │ epilogue variants        │ LinearCombination only             │
    │ bias / activation / quant│ supported by examples, not profiled│
    │ RRR / other layouts      │ catalog has types, not in input    │
    │ TF32 dtype               │ catalog has entries, not in input  │
    │ true F16/F16/F16         │ runtime correctness issues         │
    └──────────────────────────┴────────────────────────────────────┘
```

---

## 3.B Candidate Prefiltering — Reducing Search Space Before Compilation

### Motivation

The layered_bmg space (3568 candidates) requires ~6.5 days for compilation at bs=1. Many configs will underperform for a given shape. The prefilter module applies compile-time heuristics to skip unpromising configs before SYCL device code generation.

### Prefilter Levels

| Level | Candidates Kept | Compile ETA | Description |
|---|---|---|---|
| `none` | 3568 | ~6.5 days | No filtering |
| `light` | 3514 | ~6.4 days | Remove impossible: tile_k > K, tile >> problem, extreme imbalance |
| `medium` | 3406 | ~6.2 days | + remove ILP=1 + large SG + tiny plain GEMM tile |
| `aggressive` | **2032** | **~3.6 days** | + remove ILP≤2 plain GEMM, stage=1 plain GEMM, tiny SG + large tile |

### ILP (Inner Loop Product)

ILP = DPAS instructions per work-item per mainloop iteration. Higher → better latency hiding.

```
  ILP = (tile_m / sg_m / 8) × (tile_n / sg_n / 16)
```

Distribution across 3094 exhaustive GEMM entries:

| ILP | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 |
|---|---|---|---|---|---|---|---|---|
| Count | 182 | 374 | 144 | 554 | 214 | 690 | 214 | 722 |

### Priority Scoring for Screening Order

Candidates sorted by score so high-ILP/StreamK entries run first, building a TFLOPS baseline quickly.

```
  score = ILP×10 + tile_fit_bonus + SG_occupancy + stage_bonus + mode_bonus + tile_k_bonus
```

**Top-5 for 8192×4096×1536:**

| ILP | Tile | SG | st | Mode | Score |
|---|---|---|---|---|---|
| 32 | 512×256×64 | 8×4 | 2 | StreamK | 400 |
| 32 | 512×256×64 | 8×4 | 2 | DataParallel | 400 |
| 32 | 512×256×32 | 8×4 | 2 | StreamK | 390 |
| 32 | 512×256×32 | 8×4 | 2 | DataParallel | 390 |
| 32 | 512×256×64 | 8×4 | 2 | plain GEMM | 370 |

### Implementation

```
  File: test/benchmarks/intel_gemm_profiler/prefilter.py
  CLI:  --prefilter {none,light,medium,aggressive}
  
  Functions:
    compute_ilp(tile_m, tile_n, tile_k, sg_m, sg_n) → int
    prefilter_candidates(candidates, shapes, strategy) → list
    priority_score(candidate, target_shape) → int
    sort_candidates_by_priority(candidates, shapes) → list
```


## 4. Compilation Design — Why So Slow

### 4.1 Root Cause: Single-Translation-Unit SYCL Device Compilation

Intel `icpx` (clang-based) compiles all SYCL device code in a single translation unit. When `main.cpp` contains N × `CUTLASS_CREATE_GEMM_BENCHMARK(F)` macro expansions, each instantiates the full `BenchmarkRunnerGemm<F>` template chain:

```
  Template chain for each kernel:
  BenchmarkRunnerGemm<GemmConfiguration>
    ├─ GemmConfiguration::CollectiveMainloop
    │    ├─ CollectiveMma<MainloopXeL1Staged<Stages>, TileShape, ...>
    │    │    └─ TiledMma<TiledMMAHelper<...>>
    │    │         └─ XE_DPAS_TT<8, float, bfloat16_t> atom
    │    └─ GmemTiledCopyA/B (void → auto-detected)
    ├─ GemmConfiguration::CollectiveEpilogue
    │    ├─ CollectiveEpilogue<IntelXeGeneric, ...>
    │    └─ FusionCallbacks<LinearCombination, ...>
    ├─ GemmKernel = GemmUniversal<..., StreamKScheduler?>
    ├─ Gemm = GemmUniversalAdapter<GemmKernel>
    └─ defaultArguments() / problem-size dispatch / correctness verify
```

Each `GemmConfiguration` has different `TileShape`, `TiledMma`, and `PipelineStages`, preventing template reuse across kernels. The compiler must instantiate and optimize each one separately.

### 4.2 Compilation Behavior Observed

```
  Aggregate compile (330 kernels):
    ┌───────────────────────────────────────────────────┐
    │ icpx -fsycl main.cpp                              │
    │ → clang -cc1: 4+ hours, no executable produced     │
    │ → clang -cc1: infinite loop or memory exhaustion   │
    │ → Not feasible for 330+ kernels                    │
    └───────────────────────────────────────────────────┘

  Batch compile, bs=1 (1 kernel per build):
    ┌───────────────────────────────────────────────────┐
    │ per batch:                                         │
    │   cmake configure: ~2s (first: 18s for deps)       │
    │   icpx -fsycl main.cpp: ~2 min                     │
    │   link: ~30s                                       │
    │   total: ~2.5-3 min                                │
    │                                                     │
    │ 3424 batches × 3 min = 171 hours ≈ 7 days          │
    │ + screening ~1 day = ~8 days total                  │
    └───────────────────────────────────────────────────┘
```

### 4.3 Measured Progress

```
  Run: ali_one_8192_4096_1536_layered_bmg_bs1_20260520_1546
  Machine: B70 Maginfra2 (10.239.11.149), device 7 @ 2500/2500 MHz
  Compile env: oneAPI 2025.3, g++-13, 256 GRF

  After 16 hours:
    Built:     348 / 3424 batches (10.2%)
    Errors:    0
    Timeouts:  0
    Rate:      ~22 batches/hour (~2.7 min/batch)

  Projection:
    Build complete:    ~6 days from start
    Screening:         ~1 day (3568 entries, chunked, ~8s/entry)
    Confirmation:      ~0.5 day (top-8 × 2 iters)
    Total ETA:         ~7-8 days
```

### 4.4 Why NVIDIA CUTLASS Is Faster

| Factor | NVIDIA CUTLASS | Intel SYCL-TLA |
|---|---|---|
| **Kernel generation** | Generator pre-produces full legal manifest | Manual enumeration + source scan + legality filter |
| **Template compilation** | `nvcc` optimized for many similar `GemmConfiguration` instantiations per TU | `icpx`/`clang` slower for Xe kernel template chains |
| **Library pre-compile** | `sm80_gemm_*` static libraries — profiler links them | `cutlass_lib_static` mechanism in early stage for benchmark path |
| **Compiler maturity** | 5+ years of CUTLASS-specific optimization | SYCL device compilation still maturing for this pattern |
| **Build parallelism** | Many `sm80_gemm_*` targets compiled in parallel | Sequential batch builds (memory pressure on shared node) |

### 4.5 Planned Optimization Roadmap

| Priority | Action | Expected Impact |
|---|---|---|
| **P0** | Extend SYCL-TLA generator to auto-produce kernel library | Eliminate manual enum; enable pre-compiled static library approach |
| **P0** | Split device code into multiple TUs | Allow parallel compilation within a build |
| **P1** | Persistent batch build cache | Skip recompile for kernels already built in prior runs |
| **P1** | Parallel batch compilation (2-3 concurrent) | ~2-3× throughput on shared node |
| **P2** | ICX flag tuning (split-dwarf, -fno-sycl-rdc, etc.) | Reduce per-kernel compile time |
| **P2** | Hot-cold path separation in BenchmarkRunnerGemm | Reduce template instantiation depth |

---

---

## 4. Compilation Design — Detailed

### 4.0 Compilation Overview: It's CPU, Not GPU

**SYCL device compilation runs entirely on the CPU.** The GPU is only used for runtime execution (screening/confirmation).

```
  Build pipeline (all on CPU):
  
  Python profiler                      CMake + icpx (CPU)
  ┌────────────────┐                   ┌──────────────────────────┐
  │ candidate list │───filter file───▶ │ cmake configure           │
  │ (JSON)         │                   │   generates headers       │
  └────────────────┘                   │   invokes icpx            │
                                       │                           │
                    icpx pipeline:     │   main.cpp                │
                    ┌──────────────┐   │     ↓ icpx -fsycl        │
                    │ C++ frontend │   │     ↓ clang AST → IR     │
                    │ (clang AST)  │   │     ↓ opt passes         │
                    ├──────────────┤   │     ↓ SPIR-V device code │
                    │ SYCL device  │   │     ↓ AOT compile        │
                    │ compilation  │   │     ↓ .o + link          │
                    ├──────────────┤   │     ↓ executable         │
                    │ host code    │   │                           │
                    │ compilation  │   │  NO GPU INVOLVED         │
                    └──────────────┘   └──────────────────────────┘
```

The GPU is idle during the entire build phase (6+ days).  All compilation work is on Intel Xeon CPU cores on the host machine.  The resulting `cutlass_benchmarks_gemm_sycl` executable is a host binary that will later offload GEMM kernels to the GPU at runtime.

---

### 4.1 Concrete Example: How a GEMM Kernel Is Generated From Templates

**Target kernel:** `BmgGemmBF16BF16FP32_RCR_5` (BF16/BF16/FP32, RCR layout, tile=8×128×32, sg=1×4)

#### Step 1: Template alias definition (in benchmarks_sycl.hpp)

```cpp
// Line 213-228: Generic alias for BF16×BF16→FP32 RCR GEMM
template <typename TileShape, typename Tiler, typename GmemTiledCopyA, typename GmemTiledCopyB, int PipelineStages = 2>
using Gemm_Bench_BF16FP32_RCR = cutlass::gemm::device::GemmConfiguration<
    cutlass::arch::IntelXe,                     // ← GPU architecture
    cutlass::bfloat16_t, cutlass::layout::RowMajor,    // A: bf16, row-major
    cutlass::bfloat16_t, cutlass::layout::ColumnMajor, // B: bf16, column-major
    float, cutlass::layout::RowMajor,                  // C: f32, row-major
    float,                                             // Accumulator: f32
    TileShape, Scheduler::Gemm, Tiler,                 // ← tile + tile-SG mapping
    GmemTiledCopyA, GmemTiledCopyB,                    // ← void = auto-detect
    LinearCombination<float,float,float,float>,        // ← epilogue op
    PipelineStages>;                                    // ← stages count
```

#### Step 2: Concrete instantiation (lines 230-233)

```cpp
// Tile shape:  (M=8, N=128, K=32)
using BmgGemm_BF16FP32_TileShape_8_128_32 = Shape<_8, _128, _32>;

// Subgroup TiledMMA: sg(1,4,1) with XE_DPAS_TT atom
using BmgTile_6 = TiledMMAHelper<
    MMAAtom,                                                  // XE_8x16x16_F32BF16BF16F32_TT
    Layout<Shape<_8,_128,_32>>,                               // tile shape
    Layout<Shape<_1,_4,_1>, Stride<_0,_1,_0>>                // SG layout: 1×4
>::TiledMMA;

// Final kernel type: GemmConfiguration filled with concrete types
using BmgGemmBF16BF16FP32_RCR_5 = Gemm_Bench_BF16FP32_RCR<
    BmgGemm_BF16FP32_TileShape_8_128_32,   // TileShape
    BmgTile_6,                              // Tiler (= TiledMma)
    void,                                   // auto GmemTiledCopyA
    void>;                                  // auto GmemTiledCopyB
```

#### Step 3: Benchmark registration (line 233)

```cpp
CUTLASS_CREATE_GEMM_BENCHMARK(BmgGemmBF16BF16FP32_RCR_5);
```

This macro (defined in `benchmark_runner.hpp:1154-1165`) expands to:

```cpp
static void BmgGemmBF16BF16FP32_RCR_5_func(
    ::benchmark::State& state,
    cutlass::benchmark::GEMMOptions const& options,
    cutlass::KernelHardwareInfo const& hw_info)
{
    if constexpr (CUTLASS_BENCHMARK_KERNEL_ENABLED(BmgGemmBF16BF16FP32_RCR_5)) {
        // ← FILTER GATE: only instantiated if this batch enables this kernel
        auto bench = cutlass::benchmark::BenchmarkRunnerGemm<
            BmgGemmBF16BF16FP32_RCR_5>();   // ← HEAVY template instantiation
        bench.run(state, options, hw_info);
    } else {
        state.SkipWithError("benchmark disabled by build filter");
    }
}
```

#### Step 4: Full template resolution chain (what the compiler sees)

```
BenchmarkRunnerGemm<BmgGemmBF16BF16FP32_RCR_5>
  │
  ├─ GemmConfiguration<IntelXe, bf16(R), bf16(C), f32(R), f32,
  │     Shape<8,128,32>, Gemm, TiledMMA<...>, void, void,
  │     LinearCombination<f32,f32,f32,f32>, 2>
  │   │
  │   ├─ CollectiveMainloop = CollectiveMma<
  │   │     MainloopXeL1Staged<2, KernelXe>,
  │   │     Shape<8,128,32>, bf16, StrideA, bf16, StrideB,
  │   │     TiledMMA<MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>,
  │   │              Layout<Shape<8,128,32>>,
  │   │              Layout<Shape<1,4,1>>>,
  │   │     GmemTiledCopyA, void, void, identity,    // auto-detect copy atoms
  │   │     GmemTiledCopyB, void, void, identity>
  │   │
  │   ├─ CollectiveEpilogue = CollectiveEpilogue<
  │   │     IntelXeGeneric, Shape<8,128,32>, void, f32,
  │   │     StrideC, f32, StrideD,
  │   │     FusionCallbacks<IntelXeGeneric, LinearCombination<...>, ...>,
  │   │     void, void>                              // auto epilogue copy atoms
  │   │
  │   ├─ GemmKernel = GemmUniversal<ProblemShape,
  │   │     CollectiveMainloop, CollectiveEpilogue,
  │   │     StreamKScheduler?>                       // no, Scheduler::Gemm
  │   │
  │   └─ Gemm = GemmUniversalAdapter<GemmKernel>
  │
  ├─ Correctness check: BlockCompareRelativelyEqual (bf16 output)
  ├─ Google Benchmark timing harness
  ├─ Shape sweep logic (M/N/K/batch iteration)
  └─ ~10 additional type alias computations (Stride detection, etc.)
```

**Why each kernel is expensive to compile:**
- The entire chain is resolved at compile time via `if constexpr` and template specialization.
- `CollectiveMma` and `CollectiveEpilogue` are the heaviest — each contains Xe-specific memory layout, register allocation, and loop unrolling logic.
- Different `TileShape` or `TiledMma` → completely different specialization → compiler must re-instantiate everything.
- `icpx` compiles host code + SYCL device code in a single translation unit; the device-code portion (SPIR-V generation plus ahead-of-time compilation to BMG ISA) is the bottleneck.

---

### 4.2 Batch Compilation Architecture

#### Problem
Aggregating all 3424 kernels into one `main.cpp` would cause `clang -cc1` to run indefinitely (4+ hours observed with just 330 kernels, no output produced).

#### Solution: Per-Kernel Batch Build Pipeline

```
  Python profiler            CMake-generated artifacts         icpx
  ┌──────────────┐           ┌──────────────────────────┐     ┌─────────┐
  │ build manifest│           │ selected_kernel_filter     │     │         │
  │ (JSON)        │─────────▶ │   _part000.list            │────▶│ batch000│
  │               │           │   → ^KernelName_000$       │     │ compile │
  │ 3424 kernels  │           │                            │     │ ~2.5min │
  │               │           │ cutlass_benchmark_filter   │     └─────────┘
  │               │           │   _part000.hpp              │
  │               │           │   → #define ENABLE_000 1   │
  └──────────────┘           │                            │     ┌─────────┐
                             │ selected_kernel_filter     │     │ batch001│
                             │   _part001.list            │────▶│ compile │
                             │                            │     │ ~2.5min │
                             │ ... (3424 batch files)     │     └─────────┘
                             └──────────────────────────┘
```

#### Per-Batch CMake Processing (CMakeLists.txt:72-98)

```
  For each batch (e.g., batch_000):
  
  1. Read filter file: selected_kernel_filter_part000.list
     ┌──────────────────────────────────────┐
     │ ^BmgGemmBF16BF16FP32_RCR_5$          │   ← exactly 1 kernel at bs=1
     └──────────────────────────────────────┘
  
  2. CMake foreach extracts: CUTLASS_BENCHMARK_FILTER_KERNEL = "BmgGemmBF16BF16FP32_RCR_5"
  
  3. Generate filter header: cutlass_benchmark_filter.hpp
     ┌───────────────────────────────────────────────────┐
     │ #define CUTLASS_BENCHMARK_ENABLE_                │
     │     BmgGemmBF16BF16FP32_RCR_5  1                 │
     └───────────────────────────────────────────────────┘
  
  4. If CUTLASS_BENCHMARK_EXHAUSTIVE_GEMM is ON and kernel name
     matches the GemmExhaustive pattern:
     
     Generate: cutlass_benchmark_exhaustive_gemm_declare.hpp
     ┌───────────────────────────────────────────────────┐
     │ BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(           │
     │   BmgGemmBF16BF16FP32_RCR,                        │
     │   Gemm_Bench_BF16FP32_RCR,                        │
     │   MMAAtom, 8, 32, 16, 1, 2, 1)                   │
     └───────────────────────────────────────────────────┘
     
     This macro expands to the full using+CUTLASS_CREATE_GEMM_BENCHMARK chain.
  
  5. cmake --build:
     - icpx compiles main.cpp
     - Only the ENABLE'd kernel instantiates BenchmarkRunnerGemm<F>
     - All other 3423 kernels: lightweight SkipWithError path
     - ~2.5 minutes per batch
```

#### Why bs=1 Works But Is Slow

```
  bs=1 compilation time budget (per batch):
    cmake configure:  ~2s    (re-checks ONEAPI compiler, boost, googlebenchmark)
    icpx -fsycl:      ~120s  (parse main.cpp, instantiate 1 kernel, SPIR-V gen)
    link:             ~30s   (link executable + SYCL runtime)
    ─────────────────────
    Total:            ~150s

  3424 batches × 150s = 513,600s = 142.7h = 5.95 days
```

---

### 4.3 Compilation Is on CPU — Detailed Explanation

```
  ┌────────────────────────────────────────────────────────────────┐
  │                      HOST CPU (Intel Xeon)                      │
  │                                                                │
  │  icpx pipeline:                                                 │
  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌───────────────┐  │
  │  │ clang    │→│ LLVM IR  │→│ opt       │→│ SPIR-V        │  │
  │  │ frontend │  │ gen      │  │ passes   │  │ device binary │  │
  │  │ (parse   │  │ (C++ →   │  │ (inline, │  │               │  │
  │  │  C++     │  │  LLVM)   │  │  unroll, │  │ ↓ AOT compile │  │
  │  │  +SYCL)  │  │          │  │  vector) │  │               │  │
  │  └──────────┘  └──────────┘  └──────────┘  └───────┬───────┘  │
  │                                                    │          │
  │  ┌─────────────────────────────────────────────────┘          │
  │  │  IGC (Intel Graphics Compiler) — runs on CPU                │
  │  │  ┌──────────┐  ┌──────────┐  ┌──────────────────────┐     │
  │  │  │ SPIR-V   │→│ IGC IR   │→│ BMG G31 ISA (binary)  │     │
  │  │  │ reader   │  │ optimize │  │ GPU machine code      │     │
  │  │  └──────────┘  └──────────┘  └──────────────────────┘     │
  │  └────────────────────────────────────────────────────────────┘  │
  │                                                                │
  │  Host code compile:  clang → x86_64 .o → link → executable    │
  └────────────────────────────────────────────────────────────────┘
  
  ┌────────────────────────────────────────────────────────────────┐
  │                    B70 GPU (idle during build)                  │
  │                                                                │
  │  Only used at RUNTIME (screening/confirmation):                 │
  │    executable loads → SYCL runtime → pushes kernel binary      │
  │    → GPU Xe cores execute GEMM                                 │
  └────────────────────────────────────────────────────────────────┘
```

**Key insight:** The GPU is completely idle for 6+ days during compilation. All 3424 batch compiles run sequentially on CPU cores. This means CPU optimizations (more cores, better compiler flags, caching) directly reduce build time.

---

### 4.4 Acceleration Strategies

#### P0: Parallel Batch Compilation (immediate, low risk)

Build N batches concurrently instead of sequentially:

```
  Current:  3424 batches × 150s sequential = 6 days
  N=2:      1712 slots × 150s = 3 days      (2 concurrent cmake builds)
  N=4:      856 slots  × 150s = 1.5 days    (4 concurrent, needs 4× memory)
```

Implementation: change `execute_candidate_build_preflight_plans` in `workflow.py` to use `concurrent.futures.ThreadPoolExecutor` or `subprocess.Popen` with a semaphore.

**Risk:** Memory pressure. Each `icpx` build can use 4-8GB RAM. On a shared node with 256GB, N=4 is safe.

#### P0: Skip cmake Reconfiguration

Each batch runs `cmake` from scratch (2s overhead per 150s = 1.3%). For 3424 batches, that's ~6800s (~2h). After the first batch, subsequent batches only need to regenerate filter headers — not full reconfiguration.

```
  Strategy: pre-configure once with an empty filter, then for each batch:
    cp filter_file → build_dir
    cmake --build (skip configure)
  Savings: ~2h per full run
```

#### P1: Persistent Object Cache

`icpx` already caches device binaries via SYCL cache, but C++ template instantiation is re-done every batch:

```
  Strategy: ccache or sccache wrapper around icpx
  Since each batch compiles a DIFFERENT kernel, cache hit rate is low
  But across runs (re-build same kernel), cache eliminates recompile
  Savings: 100% for previously-built kernels on re-run
```

#### P1: Hot-Cold Separation in Benchmark Runner

The 3423 "disabled" kernels still go through the `if constexpr` branch in `CUTLASS_CREATE_GEMM_BENCHMARK`. While they skip `BenchmarkRunnerGemm<F>` instantiation, they still require the compiler to parse and verify the branch. Separating enabled vs disabled kernels into different C++ files would eliminate this:

```
  Strategy: split main.cpp into:
    main_enabled.cpp  — #include only the enabled kernel's declaration header
    main_disabled.cpp — empty stubs, no template code
  The batch filter CMake logic already generates the needed headers
  Savings: ~20-30% per batch
```

#### P1: Library Pre-Compilation (NVIDIA Approach)

Follow NVIDIA's model: pre-compile kernels into a static library, then link:

```
  Strategy:
    1. Add CMake target: cutlass_benchmarks_gemm_kernels (STATIC library)
    2. Each kernel compiles into its own .o in this library
    3. benchmark executable links the library → no per-batch build needed
  Benefit: parallels NVIDIA CUTLASS profiler approach
  Challenge: SYCL static library support + cross-TU device code linking maturity
```

#### P2: IGC/ICX Flag Tuning

Current flags:
```
  IGC_ExtraOCLOptions: -cl-intel-256-GRF-per-thread
  IGC_VectorAliasBBThreshold: 10000
  SYCL_PROGRAM_COMPILE_OPTIONS: -ze-opt-large-register-file -gline-tables-only
```

Potential additions:
```
  -g0 or -gline-tables-only: already in use (minimal debug info)
  -fno-sycl-rdc: disable relocatable device code (not needed for single-TU)
  -fno-sycl-early-optimizations: may help parse speed (trade: worse ISA)
  IGC_ShaderDumpEnable=0: skip shader dump to disk
```

#### Summary: Potential Speedup

| Optimization | Current | After | Speedup |
|---|---|---|---|
| Baseline (bs=1 sequential) | 6.0 days | — | 1× |
| + Parallel N=4 | 6.0d | 1.5d | 4× |
| + Skip cmake reconfig | 1.5d | 1.4d | 1.07× |
| + Hot-cold separation | 1.4d | 1.1d | 1.3× |
| + ICX flag tuning | 1.1d | 1.0d | 1.1× |
| **Total potential** | **6.0d** | **~1.0d** | **~6×** |




### 4.5 Why `--parallel` Doesn't Cause Oversubscription

Each batch build uses `cmake --build --parallel` (equivalent to `make -j` with no job limit), and 32 batches run concurrently.  At first glance this appears to create 32 × 256 = 8192 concurrent jobs.  In practice it does not:

```
  Per-batch Make targets for cutlass_benchmarks_gemm_sycl:
  ┌────────────────────────────────────────────────────────┐
  │ main.cpp (SYCL device compile)     ← 1 heavy job, ~2min │
  │ googlebenchmark objects            ← 2-3 light jobs     │
  │   (cached after first build)                            │
  │ googletest objects                 ← 2-3 light jobs     │
  │   (cached after first build)                            │
  │ library objects (cutlass_lib_static) ← 1-2 light jobs   │
  │ link step                          ← 1 serial job       │
  └────────────────────────────────────────────────────────┘

  Actual parallelism per batch: ~3-5 concurrent jobs
  32 batches × ~4 jobs ≈ 128 concurrent processes

  Measured CPU load: 95 / 256 vCPUs = 37%  ← matches the model
```

**Conclusion:** `--parallel` is not the bottleneck and does not cause oversubscription.  The true bottleneck is the ~2-minute `icpx` SYCL device compilation of `main.cpp` per batch.  The 32-batch concurrency is already well-matched to the available CPU resources (37-50% utilization).




### 4.6 What Each Batch Compilation Includes

Each batch compile (`cmake --build`) produces one statically-linked executable containing the full SYCL-TLA GEMM implementation for exactly one kernel configuration.  SYCL-TLA is **header-only C++ templates** — there is no pre-compiled GEMM library involved in the benchmark path:

```
  main.cpp compilation chain (each batch):
  ┌────────────────────────────────────────────────────────────┐
  │ main.cpp                                                    │
  │  └─ #include "benchmarks_sycl.hpp"                         │
  │       ├─ declares: Gemm_Bench_BF16FP32_RCR<Tile,SG,...>    │
  │       ├─ instantiates: CUTLASS_CREATE_GEMM_BENCHMARK(F)    │
  │       │    └─ BenchmarkRunnerGemm<GemmConfiguration>        │
  │       │         └─ GemmConfiguration<                       │
  │       │              IntelXe, bf16(row), bf16(col),         │
  │       │              f32(row), f32,                         │
  │       │              Shape<M,N,K>, Scheduler, TiledMma,    │
  │       │              void, void, Epilogue, Stages>          │
  │       │              │                                      │
  │       │              ├─ CollectiveMma<                      │
  │       │              │    MainloopXeL1Staged<Stages>,      │
  │       │              │    TileShape, ElementA/B,            │
  │       │              │    TiledMma<XE_DPAS_TT<8,...>,       │
  │       │              │           Layout<M,N,K>,             │
  │       │              │           Layout<sgM,sgN,1>>>>       │
  │       │              │    ← 2D block IO + DPAS MMA          │
  │       │              │                                      │
  │       │              ├─ CollectiveEpilogue<                 │
  │       │              │    IntelXeGeneric,                   │
  │       │              │    LinearCombination<f32,...>>       │
  │       │              │    ← epilogue fusion + store         │
  │       │              │                                      │
  │       │              ├─ GemmKernel = GemmUniversal<...>     │
  │       │              │    ← Xe kernel dispatch              │
  │       │              │                                      │
  │       │              └─ Gemm = GemmUniversalAdapter<...>    │
  │       │                   ← SYCL host-device bridge         │
  │       │                                                     │
  │       ├─ correctness verify: BlockCompareRelativelyEqual    │
  │       └─ timing: Google Benchmark harness                   │
  │                                                            │
  │  ALSO: googlebenchmark + googletest static libraries        │
  │        (compiled once, cached across batches)               │
  └────────────────────────────────────────────────────────────┘

  Compilation output per batch:
  ┌────────────────────────────────────────────────────────────┐
  │ x86_64 host code:  benchmark main(), argument parsing       │
  │ SPIR-V device code: GEMM kernel (icpx → LLVM → SPIR-V)     │
  │ BMG G31 ISA:        ahead-of-time compiled GPU binary      │
  │                      (IGC: SPIR-V → BMG machine code)       │
  └────────────────────────────────────────────────────────────┘
```

**Key point:** The 2-minute `icpx` time is dominated by instantiating the `GemmConfiguration` template chain and compiling the resulting SPIR-V device code into BMG GPU ISA.  There is no separate GEMM library — it is all header-only templates instantiated in `main.cpp`.




### 4.7 Hand-Written Benchmarks vs Generated Library — Two Architecture Paths

SYCL-TLA offers two distinct mechanisms for registering GEMM kernels with the profiler:

```
  Path A: Generated Library (cutlass_lib_static)     Path B: Hand-Written Benchmarks
  ┌──────────────────────────────────────┐           ┌──────────────────────────────────┐
  │ Python: cutlass_library/generator    │           │ C++: benchmarks_sycl.hpp          │
  │ → IntelXe manifest (JSON)            │           │ → template aliases + macros       │
  │ → auto-generate .cpp files           │           │ → CUTLASS_CREATE_GEMM_BENCHMARK() │
  │ → compile → cutlass_lib_static.a     │           │ → compiled directly into main.cpp │
  │                                      │           │                                   │
  │ benchmark runner:                     │           │ benchmark runner:                  │
  │ → links cutlass_lib_static.a         │           │ → no external library needed       │
  │ → registers via manifest iterator    │           │ → registers via CUTLASS_BENCHMARK  │
  └──────────────────────────────────────┘           └──────────────────────────────────┘
```

#### Comparison

| Aspect | Path A (library) | Path B (hand-written) |
|---|---|---|
| **Kernel source** | Python generator + manifest | C++ templates + macros in benchmarks_sycl.hpp |
| **Compile model** | Static library (`.a`), link once | Each kernel compiled into the benchmark executable TU |
| **Add new kernel** | Add entry to Python manifest/catalog | Write `using` + `CUTLASS_CREATE_GEMM_BENCHMARK()` in C++ |
| **Add new dtype/layout** | Generator handles combinatorially | Write new template alias + registration block |
| **Compile parallelism** | Library targets compiled in parallel by CMake | Single TU per batch (当前的 bs=1) |
| **Binary size** | Multiple `.o` in `.a`, linker deduplicates | One executable per batch (当前的, ~100MB each) |
| **Search space** | Determined by generator manifest | Determined by which registrations are `#ifdef`'d in |
| **Filter gating** | KERNEL_FILTER_FILE controls manifest | KERNEL_FILTER_FILE + per-batch `#define ENABLE_*` |
| **Maturity** | NVIDIA CUTLASS default approach | Our current approach |
| **Screening** | Single executable runs all kernels | Per-batch executables (当前的) |

#### Pros & Cons

**Path A (library) — Pros:**
- **Single executable**: one build, one binary, all kernels available → much simpler screening
- **Build parallelism**: CMake compiles library objects in parallel naturally
- **Ecosystem maturity**: same as NVIDIA CUTLASS profiler; all tooling expects this model
- **Incremental build**: changing one kernel only recompiles its `.o`, not everything
- **Binary reuse**: same library can be used across different shape inputs

**Path A (library) — Cons:**
- **Not yet validated in our profiler pipeline**: the `cutlass_lib_static` path exists in the CMake build but has not been tested end-to-end for BMG GEMM benchmark screening; testing is needed to confirm it works correctly with SYCL device code and batch filtering
- **Generator complexity**: must maintain Python generator → manifest → C++ codegen pipeline
- **Initial build overhead**: first build compiles ALL kernels into the static library (could be even slower than batch approach for large catalogs)
- **Overhead for small searches**: compiling 3000 kernels into a library for a 2-shape search is wasteful

**Path B (hand-written) — Pros:**
- **Direct control**: every kernel registration is explicit C++ code, easy to audit
- **No generator dependency**: no Python codegen chain, simpler toolchain
- **Per-batch isolation**: a broken kernel only fails its batch, not the whole library
- **Filter gating is natural**: `#ifdef` on a per-batch basis works cleanly with `if constexpr`

**Path B (hand-written) — Cons:**
- **Per-batch executable**: 3424 separate binaries = disk space (3424 × ~100MB ≈ 340GB)
- **Screening complexity**: must route each (shape, kernel) to the correct per-batch exe
- **Single-TU bottleneck**: one `main.cpp` per batch, no intra-batch parallelism
- **Manual enumeration**: every new kernel = new C++ code in benchmarks_sycl.hpp
- **Code bloat**: benchmarks_sycl.hpp + bmg_gemm_source_tile_sg.def already 800+ lines

#### Recommended Hybrid Architecture (Future)

```
  Layer 1: Seed + Expanded kernels  (Path B, current)
  ┌─────────────────────────────────────────────┐
  │ benchmarks_sycl.hpp                          │
  │ → hand-written, well-tested, always included │
  │ → 165 kernels/dtype (330 total for BF16+F16) │
  └─────────────────────────────────────────────┘

  Layer 2: Exhaustive regular GEMM  (Path B, current)
  ┌─────────────────────────────────────────────┐
  │ bmg_gemm_source_tile_sg.def + CMake headers  │
  │ → legality-filtered Cartesian enumeration    │
  │ → generated at CMake time from filter list   │
  │ → 1547 kernels/dtype                         │
  └─────────────────────────────────────────────┘

  Layer 3: Generator library  (Path A, future)
  ┌─────────────────────────────────────────────┐
  │ CUTLASS Python generator                     │
  │ → manifest of ALL legal IntelXe GEMM kernels │
  │ → compiled into cutlass_lib_static.a once    │
  │ → profiler links and filters at runtime      │
  │ → replaces manual enumeration for L2+L3      │
  └─────────────────────────────────────────────┘
```

**Migration path:** Once SYCL static library support is mature, Layer 2 can be migrated
from hand-written `BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE` macros to generator-produced
`cutlass_lib_static`.  Layer 1 (seed + expanded) stays as hand-written for the
well-validated "anchor" kernels.

**Key design principle:** The profiler Python layer (`catalog.py`) already abstracts the
source.  Switching from `source="exhaustive_regular_gemm_catalog"` to `source="generator_manifest"`
requires only changing the catalog generation function — the rest of the pipeline
(manifest, batch builds, screening, dispatch) is identical.


## 5. Deployment Summary

| Parameter | Value |
|---|---|
| **Machine** | Maginfra2 (10.239.11.149) |
| **GPU** | B70 device 7 @ 2500/2500 MHz (locked) |
| **oneAPI** | 2025.3 compiler + MKL + TBB + UMF |
| **Build config** | `g++-13`, `DPCPP_SYCL_TARGET=intel_gpu_bmg_g31`, `IGC_ExtraOCLOptions=-cl-intel-256-GRF-per-thread`, `IGC_VectorAliasBBThreshold=10000`, `SYCL_PROGRAM_COMPILE_OPTIONS=-ze-opt-large-register-file -gline-tables-only` |
| **Run ID** | `ali_one_8192_4096_1536_layered_bmg_bs1_20260520_1546` |
| **Input** | `ali_one_8192_4096_1536_bf16_f16.json` (2 shapes, BF16+F16 RCR) |
| **Catalog** | `layered_bmg` (expanded_bmg + exhaustive regular GEMM) |
| **Search space** | 3568 candidates / 3424 selected kernels |
| **Batches** | 3424 (bs=1) |
| **Progress** | 348/3424 built, 0 errors, 0 timeouts |
| **ETA** | ~7-8 days total |
