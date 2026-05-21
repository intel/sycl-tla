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
