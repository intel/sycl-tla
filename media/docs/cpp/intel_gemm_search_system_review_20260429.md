# Intel GEMM Search System Review - 2026-04-29

## Scope

This note summarizes three things for manual review:

1. what is already implemented,
2. what is still missing relative to the search-system design,
3. the remaining implementation plan after reviewing `example/0017_search_engin.md` and related design notes.

The target remains the Intel/BMG non-legacy GEMM profiler path, not a full CUDA profiler reimplementation.

## Implemented today and previously completed

### 0. Latest calibration and validation update

The profiler now includes a calibrated hardware reference model and spec-driven probe anomaly detection.

New repo assets:

- `test/benchmarks/intel_gemm_hw_reference_specs.json`
- `test/benchmarks/intel_gemm_profiler/hw_specs.py`

Latest validation data:

| Scope | Result |
|---|---|
| local `python3 test/python/cutlass/test_intel_gemm_profiler.py` | `29 / 29` passed |
| local wrapper smoke (`run_phase_a.py`, `run_phase_b.py`, `workflow.py`) | passed |
| remote B60 `python3 test/python/cutlass/test_intel_gemm_profiler.py` | passed |
| remote B60 Phase A BF16 run | passed |

Latest B60 Phase A BF16 evidence:

- resolved `hw_reference_spec_id = bmg_g21`
- corrected `max_slm_kb = 64`
- DPAS baseline probe: `2.97414 TFLOPS`
- compiler probe:
  - small tile: `2.93268 TFLOPS`
  - medium tile: `18.9148 TFLOPS`
  - large tile: `7.92284 TFLOPS`
- anomaly detection auto-blocked:
  - `rcr_bf16bf16f32_tm256_tn256_tk32_sg8x4_st2_sk1`
  - reason: `large_tile_slower_than_rcr_bf16bf16f32_tm16_tn64_tk32_sg2x4_st2_sk1`
- anomaly detection also auto-blocked:
  - `rcr_bf16bf16f32_tm8_tn64_tk32_sg1x4_st2_sk2`
  - reason: `severely_below_spec`

Current compiler-flag conclusion from the B60 benchmark binary:

- an explicit A/B run of the same medium-tile config with and without `IGC_VISAOptions=-perfmodel`
  produced `18.7488 TFLOPS` vs `18.8388 TFLOPS`
- the delta is only about `0.5%`
- therefore the current change does **not** promote runtime env flags into a new per-variant compiler-selection system yet
- the immediate correction is the calibrated spec model and anomaly-driven pruning
### 1. GEMM profiler MVP workflow exists

The current runner already implements the main offline tuning skeleton:

```text
candidate -> verify -> profile -> select_best -> dispatch
```

Delivered pieces:

- candidate generation for non-legacy GEMM MVP,
- benchmark subprocess execution,
- result parsing,
- screening + confirmation stages,
- dispatch table emission.

Main file:

- `test/benchmarks/intel_gemm_profiler.py`

### 2. Search is non-legacy and benchmark-backed

The active best-config search path now stays on:

- `benchmarks/gemm/cutlass_benchmarks_gemm_sycl`

It no longer depends on `benchmarks/gemm/legacy`.

### 3. BF16 and F16 RCR coverage is working

Non-legacy BF16 and F16 search paths are in place and validated on B60.

Relevant files:

- `benchmarks/gemm/benchmarks_sycl.hpp`
- `test/benchmarks/intel_gemm_profiler.py`

### 4. Split-K is available only as probe / feature validation

Current Split-K status:

- BF16 Split-K works on B60,
- F16 Split-K works on B60,
- the backend is `examples/03_bmg_gemm_streamk`,
- it is intentionally **not** part of best-kernel search.

That boundary is deliberate: examples are acceptable for capability probing and feature validation, but not for final optimal-configuration benchmarking.

### 5. Phase A probe framework exists

The runner now supports:

- `--probe-mode=off`
- `--probe-mode=static`
- `--probe-mode=run`
- `--probe-mode=auto`

Produced artifacts now include:

- `verified_hw_caps.json`
- `safe_search_constraints.json`
- `compiler_profiles.json`

### 6. Phase A / Phase B entrypoints and artifact names are aligned

The repo now contains lightweight entry wrappers:

- `test/benchmarks/run_phase_a.py`
- `test/benchmarks/run_phase_b.py`

And the runner emits design-aligned artifacts:

- `verified_hw_caps.json`
- `bmg_safe_candidates.json`
- `optimal_dispatch_table.json`
- `phase_a_summary.json`
- `phase_b_summary.json`

### 7. Boundary correction has been applied

After re-reading the design documents, the implementation now follows these rules:

1. automatic best-config search is a primary goal,
2. examples must not be used as the final best-config benchmark path,
3. search stays benchmark-backed,
4. Phase A defines boundaries and Phase B searches only inside validated space.

## What is still missing

The current system is still a GEMM MVP, not the full search system described in the planning docs.

### 1. Candidate generation is still seed-list driven

Current behavior:

- Phase B candidates still originate from a hand-maintained Python seed table.

Missing:

- a structured kernel catalog,
- instantiation levels,
- catalog-driven candidate generation,
- build-manifest driven expansion.

### 2. Search-space modeling is still incomplete

The docs, especially `0017_search_engin.md`, imply a two-layer search model:

```text
compile-time variants × runtime sweep
```

The current MVP does not yet fully model that split.

Still missing:

- compile-time variant schema,
- runtime sweep schema,
- explicit instantiation levels,
- richer runtime knobs beyond the current MVP set.

### 3. BenchmarkCodegen / build manifest is only partial

The current runner can emit configs and manifests for the existing benchmark flow, but it is not yet a true:

```text
kernel catalog -> build manifest -> generated benchmark set
```

pipeline.

Still missing:

- per-level candidate manifests,
- compile-time variant identifiers,
- a cleaner benchmark codegen boundary,
- a scalable AOT expansion path.

### 4. Split-K is not benchmark-backed for best-config search

Current state is intentionally limited:

- Split-K participates in Phase A probing,
- Split-K does **not** participate in Phase B best-kernel selection.

Before that can change, the project still needs:

1. a non-legacy benchmark-backed Split-K runner,
2. catalog integration,
3. Phase A validation of the new runner,
4. safe-candidate admission rules.

### 5. Scope is still GEMM-only

Not yet implemented:

- BlockScaledGemm search,
- GroupedGemm search,
- Attention-specific search,
- online runtime integration of the dispatch table.

## Remaining search-system plan

This is the remaining implementation order after reconciling the current code with `example/0017_search_engin.md`.

### Step 0 - Define search runtime schema

Lock the search data model first:

- compile-time variant fields,
- runtime sweep fields,
- pruning inputs,
- manifest fields,
- result evidence fields.

Without this, later catalog and build-manifest work will keep changing shape.

### Step 1 - Define kernel catalog schema

Introduce a structured Intel GEMM kernel catalog with:

- `instantiation_level`,
- dtype / layout,
- tile shape,
- subgroup layout,
- reg-blocking / ILP class,
- GRF mode,
- runner type,
- benchmark target,
- allowed runtime sweep fields.

### Step 2 - Implement catalog-driven candidate generation

Replace the current hand-maintained seed list with:

```text
kernel catalog
  -> Phase A constraints
  -> pruning rules
  -> bmg_safe_candidates.json
```

### Step 3 - Implement build manifest generation

Add the first real BenchmarkCodegen layer:

- candidate manifest,
- per-level config fragments,
- explicit compile-time vs runtime split,
- artifact structure that can scale to larger AOT search spaces.

### Step 4 - Wire SearchExecutor to the catalog flow

Refactor the search executor so it consumes:

```text
catalog -> safe candidates -> build manifest -> search
```

instead of direct seed tables.

### Step 5 - Plan benchmark-backed Split-K runner

Keep Split-K out of final search until a proper non-legacy benchmark-backed runner exists.

## Current pending backlog

- `define-search-runtime-schema`
- `define-kernel-catalog-schema`
- `implement-catalog-candidates`
- `implement-build-manifest`
- `wire-search-to-catalog`
- `plan-benchmark-splitk-runner`
- `publish-review-status`

## Review focus

The main review questions are:

1. Is the current boundary correct that **examples remain probe-only**?
2. Is the next implementation priority correct that **catalog/schema/build-manifest** comes before more search breadth?
3. Should Split-K stay out of best-config search until a benchmark-backed runner exists?
4. Is the remaining system still aligned with the intended non-legacy, Intel-specific search architecture?
