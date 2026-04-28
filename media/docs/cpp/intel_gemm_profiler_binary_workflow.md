# Intel GEMM Profiler Binary Workflow

## Goal

This document defines the Phase 1 execution workflow for the Intel GEMM profiler MVP.

The key decision is unchanged:

> do not recreate CUTLASS `Manifest + Operation`; use generated benchmark binaries plus subprocess execution.

The workflow must stay compatible with the current repository reality:

- benchmark binaries already exist
- benchmark binaries already consume `--config_file=...`
- benchmark logs already expose `avg_runtime_ms`, `avg_tflops`, and `avg_throughput`

## Pipeline overview

The Phase 1 flow is:

```text
gemm_target_shapes.json
safe_search_constraints.json
compiler_profiles.json
        |
        v
TileSpaceGenerator
        |
        v
gemm_candidate_space.json
        |
        v
BenchmarkCodegen
        |
        +--> generated benchmark source / registration
        +--> generated config .in files
        +--> build manifest
        |
        v
BatchBuilder
        |
        v
built benchmark binaries
        |
        v
BenchmarkRunner (subprocess)
        |
        +--> raw logs
        +--> gemm_profile_results.csv
        |
        v
BestSelector
        |
        v
gemm_dispatch_table.json
```

## Why subprocess is the right boundary

The Intel benchmark path is compile-time instantiated.

That means Phase 1 should treat each generated benchmark binary as an executable artifact rather than trying to model kernels as runtime-polymorphic objects.

Subprocess execution gives four benefits:

1. hard isolation between candidate variants
2. clean reuse of existing benchmark entrypoints
3. simpler logging and failure capture
4. no need to embed profiler orchestration inside the benchmark binary itself

## Module responsibilities

### `TileSpaceGenerator`

Input:

- `safe_search_constraints.json`
- `compiler_profiles.json`
- optional target-shape metadata

Output:

- `gemm_candidate_space.json`

Responsibility:

- enumerate legal candidates
- prune with search constraints
- classify candidates
- bind `compiler_profile_id`

### `BenchmarkCodegen`

Input:

- `gemm_candidate_space.json`

Output:

- generated registration source
- generated build fragments if needed
- generated `.in` config files
- `build_manifest.json`

Responsibility:

- map `candidate_id` to a concrete benchmark symbol
- emit source or registration lists for those candidates
- generate one or more `.in` files that pair candidate kernels with target shapes

### `BatchBuilder`

Input:

- generated source
- selected compiler profiles
- workspace root

Output:

- benchmark executables
- build logs
- updated `build_manifest.json`

Responsibility:

- configure and build candidate batches
- persist build success/failure per candidate
- keep incremental rebuild support

### `BenchmarkRunner`

Input:

- `build_manifest.json`
- generated `.in` files
- runtime environment

Output:

- raw logs
- `gemm_profile_results.csv`

Responsibility:

- execute benchmark subprocesses
- capture stdout/stderr
- map failures into structured CSV rows

### `ResultParser`

Input:

- raw log files

Output:

- normalized `gemm_profile_results.csv`

Responsibility:

- parse current benchmark text format
- normalize result fields into schema names
- attach `shape_id`, `candidate_id`, and `compiler_profile_id`

### `BestSelector`

Input:

- `gemm_profile_results.csv`

Output:

- `gemm_dispatch_table.json`

Responsibility:

- run top-k screening
- run confirmation rounds
- produce final exact-shape mapping

## Workspace layout

The profiler workflow should use a dedicated workspace under a generated run directory.

Recommended structure:

```text
<workspace_root>/
  inputs/
    gemm_target_shapes.json
    safe_search_constraints.json
    compiler_profiles.json
  generated/
    candidates/
    configs/
    sources/
    manifests/
  build/
    <batch_id>/
  logs/
    build/
    run/
  reports/
    gemm_candidate_space.json
    gemm_profile_results.csv
    gemm_dispatch_table.json
```

This layout keeps generated artifacts isolated from repository-tracked source.

## Codegen strategy

### Phase 1 recommendation

Generate benchmark registration code from candidates, then compile in batches.

Practical form:

1. emit one generated header or source fragment containing benchmark aliases and `CUTLASS_CREATE_GEMM_BENCHMARK(...)`
2. emit a matching registration list for `register_gemm_benchmarks()`
3. build a benchmark binary for the batch

### Why batch builds instead of one binary per candidate

One-binary-per-candidate is easy to reason about but too expensive in build overhead.

Phase 1 should batch multiple candidates into one binary, but the batch size must stay small enough that:

- compile failures are easy to localize
- incremental rebuilds remain useful

Recommended starting policy:

- batch by `candidate_class`
- then split by dtype/layout family

For example:

- `batch_small_rcr_bf16`
- `batch_medium_rcr_bf16`
- `batch_large_rcr_bf16`

## Build manifest

`build_manifest.json` is not a user-facing final report.

It is the runtime lookup table that tells the runner which executable and config file correspond to which candidates.

Recommended record shape:

```json
{
  "batch_id": "batch_medium_rcr_bf16",
  "binary_path": "build/batch_medium_rcr_bf16/cutlass_benchmarks_gemm_sycl",
  "config_path": "generated/configs/batch_medium_rcr_bf16_screening.in",
  "candidate_ids": [
    "rcr_bf16bf16f32_tm32_tn128_tk32_sg2x4_st2_sk1",
    "rcr_bf16bf16f32_tm64_tn128_tk32_sg4x4_st2_sk1"
  ],
  "compiler_profile_id": "bmg.medium_tile.default",
  "build_status": "pass"
}
```

## Generated benchmark config files

The current benchmark binary already consumes `.in` files with lines like:

```text
<kernel_name> --bm_name=<name> --m=<m> --n=<n> --k=<k>
```

Phase 1 should keep using this transport layer.

### Config generation rule

For screening:

- one config line per `(candidate_id, shape_id)` pair

For confirmation:

- one config line per `(top_k_candidate_id, shape_id)` pair

### `bm_name` rule

Use a reversible name so the parser can recover both shape and stage:

```text
bm_name=<shape_id>__<stage>__<attempt_index>
```

Example:

```text
--bm_name=rcr_bf16_1_4096_14336__screening__0
```

## Binary CLI contract

Phase 1 does not require a brand-new benchmark CLI.

The generated workflow can stay on top of:

```text
cutlass_benchmarks_gemm_sycl --config_file=<generated.in>
```

However, generated kernels should preserve a path for binary-internal correctness checking.

### Verification requirement

Each generated candidate registration must support:

- correctness verification enabled during dedicated verify runs
- performance measurement runs with verification disabled if needed

The exact switch may be implemented as:

- generated compile-time flag
- generated benchmark option
- dedicated verify-only batch

The contract requirement is:

> verification result must end up in `gemm_profile_results.csv` as `verify_status`

## Subprocess runner behavior

### Invocation

Each run step executes:

```text
<binary_path> --config_file=<config_path>
```

in the correct build directory and with the selected runtime environment.

### Runtime environment

The runner must apply:

- compiler-profile environment
- oneAPI environment
- proxy/bootstrap environment required by the node

On B60-like nodes, remote commands should use:

```bash
bash -lc 'source /home/intel/.bashrc && <oneAPI env> && <cmd>'
```

### Failure handling

A subprocess failure must still produce CSV rows.

Required behavior:

1. capture return code
2. capture stdout/stderr log path
3. emit `status=fail`
4. emit `failure_reason`

No failure should disappear just because the subprocess exits non-zero.

## Result parsing

The current benchmark logs already contain:

- `avg_runtime_ms`
- `best_runtime_ms`
- `worst_runtime_ms`
- `avg_tflops`
- `avg_throughput`

The parser should extract those counters and join them with generated metadata.

### Join strategy

Do not infer candidate identity only from the free-form kernel token.

Instead:

1. generate config lines from structured `(candidate_id, shape_id)` metadata
2. persist that mapping in a side manifest
3. use benchmark output plus the manifest to reconstruct structured CSV rows

This avoids brittle parsing when kernel symbols become longer or generated.

## Two-stage selection flow

### Stage 1: screening

For each shape:

1. run all candidate pairs once
2. rank by `avg_tflops`
3. keep top `k=3`

### Stage 2: confirmation

For each shape:

1. rerun the top `k` candidates multiple times
2. compute median runtime / median tflops
3. choose final winner
4. mark `close_call=true` if gap is below threshold

The selection policy is then written into `gemm_dispatch_table.json`.

## Incremental execution policy

The workflow should support restart after partial failure.

Recommended persistence points:

- generated candidates
- build manifest
- build logs
- run logs
- accumulated CSV rows

This allows:

- re-running only failed builds
- re-running only failed shapes
- regenerating dispatch output without recompiling

## Minimal implementation path

The fastest way to realize the workflow is:

1. generate `gemm_candidate_space.json`
2. generate one small `RCR bf16` batch
3. emit one screening `.in` file
4. compile one benchmark binary
5. run it via subprocess
6. parse logs into `gemm_profile_results.csv`
7. emit one exact-shape dispatch file

This gives a complete vertical slice before broader candidate expansion.

## Exit criteria

This workflow definition is sufficient for Phase 1 when:

1. every schema artifact has a producing module
2. build and run boundaries are explicit
3. benchmark `.in` files are treated as transport artifacts, not the source of truth
4. subprocess failures map into structured results
5. the top-k confirmation loop is part of the contract rather than an afterthought
