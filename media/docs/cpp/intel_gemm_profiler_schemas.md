# Intel GEMM Profiler Schemas

## Purpose

This document locks the data contracts for the Intel GEMM profiler MVP.

The intent is to let the following modules evolve independently without drifting on file formats:

- `PhaseAProbeRunner`
- `TileSpaceGenerator`
- `BenchmarkCodegen`
- `BenchmarkRunner`
- `ResultParser`
- `BestSelector`
- `DispatchTableEmitter`

The MVP scope is:

- GEMM only
- exact-shape dispatch
- Intel SYCL path
- `RCR` first, with schema support for other layouts
- `bf16` / `f16` first, with schema support for extension

## Contract rules

### Canonical naming

All JSON keys use `snake_case`.

Enums are lowercase strings.

Canonical names:

| Concept | Canonical field |
| --- | --- |
| layout | `layout` |
| data types | `dtype_a`, `dtype_b`, `dtype_c`, `dtype_acc` |
| problem size | `m`, `n`, `k` |
| tile shape | `tile_m`, `tile_n`, `tile_k` |
| subgroup layout | `sg_m`, `sg_n` |
| pipeline stages | `stages` |
| split-k | `split_k` |
| benchmark result | `status`, `avg_runtime_ms`, `avg_tflops` |

### Stable IDs

The profiler uses explicit IDs so CSV rows and JSON records can be joined without reparsing free-form names.

Required IDs:

- `shape_id`
- `candidate_id`
- `compiler_profile_id`
- `probe_id`
- `run_id`
- `dispatch_id`

### Shape key

Phase 1 uses an exact key:

```text
(layout, dtype_a, dtype_b, dtype_c, dtype_acc, m, n, k)
```

This key is represented in JSON as the `shape_key` object rather than a packed string.

### Separation of concerns

- `verified_hw_caps.json` records **observations**
- `safe_search_constraints.json` records **search-time limits**
- `compiler_profiles.json` records **compiler/runtime presets**
- `gemm_target_shapes.json` records **requested problem shapes**
- `gemm_candidate_space.json` records **legal candidates after filtering**
- `gemm_profile_results.csv` records **per-run measurements**
- `gemm_dispatch_table.json` records **final selected mappings**

## Shared value domains

### Layout

Allowed values for Phase 1:

- `rcr`

Reserved for later:

- `rrr`
- `rcc`
- `ccr`

### Dtype

Allowed values for Phase 1:

- `bf16`
- `f16`
- `f32`

### Status fields

Common status values:

- `pass`
- `fail`
- `skip`
- `unsupported`

### Candidate class

Used for compiler profile mapping and reporting:

- `small_tile`
- `medium_tile`
- `large_tile`

## File 1: `verified_hw_caps.json`

### Role

Stores raw probe conclusions about hardware and compiler behavior on a specific node.

### Primary keys

- top-level `node_id`
- each probe record keyed by `probe_id`

### Required top-level fields

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `schema_version` | string | yes | Start with `1.0` |
| `generated_at` | string | yes | ISO 8601 |
| `node_id` | string | yes | e.g. hostname or inventory id |
| `device` | object | yes | static device identity |
| `toolchain` | object | yes | compiler/runtime identity |
| `probe_results` | array | yes | per-probe outputs |

### Device object

| Field | Type | Required |
| --- | --- | --- |
| `vendor` | string | yes |
| `architecture` | string | yes |
| `device_name` | string | yes |
| `driver_version` | string | no |
| `subgroup_size` | integer | yes |

### Toolchain object

| Field | Type | Required |
| --- | --- | --- |
| `cxx` | string | yes |
| `cxx_version` | string | yes |
| `cmake_version` | string | no |
| `level_zero_runtime` | string | no |
| `sycl_runtime` | string | no |

### Probe result object

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `probe_id` | string | yes | e.g. `compiler_flags_probe.large_tile` |
| `probe_type` | string | yes | `compiler_flags`, `slm_limit`, `block_copy`, `occupancy`, `prefetch`, `dpas` |
| `status` | string | yes | `pass`, `fail`, `skip` |
| `summary` | string | yes | one-line conclusion |
| `metrics` | object | yes | numeric results |
| `artifacts` | object | no | raw log paths |
| `issues` | array | no | known compiler/hardware issues |

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T19:30:00+08:00",
  "node_id": "b60-172.16.114.104",
  "device": {
    "vendor": "intel",
    "architecture": "bmg",
    "device_name": "Intel Graphics",
    "subgroup_size": 16
  },
  "toolchain": {
    "cxx": "icpx",
    "cxx_version": "2026.0",
    "sycl_runtime": "oneAPI"
  },
  "probe_results": [
    {
      "probe_id": "compiler_flags_probe.large_tile",
      "probe_type": "compiler_flags",
      "status": "pass",
      "summary": "256 GRF plus perfmodel is best for large_tile.",
      "metrics": {
        "winner": "large_tile_perf_a",
        "avg_tflops": 13.2,
        "margin_percent": 18.4
      }
    }
  ]
}
```

## File 2: `safe_search_constraints.json`

### Role

Stores the distilled constraints used by `TileSpaceGenerator`.

This file is derived from `verified_hw_caps.json` or from hardcoded BMG defaults when probes are not available.

### Required top-level fields

| Field | Type | Required |
| --- | --- | --- |
| `schema_version` | string | yes |
| `generated_at` | string | yes |
| `constraint_source` | string | yes |
| `device_arch` | string | yes |
| `limits` | object | yes |
| `allowed_values` | object | yes |
| `blocked_rules` | array | no |

### `constraint_source`

Allowed values:

- `probe`
- `default_bmg`

### Limits object

| Field | Type | Required |
| --- | --- | --- |
| `max_slm_kb` | integer | yes |
| `subgroup_size` | integer | yes |
| `max_split_k` | integer | yes |
| `max_stages` | integer | yes |

### Allowed values object

| Field | Type | Required |
| --- | --- | --- |
| `tile_m` | array of integers | yes |
| `tile_n` | array of integers | yes |
| `tile_k` | array of integers | yes |
| `sg_m` | array of integers | yes |
| `sg_n` | array of integers | yes |
| `stages` | array of integers | yes |
| `split_k` | array of integers | yes |

### Blocked rule object

| Field | Type | Required |
| --- | --- | --- |
| `rule_id` | string | yes |
| `reason` | string | yes |
| `match` | object | yes |

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T19:35:00+08:00",
  "constraint_source": "default_bmg",
  "device_arch": "bmg",
  "limits": {
    "max_slm_kb": 128,
    "subgroup_size": 16,
    "max_split_k": 4,
    "max_stages": 3
  },
  "allowed_values": {
    "tile_m": [8, 16, 32, 64, 128, 256],
    "tile_n": [64, 128, 256],
    "tile_k": [32, 64],
    "sg_m": [1, 2, 4, 8],
    "sg_n": [4, 8],
    "stages": [1, 2, 3],
    "split_k": [1, 2, 4]
  },
  "blocked_rules": [
    {
      "rule_id": "block_copy.bad_16x256",
      "reason": "probe marked this block copy shape unstable",
      "match": {
        "tile_m": 16,
        "tile_n": 256
      }
    }
  ]
}
```

## File 3: `compiler_profiles.json`

### Role

Stores compiler/runtime presets selected by Phase A for candidate classes.

### Required top-level fields

| Field | Type | Required |
| --- | --- | --- |
| `schema_version` | string | yes |
| `generated_at` | string | yes |
| `profiles` | array | yes |

### Compiler profile object

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `compiler_profile_id` | string | yes | stable identifier |
| `candidate_class` | string | yes | `small_tile`, `medium_tile`, `large_tile` |
| `description` | string | yes | human-readable summary |
| `selector` | object | yes | matching rule for candidates |
| `env` | object | yes | environment variables |
| `cmake_flags` | array of strings | yes | configure-time flags |
| `compile_flags` | array of strings | no | file/target flags |
| `probe_evidence` | object | no | metrics from probe stage |

### Selector object

At least one of the following should be present:

- `tile_m_min`
- `tile_m_max`
- `sg_count_min`
- `sg_count_max`

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T19:40:00+08:00",
  "profiles": [
    {
      "compiler_profile_id": "bmg.large_tile.perf_a",
      "candidate_class": "large_tile",
      "description": "Large tiles use 256 GRF plus perfmodel.",
      "selector": {
        "tile_m_min": 128,
        "sg_count_min": 16
      },
      "env": {
        "IGC_ExtraOCLOptions": "-cl-intel-256-GRF-per-thread",
        "IGC_VISAOptions": "-perfmodel",
        "IGC_VectorAliasBBThreshold": "100000000000",
        "SYCL_PROGRAM_COMPILE_OPTIONS": "-ze-opt-large-register-file -gline-tables-only"
      },
      "cmake_flags": [
        "-DCMAKE_BUILD_TYPE=Release",
        "-DCUTLASS_ENABLE_SYCL=ON"
      ],
      "probe_evidence": {
        "winner_margin_percent": 18.4
      }
    }
  ]
}
```

## File 4: `gemm_target_shapes.json`

### Role

Defines the problem shapes requested by the user or by a curated shape set.

### Required top-level fields

| Field | Type | Required |
| --- | --- | --- |
| `schema_version` | string | yes |
| `generated_at` | string | yes |
| `shape_set_id` | string | yes |
| `source` | string | yes |
| `shapes` | array | yes |

### `source`

Allowed values:

- `cli`
- `json_file`
- `csv_file`
- `predefined`

### Shape object

| Field | Type | Required |
| --- | --- | --- |
| `shape_id` | string | yes |
| `layout` | string | yes |
| `dtype_a` | string | yes |
| `dtype_b` | string | yes |
| `dtype_c` | string | yes |
| `dtype_acc` | string | yes |
| `m` | integer | yes |
| `n` | integer | yes |
| `k` | integer | yes |
| `tags` | array of strings | no |

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T19:45:00+08:00",
  "shape_set_id": "decode_prefill_v1",
  "source": "predefined",
  "shapes": [
    {
      "shape_id": "rcr_bf16_1_4096_14336",
      "layout": "rcr",
      "dtype_a": "bf16",
      "dtype_b": "bf16",
      "dtype_c": "f32",
      "dtype_acc": "f32",
      "m": 1,
      "n": 4096,
      "k": 14336,
      "tags": ["decode"]
    }
  ]
}
```

## File 5: `gemm_candidate_space.json`

### Role

Defines the candidate kernels that survive Phase A constraints and are eligible for build/run.

### Required top-level fields

| Field | Type | Required |
| --- | --- | --- |
| `schema_version` | string | yes |
| `generated_at` | string | yes |
| `device_arch` | string | yes |
| `constraint_source` | string | yes |
| `candidates` | array | yes |

### Candidate object

| Field | Type | Required | Notes |
| --- | --- | --- | --- |
| `candidate_id` | string | yes | stable identifier |
| `layout` | string | yes | phase-1 `rcr` |
| `dtype_a` | string | yes | |
| `dtype_b` | string | yes | |
| `dtype_c` | string | yes | |
| `dtype_acc` | string | yes | |
| `tile_m` | integer | yes | |
| `tile_n` | integer | yes | |
| `tile_k` | integer | yes | |
| `sg_m` | integer | yes | |
| `sg_n` | integer | yes | |
| `stages` | integer | yes | |
| `split_k` | integer | yes | |
| `candidate_class` | string | yes | |
| `compiler_profile_id` | string | yes | selected from `compiler_profiles.json` |
| `estimated_slm_kb` | number | no | pre-build estimate |
| `filters_applied` | array of strings | no | traceability |
| `kernel_name` | string | no | generated benchmark symbol if known |

### Candidate naming rule

The recommended `candidate_id` format is:

```text
<layout>_<dtype_a><dtype_b><dtype_c>_tm<tile_m>_tn<tile_n>_tk<tile_k>_sg<sg_m>x<sg_n>_st<stages>_sk<split_k>
```

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T19:50:00+08:00",
  "device_arch": "bmg",
  "constraint_source": "default_bmg",
  "candidates": [
    {
      "candidate_id": "rcr_bf16bf16f32_tm64_tn128_tk32_sg4x4_st2_sk1",
      "layout": "rcr",
      "dtype_a": "bf16",
      "dtype_b": "bf16",
      "dtype_c": "f32",
      "dtype_acc": "f32",
      "tile_m": 64,
      "tile_n": 128,
      "tile_k": 32,
      "sg_m": 4,
      "sg_n": 4,
      "stages": 2,
      "split_k": 1,
      "candidate_class": "medium_tile",
      "compiler_profile_id": "bmg.medium_tile.default",
      "estimated_slm_kb": 96,
      "filters_applied": ["fits_slm", "legal_sg_shape"]
    }
  ]
}
```

## File 6: `gemm_profile_results.csv`

### Role

Stores all benchmark observations in a flat join-friendly format.

This is the main input to `BestSelector`.

### Row granularity

One row per:

```text
(run_id, shape_id, candidate_id, attempt_index, stage)
```

### Required columns

| Column | Type | Notes |
| --- | --- | --- |
| `run_id` | string | unique execution batch id |
| `stage` | string | `screening` or `confirm` |
| `attempt_index` | integer | 0-based |
| `shape_id` | string | joins `gemm_target_shapes.json` |
| `candidate_id` | string | joins `gemm_candidate_space.json` |
| `compiler_profile_id` | string | joins `compiler_profiles.json` |
| `status` | string | `pass`, `fail`, `skip`, `unsupported` |
| `verify_status` | string | `pass`, `fail`, `not_run` |
| `layout` | string | denormalized for reporting |
| `dtype_a` | string | denormalized for reporting |
| `dtype_b` | string | denormalized for reporting |
| `dtype_c` | string | denormalized for reporting |
| `dtype_acc` | string | denormalized for reporting |
| `m` | integer | |
| `n` | integer | |
| `k` | integer | |
| `avg_runtime_ms` | number | aligns with current benchmark counters |
| `best_runtime_ms` | number | aligns with current benchmark counters |
| `worst_runtime_ms` | number | aligns with current benchmark counters |
| `avg_tflops` | number | aligns with current benchmark counters |
| `avg_throughput` | number | aligns with current benchmark counters |
| `max_error` | number | optional zero when exact compare is used |
| `close_call_group` | string | optional selector grouping |
| `failure_reason` | string | empty on success |
| `stdout_log` | string | relative log path |

### Example

```csv
run_id,stage,attempt_index,shape_id,candidate_id,compiler_profile_id,status,verify_status,layout,dtype_a,dtype_b,dtype_c,dtype_acc,m,n,k,avg_runtime_ms,best_runtime_ms,worst_runtime_ms,avg_tflops,avg_throughput,max_error,close_call_group,failure_reason,stdout_log
202604281955_screen,screening,0,rcr_bf16_1_4096_14336,rcr_bf16bf16f32_tm64_tn128_tk32_sg4x4_st2_sk1,bmg.medium_tile.default,pass,pass,rcr,bf16,bf16,f32,f32,1,4096,14336,0.412,0.398,0.437,1.13,287.4,0,,,"logs/run_001.log"
```

## File 7: `gemm_dispatch_table.json`

### Role

Stores the final exact-shape mapping emitted by `DispatchTableEmitter`.

### Required top-level fields

| Field | Type | Required |
| --- | --- | --- |
| `schema_version` | string | yes |
| `generated_at` | string | yes |
| `dispatch_id` | string | yes |
| `selection_policy` | object | yes |
| `entries` | array | yes |

### Selection policy object

| Field | Type | Required |
| --- | --- | --- |
| `screening_top_k` | integer | yes |
| `confirm_runs` | integer | yes |
| `metric` | string | yes |
| `close_call_threshold_percent` | number | yes |

### Dispatch entry object

| Field | Type | Required |
| --- | --- | --- |
| `shape_key` | object | yes |
| `shape_id` | string | yes |
| `candidate_id` | string | yes |
| `compiler_profile_id` | string | yes |
| `status` | string | yes |
| `selected_metric` | number | yes |
| `runner_up_candidate_id` | string | no |
| `runner_up_gap_percent` | number | no |
| `close_call` | boolean | yes |
| `evidence` | object | yes |

### Evidence object

| Field | Type | Required |
| --- | --- | --- |
| `confirm_median_runtime_ms` | number | yes |
| `confirm_median_tflops` | number | yes |
| `screening_rank` | integer | yes |
| `confirm_samples` | integer | yes |

### Example

```json
{
  "schema_version": "1.0",
  "generated_at": "2026-04-28T20:00:00+08:00",
  "dispatch_id": "bmg_rcr_bf16_v1",
  "selection_policy": {
    "screening_top_k": 3,
    "confirm_runs": 5,
    "metric": "confirm_median_tflops",
    "close_call_threshold_percent": 3.0
  },
  "entries": [
    {
      "shape_key": {
        "layout": "rcr",
        "dtype_a": "bf16",
        "dtype_b": "bf16",
        "dtype_c": "f32",
        "dtype_acc": "f32",
        "m": 1,
        "n": 4096,
        "k": 14336
      },
      "shape_id": "rcr_bf16_1_4096_14336",
      "candidate_id": "rcr_bf16bf16f32_tm64_tn128_tk32_sg4x4_st2_sk1",
      "compiler_profile_id": "bmg.medium_tile.default",
      "status": "pass",
      "selected_metric": 1.18,
      "runner_up_candidate_id": "rcr_bf16bf16f32_tm32_tn128_tk32_sg2x4_st2_sk1",
      "runner_up_gap_percent": 2.1,
      "close_call": true,
      "evidence": {
        "confirm_median_runtime_ms": 0.401,
        "confirm_median_tflops": 1.18,
        "screening_rank": 1,
        "confirm_samples": 5
      }
    }
  ]
}
```

## Transient file: generated benchmark input `.in`

The current benchmark binary consumes a line-based config file:

```text
<kernel_name> --bm_name=<name> --m=<m> --n=<n> --k=<k>
```

This file remains a generated transport artifact rather than a canonical schema artifact.

Reason:

- the repository already supports this format
- it is suitable as a temporary handoff into `cutlass_benchmarks_gemm_sycl`
- the canonical source of truth should remain `gemm_target_shapes.json` plus `gemm_candidate_space.json`

## Module interfaces

### `TileSpaceGenerator`

Input:

- `safe_search_constraints.json`
- `compiler_profiles.json`

Output:

- `gemm_candidate_space.json`

### `BenchmarkCodegen`

Input:

- `gemm_candidate_space.json`
- `compiler_profiles.json`

Output:

- generated source files
- generated benchmark `.in` files
- build manifest keyed by `candidate_id`

### `BenchmarkRunner`

Input:

- `gemm_target_shapes.json`
- generated benchmark `.in` files
- build manifest

Output:

- raw logs
- `gemm_profile_results.csv`

### `BestSelector`

Input:

- `gemm_profile_results.csv`

Output:

- `gemm_dispatch_table.json`

## MVP non-goals for this schema version

The following are intentionally excluded from schema version `1.0`:

- grouped gemm
- attention
- shape bucketing or range dispatch
- oneDNN performance baseline integration
- multi-device dispatch
- auto-generated manifest compatible with CUDA CUTLASS profiler

## Versioning rule

Schema changes that rename fields or change row granularity must bump `schema_version`.

Field additions that are strictly optional may keep the same major version.
