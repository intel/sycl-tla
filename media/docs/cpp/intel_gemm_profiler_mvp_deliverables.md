# Intel GEMM Profiler MVP Deliverables

## Goal

This document defines what counts as a successful Phase 1 GEMM profiler MVP on Intel SYCL.

The MVP is complete when it delivers a usable end-to-end tuning loop, not just isolated scripts or benchmark binaries.

## Current implementation status

The GEMM MVP vertical slice is now operational for the Intel/BMG generated benchmark path.

Validated end-to-end proof:

- Ali workbook input: 76 BF16 GEMM shapes
- generated catalog source: `--kernel-catalog-source generator`
- generator instantiation level: `1`
- generated benchmark candidates: 14
- benchmark result rows: 1064
- passed rows: 1064
- failed rows: 0
- timeout rows: 0
- dispatch entries: 76
- Ali reference matches: 76
- missing dispatch entries: 0

The current workflow also supports:

- candidate benchmark auto-build through `--build-candidate-benchmark`
- local Google Benchmark source injection through `--googlebenchmark-dir`
- chunked benchmark subprocess execution through `--benchmark-entry-chunk-size`
- top-k confirmation through `--top-k` and `--confirm-runs`
- median-based final selection evidence in `gemm_dispatch_table.json`
- exact-shape runtime dispatch lookup CLI with schema validation and explicit fallback metadata
- product bundle validation CLI suitable for release/CI gates
- close-call labeling through `--close-call-threshold`
- selected-kernel batch filter emission through `--candidate-build-batch-size`
- per-batch build preflight execution through `--run-candidate-build-preflight`
- optional screening/confirmation routing to preflighted per-batch benchmark binaries through `--use-candidate-build-preflight-benchmarks`
- generated Intel Xe `StageCountAuto` candidates represented as `stages=0`
- generated candidate coverage and exception summaries in `gemm_candidate_space.json`
- native C++ `tools/profiler/cutlass_profiler` generated GEMM profiling with host-reference verification

Confirmation smoke proof:

- Ali subset: 2 BF16 GEMM shapes
- confirmation policy: `--top-k 3 --confirm-runs 2`
- benchmark rows: 40
- passed rows: 40
- failed rows: 0
- dispatch entries: 2
- entries with confirmation evidence: 2
- incomplete confirmation entries: 0

Batch routing smoke proof:

- remote BMG node
- manual BF16 RCR shape: `m=128, n=128, k=32`
- generator catalog: level 1
- compiled-kernel-list: 2 generated kernels
- batch size: 1 kernel per preflight build
- preflight batches: 2
- passed preflight batches: 2
- aggregate candidate build: not run
- screening rows: 2
- passed rows: 2
- failed rows: 0
- dispatch entries: 1
- batch-specific config/manifest/log artifacts emitted for both batches

Generated level0 StageCountAuto smoke proof:

- remote BMG node
- manual BF16 RCR shape: `m=128, n=128, k=32`
- generator catalog: level 0
- compiled-kernel-list: 1 generated StageCountAuto kernel
- candidate stage: `st0`
- preflight batches: 1
- passed preflight batches: 1
- screening rows: 1
- passed rows: 1
- failed rows: 0
- dispatch entries: 1

F16 non-RCR generated layout smoke proof:

- remote BMG node
- generator catalog: level 0
- candidate stage: `st0`
- F16 RRR shape: `m=128, n=128, k=32`
- F16 CCR shape: `m=128, n=128, k=32`
- compiled-kernel-list: 1 generated StageCountAuto kernel per run
- RRR selected candidate: `rrr_f16f16f32_tm128_tn128_tk32_sg4x4_st0_sk1`
- CCR selected candidate: `ccr_f16f16f32_tm128_tn128_tk32_sg4x4_st0_sk1`
- each run passed: 1 preflight batch, 1 screening row, 0 failed rows, 1 dispatch entry

Generated StreamK limitation tracking:

- Intel Xe `GemmUniversal` currently rejects generated `StreamKScheduler` specialization at compile time.
- The workflow keeps generated StreamK candidates out of `candidate_build_manifest.json` and records the reason in `candidate_exceptions`.
- `candidate_exception_summary` aggregates this by reason and includes sample kernel names.
- `candidate_coverage` records catalog, matched-signature, accepted, blocked, and exception counts for auditability.
- BF16/RCR level1 artifact smoke recorded 76 `intel_xe_generated_streamk_tile_scheduler_unsupported` exceptions and 28 accepted candidates.

Native `tools/profiler/cutlass_profiler` GEMM proof:

- remote BMG node
- generated manifest filtered to 1 operation
- operation: `cutlass3x_xe20_tensorop_gemm_bf16_bf16_f32_f32_f32_128x128x32_1x1x1_0_tnt_align8`
- shape: `m=128, n=128, k=32`
- tensors: `A=bf16:row, B=bf16:column, C=f32:row, D=f32:row`
- verification provider: `reference_host`
- disposition: `passed`
- status: `success`
- runtime: `0.006636 ms`
- throughput: `162.951 GFLOP/s`
- output: `/home/intel/tianfeng/cutlas_profile_validation/profiler_gemm_f32d.gemm.csv`

This proof is separate from the Python-orchestrated benchmark workflow: it validates the native CUTLASS profiler executable path can register, run, verify, time, and report a generated SYCL GEMM operation. A `D=bf16` generated kernel was also able to profile but reported `not_verified`; broader `D=bf16` native reference coverage remains a follow-up item.

## MVP scope

The MVP includes:

- GEMM only
- Intel SYCL path only
- exact-shape tuning
- `RCR` as the primary layout
- `bf16` first, with `f16` as an extension path
- offline tuning only
- dispatch-table emission

The MVP excludes:

- grouped gemm
- attention
- quantized / blockscaled gemm
- online heuristic selection
- oneDNN performance comparison in the main loop

## Required deliverables

### 1. Schema contract

Required artifact:

- `media/docs/cpp/intel_gemm_profiler_schemas.md`

Outcome:

- all intermediate files and result formats are fixed

### 2. Search-space definition

Required artifact:

- `media/docs/cpp/intel_gemm_profiler_search_space.md`

Outcome:

- the first-wave candidate domain is fixed
- target shape sets are explicit

### 3. Phase A probe definition

Required artifact:

- `media/docs/cpp/intel_gemm_profiler_phase_a_probes.md`

Outcome:

- probe inventory and output mapping are fixed

### 4. Binary/subprocess workflow definition

Required artifact:

- `media/docs/cpp/intel_gemm_profiler_binary_workflow.md`

Outcome:

- codegen/build/run/parser boundaries are explicit

### 5. Working vertical slice

Required runtime outputs:

- `gemm_target_shapes.json`
- `gemm_candidate_space.json`
- `gemm_profile_results.csv`
- `gemm_dispatch_table.json`

Outcome:

- one end-to-end tuning run can start from shapes and finish with a dispatch table

### 6. Benchmark build path

Required capability:

- generated `RCR` benchmark registrations can be built on the target node
- configure/build failures are captured in `candidate_build_summary.json`

Minimum proof:

- at least one generated `RCR bf16` candidate batch builds successfully on B60

Failure handling:

- `candidate_build_summary.json` is written before the workflow raises on candidate benchmark build failure.
- Failed summaries include `status`, `failure_step`, `failure_reason`, `selected_kernel_count`, `kernel_filter_file`, and per-step command/log/returncode metadata.
- `--candidate-build-batch-size` can emit additional `selected_kernel_filter_partXXX.list` files and `selected_kernel_batches` metadata for large-catalog preflight and retry isolation.
- `candidate_build_plan.json` includes `batch_preflight_plans` with per-batch configure/build commands and isolated build directories when batch artifacts are enabled.
- `--run-candidate-build-preflight` executes those plans and writes `candidate_build_preflight_summary.json` before raising on failed batch preflight.
- `--use-candidate-build-preflight-benchmarks` routes screening/confirmation entries to the successful per-batch benchmark binaries and emits batch-specific config/manifest/log artifacts. It requires successful preflight execution and does not silently fall back to the aggregate binary.

### 7. Correctness path

Required capability:

- generated candidates can report `verify_status`
- native `cutlass_profiler` can report a verified GEMM row for at least one generated SYCL GEMM operation

Minimum proof:

- at least one target shape completes with `verify_status=pass`
- at least one native `tools/profiler/cutlass_profiler` generated GEMM row completes with `Disposition=passed`

### 8. Selection path

Required capability:

- top-k screening plus confirmation

Minimum proof:

- at least one shape produces a single dispatch-table winner with evidence fields populated

Current implementation:

- screening uses the normalized benchmark rows in `gemm_profile_results.csv`
- confirmation entries are generated from the per-shape screening top-k candidates
- final selection uses confirmation median TFLOPS when confirmation rows are present
- `evidence` records:
  - selection stage
  - median runtime
  - median TFLOPS
  - runtime standard deviation
  - TFLOPS standard deviation
  - TFLOPS coefficient of variation
  - screening rank
  - confirmation sample count
  - expected confirmation sample count
  - confirmation completeness
  - runner-up median metrics
  - ranked top candidates
- `selection_summary` records dispatch entry count, confirmation coverage, incomplete confirmation count, and close-call count

## Minimal runtime deliverables

For the first successful MVP run, the minimum artifact set is:

| Artifact | Purpose |
| --- | --- |
| `gemm_target_shapes.json` | requested tuning shapes |
| `safe_search_constraints.json` | search boundary |
| `compiler_profiles.json` | compiler/runtime presets |
| `gemm_candidate_space.json` | buildable/searchable candidates |
| generated `.in` files | transport into benchmark binary |
| build logs | compile failure diagnosis |
| run logs | runtime failure diagnosis |
| `gemm_profile_results.csv` | normalized measurements |
| `gemm_dispatch_table.json` | final selection result |
| `optimal_dispatch_table.json` | alias of the final best-dispatch artifact for downstream consumers; accepted by runtime dispatch lookup |
| `gemm_product_bundle_manifest.json` | product handoff manifest listing required/optional artifacts and the dispatch lookup CLI template |
| `reference_comparison.json` | optional comparison against Ali/reference performance data |
| `run_summary.json` | row counts, pass/fail counts, benchmark commands, and log paths |
| `phase_a_summary.json` | Phase A/probe summary |
| `phase_b_summary.json` | candidate/search/dispatch summary |

## Acceptance criteria

### A. Functional acceptance

The MVP must:

1. accept an explicit shape set
2. generate candidate kernels from the approved search space
3. build those candidates on the Intel node
4. run them through subprocess execution
5. collect normalized results
6. emit an exact-shape dispatch table
7. load and query that dispatch table by exact runtime shape key without silent fallback

### B. Correctness acceptance

The MVP must:

1. surface verify results explicitly
2. fail loudly when correctness fails
3. never silently promote an unverified candidate into the dispatch table

### C. Operability acceptance

The MVP must:

1. support restart from partial progress
2. keep generated artifacts in a dedicated workspace
3. retain build/run logs for diagnosis

### D. Scope acceptance

The MVP must remain within the approved narrow scope.

Any attempt to add grouped gemm, attention, or BlockScaledGemm to the first vertical slice is scope creep.

## Recommended demo scenario

The MVP demo should be intentionally small.

### Demo input

Use:

- one decode-style shape
- one prefill-style shape
- `layout = rcr`
- `dtype = bf16`

Suggested shapes:

- decode: `(m=1, n=4096, k=14336)`
- prefill: `(m=64, n=4096, k=4096)`

### Demo candidate wave

Start with a small seed-adjacent candidate set:

- `8x64x32`, `sg=1x4`
- `8x128x32`, `sg=1x4`
- `16x64x32`, `sg=2x4`
- `64x128x32`, `sg=4x4`
- `256x256x32`, `sg=8x4`

With:

- `stages = 2`
- `split_k = 1`

### Demo success criteria

The demo is successful when it produces:

1. a buildable candidate batch
2. a non-empty `gemm_profile_results.csv`
3. at least one `verify_status=pass`
4. a non-empty `gemm_dispatch_table.json`

## Required test categories

### 1. Schema tests

Validate:

- required JSON keys exist
- CSV headers match the schema document
- IDs are unique where required

### 2. Candidate-generation tests

Validate:

- forbidden combinations are pruned
- compiler profiles are assigned
- candidate IDs are stable

### 3. Build tests

Validate:

- generated registrations compile
- failed builds are reported into the manifest

### 4. Correctness tests

Validate:

- verify pass/fail is surfaced
- failed verification cannot become a selected dispatch entry

### 5. Parsing tests

Validate:

- benchmark logs map to normalized CSV rows
- subprocess failures still produce structured result rows

### 6. Selection tests

Validate:

- screening top-k is correct
- confirmation median is used for final selection
- close-call threshold is applied consistently

## Reporting requirements

Each MVP run should produce a concise report containing:

- node id
- git commit id
- shape set id
- candidate count
- build success/failure count
- verification success/failure count
- selected candidate per shape
- close-call cases

This can be emitted as a markdown summary or a JSON run summary.

## Non-goals for the MVP report

The report does not need to include:

- oneDNN performance charts
- generalized heuristic rules
- compressed/ranged dispatch rules
- cross-device comparisons

## Exit criteria

The GEMM MVP is considered ready when:

1. all design documents exist
2. the vertical slice can run on the target Intel node
3. results are persisted in the approved schema artifacts
4. at least one shape produces a verified dispatch-table winner
