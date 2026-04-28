# Intel GEMM Profiler MVP Deliverables

## Goal

This document defines what counts as a successful Phase 1 GEMM profiler MVP on Intel SYCL.

The MVP is complete when it delivers a usable end-to-end tuning loop, not just isolated scripts or benchmark binaries.

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

Minimum proof:

- at least one generated `RCR bf16` candidate batch builds successfully on B60

### 7. Correctness path

Required capability:

- generated candidates can report `verify_status`

Minimum proof:

- at least one target shape completes with `verify_status=pass`

### 8. Selection path

Required capability:

- top-k screening plus confirmation

Minimum proof:

- at least one shape produces a single dispatch-table winner with evidence fields populated

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

## Acceptance criteria

### A. Functional acceptance

The MVP must:

1. accept an explicit shape set
2. generate candidate kernels from the approved search space
3. build those candidates on the Intel node
4. run them through subprocess execution
5. collect normalized results
6. emit an exact-shape dispatch table

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
