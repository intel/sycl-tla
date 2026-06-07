# Intel GEMM Profiler Search Space

## Goal

This document defines the Phase 1 search space for the Intel GEMM profiler MVP.

It answers five questions:

1. which GEMM family is in scope
2. which shapes are worth tuning first
3. which kernel dimensions are searchable
4. which combinations are pruned before build/run
5. how candidates are named and classified

The output of this document feeds:

- `gemm_target_shapes.json`
- `gemm_candidate_space.json`
- compiler-profile selection
- generated benchmark registration and config files

## Scope

Phase 1 started intentionally narrow:

- GEMM only
- Intel SYCL path only
- exact-shape dispatch only
- `RCR` first-class target
- `bf16` and `f16` inputs with `f32` accumulation

Current implementation status has expanded beyond the original MVP note:

- legacy `baseline` search is still preserved
- legacy `expanded_bmg` search is still preserved
- legacy `layered_exhaustive` search is still preserved
- `RCR` and `RRR` are both searchable in the layered exhaustive path
- `bruteforce_scheduler` now widens only the BF16 scheduler axis (`sg_m/sg_n/stages`) while keeping the preserved legacy strategies unchanged

Out of scope for this search-space version:

- grouped gemm
- attention
- quantized/blockscaled gemm
- layout bucketing
- heuristic online dispatch

## Search strategy preservation

The profiler now exposes a high-level `--search-strategy` preset so new search behavior does not overwrite older standards.

| Strategy | Preserved meaning | Candidate universe |
| --- | --- | --- |
| `baseline` | old persisted seed baseline | `persisted` catalog only |
| `expanded_bmg` | old expanded BMG search | `expanded_bmg` catalog only |
| `layered_exhaustive` | old layered exhaustive search | `layered_bmg` catalog only |
| `bruteforce_scheduler` | new scheduler brute-force mode | `layered_bmg_scheduler_expanded`: keep old regular-GEMM space, widen BF16 scheduler candidates across legal `sg_m/sg_n/stages`, and route through preflight batch benchmark execution |
| `manual` | no preset override | keep explicit low-level flags unchanged |

Compatibility rules:

- `--kernel-catalog-source` remains valid
- `--bruteforce-scheduler-search` remains valid
- `--bruteforce-scheduler-search` is treated as a compatibility alias of `--search-strategy bruteforce_scheduler`

This means the new scheduler search mode is additive. It does **not** replace the old baseline or old exhaustive standards.

## Current quantitative snapshot

The numbers below are the current BF16/BF16/FP32 snapshot for one exact-shape probe under:

- layouts: `rcr + rrr`
- `allowed_runners = (benchmark, streamk_example)`
- `default_constraints()`
- no heuristic prefilter

| Search space | Effective candidates |
| --- | ---: |
| `baseline` | 53 |
| `expanded_bmg` | 399 |
| `layered_exhaustive` | 1857 |
| `bruteforce_scheduler` | 6843 |

Interpretation:

- the preserved legacy strategies stay unchanged
- `bruteforce_scheduler` is now strictly wider because it expands BF16 scheduler candidates beyond fixed `sg=8x4, stages=2`
- the widening is limited to scheduler kernels; it does not rewrite the preserved `baseline`, `expanded_bmg`, or `layered_exhaustive` standards

## Current repository baseline

### What the new benchmark path currently exposes

`benchmarks/gemm/benchmarks_sycl.hpp` is no longer a single-kernel placeholder.

It now exposes:

- preserved `RCR` / `RRR` GEMM seed kernels
- benchmark-backed `StreamK` / `DataParallel` / `SplitK` seed tiles
- under `CUTLASS_BENCHMARK_EXPANDED_BMG_STREAMK=ON`, the benchmark-backed expanded scheduler registry needed by the current profiler catalogs
- BF16 `RRR` benchmark-backed scheduler registrations, so the layered exhaustive / brute-force scheduler catalog is not Python-only on that layout

The important remaining limitation is different:

- scheduler benchmark registration is now aligned with the current cataloged tile set
- but the scheduler path still fixes subgroup layout to `8x4` and stages to `2`

### What the legacy benchmark path already proves

`benchmarks/gemm/legacy/benchmarks_sycl.hpp` contains the most useful seed variants for the new search space:

| Seed variant | Layout | Tile | SG layout | Scheduler |
| --- | --- | --- | --- | --- |
| `RCR_5` | `rcr` | `8x128x32` | `1x4` | `gemm` |
| `RCR_6` | `rcr` | `256x256x32` | `8x4` | `gemm` |
| `RCR_7` | `rcr` | `8x128x32` | `1x8` | `gemm` |
| `RCR_9` | `rcr` | `8x64x32` | `1x4` | `gemm` |
| `RCR_16` | `rcr` | `16x64x32` | `2x4` | `gemm` |
| `SplitK_RCR_5` | `rcr` | `8x64x32` | `1x4` | `split_k` |

These seed variants were the original proof points. The current benchmark path now re-registers the relevant seed bands and a much wider scheduler tile set on top of them.

## Phase 1 target shape sets

The profiler does not auto-generate a full problem space.

It consumes an explicit shape set.

### Target shape set A: decode

Recommended starter shapes:

- `m`: `[1, 8]`
- `n`: `[1024, 2048, 4096, 8192, 16384]`
- `k`: `[4096, 8192, 14336]`
- `layout`: `rcr`
- `dtype_a/dtype_b`: `bf16` first, then `f16`
- `dtype_c/dtype_acc`: `f32`

Reason:

- matches the small-`m` region already hinted by legacy `RCR_5`, `RCR_7`, `RCR_9`
- gives fast feedback during early candidate bring-up

### Target shape set B: prefill / dense linear

Recommended starter shapes:

- `m`: `[16, 32, 64, 128, 256, 512, 1024, 2048]`
- `n`: `[4096]`
- `k`: `[4096, 8192, 14336]`
- `layout`: `rcr`
- `dtype_a/dtype_b`: `bf16` first, then `f16`
- `dtype_c/dtype_acc`: `f32`

Reason:

- covers the region currently missing from the existing handcrafted registry
- is the most likely place to see meaningful tuning wins over current defaults

## Search dimensions

The searchable kernel dimensions for Phase 1 are:

| Dimension | Meaning | Search |
| --- | --- | --- |
| `tile_m` | workgroup tile M | yes |
| `tile_n` | workgroup tile N | yes |
| `tile_k` | workgroup tile K | yes |
| `sg_m` | subgroup layout in M | yes |
| `sg_n` | subgroup layout in N | yes |
| `stages` | software pipeline depth | yes |
| `split_k` | K partition count | yes |

The following are not Phase 1 search dimensions:

| Dimension | Reason |
| --- | --- |
| `cluster` | NVIDIA-specific |
| `raster_order` | not locked for Intel MVP |
| `swizzle_size` | postpone until there is evidence it matters on Intel |
| `dpas atom shape` | treated as fixed hardware property |
| `block_copy opcode form` | handled as probe-derived constraints, not the outer search axis |

## How `RCR` / `RRR` scheduler search works now

The current implementation uses two different modes:

1. **legacy preserved search spaces**
2. **layered exhaustive / brute-force scheduler search**

### 1. Legacy preserved spaces

#### `baseline`

- source: `seed_catalog_level0`
- purpose: preserve the small, known-good seed set
- behavior: not brute force
- scheduler candidates: only what already exists in the seed catalog

For the current BF16/BF16/FP32 snapshot:

- `rcr / Gemm = 4`
- `rrr / Gemm = 1`
- `rcr / DataParallel = 8`
- `rcr / SplitK = 32`
- `rcr / StreamK = 8`

#### `expanded_bmg`

- sources:
  - `seed_catalog_level0`
  - `expanded_gemm_catalog`
  - `expanded_streamk_catalog`
  - `source_template_gemm_catalog`
- purpose: preserve the older expanded search standard
- behavior: expanded but still not full exhaustive across all legal tiles/subgroups/stages
- scheduler candidates:
  - `RCR` scheduler search is explicit enumeration over the benchmark-backed legal `8x4` tile list plus the retained legacy `512x256x32` seed tile
  - `RRR` remains GEMM-only in this preserved legacy mode

For the current BF16/BF16/FP32 snapshot:

- `rcr / Gemm = 55`
- `rrr / Gemm = 55`
- `rcr / DataParallel = 49`
- `rcr / SplitK = 196`
- `rcr / StreamK = 49`

### 2. `layered_exhaustive` and `bruteforce_scheduler`

These two strategies now share the same regular-GEMM universe, but not the same scheduler universe:

- `layered_exhaustive`: preserve the old layered exhaustive search standard
- `bruteforce_scheduler`: keep the old regular GEMM space, but widen BF16 scheduler candidates and force preflight-batch benchmark routing with scheduler-aware reporting

#### Base GEMM enumeration

Regular GEMM candidates are emitted by brute-force enumeration over the legal search domain from `default_constraints()` / `safe_search_constraints.json`:

- `tile_m` from allowed `tile_m`
- `tile_n` from allowed `tile_n`
- `tile_k` from allowed `tile_k`
- `sg_m` from allowed `sg_m`
- `sg_n` from allowed `sg_n`
- `stages` from allowed `stages`

Then the generator keeps only combinations that satisfy:

- `is_valid_xe2_tile_sg(tile_shape, sg_layout)`
- current constraint limits / blocked rules

This is a real legal-space enumeration, not a seed-neighbor expansion.

#### Scheduler enumeration (`StreamK`, `SplitK`, `DataParallel`)

Scheduler candidates are also emitted by explicit enumeration, not by “take top-k base GEMM and expand”.

Implementation contract:

1. iterate legal tile shapes from constraint space
2. require `tile_k >= 32`
3. for preserved `layered_exhaustive`, keep the benchmark-backed fixed scheduler baseline:
   - `sg = 8x4`
   - `stages = 2`
4. for `bruteforce_scheduler`, widen the scheduler axis to the currently legal BF16 scheduler domain:
   - `sg in {(2,8), (4,4), (4,8), (8,2), (8,4)}`
   - `stages in {1,2,3}`
5. emit three scheduler families for each legal scheduler point:
   - `streamk_mode = streamk`, `split_k = 1`
   - `streamk_mode = data_parallel`, `split_k = 1`
   - `streamk_mode = splitk`, `split_k = 1`
6. emit both `layout = rcr` and `layout = rrr`

Current benchmark-backed `StreamKScheduler` SplitK kernels only support the
decomposition-mode path (`split_k_slices <= 1`) on Xe. Wiring runtime
`split_k_slices = 2/3/4/6` to these benchmark kernels reuses the same compiled
kernel but hangs at runtime, so the profiler catalog intentionally keeps a
single benchmark-backed SplitK candidate per tile/subgroup/stage point.

So:

- `layered_exhaustive` keeps the old fixed-`8x4/st2` scheduler brute-force tile search
- `bruteforce_scheduler` performs **brute-force over the widened legal BF16 scheduler tile/subgroup/stage domain**

The fixed-path benchmark registry now matches the cataloged fixed-`8x4/st2` scheduler tile domain, including BF16 `RRR` scheduler variants. The widened `bruteforce_scheduler` path does not statically register every expanded scheduler kernel in `benchmarks_sycl.hpp`; instead it relies on filter-generated exhaustive scheduler headers enabled by `CUTLASS_BENCHMARK_EXHAUSTIVE_STREAMK`.

It is **not**:

- local expansion around top-k base kernels
- heuristic narrowing of the scheduler axis
- conditional expansion only after a base GEMM winner is known

The only filters are legality / safety constraints.

#### Current BF16/BF16/FP32 snapshot after the `RRR` scheduler fix

| layout / mode | candidates |
| --- | ---: |
| `rcr / Gemm` | 686 |
| `rcr / DataParallel` | 49 |
| `rcr / SplitK` | 196 |
| `rcr / StreamK` | 49 |
| `rrr / Gemm` | 686 |
| `rrr / DataParallel` | 32 |
| `rrr / SplitK` | 128 |
| `rrr / StreamK` | 32 |

This yields:

- total `Gemm = 1372`
- total `DataParallel = 81`
- total `SplitK = 324`
- total `StreamK = 81`
- total candidates = `1858`

## Phase 1 candidate domain

### Base domain

The candidate generator starts from the following base domain:

```text
tile_m   in [8, 16, 32, 64, 128, 256]
tile_n   in [64, 128, 256]
tile_k   in [32, 64]
sg_m     in [1, 2, 4, 8]
sg_n     in [4, 8]
stages   in [1, 2, 3]
split_k  in [1, 2, 4]
```

### Scheduler mapping

Scheduler is derived from `split_k`:

- `split_k = 1` -> `gemm`
- `split_k > 1` -> `split_k`

This keeps scheduler choice implicit in the candidate description instead of opening a separate axis.

## Candidate pruning rules

The base domain must be pruned before any codegen/build step.

### Rule 1: start from `RCR`

Phase 1 only emits candidates with:

- `layout = rcr`

This keeps the benchmark codegen and dispatch table aligned with the approved MVP.

### Rule 2: respect probe/default constraints

`safe_search_constraints.json` is authoritative.

If probe output exists, it overrides defaults.

If not, use `BMG_DEFAULT_CONSTRAINTS`.

### Rule 3: subgroup count cap

Phase 1 caps:

```text
sg_m * sg_n <= 32
```

Reason:

- legacy seed variants top out at `8x4 = 32`
- this avoids introducing `8x8 = 64` as a speculative first-pass candidate

### Rule 4: seed-aligned preferred SG pairs

The preferred subgroup pairs are:

- `(1, 4)`
- `(1, 8)`
- `(2, 4)`
- `(4, 4)`
- `(8, 4)`

`(2, 8)` and `(4, 8)` remain legal only if explicitly enabled by constraints.

Reason:

- the preferred set interpolates naturally from existing seed variants
- it keeps the first candidate wave closer to known-good layouts

### Rule 5: tile shape monotonicity

Reject:

```text
tile_n < 64
tile_m < 8
tile_k not in {32, 64}
```

These shapes are outside the approved base domain and do not map cleanly to the current seed patterns.

### Rule 6: estimated SLM must fit

Reject a candidate when:

```text
estimated_slm_kb > max_slm_kb
```

The exact estimator can evolve, but the filter contract is stable.

### Rule 7: block-copy blacklist

Reject any candidate that matches a `blocked_rules` entry from `safe_search_constraints.json`.

This is where Intel-specific `block_copy` instability is enforced.

### Rule 8: strategy-dependent scheduler expansion

The original MVP note gated `split_k` and recommended seed-adjacent expansion first.

That behavior still exists conceptually in the preserved legacy spaces:

- `baseline`
- `expanded_bmg`

But it is intentionally **not** used in:

- `layered_exhaustive`
- `bruteforce_scheduler`

For those two strategies, scheduler search is brute-force over the legal tile domain described above.

## Candidate classes

Candidate class is derived from tile scale and subgroup count.

### `small_tile`

Use when:

- `tile_m <= 16`
- `sg_m * sg_n <= 8`

Typical target:

- decode-style small `m`

### `medium_tile`

Use when:

- `32 <= tile_m <= 64`
- `sg_m * sg_n <= 16`

Typical target:

- prefill transition region

### `large_tile`

Use when:

- `tile_m >= 128` or `sg_m * sg_n >= 16`

Typical target:

- large dense GEMM

These classes are intentionally simple because they only need to support compiler-profile assignment in Phase 1.

## Candidate naming

The canonical `candidate_id` is:

```text
<layout>_<dtype_a><dtype_b><dtype_c>_tm<tile_m>_tn<tile_n>_tk<tile_k>_sg<sg_m>x<sg_n>_st<stages>_sk<split_k>
```

Examples:

- `rcr_bf16bf16f32_tm8_tn128_tk32_sg1x4_st2_sk1`
- `rcr_bf16bf16f32_tm256_tn256_tk32_sg8x4_st2_sk1`
- `rcr_bf16bf16f32_tm8_tn64_tk32_sg1x4_st2_sk4`

## Compiler profile binding

Compiler profile is not part of the outer search space.

Instead:

1. classify candidate as `small_tile`, `medium_tile`, or `large_tile`
2. select `compiler_profile_id` from `compiler_profiles.json`
3. store that binding in `gemm_candidate_space.json`

This avoids multiplying the search space by compiler combinations.

## Dispatch key

Phase 1 dispatch uses the exact shape key:

```text
(layout, dtype_a, dtype_b, dtype_c, dtype_acc, m, n, k)
```

No bucketing, padding, or range compression is applied in this version.

## Build-facing implications

This search space implies three practical codegen requirements:

1. the new benchmark path must be able to register multiple `RCR` candidates, not just the current single `RRR` tile
2. candidate registration must be generated from `gemm_candidate_space.json`
3. benchmark `.in` files should be generated from `(shape_id, candidate_id)` pairs instead of handwritten lists

## Recommended first-wave candidate matrix

To keep bring-up tractable, the first generated wave should prioritize these tiles:

| Class | Preferred tiles | Preferred SG layouts |
| --- | --- | --- |
| small | `8x64x32`, `8x128x32`, `16x64x32` | `1x4`, `1x8`, `2x4` |
| medium | `32x64x32`, `32x128x32`, `64x128x32` | `2x4`, `4x4` |
| large | `128x128x32`, `128x256x32`, `256x256x32` | `4x4`, `8x4` |

`stages` should start with `2`, then expand to `[1, 3]`.

`split_k` should start with `1`, then expand to `[2, 4]` on the gated shapes.

## Exit criteria for this search-space version

This search-space definition is considered sufficient for Phase 1 when:

1. `gemm_target_shapes.json` can represent decode and prefill starter sets
2. `gemm_candidate_space.json` can generate seed-adjacent `RCR` candidates
3. every candidate can be classified to a compiler profile without ambiguity
4. build-time pruning can happen before source generation
