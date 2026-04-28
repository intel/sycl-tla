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

Phase 1 is intentionally narrow:

- GEMM only
- Intel SYCL path only
- exact-shape dispatch only
- `RCR` first-class target
- `bf16` and `f16` inputs with `f32` accumulation

Out of scope for this search-space version:

- grouped gemm
- attention
- quantized/blockscaled gemm
- layout bucketing
- heuristic online dispatch

## Current repository baseline

### What the new benchmark path currently exposes

`benchmarks/gemm/benchmarks_sycl.hpp` currently registers only one Intel benchmark:

- `RRR`
- tile `512x256x32`
- subgroup layout `8x4`

This is not enough for an Intel profiler MVP, because the target workflow is centered on `RCR` search for decode/prefill style shapes.

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

These seed variants are enough to define the first search bands even though the new benchmark path has not yet re-registered them.

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

### Rule 8: split-k gating

Phase 1 only emits `split_k > 1` candidates for large-`n` or large-`k` workloads.

Recommended initial gate:

```text
n >= 4096 or k >= 8192
```

Reason:

- keeps split-k from inflating the search space on small problems
- aligns with the intended use of the existing `SplitK_RCR_5` style kernels

### Rule 9: confirmation-first expansion

The first wave should prefer seed-adjacent tile shapes:

- small-`m`: `8x64x32`, `8x128x32`, `16x64x32`
- medium-`m`: `32x64x32`, `32x128x32`, `64x128x32`
- large-`m`: `128x128x32`, `128x256x32`, `256x256x32`

`tile_k = 64` should be treated as a second-wave expansion behind `tile_k = 32`.

Reason:

- the existing repository assets skew heavily toward `tile_k = 32`
- this reduces build cost during early bring-up

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
