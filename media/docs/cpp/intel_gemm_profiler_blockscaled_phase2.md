# Intel GEMM Profiler Phase 2: BlockScaledGemm Roadmap

## Goal

This document defines how BlockScaledGemm should extend the Intel GEMM profiler **after** the Phase 1 GEMM MVP is stable.

The main rule is simple:

> BlockScaledGemm is Phase 2 because it adds new data semantics, new correctness concerns, and new candidate dimensions that would dilute Phase 1.

## Why it is not part of Phase 1

Phase 1 is intentionally limited to dense GEMM with:

- standard floating-point inputs
- exact-shape dispatch
- one clear search loop

BlockScaledGemm changes three things at once:

1. **data model**
   - additional scale tensors
   - block granularity semantics

2. **correctness path**
   - scale handling must be validated together with GEMM output
   - reference comparison becomes more sensitive to quantization behavior

3. **search space**
   - candidate quality depends not only on tile shape and SG layout
   - scale/block movement and layout decisions enter the picture

That is too much surface area for the first vertical slice.

## Phase 2 entry condition

BlockScaledGemm work should start only after Phase 1 proves:

1. schema artifacts are stable
2. candidate generation works for dense GEMM
3. build/run/parser workflow is reliable
4. dispatch-table emission is already working

In other words, Phase 2 should reuse Phase 1 infrastructure instead of redesigning it.

## What can be reused from Phase 1

The following components should be reused almost unchanged:

- `safe_search_constraints.json`
- `compiler_profiles.json`
- candidate classification by tile class
- batch build workflow
- subprocess runner
- normalized CSV result handling
- top-k screening plus confirmation
- dispatch-table emission pattern

This reuse is the reason to build the dense GEMM MVP first.

## What must be extended

### 1. Shape schema

Phase 2 will likely need extra fields beyond `(layout, dtype, m, n, k)`.

Expected additions:

- block scale data type
- block shape
- scale layout

These should be added as an extension of the existing schema, not a parallel schema family.

### 2. Candidate schema

BlockScaledGemm candidates will likely need extra dimensions such as:

- scale block granularity
- scale load/store strategy
- scale tensor layout assumptions

These dimensions should remain Phase 2-only fields.

### 3. Correctness path

Dense GEMM Phase 1 can reuse the current reference path.

BlockScaledGemm Phase 2 will need:

- reference handling for scaled inputs
- tolerance rules that reflect quantized or scaled behavior
- explicit reporting of scale-related errors

### 4. Search-space pruning

The dense GEMM Phase 1 pruning rules are not enough.

Phase 2 will likely need new pruning rules around:

- scale tensor movement cost
- block-size legality
- memory pressure from scale metadata

## Recommended Phase 2 breakdown

### Phase 2A: schema extension

Deliverables:

- extend `intel_gemm_profiler_schemas.md`
- define BlockScaledGemm shape key additions
- define candidate fields specific to scaled kernels

### Phase 2B: correctness-first vertical slice

Deliverables:

- one small BlockScaledGemm target shape set
- verify-capable benchmark batch
- structured results in the same CSV/report pipeline

The first milestone is correctness, not peak performance.

### Phase 2C: search-space design

Deliverables:

- BlockScaledGemm candidate dimensions
- probe additions if new hardware constraints are needed
- pruning rules that keep candidate count bounded

### Phase 2D: dispatch extension

Deliverables:

- dispatch-table entries that include scaled-kernel metadata
- compatibility rules with dense GEMM dispatch output

## Recommended initial Phase 2 success case

The first BlockScaledGemm success case should be deliberately small:

- one layout family
- one scale format
- one or two representative shapes
- one working correctness path
- one dispatch-table winner

This keeps the extension incremental rather than reopening Phase 1 design questions.

## Risks

### Risk 1: schema fork

Creating a separate profiler flow for BlockScaledGemm would duplicate the Phase 1 pipeline and increase maintenance cost.

### Risk 2: correctness ambiguity

If scaled-kernel verification is underspecified, the search loop may select numerically invalid winners.

### Risk 3: search-space explosion

If scale/block dimensions are added directly to the dense GEMM candidate sweep without pruning, compile cost will become unmanageable.

## Guiding rule

BlockScaledGemm Phase 2 should be treated as:

> a schema-preserving extension of the dense GEMM profiler pipeline

not as a fresh profiler architecture.

## Exit criteria

This roadmap is sufficient when the team can answer:

1. why BlockScaledGemm is deferred
2. which Phase 1 artifacts will be reused
3. which new schema and correctness fields BlockScaledGemm introduces
4. how to start with one minimal scaled vertical slice instead of a full search explosion
