# Generator and search-space gap analysis

## Files

- `python/cutlass_library/generator.py`
- `python/cutlass_library/gemm_operation.py`
- current Intel search tooling under `test/benchmarks/intel_gemm_profiler`

## Key correction

The Intel Xe generator is **not** missing all SG/workgroup expressiveness.

Today it already has:

- explicit `default_tiles_wg_sg`
- `SYCL_TLA_ADDITIONAL_TILE_SHAPES`
- explicit `TileDescription(..., sg_tile, ...)`

What it lacks is **systematic search-space expansion**, not basic expression power.

## Current Intel generation profile

- `GenerateIntelXe()` only enables `GenerateXe_TensorOp_16b_DPAS_gemm()`
- FP8, INT8, and mixed dtype helpers exist but are disabled at the unified entry
- Intel Xe 3.x emission currently uses `StageCountAuto`
- current call path uses `tile_schedulers=[TileSchedulerType.Persistent]`

## Real gap

Three spaces are not yet unified:

1. generated CUTLASS library space
2. benchmark-backed runnable space
3. current Python search/catalog space

This is why many generated ops do not translate into many currently searchable candidates.

## Design-heavy part

Compared with profiler porting, this is the area that really needs design:

- extra SG choices per tile
- explicit stage policies
- scheduler variants
- instantiation levels
- catalog/build-manifest alignment with benchmark execution
