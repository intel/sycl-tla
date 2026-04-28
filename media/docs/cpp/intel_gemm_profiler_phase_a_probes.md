# Intel GEMM Profiler Phase A Probes

## Goal

Phase A exists to answer one question before large-scale tuning starts:

> on this Intel GPU, with this compiler stack, which GEMM configurations are safe and worth searching?

Phase A does **not** try to find the best kernel for every shape.

Instead, it produces:

- `verified_hw_caps.json`
- `safe_search_constraints.json`
- `compiler_profiles.json`

These outputs bound the Phase 1 GEMM search space.

## Probe set

Phase A for the GEMM MVP contains six probes:

1. `hardware_inventory_probe`
2. `dpas_baseline_probe`
3. `block_copy_probe`
4. `slm_capacity_probe`
5. `occupancy_probe`
6. `compiler_flags_probe`
7. `prefetch_stages_probe`

The first probe is informational; the other six drive constraints.

## Probe execution rules

### Common environment

All probes should record:

- hostname / `node_id`
- GPU architecture
- compiler version
- driver/runtime version
- oneAPI environment
- proxy/bootstrap method used on the node

For remote non-interactive execution, commands should use:

```bash
bash -lc 'source /home/intel/.bashrc && <oneAPI env> && <cmd>'
```

### Common measurement rules

Unless a probe states otherwise:

- warmup count: `5`
- measured repetitions: `5`
- aggregate metric: `median`
- failures are recorded, not silently dropped

### Common representative tiles

Phase A should use three representative tile classes:

| Class | Tile | SG | Source |
| --- | --- | --- | --- |
| small | `8x128x32` | `1x4` | legacy `RCR_5` |
| medium | `64x128x32` | `4x4` | planned MVP candidate |
| large | `256x256x32` | `8x4` | legacy `RCR_6` |

These three anchors are reused by multiple probes so results stay comparable.

## Probe 1: `hardware_inventory_probe`

### Purpose

Collect static machine and toolchain identity.

### What it measures

- device name
- architecture label
- subgroup size
- driver/runtime version
- compiler version
- effective benchmark environment variables

### Output effect

This probe populates metadata in:

- `verified_hw_caps.json`

It does not directly generate pruning rules.

## Probe 2: `dpas_baseline_probe`

### Purpose

Confirm that the expected DPAS path is functional and establish a throughput baseline.

### What it measures

- kernel runs successfully with representative tiles
- baseline `avg_tflops`
- baseline runtime stability

### Input matrix

Recommended initial points:

- `(8, 128, 32), sg=(1,4)`
- `(64, 128, 32), sg=(4,4)`
- `(256, 256, 32), sg=(8,4)`

With:

- `layout = rcr`
- `dtype = bf16`
- `split_k = 1`
- `stages = 2`

### Output effect

This probe answers:

- whether the basic MMA path is healthy
- whether any representative tile is already unstable

If a representative tile cannot run, the associated class should be marked degraded in `verified_hw_caps.json`.

## Probe 3: `block_copy_probe`

### Purpose

Detect unstable or degraded block-copy shapes so Phase B can blacklist them.

### What it measures

- compile success
- run success
- correctness
- whether performance is abnormally degraded relative to peer tiles

### Input matrix

Use representative tile classes plus neighboring shapes:

- `8x64x32`
- `8x128x32`
- `16x64x32`
- `32x128x32`
- `64x128x32`
- `128x256x32`
- `256x256x32`

### Output effect

Results are transformed into:

- `blocked_rules` in `safe_search_constraints.json`

Rule shape should stay declarative, for example:

```json
{
  "rule_id": "block_copy.bad_16x256",
  "match": { "tile_m": 16, "tile_n": 256 }
}
```

## Probe 4: `slm_capacity_probe`

### Purpose

Find a practical SLM ceiling for GEMM candidates.

### What it measures

- compile success near SLM boundary
- runtime success near SLM boundary
- correctness near SLM boundary

### Input strategy

Sweep representative tiles while increasing:

- `tile_n`
- `stages`
- `split_k` only when needed for the test

The goal is not to model every candidate exactly, but to determine a safe upper bound such as:

```text
max_slm_kb = 96
```

or:

```text
max_slm_kb = 128
```

### Output effect

Writes:

- `limits.max_slm_kb` in `safe_search_constraints.json`

## Probe 5: `occupancy_probe`

### Purpose

Measure which subgroup/tile combinations collapse occupancy enough to stop being worth searching.

### What it measures

- runtime stability across subgroup layouts
- evidence of over-large SG grids
- relative penalty of large register footprints on small vs large tiles

### Input matrix

Recommended subgroup pairs:

- `(1,4)`
- `(1,8)`
- `(2,4)`
- `(4,4)`
- `(8,4)`
- optional stress cases: `(2,8)`, `(4,8)`

Across:

- small representative tile
- medium representative tile
- large representative tile

### Output effect

This probe drives:

- `allowed_values.sg_m`
- `allowed_values.sg_n`
- optional `blocked_rules`
- the Phase 1 `sg_m * sg_n <= 32` cap if stress cases prove unhelpful

## Probe 6: `compiler_flags_probe`

### Purpose

Determine the best compiler/runtime profile per candidate class.

### Why it is separate

Compiler options must not be multiplied into the Phase B outer search space.

They must be collapsed first into a small number of reusable profiles.

### Probe combinations

Minimum combinations:

| Combo | Meaning |
| --- | --- |
| `combo_a` | `256 GRF + perfmodel + Release` |
| `combo_b` | `256 GRF + no perfmodel + Release` |
| `combo_c` | `default GRF + Release` |
| `combo_d` | `256 GRF + perfmodel + O3` |

### Representative tiles

- small: `8x128x32`, `sg=1x4`
- medium: `64x128x32`, `sg=4x4`
- large: `256x256x32`, `sg=8x4`

### Output effect

Produces:

- `compiler_profiles.json`

And may also update:

- `verified_hw_caps.json` with evidence and winner margins

### Success criterion

If all three representative classes choose the same combo with small variance, Phase 1 may use a single global compiler profile.

If they diverge, keep one profile per class.

## Probe 7: `prefetch_stages_probe`

### Purpose

Determine the safe and useful range of `stages`.

### What it measures

- compile success
- correctness
- runtime benefit of `stages = 1, 2, 3`

### Input matrix

Use:

- small tile
- medium tile
- large tile

Across:

- `stages = 1`
- `stages = 2`
- `stages = 3`

### Output effect

Drives:

- `limits.max_stages`
- `allowed_values.stages`

If `stages = 3` is unstable or consistently slower, Phase 1 can clamp to `[1, 2]`.

## Probe output mapping

| Probe | `verified_hw_caps.json` | `safe_search_constraints.json` | `compiler_profiles.json` |
| --- | --- | --- | --- |
| `hardware_inventory_probe` | metadata | no | no |
| `dpas_baseline_probe` | yes | optional | no |
| `block_copy_probe` | yes | yes | no |
| `slm_capacity_probe` | yes | yes | no |
| `occupancy_probe` | yes | yes | optional |
| `compiler_flags_probe` | yes | no | yes |
| `prefetch_stages_probe` | yes | yes | no |

## Recommended execution order

Run probes in this order:

1. `hardware_inventory_probe`
2. `dpas_baseline_probe`
3. `compiler_flags_probe`
4. `slm_capacity_probe`
5. `occupancy_probe`
6. `block_copy_probe`
7. `prefetch_stages_probe`

Reason:

- hardware identity should be recorded first
- compiler profile choice affects later probe reproducibility
- SLM and occupancy bounds should be known before deeper shape expansion

## Default fallback behavior

Phase B must be able to proceed before Phase A fully completes.

If probe outputs are missing:

- use `constraint_source = default_bmg`
- use `BMG_DEFAULT_CONSTRAINTS`
- use `BMG_DEFAULT_COMPILER_PROFILE`

Once probes exist, `constraint_source` flips to `probe`.

## Exit criteria

Phase A design is considered complete when the following are true:

1. every probe has a stable `probe_id`
2. every probe has a clear output mapping into the schema files
3. representative tiles are fixed
4. compiler profile derivation does not leak into the Phase B outer search space
5. missing-probe fallback behavior is explicitly defined
