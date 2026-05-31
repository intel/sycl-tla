# Autotune Validation Plan

## Purpose
Validate autotune pipeline conclusions against 496-batch full screening ground truth.
All validations are pure statistical analysis — no additional screening required.

## Prerequisites
- 496-batch full screening completed (`results_full_clean/`)
- Unified CSV with kernel metadata (tile, sg, stages, scheduler, layout)

---

## V1: Phase 1 Top-10 Coverage

**Hypothesis**: Phase 1 (Gemm-scheduler-only, DP fixed) top-10 covers ≥80% of global top-10 (all schedulers).

**Method**:
1. From 496 ground truth, sort all kernels by TFLOPS descending
2. Take global top-10
3. Take top-10 from Gemm-scheduler-only subset
4. Compute overlap = |top-10(Gemm) ∩ top-10(global)| / 10

**Pass**: overlap ≥ 0.8

---

## V2: Scheduler Delta

**Hypothesis**: On 8192×4096×1536 with tiles ≥ 128×128, scheduler switch (DP↔StreamK↔SplitK) changes TFLOPS by <3%.

**Method**:
1. Find all (tile_m, tile_n, tile_k, sg_m, sg_n) where DP, StreamK, and SplitK all exist
2. For each config, compute: delta = max(DP, SK, SP) - min(DP, SK, SP)
3. Compute delta as % of max
4. Output histogram + mean/median delta

**Pass**: mean delta < 3%, median delta < 2%

---

## V3: Occupancy Bucket Prediction

**Hypothesis**: Medium occupancy bucket (tile_area/sg_product ∈ [512, 2048]) has highest mean and max TFLOPS.

**Method**:
1. Classify each kernel by occupancy proxy: tile_area / (sg_m × sg_n)
2. Bucket: small=[1,511], medium=[512,2047], large=[2048,8191], xlarge=[8192,∞)
3. Compute max + mean per bucket

**Pass**: medium bucket max > large bucket max AND medium bucket mean > all other bucket means

---

## V4: RRR vs RCR

**Hypothesis**: For identical compute config (same tile×sg×stages), RRR layout outperforms RCR.

**Method**:
1. Find all (tile_m, tile_n, tile_k, sg_m, sg_n) pairs that exist in both RRR and RCR
2. Compute delta = TFLOPS(RRR) - TFLOPS(RCR) for each pair
3. Output: mean delta, median delta, % of pairs where RRR wins

**Pass**: mean delta > 0 AND RRR win rate > 50%

---

## V5: Aggregate Validation Report

**Output**: `docs/autotune_validation.md`

| Section | Contains |
|---------|---------|
| V1 result | Overlap table + pass/fail |
| V2 result | Delta histogram + pass/fail |
| V3 result | Bucket table + pass/fail |
| V4 result | Paired delta stats + pass/fail |
| Summary | All pass/fail + recommendations |

## Dependencies

All validations blocked by: `496-batch screening completion`

```
496 screening
  ├── V1: validate-phase1-coverage
  ├── V2: validate-scheduler-delta
  ├── V3: validate-occupancy-bucket
  ├── V4: validate-rrr-vs-rcr
  └── V5: validate-full-report (depends on V1-V4)
```
