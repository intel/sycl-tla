# B70 GEMM Profiler — Presentation Outline (12 slides)

---

## Slide 1: Title
**B70 GEMM Auto-Profiling Pipeline on Intel Xe GPU (BMG G31)**

- 1,772 kernel configurations screened
- 154.9 TFLOPS peak (RRR layout)
- Fully automated catalog → compile → screen pipeline
- 886 batches, GPU 0–3 round-robin execution
- Problem: M=8192, N=4096, K=1536, BF16

---

## Slide 2: Profiler Architecture

```
┌──────────┐    ┌───────────┐    ┌──────────┐    ┌──────────┐
│ catalog  │───→│ gen_mini  │───→│  cmake   │───→│   GPU    │
│ .py      │    │ _hpp.py   │    │  + icpx  │    │  screen  │
│ enum     │    │ per-batch │    │  link    │    │          │
└──────────┘    └───────────┘    └──────────┘    └──────────┘
     ↓                ↓               ↓              ↓
 constraints   macro strip      perf flags       CSV output
 + ali shapes  + declare gen    baked at compile unified results
```

**Pipeline metrics:**
- 1,772 kernels → 886 batches (2 kernels/bin)
- 33s compile + 30s screen ≈ 1.5 min/batch
- Full run: ~22 hours sequential, ~7h with 4-GPU round-robin
- Result: single CSV with per-kernel TFLOPS + metadata

---

## Slide 3: Kernel Search Space

| Dimension | Values | Count |
|-----------|--------|-------|
| tile_m    | {8, 16, 32, 64, 128, 256, 512} | 7 |
| tile_n    | {32, 64, 96, 128, 192, 256, 512} | 7 |
| tile_k    | {16, 32, 64} | 3 |
| sg_m      | {1, 2, 4, 8} | 4 |
| sg_n      | {2, 4, 8} | 3 |
| stages    | {1, 2, 3} | 3 |
| scheduler | {Gemm, StreamK, DataParallel, SplitK} | 4 |
| layout    | {RCR, RRR} | 2 |

**Constraints:**
- B70 hardware max subgroup = 32 → sg_m × sg_n ∈ {16, 32}
- SG 8×8 invalid (product = 64 > 32)
- StreamK / DP / SplitK: SG fixed at 8×4

**BF16 coverage breakdown:**

| Type | RCR | RRR | Total |
|------|-----|-----|-------|
| Gemm_ SG exhaustive | 76 | 76 | 152 |
| GemmExhaustive (stages) | 684 | 684 | 1,368 |
| StreamK | 49 | 32 | 81 |
| DataParallel | 49 | 32 | 81 |
| SplitK | 49 | 32 | 81 |
| Hand-written | 9 | 2 | 11 |
| **Total** | | | **1,772** |

---

## Slide 4: Two-Phase Search Strategy

**Phase 1 — Compute Shape (Gemm scheduler fixed)**
- Search: tile_m × tile_n × tile_k × sg_m × sg_n × stages
- Goal: find most compute-efficient configurations
- Output: top-10 per occupancy bucket (not pure GFLOPS)

**Phase 2 — Execution Strategy (scheduler sweep)**
- Input: Phase 1 top-10 compute configs
- Search: scheduler × splits on fixed tile + SG
- Goal: find optimal scheduler per compute config
- Expected: ~10 configs × 4 schedulers × 3 splits ≈ 120 variants (vs. 886)

**Key data insight:**
- On large GEMM (8192×4096): all schedulers within 1% of each other
- Scheduler matters at K-skinny or small M/N problems
- Phase 2 enables reusing Phase 1 results across problem shapes

**Occupancy-bucket diversity (critical):**
- Large tile, low occupancy → favors DataParallel
- Small tile, high occupancy → favors StreamK / SplitK
- Retain top-N from MULTIPLE occupancy classes, not just top GFLOPS

---

## Slide 5: gen_mini_hpp.py — Per-Batch Kernel Generation

**Problem:** Full `benchmarks_sycl.hpp` = 835 lines, 300+ macro-generated types. Compile time > 30 minutes with all types. Each batch needs only 2 kernel types.

**Solution:** `gen_mini_hpp.py` strips preamble and generates batch-specific code.

**Final design (after 12+ iterations):**
1. Read full HPP → extract macro definitions + template types
2. **Strip** all `BMG_DECLARE_*`, `BMG_REGISTER_*`, `BMG_SOURCE_*` macro calls and `.def` includes
3. For each batch kernel, classify by regex pattern and generate:
   - Gemm_ SG: `BMG_DECLARE_GEMM_TILE_SG(PREFIX, CONFIG, ATOM, M,N,K,SGM,SGN)`
   - GemmExhaustive: `BMG_DECLARE_EXHAUSTIVE_GEMM_TILE_STAGE(...)`
   - StreamK / DP / SplitK: direct `GemmConfiguration<...>` + `CUTLASS_CREATE_GEMM_BENCHMARK`
4. Skip kernels already covered by `EXPANDED_STREAMK_TILES` or hand-written declarations
5. Override `register_gemm_benchmarks()` with batch-specific registrations

**Key pitfalls fixed:**
- Preamble strip prevents redefinition with 300+ macro-generated types
- `covered()` checks both `using Name =` and `CUTLASS_CREATE_GEMM_BENCHMARK`
- `EXPANDED_STREAMK_TILES` set must match full HPP individual calls (29 tiles)
- `.DELETE_ON_ERROR`: cmake deletes `.o` on link fail → `sed -i` to remove

**Performance impact of preamble stripping:**
- Before: 835 lines, IGC sees 300+ extra types → conservative optimization → 148 TFLOPS
- After: ~500 lines, only batch types → IGC aggressive optimization → **155 TFLOPS (+4-6%)**

---

## Slide 6: Build System

**Toolchain:**
- Intel oneAPI 2025.3 (icpx compiler)
- cmake 3.28 (Unix Makefiles generator)
- SYCL target: `spir64_gen -device bmg-g31`
- SPIR-V extensions: `+SPV_INTEL_split_barrier,+SPV_INTEL_2d_block_io,+SPV_INTEL_subgroup_matrix_multiply_accumulate`

**Critical compile flags (must be exported BEFORE make):**
```bash
export SYCL_PROGRAM_COMPILE_OPTIONS="-ze-opt-large-register-file -gline-tables-only"
export IGC_VectorAliasBBThreshold=10000
export IGC_ExtraOCLOptions="-cl-intel-256-GRF-per-thread"
```
Without these: **~4 TFLOPS**. With them: **154.9 TFLOPS**.

**Build flow per batch:**
```
gen_mini_hpp.py → benchmark_sycl.hpp (~500 lines)
       ↓
cmake make cutlass_benchmarks_gemm_sycl -j128  →  main.cpp.o (33s)
       ↓   (.DELETE_ON_ERROR disabled via sed)
icpx manual link  →  binary (~1s, 3–4 MB)
       ↓   (Google Benchmark stub: headers needed, library not)
ZE_AFFINITY_MASK=N ./binary --kernel=NAME --m=8192 --n=4096 --k=1536
```

**Google Benchmark handling:**
- `benchmark::State` type needed → headers required
- Library NOT needed (uses `run_direct` bypassing GB)
- `cmake/googlebenchmark_stub.cmake`: INTERFACE targets + ALIAS `benchmark::benchmark`
- cmake link fails (no `.a`) → manual `icpx` link with pre-built `libbenchmark.a`

---

## Slide 7: Runtime & Perf Optimization

**Scheduler types (all share identical CollectiveMma):**

| Scheduler | Kernel | Behavior | Best for |
|-----------|--------|----------|----------|
| Gemm | KernelXe | Static tile assignment | General |
| StreamK | KernelXeCooperative | Dynamic tile dispatch | K-skinny, small batch |
| DataParallel | KernelXeCooperative | Splits M×N grid | Large problems |
| SplitK | KernelXeCooperative | K slices + partial reduce | K-large, M/N-small |

**SplitK on B70 hardware:**
- `PersistentTileSchedulerXeStreamKParams` exposes: `DecompositionMode`, `splits`, `ReductionMode`
- `splits=1`: works reliably ✅ (verified via example 03 binary)
- `splits=2`: runtime hang ❌ (hardware limitation on current driver)
- Fixed: `GemmConfiguration::defaultArguments()` changed from `splits=2` → `splits=1`
- Fixed: `run_direct()` now calls `set_scheduler_splits()` (was only in `run()`)

**CPU & GPU frequency:**
```bash
# CPU: performance governor
for gov in /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor; do
  echo performance > $gov
done

# GPU frequency managed by Xe driver firmware (no userspace control on BMG G31)
```

---

## Slide 8: Top 20 Results (Problem: 8192×4096×1536, BF16)

| # | TFLOPS | Kernel | Layout |
|---|--------|--------|--------|
| 1 | **154.9** | Gemm_128×128×32_SG2×4 | RRR |
| 2 | 149.6 | Gemm_128×256×32_SG4×4 | RRR |
| 3 | 149.2 | Gemm_256×128×32_SG4×4 | RRR |
| 4 | 147.5 | Gemm_128×128×32_SG2×4 | RCR |
| 5 | 147.5 | Gemm_256×128×32_SG8×2 | RRR |
| 6 | 146.6 | Gemm_256×128×32_SG4×4 | RCR |
| 7 | 145.6 | GemmExhaustive_128×256×32_SG2×8_ST2 | RCR |
| 8 | 144.7 | GemmExhaustive_256×128×32_SG4×4_ST2 | RCR |
| 9 | 144.4 | GemmExhaustive_128×256×32_SG2×8_ST3 | RCR |
| 10 | 144.3 | DataParallel_256×256×32 | RCR |
| 11 | 144.2 | SplitK_256×256×32 | RCR |
| 12 | 143.5 | GemmExhaustive_256×128×32_SG4×4_ST1 | RCR |
| 13 | 143.2 | GemmExhaustive_256×128×32_SG4×4_ST3 | RCR |
| 14 | 143.2 | GemmExhaustive_64×256×64_SG2×8_ST1 | RCR |
| 15 | 143.2 | Gemm_64×128×32_SG1×8 | RCR |
| 16 | 142.3 | StreamK_256×256×32 | RCR |
| 17 | 142.2 | GemmExhaustive_128×256×32_SG2×8_ST1 | RCR |
| 18 | 142.1 | GemmExhaustive_256×128×32_SG8×2_ST1 | RCR |
| 19 | 140.9 | Gemm_128×128×64_SG4×4 | RCR |
| 20 | 140.5 | GemmExhaustive_256×128×32_SG8×2_ST2 | RCR |

**Statistics (992 kernels):** mean = 76.3, median = 68.9, top RRR = 154.9

---

## Slide 9: Scheduler Comparison — Same Tile, Same Problem

**Tile: 256×256×32, SG 8×4, Problem: 8192×4096×1536**

| Scheduler | TFLOPS | vs Gemm baseline |
|-----------|--------|-----------------|
| Gemm | 136.5 | — |
| StreamK | 142.3 | +4.2% |
| DataParallel | **144.3** | +5.7% |
| SplitK (splits=1) | **144.2** | +5.6% |

**Key finding:**
- On large GEMM, all schedulers perform within 1.4% of each other
- Scheduler impact is small when GPU is already saturated
- StreamK / SplitK benefits manifest at K-skinny or small-batch problems
- DataParallel has slight edge on large problems (simple, no reduction overhead)

**CPU overhead comparison:**
- Gemm: 0% extra CPU work
- DataParallel: low overhead (grid repartition only)
- StreamK: moderate overhead (dynamic work stealing)
- SplitK: highest overhead (reduction / atomic accumulate)

---

## Slide 10: RRR vs RCR Layout Comparison

**Identical compute configs, different memory layout:**

| Tile + SG | RCR TFLOPS | RRR TFLOPS | RRR Win |
|-----------|-----------|-----------|---------|
| 128×128×32 SG2×4 | 147.5 | **154.9** | +5.0% |
| 128×256×32 SG4×4 | 131.1 | **149.6** | +14.1% |
| 256×128×32 SG4×4 | 146.6 | **149.2** | +1.8% |

**Why RRR wins:**
- `B: RowMajor` → contiguous memory access in TiledMMA
- Better cache locality and register reuse
- Same DPAS atom, same computation, different data layout

**RRR is NOT always faster:**
- RCR benefits C-transposed access patterns
- Best layout depends on problem dimensions
- Current data: RRR wins on 8192×4096

---

## Slide 11: Build & Screen SOP

**One-command full screening:**
```bash
RESULTS_DIR=$WS/results_final_full BATCHES=all bash tools/run_seq.sh
```

**Pre-flight checklist:**
- [ ] `source /opt/intel/oneapi/compiler/2025.3/env/vars.sh`
- [ ] Export `SYCL_PROGRAM_COMPILE_OPTIONS`, `IGC_VectorAliasBBThreshold`, `IGC_ExtraOCLOptions`
- [ ] CPU governor = performance
- [ ] Restore `_deps/` (googlebenchmark + googletest headers)
- [ ] `touch compiler_depend.ts` in cmake build directory
- [ ] Verify `benchmark.h` exists: `ls _deps/googlebenchmark-src/include/benchmark/benchmark.h`

**Result directory convention (never mix runs):**

| Directory | Content |
|-----------|---------|
| `results_original/` | First screening (batches 0066–0545) |
| `results_rerun/` | Rerun of lost batches |
| `results_dp_only/` | DataParallel targeted test |
| `results_sk_dp_sp/` | StreamK + DP + SplitK test |
| `results_full_fixed/` | 496-batch run (old manifest) |
| `results_final_full/` | Current 886-batch run |

**Monitor:** `tail -f $WS/run_final2.log`

---

## Slide 12: Lessons Learned & Future Work

**Top pitfalls (and their fixes):**

| # | Pitfall | Impact | Fix |
|---|---------|--------|-----|
| 1 | Perf flags not baked at compile | 4.4→136 TFLOPS | Export before make |
| 2 | GB link failure | cmake error | Stub cmake + manual icpx link |
| 3 | .DELETE_ON_ERROR | .o lost every batch | `sed -i` to strip |
| 4 | gen_mini preamble stale macros | redefinition error | Strip all BMG_DECLARE_* |
| 5 | Python can't see C preprocessor output | redefinition | `covered()` via `using` text search + expanded tile sets |
| 6 | remote git pull with unstaged files | stale code | `git fetch && git reset --hard origin/main` |
| 7 | results dir deleted mid-run | data loss | Add `mkdir -p` in batch loop |

**Future work:**
- [ ] RRR StreamK/DP/SplitK data (886-batch running, ~20h ETA)
- [ ] Autotune cache: (M,N,K,dtype,arch) → best config
- [ ] Multi-problem screening (M/N/K sweeps)
- [ ] StreamK chunk/waves parameters (if exposed in driver)
- [ ] Occupancy-bucket top-N diversity for Phase 1 → Phase 2 bridge

**Code & data:** [github.com/tinafengfun/sycl-tla](https://github.com/tinafengfun/sycl-tla)
