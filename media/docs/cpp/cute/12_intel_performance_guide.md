# Intel GPU Performance Tuning Guide for CuTe on Xe

> **Prerequisites:** This guide assumes you are writing or modifying a CuTe mainloop kernel
> for Intel Xe GPUs.  Read [10_intel_overview.md](10_intel_overview.md) for the API map, and
> [11_intel_gemm_companion.md](11_intel_gemm_companion.md) for a step-by-step GEMM walkthrough
> before diving into tuning.
>
> **Note:** This guide focuses on **CuTe-level manual tuning** — choosing tile sizes, pipeline
> depths, and copy/MMA atoms directly.  For a complete CuTe GEMM walkthrough, see
> [11_intel_gemm_companion.md](11_intel_gemm_companion.md).

---

## Key reasons for tuning

- **Bandwidth vs compute utilization.**
  Intel Xe GEMM kernels can be limited by either global memory bandwidth or XMX throughput.
  Identifying which bottleneck dominates tells you whether to increase tile size (more compute
  per load) or add prefetch depth (hide memory latency).

- **Register (GRF) pressure.**
  Each Xe thread has 256 general-purpose registers.  Larger tiles consume more GRF;
  the compiler spills to SLM when the budget is exceeded, erasing the benefit of larger tiles.

- **Subgroup utilization.**
  With 16-wide subgroups and up to 32 subgroups per work-group, the tile and subgroup layout
  determine how much work each subgroup does and how many subgroups stay busy.

- **Prefetch depth.**
  2D block loads have issue overhead.  Prefetching future K-blocks while the current block
  computes on XMX hides global memory latency.  Too many stages waste GRF; too few leave
  XMX idle.

- **Alignment.**
  2D block load/store hardware has strict alignment and size requirements.  Violating them
  produces silently wrong results.

---

## Data flow

Most SYCL\*TLA GEMM kernels on Intel Xe bypass Shared Local Memory (SLM) and stream data
directly from global memory into registers via hardware 2D block loads.  The typical per-iteration
data flow is:

```
Global Memory (GDDR6 on BMG / HBM2e on PVC)
    │
    │  XE_LOAD_2D_TRANSPOSE (A), XE_LOAD_2D_VNNI (B)
    │  ─── copy fragments (tArA, tBrB) land in GRF ───
    ▼
Registers (GRF)
    │
    │  reorder(tArA → tCrA),  reorder(tBrB → tCrB)
    │  ─── shuffle data from copy layout to MMA layout ───
    ▼
XMX Compute
    │
    │  cute::gemm(tiled_mma, tCrA, tCrB, accum)
    │  ─── DPAS: D[M×16] = A[M×K] × B[K×16] + C[M×16] ───
    ▼
Accumulators (GRF)
    │
    │  Epilogue (e.g. LinearCombination: D = α·accum + β·C)
    │  XE_STORE_2D
    ▼
Global Memory
```

Running in parallel with the compute path, the prefetch path issues non-blocking hints
`PipelineStages` K-blocks ahead:

```
Global Memory ──[XE_PREFETCH_2D]──► L1/L2 cache   (no register allocation)
```

> **SLM note:** SLM staging is needed only when multiple subgroups share the same loaded tile
> or when a tile exceeds the single 2D block load limit.  For the standard
> `Shape<_256, _256, _32>` BF16 GEMM, each subgroup loads its own tile and SLM is skipped.

---

## The mainloop K-loop

The reference mainloop implementation lives in
`examples/cute/tutorial/xe_gemm.cpp`.  Understanding this loop is essential for
tuning.  Below is the annotated structure:

```cpp
// ── Setup ──────────────────────────────────────────────────────────
// Build TiledCopy objects from the global tensor (auto-selects copy atom).
auto copy_a = get_block_2d_copy_A<GmemTiledCopyA>(TiledMma{}, gA_batch);
auto copy_b = get_block_2d_copy_B<GmemTiledCopyB>(TiledMma{}, gB_batch);

// Allocate register fragments: separate sets for copy and MMA layouts.
auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));   // MMA layout
auto tArA = thr_copy_a.partition_sg_fragment_D(gA(_,_,0)); // copy layout

// Build prefetch TiledCopy (no register output).
auto prefetch_a = make_block_2d_prefetch(copy_a);

// ── Warmup: prefetch the first PipelineStages K-blocks ────────────
for (; prefetch_k < DispatchPolicy::Stages; prefetch_k++) {
    prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
    prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
}

// ── Main K-loop ───────────────────────────────────────────────────
for (int k_tile = 0; k_tile < k_tile_count; k_tile++, prefetch_k++) {
    barrier_arrive(barrier_scope);            // split barrier — first half

    copy(copy_a, tAgA(_,_,_,k_tile), tArA);  // 2D block load A → copy fragment
    copy(copy_b, tBgB(_,_,_,k_tile), tBrB);  // 2D block load B → copy fragment

    if (prefetch_k < k_tile_count) {          // guard: don't prefetch past end
        prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));
        prefetch(prefetch_b, pBgB(_, _, _, prefetch_k));
    }

    reorder(tArA, tCrA);                      // copy layout → MMA layout for A
    reorder(tBrB, tCrB);                      // copy layout → MMA layout for B

    cute::gemm(tiled_mma, tCrA, tCrB, accum); // XMX DPAS
    barrier_wait(barrier_scope);              // split barrier — second half
}
```

**Key observations for tuning:**

1. **`reorder` is not free.** It shuffles data between copy and MMA fragment layouts.  If
   the copy atom's data layout already matches the MMA atom's expected layout, the compiler
   removes the `reorder` entirely.  Otherwise it becomes a register-to-register shuffle.

2. **Split barriers (`barrier_arrive` / `barrier_wait`)** bracket each iteration.  The
   scope parameter (`barrier_scope = 2`) covers the work-group.  Adding or removing
   barriers changes synchronization granularity — only audit these if you modify the loop
   structure.

3. **Prefetch distance** equals `DispatchPolicy::Stages`.  The warmup loop fills the first
   `Stages` slots before the first compute iteration begins.

---

## Tuning knobs

### 1. Subgroup sizing

Intel Xe always uses **16-wide subgroups**.  The subgroup size must be set to 16 in all kernel configurations.

**Checklist:**
- [ ] `sycl::ext::oneapi::experimental::sub_group_size<16>` is set in kernel properties.
- [ ] `TiledMMAHelper` subgroup layout shape is consistent with 16-wide lanes.

A mismatch causes **silent correctness failures** — the kernel computes but produces wrong
results.

### 2. Tile size selection

The workgroup tile `Shape<M, N, K>` is the most impactful tuning knob.

**BF16 starting point** (from
[`examples/00_bmg_gemm/00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp)):

```cpp
using TileShape = Shape<_256, _256, _32>;

// 8×4 subgroup layout → 32 subgroups per work-group.
// Each subgroup handles a contiguous 32×64×32 chunk (4×4×2 DPAS iterations).
using TiledMma = typename TiledMMAHelper<
    MMA_Atom<XE_DPAS_TT<8, float, cute::bfloat16_t>>,
    Layout<TileShape>,
    Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>
>::TiledMMA;
```

**Rules of thumb:**

| Knob | Effect of increasing | Watch out for |
|------|---------------------|---------------|
| **M, N** | More compute per work-group → better XMX utilization | GRF spill if tile exceeds register budget; fewer concurrent work-groups |
| **K** | Fewer 2D block load issues per K-loop → amortizes load overhead | GRF spill from larger copy fragments; K must be a multiple of DPAS K dimension |
| **Subgroup count** | More subgroups per work-group → better occupancy | Each subgroup's tile shrinks; diminishing returns past 32 |

**Standard tile sizes in SYCL\*TLA:**

| Data type | Tile shape (M × N × K) | Subgroup layout | Notes |
|-----------|------------------------|-----------------|-------|
| BF16 / FP16 | `Shape<_256, _256, _32>` | `Shape<_8, _4, _1>` | 32 subgroups, standard large tile |
| TF32 | `Shape<_256, _256, _16>` | `Shape<_8, _4, _1>` | K=16 (TF32 DPAS K=8, 2 iterations) |

### 3. Pipeline stages (prefetch depth)

`PipelineStages` controls how many K-blocks ahead to prefetch.

A common default is `PipelineStages = 3` for standard GEMM and `2` for grouped GEMM.
Flash Attention Decode typically uses `PipelineStages = 1`.

> **Reference:** [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp)

```cpp
constexpr int PipelineStages = 2;  // 00_bmg_gemm starting point; try 3 for more latency hiding
```

**Tradeoff:** Each additional stage keeps one more K-block's worth of copy fragments live in
GRF.  Increasing from 2 → 3 can improve latency hiding but may trigger register spill.
Profile with [Intel PTI for GPU](https://github.com/intel/pti-gpu) or
[Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
to verify.

### 4. Prefetch strategy

Every 2D block load atom exposes a `PREFETCH` nested type:

```cpp
using Prefetch = XE_LOAD_2D<16, 8, 16>::PREFETCH;  // → XE_PREFETCH_2D<16, 8, 16>
```

The mainloop constructs a separate `TiledCopy` from this type — it issues a non-blocking hint
to the L1/L2 cache without consuming registers:

```cpp
auto prefetch_a = make_block_2d_prefetch(copy_a);
// ...
prefetch(prefetch_a, pAgA(_, _, _, prefetch_k));   // no destination argument
```

The warmup loop (before the first compute iteration) fills the prefetch pipeline, and
the guarded `if (prefetch_k < k_tile_count)` prevents out-of-bounds prefetches.

### 5. SLM usage

SLM is **not used** for the standard Xe GEMM mainloop — data goes directly
Global Memory → GRF via 2D block loads.

**Consider SLM only when all three conditions hold:**
1. **Register pressure is high** — the tile size or pipeline depth has exhausted the GRF
   budget, and spilling to SLM explicitly is cheaper than letting the compiler spill
   implicitly.
2. **The pipeline is written for multiple buffers** — the mainloop must double-buffer (or
   multi-buffer) SLM slots so that loads into one buffer can overlap with compute from
   another, avoiding stalls from barrier synchronization.
3. **Data dependencies are low** — subgroups reading the same SLM buffer must not have
   producer–consumer dependencies that force serialization; otherwise the extra barriers
   negate the benefit.

SLM may also be needed when a tile exceeds the maximum 2D block load size (see
[hardware constraints](#hardware-constraints) below) or when multiple subgroups must
share the same loaded data (e.g. broadcast B tile).

### 6. Reorder step

The `reorder(src, dst)` call shuffles data from the copy fragment's register layout to
the MMA fragment's expected layout.  This is a subgroup-scope register-to-register operation.

- If the copy and MMA layouts match, the compiler elides `reorder` entirely (no cost).
- If they don't match, `reorder` becomes a shuffle.  This is a necessary cost — it
  replaces the alternative of storing to SLM and reloading.
- See `include/cute/arch/reorder_xe.hpp` for the implementation.

---

## Hardware constraints

All constraints are enforced at compile time (`static_assert`) or at runtime when
`-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1` is passed to the build.

### 2D block operation limits

These constraints come from `include/cute/arch/copy_xe_2d.hpp` and
`include/cute/atom/copy_traits_xe_2d.hpp`.
All Width/Pitch values below are **in bytes** unless otherwise noted.

| Constraint | Value | Applies to |
|------------|-------|------------|
| Base pointer alignment | 64 bytes | All 2D ops |
| Pitch (stride) alignment | 16 bytes (4 bytes may work on some PVC configs) | All 2D ops |
| Width alignment | 4 bytes | All 2D ops |
| Max height (loads / prefetch) | 32 rows | `XE_LOAD_2D`, `XE_LOAD_2D_VNNI`, `XE_LOAD_2D_TRANSPOSE`, `XE_PREFETCH_2D` |
| **Max height (stores)** | **8 rows** | `XE_STORE_2D` — much more restrictive than loads |
| Max total width | 64 bytes (`Bits × Width ≤ 512`) | All 2D ops |
| Block count | 1, 2, or 4 (not 3) | `XE_LOAD_2D`, `XE_LOAD_2D_VNNI` |
| Bits per element | 8, 16, 32, or 64 | All 2D ops |
| Width/pitch/height max dimension | 2^24 | All 2D ops |

> **References:**
> [`include/cute/arch/copy_xe_2d.hpp`](../../../../include/cute/arch/copy_xe_2d.hpp),
> [`include/cute/atom/copy_traits_xe_2d.hpp`](../../../../include/cute/atom/copy_traits_xe_2d.hpp),
> [`xe_rearchitecture.md`](../xe_rearchitecture.md)

### Per-atom specific constraints

| Atom | Bits | Extra constraint |
|------|------|-----------------|
| `XE_LOAD_2D_VNNI` | **8 or 16 only** | For B-matrix VNNI-packed loads (BF16, FP16, INT8) |
| `XE_LOAD_2D_TRANSPOSE` | **32 or 64 only** | Width ≤ 8; if 64-bit: Height must be 8 and Width < 4 |
| `XE_STORE_2D` | 8/16/32/64 | Height ≤ 8 (vs ≤ 32 for loads) |

### Other hardware constraints

| Constraint | Value | Notes |
|------------|-------|-------|
| Subgroup size | **16** (always) | Must match `sub_group_size<16>` kernel attribute |
| GRF per thread | **256 registers** | Exceeding this causes compiler to spill to SLM |
| DPAS M range | 1–8 | `XE_DPAS_TT<M, ...>` |
| DPAS N | 16 (fixed) | All Intel Xe GPUs |
| DPAS K | Derived: `256 / max(sizeof_bits(TypeA), sizeof_bits(TypeB))` | BF16→16, TF32→8, INT8→32, INT4→64 |

### Enabling runtime alignment checks

Compile with `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1` to enable expensive runtime assertions
that check base pointer, width, and pitch alignment before every 2D block operation.  Use
this in debug builds to catch alignment issues early:

```bash
cmake .. -G Ninja \
  -DCUTLASS_ENABLE_SYCL=ON \
  -DDPCPP_SYCL_TARGET=intel_gpu_bmg_g21 \
  -DCMAKE_CXX_FLAGS="-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1"
```

---

## Tuning workflow

1. **Start with a known-good configuration.**
   Use `Shape<_256, _256, _32>` with `PipelineStages = 2` for BF16 on BMG
   (from [`00_bmg_gemm.cpp`](../../../../examples/00_bmg_gemm/00_bmg_gemm.cpp)).

2. **Profile.**
   Use [Intel PTI for GPU](https://github.com/intel/pti-gpu) (`unitrace --device-timing`)
   or [Intel VTune](https://www.intel.com/content/www/us/en/developer/tools/oneapi/vtune-profiler.html)
   GPU Hotspot analysis to determine if the kernel is bandwidth-bound or compute-bound.

3. **Adjust tile size.**
   - Bandwidth-bound → increase K to amortize 2D block load overhead.
   - Compute-bound → increase M or N to give XMX more work per iteration.
   - Watch for GRF spill (compiler `-v` output or PTI register metrics).

4. **Adjust prefetch depth.**
   Try `PipelineStages = 3`.  If register spill increases, revert to 2.

5. **Verify alignment.**
   Build with `-DCUTE_ENABLE_XE_BLOCK_2D_ASSERT=1` and run.  Fix any assertion failures
   by padding matrices or adjusting strides.

6. **Verify subgroup size.**
   Confirm `sub_group_size<16>` is set.  A mismatch produces wrong results with no error.

---

## Further reading

- [`xe_rearchitecture.md`](../xe_rearchitecture.md) — Xe CuTe architecture redesign (authoritative reference for atoms and data flow)
- [`10_intel_overview.md`](10_intel_overview.md) — Intel CuTe API map and terminology
- [`11_intel_gemm_companion.md`](11_intel_gemm_companion.md) — Step-by-step GEMM walkthrough
- [`xe_2d_copy.md`](xe_2d_copy.md) — Full 2D copy atom reference
- [`examples/00_bmg_gemm/`](../../../../examples/00_bmg_gemm/) — Reference GEMM example
