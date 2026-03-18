# Intel GPU Performance Tuning Guide for SYCL\*TLA

> **Prerequisites:** This guide assumes familiarity with CuTe concepts
> ([intel_overview.md](intel_overview.md)) and the GEMM tutorial
> ([0x_gemm_tutorial.md](0x_gemm_tutorial.md)).
> For SYCL-specific execution model details (kernel launch, copy atom wiring, epilogue patterns),
> see the [Intel SYCL GEMM Companion](intel_gemm_companion.md).

## Why tuning matters

Out-of-the-box tile sizes and pipeline depths may leave significant performance on the table.
Understanding whether a kernel is **bandwidth-bound** (memory throughput is the bottleneck) or
**compute-bound** (XMX throughput is the bottleneck) determines which knobs to turn first.

**Symptoms at a glance:**

| Symptom | Likely bottleneck | First action |
|---------|------------------|-------------|
| Low memory bandwidth utilization | Register/SLM staging mismatch | Check tile size vs cache line size |
| Low XMX utilization | Tile too small, pipeline too shallow | Increase tile M/N, add pipeline stage |
| Frequent barrier stalls | Over-synchronization | Audit `barrier()` calls in epilogue |
| High register spill | Tile too large | Reduce tile M or tile N |

## Memory hierarchy overview

```
Global Memory (HBM)
    │
    │  2D block loads: XE_LOAD_2D / XE_LOAD_2D_TRANSPOSE / XE_LOAD_2D_VNNI
    ▼
Shared Local Memory (SLM)       ← optional staging; many Xe kernels skip SLM
    │                               and go directly Global → Register
    ▼
Registers (GRF)
    │
    │  XMX compute: XE_DPAS_TT atoms
    ▼
Compute
    │
    │  2D block stores: XE_STORE_2D
    ▼
Global Memory (HBM)
```

Many SYCL\*TLA GEMM kernels on Intel Xe use **direct Global→Register** 2D block loads,
bypassing SLM entirely.  This is valid when the tile size fits in the GRF budget and avoids the
extra SLM round-trip.

> **Note:** For CUTLASS-level automatic configuration (auto-selecting copy atoms, tile sizes, and
dispatch policies), see `CollectiveBuilder` in
> `examples/01_bmg_gemm_with_collective_builder/`. The rest of this guide focuses on
> **CuTe-level manual tuning** for cases where `CollectiveBuilder` does not fit your needs.

## Optimization strategies

### Subgroup sizing

Intel Xe uses **16-wide subgroups**.  The dispatch policy `IntelXeXMX16` sets `SubgroupSize = 16`.
Mismatching the subgroup size in the kernel attributes and the `TiledMMA` construction leads to
silent correctness failures.

**Checklist:**
- [ ] Verify `sycl::ext::oneapi::experimental::sub_group_size<16>` is set in the kernel properties.
- [ ] Verify `TiledMMAHelper` is instantiated with the correct `SubgroupSize`.

### SLM usage

SLM (Shared Local Memory) is optional for many Xe GEMM kernels because 2D block loads can stream
data directly into registers.

**Use SLM when:**
- The A or B tile does not fit in a single 2D block load operation.
- Multiple subgroups need to share the same loaded tile.

**Skip SLM when:**
- Each subgroup loads its own tile from global memory using `XE_LOAD_2D` / `XE_LOAD_2D_TRANSPOSE` / `XE_LOAD_2D_VNNI` operations.
- Pipeline stages are used instead (`PipelineStages ≥ 2`) to hide latency.

### Prefetch strategy

Intel Xe 2D block load traits expose a `PREFETCH` nested type.  Issue prefetches a few iterations
ahead of actual loads to hide HBM latency:

```cpp
// Example: prefetch A tile one iteration ahead
cute::copy(prefetchA, tAgA(_, _, _, k + 1), tAsA(_, _, _, (k + 1) % Stages));
```

The `PREFETCH` struct for a given copy trait (e.g., `XE_LOAD_2D<16, 8, 16>::PREFETCH`) issues a
non-blocking prefetch request.

### Tile size selection

Common tile sizes in this codebase:

| Data type | Typical tile shape (M × N × K) | Notes |
|-----------|-------------------------------|-------|
| BF16 / FP16 | `Shape<_256, _256, _32>` | Standard large tile for BMG and PVC |
| INT8 | `Shape<_32, _128, _32>` | Mixed-precision; smaller tile is common |

**Rules of thumb:**
- Start with `(256, 256, 32)` for BF16 on BMG/PVC.
- Reduce M or N if register spill is observed (check with Intel VTune or compiler `-v` output).
- Increase K-depth for memory-bound kernels to amortize the 2D block load overhead.

#### Real-world example: Flash Attention BF16 tile-size tuning on Intel Xe BMG

An improvement to the BF16 Flash Attention prefill kernel on Intel Arc BMG demonstrated that
doubling the K-tile in the QK GEMM stage meaningfully improves performance by amortizing 2D block
load overhead over more XMX compute.

**Before** (conservative K-tile = 32):

```cpp
// HEAD_DIM = 64 or 128
using ShapeQK = Shape<_128, _64, _32>;      // K-tile is 32 elements wide
using ShapePV = Shape<_128, _32, _64>;
using SubgroupLayout = Layout<Shape<_8, _1, _1>>;
```

**After** (doubled K-tile = 64, from `examples/06_bmg_flash_attention/06_bmg_prefill_attention.cpp`):

```cpp
// HEAD_DIM = 64
using ShapeQK = Shape<_128, _64, _64>;      // K-tile doubled to 64 elements
using ShapePV = Shape<_128, _32, _64>;
using ShapeOutPut = Shape<_128, _64, _64>;
using SubgroupLayout = Layout<Shape<_8, _1, _1>, Stride<_1, _1, _1>>;

// HEAD_DIM = 128
using ShapeQK = Shape<_128, _64, _64>;      // K-tile doubled to 64 elements
using ShapePV = Shape<_128, _32, _64>;
using ShapeOutPut = Shape<_128, _128, _64>;
using SubgroupLayout = Layout<Shape<_16, _1, _1>, Stride<_1, _1, _1>>;
```

**Why it helped:**  Each `XE_LOAD_2D_TRANSPOSE` / `XE_LOAD_2D_VNNI` 2D block load carries a fixed issue
overhead.  With K-tile = 32, loads were issued frequently relative to the XMX work they fed.
Doubling to K-tile = 64 halves the number of block loads per K-loop iteration, giving XMX more
sustained work per memory transaction and improving bandwidth utilization.

**When to apply this pattern:**
- The kernel is memory-bound (XMX utilization is low relative to bandwidth utilization).
- The head dimension (or K extent) is large enough that the larger K-tile does not overflow the
  GRF budget (verify with Intel VTune or `-v` compiler output; register spill negates the gain).
- The K dimension of the problem is a multiple of the new tile size.

### Pipeline stages

`PipelineStages = 2` is the standard starting point.  It overlaps one iteration of loads with the
compute of the previous iteration.

```cpp
static constexpr int PipelineStages = 2;
```

Increasing to 3 or 4 can help on high-latency HBM systems, but raises register pressure.

## Common pitfalls

| Pitfall | Description | Fix |
|---------|-------------|-----|
| **Alignment** | `XE_LOAD_2D` / `XE_STORE_2D` operations require the base pointer to be 64-byte aligned and the row stride to be a multiple of 16 elements. Unaligned access silently produces garbage. | Pad matrices to alignment boundaries. |
| **Over-synchronization** | Inserting `barrier()` after every copy wastes throughput. The epilogue often only needs one barrier. | Audit barrier placement; consolidate where possible. |
| **Register pressure** | Large tiles (`512×512×32`) can exceed the 256-GRF budget per thread. The compiler will spill to SLM, hurting performance. | Reduce tile M or N; use `-cl-intel-256-GRF-per-thread` with awareness. |
| **VNNI format** | B-matrix loads for XMX must use VNNI-packed layout. Using a non-VNNI load for the B matrix causes wrong results or poor performance. | Use `XE_LOAD_2D_VNNI<Bits, H, W>` for B-matrix loads. |

## Fast diagnosis — what to check first

1. **Bandwidth-bound or compute-bound?**
   Run with Intel VTune "GPU Hotspot" analysis.  Compare achieved memory bandwidth to HBM peak and
   achieved XMX TFLOPS to peak.

2. **Tile sizes appropriate for problem dimensions?**
   If M or N < tile size, many subgroups will be idle or padding-dominated.
   Consider a "residue" kernel or smaller tiles for non-multiple sizes.

3. **Pipeline depth sufficient?**
   If memory latency is high and XMX utilization is low, increase `PipelineStages` by one and
   re-benchmark.

4. **Alignment verified?**
   Print or assert `reinterpret_cast<uintptr_t>(ptr) % 64 == 0` in a debug build.

5. **Subgroup size matches kernel attributes?**
   Mismatched subgroup sizes cause silent correctness issues on Xe.  Always set
   `sub_group_size<16>` explicitly.
