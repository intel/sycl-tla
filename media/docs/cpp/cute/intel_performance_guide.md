# Intel GPU Performance Tuning Guide for SYCL\*TLA

## Why tuning matters

Out-of-the-box tile sizes and pipeline depths may leave significant performance on the table.
Understanding whether a kernel is **bandwidth-bound** (memory throughput is the bottleneck) or
**compute-bound** (XMX throughput is the bottleneck) determines which knobs to turn first.

**Symptoms at a glance:**

| Symptom | Likely bottleneck | First action |
|---------|------------------|-------------|
| Low memory bandwidth utilization | Register/SLM staging mismatch | Check tile size vs cache line size |
| Low XMX utilization | Tile too small, pipeline too shallow | Increase tile M/N, add pipeline stage |
| Frequent barrier stalls | Over-synchronisation | Audit `barrier()` calls in epilogue |
| High register spill | Tile too large | Reduce tile M or tile N |

## Memory hierarchy overview

```
Global Memory (HBM)
    │
    │  2D block loads: XE_2D_*_LD_N / _LD_T / _LD_V
    ▼
Shared Local Memory (SLM)       ← optional staging; many Xe kernels skip SLM
    │                               and go directly Global → Register
    ▼
Registers (GRF)
    │
    │  XMX compute: XE_8x16x16_* atoms
    ▼
Compute
    │
    │  2D block stores: XE_2D_*_ST_N
    ▼
Global Memory (HBM)
```

Many SYCL\*TLA GEMM kernels on Intel Xe use **direct Global→Register** 2D block loads,
bypassing SLM entirely.  This is valid when the tile size fits in the GRF budget and avoids the
extra SLM round-trip.

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
- Each subgroup loads its own tile from global memory using `XE_2D_*_LD_*` operations.
- Pipeline stages are used instead (`PipelineStages ≥ 2`) to hide latency.

### Prefetch strategy

Intel Xe 2D block load traits expose a `PREFETCH` nested type.  Issue prefetches a few iterations
ahead of actual loads to hide HBM latency:

```cpp
// Example: prefetch A tile one iteration ahead
cute::copy(prefetchA, tAgA(_, _, _, k + 1), tAsA(_, _, _, (k + 1) % Stages));
```

The `PREFETCH` struct for a given copy trait (e.g., `XE_2D_U16x8x16_LD_N::PREFETCH`) issues a
non-blocking prefetch request.

### Tile size selection

Common tile sizes in this codebase:

| Data type | Typical tile shape (M × N × K) | Notes |
|-----------|-------------------------------|-------|
| BF16 / FP16 | `Shape<_256, _256, _32>` | Standard large tile for BMG and PVC |
| FP8 | `Shape<_256, _256, _32>` | Use with `F32F8F8F32` atom |
| INT8 | `Shape<_32, _128, _32>` | Mixed-precision; smaller tile is common |

**Rules of thumb:**
- Start with `(256, 256, 32)` for BF16 on BMG/PVC.
- Reduce M or N if register spill is observed (check with Intel VTune or compiler `-v` output).
- Increase K-depth for memory-bound kernels to amortize the 2D block load overhead.

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
| **Alignment** | `XE_2D_*` loads require the base pointer to be 64-byte aligned and the row stride to be a multiple of 16 elements. Unaligned access silently produces garbage. | Pad matrices to alignment boundaries. |
| **Over-synchronisation** | Inserting `barrier()` after every copy wastes throughput. The epilogue often only needs one barrier. | Audit barrier placement; consolidate where possible. |
| **Register pressure** | Large tiles (`512×512×32`) can exceed the 256-GRF budget per thread. The compiler will spill to SLM, hurting performance. | Reduce tile M or N; use `-cl-intel-256-GRF-per-thread` with awareness. |
| **VNNI format** | B-matrix loads for XMX must use `_LD_V` (VNNI-packed) layout. Using `_LD_N` for the B matrix causes wrong results or poor performance. | Use `XE_2D_U16x*x*_LD_V` for B-matrix loads. |

## Fast diagnosis — what to check first

1. **Bandwidth-bound or compute-bound?**
   Run with Intel VTune "GPU Hotspot" analysis.  Compare achieved memory bandwidth to HBM peak and
   achieved XMX TFLOPS to peak.

2. **Tile sizes appropriate for problem dimensions?**
   If M or N < tile size, many subgroups will be idle or padding-dominated.
   Consider a "residue" kernel or smaller tiles for non-multiple sizes.

3. **Pipeline depth sufficient?**
   If memory latency is high and XMX utilisation is low, increase `PipelineStages` by one and
   re-benchmark.

4. **Alignment verified?**
   Print or assert `reinterpret_cast<uintptr_t>(ptr) % 64 == 0` in a debug build.

5. **Subgroup size matches kernel attributes?**
   Mismatched subgroup sizes cause silent correctness issues on Xe.  Always set
   `sub_group_size<16>` explicitly.
