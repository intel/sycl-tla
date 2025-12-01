# Xe SLM Pipelined GEMM
## Limitations of the L1 pipelined GEMM
* Current GEMM implementation is not well-performed. SLM pipelined GEMM maybe can get better performance.

## Goals

The goal of introduing Shared Local Memory ([SLM](https://www.intel.com/content/www/us/en/docs/oneapi/optimization-guide-gpu/2025-2/shared-local-memory.html#id-d26320e84)) into pipelined GEMM, In more detail, we want to:

* Fullly control of Shared local Memory
  - Programmers take over and optimize shared memory path.
  - Avoid competition between multiple work items.

* Avoid bank conflict with cutlass component
  - There is potential to use exsiting component (see `include/cute/swizzle.hpp`) in Cutlass.
  - Using SLM as Cache, higher parallelism

* Ease of gathering
  - L1 cache is non-deterministic
  - Gathering and reordering through controllable and high-bandwith memory

## (Pros) Register->SLM->Register is more efficient than L1 path
  - The key improvement is in the global -> register -> SLM part. The amount of data each thread needs to copy from global memory is smaller because threads collaborate to share the work of copying/reordering/dequantizing the data.
  - Subgroup specialization in each workgroup could avoid duplicated data tiles movemonet.
  - In the current mainloop, each subgroup copies the same data from L1 -> register and does any necessary reorder/data conversion/dequantization on that data.
  - To make an SLM flow efficient, we need heavy pipelining, typically triple or quadruple buffering in SLM.
  - We can also combine the SLM flow with prefetch to L1, though prefetch is much less crucial in this case.
  - Due to lack of asynchronous hardware features on Xe3p platform, programming model should be switched to Producer-Consumer pattern. Subgroups producers only copy from global memory to SLM.

## (Cons) Can not fullly utilize hardware engines in SLM pipelines
  - Type conversion/dequantization is expensive and can't be hidden behind dpas (or increases power). For instance, e4m3 -> f16/bf16 upconversion on BMG. We need an SLM pipeline to reach full performance.
  - Loading is expensive. Generally this happens whenever we can't use block 2D atoms. Sometimes the loads themselves are slow (e.g. non-4-byte-aligned loads), and sometimes there are so many individual loads needed that HW is unable to keep them all in flight.


## Implementation of SLM Pipelined GEMM 
  ### Provide high efficiency global memory to SLM interface
  * The copy operator from global memory to SLM can  be obtained by wrapping 2-D block(G->R) and the vectorized copy / 1-D instruction(R->SLM).
    - 2-D block instruction exists in Cutlass.
    - 1-D instruction or vectorized copy exist in Cutlass.
    - Reorder Shared local memory layout can avoid bank conflicts.
    - Meeting the alignment (32 bits) rule is a fundamental requirement for vectorized copy / 1-D instruction, which can be satisfied by tuning stages for various data types in most conditions.

  ### Producer-Consumer Programming Model
  * A kernel-level design pattern in which different subgroups are statically assigned to asymmetric roles:
    - Producers – issue asynchronous copy instructions (block 2d load / vectorized load) that move data from global memory into shared memory / registers.
    - Consumers – issue compute instructions (dpas, math calculation etc.) that read the freshly delivered data and produce results.
  * Key properties
    - Subgroups-specialised: only a subset of subgroups in a block act as producers; the rest are consumers.
    - Scale-out: the same kernel can run with 2 producer subgroups + 6 consumer subgroups, or 4 + 12, etc.;

<!-- ### Subgroup specialization
* `Subgroup specialization`originates from concept `Warp specialization` On NVIDIA Hopper.
  - Warp Groups (warpgroup) – 4 consecutive warps (128 threads) that act as one schedulable unit for the new matrix-multiply-and-accumulate instruction WGMMA.
  - Asynchronous special-purpose units – TMA (Tensor Memory Accelerator) for global→shared bulk copy and WGMMA for shared→register MMA; both run without stalling the issuing warp.
  - Role declaration + resource budget – you statically label warps as producer (TMA), consumer (WGMMA), reduction, etc., and you may re-allocate register file per warpgroup with the new PTX directive. -->
