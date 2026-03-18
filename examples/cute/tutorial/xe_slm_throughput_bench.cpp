/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

//
// xe_slm_throughput_bench.cpp — SLM copy throughput microbenchmark
//
// Measures the throughput of the two SLM copy paths:
//   Variant 1 (UniversalCopy):  Regs→SLM→Regs   via make_A_slm_copies (compiler-vectorized dst=src)
//   Variant 2 (VISA):           Regs→SLM→Regs   via XE_1D_STSM / XE_1D_LDSM (explicit lsc_store/load.slm asm)
//
// Timing: SYCL event profiling (device-side, pure kernel execution)
// Clocks: Level Zero Sysman API (actual GPU frequency + throttle reasons)
//
// Usage:
//   ./xe_slm_throughput_bench [--iterations=100] [--inner_iters=100] [--num_wgs=64]
//

#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/backend/level_zero.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/util/command_line.h"

#include "../../common/sycl_cute_common.hpp"

// Device-side hardware clock intrinsic
#include <sycl/ext/oneapi/experimental/clock.hpp>

// Level Zero for GPU clock cycle measurement + Sysman for frequency monitoring
#include <level_zero/ze_api.h>
#include <level_zero/zes_api.h>

#include <thread>
#include <atomic>
#include <vector>
#include <algorithm>
#include <numeric>

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;

// ============================================================================
// GPU Frequency Monitor  (Level Zero Sysman + sysfs fallback)
// ============================================================================

struct GpuFreqMonitor {
  // Level Zero handles
  zes_device_handle_t  zes_device = nullptr;
  zes_freq_handle_t    freq_handle = nullptr;
  bool                 sysman_ok = false;

  // GPU timestamp clock frequency (cycles per second)
  uint64_t gpu_timer_freq_hz = 0;   // from ze_device_properties_t.timerResolution (v1.2 = cycles/s)
  uint32_t core_clock_mhz = 0;      // from ze_device_properties_t.coreClockRate

  // Sampling thread state
  std::atomic<bool>       running{false};
  std::thread             worker;
  std::vector<double>     freq_samples;    // actual MHz per sample
  std::vector<double>     req_samples;     // requested MHz per sample
  std::vector<uint32_t>   throttle_samples;

  // Initialise Sysman from a SYCL queue
  void init(sycl::queue &Q) {
    try {
      // Get the Level Zero device handle from the SYCL device
      auto ze_dev = sycl::get_native<sycl::backend::ext_oneapi_level_zero>(
                        Q.get_device());
      zes_device = reinterpret_cast<zes_device_handle_t>(ze_dev);

      // Query device properties (v1.2) for timer resolution in cycles/sec
      ze_device_properties_t devProps{};
      devProps.stype = ZE_STRUCTURE_TYPE_DEVICE_PROPERTIES_1_2;
      devProps.pNext = nullptr;
      if (zeDeviceGetProperties(ze_dev, &devProps) == ZE_RESULT_SUCCESS) {
        gpu_timer_freq_hz = devProps.timerResolution;  // cycles/sec with v1.2
        core_clock_mhz    = devProps.coreClockRate;
      }

      // Enumerate frequency domains and pick the first one (GPU core)
      uint32_t count = 0;
      if (zesDeviceEnumFrequencyDomains(zes_device, &count, nullptr) != ZE_RESULT_SUCCESS || count == 0) {
        printf("  [freq] Warning: no frequency domains found\n");
        return;
      }
      std::vector<zes_freq_handle_t> handles(count);
      zesDeviceEnumFrequencyDomains(zes_device, &count, handles.data());
      freq_handle = handles[0];  // first domain = GPU core clock
      sysman_ok = true;
    } catch (...) {
      printf("  [freq] Warning: Level Zero Sysman init failed — no frequency data\n");
    }
  }

  // Read current frequency state (call from any thread)
  bool read_one(double &actual_mhz, double &request_mhz, uint32_t &throttle) {
    if (!sysman_ok) return false;
    zes_freq_state_t state{};
    state.stype = ZES_STRUCTURE_TYPE_FREQ_STATE;
    state.pNext = nullptr;
    if (zesFrequencyGetState(freq_handle, &state) != ZE_RESULT_SUCCESS) return false;
    actual_mhz  = state.actual;
    request_mhz = state.request;
    throttle    = state.throttleReasons;
    return true;
  }

  // Start background sampling thread (~200 µs interval)
  void start() {
    freq_samples.clear();
    req_samples.clear();
    throttle_samples.clear();
    running = true;
    worker = std::thread([this]() {
      while (running.load(std::memory_order_relaxed)) {
        double act = 0, req = 0;
        uint32_t thr = 0;
        if (read_one(act, req, thr)) {
          if (act > 0) {             // ignore 0 = idle
            freq_samples.push_back(act);
            req_samples.push_back(req);
            throttle_samples.push_back(thr);
          }
        }
        std::this_thread::sleep_for(std::chrono::microseconds(200));
      }
    });
  }

  void stop() {
    running.store(false, std::memory_order_relaxed);
    if (worker.joinable()) worker.join();
  }

  static const char* throttle_flag_name(uint32_t bit) {
    switch (bit) {
      case ZES_FREQ_THROTTLE_REASON_FLAG_AVE_PWR_CAP:  return "AVE_PWR(PL1)";
      case ZES_FREQ_THROTTLE_REASON_FLAG_BURST_PWR_CAP:return "BURST_PWR(PL2)";
      case ZES_FREQ_THROTTLE_REASON_FLAG_CURRENT_LIMIT:return "CURRENT(PL4)";
      case ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL_LIMIT:return "THERMAL_LIMIT";
      case ZES_FREQ_THROTTLE_REASON_FLAG_PSU_ALERT:    return "PSU_ALERT";
      case ZES_FREQ_THROTTLE_REASON_FLAG_SW_RANGE:     return "SW_RANGE";
      case ZES_FREQ_THROTTLE_REASON_FLAG_HW_RANGE:     return "HW_RANGE";
      case ZES_FREQ_THROTTLE_REASON_FLAG_VOLTAGE:      return "VOLTAGE";
      case ZES_FREQ_THROTTLE_REASON_FLAG_THERMAL:      return "THERMAL";
      case ZES_FREQ_THROTTLE_REASON_FLAG_POWER:        return "POWER";
      default: return "UNKNOWN";
    }
  }

  void report() const {
    if (freq_samples.empty()) {
      printf("  GPU Freq : (no active samples)\n");
      return;
    }
    double mn  = *std::min_element(freq_samples.begin(), freq_samples.end());
    double mx  = *std::max_element(freq_samples.begin(), freq_samples.end());
    double avg = std::accumulate(freq_samples.begin(), freq_samples.end(), 0.0)
                 / double(freq_samples.size());

    printf("  GPU Freq : min=%.0f  max=%.0f  avg=%.0f MHz  (%zu samples)\n",
           mn, mx, avg, freq_samples.size());

    // Aggregate throttle reasons across all samples
    uint32_t all_throttle = 0;
    for (auto t : throttle_samples) all_throttle |= t;
    if (all_throttle == 0) {
      printf("  Throttle : none\n");
    } else {
      printf("  Throttle : ");
      bool first = true;
      for (int bit = 0; bit < 10; ++bit) {
        uint32_t flag = 1u << bit;
        if (all_throttle & flag) {
          if (!first) printf(" | ");
          printf("%s", throttle_flag_name(flag));
          first = false;
        }
      }
      printf("\n");
    }
  }

  // Convert nanoseconds to GPU timer ticks using the device timer frequency
  double ns_to_timer_ticks(double ns) const {
    if (gpu_timer_freq_hz == 0) return 0.0;
    return ns * double(gpu_timer_freq_hz) / 1e9;
  }

  // Convert nanoseconds to core GPU clock cycles using actual sampled frequency
  // (or coreClockRate if no samples available)
  double ns_to_core_clocks(double ns) const {
    double freq_mhz = 0;
    if (!freq_samples.empty()) {
      freq_mhz = std::accumulate(freq_samples.begin(), freq_samples.end(), 0.0)
                 / double(freq_samples.size());
    } else if (core_clock_mhz > 0) {
      freq_mhz = double(core_clock_mhz);
    }
    if (freq_mhz <= 0) return 0.0;
    return ns * freq_mhz / 1e3;  // ns * (MHz * 1e6) / 1e9 = ns * MHz / 1e3
  }

  void print_timer_info() const {
    if (gpu_timer_freq_hz > 0) {
      printf("  GPU Timer Freq : %lu Hz (%.2f MHz)\n",
             (unsigned long)gpu_timer_freq_hz, double(gpu_timer_freq_hz) / 1e6);
    }
    if (core_clock_mhz > 0) {
      printf("  Core Clock Rate: %u MHz\n", core_clock_mhz);
    }
  }
};

// ============================================================================
// Command-line options
// ============================================================================

struct Options {
  bool help  = false;
  bool error = false;
  int iterations  = 100;    // number of kernel launches for timing
  int inner_iters = 100;    // loop count inside each kernel launch
  int num_wgs     = 64;     // number of workgroups to launch

  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);
    if (cmd.check_cmd_line_flag("help")) { help = true; return; }
    cmd.get_cmd_line_argument("iterations",  iterations,  100);
    cmd.get_cmd_line_argument("inner_iters", inner_iters, 100);
    cmd.get_cmd_line_argument("num_wgs",     num_wgs,     64);
  }

  std::ostream& print_usage(std::ostream &out) const {
    out << "SLM Throughput Benchmark\n\n"
        << "Options:\n"
        << "  --help                Displays this message\n"
        << "  --iterations=<int>    Kernel launches for timing       (default 100)\n"
        << "  --inner_iters=<int>   Loop iterations inside kernel    (default 100)\n"
        << "  --num_wgs=<int>       Number of workgroups to launch   (default 64)\n\n";
    return out;
  }
};

// ============================================================================
// MMA setup — bf16 DPAS, 256×256 WG tile, 8×4 subgroup layout
// ============================================================================

static auto make_bench_mma() {
  auto op = XE_DPAS_TT<8, float, bfloat16_t>{};
  using WGTile   = Shape<_256, _256, C<decltype(op)::K * 2>>;
  using SGLayout  = Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>;
  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  return MMA{};
}

using BenchMMA = decltype(make_bench_mma());

using gpu_clock = sycl::ext::oneapi::experimental::clock_scope;
// BMG supports sub_group clock scope (not device/work_group)

// ============================================================================
// Variant 1: UniversalCopy path  (high-level make_A_slm_copies API)
// ============================================================================

template <class ATensor, class OutTensor>
void
slm_bench_universal_copy_device(ATensor   const& A,
                                 OutTensor      & out,
                                 BenchMMA  const& mma,
                                 int              inner_iters,
                                 uint64_t*        clock_buf)
{
  using TA = typename ATensor::element_type;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto local_id = int(item.get_local_id(0));

  // Coordinate tensors
  Tensor cA = make_identity_tensor(A.shape());     // (M, K)
  auto wg_tile = mma.tile_mnk();
  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _));  // (BLK_M, BLK_K, k)

  // -- Build copies --
  auto coop_copy_a  = make_coop_block_2d_copy_A(mma, A);
  auto coop_copy_a_ = make_coop_block_2d_copy_A(mma, make_tensor(A.data(), make_layout(shape(A), LayoutRight{})));
  auto [r2s_A, s2r_A] = make_A_slm_copies(mma, coop_copy_a);

  // SLM buffer (single stage — no double buffering needed for pure throughput test)
  Layout a_slm_layout = make_layout(typename decltype(r2s_A)::Tiler_MN{});
  auto smemA = compat::local_mem<TA[size(a_slm_layout)]>();
  Tensor sA = make_tensor(make_smem_ptr(smemA), a_slm_layout);

  // Slice to this thread
  auto thr_mma         = mma.get_slice(local_id);
  auto coop_thr_copy_a = coop_copy_a.get_slice(local_id);
  auto coop_thr_copy_a_ = coop_copy_a_.get_slice(local_id);
  auto thr_r2s_A       = r2s_A.get_slice(local_id);
  auto thr_s2r_A       = s2r_A.get_slice(local_id);

  // MMA register fragment (destination of s2r)
  auto tCrA = thr_mma.partition_sg_fragment_A(gA(_,_,0));

  // Copy register fragments
  auto tArA_in   = coop_thr_copy_a.partition_sg_fragment_D(gA(_,_,0));
  auto tArA_in_  = coop_thr_copy_a_.partition_sg_fragment_D(gA(_,_,0));

  auto tAsA_out  = thr_r2s_A.partition_D(sA);
  auto tArA_out  = thr_r2s_A.retile_S(tArA_in_);

  auto tAsA_in   = thr_s2r_A.partition_S(sA);
  auto tCrA_in   = thr_s2r_A.retile_D(tCrA);

  // Accumulator (used as a sink to prevent DCE)
  float accum = 0.f;

  // NO global memory access — register fragments contain uninitialized data.
  // Content is irrelevant for throughput measurement; only SLM bandwidth
  // matters and the full copy chain is kept alive by the accum DCE sink.

  // Barrier to synchronize all subgroups before starting the inner loop
  barrier_arrive(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
  barrier_wait(SPIRVScope::ScopeWorkgroup,
               SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

  // Read hardware GPU clock BEFORE inner loop (sub_group scope = HW timestamp register)
  uint64_t t_start = sycl::ext::oneapi::experimental::clock<gpu_clock::sub_group>();

  // Inner loop — measures ONLY SLM <-> Register throughput
  for (int i = 0; i < inner_iters; ++i) {
    // Registers → SLM  (UniversalCopy: dst = src)
    copy(r2s_A, tArA_out, tAsA_out);

    // Barrier
    barrier_arrive(SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

    // SLM → Registers  (UniversalCopy: dst = src)
    copy(s2r_A, tAsA_in, tCrA_in);

    // Barrier
    barrier_arrive(SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

    // Lightweight sink to prevent dead-code elimination
    accum += float(tCrA(0));
  }

  // Read hardware GPU clock AFTER inner loop
  uint64_t t_end = sycl::ext::oneapi::experimental::clock<gpu_clock::sub_group>();

  // Write one value per WG to global memory to prevent full DCE
  if (local_id == 0) {
    out(wg_m) = TA(accum);
    // Store measured clock ticks: [wg*2] = start, [wg*2+1] = end
    clock_buf[wg_m * 2]     = t_start;
    clock_buf[wg_m * 2 + 1] = t_end;
  }
}

// ============================================================================
// Variant 2: Explicit VISA path  (XE_1D_STSM / XE_1D_LDSM atoms)
//
// Each work-item uses lsc_store.slm / lsc_load.slm to write/read its own
// stripe of SLM.  The stripe assignment is:
//   work-item i within its subgroup (lane) accesses: base + lane * 16 bytes
//   where base advances for each subgroup and each iteration chunk.
// This mirrors how the 1D SLM atoms work internally (see copy_xe.hpp).
// ============================================================================

template <class ATensor, class OutTensor>
void
slm_bench_visa_device(ATensor   const& A,
                       OutTensor      & out,
                       BenchMMA  const& mma,
                       int              inner_iters,
                       uint64_t*        clock_buf)
{
  using TA = typename ATensor::element_type;

  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m = int(item.get_group(1));
  auto local_id = int(item.get_local_id(0));

  // Coordinate tensors
  Tensor cA = make_identity_tensor(A.shape());
  auto wg_tile = mma.tile_mnk();
  Tensor gA = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _));

  // -- SLM buffer: flat contiguous layout sized for the tile --
  constexpr int tile_M = get<0>(BenchMMA{}.tile_mnk());
  constexpr int tile_K = get<2>(BenchMMA{}.tile_mnk());
  constexpr int SLM_ELEMS = tile_M * tile_K;
  // Number of subgroups per workgroup
  constexpr int NUM_SGS = 32;      // 8x4 SG layout
  constexpr int SG_SIZE = 16;
  // Elements per work-item per 1D atom access (uint128_t = 16 bytes = 8 bf16)
  constexpr int VEC_ELEMS = int(sizeof(cutlass::uint128_t) / sizeof(TA));
  // Elements per subgroup per 1D access = SG_SIZE * VEC_ELEMS = 16 * 8 = 128
  constexpr int ELEMS_PER_SG = SG_SIZE * VEC_ELEMS;
  // How many 1D-atom "chunks" each subgroup must do to cover SLM_ELEMS / NUM_SGS
  constexpr int ELEMS_PER_SG_TOTAL = SLM_ELEMS / NUM_SGS;
  constexpr int CHUNKS_PER_SG = ELEMS_PER_SG_TOTAL / ELEMS_PER_SG;

  auto smem = compat::local_mem<TA[SLM_ELEMS]>();

  // Each subgroup gets a contiguous stripe of SLM.
  // Subgroup index within the workgroup:
  int sg_idx = local_id / SG_SIZE;
  int lane   = local_id % SG_SIZE;

  // Build VISA-atom copies for a single-subgroup chunk
  using Element_uint = typename uint_bit<sizeof_bits_v<TA>>::type;

  using traits_stsm = Copy_Traits<XE_1D_STSM<cutlass::uint128_t, Element_uint>>;
  using Atom_stsm   = Copy_Atom<traits_stsm, TA>;

  using traits_ldsm = Copy_Traits<XE_1D_LDSM<Element_uint, cutlass::uint128_t>>;
  using Atom_ldsm   = Copy_Atom<traits_ldsm, TA>;

  // For one chunk: shape (1, ELEMS_PER_SG) with ThreadLayout (1, 16), VecLayout (1, VEC_ELEMS)
  auto VecLayout_   = make_layout(make_shape(_1{}, Int<VEC_ELEMS>{}),
                                   Stride<Int<VEC_ELEMS>, _1>{});
  auto ThreadLayout_ = make_layout(make_shape(_1{}, _16{}));

  auto tiled_stsm = make_tiled_copy(Atom_stsm{}, ThreadLayout_, VecLayout_);
  auto tiled_ldsm = make_tiled_copy(Atom_ldsm{}, ThreadLayout_, VecLayout_);

  float accum = 0.f;

  // Pre-fill register fragments ONCE (outside timed loop)
  // Use one chunk's worth of fragments, filled with data to prevent DCE
  // We'll reuse these same fragments every iteration in the timed loop.
  constexpr int FRAG_ELEMS = VEC_ELEMS;  // elements per work-item per atom access

  // Allocate per-chunk store fragments and fill them once
  // (CHUNKS_PER_SG fragments, each of size FRAG_ELEMS)
  TA store_data[CHUNKS_PER_SG][FRAG_ELEMS];
  CUTE_UNROLL
  for (int c = 0; c < CHUNKS_PER_SG; ++c) {
    CUTE_UNROLL
    for (int j = 0; j < FRAG_ELEMS; ++j) {
      store_data[c][j] = static_cast<TA>(float((j + c) & 0x7F) * 0.01f);
    }
  }

  // Barrier to ensure all subgroups are ready
  barrier_arrive(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
  barrier_wait(SPIRVScope::ScopeWorkgroup,
               SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

  // Read hardware GPU clock BEFORE inner loop
  uint64_t t_start = sycl::ext::oneapi::experimental::clock<gpu_clock::sub_group>();

  // Inner loop — measures ONLY SLM <-> Register throughput
  for (int i = 0; i < inner_iters; ++i) {
    // Each subgroup writes its stripe using explicit VISA store atoms
    CUTE_UNROLL
    for (int c = 0; c < CHUNKS_PER_SG; ++c) {
      int base_elem = sg_idx * ELEMS_PER_SG_TOTAL + c * ELEMS_PER_SG;
      Tensor sChunk = make_tensor(make_smem_ptr(smem + base_elem),
                                  make_layout(Shape<Int<1>, Int<ELEMS_PER_SG>>{},
                                              Stride<Int<ELEMS_PER_SG>, _1>{}));
      auto thr_stsm = tiled_stsm.get_thread_slice(ThreadIdxX());
      Tensor thr_dst = thr_stsm.partition_D(sChunk);
      Tensor frag    = make_fragment_like(thr_stsm.partition_S(sChunk));

      // Copy pre-filled data into fragment
      CUTE_UNROLL
      for (int j = 0; j < size(frag); ++j) {
        frag(j) = store_data[c][j];
      }

      copy(tiled_stsm, frag, thr_dst);
    }

    // Barrier
    barrier_arrive(SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

    // Each subgroup reads its stripe using explicit VISA load atoms
    CUTE_UNROLL
    for (int c = 0; c < CHUNKS_PER_SG; ++c) {
      int base_elem = sg_idx * ELEMS_PER_SG_TOTAL + c * ELEMS_PER_SG;
      Tensor sChunk = make_tensor(make_smem_ptr(smem + base_elem),
                                  make_layout(Shape<Int<1>, Int<ELEMS_PER_SG>>{},
                                              Stride<Int<ELEMS_PER_SG>, _1>{}));
      auto thr_ldsm = tiled_ldsm.get_thread_slice(ThreadIdxX());
      Tensor thr_src = thr_ldsm.partition_S(sChunk);
      Tensor frag    = make_fragment_like(thr_ldsm.partition_D(sChunk));

      copy(tiled_ldsm, thr_src, frag);

      accum += float(frag(0));
    }

    // Barrier
    barrier_arrive(SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);
  }

  // Read hardware GPU clock AFTER inner loop
  uint64_t t_end = sycl::ext::oneapi::experimental::clock<gpu_clock::sub_group>();

  if (local_id == 0) {
    out(wg_m) = TA(accum);
    clock_buf[wg_m * 2]     = t_start;
    clock_buf[wg_m * 2 + 1] = t_end;
  }
}

// ============================================================================
// Kernel launch wrappers
// ============================================================================

class SLM_Bench_UniversalCopy;
class SLM_Bench_VISA;

template <class ATensor, class OutTensor>
sycl::event launch_universal_copy(sycl::queue &Q,
                           ATensor const& A,
                           OutTensor    & out,
                           BenchMMA const& mma,
                           int inner_iters,
                           int num_wgs,
                           uint64_t* clock_buf)
{
  sycl::range<2> local  = {static_cast<size_t>(size(mma)), 1};
  sycl::range<2> global = {local[0] * num_wgs, 1};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  return Q.parallel_for<SLM_Bench_UniversalCopy>(
    sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      slm_bench_universal_copy_device(A, out, mma, inner_iters, clock_buf);
    }
  );
}

template <class ATensor, class OutTensor>
sycl::event launch_visa(sycl::queue &Q,
                 ATensor const& A,
                 OutTensor    & out,
                 BenchMMA const& mma,
                 int inner_iters,
                 int num_wgs,
                 uint64_t* clock_buf)
{
  sycl::range<2> local  = {static_cast<size_t>(size(mma)), 1};
  sycl::range<2> global = {local[0] * num_wgs, 1};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  return Q.parallel_for<SLM_Bench_VISA>(
    sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      slm_bench_visa_device(A, out, mma, inner_iters, clock_buf);
    }
  );
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, const char** argv)
{
  Options options;
  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout);
    return 0;
  }

  // Create queue with profiling enabled for device-side kernel timing
  sycl::queue Q(sycl::gpu_selector_v,
                sycl::property_list{sycl::property::queue::enable_profiling{},
                                    sycl::property::queue::in_order{}});

  // Initialise GPU frequency monitor
  GpuFreqMonitor freq_mon;
  freq_mon.init(Q);

  // Print static GPU clock info
  {
    auto dev = Q.get_device();
    auto max_clock = dev.get_info<sycl::info::device::max_clock_frequency>();
    printf("GPU: %s\n", dev.get_info<sycl::info::device::name>().c_str());
    printf("  Max Clock (SYCL): %u MHz\n", max_clock);

    // Read one Sysman sample to show current state
    double act = 0, req = 0; uint32_t thr = 0;
    if (freq_mon.read_one(act, req, thr)) {
      printf("  Current   : actual=%.0f  requested=%.0f MHz\n", act, req);
    }
    freq_mon.print_timer_info();
    printf("\n");
  }

  auto mma = make_bench_mma();
  auto wg_tile = mma.tile_mnk();
  int tile_M = get<0>(wg_tile);
  int tile_K = get<2>(wg_tile);

  int M = tile_M * options.num_wgs;   // total rows so each WG gets its own tile row
  int K = tile_K;                      // single K-tile

  const int iterations  = options.iterations;
  const int inner_iters = options.inner_iters;
  const int num_wgs     = options.num_wgs;

  using TA = bfloat16_t;

  // Allocate global tensor A  (M × K, row-major)
  auto A_ptr = sycl::malloc_shared<TA>(M * K, Q);
  // Fill with some data
  for (int i = 0; i < M * K; i++) {
    A_ptr[i] = TA(float(i % 128) * 0.01f);
  }
  auto A = make_tensor(make_gmem_ptr(A_ptr), make_layout(make_shape(M, K), make_stride(K, _1{})));

  // Output sink (one element per WG, prevents DCE)
  auto out_ptr = sycl::malloc_shared<TA>(num_wgs, Q);
  for (int i = 0; i < num_wgs; i++) out_ptr[i] = TA(0);
  auto out = make_tensor(make_gmem_ptr(out_ptr), make_layout(make_shape(num_wgs)));

  // Clock buffer: 2 uint64_t per WG (start, end) for hardware clock measurement
  auto clock_buf = sycl::malloc_shared<uint64_t>(num_wgs * 2, Q);

  // Bytes moved through SLM per kernel launch
  // Each WG: r2s writes tile_M*tile_K elements, s2r reads them back = 2× tile bytes
  double slm_bytes_per_launch = double(num_wgs) * double(tile_M) * double(tile_K) * sizeof(TA) * 2.0 * inner_iters;

  printf("============================================================\n");
  printf("SLM <-> Register Throughput Benchmark\n");
  printf("  Measures ONLY SLM store/load — zero global memory in kernel\n");
  printf("  Timing via SYCL event profiling (device-side, pure kernel execution)\n");
  printf("  Data type    : bf16\n");
  printf("  WG tile      : %d x %d\n", tile_M, tile_K);
  printf("  Num WGs      : %d\n", num_wgs);
  printf("  WG size      : %d work-items\n", int(size(mma)));
  printf("  Inner iters  : %d\n", inner_iters);
  printf("  Outer iters  : %d\n", iterations);
  printf("  SLM bytes/launch : %.0f  (%.2f KB)\n", slm_bytes_per_launch, slm_bytes_per_launch / 1024.0);
  printf("============================================================\n\n");

  // ====================
  // Variant 1: UniversalCopy
  // ====================
  {
    printf("Variant 1: UniversalCopy (make_A_slm_copies)\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
      launch_universal_copy(Q, A, out, mma, inner_iters, num_wgs, clock_buf).wait();
    }

    // Start frequency sampling
    freq_mon.start();

    // Timed — use SYCL event profiling (device-side kernel execution time)
    double total_ns = 0;
    std::vector<double> per_launch_ns(iterations);
    std::vector<uint64_t> per_launch_hw_clk(iterations);  // median per-WG delta each launch
    std::vector<uint64_t> per_launch_max_clk(iterations);
    std::vector<uint64_t> per_launch_min_clk(iterations);
    for (int i = 0; i < iterations; i++) {
      auto ev = launch_universal_copy(Q, A, out, mma, inner_iters, num_wgs, clock_buf);
      ev.wait();
      auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
      per_launch_ns[i] = double(t1 - t0);
      total_ns += per_launch_ns[i];

      // Collect per-WG hardware clock deltas
      std::vector<uint64_t> wg_deltas;
      for (int w = 0; w < num_wgs; w++) {
        uint64_t delta = clock_buf[w * 2 + 1] - clock_buf[w * 2];
        if (delta > 0 && delta < UINT64_MAX/2) wg_deltas.push_back(delta);  // filter invalid
      }
      if (!wg_deltas.empty()) {
        std::sort(wg_deltas.begin(), wg_deltas.end());
        per_launch_hw_clk[i] = wg_deltas[wg_deltas.size() / 2];  // median
        per_launch_min_clk[i] = wg_deltas.front();
        per_launch_max_clk[i] = wg_deltas.back();
      }
    }

    freq_mon.stop();

    double elapsed = total_ns / 1e9;
    double total_bytes = slm_bytes_per_launch * iterations;
    double bw_GBs = total_bytes / elapsed / 1e9;
    double avg_ns = total_ns / iterations;
    double min_ns = *std::min_element(per_launch_ns.begin(), per_launch_ns.end());
    double max_ns = *std::max_element(per_launch_ns.begin(), per_launch_ns.end());

    printf("  Elapsed        : %.3f s  (device-side)\n", elapsed);
    printf("  Avg/launch     : %.4f ms\n", avg_ns / 1e6);
    printf("  Min/launch     : %.4f ms\n", min_ns / 1e6);
    printf("  Max/launch     : %.4f ms\n", max_ns / 1e6);
    printf("  SLM BW         : %.2f GB/s  (r2s + s2r combined)\n", bw_GBs);
    printf("  SLM BW r2s     : %.2f GB/s\n", bw_GBs / 2.0);
    printf("  SLM BW s2r     : %.2f GB/s\n", bw_GBs / 2.0);

    // Hardware-measured GPU clock ticks (from __spirv_ReadClockKHR on device)
    {
      uint64_t hw_min = *std::min_element(per_launch_min_clk.begin(), per_launch_min_clk.end());
      uint64_t hw_max = *std::max_element(per_launch_max_clk.begin(), per_launch_max_clk.end());
      double hw_median_avg = 0;
      for (auto c : per_launch_hw_clk) hw_median_avg += double(c);
      hw_median_avg /= iterations;

      printf("  HW Clocks (measured on device, median across WGs):\n");
      printf("    median-avg=%.0f  WG-min=%lu  WG-max=%lu  ticks/launch\n", hw_median_avg,
             (unsigned long)hw_min, (unsigned long)hw_max);
      if (hw_median_avg > 0) {
        printf("    Bytes/HW-tick    : %.2f\n", slm_bytes_per_launch / hw_median_avg);
        printf("    HW-ticks/byte    : %.6f\n", hw_median_avg / slm_bytes_per_launch);
      }
      printf("    Last launch WG spread: min=%lu  median=%lu  max=%lu  ticks\n",
             (unsigned long)per_launch_min_clk.back(),
             (unsigned long)per_launch_hw_clk.back(),
             (unsigned long)per_launch_max_clk.back());
    }
    freq_mon.report();
    printf("\n");
  }

  // ====================
  // Variant 2: Explicit VISA (XE_1D_STSM / XE_1D_LDSM)
  // ====================
  {
    printf("Variant 2: Explicit VISA (XE_1D_STSM / XE_1D_LDSM)\n");

    // Warmup
    for (int i = 0; i < 5; i++) {
      launch_visa(Q, A, out, mma, inner_iters, num_wgs, clock_buf).wait();
    }

    // Start frequency sampling
    freq_mon.start();

    // Timed — use SYCL event profiling (device-side kernel execution time)
    double total_ns = 0;
    std::vector<double> per_launch_ns(iterations);
    std::vector<uint64_t> per_launch_hw_clk(iterations);
    std::vector<uint64_t> per_launch_max_clk(iterations);
    std::vector<uint64_t> per_launch_min_clk(iterations);
    for (int i = 0; i < iterations; i++) {
      auto ev = launch_visa(Q, A, out, mma, inner_iters, num_wgs, clock_buf);
      ev.wait();
      auto t0 = ev.get_profiling_info<sycl::info::event_profiling::command_start>();
      auto t1 = ev.get_profiling_info<sycl::info::event_profiling::command_end>();
      per_launch_ns[i] = double(t1 - t0);
      total_ns += per_launch_ns[i];

      std::vector<uint64_t> wg_deltas;
      for (int w = 0; w < num_wgs; w++) {
        uint64_t delta = clock_buf[w * 2 + 1] - clock_buf[w * 2];
        if (delta > 0 && delta < UINT64_MAX/2) wg_deltas.push_back(delta);
      }
      if (!wg_deltas.empty()) {
        std::sort(wg_deltas.begin(), wg_deltas.end());
        per_launch_hw_clk[i] = wg_deltas[wg_deltas.size() / 2];
        per_launch_min_clk[i] = wg_deltas.front();
        per_launch_max_clk[i] = wg_deltas.back();
      }
    }

    freq_mon.stop();

    double elapsed = total_ns / 1e9;
    double total_bytes = slm_bytes_per_launch * iterations;
    double bw_GBs = total_bytes / elapsed / 1e9;
    double avg_ns = total_ns / iterations;
    double min_ns = *std::min_element(per_launch_ns.begin(), per_launch_ns.end());
    double max_ns = *std::max_element(per_launch_ns.begin(), per_launch_ns.end());

    printf("  Elapsed        : %.3f s  (device-side)\n", elapsed);
    printf("  Avg/launch     : %.4f ms\n", avg_ns / 1e6);
    printf("  Min/launch     : %.4f ms\n", min_ns / 1e6);
    printf("  Max/launch     : %.4f ms\n", max_ns / 1e6);
    printf("  SLM BW         : %.2f GB/s  (r2s + s2r combined)\n", bw_GBs);
    printf("  SLM BW r2s     : %.2f GB/s\n", bw_GBs / 2.0);
    printf("  SLM BW s2r     : %.2f GB/s\n", bw_GBs / 2.0);

    // Hardware-measured GPU clock ticks
    {
      uint64_t hw_min = *std::min_element(per_launch_min_clk.begin(), per_launch_min_clk.end());
      uint64_t hw_max = *std::max_element(per_launch_max_clk.begin(), per_launch_max_clk.end());
      double hw_median_avg = 0;
      for (auto c : per_launch_hw_clk) hw_median_avg += double(c);
      hw_median_avg /= iterations;

      printf("  HW Clocks (measured on device, median across WGs):\n");
      printf("    median-avg=%.0f  WG-min=%lu  WG-max=%lu  ticks/launch\n", hw_median_avg,
             (unsigned long)hw_min, (unsigned long)hw_max);
      if (hw_median_avg > 0) {
        printf("    Bytes/HW-tick    : %.2f\n", slm_bytes_per_launch / hw_median_avg);
        printf("    HW-ticks/byte    : %.6f\n", hw_median_avg / slm_bytes_per_launch);
      }
      printf("    Last launch WG spread: min=%lu  median=%lu  max=%lu  ticks\n",
             (unsigned long)per_launch_min_clk.back(),
             (unsigned long)per_launch_hw_clk.back(),
             (unsigned long)per_launch_max_clk.back());
    }
    freq_mon.report();
    printf("\n");
  }

  printf("============================================================\n");
  printf("Done.\n");

  sycl::free(A_ptr, Q);
  sycl::free(out_ptr, Q);
  sycl::free(clock_buf, Q);

  return 0;
}
