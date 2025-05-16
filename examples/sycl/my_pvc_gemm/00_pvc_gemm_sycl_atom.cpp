#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/atom/copy_atom.hpp"
#include "cute/atom/copy_traits_xe.hpp"
#include "cute/atom/mma_atom.hpp"
#include "cutlass/detail/layout.hpp"
#include "cutlass/device_kernel.h"
#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/epilogue/collective/xe_epilogue.hpp"
#include "cutlass/epilogue/fusion/xe_callbacks.hpp"
#include "cutlass/gemm/collective/collective_mma.hpp"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm_coord.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include <cute/tensor.hpp>
#include <random>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "helper.h"
#include "sycl_common.hpp"

using namespace cute;

// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, l, iterations;
  float alpha, beta;

  Options()
      : help(false), error(false), m(5120), n(4096), k(4096), l(1),
        iterations(20), alpha(1.f), beta(0.f) {}

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 5120);
    cmd.get_cmd_line_argument("n", n, 4096);
    cmd.get_cmd_line_argument("k", k, 4096);
    cmd.get_cmd_line_argument("l", l, 1);
    cmd.get_cmd_line_argument("alpha", alpha, 1.f);
    cmd.get_cmd_line_argument("beta", beta, 0.f);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
  }

  /// Prints the usage statement.
  std::ostream &print_usage(std::ostream &out) const {

    out << "PVC GEMM Example\n\n"
        << "Options:\n\n"
        << "  --help                      If specified, displays this usage "
           "statement\n\n"
        << "  --m=<int>                   Sets the M extent of the GEMM\n"
        << "  --n=<int>                   Sets the N extent of the GEMM\n"
        << "  --k=<int>                   Sets the K extent of the GEMM\n"
        << "  --l=<int>                   Sets the L extent (batch count) of "
           "the GEMM\n"
        << "  --alpha=<s32>               Epilogue scalar alpha\n"
        << "  --beta=<s32>                Epilogue scalar beta\n\n"
        << "  --iterations=<int>          Iterations\n\n";

    return out;
  }
};

//////////////////////////////////////////////////////////////////////////////////////////////////
struct MySYCLExampleRunner {
  using ElementAccumulator = float;     // <- data type of accumulator
  using ElementComputeEpilogue = float; // <- data type of epilogue operations
  using ElementInputA =
      bfloat16_t; // <- data type of elements in input matrix A
  using ElementInputB =
      bfloat16_t;              // <- data type of elements in input matrix B
  using ElementOutput = float; // <- data type of elements in output matrix D

  using LayoutA = cutlass::layout::RowMajor;
  using LayoutB = cutlass::layout::RowMajor;
  using LayoutC = cutlass::layout::RowMajor;
  using LayoutD = cutlass::layout::RowMajor;

  using StrideA = cutlass::gemm::TagToStrideA_t<LayoutA>;
  using StrideB = cutlass::gemm::TagToStrideB_t<LayoutB>;
  using StrideC = cutlass::gemm::TagToStrideC_t<LayoutC>;
  using StrideD = cutlass::gemm::TagToStrideC_t<LayoutD>;

  using ElementA = ElementInputA;
  using ElementB = ElementInputB;
  using ElementAcc = ElementAccumulator;

  using ElementC = ElementAccumulator;
  using ElementCompute = ElementComputeEpilogue;
  using ProblemShapeType = Shape<int, int, int, int>;

  // Workgroup-level tile
  using TileShape = Shape<_256, _256, _32>;
  using ClusterShape = Shape<_1, _1, _1>;
  using GmemTiledCopyA = XE_2D_U16x32x32_LD_N;
  using GmemTiledCopyB = XE_2D_U16x32x32_LD_V;
  static constexpr int PipelineStages = 2;
  using DispatchPolicy = cutlass::gemm::MainloopIntelXeXMX16<PipelineStages>;
  using TiledMma = typename TiledMMAHelper<
      MMA_Atom<XE_8x16x16_F32BF16BF16F32_TT>, Layout<TileShape>,
      Layout<Shape<_8, _4, _1>, Stride<_4, _1, _0>>>::TiledMMA;

  using CopyThreadShape = Shape<_1, Int<DispatchPolicy::SubgroupSize>>;

  using traits_load_A = Copy_Traits<GmemTiledCopyA, StrideA>;
  using atom_load_A = Copy_Atom<traits_load_A, ElementA>;
  using val_layout_load_A = decltype(make_layout(
      shape_div(typename traits_load_A::BlockShape{}, CopyThreadShape{})));
  using Copy_A = decltype(make_tiled_copy(
      atom_load_A{}, Layout<CopyThreadShape>{}, val_layout_load_A{}));

  using traits_load_B = Copy_Traits<GmemTiledCopyB, StrideB>;
  using atom_load_B = Copy_Atom<traits_load_B, ElementB>;
  using val_layout_load_B = decltype(make_layout(
      shape_div(typename traits_load_B::BlockShape{}, CopyThreadShape{})));
  using Copy_B = decltype(make_tiled_copy(
      atom_load_B{}, Layout<CopyThreadShape>{}, val_layout_load_B{}));


  StrideA stride_A;
  StrideB stride_B;
  StrideC stride_C;
  StrideD stride_D;
  uint64_t seed = 0;

  using TileScheduler =
      typename cutlass::gemm::kernel::detail::TileSchedulerSelector<
          cutlass::gemm::PersistentScheduler, cutlass::arch::IntelXe, TileShape,
          cute::Shape<cute::Int<1>, cute::Int<1>, cute::Int<1>>>::Scheduler;
  using TileSchedulerParams = typename TileScheduler::Params;

  struct MainloopParams {
    Copy_A tiled_copy_a;
    Copy_B tiled_copy_b;
  };

  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShapeType problem_shape{};
    MainloopParams mainloop{};
    int epilogue;
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  struct Params {
    cutlass::gemm::GemmUniversalMode mode{};
    ProblemShapeType problem_shape{};
    MainloopParams mainloop{};
    int epilogue;
    cutlass::KernelHardwareInfo hw_info{};
    TileSchedulerParams scheduler{};
  };

  static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;

  using MmaAtomShape = typename TiledMma::AtomShape_MNK;

  static constexpr auto BLK_M = get<0>(TileShape{});
  static constexpr auto BLK_N = get<1>(TileShape{});
  static constexpr auto BLK_K = get<2>(TileShape{});

  static constexpr auto ATOM_M =
      get<1>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMma::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_K =
      get<3>(typename TiledMma::ThrLayoutVMNK{}.shape());

  static_assert(BLK_M % TiledMma{}.template tile_size_mnk<0>() == 0,
                "TiledMma permutation size must match block size.");
  static_assert(BLK_N % TiledMma{}.template tile_size_mnk<1>() == 0,
                "TiledMma permutation size must match block size.");
  static_assert(BLK_K % TiledMma{}.template tile_size_mnk<2>() == 0,
                "TiledMma permutation size must match block size.");

  static constexpr auto SG_M = ceil_div(BLK_M, ATOM_M);
  static constexpr auto SG_N = ceil_div(BLK_N, ATOM_N);
  static constexpr auto SG_K = ceil_div(BLK_K, ATOM_K);

  using SubgroupTileShape =
      Shape<decltype(SG_M), decltype(SG_N), decltype(SG_K)>;
  struct Memory {
    ElementA *block_A;
    ElementB *block_B;
    ElementC *block_C;
    ElementOutput *block_D;
    ElementOutput *block_ref_D;
    sycl::queue q;

    Memory(sycl::queue q, ProblemShapeType problem_shape_MNKL) : q(q) {
      auto [M, N, K, L] = problem_shape_MNKL;
      block_A =
          sycl::malloc_device<ElementA>(static_cast<std::size_t>(M) * K * L, q);
      block_B =
          sycl::malloc_device<ElementB>(static_cast<std::size_t>(N) * K * L, q);
      block_C =
          sycl::malloc_device<ElementC>(static_cast<std::size_t>(M) * N * L, q);
      block_D = sycl::malloc_device<ElementOutput>(
          static_cast<std::size_t>(M) * N * L, q);
      block_ref_D = sycl::malloc_device<ElementOutput>(
          static_cast<std::size_t>(M) * N * L, q);
    }

    ~Memory() {
      sycl::free(block_A, q);
      sycl::free(block_B, q);
      sycl::free(block_C, q);
      sycl::free(block_D, q);
      sycl::free(block_ref_D, q);
    }

    // delete other constructors so avoiding leaks is easy
    Memory(const Memory &) = delete;
    Memory(Memory &&) noexcept = delete;
    Memory &operator=(const Memory &) = delete;
    Memory &operator=(Memory &&) noexcept = delete;
  };

  bool verify(Memory &mem, const ProblemShapeType &problem_size,
              ElementCompute alpha, ElementCompute beta) {
    auto [M, N, K, L] = problem_size;

    std::size_t sizeA = std::size_t(M) * K * L;
    std::size_t sizeB = std::size_t(K) * N * L;
    std::size_t sizeC = std::size_t(M) * N * L;
    std::size_t sizeD = sizeC;

    std::vector<ElementA> host_A(sizeA);
    std::vector<ElementB> host_B(sizeB);
    std::vector<ElementC> host_C(sizeC);
    std::vector<ElementOutput> host_D(sizeD);
    std::vector<ElementOutput> host_ref_D(sizeD);

    mem.q.memcpy(host_A.data(), mem.block_A, sizeof(ElementA) * sizeA);
    mem.q.memcpy(host_B.data(), mem.block_B, sizeof(ElementB) * sizeB);
    mem.q.memcpy(host_C.data(), mem.block_C, sizeof(ElementC) * sizeC);
    mem.q.memcpy(host_D.data(), mem.block_D, sizeof(ElementOutput) * sizeD);
    mem.q.wait();

    for (int l = 0; l < L; ++l) {
      std::size_t offA = std::size_t(l) * M * K;
      std::size_t offB = std::size_t(l) * K * N;
      std::size_t offC = std::size_t(l) * M * N;
      std::size_t offD = std::size_t(l) * M * N;

      for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
          ElementAccumulator acc = ElementAccumulator(0);
          for (int k = 0; k < K; ++k) {
            acc += ElementAccumulator(host_A[offA + m * K + k]) *
                   ElementAccumulator(host_B[offB + k * N + n]);
          }
        }
      }
    }

    for (std::size_t i = 0; i < sizeD; ++i) {
      if (host_ref_D[i] != host_D[i]) {
        std::cout << "host_ref_D[" << i << "] != host_D[" << i << "]"
                  << std::endl;
        return false;
      }
    }

    return true;
  }

  /// Initialize operands to be used in the GEMM and reference GEMM
  void initialize(const ProblemShapeType &problem_size, Memory &mem) {
    auto problem_shape_MNKL = cute::append<4>(problem_size, 1);
    auto [M, N, K, L] = problem_shape_MNKL;

    stride_A =
        cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    stride_B =
        cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    stride_C =
        cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    stride_D =
        cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, L));

    initialize_block(mem.block_A, M * K * L, seed + 2023);
    initialize_block(mem.block_B, N * K * L, seed + 2022);
    initialize_block(mem.block_C, M * N * L, seed + 2021);
  }

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    // SharedStorage& shared_storage =
    // *reinterpret_cast<SharedStorage*>(smem_buf); Preconditions
    CUTE_STATIC_ASSERT(is_static<TileShape>::value);

    // Separate out problem shape for convenience
    // Optionally append 1s until problem shape is rank-4 in case its is only
    // rank-3 (MNK)
    auto problem_shape_MNKL = append<4>(params.problem_shape, Int<1>{});
    auto M = get<0>(problem_shape_MNKL);
    auto N = get<1>(problem_shape_MNKL);
    auto K = get<2>(problem_shape_MNKL);
    auto L = get<3>(problem_shape_MNKL);

    // Preconditions
    static_assert(cute::rank(StrideA{}) == 3,
                  "StrideA must be rank-3: [M, K, L]. If batch mode is not "
                  "needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideB{}) == 3,
                  "StrideB must be rank-3: [N, K, L]. If batch mode is not "
                  "needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideC{}) == 3,
                  "StrideC must be rank-3: [M, N, L]. If batch mode is not "
                  "needed, set L stride to Int<0>.");
    static_assert(cute::rank(StrideD{}) == 3,
                  "StrideD must be rank-3: [M, N, L]. If batch mode is not "
                  "needed, set L stride to Int<0>.");

    // Get the appropriate blocks for this sub_group -- potential for sub_group
    // locality
    int thread_idx = int(ThreadIdxX());
    auto blk_shape = TileShape{};
    int m_coord, n_coord, l_coord;
    if (params.scheduler.raster_order_ == TileScheduler::RasterOrder::AlongN) {
      m_coord = BlockIdxY();
      n_coord = BlockIdxX();
      l_coord = BlockIdxZ();
    } else {
      m_coord = BlockIdxX();
      n_coord = BlockIdxY();
      l_coord = BlockIdxZ();
    }

    auto blk_coord_mnkl = make_coord(m_coord, n_coord, _, l_coord);
    constexpr auto workgroup_shape = TileShape{}; // (SUB_M,SUB_N,SUB_K)
    constexpr auto subgroup_shape = SubgroupTileShape{};

    Tensor mA_mkl = cute::get_xe_tensor(make_shape(M, K, L)); //(m,k,l)
    Tensor mB_nkl = cute::get_xe_tensor(make_shape(N, K, L)); //(n,k,l)

    Tensor gA = local_tile(mA_mkl, select<0, 2>(blk_shape),
                           make_coord(m_coord, _, l_coord));
    Tensor gB = local_tile(mB_nkl, select<1, 2>(blk_shape),
                           make_coord(n_coord, _, l_coord));

    // Allocate the tiled_mma and the accumulators for the (M,N) subgroup_shape
    TiledMma tiled_mma;

    Tensor accumulators =
        partition_fragment_C(tiled_mma, take<0, 2>(blk_shape));
    clear(accumulators);

    auto k_tile_iter =
        cute::make_coord_iterator(idx2crd(0, make_shape(K)), make_shape(K));
    int k_tile_count = ceil_div(K, get<2>(workgroup_shape));

    // Perform the collective scoped MMA
    cutlass::gemm::collective::CollectiveMma<
        DispatchPolicy, TileShape, ElementInputA,
        cutlass::gemm::TagToStrideA_t<LayoutA>, ElementInputB,
        cutlass::gemm::TagToStrideB_t<LayoutB>, TiledMma, GmemTiledCopyA, void,
        void, cute::identity,                      // A
        GmemTiledCopyB, void, void, cute::identity // B
        >
        collective_mma;

    collective_mma(
        accumulators, gA, gB, accumulators, k_tile_iter, k_tile_count,
        blk_coord_mnkl, // TODO(codeplay): Remove this once unneeded in
                        // xe_mma_mixed_input.hpp
        K, thread_idx,
        {params.mainloop.tiled_copy_a, params.mainloop.tiled_copy_b});
  }

  cutlass::Status run_sycl(const Options &options,
                           const cutlass::KernelHardwareInfo &hw_info) {
    // sycl::queue q = syclcompat::get_default_queue();
    auto q = syclcompat::create_queue();
    auto problem_size =
        ProblemShapeType{options.m, options.n, options.k, options.l};
    auto [M, N, K, L] = problem_size;
    Memory mem(q, problem_size);
    initialize(problem_size, mem);

    static constexpr int SubgroupSize = DispatchPolicy::SubgroupSize;
    static constexpr uint32_t MaxThreadsPerBlock = size(TiledMma{});
    dim3 const block = dim3(MaxThreadsPerBlock, 1, 1);
    ;

    auto cta_m = cute::size(cute::ceil_div(cute::shape<0>(problem_size),
                                           cute::shape<0>(TileShape{})));
    auto cta_n = cute::size(cute::ceil_div(cute::shape<1>(problem_size),
                                           cute::shape<1>(TileShape{})));
    auto cluster_shape = cutlass::gemm::to_gemm_coord(ClusterShape{});
    auto problem_blocks_m =
        ((cta_m + cluster_shape.m() - 1) / cluster_shape.m()) *
        cluster_shape.m();
    auto problem_blocks_n =
        ((cta_n + cluster_shape.n() - 1) / cluster_shape.n()) *
        cluster_shape.n();
    // Need adjust x and y?
    dim3 const grid = dim3(problem_blocks_m, problem_blocks_n, L);

    const syclcompat::dim3 sycl_block(block.x, block.y, block.z);
    const syclcompat::dim3 sycl_grid(grid.x, grid.y, grid.z);

    auto mA_mkl = make_tensor(make_gmem_ptr(mem.block_A),
                              make_layout(make_shape(M, K, L), stride_A));
    auto mB_nkl = make_tensor(make_gmem_ptr(mem.block_B),
                              make_layout(make_shape(N, K, L), stride_B));
    Copy_A tiled_copy_a{Copy_A{}.with(mA_mkl)};
    Copy_B tiled_copy_b{Copy_B{}.with(mB_nkl)};

    Params params;
    params.mode = cutlass::gemm::GemmUniversalMode::kGemm;
    params.problem_shape = problem_size;
    params.mainloop.tiled_copy_a = tiled_copy_a;
    params.mainloop.tiled_copy_b = tiled_copy_b;

    TileSchedulerParams scheduler = TileScheduler::to_underlying_arguments(
        problem_size, TileShape{}, ClusterShape{}, hw_info,
        {1, cutlass::gemm::kernel::detail::RasterOrderOptions::Heuristic},
        nullptr);
    params.scheduler = scheduler;

    // int smem_size = GemmKernel_::SharedStorageSize;

    using namespace syclcompat::experimental;
    auto event = launch<cutlass::device_kernel<MySYCLExampleRunner>>(
        launch_policy{
            sycl_grid, sycl_block,
            kernel_properties{sycl_exp::sub_group_size<SubgroupSize>}},
        q, params);
    event.wait_and_throw();
    std::cout << "sycl kernel done" << std::endl;
    bool passed = verify(mem, problem_size, options.alpha, options.beta);
    std::cout << "Disposition: " << (passed ? "Passed" : "Failed") << std::endl;

    if (!passed)
      return cutlass::Status::kErrorInternal;

    return cutlass::Status::kSuccess;
  }
};

int main(int argc, const char **argv) {
  //
  // Parse options
  //

  Options options;

  options.parse(argc, argv);

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  //
  // Run examples
  //

  // The KernelHardwareInfo struct holds the number of EUs on the GPU with a
  // given device ID. This information is used by the underlying kernel.
  cutlass::KernelHardwareInfo hw_info;

  // Change device_id to another value if you are running on a machine with
  // multiple GPUs and wish to use a GPU other than that with device ID 0.
  hw_info.sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
          hw_info.device_id);
  std::cout << "SM count: " << hw_info.sm_count << std::endl;
  bool passed;

  MySYCLExampleRunner runner_sycl;
  CUTLASS_CHECK(runner_sycl.run_sycl(options, hw_info));

  return 0;
}
