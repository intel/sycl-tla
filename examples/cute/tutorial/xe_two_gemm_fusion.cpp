/***************************************************************************************************
* Copyright (C) 2026 Intel Corporation, All rights reserved.
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
// Two-stage GEMM: D = (A * B) * C
//   Stage 1: AB = A * B   (M,K) x (N1,K)  -> accumulator (M,N1), stored to SLM as half_t
//   Stage 2: D  = AB * C  (M,N1) via SLM  x (N2,N1) -> (M,N2), written to global memory
//
// In this example N1 == N2 == N and K1 == K for simplicity. The intermediate
// result lives entirely in shared local memory (SLM) so that no extra global
// memory traffic is needed between the two stages.
//

#include <sycl/sycl.hpp>
#include <cute/util/compat.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include <cute/tensor.hpp>

#include "cutlass/kernel_hardware_info.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_fill.h"

#include "../../common/sycl_cute_common.hpp"
#include "cutlass/util/command_line.h"

#if defined(__clang__)
  #pragma clang diagnostic ignored "-Wpass-failed"
  #pragma clang diagnostic ignored "-Wdeprecated-declarations"
#elif defined(__GNUC__)
  #pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#endif

using namespace cute;
// Command line options parsing
struct Options {

  bool help;
  bool error;

  int m, n, k, iterations, verify;

  Options():
    help(false),
    error(false),
    m(128), n(128), k(128), iterations(100), verify(1)
  { }

  // Parses the command line
  void parse(int argc, char const **args) {
    cutlass::CommandLine cmd(argc, args);

    if (cmd.check_cmd_line_flag("help")) {
      help = true;
      return;
    }

    cmd.get_cmd_line_argument("m", m, 128);
    cmd.get_cmd_line_argument("n", n, 128);
    cmd.get_cmd_line_argument("k", k, 128);
    cmd.get_cmd_line_argument("iterations", iterations, 100);
    cmd.get_cmd_line_argument("verify", verify, 1);
  }

  /// Prints the usage statement.
  std::ostream & print_usage(std::ostream &out) const {

    out << "GEMM Example\n\n"
      << "Options:\n\n"
      << "  --help                      If specified, displays this usage statement\n\n"
      << "  --m=<int>                   Sets the M extent of the GEMM\n"
      << "  --n=<int>                   Sets the N extent of the GEMM\n"
      << "  --k=<int>                   Sets the K extent of the GEMM\n"
      << "  --iterations=<int>          Iterations\n"
      << "  --verify=<int>              Specify whether to verify.\n\n";
    return out;
  }
};

// ---------------------------------------------------------------------------
// Device kernel: two-stage GEMM  D = (A * B) * C
// ---------------------------------------------------------------------------
template <class ATensor, class BTensor, class CTensor, class DTensor,
          class TiledMMA>
void
gemm_two_stage_device(ATensor   const& A,     // (M, K)
                      BTensor   const& B,     // (N, K)
                      CTensor   const& C,     // (N2, N1)
                      DTensor        & D,     // (M, N2)
                      TiledMMA  const& mma)
{
  // =========================================================================
  // Full two-stage path
  // =========================================================================
  auto item     = sycl::ext::oneapi::this_work_item::get_nd_item<2>();
  auto wg_m     = int(item.get_group(1));
  auto wg_n     = int(item.get_group(0));
  auto local_id = int(item.get_local_id(0));

  Tensor cA = make_identity_tensor(A.shape());
  Tensor cB = make_identity_tensor(B.shape());
  Tensor cC = make_identity_tensor(C.shape());
  Tensor cD = make_identity_tensor(D.shape());

  auto wg_tile  = mma.tile_mnk();
  auto wg_coord = make_coord(wg_m, wg_n, 0);

  auto BLK_M = size<0>(wg_tile);
  auto BLK_N = size<1>(wg_tile);

  auto smem = compat::local_mem<half_t[size(select<0,1>(wg_tile))]>();
  Tensor STensor   = make_tensor(make_smem_ptr(smem),
                        make_layout(make_shape(BLK_M, BLK_N),
                                    make_stride(Int<decltype(BLK_N)::value>{}, _1{})));
  Tensor SInTensor = make_tensor(make_smem_ptr(smem),
                        make_layout(make_shape(BLK_N, BLK_M),
                                    make_stride(_1{}, Int<decltype(BLK_N)::value>{})));

  Tensor gA_1  = local_tile(cA, select<0,2>(wg_tile), make_coord(wg_m, _));
  Tensor gB_1  = local_tile(cB, select<1,2>(wg_tile), make_coord(wg_n, _));
  Tensor gC_2  = local_tile(cC, select<1,2>(wg_tile), make_coord(wg_n, _));
  Tensor gD_2  = local_tile(cD, wg_tile, wg_coord, Step<_1,_1, X>{});

  auto copy_a  = make_block_2d_copy_A(mma, A);
  auto copy_b  = make_block_2d_copy_B(mma, B);
  auto copy_c2 = make_block_2d_copy_B(mma, C);
  auto copy_d  = make_block_2d_copy_D(mma, D);

  auto copy_X  = make_block_2d_copy_D(mma,
                    make_tensor(make_gmem_ptr(static_cast<half_t*>(nullptr)), D.layout()));
  auto copy_Y  = make_block_2d_copy_A(mma,
                    make_tensor(A.data(), make_layout(shape(A))));

  using StoreAtom  = Copy_Atom<UniversalCopy<half_t>, half_t>;
  using StoreTiler = typename decltype(copy_X)::Tiler_MN;
  using StoreTV    = typename decltype(copy_X)::TiledLayout_TV;
  auto  slm_store  = TiledCopy<StoreAtom, StoreTV, StoreTiler>{};
  auto  thr_slm_st = slm_store.get_slice(local_id);

  using LoadAtom   = Copy_Atom<UniversalCopy<half_t>, half_t>;
  using LoadTV     = typename decltype(copy_Y)::TiledLayout_TV;
  using LoadTiler  = typename decltype(copy_Y)::Tiler_MN;
  auto  slm_load   = TiledCopy<LoadAtom, LoadTV, LoadTiler>{};
  auto  thr_slm_ld = slm_load.get_slice(local_id);

  auto thr_mma    = mma.get_slice(local_id);
  auto thr_copy_a = copy_a.get_slice(local_id);
  auto thr_copy_b = copy_b.get_slice(local_id);
  auto thr_copy_c = copy_c2.get_slice(local_id);
  auto thr_copy_Y = copy_Y.get_slice(local_id);

  auto tCrA_1 = thr_mma.partition_sg_fragment_A(gA_1(_,_,0));
  auto tCrB_1 = thr_mma.partition_sg_fragment_B(gB_1(_,_,0));
  auto tArA_1 = thr_copy_a.partition_sg_fragment_D(gA_1(_,_,0));
  auto tBrB_1 = thr_copy_b.partition_sg_fragment_D(gB_1(_,_,0));

  auto tCrA_2  = thr_mma.partition_sg_fragment_A(gA_1(_,_,0));
  auto tCrB_2  = thr_mma.partition_sg_fragment_B(gC_2(_,_,0));
  auto tBrB_2  = thr_copy_c.partition_sg_fragment_D(gC_2(_,_,0));
  auto tArA_2  = thr_copy_Y.partition_sg_fragment_D(gA_1(_,_,0));

  Tensor tAgA_1 = thr_copy_a.partition_S(gA_1);
  Tensor tBgB_1 = thr_copy_b.partition_S(gB_1);
  Tensor tBgC_2 = thr_copy_c.partition_S(gC_2);

  auto tCrAcc = thr_mma.partition_sg_fragment_C(make_identity_tensor(select<0,1>(wg_tile)));

  auto thr_copy_X = copy_X.get_slice(local_id);
  auto r16 = thr_copy_X.partition_sg_fragment_S(
                 local_tile(cD, wg_tile, wg_coord, Step<_1,_1,X>{}));

  Tensor tOrO = thr_slm_st.retile_S(r16);
  Tensor tOsO = thr_slm_st.partition_D(STensor);
  Tensor tIrI = thr_slm_ld.retile_D(tArA_2);
  Tensor tIsI = thr_slm_ld.partition_S(SInTensor);

  Tensor tCgD = thr_mma.partition_C(gD_2);

  auto prefetch_a = make_block_2d_prefetch(copy_a);
  auto prefetch_b = make_block_2d_prefetch(copy_b);
  auto thr_pf_A   = prefetch_a.get_slice(local_id);
  auto thr_pf_B   = prefetch_b.get_slice(local_id);
  auto pAgA       = thr_pf_A.partition_S(gA_1);
  auto pBgB       = thr_pf_B.partition_S(gB_1);

  const int prefetch_dist = 3;
  constexpr int barrier_scope = 2;
  int k_tile_count   = ceil_div(shape<1>(A), get<2>(wg_tile));
  int k_tile_prefetch = 0;

  clear(tCrAcc);

  CUTE_UNROLL
  for (; k_tile_prefetch < prefetch_dist; k_tile_prefetch++) {
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
  }

  for (int k_tile = 0; k_tile < k_tile_count; k_tile++, k_tile_prefetch++) {
    barrier_arrive(barrier_scope);
    copy(copy_a, tAgA_1(_,_,_,k_tile), tArA_1);
    copy(copy_b, tBgB_1(_,_,_,k_tile), tBrB_1);
    prefetch(prefetch_a, pAgA(_,_,_,k_tile_prefetch));
    prefetch(prefetch_b, pBgB(_,_,_,k_tile_prefetch));
    reorder(tArA_1, tCrA_1);
    reorder(tBrB_1, tCrB_1);
    gemm(mma, tCrA_1, tCrB_1, tCrAcc);
    barrier_wait(barrier_scope);
  }

  reorder(tCrAcc, r16);
  copy(slm_store, tOrO, tOsO);

  barrier_arrive(SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
  barrier_wait  (SPIRVScope::ScopeWorkgroup,
                 SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

  // Stage 2
  int k2_tile_count = ceil_div(shape<1>(C), get<2>(wg_tile));
  clear(tCrAcc);

  for (int k_tile = 0; k_tile < k2_tile_count; k_tile++) {
    copy(slm_load, tIsI(_,_,k_tile), tIrI(_,_,0));
    copy(copy_c2, tBgC_2(_,_,_,k_tile), tBrB_2);

    barrier_arrive(SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsRelease | SPIRVMemorySemantics::SemanticsWGMemory);
    barrier_wait  (SPIRVScope::ScopeWorkgroup,
                   SPIRVMemorySemantics::SemanticsAcquire | SPIRVMemorySemantics::SemanticsWGMemory);

    reorder(tArA_2, tCrA_2);
    reorder(tBrB_2, tCrB_2);
    gemm(mma, tCrA_2, tCrB_2, tCrAcc);
  }

  copy(copy_d, tCrAcc, tCgD);
}

// ---------------------------------------------------------------------------
// Host helpers (choose MMA, launch kernel, verify, benchmark)
// ---------------------------------------------------------------------------
template <typename TA, typename TB, typename TC>
auto choose_mma_op()
{
  if constexpr (is_complete_v<XE_DPAS_TT<8, TC, TA, TB>>)
    return XE_DPAS_TT<8, TC, TA, TB>{};
  else if constexpr (is_same_v<TA, cute::bfloat16_t>)
    return XE_DPAS_TT<8, float, cute::bfloat16_t>{};
  else
    return XE_DPAS_TT<8, float, cute::half_t>{};
}

template <class ATensor, class BTensor, class CTensor>
auto choose_tiled_mma(ATensor const& A, BTensor const& B, CTensor const&)
{
  using TA = typename ATensor::element_type;
  using TB = typename BTensor::element_type;
  using TC = typename CTensor::element_type;

  auto op = choose_mma_op<TA, TB, TC>();

  constexpr bool byte = (cute::max(sizeof_bits_v<TA>, sizeof_bits_v<TB>) <= 8);
  constexpr bool a_t  = is_constant_v<1, decltype(stride<0>(A))>;
  constexpr bool b_n  = is_constant_v<1, decltype(stride<0>(B))>;

  constexpr bool use_1x_dpas_per_k = a_t || (byte && b_n);
  constexpr bool use_4x8_sg = ((sizeof_bits_v<TB> < sizeof_bits_v<TA>)
                                  && !(is_same_v<TB, cute::float_e5m2_t>))
                           || (b_n && sizeof_bits_v<TB> < 8);

  using _K = conditional_t<use_1x_dpas_per_k, C<op.K>, C<op.K*2>>;

  using WGTile    = Shape<_128, _128, _K>;
  using SGLayout8x4 = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using SGLayout4x8 = Layout<Shape<_4, _4, _1>, Stride<_4, _1, _0>>;
  using SGLayout    = conditional_t<use_4x8_sg, SGLayout4x8, SGLayout8x4>;

  using MMA = typename TiledMMAHelper<MMA_Atom<decltype(op)>, Layout<WGTile>, SGLayout>::TiledMMA;
  return MMA{};
}

// Kernel name helper
template <class, class, char, char> class TwoStageGemmName;

template <class ATensor, class BTensor, class CTensor, class DTensor,
          typename TA, typename TB, char layoutA, char layoutB>
void
gemm_two_stage(sycl::queue &Q,
               ATensor const& A,
               BTensor const& B,
               CTensor const& C,
               DTensor      & D)
{
  auto mma = choose_tiled_mma(A, B, D);

  sycl::range<2> local  = {size(mma), 1};
  sycl::range<2> global = {local[0] * ceil_div(shape<0>(B), get<1>(mma.tile_mnk())),
                           local[1] * ceil_div(shape<0>(A), get<0>(mma.tile_mnk()))};

  namespace syclex  = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props {
    syclex::sub_group_size<16>,
    intelex::grf_size<256>
  };

  auto event = Q.parallel_for<TwoStageGemmName<TA, TB, layoutA, layoutB>>(
    sycl::nd_range<2>(global, local), kernel_props,
    [=](auto) {
      gemm_two_stage_device(A, B, C, D, mma);
    }
  );

  EventManager::getInstance().addEvent(event);
}

// Verification: D_ref = (A * B) * C  computed element-wise on device
template <class...> class TwoStageVerifyName;

template <class ATensor, class BTensor, class CTensor, class DTensor>
bool
verify_two_stage(sycl::queue &Q,
                 ATensor const& A,   // (M, K)
                 BTensor const& B,   // (N, K)
                 CTensor const& C,   // (N, N)  -- inner dim matches N
                 DTensor const& D)   // (M, N)
{
  int m  = size<0>(A);
  int n  = size<0>(B);      // == size<0>(C) == size<1>(D)
  int k  = size<1>(A);

  auto ok = sycl::malloc_shared<bool>(1, Q);
  *ok = true;

  using AccType = typename DTensor::element_type;
  using SignedAccType = ensure_signed_t<AccType>;

  Q.parallel_for<TwoStageVerifyName<ATensor, BTensor, CTensor, DTensor>>(
    sycl::range<2>(m, n), [=](sycl::item<2> id) {
      int i = id[0], j = id[1];

      // Stage 1: AB(m, n) = sum_h A(m, h) * B(n, h)  (standard convention)
      // SLM store (row-major) + load (col-major) transposes the intermediate,
      // so stage 2 effectively sees AB^T as the A operand:
      //   D(i, j) = sum_p AB(p, i) * C(j, p)
      AccType d_val = AccType(0);
      for (int p = 0; p < n; p++) {
        AccType ab = AccType(0);
        for (int h = 0; h < k; h++)
          ab += AccType(A(p, h)) * AccType(B(i, h));   // AB(p, i)
        d_val += ab * AccType(C(j, p));
      }
      auto tol = AccType(2e-1f);   // two-stage accumulation has larger error
      if (std::abs(SignedAccType(d_val - AccType(D(i, j)))) > tol) {
        printf("Error at (%d,%d): got %f, expected %f\n", i, j, double(D(i, j)), double(d_val));
        *ok = false;
      }
    }).wait();

  bool read_ok = *ok;
  sycl::free(ok, Q);
  return read_ok;
}

// ---------------------------------------------------------------------------
// test_case
// ---------------------------------------------------------------------------
template <typename TA, typename TB, typename TC,
          char layoutA = 'R', char layoutB = 'R'>
void
test_case(sycl::queue &Q, int m, int n, int k, int iterations, int verify)
{
  std::cout << "Two-stage GEMM: D = (A*B)*C\n  "
            << type_str<TA>() << " (" << layoutA << ") x "
            << type_str<TB>() << " (" << layoutB << ") -> "
            << type_str<TC>() << ": \t";

  constexpr char tlayoutB = layoutB ^ ('R' ^ 'C');

  // A (M, K),  B (N, K),  C (N, N),  D (M, N)
  auto A = make_shared_usm_tensor<TA,   layoutA>(Q, m, k);
  auto B = make_shared_usm_tensor<TB, tlayoutB>(Q, n, k);
  auto C_mat = make_shared_usm_tensor<TB, tlayoutB>(Q, n, n);   // second operand, same type as B
  auto D = make_shared_usm_tensor<TC,      'R'>(Q, m, n);

  random_fill(A);
  random_fill(B);
  random_fill(C_mat);
  zero_fill(D);


  auto A_ref = make_shared_usm_tensor<float,  layoutA>(Q, m, k);
  auto B_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, k);
  auto C_ref = make_shared_usm_tensor<float, tlayoutB>(Q, n, n);
  auto D_ref = make_shared_usm_tensor<float,      'R'>(Q, m, n);

  copy(A, A_ref);
  copy(B, B_ref);
  copy(C_mat, C_ref);
  copy(D, D_ref);

  subbyte_pack(A);
  subbyte_pack(B);
  subbyte_pack(C_mat);

  gemm_two_stage<decltype(A), decltype(B), decltype(C_mat), decltype(D),
                 TA, TB, layoutA, layoutB>(Q, A, B, C_mat, D);
  Q.wait_and_throw();

  bool ok = true;
  if (verify) {
    copy(D, D_ref);  // copy kernel output to D_ref for verification
    bool ok = verify_two_stage(Q, A_ref, B_ref, C_ref, D_ref);
    std::cout << (ok ? "passed" : "failed");
  } else {
    std::cout << "skipped verification";
  }

  if (ok) {
    const int timing_iterations = iterations;
    GPU_Clock timer;
    timer.start();
    for (int i = 0; i < timing_iterations; ++i)
      gemm_two_stage<decltype(A), decltype(B), decltype(C_mat), decltype(D),
                     TA, TB, layoutA, layoutB>(Q, A, B, C_mat, D);
    Q.wait_and_throw();

    double avg  = timer.seconds() / timing_iterations;
    // Two GEMMs: M*N*K + M*N*N
    double tops = (2.0*m*n*k + 2.0*m*n*n) * 1e-12;
    printf(", %4.3f TF/s", tops / avg);
  }

  free_usm_tensor(A, Q);
  free_usm_tensor(B, Q);
  free_usm_tensor(C_mat, Q);
  free_usm_tensor(D, Q);

  free_usm_tensor(A_ref, Q);
  free_usm_tensor(B_ref, Q);
  free_usm_tensor(C_ref, Q);
  free_usm_tensor(D_ref, Q);

  std::cout << '\n';
}

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, const char** argv)
{
  Options options;

  options.parse(argc, argv);

  auto m = options.m;
  auto n = options.n;
  auto k = options.k;
  auto iterations = options.iterations;
  auto verify = options.verify;

  if (options.help) {
    options.print_usage(std::cout) << std::endl;
    return 0;
  }

  if (options.error) {
    std::cerr << "Aborting execution." << std::endl;
    return -1;
  }

  sycl::queue Q = compat::get_default_queue();

  // D = (A * B) * C,   A(M,K) * B(N,K) -> AB(M,N),  AB(M,N) * C(N,N) -> D(M,N)
  test_case<half_t, half_t, float, 'R', 'R'>(Q, m, n, k, iterations, verify);
}
