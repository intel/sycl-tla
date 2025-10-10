#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "default_gemm_configuration.hpp"
#include "gemm_testbed_3x.hpp"

namespace cutlass {
namespace {

template <typename LayoutA, typename LayoutB>
struct XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative {
  using ElementA = float_e5m2_t;
  using ElementB = float_e5m2_t;

  using Config = gemm::device::DefaultGemmConfigurationToCutlass3Types<
    arch::OpClassTensorOp, arch::IntelXe,
    ElementA, LayoutA,
    ElementB, LayoutB,
    float, layout::RowMajor,
    float>;

  using DispatchPolicy = gemm::MainloopIntelW8A8<2, gemm::KernelXe>;

  using CollectiveMainloop = gemm::collective::CollectiveMma<
    DispatchPolicy, typename Config::TileShape,
    ElementA, detail::TagToStrideA_t<LayoutA>,
    ElementB, detail::TagToStrideB_t<LayoutB>,
    typename Config::TiledMma,
    typename Config::GmemTiledCopyA, void, void, cute::identity,
    typename Config::GmemTiledCopyB, void, void, cute::identity
  >;

  using GemmKernel = gemm::kernel::GemmUniversal<
      cute::Shape<int,int,int,int>,
      CollectiveMainloop,
      typename Config::CollectiveEpilogue
      >;

  using Gemm = gemm::device::GemmUniversalAdapter<GemmKernel>;
};

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, llama2_7b) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, gpt3) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, mistral_7b) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, tensor_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, micro_batch) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8t_f32t_tensor_op_f32_cooperative, large_batch) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8t_fp8n_f32t_tensor_op_f32_cooperative, tensor_parallel) {
  using LayoutA = layout::RowMajor;
  using LayoutB = layout::ColumnMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8n_fp8t_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::ColumnMajor;
  using LayoutB = layout::RowMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

TEST(XE_Device_Gemm_fp8n_fp8n_f32t_tensor_op_f32_cooperative, model_parallel) {
  using LayoutA = layout::ColumnMajor;
  using LayoutB = layout::ColumnMajor;
  using Gemm = XE_Device_Gemm_fp8_fp8_f32_tensor_op_f32_cooperative<LayoutA, LayoutB>::Gemm;
  EXPECT_TRUE(test::gemm::device::TestXe<Gemm>(1.0, 0.0, true));
}

} // namespace
} // namespace cutlass
 