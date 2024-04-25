#pragma once

#include <cute/atom/copy_atom.hpp>
#include <cute/atom/copy_traits.hpp>

#include <cute/arch/copy_xe.hpp>

namespace cute
{
template <class CopyOp>
struct XE_2D_LD_Unpack
{
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits const &traits,
                Tensor<ViewEngine<ArithmeticTupleIterator<TS>>, SLayout> const &src,
                Tensor<TD, DLayout> &dst)
    {
        static_assert(is_rmem<TD>::value);
        int H = size<0>(traits.tensor);
        // int W = size<1>(traits.tensor) * sizeof(typename decltype(traits.tensor)::engine_type::value_type);
        int W = size<1>(traits.tensor) * sizeof(typename Copy_Traits::CopyInternalType); //TODO: inconsistent to give the size in elements but use vector for copy
        auto [y, x] = src.data().coord_;
        CopyOp::copy(traits.tensor.data().get(), W, H, W, int2_{x, y}, &*dst.data());
    }
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X8X16X1X1_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X8X16X1X1_LD_N>
{
    // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per subgroup) - but static_assert fails
    using ThrID = Layout<_1>;
    using NumBits = Int<sizeof(typename GTensor::engine_type::value_type) * 8>; // hacky: does vec of 8
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, NumBits>>; // TODO:  is _1 correct?
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, NumBits>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
    using CopyInternalType = ushort;

    GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U32X8X16X1X1_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U32X8X16X1X1_LD_N>
{
    // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per subgroup) - but static_assert fails
    using ThrID = Layout<_1>;
    using NumBits = Int<sizeof(typename GTensor::engine_type::value_type) * 8>; // hacky: does vec of 8
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, NumBits>>; // TODO:  is _1 correct?
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, NumBits>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
    using CopyInternalType = uint;
    
    GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X8X16X4X2_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X8X16X4X2_LD_N>
{
  // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per
  // subgroup) - but static_assert fails
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =
      Layout<Shape<_16, Shape<Shape<_8, _4>, Shape<_16, _2>>>,
             Stride<_16, Stride<Stride<_512, _4096>, Stride<_1, _256>>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;

  GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X8X16X2X2_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X8X16X2X2_LD_N>
{
  // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per
  // subgroup) - but static_assert fails
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =
      Layout<Shape<_16, Shape<Shape<_8, _2>, Shape<_16, _2>>>,
             Stride<_16, Stride<Stride<_512, _4096>, Stride<_1, _256>>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;

  GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X8X16X1X2_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X8X16X1X2_LD_N>
{
  // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per
  // subgroup) - but static_assert fails
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout = Layout<Shape<_16, Shape<_8, Shape<_16, _2>>>,
                           Stride<_16, Stride<_512, Stride<_1, _256>>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;

  GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X8X16X4X1_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X8X16X4X1_LD_N>
{
  // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per
  // subgroup) - but static_assert fails
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =
      Layout<Shape<_16, Shape<_32, _16>>, Stride<_16, Stride<_256, _1>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  using CopyInternalType = ushort;

  GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U32X8X16X2X1_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U32X8X16X2X1_LD_N>
{
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =
      Layout<Shape<_16, Shape<_16, _32>>, Stride<_32, Stride<_512, _1>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  // 32 bits register file
  using CopyInternalType = uint;

  GTensor tensor;
};

template <class GTensor>
struct Copy_Traits<XE_2D_U16X16X16X2X1_LD_N, GTensor>
     : XE_2D_LD_Unpack<XE_2D_U16X16X16X2X1_LD_N>
{
  // // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is
  // per subgroup) - but static_assert fails
  using ThrID = Layout<_16>;
  // Map from (src-thr,src-val) to bit
  using SrcLayout = Layout<Shape<_16, _64>, Stride<_0, _1>>;
  // Map from (dst-thr,dst-val) to bit
  using DstLayout =
      Layout<Shape<_16, Shape<_16, _32>>, Stride<_32, Stride<_512, _1>>>;
  // Reference map from (thr,val) to bit
  using RefLayout = DstLayout;
  // 32 bits register file
  using CopyInternalType = uint;

  GTensor tensor;
};

template <class CopyOp>
struct XE_2D_ST_Unpack
{
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits const &traits,
                Tensor<TS, SLayout> const &src,
                Tensor<ViewEngine<ArithmeticTupleIterator<TD>>, DLayout> &dst)
    {
        static_assert(is_rmem<TS>::value);
        int H = size<0>(traits.tensor);
        int W = size<1>(traits.tensor) * sizeof(typename decltype(traits.tensor)::engine_type::value_type);
        auto [y, x] = dst.data().coord_;
        CopyOp::copy(traits.tensor.data().get(), W, H, W, int2_{x, y}, &*src.data());
    }
};

template <class GTensor>
struct Copy_Traits<XE_2D_U32X8X16X1X1_ST_N, GTensor>
     : XE_2D_ST_Unpack<XE_2D_U32X8X16X1X1_ST_N>
{
    // using ThrID   = Layout<_16>; //TODO: I think it should be 16 (copy is per subgroup) - but static_assert fails
    using ThrID = Layout<_1>;
    using NumBits = Int<sizeof(typename GTensor::engine_type::value_type) * 8>; // hacky: does vec of 8
    // Map from (src-thr,src-val) to bit
    using SrcLayout = Layout<Shape<_1, NumBits>>; // TODO:  is _1 correct?
    // Map from (dst-thr,dst-val) to bit
    using DstLayout = Layout<Shape<_1, NumBits>>;
    // Reference map from (thr,val) to bit
    using RefLayout = SrcLayout;
    GTensor tensor;
    template <class TS, class SLayout,
              class TD, class DLayout>
    CUTE_HOST_DEVICE friend constexpr void
    copy_unpack(Copy_Traits const &traits,
                Tensor<TS, SLayout> const &src,
                Tensor<ViewEngine<ArithmeticTupleIterator<TD>>, DLayout> &dst)
    {
        static_assert(is_rmem<TS>::value);
        int H = size<0>(traits.tensor);
        int W = size<1>(traits.tensor) * sizeof(typename decltype(traits.tensor)::engine_type::value_type);
        auto [y, x] = dst.data().coord_;
        XE_2D_SAVE::copy(traits.tensor.data().get(), W, H, W, int2_{x, y}, &*src.data());
    }
};

template <class Copy, class GEngine, class GLayout>
auto make_xe_2d_copy(Tensor<GEngine, GLayout> gtensor)
{
    using GTensor = Tensor<GEngine, GLayout>;
    using Traits = Copy_Traits<Copy, GTensor>;
    Traits traits{gtensor};
    return Copy_Atom<Traits, typename GEngine::value_type>{traits};
}
} // end namespace cute
