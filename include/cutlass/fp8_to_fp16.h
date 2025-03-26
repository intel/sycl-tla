#pragma once

#include <stdint.h>
#include <cstdint>
#include <cutlass/half.h> 
#include <cute/util/sycl_vec.hpp>

using half_t = cutlass::half_t;
using uchar4 = cute::intel::uchar4;
using ushort4 = cute::intel::ushort4;


union FP16Union {
    uint16_t i;
    half_t f;
};

static inline cute::intel::ushort16 convert_ushort16(cute::intel::uchar16 x) {
    cute::intel::ushort16 result;
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        result[i] = static_cast<uint16_t>(x[i]);
    }
    return result;
}

static inline cute::intel::ushort16 E4M3_to_FP16_vec16(cute::intel::uchar16 xin) {
    using namespace cute::intel;

    uchar16 xa = xin & 0x7F;
    uchar16 sgn_x = xin ^ xa;

    uchar16 zero_mask;
    for (int i = 0; i < 16; ++i) {
        zero_mask[i] = (xa[i] == 0) ? 1 : 0;
    }
    uchar16 nan_mask = (0x7E - xa) & 0x80;
    uchar16 den_mask = ((xa - 8) >> 7) & 0x01;

    xa += (nan_mask >> 1);
    xa |= (den_mask & 8);
    den_mask &= 0x48;
    xa += 0x40 & ~(zero_mask * 0x40);

    ushort16 x16 = convert_ushort16(xa) << 7;
    ushort16 den_corr = convert_ushort16(den_mask & ~zero_mask) << 7;

    ushort16 result = x16 - den_corr;
    result &= ~(convert_ushort16(zero_mask) << 7);

    ushort16 sign_ext = convert_ushort16(sgn_x) << 8;
    result ^= sign_ext;

    return result;
}

static inline half_t E4M3_to_FP16(uint8_t xin) {
    uint8_t xa = xin & 0x7F;
    uint8_t sgn_x = xin ^ xa;

    uint8_t nan_mask = (0x7E - xa) & 0x80;
    uint8_t den_mask = (((int8_t)(xa - 8)) >> 7);

    xa += (nan_mask >> 1);
    xa |= (den_mask & 8);
    den_mask &= 0x48;
    xa += 0x40;

    FP16Union x16, den_corr;
    x16.i = xa << 7;
    den_corr.i = den_mask << 7;

    x16.f -= den_corr.f;

    x16.i ^= ((uint16_t)sgn_x << 8);

    return x16.f;
}


template <int N>
struct E4M3_to_FP16_NumericArrayConverter {

    using result_type = cutlass::Array<half_t, N>;
    using source_type = cutlass::Array<uint8_t, N>;

    CUTLASS_HOST_DEVICE
    static result_type convert(source_type const& s) {
        result_type result;

        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; ++i) {
            result[i] = E4M3_to_FP16(s[i]);
        }

        return result;
    }

    CUTLASS_HOST_DEVICE
    result_type operator()(source_type const& s) const {
    return convert(s);
    }
};
