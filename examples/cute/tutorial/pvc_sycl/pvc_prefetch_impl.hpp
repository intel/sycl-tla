// #pragma once

#include "pvc_sycl_builtins.hpp"

#define HELPER_NAMEX(PREFIX, MM, NN) PREFIX##_m##MM##_n##NN
#define HELPER_NAME(PREFIX, MM, NN) HELPER_NAMEX(PREFIX, MM, NN)

void HELPER_NAME(atile_block_prefetch_rowmajor, MM,
                 NN)(global ushort *A, int tM, uint32_t M, uint32_t K, int m,
                     int k) {
  if constexpr (KK == 2 & MM == 4 & SGS_PER_WG_X >= 4) {
    const uint32_t sg_index_x =
        get_sub_group_id() % SGS_PER_WG_X; // index in [0, SGS_PER_WG_X)
    const uint32_t kk = 0;
    const int mm = sg_index_x % 4;
    // if (get_sub_group_local_id() == 0) {
    //     printf("atile block prefetch: %d, %d, %2d: sg_x = %d, m = %3d, k =
    //     %3d, mm = %2d, kk = %2d, coord = %3d, %3d\n", (int)get_group_id(1),
    //     (int)get_group_id(0), get_sub_group_id(), sg_index_x, m, k, mm, kk, k
    //     + kk * tK, m + mm * tM);
    // }
    intel_subgroup_block_prefetch_u16_m8k16v2(
        A, K * sizeof(ushort), M, K * sizeof(ushort), (int2_){k, m + mm * tM});
  } else if constexpr (KK % 2 == 0 & MM % 4 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int mm = 0; mm < MM; mm += 4) {
        intel_subgroup_block_prefetch_u16_m32k16v2(
            A, K * sizeof(ushort), M, K * sizeof(ushort),
            (int2_){k + kk * tK, m + mm * tM});
      }
    }
  } else if constexpr (KK % 2 == 0 & MM % 2 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int mm = 0; mm < MM; mm += 2) {
        intel_subgroup_block_prefetch_u16_m16k16v2(
            A, K * sizeof(ushort), M, K * sizeof(ushort),
            (int2_){k + kk * tK, m + mm * tM});
      }
    }
  } else if constexpr (KK % 2 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int mm = 0; mm < MM; mm++) {
        intel_subgroup_block_prefetch_u16_m8k16v2(
            A, K * sizeof(ushort), M, K * sizeof(ushort),
            (int2_){k + kk * tK, m + mm * tM});
      }
    }
  } else if constexpr (MM % 4 == 0) {
    for (int kk = 0; kk < KK; kk++) {
      for (int mm = 0; mm < MM; mm += 4) {
        intel_subgroup_block_prefetch_u16_m32k16(
            A, K * sizeof(ushort), M, K * sizeof(ushort),
            (int2_){k + kk * tK, m + mm * tM});
      }
    }
  } else {
    for (int kk = 0; kk < KK; kk++) {
      for (int mm = 0; mm < MM; mm++) {
        intel_subgroup_block_prefetch_u16_m8k16(
            A, K * sizeof(ushort), M, K * sizeof(ushort),
            (int2_){k + kk * tK, m + mm * tM});
      }
    }
  }
}

void HELPER_NAME(btile_block_prefetch_rowmajor, MM,
                 NN)(global ushort *B, int tN, int K, int N, int k, int n) {
  if (KK == 2 & NN == 4 & SGS_PER_WG_Y >= 4) {
    const int sg_index_y =
        get_sub_group_id() / SGS_PER_WG_X; // index in [0, SGS_PER_WG_Y)
    const int nn =
        sg_index_y % 2 * 2; // nn(sg_index_y) == 0, 2, 0, 2, 0, 2, 0, 2, ...
    const int kk =
        sg_index_y / 2 % 2; // kk(sg_index_y) == 0, 0, 1, 1, 0, 0, 1, 1, ...
    // if (get_sub_group_local_id() == 0) {
    //     printf("btile block prefetch: %d, %d, %2d: sg_y = %d, n = %3d, k =
    //     %3d, nn = %2d, kk = %2d, coord = %3d, %3d\n", (int)get_group_id(1),
    //     (int)get_group_id(0), get_sub_group_id(), sg_index_y, n, k, nn, kk, n
    //     + nn * tN, k + kk * tK);
    // }
    intel_subgroup_block_prefetch_u16_m16k16v2(
        B, N * sizeof(ushort), K, N * sizeof(ushort),
        (int2_){n + nn * tN, k + kk * tK});
  } else if (KK % 2 == 0 & NN % 2 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int nn = 0; nn < NN; nn += 2) {
        intel_subgroup_block_prefetch_u16_m32k16v2(
            B, N * sizeof(ushort), K, N * sizeof(ushort),
            (int2_){n + nn * tN, k + kk * tK});
      }
    }
  } else if (NN % 2 == 0) {
    for (int kk = 0; kk < KK; kk++) {
      for (int nn = 0; nn < NN; nn += 2) {
        intel_subgroup_block_prefetch_u16_m16k16v2(
            B, N * sizeof(ushort), K, N * sizeof(ushort),
            (int2_){n + nn * tN, k + kk * tK});
      }
    }
  } else if (KK % 2 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int nn = 0; nn < NN; nn++) {
        intel_subgroup_block_prefetch_u16_m32k16(
            B, N * sizeof(ushort), K, N * sizeof(ushort),
            (int2_){n + nn * tN, k + kk * tK});
      }
    }
  } else {
    for (int kk = 0; kk < KK; kk++) {
      for (int nn = 0; nn < NN; nn++) {
        intel_subgroup_block_prefetch_u16_m16k16(
            B, N * sizeof(ushort), K, N * sizeof(ushort),
            (int2_){n + nn * tN, k + kk * tK});
      }
    }
  }
}

void HELPER_NAME(btile_block_prefetch_vnni, MM,
                 NN)(global ushort *B, int tN, int K, int N, int k, int n) {
  if constexpr (KK == 2 & NN == 4 & SGS_PER_WG_Y >= 4) {
    const unsigned int sg_index_y =
        get_sub_group_id() / SGS_PER_WG_X; // index in [0, SGS_PER_WG_Y)
    const int nn = sg_index_y % 4; // nn(sg_index_y) == 0, 1, 2, 3, 0, 1, 2, 3
    // static const unsigned int kk = 0;              // kk(sg_index_y) == 0, 0,
    // 0, 0, 0, 0, 0, 0
    intel_subgroup_block_prefetch_u32_m16k16(
        B, N * sizeof(uint), K, N * sizeof(uint), (int2_){n + nn * tN, k / 2});
  } else if constexpr (KK % 2 == 0) {
    for (int kk = 0; kk < KK; kk += 2) {
      for (int nn = 0; nn < NN; nn++) {
        intel_subgroup_block_prefetch_u32_m16k16(
            B, N * sizeof(uint), K, N * sizeof(uint),
            (int2_){n + nn * tN, (k + kk * tK) / 2});
      }
    }
  } else {
    for (int kk = 0; kk < KK; kk++) {
      for (int nn = 0; nn < NN; nn++) {
        intel_subgroup_block_prefetch_u32_m8k16(
            B, N * sizeof(uint), K, N * sizeof(uint),
            (int2_){n + nn * tN, (k + kk * tK) / 2});
      }
    }
  }
}
