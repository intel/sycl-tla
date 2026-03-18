#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <cassert>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/command_line.h"
#include "params.hpp"
#include "gemm_dq_kernel.hpp"

using namespace cute;

template<class...> class gemmDqDeviceName;

// ---------------------------------------------------------------------------
// CPU reference: dQ[m,d] = sum_n dP[m,n] * K[n,d]
// dP is column-major (stride 1 on M, kBlockM on N)
// K  is column-major in the transposed view: K[D,N] with stride 1 on D
// dQ is row-major (stride head_dim on M, 1 on D)
// ---------------------------------------------------------------------------
template<typename T>
void gemm_dq_reference(const T *dP, const T *K,
                        float *dQ,
                        int M, int N, int D,
                        int dP_m_stride, int dP_n_stride,
                        int K_d_stride, int K_n_stride,
                        int dQ_m_stride, int dQ_d_stride) {
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int n = 0; n < N; ++n) {
                float a = static_cast<float>(dP[m * dP_m_stride + n * dP_n_stride]);
                float b = static_cast<float>(K[d * K_d_stride + n * K_n_stride]);
                acc += a * b;
            }
            dQ[m * dQ_m_stride + d * dQ_d_stride] += acc;
        }
    }
}

template<typename T>
void uniform_init(int seed, T *dst, size_t N, float a = -0.5f, float b = 0.5f) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<float> dis(a, b);
    for (size_t i = 0; i < N; ++i)
        dst[i] = static_cast<T>(dis(gen));
}

template<typename T>
bool verify(const float *ref, const float *test, int M, int D,
            float atol = 1e-2f, float rtol = 1e-2f) {
    int mismatches = 0;
    float max_diff = 0.0f;
    for (int m = 0; m < M; ++m) {
        for (int d = 0; d < D; ++d) {
            float r = ref[m * D + d];
            float t = test[m * D + d];
            float diff = std::abs(r - t);
            float denom = std::max(std::abs(r), std::abs(t));
            max_diff = std::max(max_diff, diff);
            if (diff > atol + rtol * denom) {
                if (mismatches < 10)
                    printf("  MISMATCH (%d,%d) ref=%.6f test=%.6f diff=%.6f\n",
                           m, d, r, t, diff);
                mismatches++;
            }
        }
    }
    printf("  max_diff=%.6f mismatches=%d/%d %s\n",
           max_diff, mismatches, M * D,
           mismatches == 0 ? "\033[32mPASS\033[0m" : "\033[31mFAIL\033[0m");
    return mismatches == 0;
}

// ---------------------------------------------------------------------------
// Kernel launcher: reproduces the dQ GEMM from sdpa_backward
//
// The backward pass iterates over N-blocks (grid X), and for each N-block
// iterates over M-blocks, computing dQ[M_tile, D] += dP[M_tile, N_tile] * K[D, N_tile]^T
// with atomicAdd to dQaccum.
// ---------------------------------------------------------------------------
template<class...> class gemmDqKernel;

template<typename T, int kBlockM_, int kBlockN_, int kHeadDim_,
         int kNSGs_, int AtomLayoutMdQ_>
struct DqTrait {
    using DType = T;
    using VType = float;
    static constexpr int kBlockM = kBlockM_;
    static constexpr int kBlockN = kBlockN_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kNSGs = kNSGs_;
    static constexpr int AtomLayoutMdQ = AtomLayoutMdQ_;
    static constexpr int SubgroupSize = 16;
    using MMA_Atom_ARCH = XE_DPAS_TT<8, VType, DType>;
    using _K = Int<MMA_Atom_ARCH::K>;
    using SubgroupLayoutdQ = Layout<Shape<Int<AtomLayoutMdQ>, Int<kNSGs / AtomLayoutMdQ>, _1>>;
    using TileShapedQ = Layout<Shape<Int<kBlockM>, Int<kHeadDim>, _K>>;
    using TiledMmadQ = typename TiledMMAHelper<MMA_Atom<MMA_Atom_ARCH>,
                                                TileShapedQ,
                                                SubgroupLayoutdQ>::TiledMMA;
};

template<typename Trait>
void gemm_dq_device_kernel(Trait trait,
                           const typename Trait::DType *dP_ptr,
                           const typename Trait::DType *K_ptr,
                           float *dQ_ptr,
                           int seq_len_q, int seq_len_kv, int head_dim,
                           int dP_row_stride,     // stride between rows of dP in the buffer
                           int K_col_stride,      // stride between columns (rows in original K)
                           int dQ_row_stride,     // stride between rows of dQ
                           int n_blocks) {
    using T = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kHeadDim = Trait::kHeadDim;

    // This workgroup handles one N-block, iterates over all M-blocks
    const int n_block = BlockIdxX();
    const int bidh = BlockIdxY();
    const int bidb = BlockIdxZ();

    const int batch_head_offset_dP = (bidb * GridDimY() + bidh) * seq_len_q * seq_len_kv;
    const int batch_head_offset_K = (bidb * GridDimY() + bidh) * seq_len_kv * head_dim;
    const int batch_head_offset_dQ = (bidb * GridDimY() + bidh) * seq_len_q * head_dim;

    const int n_offset = n_block * kBlockN;
    const int actual_n = (n_offset + kBlockN <= seq_len_kv) ? kBlockN : (seq_len_kv - n_offset);
    // Round up to even for block2D
    const int block_n_dim = (actual_n + 1) & ~1;

    const int M_blocks = (seq_len_q + kBlockM - 1) / kBlockM;

    typename Trait::TiledMmadQ tiled_mma_dq;

    for (int m_block = 0; m_block < M_blocks; ++m_block) {
        const int m_offset = m_block * kBlockM;
        const int actual_m = (m_offset + kBlockM <= seq_len_q) ? kBlockM : (seq_len_q - m_offset);

        // dP tile: [actual_m, block_n_dim] column-major
        // dP layout in memory: dP[m, n] at offset m + n * dP_row_stride
        // For the GEMM: A = dP[M, N] with stride (_1, dP_row_stride)
        auto mdP = make_tensor(
            make_gmem_ptr(dP_ptr + batch_head_offset_dP + m_offset + n_offset * dP_row_stride),
            make_layout(
                make_shape(actual_m, block_n_dim),
                make_stride(_1{}, dP_row_stride)));

        // K tile: [kHeadDim, block_n_dim] column-major
        // K[d, n] at offset d + n * K_col_stride
        auto mK = make_tensor(
            make_gmem_ptr(K_ptr + batch_head_offset_K + n_offset * K_col_stride),
            make_layout(
                make_shape(Int<kHeadDim>{}, block_n_dim),
                make_stride(_1{}, K_col_stride)));

        // dQaccum: [kBlockM, kHeadDim] row-major f32
        auto mdQaccum = make_tensor(
            make_gmem_ptr(dQ_ptr + batch_head_offset_dQ + m_offset * dQ_row_stride),
            make_layout(
                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                make_stride(dQ_row_stride, _1{})));

        gemm_dQ_atomic(trait, mdP, mK, mdQaccum, tiled_mma_dq);
    }
}

// ---------------------------------------------------------------------------
// SIMD32 atomic variant device kernel.
// Same grid/structure as baseline, but uses gemm_dQ_atomic_simd32.
// ---------------------------------------------------------------------------
template<class...> class gemmDqSimd32Name;

template<typename Trait>
void gemm_dq_simd32_device_kernel(Trait trait,
                                   const typename Trait::DType *dP_ptr,
                                   const typename Trait::DType *K_ptr,
                                   float *dQ_ptr,
                                   int seq_len_q, int seq_len_kv, int head_dim,
                                   int dP_row_stride, int K_col_stride, int dQ_row_stride,
                                   int n_blocks) {
    using T = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kHeadDim = Trait::kHeadDim;

    const int n_block = BlockIdxX();
    const int bidh = BlockIdxY();
    const int bidb = BlockIdxZ();

    const int batch_head_offset_dP = (bidb * GridDimY() + bidh) * seq_len_q * seq_len_kv;
    const int batch_head_offset_K = (bidb * GridDimY() + bidh) * seq_len_kv * head_dim;
    const int batch_head_offset_dQ = (bidb * GridDimY() + bidh) * seq_len_q * head_dim;

    const int n_offset = n_block * kBlockN;
    const int actual_n = (n_offset + kBlockN <= seq_len_kv) ? kBlockN : (seq_len_kv - n_offset);
    const int block_n_dim = (actual_n + 1) & ~1;

    const int M_blocks = (seq_len_q + kBlockM - 1) / kBlockM;
    typename Trait::TiledMmadQ tiled_mma_dq;

    for (int m_block = 0; m_block < M_blocks; ++m_block) {
        const int m_offset = m_block * kBlockM;
        const int actual_m = (m_offset + kBlockM <= seq_len_q) ? kBlockM : (seq_len_q - m_offset);

        auto mdP = make_tensor(
            make_gmem_ptr(dP_ptr + batch_head_offset_dP + m_offset + n_offset * dP_row_stride),
            make_layout(
                make_shape(actual_m, block_n_dim),
                make_stride(_1{}, dP_row_stride)));

        auto mK = make_tensor(
            make_gmem_ptr(K_ptr + batch_head_offset_K + n_offset * K_col_stride),
            make_layout(
                make_shape(Int<kHeadDim>{}, block_n_dim),
                make_stride(_1{}, K_col_stride)));

        auto mdQaccum = make_tensor(
            make_gmem_ptr(dQ_ptr + batch_head_offset_dQ + m_offset * dQ_row_stride),
            make_layout(
                Shape<Int<kBlockM>, Int<kHeadDim>>{},
                make_stride(dQ_row_stride, _1{})));

        gemm_dQ_atomic_simd32(trait, mdP, mK, mdQaccum, tiled_mma_dq);
    }
}

// ---------------------------------------------------------------------------
// K-sliced device kernel.
// Grid: (N_BLOCKS, heads, batch) — same as baseline.
// WG has kNSGs * K_SLICES SGs total (128-GRF for 2x occupancy).
// K_SLICES groups of kNSGs SGs each handle kBlockN/K_SLICES of the K dim.
// Each group produces partial dQ, atomicAdds to SLM. Then one global atomicAdd.
// ---------------------------------------------------------------------------
template<class...> class gemmDqKsliceName;

template<typename Trait, int K_SLICES>
void gemm_dq_kslice_device_kernel(Trait trait,
                                   const typename Trait::DType *dP_ptr,
                                   const typename Trait::DType *K_ptr,
                                   float *dQ_ptr,
                                   int seq_len_q, int seq_len_kv, int head_dim,
                                   int dP_row_stride, int K_col_stride, int dQ_row_stride,
                                   int n_blocks) {
    using T = typename Trait::DType;
    constexpr int kBlockM = Trait::kBlockM;
    constexpr int kBlockN = Trait::kBlockN;
    constexpr int kHeadDim = Trait::kHeadDim;
    constexpr int kNSGs = Trait::kNSGs;              // SGs per K-slice group
    constexpr int SubgroupSize = Trait::SubgroupSize;
    constexpr int kNSGs_total = kNSGs * K_SLICES;
    constexpr int kBlockN_per_slice = kBlockN / K_SLICES;

    static_assert(kBlockN % K_SLICES == 0, "kBlockN must be divisible by K_SLICES");

    const int n_block = BlockIdxX();
    const int bidh = BlockIdxY();
    const int bidb = BlockIdxZ();

    const int batch_head_offset_dP = (bidb * GridDimY() + bidh) * seq_len_q * seq_len_kv;
    const int batch_head_offset_K = (bidb * GridDimY() + bidh) * seq_len_kv * head_dim;
    const int batch_head_offset_dQ = (bidb * GridDimY() + bidh) * seq_len_q * head_dim;

    const int n_offset = n_block * kBlockN;
    const int actual_n = (n_offset + kBlockN <= seq_len_kv) ? kBlockN : (seq_len_kv - n_offset);

    // SLM accumulator: [kBlockM, kHeadDim] f32
    auto slm_raw = compat::local_mem<float[kBlockM * kHeadDim]>();
    auto group = compat::get_nd_item<1>().get_group();
    auto sg = compat::get_nd_item<1>().get_sub_group();
    const int tid = ThreadIdxX();
    const int num_threads = kNSGs_total * SubgroupSize;

    // Which K-slice group does this SG belong to?
    const int sg_id_global = sg.get_group_linear_id();      // 0..kNSGs_total-1
    const int k_group = sg_id_global / kNSGs;                // 0..K_SLICES-1
    const int sg_id_local = sg_id_global % kNSGs;            // 0..kNSGs-1
    const int local_thr_idx = sg_id_local * SubgroupSize;    // thread idx within group
    const int local_id = sg.get_local_id();                  // lane id

    typename Trait::TiledMmadQ tiled_mma_dq;
    auto tile_mnk = tiled_mma_dq.tile_mnk();

    // C partition — use local_thr_idx so each group sees the same MMA layout
    auto dQ_shape = Shape<Int<kBlockM>, Int<kHeadDim>>{};
    Tensor cC = make_identity_tensor(dQ_shape);
    Tensor gC = local_tile(cC, select<0, 1>(tile_mnk), make_coord(0, 0));
    auto thr_mma = tiled_mma_dq.get_slice(local_thr_idx);
    Tensor tCgC = thr_mma.partition_C(gC);

    const int M_blocks = (seq_len_q + kBlockM - 1) / kBlockM;

    // K-offset for this group's slice
    const int k_slice_offset = k_group * kBlockN_per_slice;

    for (int m_block = 0; m_block < M_blocks; ++m_block) {
        const int m_offset = m_block * kBlockM;
        const int actual_m = (m_offset + kBlockM <= seq_len_q) ? kBlockM : (seq_len_q - m_offset);

        // 1. Clear SLM cooperatively (all kNSGs_total * SubgroupSize threads)
        for (int i = tid; i < kBlockM * kHeadDim; i += num_threads)
            slm_raw[i] = 0.0f;
        sycl::group_barrier(group);

        // 2. Each K-slice group computes partial GEMM on its K-slice
        //    A = dP[m_offset:, n_offset+k_slice_offset:] shape (actual_m, kBlockN_per_slice)
        //    B = K[:, n_offset+k_slice_offset:]           shape (kHeadDim, kBlockN_per_slice)
        {
            auto mdP = make_tensor(
                make_gmem_ptr(dP_ptr + batch_head_offset_dP + m_offset
                              + (n_offset + k_slice_offset) * dP_row_stride),
                make_layout(
                    make_shape(actual_m, Int<kBlockN_per_slice>{}),
                    make_stride(_1{}, dP_row_stride)));

            auto mK = make_tensor(
                make_gmem_ptr(K_ptr + batch_head_offset_K
                              + (n_offset + k_slice_offset) * K_col_stride),
                make_layout(
                    make_shape(Int<kHeadDim>{}, Int<kBlockN_per_slice>{}),
                    make_stride(_1{}, K_col_stride)));

            // GEMM: partial dQ = dP_slice × K_slice^T
            auto tCrC = thr_mma.partition_sg_fragment_C(
                make_identity_tensor(select<0,1>(tile_mnk)));
            gemm_engine_no_barrier(trait, mdP, mK, tCrC, tiled_mma_dq, local_thr_idx);

            // atomicAdd partial to SLM
            CUTLASS_PRAGMA_UNROLL
            for (int i = 0; i < size(tCgC); ++i) {
                auto [m, n] = tCgC(i);
                slm_atomic_add(&slm_raw[m * kHeadDim + n + local_id], tCrC(i));
            }
        }

        // 3. Barrier: all K-slice groups done
        sycl::group_barrier(group);

        // 4. Single atomicAdd from SLM to global
        float *dQ_global = dQ_ptr + batch_head_offset_dQ + m_offset * dQ_row_stride;
        for (int i = tid; i < actual_m * kHeadDim; i += num_threads) {
            int m = i / kHeadDim;
            int d = i % kHeadDim;
            cutlass::atomicAdd(&dQ_global[m * dQ_row_stride + d], slm_raw[m * kHeadDim + d]);
        }

        // Barrier before next M-block
        sycl::group_barrier(group);
    }
}

struct Options {
    int batch = 4;
    int num_heads = 5;
    int seq_len_q = 512;
    int seq_len_kv = 512;
    int head_dim = 128;
    int iterations = 20;
    bool is_bf16 = true;
    bool help = false;

    void parse(int argc, const char **argv) {
        cutlass::CommandLine cmd(argc, argv);
        cmd.get_cmd_line_argument("batch", batch);
        cmd.get_cmd_line_argument("num_heads", num_heads);
        cmd.get_cmd_line_argument("seq_len_q", seq_len_q);
        cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv);
        cmd.get_cmd_line_argument("head_dim", head_dim);
        cmd.get_cmd_line_argument("iterations", iterations);
        if (cmd.check_cmd_line_flag("is_bf16")) is_bf16 = true;
        if (cmd.check_cmd_line_flag("is_fp16")) is_bf16 = false;
        if (cmd.check_cmd_line_flag("help")) help = true;
    }

    void print_usage(std::ostream &os) {
        os << "gemm_dQ reproducer\n\n"
           << "  --batch=<int>       Batch size [" << batch << "]\n"
           << "  --num_heads=<int>   Number of heads [" << num_heads << "]\n"
           << "  --seq_len_q=<int>   Query sequence length [" << seq_len_q << "]\n"
           << "  --seq_len_kv=<int>  KV sequence length [" << seq_len_kv << "]\n"
           << "  --head_dim=<int>    Head dimension [" << head_dim << "]\n"
           << "  --iterations=<int>  Benchmark iterations [" << iterations << "]\n"
           << "  --is_bf16           Use BF16 [default]\n"
           << "  --is_fp16           Use FP16\n";
    }
};

template<typename T, int kBlockM, int kBlockN, int kHeadDim, int kNSGs, int AtomLayoutMdQ>
void run_gemm_dq(int batch, int num_heads, int seq_len_q, int seq_len_kv,
                 int iterations, bool check) {
    using Trait = DqTrait<T, kBlockM, kBlockN, kHeadDim, kNSGs, AtomLayoutMdQ>;
    Trait trait{};

    const size_t dP_size = (size_t)batch * num_heads * seq_len_q * seq_len_kv;
    const size_t K_size = (size_t)batch * num_heads * seq_len_kv * kHeadDim;
    const size_t dQ_size = (size_t)batch * num_heads * seq_len_q * kHeadDim;

    // Host
    std::vector<T> dP_h(dP_size), K_h(K_size);
    std::vector<float> dQ_h(dQ_size, 0.0f), dQ_ref(dQ_size, 0.0f);

    uniform_init(42, dP_h.data(), dP_size);
    uniform_init(43, K_h.data(), K_size);

    // Device
    T *dP_d = compat::malloc<T>(dP_size);
    T *K_d = compat::malloc<T>(K_size);
    float *dQ_d = compat::malloc<float>(dQ_size);

    compat::memcpy<T>(dP_d, dP_h.data(), dP_size);
    compat::memcpy<T>(K_d, K_h.data(), K_size);
    compat::fill(dQ_d, 0.0f, dQ_size);
    compat::wait_and_throw();

    const int N_BLOCKS = (seq_len_kv + kBlockN - 1) / kBlockN;

    // dP layout: [batch, head, M, N] row-major → for column-major GEMM view,
    // dP_row_stride = seq_len_kv (stride between M rows when indexing as dP[m + n*stride])
    // Wait — we need dP in column-major for the GEMM. In SDPA backward, dP is stored
    // in a scratch buffer with stride kBlockM on the N dimension.
    // For the reproducer, store dP as [batch*head, seq_len_q, seq_len_kv] row-major,
    // then the GEMM views a [M_tile, N_tile] slice with stride (_1, seq_len_q) for column-major.
    // Actually: dP[m,n] stored as row-major means dP[m*seq_len_kv + n].
    // For the GEMM A matrix (column-major view): A[m, n] = dP[m + n * seq_len_q]
    // So we need dP in column-major: dP[m + n * seq_len_q]
    // Let's just store dP column-major to match the backward's scratch buffer layout.
    // dP_row_stride = seq_len_q (stride to next column = stride on the N dimension)

    // Actually in the real backward, dP is in a scratch buffer with:
    //   shapeSP = (kBlockM, block_n_dim), stride = (_1, kBlockM)
    // So it's column-major with M-stride=1 and N-stride=kBlockM.
    // For the full matrix, we can use M-stride=1 and N-stride=seq_len_q.

    // K layout: [batch, head, N, D] row-major → K[n*head_dim + d]
    // For GEMM B = K[D, N] column-major: B[d, n] = K[n*head_dim + d]
    // So B[d + n * head_dim] → K_col_stride = head_dim

    // dQ layout: [batch, head, M, D] row-major → dQ[m*head_dim + d]
    // dQ_row_stride = head_dim

    const int dP_row_stride = seq_len_q;  // column-major: stride on N dim = seq_len_q
    const int K_col_stride = kHeadDim;    // stride between K rows (N dim)
    const int dQ_row_stride = kHeadDim;

    // Re-init dP in column-major layout for correctness
    // dP_colmaj[m + n * seq_len_q] = dP_h[bh * sq * skv + m * skv + n] (conceptually)
    // For simplicity, just use dP as-is and set strides accordingly
    // dP stored as [batch*head*seq_len_q*seq_len_kv] where dP[m,n] = dP_h[m*seq_len_kv + n]
    // For column-major GEMM: A[m,n] at address m + n * stride
    // So dP_row_stride (N stride) = seq_len_q won't work with row-major storage.
    // Let's just store in column-major: dP_cm[m + n * seq_len_q]
    {
        std::vector<T> dP_cm(dP_size);
        for (int bh = 0; bh < batch * num_heads; ++bh) {
            for (int m = 0; m < seq_len_q; ++m) {
                for (int n = 0; n < seq_len_kv; ++n) {
                    // row-major index: bh * sq * skv + m * skv + n
                    // col-major index: bh * sq * skv + m + n * sq
                    dP_cm[bh * seq_len_q * seq_len_kv + m + n * seq_len_q] =
                        dP_h[bh * seq_len_q * seq_len_kv + m * seq_len_kv + n];
                }
            }
        }
        compat::memcpy<T>(dP_d, dP_cm.data(), dP_size);
        compat::wait_and_throw();
    }

    // Reference: dQ[m,d] += sum_n dP[m,n] * K[n,d]
    if (check) {
        printf("Computing reference...\n");
        for (int bh = 0; bh < batch * num_heads; ++bh) {
            const T *dP_bh = dP_h.data() + bh * seq_len_q * seq_len_kv;  // row-major
            const T *K_bh = K_h.data() + bh * seq_len_kv * kHeadDim;      // row-major K[n,d]
            float *dQ_bh = dQ_ref.data() + bh * seq_len_q * kHeadDim;
            // dQ[m,d] = sum_n dP[m,n] * K[n,d]
            // dP row-major: dP[m*skv + n], K row-major: K[n*hd + d]
            for (int m = 0; m < seq_len_q; ++m) {
                for (int d = 0; d < kHeadDim; ++d) {
                    float acc = 0.0f;
                    for (int n = 0; n < seq_len_kv; ++n) {
                        acc += (float)dP_bh[m * seq_len_kv + n] * (float)K_bh[n * kHeadDim + d];
                    }
                    dQ_bh[m * kHeadDim + d] = acc;
                }
            }
        }
    }

    // SLM size for K-sliced variant
    constexpr int slm_bytes = kBlockM * kHeadDim * sizeof(float);

    // --- Baseline launcher: N-parallel grid, atomicAdd to global ---
    auto launch_baseline = [&]() {
        compat::fill(dQ_d, 0.0f, dQ_size);
        compat::wait_and_throw();

        auto dimGrid = compat::dim3(size(N_BLOCKS), size(num_heads), size(batch));
        auto dimBlock = compat::dim3(size(kNSGs * Trait::SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props{};
        compat::experimental::kernel_properties kernel_props{
            sycl::ext::oneapi::experimental::sub_group_size<Trait::SubgroupSize>};
        compat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
        auto event = compat::experimental::launch<
            gemm_dq_device_kernel<Trait>,
            gemmDqDeviceName<Trait>>(
                policy, trait,
                dP_d, K_d, dQ_d,
                seq_len_q, seq_len_kv, kHeadDim,
                dP_row_stride, K_col_stride, dQ_row_stride,
                N_BLOCKS);
        EventManager::getInstance().addEvent(event);
        compat::wait_and_throw();
    };

    // --- SIMD32 atomic launcher: same grid as baseline, halved atomics ---
    auto launch_simd32 = [&]() {
        compat::fill(dQ_d, 0.0f, dQ_size);
        compat::wait_and_throw();

        auto dimGrid = compat::dim3(size(N_BLOCKS), size(num_heads), size(batch));
        auto dimBlock = compat::dim3(size(kNSGs * Trait::SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props{};
        compat::experimental::kernel_properties kernel_props{
            sycl::ext::oneapi::experimental::sub_group_size<Trait::SubgroupSize>};
        compat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
        auto event = compat::experimental::launch<
            gemm_dq_simd32_device_kernel<Trait>,
            gemmDqSimd32Name<Trait>>(
                policy, trait,
                dP_d, K_d, dQ_d,
                seq_len_q, seq_len_kv, kHeadDim,
                dP_row_stride, K_col_stride, dQ_row_stride,
                N_BLOCKS);
        EventManager::getInstance().addEvent(event);
        compat::wait_and_throw();
    };

    // --- K-sliced launcher: split kBlockN across K_SLICES groups, SLM reduce ---
    // Same grid as baseline, but 2x SGs (128-GRF for occupancy)
    constexpr int K_SLICES = 2;
    constexpr int kNSGs_total = kNSGs * K_SLICES;
    auto launch_kslice = [&]() {
        compat::fill(dQ_d, 0.0f, dQ_size);
        compat::wait_and_throw();

        auto dimGrid = compat::dim3(size(N_BLOCKS), size(num_heads), size(batch));
        auto dimBlock = compat::dim3(size(kNSGs_total * Trait::SubgroupSize), size(1), size(1));
        compat::experimental::launch_properties launch_props{
            sycl::ext::oneapi::experimental::work_group_scratch_size(slm_bytes),
        };
        compat::experimental::kernel_properties kernel_props{
            sycl::ext::oneapi::experimental::sub_group_size<Trait::SubgroupSize>,
            sycl::ext::intel::experimental::grf_size<128>};
        compat::experimental::launch_policy policy{dimGrid, dimBlock, launch_props, kernel_props};
        auto event = compat::experimental::launch<
            gemm_dq_kslice_device_kernel<Trait, K_SLICES>,
            gemmDqKsliceName<Trait>>(
                policy, trait,
                dP_d, K_d, dQ_d,
                seq_len_q, seq_len_kv, kHeadDim,
                dP_row_stride, K_col_stride, dQ_row_stride,
                N_BLOCKS);
        EventManager::getInstance().addEvent(event);
        compat::wait_and_throw();
    };

    auto benchmark = [&](auto launch_fn, const char *name) {
        // Accuracy check
        if (check) {
            launch_fn();
            compat::memcpy<float>(dQ_h.data(), dQ_d, dQ_size);
            compat::wait_and_throw();
            printf("[%s] dQ accuracy: ", name);
            verify<T>(dQ_ref.data(), dQ_h.data(), batch * num_heads * seq_len_q, kHeadDim);
        }

        // Warmup
        for (int i = 0; i < 3; ++i) launch_fn();

        // Benchmark
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) launch_fn();
        auto t1 = std::chrono::high_resolution_clock::now();
        double us = std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()
                    / (double)iterations;

        double flops = 2.0 * batch * num_heads * seq_len_q * seq_len_kv * kHeadDim;
        double tflops = flops / (us * 1e-6) / 1e12;
        printf("[%s] gemm_dQ: %.1f us  %.2f TFLOPS  "
               "(batch=%d heads=%d sq=%d skv=%d hd=%d)\n",
               name, us, tflops, batch, num_heads, seq_len_q, seq_len_kv, kHeadDim);
    };

    benchmark(launch_baseline, "baseline");
    benchmark(launch_simd32,   "simd32 ");
    benchmark(launch_kslice,   "kslice ");

    compat::free(dP_d);
    compat::free(K_d);
    compat::free(dQ_d);
}

int main(int argc, const char **argv) {
    Options opts;
    opts.parse(argc, argv);
    if (opts.help) {
        opts.print_usage(std::cout);
        return 0;
    }

    printf("gemm_dQ reproducer: batch=%d heads=%d sq=%d skv=%d hd=%d %s\n",
           opts.batch, opts.num_heads, opts.seq_len_q, opts.seq_len_kv,
           opts.head_dim, opts.is_bf16 ? "BF16" : "FP16");

    constexpr int kBlockM = 64;
    constexpr int kBlockN = 64;
    constexpr int kNSGs = 8;
    constexpr int AtomLayoutMdQ = 4;  // for hd=128

    if (opts.head_dim == 128) {
        if (opts.is_bf16) {
            run_gemm_dq<cute::bfloat16_t, kBlockM, kBlockN, 128, kNSGs, AtomLayoutMdQ>(
                opts.batch, opts.num_heads, opts.seq_len_q, opts.seq_len_kv,
                opts.iterations, true);
        } else {
            run_gemm_dq<cute::half_t, kBlockM, kBlockN, 128, kNSGs, AtomLayoutMdQ>(
                opts.batch, opts.num_heads, opts.seq_len_q, opts.seq_len_kv,
                opts.iterations, true);
        }
    } else {
        printf("Only head_dim=128 supported in reproducer\n");
        return 1;
    }

    return 0;
}
