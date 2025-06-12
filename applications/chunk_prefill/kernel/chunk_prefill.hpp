#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"
#include "flash_attention_v2/collective/xe_flash_attn_prefill_mma.hpp"

namespace cutlass::flash_attention::kernel {

template <class ProblemShape_, class CollectiveMainloop_, class CollectiveSoftmaxEpilogue_, 
          class CollectiveEpilogue_, class TileScheduler_ = void>
class FMHAPrefillChunk {

public:
  using ProblemShape = ProblemShape_;
  
  // ProblemShape: <batch, num_heads_q, num_head_kv, seq_len_qo, seq_len_kv, chunk_size, head_size_qk, head_size_vo>
  static_assert(rank(ProblemShape{}) == 8, 
    "ProblemShape{} should be <batch, num_heads_q, num_head_kv, seq_len_qo, seq_len_kv, chunk_size, head_size_qk, head_size_vo>");

  // ... 其他类型定义与 cachedKV 版本相同 ...

  CUTLASS_DEVICE
  void operator()(Params const &params, char *smem_buf) {
    // ... 前置代码与 cachedKV 版本类似 ...
    
    auto& chunk_size = get<5>(params.problem_shape);  // 获取 chunk_size
    auto& seq_len_kv = get<4>(params.problem_shape); // 总的 seq_len_kv
    
    // 计算需要处理的 chunk 数量
    int num_chunks = cute::ceil_div(seq_len_kv, chunk_size);
    
    // 基础 tensor 定义
    Tensor mQ_mkl = cute::get_xe_tensor(make_shape(seq_len_qo, head_size_qk, 
                                      (is_var_len ? 1 : batch) * num_heads_q));   //(m,k,l)
    
    // 为每个 chunk 创建 K/V tensor
    for(int chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
      // 计算当前 chunk 的实际大小
      int current_chunk_size = cute::min(chunk_size, 
                                       seq_len_kv - chunk_idx * chunk_size);
      
      // 创建当前 chunk 的 K/V tensor
      Tensor mK_chunk_nkl = cute::get_xe_tensor(
        make_shape(current_chunk_size, head_size_qk, 
                  (is_var_len ? 1 : batch) * num_head_kv));   //(n_chunk,k,l)
      
      Tensor mV_chunk_nkl = cute::get_xe_tensor(
        make_shape(head_size_vo, current_chunk_size,
                  (is_var_len ? 1 : batch) * num_head_kv));   //(n_chunk,k,l)
      
      // 获取当前 chunk 的局部视图
      Tensor mK_chunk_nk = mK_chunk_nkl(_, _, blk_l_coord/group_heads_q);  // (n_chunk,k)
      Tensor mV_chunk_nk = mV_chunk_nkl(_, _, blk_l_coord/group_heads_q);  // (n_chunk,k)
      
      // 创建局部 tile
      auto gK_chunk = local_tile(mK_chunk_nk, TileShapeQK{}, 
                                make_coord(_, _, _), Step<X, _1, _1>{});
      auto gV_chunk = local_tile(mV_chunk_nk, TileShapeOutput{}, 
                                make_coord(_, blk_n_coord, _), Step<X, _1, _1>{});
      
      // ... prefetch 逻辑 ...
      auto tiled_prefetch_k_chunk = cute::prefetch_selector<
        Shape<Int<QK_BLK_N>, Int<cute::max(cute::gcd(QK_BLK_K, 64), 32)>>, 
        Num_SGs>(mainloop_params.gmem_tiled_copy_k);
        
      auto tiled_prefetch_v_chunk = cute::prefetch_selector<
        Shape<Int<cute::max(cute::gcd(Epilogue_BLK_N, 64), 32)>, Int<Epilogue_BLK_K>>, 
        Num_SGs>(mainloop_params.gmem_tiled_copy_v);
      
      // ... 主循环逻辑 ...
      for (int nblock = 0; nblock < cute::ceil_div(current_chunk_size, QK_BLK_N); ++nblock) {
        barrier_arrive(barrier_scope);
        
        // 1) 创建 S tensor
        Tensor tSr = make_tensor<ElementAccumulator>(
          Shape<Int<Vec>, Int<FragsM>, Int<FragsN>>{});
        clear(tSr);
        
        // 2) 执行 Q*K GEMM
        collective_mma.mmaQK(tSr, gQ, gK_chunk(_, _, nblock, _), tSr, 
                           ceil_div(head_size_qk, QK_BLK_K), 
                           mainloop_params, false);
        
        // 3) Softmax 处理
        CollectiveSoftmaxEpilogue softmax(params.softmax);
        softmax(chunk_idx == 0 && nblock == 0, tSr, max_reg, sum_reg, out_reg);
        
        // 4) 执行 S*V GEMM
        collective_mma.template mmaPV<VSlicer>(
          out_reg, tSr, gV_chunk(_, _, nblock), out_reg, 
          mainloop_params, false);
        
        // ... prefetch 下一个 tile ...
        
        barrier_wait(barrier_scope);
      }
    }
    
    // Epilogue 处理
    auto epilogue_params = CollectiveEpilogue::template get_updated_copies<is_var_len>(
      params.epilogue, params.problem_shape, sequence_length_shape, batch_coord);
    CollectiveEpilogue epilogue{epilogue_params, shared_storage.epilogue};
    auto blk_coord_mnkl = make_coord(blk_m_coord, blk_n_coord, _, blk_l_coord);
    epilogue(params.problem_shape, sequence_length_shape, blk_coord_mnkl, 
             out_reg, max_reg, sum_reg);
  }
};

} // namespace cutlass::flash_attention::kernel