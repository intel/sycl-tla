#include "src/prefill.hpp"
#include <torch/all.h>

namespace FLASH_NAMESPACE {

std::vector<at::Tensor>
mha_fwd(
        at::Tensor &q,         // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        const at::Tensor &k,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        const at::Tensor &v,         // batch_size x seqlen_k x num_heads_k x round_multiple(head_size, 8)
        std::optional<at::Tensor> &out_,             // batch_size x seqlen_q x num_heads x round_multiple(head_size, 8)
        std::optional<at::Tensor> &alibi_slopes_, // num_heads or batch_size x num_heads
        const float p_dropout,
        const float softmax_scale,
        bool is_causal,
        int window_size_left,
        int window_size_right,
        const float softcap,
        const bool return_softmax,
        std::optional<at::Generator> gen_) {
  
    at::Tensor out;
    if (out_.has_value()) {
      out = *out_;
    } else {
      out = torch::zeros_like(q);
    }

    cutlass_prefill_fixed_impl(q, k, v, out, softmax_scale, is_causal);

    // TODO: current do not support store softmax_lse out
    // hard code to return empty tensor for softmax_lse, S_dmask, rng_state
    auto softmax_lse = torch::empty_like(out);
    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }

std::vector<at::Tensor>
mha_varlen_fwd(
              at::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
              const at::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              const at::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
              std::optional<at::Tensor> &out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
              const at::Tensor &cu_seqlens_q,  // b+1
              const at::Tensor &cu_seqlens_k,  // b+1
              std::optional<at::Tensor> &seqused_k, // b. If given, only this many elements of each batch element's keys are used.
              std::optional<const at::Tensor> &leftpad_k_, // batch_size
              std::optional<at::Tensor> &block_table_, // batch_size x max_num_blocks_per_seq
              std::optional<at::Tensor> &alibi_slopes_, // num_heads or b x num_heads
              int max_seqlen_q,
              const int max_seqlen_k,
              const float p_dropout,
              const float softmax_scale,
              const bool zero_tensors,
              bool is_causal,
              int window_size_left,
              int window_size_right,
              const float softcap,
              const bool return_softmax,
              std::optional<at::Generator> gen_) {
    at::Tensor out;
    if (out_.has_value()) {
      out = *out_;
    } else {
      out = torch::zeros_like(q);
    }

    cutlass_prefill_varlen_impl(q, k, v, out,
                              cu_seqlens_q, cu_seqlens_k,
                              max_seqlen_q, max_seqlen_k,
                              softmax_scale, is_causal);

    // TODO: current do not support store softmax_lse out
    // hard code to return empty tensor for softmax_lse, S_dmask, rng_state
    auto softmax_lse = torch::empty_like(out);
    at::Tensor S_dmask;
    at::Tensor rng_state;
    return {out, softmax_lse, S_dmask, rng_state};
  }
}  // namespace FLASH_NAMESPACE

// std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
std::vector<torch::Tensor>
mha_fwd(
    torch::Tensor &q, 
    const torch::Tensor &k, 
    const torch::Tensor &v,
    c10::optional<torch::Tensor> out_,
    c10::optional<torch::Tensor> alibi_slopes_,
    const double p_dropout, 
    const double softmax_scale, 
    bool is_causal,
    const int64_t window_size_left, 
    const int64_t window_size_right,
    const double softcap, 
    const bool return_softmax,
    c10::optional<at::Generator> gen_) {
    return FLASH_NAMESPACE::mha_fwd(
      q, 
      k, 
      v, 
      out_, 
      alibi_slopes_, 
      static_cast<float>(p_dropout),
      static_cast<float>(softmax_scale), 
      is_causal,
      static_cast<int>(window_size_left), 
      static_cast<int>(window_size_right),
      static_cast<float>(softcap), 
      return_softmax,
      gen_
    );
}

std::vector<torch::Tensor>
mha_varlen_fwd(
    torch::Tensor &q,  // total_q x num_heads x head_size, total_q := \sum_{i=0}^{b} s_i
    const torch::Tensor &k,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    const torch::Tensor &v,  // total_k x num_heads_k x head_size, total_k := \sum_{i=0}^{b} s_i or num_blocks x page_block_size x num_heads_k x head_size if there's a block_table.
    std::optional<torch::Tensor> out_, // total_q x num_heads x head_size, total_k := \sum_{i=0}^{b} s_i
    const torch::Tensor &cu_seqlens_q,  // b+1
    const torch::Tensor &cu_seqlens_k,  // b+1
    std::optional<torch::Tensor> seqused_k, // b. If given, only this many elements of each batch element's keys are used.
    std::optional<torch::Tensor> leftpad_k_, // batch_size
    std::optional<torch::Tensor> block_table_, // batch_size x max_num_blocks_per_seq
    std::optional<torch::Tensor> alibi_slopes_, // num_heads or b x num_heads
    int64_t max_seqlen_q,
    const int64_t max_seqlen_k,
    const double p_dropout,
    const double softmax_scale,
    const bool zero_tensors,
    bool is_causal,
    int64_t window_size_left,
    int64_t window_size_right,
    const double softcap,
    const bool return_softmax,
    std::optional<at::Generator> gen_) {    
    return FLASH_NAMESPACE::mha_varlen_fwd(
        const_cast<at::Tensor &>(q), 
        k, 
        v, 
        out_, 
        cu_seqlens_q, 
        cu_seqlens_k,
        seqused_k,
        reinterpret_cast<std::optional<const at::Tensor>&>(leftpad_k_),
        block_table_,
        alibi_slopes_,
        static_cast<int>(max_seqlen_q),
        static_cast<int>(max_seqlen_k),
        static_cast<float>(p_dropout),
        static_cast<float>(softmax_scale),
        zero_tensors,
        is_causal,
        static_cast<int>(window_size_left), 
        static_cast<int>(window_size_right),
        static_cast<float>(softcap),
        return_softmax,
        gen_
    );
}