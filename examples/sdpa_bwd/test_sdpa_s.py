import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
import numpy as np
def is_close(refe: torch.Tensor,
             test: torch.Tensor):
    test = test.to(torch.float32)
    refe = refe.to(torch.float32)
    cosfactor = F.cosine_similarity(test.reshape(-1), refe.reshape(-1), dim=0) > 0.99
    allclose = torch.allclose(test, refe, atol=3e-3, rtol=3e-3)
    return cosfactor and allclose

def num_head_bcast(query: torch.Tensor,
                   key: torch.Tensor,
                   value: torch.Tensor):
    q_num_heads = query.size(-3)
    k_num_heads = key.size(-3)
    v_num_heads = value.size(-3)
    k_dim0 = key.size(0)
    k_dim1 = key.size(1)
    k_dim2 = key.size(2)
    k_dim3 = key.size(3)
    v_dim0 = value.size(0)
    v_dim1 = value.size(1)
    v_dim2 = value.size(2)
    v_dim3 = value.size(3)
    if (q_num_heads == k_num_heads) and (q_num_heads == v_num_heads):
        return key, value
    k_repeat = q_num_heads // k_num_heads
    v_repeat = q_num_heads // v_num_heads
    key = key.repeat_interleave(k_repeat, 1).reshape(k_dim0, k_repeat * k_dim1, k_dim2, k_dim3)
    value = value.repeat_interleave(v_repeat, 1).reshape(v_dim0, v_repeat * v_dim1, v_dim2, v_dim3)
    return key, value

def num_head_reduce(expand_grad: torch.Tensor,
                    x: torch.Tensor):
    num_heads_expand = expand_grad.size(-3)
    num_heads_orig = x.size(-3)
    if (num_heads_expand == num_heads_orig):
        return expand_grad
    n_repeat = num_heads_expand // num_heads_orig
    assert len(x.shape) == 4
    batch, num_head, seq_len, head_size = x.size()
    expand_grad = expand_grad.reshape(batch, num_head, n_repeat, seq_len, head_size)
    grad = torch.sum(expand_grad, dim=2).reshape(batch, num_head, seq_len, head_size)
    return grad

GRAD_DICT = {}

def dump_grad(name, value):
    global GRAD_DICT
    if name not in GRAD_DICT:
        GRAD_DICT[name] = value.clone()
    else:
        print(f'duplicated grad {name}')
    return

def softmax_backward(y: torch.Tensor,
                     grad_y: torch.Tensor,
                     scale: float):
    orig_dtype = y.dtype
    rest_dim = y.shape[:-1]
    dim = y.shape[-1]
    y = y.to(torch.float32)
    grad_y = grad_y.to(torch.float32)
    ydy = grad_y * y
    sum_row = torch.sum(ydy, dim= -1).reshape(*rest_dim, 1)
    grad_x2 = ydy - y * sum_row
    grad_x = grad_x2.reshape(*rest_dim, dim) * scale
    return grad_x.to(orig_dtype)

def softmax_backward_odo(p: torch.Tensor,
                         dp: torch.Tensor,
                         o: torch.Tensor,
                         do: torch.Tensor,
                         scale: float):
    orig_dtype = p.dtype
    o = o.to(torch.float32)
    do = do.to(torch.float32)
    p = p.to(torch.float32)
    dp = dp.to(torch.float32)
    odo = o * do
    sum_odo = torch.sum(odo, dim= -1, keepdim=True)
    ds = p * (dp - sum_odo) * scale
    ds = ds.to(orig_dtype)
    return ds, sum_odo

def dropout_backward(mask: torch.Tensor,
                     grad_y: torch.Tensor,
                     dropout_p: float):
    return mask * grad_y / (1 - dropout_p)

def dropout_backward2(grad_y: torch.Tensor,
                      dropout_p: float):
    return dropout_backward(mask, grad_y, dropout_p)

def dropout_forward(seed: int,
                    dropout_p: float,
                    x: torch.Tensor):
    torch.manual_seed(seed)
    mask = torch.empty_like(x).fill_(dropout_p)
    prob = torch.bernoulli(mask).logical_not()
    y = x * prob / (1 - dropout_p)
    return y

def softmax_causal_backward(y1: torch.Tensor,
                            y2: torch.Tensor,
                            grad_y: torch.Tensor):
    # y1 attn2 after dropout mask
    # y2 attn after softmax, only half mask
    orig_dtype = y1.dtype
    rest_dim = y1.shape[:-1]
    dim = y1.shape[-1]
    # seq_len_q = y.size()[-2]
    # seq_len_k = y.size()[-1]
    # seq_len_q = grad_y.size()[-2]
    # seq_len_k = grad_y.size()[-1]
    # mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool).tril(diagonal=0)
    y1 = y1.to(torch.float32)
    grad_y = grad_y.to(torch.float32)
    grad_y = grad_y
    ydy = grad_y * y1
    sum_row = torch.sum(ydy, dim= -1).reshape(*rest_dim, 1)
    grad_x2 = ydy - y2 * sum_row
    grad_x = grad_x2.reshape(*rest_dim, dim)
    return grad_x.to(orig_dtype)

class SDPA(nn.Module):
    def __init__(self, dropout_p) -> None:
        super().__init__()
        if dropout_p > 0.0:
            self.do_m = nn.Dropout(p=dropout_p)

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale: float = None):
        dtype = q.dtype
        self.head_size_q = q.size()[-1]
        self.q = q.clone()
        self.k = k.clone()
        self.v = v.clone()
        seq_len_q, seq_len_k = q.size(-2), k.size(-2)

        attn_bias = torch.zeros(seq_len_q, seq_len_k, dtype=dtype)
        self.is_causal = is_causal
        if is_causal:
            diagonal_offset = seq_len_k - seq_len_q
            temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool).tril(diagonal=diagonal_offset)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(dtype)

        k_expand, v_expand = num_head_bcast(q, k, v)
        #k_expand = k
        #v_expand = v
        # k_expand.register_hook(lambda x: dump_grad('k_expand_grad', k_expand))
        # v_expand.register_hook(lambda x: dump_grad('v_expand_grad', v_expand))
        #self.p = q@k_expand.transpose(-2, -1)
        s = torch.matmul(q, k_expand.transpose(-1, -2))
        s.register_hook(lambda x: dump_grad('s_grad', x))
        self.s = s
        s = s.to(torch.float32)
        self.softmax_scale = 1 / np.sqrt(self.head_size_q) if scale is None else scale
        s = s * self.softmax_scale + attn_bias
        if is_causal:
            diagonal_offset = seq_len_q - seq_len_k
            if (diagonal_offset > 0):
                for i in range(diagonal_offset):
                    s[:, :, i, :] = 0.0

        sum_row, _ = torch.max(s, dim= -1, keepdim=True)
        s = s - sum_row
        self.lse = torch.logsumexp(s, dim=-1, keepdim=True) + sum_row
        p = torch.softmax(s, dim= -1).to(dtype)
        self.p = p
        p.register_hook(lambda x: dump_grad('p_grad', x))
        attn = torch.matmul(p, v_expand)
        attn.register_hook(lambda x: dump_grad('O_grad', x))
        self.o = attn
        return attn

    def backward_ref(self,
                     o_grad: torch.Tensor):
        q_grad = torch.empty_like(self.q)
        k_grad = torch.empty_like(self.k)
        v_grad = torch.empty_like(self.v)
        k_expand, v_expand = num_head_bcast(self.q, self.k, self.v)
        # forward
        s = torch.matmul(self.q, k_expand.transpose(-1, -2))
        s = s.to(torch.float32)
        dtype = self.q.dtype
        seq_len_q, seq_len_k = q_grad.size(-2), k_grad.size(-2)
        attn_bias = torch.zeros(seq_len_q, seq_len_k, dtype=dtype)
        if self.is_causal:
            diagonal_offset = seq_len_k - seq_len_q
            temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool).tril(diagonal=diagonal_offset)
            attn_bias.masked_fill_(temp_mask.logical_not(), float("-inf"))
            attn_bias.to(dtype)
        s = s * self.softmax_scale + attn_bias
        p = torch.exp(s - self.lse).to(dtype)
        # backward
        v_grad = torch.matmul(p.transpose(-1, -2), o_grad)

        p_grad = torch.matmul(o_grad, v_expand.transpose(-1, -2))
        s_grad, odo = softmax_backward_odo(p, p_grad, self.o, o_grad, self.softmax_scale)
        self.odo = odo
        # s_grad = softmax_backward(p, p_grad, self.softmax_scale)
        k_grad = torch.matmul(s_grad.transpose(-1, -2), self.q)
        q_grad = torch.matmul(s_grad, k_expand)
        k_grad = num_head_reduce(k_grad, self.k)
        v_grad = num_head_reduce(v_grad, self.v)
        return (q_grad, k_grad, v_grad, p_grad, s_grad)

class ptSDPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self,
                q: torch.Tensor,
                k: torch.Tensor,
                v: torch.Tensor,
                dropout_p: float = 0.0,
                is_causal: bool = False,
                scale: float = None):
        dtype = q.dtype
        if is_causal:
            seq_len_q, seq_len_k = q.size(-2), k.size(-2)
            diagonal_offset = seq_len_k - seq_len_q
            temp_mask = torch.ones(seq_len_q, seq_len_k, dtype=torch.bool).tril(diagonal=diagonal_offset)
        with sdpa_kernel(backends=[SDPBackend.MATH]):
            return F.scaled_dot_product_attention(q, k, v,
                                                  dropout_p=dropout_p,
                                                  scale=scale,
                                                  attn_mask=temp_mask,
                                                  enable_gqa=True)

def set_dict(dump_dict, name, value):
    if value.dtype == torch.bfloat16 or value.dtype == torch.float16:
        dump_dict[name] = value.detach().clone().view(torch.uint16).numpy()
    elif value.dtype == torch.bool:
        dump_dict[name] = value.detach().clone().to(torch.uint16).numpy()
    else:
        dump_dict[name] = value.detach().clone().numpy()

def test_sdpa(dtype,
              seed: int,
              batch: int,
              num_heads_q: int,
              num_heads_kv: int,
              seq_len_qo: int,
              seq_len_kv: int,
              head_size_qk: int,
              head_size_vo: int,
              dropout_p: float = 0.0,
              is_causal: bool = False,
              is_bhsd: bool = True):
    torch.manual_seed(seed)
    q = torch.randn(batch, num_heads_q, seq_len_qo, head_size_qk, requires_grad=True).to(dtype)
    k = torch.randn(batch, num_heads_kv, seq_len_kv, head_size_qk, requires_grad=True).to(dtype)
    v = torch.randn(batch, num_heads_kv, seq_len_kv, head_size_vo, requires_grad=True).to(dtype)
    q2 = q.clone()
    k2 = k.clone()
    v2 = v.clone()
    q.retain_grad()
    k.retain_grad()
    v.retain_grad()
    q2.retain_grad()
    k2.retain_grad()
    v2.retain_grad()
    test_model = SDPA(dropout_p).to(dtype)
    refe_model = ptSDPA().to(dtype)

    torch.manual_seed(seed)
    attn_out = test_model(q, k, v, dropout_p, is_causal)
    torch.manual_seed(seed)
    attn_out_pt = refe_model(q2, k2, v2, dropout_p, is_causal)
    grad = torch.empty_like(attn_out)
    torch.manual_seed(seed)
    grad.uniform_(-1, 1)
    grad = grad.to(dtype)
    attn_out.backward(grad)
    attn_out_pt.backward(grad)
    q_grad, k_grad, v_grad, p_grad, s_grad = test_model.backward_ref(grad)
    dump_dict = {}
    print(f"seed {seed} bsz {batch} nh_q {num_heads_q} nh_kv {num_heads_kv} sl_qo {seq_len_qo} sl_kv {seq_len_kv} hs_qk {head_size_qk} hs_vo {head_size_vo} dp {dropout_p} is_causal {is_causal} is_bhsd {is_bhsd}")
    print('attn_out ', is_close(attn_out, attn_out_pt))
    print('p_grad ', is_close(GRAD_DICT['p_grad'], p_grad))
    # print('s2_grad ', is_close(GRAD_DICT['s2_grad'], s2_grad))
    print('s_grad ', is_close(GRAD_DICT['s_grad'], s_grad))
    print('k_grad ', is_close(k_grad, k2.grad))
    print('q_grad ', is_close(q_grad, q2.grad))
    print('v_grad ', is_close(v_grad, v2.grad))
    if is_bhsd:
        set_dict(dump_dict, 'out', attn_out)
        set_dict(dump_dict, 'grad', grad)
        set_dict(dump_dict, 'v_grad', v_grad)
        set_dict(dump_dict, 'k_grad', k_grad)
        set_dict(dump_dict, 'q_grad', q_grad)
        set_dict(dump_dict, 'q', q)
        set_dict(dump_dict, 'k', k)
        set_dict(dump_dict, 'v', v)
    else:
        set_dict(dump_dict, 'out', attn_out.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'grad', grad.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'v_grad', v_grad.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'k_grad', k_grad.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'q_grad', q_grad.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'q', q.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'k', k.transpose(1, 2).contiguous())
        set_dict(dump_dict, 'v', v.transpose(1, 2).contiguous())
    set_dict(dump_dict, 'lse', test_model.lse)
    set_dict(dump_dict, 'odo', test_model.odo)
    set_dict(dump_dict, 's', test_model.s)
    set_dict(dump_dict, 'p', test_model.p)
    set_dict(dump_dict, 'p_grad', p_grad)
    set_dict(dump_dict, 's_grad', s_grad)
    shape = np.array([batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo, is_causal, is_bhsd], dtype=np.int32)
    dump_dict['shape'] = shape
    # print('test', v_grad[0,0:4,0,0:16])
    # print('upstream', v2.grad[0,0:4,0,0:16])
    np.savez(f'mha-{batch}-{num_heads_q}-{num_heads_kv}-{seq_len_qo}-{seq_len_kv}-{head_size_qk}-{head_size_vo}-{dropout_p}-{int(is_causal)}-{int(is_bhsd)}.npz', **dump_dict)

def loop_run():
    global GRAD_DICT
    for h in [4]:
        # for seq_q in list(range(512, 512+32)):
        #     for seq_k in list(range(512, 512+32)):
        for seq_q in [512, 513, 523, 527, 535, 543]:
            for seq_k in [512, 513, 523, 527, 535, 543]:
                for dim in [64, 96, 128, 192]:
                    # print('test_run', 4, 4, h, seq_q, seq_k, dim, dim)
                    # bhsd
                    test_sdpa(torch.float16, 123, 4, 4, h, seq_q, seq_k, dim, dim, is_causal=True, is_bhsd = True)
                    GRAD_DICT = {}
                    # bshd
                    test_sdpa(torch.float16, 123, 4, 4, h, seq_q, seq_k, dim, dim, is_causal=True, is_bhsd = False)
                    GRAD_DICT = {}

if __name__ == '__main__':
    # test_sdpa(torch.float16, 123, 4, 4, 4, 513, 512, 128, 128, is_causal=True)
    loop_run()
    # test_sdpa(torch.float16, 123, 4, 4, 4, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 4, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 2, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 2, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 1, 513, 784, 128, 128, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.float16, 123, 4, 4, 1, 513, 784, 128, 128, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, is_causal=True)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, 0.3, is_causal=False)
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 2, 513, 513, 128, 64, is_causal=False)
    # GRAD_DICT = {}
    # GRAD_DICT = {}
    # test_sdpa(torch.bfloat16, 123, 4, 4, 4, 513, 513, 128, 64, is_causal=False)
    # test_sdpa(torch.bfloat16, 123, 4, 4, 1, 513, 513, 128, 64, False)
    # test_sdpa(torch.bfloat16, 123, 4, 4, 4, 1024, 513, 128, 128)
    # test_sdpa(123, 2, 16, 1, 513, 513, 128, 128)
