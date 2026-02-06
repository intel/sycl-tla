#pragma once
#include <sycl/sycl.hpp>
#include <cassert>
#include <cute/util/compat.hpp>
#include <cute/tensor.hpp>

#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/print_error.hpp"
#include "cutlass/util/sycl_event_manager.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include "params.hpp"


#include "cutlass/util/command_line.h"

void read_args(int argc, char**argv, int n, int64_t *p) {
    if (argc >= n + 1)
        sscanf(argv[n], "%ld", p);
}

void
debug_info() {
    print("block idx (%d,%d,%d) dim (%d,%d,%d) thread idx (%d,%d,%d) dim (%d,%d,%d)\n",
          BlockIdxX(), BlockIdxY(), BlockIdxZ(),
          GridDimX(), GridDimY(), GridDimZ(),
          ThreadIdxX(), ThreadIdxY(), ThreadIdxZ(),
          BlockDimX(), BlockDimY(), BlockDimZ());
}

template<class Engine, class Layout>
void print_t(Tensor<Engine, Layout> &r) {
    print(r);
    for (int i = 0; i < size(r); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)r(i));
    }
    print("\n");
}

template<class T1, class T2>
void print_t(T1 m, T2 g) {
    print(m);
    for (int i = 0; i < size(g); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10.7f ", (float)m(g(i)));
    }
    print("\n");
}

template<class T>
void print_t_2d(T t) {
    static_assert(rank(t) == 2, "Only support 2D Tensor");
    print(t);
    for (int i = 0; i <  size < 0>(t); ++i) {
        print("\n(%03d): ", i);
        for (int j = 0; j < size<1>(t); ++j) {
            print("%10.7f ", (float)t(i,j));
        }
    }
    print("\n");
}

template<class T>
void print_d(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print("%10u ", t(i));
    }
    print("\n");
}

template<class T>
void print_c(T t) {
    print(t);
    for (int i = 0; i < size(t); ++i) {
        if (i % 8 == 0)
            print("\n(%03d): ", i / 8);
        print(t(i));
    }
    print("\n");
}

using ProblemShapeRegular = cute::tuple<int, int, int, int, int, int, int>; // batch, num_head_q,num_head_kv,seq_len_qo,seq_len_kv,head_size_qk,head_size_vo

struct Options {

    bool help;
    bool error;
    bool is_causal;
    bool is_bhsd;
    bool is_bf16;

    int batch, num_heads_q, num_heads_kv, seq_len_qo, seq_len_kv, head_size_qk, head_size_vo, iterations;
    float softmax_scale;

    Options()
        : help(false), error(false), is_causal(false), is_bhsd(true), is_bf16(false),
          batch(32), num_heads_q(16), num_heads_kv(16), seq_len_qo(1), head_size_qk(128),
          seq_len_kv(512), head_size_vo(128), iterations(100), softmax_scale(1.f) {}

    // Parses the command line
    void parse(int argc, char const **args) {
        cutlass::CommandLine cmd(argc, args);

        if (cmd.check_cmd_line_flag("help")) {
            help = true;
            return;
        }

        if (cmd.check_cmd_line_flag("is_causal")) {
            is_causal = true;
        } else {
            is_causal = false;
        }

        if (cmd.check_cmd_line_flag("is_bhsd")) {
            is_bhsd = true;
        } else {
            is_bhsd = false;
        }

        if (cmd.check_cmd_line_flag("is_bf16")) {
            is_bf16 = true;
        } else {
            is_bf16 = false;
        }

        cmd.get_cmd_line_argument("batch", batch, 32);
        cmd.get_cmd_line_argument("num_heads_q", num_heads_q, 16);
        cmd.get_cmd_line_argument("num_heads_kv", num_heads_kv, num_heads_q);
        cmd.get_cmd_line_argument("seq_len_qo", seq_len_qo, 1);
        cmd.get_cmd_line_argument("seq_len_kv", seq_len_kv, 512);
        cmd.get_cmd_line_argument("head_size_vo", head_size_vo, head_size_qk);
        cmd.get_cmd_line_argument("head_size_qk", head_size_qk, head_size_vo);
        cmd.get_cmd_line_argument("iterations", iterations, 100);

        softmax_scale = 1 / sqrt(static_cast<float>(head_size_qk));
    }

    /// Prints the usage statement.
    std::ostream &print_usage(std::ostream &out) const {

        out << "BMG Flash Attention v2 Example\n\n"
            << "Options:\n\n"
            << "  --help                      If specified, displays this usage statement\n\n"
            << "  --is_causal                 Apply Causal Mask to the output of first Matmul\n"
            << "  --is_bhsd                   Use Batch, Head, Seq, Dim layout for Q/K/V/O\n"
            << "  --is_bf16                   Use bf16 for input and output data type"
            << "  --batch=<int>               Sets the Batch Size of the Multi-Head Self Attention module\n"
            << "  --num_heads_q=<int>         Sets the Number of Attention Heads for Key-Value pair the Multi-Head Self Attention module\n"
            << "  --num_heads_kv=<int>        Sets the Number of Attention Heads for Query input in the Multi-Head Self Attention module\n"
            << "  --seq_len_qo=<int>          Sets the Sequence length of the Query input in Multi-Head Self Attention module\n"
            << "  --seq_len_kv=<int>          Sets the Sequence length of the Key-Value pair in Multi-Head Self Attention module\n"
            << "  --head_size_qk=<int>        Sets the Attention Head dimension of the 1st Matrix Multiplication in Multi-Head Self Attention module\n"
            << "  --head_size_vo=<int>        Sets the Attention Head dimension of the 2nd Matrix Multiplication in Multi-Head Self Attention module\n"
            << "  --iterations=<int>          Iterations\n\n";

        return out;
    }
};

#include "sdpa_backward.hpp"
#include "sdpa_preprocessing.hpp"
