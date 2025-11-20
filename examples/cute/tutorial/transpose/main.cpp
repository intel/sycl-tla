#include "block_2d_transposed_copy.h"
#include "copy_direct.h"
#include "copy_smem.h"
#include "transpose_naive.h"
#include "transpose_smem.h"
#include "util.h"

int main(int argc, char const **argv) {

  using Element = float;

  int size = 16384;
  int M = size, N = size, iterations = 10;

  std::cout << "Matrix size: " << M << " x " << N << std::endl;

  printf("Baseline copy.\n");
  benchmark<Element, false>(copy_direct<Element>, M, N, iterations);

  printf("\nNaive transpose (no smem):\n");
  benchmark<Element>(transpose_naive<Element>, M, N, iterations);

  printf("\nCopy through SMEM.\n");
  benchmark<Element, false>(copy_smem<Element>, M, N, iterations);

  printf("\nTranspose through SMEM.:\n");
  benchmark<Element>(transpose_smem<Element>, M, N, iterations);

  printf("Block 2d Transposed load\n");
  benchmark<Element, true, false, false>(block_2d_transposed_copy<Element>, M,
                                          N, iterations);

  return 0;
}
