/*
 * Objective: Transpose a square matrix tile of size 32 on a side
 *
 * */

/*
 * Work Group Configuration: Each work group handles one tile.

   Dimensions: (TILE_DIM x BLOCK_ROWS) = (32 x 8) work-items.
   This means each work group contains 32 * 8 = 256 work-items.
   For a 32x32 tile (1024 elements), each work-item processes 1024 / 256 = 4
 elements.

   Example Work Group for Tile (0,0):
   Thread indices within the work group (local_id):
   localID[0]
      ^
      |
   7  | t(0,7) t(1,7) t(2,7) ... t(31,7)
   6  | t(0,6) t(1,6) t(2,6) ... t(31,6)
   5  | t(0,5) t(1,5) t(2,5) ... t(31,5)
   4  | t(0,4) t(1,4) t(2,4) ... t(31,4)
   3  | t(0,3) t(1,3) t(2,3) ... t(31,3)
   2  | t(0,2) t(1,2) t(2,2) ... t(31,2)
   1  | t(0,1) t(1,1) t(2,1) ... t(31,1)
   0  | t(0,0) t(1,0) t(2,0) ... t(31,0)  --> localId[1]: 0  1  2  ... 31
      +------------------------------------->
 */

#include <sycl/sycl.hpp>

#include <iostream>
#include <stdexcept>
#include <vector>

#include "benchmark.h"

// size of the entire square matrix NrN
// we still use separate variables for the sides so we can
// think about tile and block indexing in the matrix rows/cols
constexpr size_t N = 16384;
constexpr size_t Nr = N;
constexpr size_t Nc = N;

// size of a single data tile that we will work with
// we use 16 here to demonstrate bank conflicts on intel gpus
constexpr size_t TILE_DIM = 64;

// number of rows in our workgroup
// intentionally this is a smaller number because we want to use
// a single thread to copy 4 elements
constexpr size_t BLOCK_ROWS = TILE_DIM / 4;

constexpr size_t numIters = 100;

typedef unsigned int uint;
using T = float;

template <typename AccT> auto get_accessor_pointer(const AccT &acc) {
  return acc.template get_multi_ptr<sycl::access::decorated::no>().get();
}

int main() {
  std::vector<T> A(Nr * Nc);
  std::vector<T> A_T(Nr * Nc);
  std::vector<T> A_T_ref(Nr * Nc);

  if (Nr % TILE_DIM or Nc % TILE_DIM) {
    throw std::runtime_error("Nr and Nc must be a multiple of TILE_DIM");
  }

  if (TILE_DIM % BLOCK_ROWS) {
    throw std::runtime_error("TILE_DIM must be a multiple of BLOCK_ROWS");
  }

  // fill the matrix and prep ref output on the host
  for (int i = 0; i < Nr; i++)
    for (int j = 0; j < Nc; j++)
      A[i * Nr + j] = i * Nr + j; // data same as the linear physical index

  // for the ref transpose out, flip the quickest varying index on the reads
  for (int i = 0; i < Nr; i++)
    for (int j = 0; j < Nc; j++)
      A_T_ref[i * Nr + j] = j * Nr + i;

  try {
    auto q = sycl::queue{sycl::property::queue::enable_profiling{}};

    std::cout << "Running on "
              << q.get_device().get_info<sycl::info::device::name>() << "\n";
    std::cout << "Local Memory Size: "
              << q.get_device().get_info<sycl::info::device::local_mem_size>() /
                     1024
              << "KB" << std::endl;
    std::cout
        << "Max Work Group Size: "
        << q.get_device().get_info<sycl::info::device::max_work_group_size>()
        << std::endl;

    sycl::range dataRange{Nr, Nc};
    // div y dim by 4 as we use a single work-item to move 4 values
    sycl::range globalRange{Nr / 4, Nc};
    sycl::range localRange{BLOCK_ROWS, TILE_DIM};
    sycl::nd_range ndRange{globalRange, localRange};

    {
      sycl::buffer h_idata{A.data(), dataRange};
      sycl::buffer h_odata{A_T.data(), dataRange};

      // Simple copy without coalescing to demonstrate its inefficiency
      auto simple_copy = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};

          cgh.parallel_for<class copy_gmem_direct_no_coalescing>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto row_id = item.get_group(0) * TILE_DIM + localId[0];
                auto col_id = globalId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{col_id, row_id + i};
                  d_odata[dataIdx] = d_idata[dataIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // Simple copy with coalescing used as reference for best effective
      // bandwidth
      auto simple_coalesced_copy = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};

          cgh.parallel_for<class copy_gmem_direct>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                // get_group(0) gives the work-group id along the row dim
                // since we need to compute the group offset here with
                // TILE_DIM;  Just using the global id wouldn't work here
                // because we don't have a 1:1 thread:value
                // mapping here)
                auto row_id = item.get_group(0) * TILE_DIM + localId[0];
                // work-items of the fastest varying dimension (1) access
                // consecutive memory locations so that loads and stores
                // are coalesced
                auto col_id = globalId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  d_odata[dataIdx] = d_idata[dataIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // Naive Transpose
      // reads are coalesced, but writes are not
      auto naive_transpose = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};

          cgh.parallel_for<class naive_transpose>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto row_id = item.get_group(0) * TILE_DIM + localId[0];
                auto col_id = globalId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id idataIdx{row_id + i, col_id};
                  // swap the output buffer's indices to transpose it
                  sycl::id odataIdx{col_id, row_id + i};
                  d_odata[odataIdx] = d_idata[idataIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // Tiled copy through SMEM as a baseline for SMEM transpose
      auto smem_copy = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};
          sycl::range tileRange{TILE_DIM, TILE_DIM};
          sycl::local_accessor<T, 2> sharedMemTile{tileRange, cgh};

          cgh.parallel_for<class tiled_copy>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto groupOffset_0 = item.get_group(0) * TILE_DIM;
                auto groupOffset_1 = item.get_group(1) * TILE_DIM;

                auto row_id = groupOffset_0 + localId[0];
                auto col_id = groupOffset_1 + localId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced read from gmem into smem
                  sharedMemTile[smemTileIdx] = d_idata[dataIdx];
                }

                // We need to wait here to ensure that all work items
                // have written to local memory before we start reading
                // from it.
                sycl::group_barrier(item.get_group());

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced write to gmem from smem
                  d_odata[dataIdx] = sharedMemTile[smemTileIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // Coalesce reads and writes to global memory but do the strided
      // access required for the transpose in shared local memory as it
      // doesn't levy as much a much penalty when done in SLM compared to
      // GMEM
      auto smem_transpose = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};
          sycl::range tileRange{TILE_DIM, TILE_DIM};
          sycl::local_accessor<T, 2> sharedMemTile{tileRange, cgh};

          cgh.parallel_for<class tiled_transpose>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto groupOffset_0 = item.get_group(0) * TILE_DIM;
                auto groupOffset_1 = item.get_group(1) * TILE_DIM;

                auto row_id = groupOffset_0 + localId[0];
                auto col_id = groupOffset_1 + localId[1];

                auto row_id_T = groupOffset_1 + localId[0];
                auto col_id_T = groupOffset_0 + localId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced read from gmem into smem
                  sharedMemTile[smemTileIdx] = d_idata[dataIdx];
                }

                // We need to wait here to ensure that all work items
                // have written to local memory before we start reading
                // from it.
                sycl::group_barrier(item.get_group());

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id_T + i, col_id_T};
                  // this creates strided reads in smem, but the writes
                  // to gmem are still coalesced
                  sycl::id smemTileIdx{localId[1], localId[0] + i};
                  d_odata[dataIdx] = sharedMemTile[smemTileIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // SMEM Transpose avoiding bank conflict by allocating TILE_DIM + 1 on
      // SMEM column dimension, causing every element in the smem to fall in
      // a different shared memory bank; kernel is the same as above
      auto smem_transpose_no_bank_conflict = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};
          sycl::range tileRange{TILE_DIM, TILE_DIM + 1};
          sycl::local_accessor<T, 2> sharedMemTile{tileRange, cgh};

          cgh.parallel_for<class tiled_transpose_padded_smem>(
              ndRange, [=](sycl::nd_item<2> item) {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto groupOffset_0 = item.get_group(0) * TILE_DIM;
                auto groupOffset_1 = item.get_group(1) * TILE_DIM;

                auto row_id = groupOffset_0 + localId[0];
                auto col_id = groupOffset_1 + localId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced read from gmem into smem
                  sharedMemTile[smemTileIdx] = d_idata[dataIdx];
                }

                // We need to wait here to ensure that all work items
                // have
                // written to local memory before we start reading from
                // it
                sycl::group_barrier(item.get_group());

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  // output tile offsets need to be swapped
                  sycl::id dataIdx{groupOffset_1 + localId[0] + i,
                                   groupOffset_0 + localId[1]};
                  // this creates strided reads in smem, but the writes
                  // to
                  // gmem are still coalesced
                  sycl::id smemTileIdx{localId[1], localId[0] + i};
                  d_odata[dataIdx] = sharedMemTile[smemTileIdx];
                }
              });
        });
        q.wait_and_throw();
      };

      // // transpose using subgroup shuffle functions
      // util::benchmark(
      //     [&]() {
      //       q.submit([&](sycl::handler &cgh) {
      //         sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
      //         sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
      //                                sycl::property::no_init{}};
      //
      //         // this kernel requires the tile size to be equal to the
      //         // sub-group size used so we can use the sub-group shuffle
      //         // functions
      //         constexpr size_t BLOCK_SIZE = 16;
      //         cgh.parallel_for<class tiled_transpose_subgroup_shuffle>(
      //             sycl::nd_range<2>(sycl::range<2>(Nr / BLOCK_SIZE, Nc),
      //                               sycl::range<2>(1, BLOCK_SIZE)),
      //             [=](sycl::nd_item<2> it)
      //             [[sycl::reqd_sub_group_size(16)]]
      //             {
      //               auto localId = it.get_local_id();
      //               int gi = it.get_group(0);
      //               int gj = it.get_group(1);
      //
      //               auto sg = it.get_sub_group();
      //               uint sgId = sg.get_local_id()[0];
      //
      //               float bcol[BLOCK_SIZE];
      //               int ai = BLOCK_SIZE * gi;
      //               int aj = BLOCK_SIZE * gj;
      //
      //               for (uint k = 0; k < BLOCK_SIZE; k++) {
      //                 // load columns of the matrix tile into the
      //                 subgroup bcol[k] =
      //                 sg.load(get_accessor_pointer(d_idata) +
      //                                   (ai + k) * Nc + aj);
      //               }
      //
      //               // no barriers required here because the threads of a
      //               // sub-group execute concurrently, so all columns of
      //               the
      //               // matrix were loaded into bcol already
      //
      //               float tcol[BLOCK_SIZE];
      //               for (uint n = 0; n < BLOCK_SIZE; n++) {
      //                 if (sgId == n) {
      //                   for (uint k = 0; k < BLOCK_SIZE; k++) {
      //                     // returns the value of bcol[n] from the k-th
      //                     // work-item
      //                     tcol[k] = sycl::select_from_group(sg, bcol[n],
      //                     k);
      //                   }
      //                 }
      //               }
      //
      //               for (uint k = 0; k < BLOCK_SIZE; k++) {
      //                 sg.store(get_accessor_pointer(d_odata) + (aj + k) *
      //                 Nc
      //                 +
      //                              ai,
      //                          tcol[k]);
      //               }
      //             });
      //       });
      //       q.wait_and_throw();
      //     },
      //     numIters, Nc * Nr,
      //     "Tiled GMEM Transpose with sub-group shuffle functions");

      // Tiled Transpose using the sub-group shuffle function
      // where loads and stores are to shared local memory
      auto tiled_subgroup_shuffle = [&]() {
        q.submit([&](sycl::handler &cgh) {
          sycl::accessor d_idata{h_idata, cgh, sycl::read_only};
          sycl::accessor d_odata{h_odata, cgh, sycl::write_only,
                                 sycl::property::no_init{}};
          // sub-group size == data block size in smem to be transposed
          constexpr size_t BLOCK_SIZE = 16;
          sycl::range tileRange{TILE_DIM, TILE_DIM};
          sycl::local_accessor<T, 2> sMemTile{tileRange, cgh};
          sycl::local_accessor<T, 2> sMemTileTransposed{tileRange, cgh};

          cgh.parallel_for<class tiled_transpose_smem_subgroup_shuffle>(
              ndRange,
              [=](sycl::nd_item<2> item) [[sycl::reqd_sub_group_size(16)]] {
                auto globalId = item.get_global_id();
                auto localId = item.get_local_id();

                auto groupOffset_0 = item.get_group(0) * TILE_DIM;
                auto groupOffset_1 = item.get_group(1) * TILE_DIM;

                auto row_id = groupOffset_0 + localId[0];
                auto col_id = groupOffset_1 + localId[1];

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{row_id + i, col_id};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced read from gmem into smem
                  sMemTile[smemTileIdx] = d_idata[dataIdx];
                }

                // Now sMem holds a 32x32 size data tile that we
                // transpose with sub-group select function
                // Each 1x16 sub-group of work-item can transpose one
                // 16x16 tile in the SMEM. So we need to loop over the
                // process two times to transpose an entire 32x32 tile
                {
                  auto sg = item.get_sub_group();
                  uint sgId = sg.get_local_id()[0];

                  float bcol[BLOCK_SIZE];
                  int ai = BLOCK_SIZE * 0;
                  int aj = BLOCK_SIZE * sg.get_group_id();

                  for (uint k = 0; k < BLOCK_SIZE; k++) {
                    // load columns of the matrix
                    // tile into the subgroup
                    bcol[k] = sg.load(get_accessor_pointer(sMemTile) +
                                      (ai + k) * TILE_DIM + aj);
                  }

                  float tcol[BLOCK_SIZE];
                  for (uint n = 0; n < BLOCK_SIZE; n++) {
                    if (sgId == n) {
                      for (uint k = 0; k < BLOCK_SIZE; k++) {
                        // returns the value of bcol[n] from the k-th
                        // work-item
                        tcol[k] = sycl::select_from_group(sg, bcol[n], k);
                      }
                    }
                  }

                  for (uint k = 0; k < BLOCK_SIZE; k++) {
                    sg.store(get_accessor_pointer(sMemTileTransposed) +
                                 (aj + k) * TILE_DIM + ai,
                             tcol[k]);
                  }
                }

                {
                  auto sg = item.get_sub_group();
                  uint sgId = sg.get_local_id()[0];

                  float bcol[BLOCK_SIZE];
                  int ai = BLOCK_SIZE * 1;
                  int aj = BLOCK_SIZE * sg.get_group_id();

                  for (uint k = 0; k < BLOCK_SIZE; k++) {
                    // load columns of the matrix
                    // tile into the subgroup
                    bcol[k] = sg.load(get_accessor_pointer(sMemTile) +
                                      (ai + k) * TILE_DIM + aj);
                  }

                  float tcol[BLOCK_SIZE];
                  for (uint n = 0; n < BLOCK_SIZE; n++) {
                    if (sgId == n) {
                      for (uint k = 0; k < BLOCK_SIZE; k++) {
                        // returns the value of bcol[n] from the k-th
                        // work-item
                        tcol[k] = sycl::select_from_group(sg, bcol[n], k);
                      }
                    }
                  }

                  for (uint k = 0; k < BLOCK_SIZE; k++) {
                    sg.store(get_accessor_pointer(sMemTileTransposed) +
                                 (aj + k) * TILE_DIM + ai,
                             tcol[k]);
                  }
                }

                // We need to wait here to ensure that all work items
                // have written to local memory before we start reading
                // from it.
                sycl::group_barrier(item.get_group());

                for (int i = 0; i < TILE_DIM; i += BLOCK_ROWS) {
                  sycl::id dataIdx{groupOffset_1 + localId[0] + i,
                                   groupOffset_0 + localId[1]};
                  sycl::id smemTileIdx{localId[0] + i, localId[1]};

                  // coalesced write to gmem from smem
                  d_odata[dataIdx] = sMemTileTransposed[smemTileIdx];
                }
              });
        });
        q.wait_and_throw();
      };
      util::benchmark(simple_copy, numIters, Nc * Nr,
                      "Simple Non-Coalesced Tiled Copy");
      util::benchmark(simple_coalesced_copy, numIters, Nc * Nr,
                      "Simple Tiled Copy");
      util::benchmark(naive_transpose, numIters, Nc * Nr, "Naive Transpose");
      util::benchmark(smem_copy, numIters, Nc * Nr, "Tiled SMEM Copy");
      util::benchmark(smem_transpose, numIters, Nc * Nr,
                      "Tiled SMEM Transpose");
      util::benchmark(smem_transpose_no_bank_conflict, numIters, Nc * Nr,
                      "Tiled SMEM Transpose avoiding Bank Conflict");
      util::benchmark(tiled_subgroup_shuffle, numIters, Nc * Nr,
                      "Tiled SMEM Transpose with sub-group shuffle functions");
    }
  } catch (const sycl::exception &e) {
    std::cout << "Exception caught: " << e.what() << std::endl;
  }
