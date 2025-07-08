#pragma once

#include "cuComplex.h"
#include <complex>
#include <cuda.h>
#include <iomanip>
#include <stdint.h>
#include <stdio.h>
#include <vector>

namespace matx {

namespace detail_filter {
  constexpr uint32_t MAX_BATCHES = 10000;
  constexpr uint32_t BLOCK_SIZE_RECURSIVE = 1024;
  constexpr uint32_t CORR_COLS = BLOCK_SIZE_RECURSIVE;
  constexpr uint32_t MAX_BLOCKS_PER_BATCH = 1000;
  constexpr uint32_t RECURSIVE_VALS_PER_THREAD = 8;
  constexpr uint32_t MAX_NON_RECURSIVE_COEFFS = 4;
  constexpr uint32_t MAX_RECURSIVE_COEFFS = 4;
  constexpr uint32_t WARP_SIZE = 32;
  using COMPLEX_TYPE = cuComplex;
  constexpr uint32_t RECURSIVE_CHUNK_SIZE = BLOCK_SIZE_RECURSIVE * RECURSIVE_VALS_PER_THREAD;
  constexpr uint32_t MAX_SIGNAL_LEN_PER_BATCH =
    (BLOCK_SIZE_RECURSIVE * RECURSIVE_VALS_PER_THREAD * MAX_BLOCKS_PER_BATCH);
};
using namespace detail_filter;

typedef enum {
  STATUS_FLAG_INCOMPLETE = 0,
  STATUS_FLAG_PARTIAL_COMPLETE = 1,
  STATUS_FLAG_FULL_COMPLETE = 2,
} STATUS_FLAGS;

#ifdef __CUDACC__
// Chunk ID assignment used for atomic incrementing between blocks
static __device__ uint32_t cid_assign[MAX_BATCHES] = {0};


template <uint32_t num_recursive, uint32_t num_non_recursive, typename OutType,
          typename InType, typename FilterType>
__global__ __launch_bounds__(BLOCK_SIZE_RECURSIVE, 1) void RecursiveFilter(
    OutType d_out, InType d_in, const FilterType *__restrict__ d_nrec,
    const FilterType *__restrict__ d_corr,
    volatile FilterType *__restrict__ d_full_carries,
    volatile FilterType *__restrict__ d_part_carries, index_t len,
    volatile int *__restrict__ d_status,
    const FilterType *__restrict__ d_last_corrections)
{
  using intype_strip = typename InType::type;

  __shared__ intype_strip
      s_exch[1 + (1 + BLOCK_SIZE_RECURSIVE) *
                     cuda::std::max(num_non_recursive - 1,
                         num_recursive)]; // Data exchange between threads
  __shared__ uint32_t s_chunk_id;
  __shared__ FilterType
      s_corr[1 + CORR_COLS * num_recursive]; // Local register cache of
                                             // correction values. Add one in
                                             // case where num_recursive is 0
                                             // since nvcc doesn't like that
  intype_strip tmp[RECURSIVE_VALS_PER_THREAD];
  intype_strip vals[RECURSIVE_VALS_PER_THREAD];
  intype_strip r_nonr[cuda::std::max(MAX_NON_RECURSIVE_COEFFS, MAX_RECURSIVE_COEFFS)];
  const uint32_t lane = threadIdx.x & 31;
  const uint32_t warp_id = threadIdx.x / WARP_SIZE;
// const index_t batch_offset = blockIdx.y * len;

// Load non-recursive coefficients
MATX_LOOP_UNROLL
  for (uint8_t i = 0; i < MAX_NON_RECURSIVE_COEFFS; i++) {
    if (i < num_non_recursive) {
      r_nonr[i] = d_nrec[i];
    }
    else {
      if constexpr (is_cuda_complex_v<InType>) {
        r_nonr[i] = make_cuFloatComplex(0.0, 0.0);
      }
      else {
        r_nonr[i] = 0.0;
      }
    }
  }

  // Get out chunk ID
  if (threadIdx.x == 0) {
    s_chunk_id = atomicInc(&cid_assign[blockIdx.y], gridDim.x - 1);
  }

#ifdef DEBUG
  if (threadIdx.x == 0)
    printf("batch=%d chunk=%d block=%d threadIdx.x=%d scheduled\n", blockIdx.y,
           s_chunk_id, blockIdx.x, threadIdx.x);
#endif
  __syncthreads();

  // We use the same nomenclature as the PLR paper here. Namely, a chunk is the
  // entire set of data a block processes
  const int chunk_id =
      s_chunk_id; // register this value since it's used in many places

  index_t tid =
      static_cast<index_t>(chunk_id) * blockDim.x * RECURSIVE_VALS_PER_THREAD +
      threadIdx.x;

  // Copy signal input. If we're a thread that needs to share data with other
  // blocks for the map step, store that value as well
  if (tid < len) {
MATX_LOOP_UNROLL
    for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
      vals[r] = d_in(blockIdx.y, tid + BLOCK_SIZE_RECURSIVE * r);
    }

    if (lane > WARP_SIZE - num_non_recursive) {
MATX_LOOP_UNROLL
      for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
        s_exch[((BLOCK_SIZE_RECURSIVE / WARP_SIZE) * r + (warp_id + 1)) *
                   (num_non_recursive - 1) +
               (WARP_SIZE - lane - 1)] =
            vals[r]; // Leave one spot for first warp to pull from last block
      }
    }

    if (threadIdx.x < num_non_recursive - 1) {
      if (chunk_id == 0) {
        if constexpr (is_cuda_complex_v<InType>) {
          s_exch[threadIdx.x] = make_cuFloatComplex(0.0, 0.0);
        }
        else {
          s_exch[threadIdx.x] = 0;
        }
      }
      else {
        s_exch[threadIdx.x] =
            d_in(blockIdx.y, chunk_id * blockDim.x - threadIdx.x - 1);
      }
    }
  }

  // Copy all correction coefficients
  for (uint32_t i = 0; i < CORR_COLS * num_recursive; i += blockDim.x) {
    s_corr[i + threadIdx.x] = d_corr[i + threadIdx.x];
  }

  __syncthreads();

  // TODO: Do a map operation optimized for num_recursive == 0. This would
  // ignore the data layout needed for most of this code, and instead do a
  // sliding window FIR with haloing in shared memory

  // Map operation
  if constexpr (num_non_recursive > 0) {
MATX_LOOP_UNROLL
    for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
// Load all values from other threads in our warp. Some values will not be used
// in lanes that straddle warps
MATX_LOOP_UNROLL
      for (uint32_t nrec = 1; nrec < num_non_recursive; nrec++) {
        if constexpr (is_cuda_complex_v<InType>) {
          *reinterpret_cast<uint64_t *>(&tmp[nrec - 1]) =
              __shfl_up_sync(~0, *reinterpret_cast<uint64_t *>(&vals[r]), nrec);
        }
        else {
          tmp[nrec - 1] = __shfl_up_sync(~0, vals[r], nrec);
        }
      }

      if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * r) <
          static_cast<uint32_t>(len)) { // Make sure this value is within bounds of the signal
        if constexpr (is_cuda_complex_v<InType>) {
          vals[r] = cuCmulf(vals[r], r_nonr[0]);
        }
        else {
          vals[r] *= r_nonr[0];
        }

        int32_t lid = lane - 1;
// Now apply all coefficients to previous values
MATX_LOOP_UNROLL
        for (uint32_t nrec = 1; nrec < num_non_recursive; nrec++) {
          if (lid < 0) { // need to grab from shm since it's from another warp
            if constexpr (is_cuda_complex_v<InType>) {
              vals[r] = cuCaddf(
                  vals[r],
                  cuCmulf(s_exch[(((BLOCK_SIZE_RECURSIVE / WARP_SIZE)) * r +
                                  warp_id + 1) +
                                 lid],
                          r_nonr[nrec]));
            }
            else {
              vals[r] += s_exch[(((BLOCK_SIZE_RECURSIVE / WARP_SIZE)) * r +
                                 (warp_id + 1)) *
                                    (num_non_recursive - 1) +
                                lid] *
                         r_nonr[nrec];
            }
          }
          else {
            if constexpr (is_cuda_complex_v<InType>) {
              vals[r] = cuCaddf(vals[r], cuCmulf(tmp[nrec - 1], r_nonr[nrec]));
            }
            else {
              vals[r] += tmp[nrec - 1] * r_nonr[nrec];
            }
          }

          lid--;
        }
      }
    }
  }

  // First pass of corrections for phase 1
  int grptid;

// 1->2
MATX_LOOP_UNROLL
  for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
    if constexpr (is_cuda_complex_v<InType>) {
      *reinterpret_cast<uint64_t *>(&tmp[0]) =
          __shfl_sync(~0, *reinterpret_cast<uint64_t *>(&vals[r]), 0, 2);
    }
    else {
      tmp[0] = __shfl_sync(~0, vals[r], 0, 2);
    }

    if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * r) <
        len) { // Make sure this value is within bounds of the signal
      if ((threadIdx.x & 1) == 1) {
        if constexpr (is_cuda_complex_v<InType>) {
          vals[r] = cuCaddf(vals[r], cuCmulf(s_corr[0], tmp[0]));
        }
        else {
          vals[r] += s_corr[0] * tmp[0];
        }
      }
    }
  }

// Loop through correcting 2->4, 4->8, 8->16, and 16->32. At the end of this
// step the first 32 values will be correct
MATX_LOOP_UNROLL
  for (int32_t wl = 2; wl <= 16; wl *= 2) {
    grptid = threadIdx.x & (2 * wl - 1);

    // overload register tmp[0] with a predicate instead of branching later
    if constexpr (is_cuda_complex_v<InType>) {
      tmp[0] = (grptid > (wl - 1)) ? make_cuFloatComplex(1.0, 0.0)
                                   : make_cuFloatComplex(0.0, 0.0);
    }
    else {
      tmp[0] = (grptid > (wl - 1)) ? 1.0 : 0.0;
    }

MATX_LOOP_UNROLL
    for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
// Load all of the values we need from other threads in the warp
MATX_LOOP_UNROLL
      for (uint32_t rec = 0; rec < cuda::std::min(num_recursive, static_cast<uint32_t>(wl)); rec++) {
        if constexpr (is_cuda_complex_v<InType>) {
          *reinterpret_cast<uint64_t *>(&tmp[rec + 1]) =
              __shfl_sync(~0, *reinterpret_cast<uint64_t *>(&vals[r]),
                          wl - rec - 1, wl * 2);
        }
        else {
          tmp[rec + 1] = __shfl_sync(~0, vals[r], wl - rec - 1, wl * 2);
        }
      }

      if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * r) <
          static_cast<uint32_t>(len)) { // Make sure this value is within bounds of the signal
// Now apply those values
MATX_LOOP_UNROLL
        for (uint32_t rec = 0; rec < cuda::std::min(num_recursive, static_cast<uint32_t>(wl)); rec++) {
          if constexpr (is_cuda_complex_v<InType>) {
            vals[r] =
                cuCaddf(vals[r],
                        cuCmulf(cuCmulf(s_corr[CORR_COLS * rec + (grptid - wl)],
                                        tmp[rec + 1]),
                                tmp[0]));
          }
          else {
            vals[r] +=
                s_corr[CORR_COLS * rec + (grptid - wl)] * tmp[rec + 1] * tmp[0];
          }
        }
      }
    }
  }

  __syncthreads();

  // Warp-level operations are done. Now we need to do further aggregations
  // until the chunk size is equal to MAX_CHUNK_SIZE. With 8 values per thread,
  // each block is finishing 8 chunks (assuming a chunk size of 1024). At this
  // stage the first 32 values are correct, and we need to do 5 more iterations
  // (64, 128, 256, 512, 1024) to hit our chunk size before we switch to
  // correcting 1024-element chunks. Prime the next stage by pushing the local
  // carries into shared memory for the next stage. Note that since we're
  // starting by processing Batches of 64 at a time where only the second half
  // do the work, only the first warp must write its values out. Do log2 steps
  // of the block size for the remaining values to correct until the entire
  // block is done. This loop will always require blockDim.x/2 threads active
  // while correcting
  int32_t sub_group_base;
  int32_t sub_group_idx;
  uint32_t cor_size = 32;
  int32_t dcor = 2 * cor_size - 1;
  int32_t cor_log2 = 5;
MATX_LOOP_UNROLL
  do {
    sub_group_base =
        threadIdx.x >>
        (cor_log2 + 1); // base subgroup for each of the 8 iterations
    sub_group_idx = (threadIdx.x & dcor) - cor_size;

    __syncthreads();

    // Pick off the last num_recursive threads in the block
    if (sub_group_idx < 0 && (-sub_group_idx <= static_cast<int32_t>(num_recursive))) {
MATX_LOOP_UNROLL
      for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
        s_exch[num_recursive * (sub_group_base +
                                r * (BLOCK_SIZE_RECURSIVE / 2) / cor_size) -
               sub_group_idx - 1] = vals[r];
      }
    }

    __syncthreads();

    // Each subgroup applies the corrections
    if (sub_group_idx >= 0) {
MATX_LOOP_UNROLL
      for (uint32_t vpt = 0; vpt < RECURSIVE_VALS_PER_THREAD; vpt++) {
        if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * vpt) <
            len) { // Make sure this value is within bounds of the signal
MATX_LOOP_UNROLL
          for (uint32_t r = 0; r < num_recursive; r++) {
            tmp[r] = s_exch[(sub_group_base +
                             vpt * ((BLOCK_SIZE_RECURSIVE / 2) / cor_size)) *
                                num_recursive +
                            r];
            if constexpr (is_cuda_complex_v<InType>) {
              vals[vpt] = cuCaddf(
                  vals[vpt],
                  cuCmulf(s_corr[CORR_COLS * r + sub_group_idx], tmp[r]));
            }
            else {
              vals[vpt] += s_corr[CORR_COLS * r + sub_group_idx] * tmp[r];
            }
          }
        }
      }
    }

    cor_size *= 2;
    cor_log2 += 1;
    dcor = 2 * cor_size - 1;
  } while (cor_size <= (BLOCK_SIZE_RECURSIVE >> 1));

  __syncthreads();

#ifdef DEBUG
  if (threadIdx.x < 65) {
    printf("after block reduce thread %d, vals=%.2f %.2f %.2f %.2f %.2f %.2f "
           "%.2f %.2f %.2f %.2f\n",
           threadIdx.x, vals[0], vals[1], vals[2], vals[3], vals[4], vals[5],
           vals[6], vals[7], s_corr[0], s_corr[1]);
  }
#endif

  // Start phase 2. Loop starts at 1 since the first block is already done. In
  // order to keep things in registers, only a subset of warps in the block are
  // doing active work for each loop, and it's a different subset in each loop.
  uint32_t block_idx = 1;
  if (chunk_id == 0) {
    do {
      if (threadIdx.x >= BLOCK_SIZE_RECURSIVE - num_recursive) {
        s_exch[BLOCK_SIZE_RECURSIVE - threadIdx.x - 1] = vals[block_idx - 1];
      }

      __syncthreads();

      if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * block_idx) <
          len) { // Make sure this value is within bounds of the signal
MATX_LOOP_UNROLL
        for (uint32_t r = 0; r < num_recursive; r++) {
          if constexpr (is_cuda_complex_v<InType>) {
            vals[block_idx] = cuCaddf(
                vals[block_idx],
                cuCmulf(s_exch[r], s_corr[r * CORR_COLS + threadIdx.x]));
          }
          else {
            vals[block_idx] += s_exch[r] * s_corr[r * CORR_COLS + threadIdx.x];
          }
        }
      }

      __syncthreads();
    } while (++block_idx < RECURSIVE_VALS_PER_THREAD);
  }

  // Write out the full and partial carry values. Only do this if this isn't the
  // last block
  if (threadIdx.x >= (BLOCK_SIZE_RECURSIVE - num_recursive)) {
    if (chunk_id == 0) {
      if constexpr (is_cuda_complex_v<InType>) {
        *reinterpret_cast<volatile uint64_t *>(
            &d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                            chunk_id * num_recursive + num_recursive -
                            (BLOCK_SIZE_RECURSIVE - threadIdx.x)]) =
            *reinterpret_cast<volatile uint64_t *>(
                &vals[RECURSIVE_VALS_PER_THREAD - 1]);
      }
      else {
        d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                       chunk_id * num_recursive + num_recursive -
                       (BLOCK_SIZE_RECURSIVE - threadIdx.x)] =
            vals[RECURSIVE_VALS_PER_THREAD - 1];
      }

      __threadfence(); // Ensure carries are written before flag update is
                       // observed

      d_status[blockIdx.y * gridDim.x + chunk_id] = STATUS_FLAG_FULL_COMPLETE;
    }
    else {
      if constexpr (is_cuda_complex_v<InType>) {
        *reinterpret_cast<volatile uint64_t *>(
            &d_part_carries[blockIdx.y * gridDim.x * num_recursive +
                            chunk_id * num_recursive + num_recursive -
                            (BLOCK_SIZE_RECURSIVE - threadIdx.x)]) =
            *reinterpret_cast<volatile uint64_t *>(
                &vals[RECURSIVE_VALS_PER_THREAD - 1]);
      }
      else {
        d_part_carries[blockIdx.y * gridDim.x * num_recursive +
                       chunk_id * num_recursive + num_recursive -
                       (BLOCK_SIZE_RECURSIVE - threadIdx.x)] =
            vals[RECURSIVE_VALS_PER_THREAD - 1];
      }

      __threadfence(); // Ensure carries are written before flag update is
                       // observed

      d_status[blockIdx.y * gridDim.x + chunk_id] =
          STATUS_FLAG_PARTIAL_COMPLETE;
    }
  }

  __syncthreads();

  // At this point our block is corrected based on all the information the
  // threads had. Now we do the grid synchronization by setting a flag saying we
  // are done, and look back at our immediate predecessor to see if it's done.
  // Have the first warp loop through to see if anything is ready.
  if (chunk_id > 0) {
    int to_check = chunk_id - threadIdx.x - 1;
    int lstatus = STATUS_FLAG_PARTIAL_COMPLETE;
    int full_complete;
    unsigned int last_full;

    if (threadIdx.x < 32) {
      // Keep looping until we have both a full carry present, and all the
      // carries after the full are partial. If a carry is full, but the partial
      // ones in between are not complete, we can't finish since there's not
      // enough information to finish the computation
      do {
        if (to_check >= 0) {
          lstatus = d_status[blockIdx.y * gridDim.x + to_check];
        }
      } while (__any_sync(~0, lstatus == STATUS_FLAG_INCOMPLETE) ||
               __all_sync(~0, lstatus != STATUS_FLAG_FULL_COMPLETE));

      // Get a bit mask of which threads. Note that the threads are "best"
      // starting from the LSB since those are closest to the answer
      full_complete = __ballot_sync(~0, lstatus == STATUS_FLAG_FULL_COMPLETE);
      last_full = __ffs(full_complete) - 1;

      // Pull in all the partial carries first by as many threads that can get
      // them
      if (threadIdx.x < last_full) {
MATX_LOOP_UNROLL
        for (uint32_t r = 0; r < num_recursive; r++) {
          if constexpr (is_cuda_complex_v<InType>) {
            *reinterpret_cast<volatile uint64_t *>(
                &s_exch[threadIdx.x * num_recursive + r]) =
                *reinterpret_cast<volatile uint64_t *>(
                    &d_part_carries[blockIdx.y * gridDim.x * num_recursive +
                                    (chunk_id - 1 - threadIdx.x) *
                                        num_recursive +
                                    r]);
          }
          else {
            s_exch[threadIdx.x * num_recursive + r] =
                d_part_carries[blockIdx.y * gridDim.x * num_recursive +
                               (chunk_id - 1 - threadIdx.x) * num_recursive +
                               r];
          }
        }
      }

      __syncwarp();

      // The first thread will do all the FMAs until we can finally apply the
      // value to all threads
      if (threadIdx.x == 0) {
MATX_LOOP_UNROLL
        for (uint32_t r = 0; r < num_recursive; r++) {
          if constexpr (is_cuda_complex_v<InType>) {
            *reinterpret_cast<volatile uint64_t *>(&tmp[r]) =
                *reinterpret_cast<volatile uint64_t *>(
                    &d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                                    (chunk_id - 1 - last_full) * num_recursive +
                                    r]);
          }
          else {
            tmp[r] =
                d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                               (chunk_id - 1 - last_full) * num_recursive + r];
          }
        }

        // Reuse the non-recursive registers here to save some shared memory
        for (uint32_t r = 0; r < num_recursive; r++) {
          r_nonr[r] = d_last_corrections[r];
        }

        // Roll up the lookback values
        for (uint32_t lookback = 0; lookback < last_full; lookback++) {
          for (uint32_t r = 0; r < num_recursive; r++) {
            if constexpr (is_cuda_complex_v<InType>) {
              tmp[r] = cuCaddf(s_exch[lookback * num_recursive + r],
                               cuCmulf(tmp[r], r_nonr[r]));
            }
            else {
              tmp[r] =
                  s_exch[lookback * num_recursive + r] + tmp[r] * r_nonr[r];
            }
          }
        }

        for (uint32_t r = 0; r < num_recursive; r++) {
          s_exch[num_recursive - r - 1] = tmp[r];
        }
      }
    }

    __syncthreads();

    // Now loop through applying all the correction factors. We've already
    // loaded the exchange values for the initial priming above
    block_idx = 0;
    do {
MATX_LOOP_UNROLL
      for (uint32_t r = 0; r < num_recursive; r++) {
        if ((blockIdx.x * RECURSIVE_VALS_PER_THREAD + threadIdx.x * r) < len) {
          if constexpr (is_cuda_complex_v<InType>) {
            vals[block_idx] = cuCaddf(
                vals[block_idx],
                cuCmulf(s_exch[r], s_corr[r * CORR_COLS + threadIdx.x]));
          }
          else {
            vals[block_idx] += s_exch[r] * s_corr[r * CORR_COLS + threadIdx.x];
          }
        }
      }

      __syncthreads();

      if (threadIdx.x >= BLOCK_SIZE_RECURSIVE - num_recursive) {
        s_exch[BLOCK_SIZE_RECURSIVE - threadIdx.x - 1] = vals[block_idx - 1];
      }

      __syncthreads();
    } while (++block_idx < RECURSIVE_VALS_PER_THREAD);

    // Finally, let everyone know our chunk is done
    if (threadIdx.x >= BLOCK_SIZE_RECURSIVE - num_recursive) {
      if constexpr (is_cuda_complex_v<InType>) {
        *reinterpret_cast<volatile uint64_t *>(
            &d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                            chunk_id * num_recursive + num_recursive -
                            (BLOCK_SIZE_RECURSIVE - threadIdx.x)]) =
            *reinterpret_cast<volatile uint64_t *>(
                &vals[RECURSIVE_VALS_PER_THREAD - 1]);
      }
      else {
        d_full_carries[blockIdx.y * gridDim.x * num_recursive +
                       chunk_id * num_recursive + num_recursive -
                       (BLOCK_SIZE_RECURSIVE - threadIdx.x)] =
            vals[RECURSIVE_VALS_PER_THREAD - 1];
      }

      __threadfence();
      d_status[blockIdx.y * gridDim.x + chunk_id] = STATUS_FLAG_FULL_COMPLETE;
    }
  }

// Write solution out
MATX_LOOP_UNROLL
  for (uint32_t r = 0; r < RECURSIVE_VALS_PER_THREAD; r++) {
    if ((tid + r * BLOCK_SIZE_RECURSIVE) < len) {
      d_out(blockIdx.y, tid + r * BLOCK_SIZE_RECURSIVE) = vals[r];
    }
  }

  return;
}
#endif

}; // namespace matx
