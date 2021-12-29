////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx_conv.h"
#include "matx_error.h"
#include "matx_filter_kernels.cuh"
#include "matx_tensor.h"
#include <any>
#include <array>
#include <cstdio>
#include <stdint.h>
#include <type_traits>

namespace matx {
namespace detail {
template <size_t num_recursive, size_t num_non_recursive, 
          typename OutType, typename InType, typename FilterType>
class matxFilter_t {
  static constexpr int RANK = OutType::Rank();
  using filter_tensor = matx::tensor_t<FilterType, 1>;

public:
  matxFilter_t([[maybe_unused]] OutType &o, const InType &i,
               const filter_tensor &h_rec,
               const filter_tensor &h_nonrec)
      : h_nonr_copy(h_nonrec)
  {

    if constexpr (RANK == 1) {
      MATX_ASSERT(o.Size(0) == i.Size(0), matxInvalidSize);

      sig_len = i.Size(0);
      batches = 1;
    }
    else if constexpr (RANK == 2) {
      MATX_ASSERT(o.Size(0) == i.Size(0), matxInvalidSize);
      MATX_ASSERT(o.Size(1) == i.Size(1), matxInvalidSize);

      sig_len = i.Size(1);
      batches = i.Size(0);
    }
    else if constexpr (RANK == 3) {
      MATX_ASSERT(o.Size(0) == i.Size(0), matxInvalidSize);
      MATX_ASSERT(o.Size(1) == i.Size(1), matxInvalidSize);
      MATX_ASSERT(o.Size(2) == i.Size(2), matxInvalidSize);

      sig_len = i.Size(2);
      batches = i.Size(0) * i.Size(1);
    }
    else {
      MATX_ASSERT(o.Size(0) == i.Size(0), matxInvalidSize);
      MATX_ASSERT(o.Size(1) == i.Size(1), matxInvalidSize);
      MATX_ASSERT(o.Size(2) == i.Size(2), matxInvalidSize);
      MATX_ASSERT(o.Size(3) == i.Size(3), matxInvalidSize);

      sig_len = i.Size(3);
      batches = i.Size(0) * i.Size(1) * i.Size(2);
      static_assert(RANK == 4);
    }

    Alloc(h_nonrec, h_rec);
  }

  int Alloc(const filter_tensor &h_nonrec,
            const filter_tensor &h_rec)
  {
    MATX_ASSERT(h_rec.Size(0) == num_recursive, matxInvalidSize);
    MATX_ASSERT(h_nonrec.Size(0) == num_non_recursive, matxInvalidSize);

    matxAlloc((void **)&d_nrec, num_non_recursive * sizeof(*d_nrec),
              MATX_DEVICE_MEMORY);
    MATX_CUDA_CHECK(cudaMemcpyAsync(d_nrec, h_nonrec.Data(),
                                    num_non_recursive * sizeof(FilterType),
                                    cudaMemcpyHostToDevice, 0));

    // If we have at least one recursive coefficient,
    // we need to allocate the memory for those
    if (num_recursive > 0) {
      matxAlloc((void **)&d_corr,
                sizeof(FilterType) * num_recursive * CORR_COLS,
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_status,
                sizeof(int) * (sig_len / RECURSIVE_CHUNK_SIZE + 1) * batches);
      matxAlloc((void **)&d_full_carries,
                sizeof(*d_full_carries) * num_recursive *
                    (sig_len / RECURSIVE_CHUNK_SIZE) * batches,
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_part_carries,
                sizeof(*d_part_carries) * num_recursive *
                    (sig_len / RECURSIVE_CHUNK_SIZE) * batches,
                MATX_DEVICE_MEMORY);
      matxAlloc((void **)&d_last_carries,
                sizeof(*d_last_carries) * num_recursive, MATX_DEVICE_MEMORY);
    }

    ComputeCorrectionFactors(h_rec.Data());

    ClearState();

    return 0;
  }

  ~matxFilter_t()
  {
    FreeGpuMem(d_nrec);
    FreeGpuMem(d_corr);
    FreeGpuMem((void *)d_status);
    FreeGpuMem(d_full_carries);
    FreeGpuMem(d_part_carries);
    FreeGpuMem(d_last_carries);
  }

  void FreeGpuMem(void *ptr)
  {
    if (ptr != nullptr) {
      matxFree(ptr);
    }
  }

  void Exec(OutType &o, const InType &i, cudaStream_t stream)
  {
    if (num_recursive > 0) {
      auto grid =
          dim3(static_cast<int>(
                   (sig_len +
                    (BLOCK_SIZE_RECURSIVE * RECURSIVE_VALS_PER_THREAD - 1)) /
                   BLOCK_SIZE_RECURSIVE / RECURSIVE_VALS_PER_THREAD),
               static_cast<int>(batches));
      // Fix this to support different R/N types later
      RecursiveFilter<num_recursive, num_non_recursive,
                      OutType, InType, FilterType>
          <<<grid, BLOCK_SIZE_RECURSIVE, 0, stream>>>(
              o, i, d_nrec, d_corr, d_full_carries, d_part_carries, sig_len,
              d_status, d_last_carries);
    }
    else {
      // Just call the convolution kernel directly if they
      // don't require recursive coefficients. Default to SAME. Do we want to
      // use SAME here or give them an option? IIR doesn't have the same concept
      conv1d(o, i, h_nonr_copy, matxConvCorrMode_t::MATX_C_MODE_SAME, stream);
    }
  }

private:
  void ClearState()
  {
    if (num_recursive > 0) {
      MATX_CUDA_CHECK(cudaMemset(
          (void *)d_status, 0,
          sizeof(int) * (sig_len / RECURSIVE_CHUNK_SIZE + 1) * batches));
      MATX_CUDA_CHECK(cudaMemset(d_full_carries, 0,
                                 sizeof(*d_full_carries) * num_recursive *
                                     (sig_len / RECURSIVE_CHUNK_SIZE) *
                                     batches));
      MATX_CUDA_CHECK(cudaMemset(d_part_carries, 0,
                                 sizeof(*d_part_carries) * num_recursive *
                                     (sig_len / RECURSIVE_CHUNK_SIZE) *
                                     batches));
    }
  }

  void ComputeCorrectionFactors(const FilterType *coeffs)
  {
    FilterType *out = reinterpret_cast<FilterType *>(
        malloc(sizeof(FilterType) * num_recursive * CORR_COLS));
    FilterType *last = reinterpret_cast<FilterType *>(
        malloc(sizeof(*d_last_carries) * num_recursive));
    if (out == nullptr || last == nullptr) {

      if (last != nullptr)
        free(last);
      if (out != nullptr)
        free(out);

      MATX_THROW(matxCudaError, "allocation error");
    }

    for (uint32_t row = 0; row < num_recursive; row++) {
      for (uint32_t col = 0; col < CORR_COLS * RECURSIVE_VALS_PER_THREAD;
           col++) {
        // There are an implicit num_recursive columns prior to
        // the actual coefficients beginning that are needed for computation
        FilterType res;
        if constexpr (std::is_same_v<FilterType, COMPLEX_TYPE>) {
          res.x = 0;
          res.y = 0;
        }
        else {
          res = 0;
        }

        for (uint32_t k = 0; k < num_recursive; k++) {
          int offs = col - k - 1;
          if (offs < 0) {
            // We're in the implicit coefficients
            if constexpr (std::is_same_v<FilterType, COMPLEX_TYPE>) {
              res = (static_cast<int>(row) + 1 == -offs)
                        ? cuCaddf(res, coeffs[k])
                        : res;
            }
            else {
              res += (static_cast<int>(row) + 1 == -offs) ? coeffs[k] : 0;
            }
          }
          else {
            if constexpr (std::is_same_v<FilterType, COMPLEX_TYPE>) {
              res = cuCaddf(
                  res, cuCmulf(coeffs[k], out[row * CORR_COLS + col - k - 1]));
            }
            else {
              res += coeffs[k] * out[row * CORR_COLS + col - k - 1];
            }
          }
        }

        if (col < CORR_COLS) {
          out[row * CORR_COLS + col] = res;
        }
        else if (col == CORR_COLS * RECURSIVE_VALS_PER_THREAD - 1) {
          last[row] = res;
        }
      }
    }

    // Copy to device
    MATX_CUDA_CHECK(cudaMemcpy(d_corr, out,
                               sizeof(*out) * num_recursive * CORR_COLS,
                               cudaMemcpyHostToDevice));
    MATX_CUDA_CHECK(cudaMemcpy(d_last_carries, last,
                               sizeof(*last) * num_recursive,
                               cudaMemcpyHostToDevice));

    free(out);
    free(last);
  }

  // Keep copy if we take the fast path and don't need pointer transformations
  matx::tensor_t<FilterType, 1> h_nonr_copy;
  FilterType *d_corr = nullptr;
  FilterType *d_nrec = nullptr;
  FilterType *d_full_carries = nullptr;
  FilterType *d_part_carries = nullptr;
  volatile int *d_status = nullptr;
  FilterType *d_last_carries = nullptr;
  index_t batches;
  index_t sig_len;
};

/**
 * Parameters needed to execute a recursive filter.
 */
struct FilterParams_t {
  std::vector<std::any> rec; // Type erasure on filter type
  std::vector<std::any> nonrec;
  MatXDataType_t dtype; // Input type
  MatXDataType_t ftype; // Filter type
  size_t hash;
};

/**
 * Crude hash on filter to get a reasonably good delta for collisions. This
 * doesn't need to be perfect, but fast enough to not slow down lookups, and
 * different enough so the common filter parameters change
 */
struct FilterParamsKeyHash {
  std::size_t operator()(const FilterParams_t &k) const noexcept
  {
    return k.hash;
  }
};

/**
 * Test filter parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct FilterParamsKeyEq {
  bool operator()(const FilterParams_t &l, const FilterParams_t &t) const
      noexcept
  {
    if (l.dtype == t.dtype && l.ftype == t.ftype &&
        l.rec.size() == t.rec.size() && l.nonrec.size() == t.nonrec.size()) {
      // // Now check the coefficients match
      // for (size_t i = 0; i < l.rec.size(); i++) {
      //   if (MatXAnyCmp(l.rec[i], t.rec[i], l.ftype) == false) {
      //     return false;
      //   }
      // }
      // for (size_t i = 0; i < l.nonrec.size(); i++) {
      //   if (MatXAnyCmp(l.nonrec[i], t.nonrec[i], l.ftype) == false) {
      //     return false;
      //   }
      // }
    }

    return true;
  }
};

// Static caches of 1D and 2D FFTs
static matxCache_t<FilterParams_t, FilterParamsKeyHash, FilterParamsKeyEq>
    filter_cache;

} // end namspace detail


/**
 * FIR and IIR filtering
 *
 * matxFilter_t provides an interface for executing recursive (IIR) and
 *non-recursive (FIR) filters. The IIR filter uses the algorithm from "S. Maleki
 *and M. Burtscher. "Automatic Hierarchical Parallelization of Linear
 *Recurrences." 23rd ACM International Conference on Architectural Support for
 *Programming Languages and Operating Systems. March 2018." for an optimized
 *implementation on highly-parallel processors. While the IIR implementation is
 *fast for recursive filters, it is inefficient for non-recursive filtering. If
 *the number of recursive coefficients is 0, the filter operation will revert to
 *use an algorithm optimized for non-recursive filters.
 *
 * @note If you are only using non-recursive filters, it's advised to use the
 *convolution API directly instead since it can be easier to use.
 *
 * @tparam NR
 *   Number of recursive coefficients
 * @tparam NNR
 *   Number of non-recursive coefficients
 * @tparam RANK
 *   Rank of input and output signal
 * @tparam OutType
 *   Ouput type
 * @tparam InType
 *   Input type
 * @tparam FilterType
 *   Filter type
 *
 * @param o
 *   Output tensor
 * @param i
 *   Input tensor
 * @param h_rec
 *   1D input tensor of recursive filter coefficients
 * @param h_nonrec
 *   1D input tensor of recursive filter coefficients
 **/
template <size_t NR, size_t NNR, typename OutType, typename InType,
          typename FilterType>
static auto matxMakeFilter(OutType &o, const InType &i,
                           tensor_t<FilterType, 1> &h_rec,
                           tensor_t<FilterType, 1> &h_nonrec)
{
  return detail::matxFilter_t<NR, NNR, OutType, InType, FilterType>{o, i, h_rec,
                                                                  h_nonrec};
}

/**
 * FIR and IIR filtering
 *
 * matxFilter_t provides an interface for executing recursive (IIR) and
 *non-recursive (FIR) filters. The IIR filter uses the algorithm from "S. Maleki
 *and M. Burtscher. "Automatic Hierarchical Parallelization of Linear
 *Recurrences." 23rd ACM International Conference on Architectural Support for
 *Programming Languages and Operating Systems. March 2018." for an optimized
 *implementation on highly-parallel processors. While the IIR implementation is
 *fast for recursive filters, it is inefficient for non-recursive filtering. If
 *the number of recursive coefficients is 0, the filter operation will revert to
 *use an algorithm optimized for non-recursive filters.
 *
 * @note If you are only using non-recursive filters, it's advised to use the
 *convolution API directly instead since it can be easier to use.
 *
 * @tparam NR
 *   Number of recursive coefficients
 * @tparam NNR
 *   Number of non-recursive coefficients
 * @tparam RANK
 *   Rank of input and output signal
 * @tparam OutType
 *   Ouput type
 * @tparam InType
 *   Input type
 * @tparam FilterType
 *   Filter type
 *
 * @param o
 *   Output tensor
 * @param i
 *   Input tensor
 * @param h_rec
 *   Vector of recursive coefficients
 * @param h_nonrec
 *   Vector of non-recursive coefficients
 **/
template <size_t NR, size_t NNR, typename OutType, typename InType,
          typename FilterType>
static auto matxMakeFilter(OutType &o, const InType &i,
                           const std::array<FilterType, NR> &h_rec,
                           const std::array<FilterType, NNR> &h_nonrec)
{
  tensor_t<FilterType, 1> rec_v({static_cast<index_t>(h_rec.size())});
  tensor_t<FilterType, 1> nonrec_v({static_cast<index_t>(h_nonrec.size())});

  for (size_t j = 0; j < h_rec.size(); j++) {
    rec_v(static_cast<index_t>(j)) = h_rec[j];
  }

  for (size_t j = 0; j < h_nonrec.size(); j++) {
    nonrec_v(static_cast<index_t>(j)) = h_nonrec[j];
  }

  return new detail::matxFilter_t<NR, NNR, OutType, InType, FilterType>{
      o, i, rec_v, nonrec_v};
}


/**
 * FIR and IIR filtering without a plan
 *
 * matxFilter_t provides an interface for executing recursive (IIR) and
 *non-recursive (FIR) filters. The IIR filter uses the algorithm from "S. Maleki
 *and M. Burtscher. "Automatic Hierarchical Parallelization of Linear
 *Recurrences." 23rd ACM International Conference on Architectural Support for
 *Programming Languages and Operating Systems. March 2018." for an optimized
 *implementation on highly-parallel processors. While the IIR implementation is
 *fast for recursive filters, it is inefficient for non-recursive filtering. If
 *the number of recursive coefficients is 0, the filter operation will revert to
 *use an algorithm optimized for non-recursive filters.
 *
 * @note If you are only using non-recursive filters, it's advised to use the
 *convolution API directly instead since it can be easier to use.
 *
 * @tparam NR
 *   Number of recursive coefficients
 * @tparam NNR
 *   Number of non-recursive coefficients
 * @tparam RANK
 *   Rank of input and output signal
 * @tparam OutType
 *   Ouput type
 * @tparam InType
 *   Input type
 * @tparam FilterType
 *   Filter type
 *
 * @param o
 *   Output tensor
 * @param i
 *   Input tensor
 * @param h_rec
 *   Vector of recursive coefficients
 * @param h_nonrec
 *   Vector of non-recursive coefficients
 * @param stream
 *   CUDA stream
 *
 **/
// TODO: Update later once we support compile-time shapes
template <size_t NR, size_t NNR, typename OutType, typename InType,
          typename FilterType>
void filter(OutType &o, const InType &i,
            const std::array<FilterType, NR> h_rec,
            const std::array<FilterType, NNR> h_nonrec, cudaStream_t stream = 0)
{
  // Get parameters required by these tensors
  auto params = detail::FilterParams_t();
  auto rhash = detail::PodArrayToHash<FilterType, NR>(h_rec);
  auto nrhash = detail::PodArrayToHash<FilterType, NNR>(h_nonrec);

  for (size_t j = 0; j < h_rec.size(); j++) {
    params.rec.push_back(h_rec[j]);
  }
  for (size_t j = 0; j < h_nonrec.size(); j++) {
    params.nonrec.push_back(h_nonrec[j]);
  }

  params.dtype = detail::TypeToInt<typename InType::scalar_type>();
  params.ftype = detail::TypeToInt<FilterType>(); // Update when we support different types
  params.hash = rhash + nrhash;

  // Get cache or new FFT plan if it doesn't exist
  auto ret = detail::filter_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = matxMakeFilter<NR, NNR, OutType, InType, FilterType>(
        o, i, h_rec, h_nonrec);
    detail::filter_cache.Insert(params, static_cast<void *>(tmp));

    tmp->Exec(o, i, stream);
  }
  else {
    auto filter_type =
        static_cast<detail::matxFilter_t<NR, NNR, OutType, InType, FilterType> *>(
            ret.value());
    filter_type->Exec(o, i, stream);
  }
}

} // end namespace matx
