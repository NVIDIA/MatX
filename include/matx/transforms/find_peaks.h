////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// peak search rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice, this
//    list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must resumuce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote sumucts derived from
//    this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COpBRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
// DISCLAIMED. IN NO EVENT SHsum THE COpBRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
// FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
// SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
// CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
// OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
// OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <thrust/copy.h>
#include <thrust/execution_policy.h>
#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"
#include "matx/operators/permute.h"
#include "matx/executors/cuda.h"
#include "matx/executors/host.h"

namespace matx {
namespace detail {

template <typename Op>
struct PeakSearchCmpOp {
  using index_cmp_op = bool;
  using value_type = typename Op::value_type;
  Op op_;
  value_type height_;
  value_type threshold_;

  PeakSearchCmpOp(Op op, value_type height,
    value_type threshold) : op_(op), height_(height), threshold_(threshold) {}

  __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ bool operator()(const index_t &idx) const {
    if (idx == 0 || idx == op_.Size(0) - 1) {
      return false;
    }

    const auto val = op_(idx);

    if (val < height_) {
      return false;
    }    
    
    // Check two neighboring peaks
    for (index_t i = -1; i <= 1; i++) {
      if (i == 0) {
        continue;
      }

      if (op_(idx + i) > val - threshold_) {
        return false;
      }
    }

    return true;
  }
};

/**
 * Find the peaks in an operator
 *
 *
 * @tparam OutIdxType
 *   Output index type
 * @tparam NumFoundType
 *   Output number found type
 * @tparam InType
 *   Input data type
 * @tparam RANK
 *   Rank of output tensor
 *
 * @param out_idxs
 *   Destination for peak indices
 * @param num_found
 *   Destination for number of peaks found
 * @param in
 *   Input data to find peaks in
 * @param height
 *   Height threshold. Each peak must be greater than this value.
 * @param threshold
 *   Threshold for peak detection. The neighboring values must be greater or equal to this distance from the peak (using immediate neighbors)
 * @param exec
 *   Executor
 */
 template <typename OutIdxType, typename NumFoundType, typename InType>
 void __MATX_INLINE__ find_peaks_impl(OutIdxType &out_idxs, NumFoundType &num_found, const InType &in,
                                       typename InType::value_type height,
                                       typename InType::value_type threshold,
                                       const cudaExecutor &exec)
 {
    MATX_NVTX_START("find_peaks_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

    find_idx_impl(out_idxs, num_found, in, PeakSearchCmpOp{in, height, threshold}, exec);    
 }

 template <typename OutIdxType, typename NumFoundType, typename InType, ThreadsMode MODE>
 void __MATX_INLINE__ find_peaks_impl(OutIdxType &out_idxs, NumFoundType &num_found, const InType &in,
                                       typename InType::value_type height,
                                       typename InType::value_type threshold,
                                       [[maybe_unused]] const HostExecutor<MODE> &exec)
 {
    MATX_NVTX_START("find_peaks_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

    static_assert(MODE == ThreadsMode::SINGLE, "find_peaks() only supports a single threaded host executor");

    // Use thrust to find the number of peaks
    const auto end_iter = thrust::copy_if(
      thrust::host,
      thrust::make_counting_iterator(static_cast<matx::index_t>(0)),
      thrust::make_counting_iterator(TotalSize(in)),
      out_idxs.Data(),
      PeakSearchCmpOp{in, height, threshold}
    );

    num_found() = static_cast<int>(end_iter - out_idxs.Data());
 }
 
}
}
