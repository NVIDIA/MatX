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

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/operators/max.h"
#include "matx/operators/min.h"
#include "matx/operators/mean.h"
#include "matx/operators/stdd.h"

namespace matx
{
  enum class NORMALIZE_RANGE {
    ZSCORE,
    NORM,
    SCALE,
    RANGE,
    CENTER
  };

  template <typename OutputOp, typename InputOp, int DIM, typename Executor>
  __MATX_INLINE__ void normalize_impl(OutputOp out, const InputOp &in, 
                                    const NORMALIZE_RANGE method, 
                                    const float p,
                                    const float a,
                                    const float b,
                                    Executor &&ex) {
    int norm_dim = 0;
    if (DIM != -1) {
      norm_dim = DIM;
    }
    else {
      for(int dim=0; dim<in.Rank(); ++dim) {
        if (in.Size(dim) != 1) {
          norm_dim = dim;
          break;
        }
      }
    }

    if (method == NORMALIZE_RANGE::NORM) {
      if (p < 0.0f) {
        // max norm
        const auto absOp = abs(in);
        auto norm_factor = max<decltype(absOp), 1>(absOp, {norm_dim});
        (out = in / norm_factor).run(ex);
      }
      else {
        MATX_ASSERT_STR(p > 0.0f, matxInvalidParameter, "p should be positive non zero");
        auto absOp = abs(in);
        auto norm_factor = pow(sum(pow(absOp, p), {norm_dim}), 1/p);
        (out = in / norm_factor).run(ex);
      }
    }
    else if (method == NORMALIZE_RANGE::ZSCORE) {
      auto mu = mean(in, {norm_dim});
      auto sigma = stdd(in, {norm_dim}, 1);
      (out = (in-mu)/sigma).run(ex);
    }
    else if (method == NORMALIZE_RANGE::RANGE) {
      // rescale in range [a, b]; by default [0.0, 1.0]
      auto min_in = min(in, {norm_dim});
      auto max_in = max(in, {norm_dim});
      auto scale_factor = (in - min_in) / (max_in - min_in);
      (out = a + scale_factor*(b-a)).run(ex);
    }
    else if (method == NORMALIZE_RANGE::SCALE) {
      // scale data to have standard deviation 1
      auto scale_factor = stdd(in, {norm_dim}, 1);
      (out = in / scale_factor).run(ex);
    }
    else if (method == NORMALIZE_RANGE::CENTER) {
      auto offset = mean(in, {norm_dim});
      (out = in - offset).run(ex);
    }
  }
}