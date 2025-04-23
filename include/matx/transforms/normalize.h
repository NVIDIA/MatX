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
                                    const NORMALIZE_RANGE method, Executor &&ex) {
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
      // max norm
      const auto absOp = abs(in);
      auto norm_factor = max<decltype(absOp), 1>(absOp, {norm_dim});
      (out = in / norm_factor).run(ex);
    }
    else if (method == NORMALIZE_RANGE::ZSCORE) {
      auto mu = mean(in, {norm_dim});
      auto sigma = stdd(in, {norm_dim}, 1);
      (out = (in-mu)/sigma).run(ex);
    }
    else if (method == NORMALIZE_RANGE::RANGE) {
      // rescale in range a=0 to b=1
      float a = 0.0;
      float b = 1.0;
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