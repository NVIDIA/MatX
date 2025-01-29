#pragma once

#include <cstdint>
#include <cstdio>
#include <type_traits>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/operators/sum.h"

namespace matx
{
  enum class NORMALIZE_RANGE {
    ZSCORE,
    NORM,
    SCALE,
    RANGE
  };

  template <typename OutputOp, typename InputOp, typename Executor>
  __MATX_INLINE__ void normalize_impl(OutputOp out, const InputOp &in, 
                                    const NORMALIZE_RANGE method, Executor &&ex) {
    int norm_dim = 0;
    for(int dim=0; dim<in.Rank(); ++dim) {
      if (in.Size(dim) != 1) {
        norm_dim = dim;
        break;
      }
    }

    if (method == NORMALIZE_RANGE::NORM) {
      // max norm
      const auto absOp = abs(in);
      auto norm_factor = max<decltype(absOp), 1>(absOp, {norm_dim});
      (out = in / norm_factor).run(ex);
    }
  }
}