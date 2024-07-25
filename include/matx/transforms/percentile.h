#pragma once

#include <cfloat>

#include "matx/core/cache.h"
#include "matx/core/error.h"
#include "matx/core/get_grid_dims.h"
#include "matx/core/nvtx.h"
#include "matx/core/tensor.h"
#include "matx/core/type_utils.h"
#include "matx/core/utils.h"
#include "matx/transforms/cub.h"
#include "matx/transforms/copy.h"
#include "matx/core/half.h"

namespace matx {

  enum class PercentileMethod {
    LINEAR,
    LOWER,
    HIGHER,
    HAZEN,
    WEIBULL,
    MEDIAN_UNBIASED,
    NORMAL_UNBIASED,
    MIDPOINT,
    NEAREST
  }; 

namespace detail {

/**
 * Calculate the median of values in a tensor
 *
 * Calculates the median of rows in a tensor. The median is computed by sorting
 * the data into a temporary tensor, then picking the middle element of each
 * row. For an even number of items, the mean of the two middle elements is
 * selected. Currently only works on tensor views as input since it uses CUB
 * sorting as a backend, and the tensor views must be rank 2 reducing to rank 1,
 * or rank 1 reducing to rank 0.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam RANK_IN
 *   Input rank
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType, typename Executor>
void __MATX_INLINE__ percentile_impl(OutType dest, const InType &in, uint32_t q, PercentileMethod method, Executor &&exec)
{
  MATX_NVTX_START("percentile_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  double alpha = 0.;
  double beta = 0.;
  double subidx;

  // Compute alpha and beta if used
  switch (method) {
    case PercentileMethod::LINEAR: {
      alpha = 1.0;
      beta  = 1.0;
      break;
    }
    case PercentileMethod::HAZEN: {
      alpha = 0.5;
      beta  = 0.5;
      break;
    }
    case PercentileMethod::WEIBULL: {
      alpha = 0;
      beta  = 0;
      break;
    }
    case PercentileMethod::MEDIAN_UNBIASED: {
      alpha = 1./3;
      beta  = 1./3;
      break;
    }
    case PercentileMethod::NORMAL_UNBIASED: {
      alpha = 3./8;
      beta  = 3./8;
      break;
    }
    case PercentileMethod::LOWER: [[fallthrough]];
    case PercentileMethod::HIGHER: [[fallthrough]];
    case PercentileMethod::NEAREST: [[fallthrough]];
    case PercentileMethod::MIDPOINT:
      break;
    default: {
      MATX_ASSERT_STR(false, matxInvalidParameter, "Method for percentile() not supported yet");
      return;
    }
  }

  if constexpr (OutType::Rank() == 0) {
    auto insize = TotalSize(in);
    matx::tensor_t<typename InType::value_type, 1> sort_out;
    if constexpr (is_cuda_executor_v<Executor>) {
      make_tensor(sort_out, {insize}, MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
    }
    else {
      make_tensor(sort_out, {insize}, MATX_HOST_MEMORY);  
    }

    sort_impl(sort_out, flatten(in), SORT_DIR_ASC, exec);

    // If we're landing directly onto an index after the q multiplication we should make sure that's the case
    // and not allow floating point error to move us to the wrong index.
    double base_index = ((q * (insize - 1)) % 100) == 0 ? static_cast<double>(q * (insize - 1) / 100) : static_cast<double>(insize - 1) * q/100.;

    if (q == 0) {
      (dest = at(sort_out, 0)).run(exec);
    }
    else if (q == 100) {
      (dest = at(sort_out, insize - 1)).run(exec);
    }
    else {
      switch (method) {
        case PercentileMethod::LINEAR: [[fallthrough]];
        case PercentileMethod::HAZEN: [[fallthrough]];
        case PercentileMethod::WEIBULL: [[fallthrough]];
        case PercentileMethod::MEDIAN_UNBIASED: [[fallthrough]];
        case PercentileMethod::NORMAL_UNBIASED:
        {
          subidx = q/100. * (static_cast<double>(insize) - alpha - beta + 1) + alpha - 1; 
          auto int_val = at(sort_out, static_cast<index_t>(subidx));
          (dest = at(sort_out,  static_cast<index_t>(subidx)) + 
                                as_type<typename InType::value_type>(
                                  (subidx - std::floor(subidx)) * 
                                  as_type<double>(at(sort_out, static_cast<index_t>(subidx + 1)) - int_val)
                                )
          ).run(exec);
          break;
        }
        case PercentileMethod::LOWER: {
          (dest = at(sort_out, static_cast<index_t>(base_index))).run(exec);
          break;
        }
        case PercentileMethod::HIGHER: {
          (dest = at(sort_out, static_cast<index_t>(cuda::std::ceil(base_index)))).run(exec);
          break;
        }
        case PercentileMethod::MIDPOINT: {
          (dest = as_type<typename InType::value_type>(
                  (at(sort_out, static_cast<index_t>(cuda::std::ceil(base_index))) + 
                   at(sort_out, static_cast<index_t>(cuda::std::ceil(base_index + 1)))) / 
                   static_cast<typename InType::value_type>(2))
          ).run(exec);
          break;
        }
        case PercentileMethod::NEAREST: {
          (dest = at(sort_out, static_cast<index_t>(cuda::std::round(base_index)))).run(exec);
          break;
        }        
        default:
          break;
      }
    }
  }
}

  }
}