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

union HalfBits {
  constexpr HalfBits(short x) : i(x) {}
  HalfBits() = default;
  short i;
  __half h;
  __nv_bfloat16 b;
};

union PascalHalfBits {
  constexpr PascalHalfBits(unsigned short x) : i(x) {}
  PascalHalfBits() = default;
  unsigned int i;
  __half h[2];
  __nv_bfloat16 b[2];
};


namespace matx {
namespace detail {

#ifdef __CUDACC__
template <typename T> constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T maxVal() {
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __half>) {
    constexpr HalfBits tmp{0x7BFF};
    return tmp.h;
  }
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __nv_bfloat16>) {
    constexpr HalfBits tmp{0x7F7F};
    return tmp.b;
  }
  else {
    return cuda::std::numeric_limits<T>::max();
  }
}

template <typename T> constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T minVal() {
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __half>) {
    constexpr HalfBits tmp{0x0400};
    return tmp.h;
  }
  if constexpr (std::is_same_v<convert_matx_type_t<T>, __nv_bfloat16>) {
    constexpr HalfBits tmp{0x0080};
    return tmp.b;
  }
  else {
    return cuda::std::numeric_limits<T>::lowest();
  }
}


/**
 * Operator for performing a sum reduction
 *
 * Performs a reduction of two values of type T by summing the
 * values
 */
template <typename T> class reduceOpSum {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) const { return v1 + v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return T(0); }
};

/**
 * Operator for performing a product reduction
 *
 * Performs a reduction of two values of type T by multiplying the
 * values
 */
template <typename T> class reduceOpProd {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) const { return v1 * v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return T(1); }
};

/**
 * Operator for performing a max reduction
 *
 * Performs a reduction of two values of type T by taking the max
 * of the two values. Type must have operator> defined to perform
 * max
 */
template <typename T> class reduceOpMax {
public:
  using matx_reduce = bool;
  using matx_reduce_index = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) { return v1 > v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return minVal<T>(); }
};

/**
 * Operator for performing an any reduction
 *
 * Performs a reduction of two values of type T by returning 1 if either
 * of the values are non-zero.
 */
template <typename T> class reduceOpAny {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2)
  {
    return (v1 != 0) || (v2 != 0);
  }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return (T)(0); }
};

/**
 * Operator for performing an all reduction
 *
 * Performs a reduction of two values of type T by returning 1 if all
 * of the values are non-zero.
 */
template <typename T> class reduceOpAll {
public:
  using matx_reduce = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2)
  {
    return (v1 != 0) && (v2 != 0);
  }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return (T)(1); }
};

/**
 * Operator for performing a min reduction
 *
 * Performs a reduction of two values of type T by taking the min
 * of the two values. Type must have operator< defined to perform
 * min
 */
template <typename T> class reduceOpMin {
public:
  using matx_reduce = bool;
  using matx_reduce_index = bool;
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T operator()(const T &v1, const T &v2) { return v1 < v2 ? v1 : v2; }
  __MATX_HOST__ __MATX_DEVICE__ __MATX_INLINE__ T Init() { return maxVal<T>(); }
};

#endif

} // namespace detail


/**
 * Perform a reduction
 *
 * Performs a reduction from tensor "in" into tensor "dest" using reduction
 * operation ReduceOp. The output tensor dictates which elements the reduction
 * is performed over. In general, the reductions are performed over the
 * innermost dimensions, where the number of dimensions is the difference
 * between the input and output tensor ranks. For example, for a 0D (scalar)
 * output tensor, the reduction is performed over the entire tensor. For
 * anything higher, the reduction is performed across the number of ranks below
 * the input tensor that the output tensor is. For example, if the input tensor
 * is a 4D tensor and the output is a 1D tensor, the reduction is performed
 * across the innermost dimension of the input. If the output is a 2D tensor,
 * the reduction is performed across the two innermost dimensions of the input,
 * and so on.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam ReduceOp
 *   Reduction operator to apply
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param op
 *   Reduction operator
 * @param stream
 *   CUDA stream
 * @param init
 *   if true dest will be initialized with ReduceOp::Init()
 *   otherwise the values in the destination will be included
 *   in the reduction.
 */
template <typename OutType, typename InType, typename ReduceOp>
void __MATX_INLINE__ reduce(OutType dest, const InType &in, ReduceOp op,
                   cudaStream_t stream = 0, [[maybe_unused]] bool init = true)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)
  // Use CUB implementation if we have a tensor on the RHS and it's not blocked from using CUB
  cub_reduce<OutType, InType, ReduceOp>(dest, in, op.Init(), stream);
}

/**
 * Calculate the mean of values in a tensor
 *
 * Performs a sum reduction from tensor "in" into tensor "dest" , followed by
 * a division by the number of elements in the reduction. Similar to the reduce
 * function, the type of reduction is dependent on the rank of the output
 * tensor. A single value denotes a reduction over the entire input, a 1D tensor
 * denotes a reduction over each row independently, etc.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ mean_impl(OutType dest, const InType &in,
                 cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("mean_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  static_assert(OutType::Rank() < InType::Rank(), "reduction dimensions must be <= Rank of input");

  using inner_type = typename inner_op_type_t<typename InType::value_type>::type;
  inner_type scale = 1;

  cudaStream_t stream = exec.getStream();


  sum_impl(dest, in, stream);

  // The reduction is performed over the difference in ranks between input and
  // output. This loop computes the number of elements it was performed over.
  for (int i = 1; i <= InType::Rank() - OutType::Rank(); i++) {
    scale *= static_cast<inner_type>(in.Size(InType::Rank() - i));
  }

  (dest = dest * static_cast<inner_type>(1) / scale).run(stream);
#endif
}

/**
 * Calculate the mean of values in a tensor
 *
 * Performs a sum reduction from tensor "in" into tensor "dest" , followed by
 * a division by the number of elements in the reduction. Similar to the reduce
 * function, the type of reduction is dependent on the rank of the output
 * tensor. A single value denotes a reduction over the entire input, a 1D tensor
 * denotes a reduction over each row independently, etc.
 *
 * @tparam T
 *   Output data type
 * @tparam RANK
 *   Rank of output tensor
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ mean_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("mean_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  static_assert(OutType::Rank() < InType::Rank(), "reduction dimensions must be <= Rank of input");
  using inner_type = typename inner_op_type_t<typename InType::value_type>::type;

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      auto ts = TotalSize(in);
      *lout = std::accumulate(lin, lin + ts, static_cast<typename InType::value_type>(0)) / static_cast<inner_type>(ts);
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        *(lout + b) = std::accumulate(lin + lbegin[b], lin + lend[b], static_cast<typename InType::value_type>(0)) / static_cast<inner_type>(lin.Size(1));
      }
    }
  };

  ReduceInput(ft, dest, in);
}



/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. By default the sum is performed over all dimensions. Note
 * that traditional definitions of softmax are simply exp(x)/sum(exp(x)), but this is not
 * how most libraries are implemented. Instead, x is biased by a correction factor of max(x).
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam RANK
 *   Rank of output tensor
 *
 * @param dest
 *   Destination for softmax output
 * @param in
 *   Input data to compute the softmax
 * @param stream
 *   CUDA stream
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ softmax_impl(OutType dest, const InType &in,
                 cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("softmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto tmp_sum = make_tensor<typename InType::value_type>({}, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmp_max = make_tensor<typename InType::value_type>({}, MATX_ASYNC_DEVICE_MEMORY, stream);
  max_impl(tmp_max, in, cudaExecutor{stream});
  sum_impl(tmp_sum, exp(in - tmp_max), stream);
  (dest = exp(in - tmp_max) / tmp_sum).run(stream);
#endif
}

/**
 * Calculate the softmax of values in a tensor treated as a flat vector
 *
 * softmax computes the exponential of each value divided by the sum of the exponentials
 * of items in the reduced set. The axes in which to perform the softmax over determine
 * which axes the sum will be computed over, but the input tensor rank and sizes match
 * between input and output. Note that traditional definitions of softmax are simply
 * exp(x)/sum(exp(x)), but this is not how most libraries are implemented. Instead, x
 * is biased by a correction factor of max(x).
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam PermDims
 *   Permutation array
 *
 * @param dest
 *   Destination for softmax output
 * @param in
 *   Input data to compute the softmax
 * @param dims
 *   C-style array containing the dimensions to sum over
 * @param stream
 *   CUDA stream
 */
template <typename OutType, typename InType, typename PermDims>
void __MATX_INLINE__ softmax_impl(OutType dest, const InType &in, PermDims dims, cudaStream_t stream = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("softmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  static_assert(dims.size() < InType::Rank(), "softmax dimensions must be <= Rank of input");
  static_assert(OutType::Rank() == InType::Rank(), "softmax output rank must equal input rank");

  auto perm = detail::getPermuteDims<InType::Rank()>(dims);

  // Create the shape of the summed tensor based on the permutation params
  cuda::std::array<index_t, InType::Rank() - (int)dims.size()> red_shape{};
  #pragma unroll
  for (int r = 0; r < in.Rank() - (int)dims.size(); r++) {
    red_shape[r] = in.Size(perm[r]);
  }

  // With the sum calculated, we have a tensor that's not compatible in sizes with the new one for dividing.
  // We need to clone the summed tensor on the appropriate dims for the final divide.
  cuda::std::array<index_t, InType::Rank()> clone_dims;
  int axis_ptr = 0;
  #pragma unroll
  for (int r = 0; r < InType::Rank(); r++) {
    if (axis_ptr >= 0 && dims[axis_ptr] == r) {
      clone_dims[r] = in.Size(r);
      if (static_cast<decltype(dims.size())>(++axis_ptr) == dims.size()) {
        axis_ptr = -1;
      }
    }
    else {
      clone_dims[r] = matxKeepDim;
    }
  }

  auto tmp_sum = make_tensor<typename InType::value_type>(red_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
  auto tmp_max = make_tensor<typename InType::value_type>(red_shape, MATX_ASYNC_DEVICE_MEMORY, stream);
  max_impl(tmp_max, permute(in, perm), stream);
  sum_impl(tmp_sum, exp(permute(in, perm) - clone<InType::Rank()>(tmp_max, clone_dims)), stream);

  (dest = exp(in - clone<InType::Rank()>(tmp_max, clone_dims)) / clone<InType::Rank()>(tmp_sum, clone_dims)).run(stream);
#endif
}

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
template <typename OutType, typename InType>
void __MATX_INLINE__ median_impl(OutType dest,
                   const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  if constexpr ( OutType::Rank() <= 1 && InType::Rank() <=2 ) {
    MATX_NVTX_START("median_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
    using T = typename OutType::value_type;
    constexpr int RANK_IN = InType::Rank();
    static_assert(RANK_IN <= 2 && (RANK_IN == OutType::Rank() + 1));

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    cudaStream_t stream = exec.getStream();

    auto tmp_sort = make_tensor<T>(in.Shape(), MATX_ASYNC_DEVICE_MEMORY, stream);

    // If the rank is 0 we're finding the median of a vector
    if constexpr (RANK_IN == 1) {
      matx::sort_impl(tmp_sort, in, SORT_DIR_ASC, stream);

      // Store median
      if (tmp_sort.Lsize() & 1) {
        auto middlev =
          slice<0>(tmp_sort, {tmp_sort.Lsize() / 2}, {matxDropDim});
        matx::copy(dest, middlev, stream);
      }
      else {
        auto middle1v =
          slice<0>(tmp_sort, {tmp_sort.Lsize() / 2 - 1}, {matxDropDim});
        auto middle2v =
          slice<0>(tmp_sort, {tmp_sort.Lsize() / 2}, {matxDropDim});
        (dest = (middle1v + middle2v) / 2.0f).run(stream);
      }
    }
    else if constexpr (RANK_IN == 2) {
      MATX_ASSERT(dest.Size(0) == in.Size(0), matxInvalidSize);

      matx::sort_impl(tmp_sort, in, SORT_DIR_ASC, stream);

      if (tmp_sort.Lsize() & 1) {
        auto sv = slice<1>(tmp_sort, {0, tmp_sort.Lsize() / 2},
            {matxEnd, matxDropDim});
        (dest = self(sv)).run(stream);
      }
      else {
        auto sv = slice<1>(tmp_sort, {0, tmp_sort.Lsize() / 2 - 1},
            {matxEnd, matxDropDim});
        auto sv2 = slice<1>(tmp_sort, {0, tmp_sort.Lsize() / 2},
            {matxEnd, matxDropDim});
        (dest = (sv + sv2) / 2.0f).run(stream);
      }
    }
  } else {

#if 1  // sort doesn't currently work on non-tensor input
    static_assert(InType::Rank() <= 2 && OutType::Rank() <= 1, "median only supported with output rank <= 1 and input rank <= 2");
#else
    constexpr int out_dims = OutType::Rank();
    constexpr int red_dims = InType::Rank() - OutType::Rank();

    if constexpr ( out_dims > 1) {
      // collapse batch dimensions to a single dimension
      auto oop = lcollapse<out_dims>(dest);
      auto iop = lcollapse<out_dims>(in);

      static_assert(oop.Rank() == 1);
      median_impl(oop, iop, stream);

    } else if constexpr ( red_dims > 1) {

      // collapse reduction dim to a single dim
      auto iop = rcollapse<red_dims>(in);

      static_assert(dest.Rank() <= 1);
      static_assert(iop.Rank() <= 2);
      median_impl(dest, iop, stream);
    } else {
      static_assert(false, "median ranks not supported");
    }
#endif
  }
#endif
}

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
 *   Single thread host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ median_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("median_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      auto insize = TotalSize(in);
      auto tin = new typename InType::value_type[insize];
      std::partial_sort_copy( lin,
                              lin + insize,
                              tin,
                              tin + insize);
      if ((insize % 2) == 0) {
        *lout = (tin[insize / 2] + tin[insize / 2 - 1]) / 2.0f;
      }
      else {
        *lout = tin[insize / 2];
      }

      delete [] tin;
    }
    else {
      auto insize = lin.Size(1);
      auto tin = new typename InType::value_type[insize];
      for (index_t b = 0; b < lin.Size(0); b++) {
        std::partial_sort_copy( lin + lbegin[b],
                                lin + lend[b],
                                tin,
                                tin + insize);

        if ((insize % 2) == 0) {
          *(lout + b) = (tin[insize / 2] + tin[insize / 2 - 1]) / 2.0f;
        }
        else {
          *(lout + b) = tin[insize / 2];
        }
      }

      delete [] tin;
    }
  };

  ReduceInput(ft, dest, in);
}



/**
 * Compute sum of numbers
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ sum_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("sum_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_sum<OutType, InType>(dest, in, stream);
#endif
}

/**
 * Compute sum of numbers
 *
 * Returns a tensor representing the sum of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ sum_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("sum_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = std::accumulate(lin, lin + lin.Size(0), static_cast<typename InType::value_type>(0));
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        auto f = std::accumulate(lin + lbegin[b], lin + lend[b], static_cast<typename InType::value_type>(0));
        *(lout + b) = f;
      }
    }
  };

  ReduceInputNoConvert(ft, dest, in);
}



/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ prod_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("prod_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  // example-begin reduce-1
  // Reduce "in" into "dest" using a product operation as the reduction type
  reduce(dest, in, detail::reduceOpProd<typename OutType::value_type>(), stream, true);
  // example-end reduce-1
#endif
}

/**
 * Compute product of numbers
 *
 * Returns a tensor representing the product of all items in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single thread host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ prod_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("prod_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = std::accumulate(lin,
                              lin + TotalSize(in),
                              static_cast<typename InType::value_type>(1),
                              std::multiplies<typename InType::value_type>());
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        *(lout + b) = std::accumulate(lin + lbegin[b],
                                      lin + lend[b],
                                      static_cast<typename InType::value_type>(1),
                                      std::multiplies<typename InType::value_type>());
      }
    }
  };

  ReduceInput(ft, dest, in);
}



/**
 * Compute max reduction of an operator
 *
 * Returns a tensor representing the max of all numbers in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ max_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("max_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_max<OutType, InType>(dest, in, stream);
#endif
}

/**
 * Compute max reduction of an operator
 *
 * Returns a tensor representing the max of all numbers in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ max_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("max_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = *std::max_element(lin, lin + TotalSize(in));
    }
    else {
      const index_t BATCHES = TotalSize(dest);
      for (index_t b = 0; b < BATCHES; b++) {
        lout[b] = *std::max_element(lin + lbegin[b], lin + lend[b]);
      }
    }
  };

  ReduceInput(ft, dest, in);
}


/**
 * Compute max reduction of an operator and returns value + index
 *
 * Returns a tensor with maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmax_impl(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("argmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  const auto initial_value = thrust::make_tuple(static_cast<matx::index_t>(-1), std::numeric_limits<typename InType::value_type>::lowest());
  using reduce_param_type = typename detail::ReduceParams_t<typename detail::CustomArgMaxCmp, decltype(initial_value)>;
  auto reduce_params = reduce_param_type{detail::CustomArgMaxCmp{}, initial_value};

  cudaStream_t stream = exec.getStream();
  cub_argreduce(dest, idest, in, reduce_params, stream);
#endif
}

/**
 * Compute max reduction of an operator and returns value + index
 *
 * Returns a tensor with maximums and a tensor with indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename TensorIndexType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ argmax_impl(OutType dest, TensorIndexType &idest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("argmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = static_cast<index_t>(cuda::std::max_element(lin, lin + TotalSize(in)) - lin);
    }
    else {
      const index_t BATCHES = TotalSize(dest);
      for (index_t b = 0; b < BATCHES; b++) {
        lout[b] = static_cast<index_t>(cuda::std::max_element(lin + lbegin[b], lin + lend[b]) - lin);
      }
    }
  };

  // This could be more efficient by not running two reductions to find the same values, but
  // for brevity this is faster
  ReduceInput(ft, idest, in);
  max_impl(dest, in, exec);
}



/**
 * Compute min reduction of an operator
 *
 * Returns a tensor representing the min of all numbers in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ min_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("min_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  cudaStream_t stream = exec.getStream();
  cub_min<OutType, InType>(dest, in, stream);
#endif
}

/**
 * Compute min reduction of an operator
 *
 * Returns a tensor representing the min of all numbers in the reduction
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ min_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("min_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = *std::min_element(lin, lin + TotalSize(in));
    }
    else {
      const index_t BATCHES = TotalSize(dest);
      for (index_t b = 0; b < BATCHES; b++) {
        lout[b] = *std::min_element(lin + lbegin[b], lin + lend[b]);
      }
    }
  };

  ReduceInput(ft, dest, in);
}


/**
 * Compute min reduction of an operator and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argmin_impl(OutType dest, TensorIndexType &idest, const InType &in, cudaExecutor exec = 0)
{
  static_assert(OutType::Rank() == TensorIndexType::Rank());
#ifdef __CUDACC__
  MATX_NVTX_START("argmin_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  const auto initial_value = thrust::make_tuple(static_cast<matx::index_t>(-1), std::numeric_limits<typename InType::value_type>::max());
  using reduce_param_type = typename detail::ReduceParams_t<typename detail::CustomArgMinCmp, decltype(initial_value)>;
  auto reduce_params = reduce_param_type{detail::CustomArgMinCmp{}, initial_value};

  cudaStream_t stream = exec.getStream();
  cub_argreduce(dest, idest, in, reduce_params, stream);
#endif
}

/**
 * Compute min reduction of an operator and returns value + index
 *
 * Returns a tensor with minimums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 * @tparam MODE
 *   Host executor threads mode
 *
 * @param dest
 *   Destination view of reduction
 * @param idest
 *   Destination for indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single host executor
 */
template <typename OutType, typename TensorIndexType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ argmin_impl(OutType dest, TensorIndexType &idest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("argmin_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = static_cast<index_t>(cuda::std::min_element(lin, lin + TotalSize(in)) - lin);
    }
    else {
      const index_t BATCHES = TotalSize(dest);
      for (index_t b = 0; b < BATCHES; b++) {
        lout[b] = static_cast<index_t>(cuda::std::min_element(lin + lbegin[b], lin + lend[b]) - lin);
      }
    }
  };

  // This could be more efficient by not running two reductions to find the same values, but
  // for brevity this is faster
  ReduceInput(ft, idest, in);
  min_impl(dest, in, exec);
}

/**
 * Compute min and max reduction of an operator and returns value + index
 *
 * Returns tensors with minimums and indices, and maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 *
 * @param destmin
 *   Destination view of min reduction
 * @param idestmin
 *   Destination for min indices
 * @param destmax
 *   Destination view of max reduction
 * @param idestmax
 *   Destination for max indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename TensorIndexType, typename InType>
void __MATX_INLINE__ argminmax_impl(OutType destmin, TensorIndexType &idestmin, OutType destmax, TensorIndexType &idestmax, const InType &in, cudaExecutor exec = 0)
{
  static_assert(OutType::Rank() == TensorIndexType::Rank());
#ifdef __CUDACC__
  MATX_NVTX_START("argminmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  const auto initial_value = thrust::make_tuple(
    static_cast<matx::index_t>(-1),
    std::numeric_limits<typename InType::value_type>::max(),
    static_cast<matx::index_t>(-1),
    std::numeric_limits<typename InType::value_type>::lowest()
  );
  using reduce_param_type = typename detail::ReduceParams_t<typename detail::CustomArgMinMaxCmp, decltype(initial_value)>;
  auto reduce_params = reduce_param_type{detail::CustomArgMinMaxCmp{}, initial_value};

  cudaStream_t stream = exec.getStream();
  cub_dualargreduce(destmin, idestmin, destmax, idestmax, in, reduce_params, stream);
#endif
}

/**
 * Compute min and max reduction of an operator and returns value + index
 *
 * Returns tensors with minimums and indices, and maximums and indices
 *
 * @tparam OutType
 *   Output data type
 * @tparam TensorIndexType
 *   Output type stpring indices
 * @tparam InType
 *   Input data type
 * @tparam MODE
 *   Host executor threads mode
 *
 * @param destmin
 *   Destination view of min reduction
 * @param idestmin
 *   Destination for min indices
 * @param destmax
 *   Destination view of max reduction
 * @param idestmax
 *   Destination for max indices
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single host executor
 */
template <typename OutType, typename TensorIndexType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ argminmax_impl(OutType destmin, TensorIndexType &idestmin, OutType destmax, TensorIndexType &idestmax, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  static_assert(OutType::Rank() == TensorIndexType::Rank());
  MATX_NVTX_START("argminmax_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  // This could be more efficient by not running argmin and argmax separately but
  // for brevity this is faster
  argmin_impl(destmin, idestmin, in, exec);
  argmax_impl(destmax, idestmax, in, exec);
}


/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ any_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("any_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  cudaStream_t stream = exec.getStream();
  reduce(dest, in, detail::reduceOpAny<typename OutType::value_type>(), stream, true);
#endif
}

/**
 * Find if any value is != 0
 *
 * Returns a boolean value indicating whether any value in the set of inputs are
 * non-zero. The same aggregation rules apply for input vs output tensor size
 * and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ any_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("any_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = std::any_of(lin, lin + TotalSize(in), [](typename InType::value_type vin) {
          return vin != static_cast<typename InType::value_type>(0);
        });
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        lout[b] = std::any_of(lin + lbegin[b], lin + lend[b], [](typename InType::value_type vin) {
          return vin != static_cast<typename InType::value_type>(0);
        });
      }
    }
  };

  ReduceInput(ft, dest, in);
}



/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ all_impl(OutType dest, const InType &in, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("all_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  cudaStream_t stream = exec.getStream();
  reduce(dest, in, detail::reduceOpAll<typename OutType::value_type>(), stream, true);
#endif
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType, ThreadsMode MODE>
void __MATX_INLINE__ all_impl(OutType dest, const InType &in, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("all_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    if constexpr (OutType::Rank() == 0) {
      *lout = std::all_of(lin, lin + TotalSize(in), [](typename InType::value_type vin) {
          return vin != static_cast<typename InType::value_type>(0);
        });
    }
    else {
      for (index_t b = 0; b < lin.Size(0); b++) {
        lout[b] = std::all_of(lin + lbegin[b], lin + lend[b], [](typename InType::value_type vin) {
          return vin != static_cast<typename InType::value_type>(0);
        });
      }
    }
  };

  ReduceInput(ft, dest, in);
}


/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of output boolean
 * @param in1
 *   First input data to compare
 * @param in2
 *   Second input data to compare
 * @param rtol
 *   Relative tolerance for comparison
 * @param atol
 *   Absolute tolerance for comparison
 * @param exec
 *   CUDA executor or stream ID
 */
template <typename OutType, typename InType1, typename InType2>
void __MATX_INLINE__ allclose(OutType dest, const InType1 &in1, const InType2 &in2, double rtol, double atol, cudaExecutor exec = 0)
{
#ifdef __CUDACC__
  MATX_NVTX_START("allclose(" + get_type_str(in1) + ", " + get_type_str(in2) + ")", matx::MATX_NVTX_LOG_API)
  static_assert(OutType::Rank() == 0, "allclose output must be rank 0");

  cudaStream_t stream = exec.getStream();
  reduce(dest, isclose(in1, in2, rtol, atol), detail::reduceOpAll<int>(), stream, true);
#endif
}

/**
 * Find if all values are != 0
 *
 * Returns a boolean value indicating whether all values in the set of inputs
 * are non-zero. The same aggregation rules apply for input vs output tensor
 * size and what type of reduction is done.
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of output boolean
 * @param in1
 *   First input data to compare
 * @param in2
 *   Second input data to compare
 * @param rtol
 *   Relative tolerance for comparison
 * @param atol
 *   Absolute tolerance for comparison
 * @param exec
 *   Single threaded host executor
 */
template <typename OutType, typename InType1, typename InType2, ThreadsMode MODE>
void __MATX_INLINE__ allclose(OutType dest, const InType1 &in1, const InType2 &in2, double rtol, double atol, [[maybe_unused]] const HostExecutor<MODE> &exec)
{
  MATX_NVTX_START("allclose(" + get_type_str(in1) + ", " + get_type_str(in2) + ")", matx::MATX_NVTX_LOG_API)
  static_assert(OutType::Rank() == 0, "allclose output must be rank 0");

  auto isc = isclose(in1, in2, rtol, atol);

  auto ft = [&](auto &&lin, auto &&lout, [[maybe_unused]] auto &&lbegin, [[maybe_unused]] auto &&lend) {
    *lout = std::all_of(lin, lin + TotalSize(in1), [](int vin) {
        return vin != 0;
      });
  };


  ReduceInput(ft, dest, isc);
}


/**
 * Compute a variance reduction
 *
 * Computes the variance of the input according to the output tensor rank and
 * size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 * @tparam Executor
 *   Executor type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 * @param ddof
 *   Delta Degrees Of Freedom used in the divisor of the result as N - ddof. Defaults
 *   to 1 to give an unbiased estimate
 */
#ifndef DOXYGEN_ONLY
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
#else
template <typename OutType, typename InType, typename Executor>
#endif
void __MATX_INLINE__ var_impl(OutType dest, const InType &in, Executor &&exec, int ddof = 1)
{
  MATX_NVTX_START("var_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  matxMemorySpace_t space;
  using inner_type = typename inner_op_type_t<typename InType::value_type>::type;

  if constexpr (is_cuda_executor_v<Executor>) {
    space = MATX_ASYNC_DEVICE_MEMORY;
  }
  else {
    space = MATX_HOST_MALLOC_MEMORY;
  }

  auto mean_tns = make_tensor<typename InType::value_type>(dest.Descriptor(), space);
  mean_impl(mean_tns, in, exec);

  // need to clone along right most dims
  cuda::std::array<index_t, InType::Rank()> cdims;
  for(int i = 0; i < OutType::Rank(); i++) {
    cdims[i] = matxKeepDim;
  }
  for(int i = OutType::Rank(); i < InType::Rank(); i++) {
    cdims[i] = in.Size(i);
  }

  auto mean_op = mean_tns.template Clone<InType::Rank()>(cdims);

  sum_impl(dest, pow(abs(in - mean_op), static_cast<inner_type>(2)), exec);

  // The length of what we are taking the variance over is equal to the product
  // of the outer dimensions covering the different in input/output ranks
  index_t N = in.Size(in.Rank() - 1);
  for (int i = 2; i <= in.Rank() - OutType::Rank(); i++) {
    N *= in.Size(in.Rank() - i);
  }

  // Sample variance for an unbiased estimate uses ddof == 1, whereas 0 gives an ML estimate
  (dest = dest / static_cast<inner_type>(N - ddof)).run(exec);
}


/**
 * Compute a standard deviation reduction
 *
 * Computes the standard deviation of the input according to the output tensor
 * rank and size
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 * @param ddof
 *   Delta Degrees Of Freedom used in the divisor of the result as N - ddof. Defaults
 *   to 1 to give an unbiased estimate
 */
#ifndef DOXYGEN_ONLY
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
#else
template <typename OutType, typename InType, typename Executor>
#endif
void __MATX_INLINE__ stdd_impl(OutType dest, InType &&in, Executor &&exec, int ddof = 1)
{
  MATX_NVTX_START("stdd_impl(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)
  var_impl(dest, in, exec, ddof);
  (dest = sqrt(dest)).run(exec);
}


/**
 * Computes the trace of a tensor
 *
 * Computes the trace of a square matrix by summing the diagonal
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param exec
 *   Executor type
 */
#ifndef DOXYGEN_ONLY
template <typename OutType, typename InType, typename Executor, std::enable_if_t<is_executor_t<Executor>(), bool> = true>
#else
template <typename OutType, typename InType, typename Executor>
#endif
void __MATX_INLINE__ trace_impl(OutType dest, const InType &in, Executor &&exec)
{
  MATX_NVTX_START("trace(" + get_type_str(in) + ")", matx::MATX_NVTX_LOG_API)

  auto d = diag(in);
  sum_impl(dest, d, exec);
}

/**
 * Computes the trace of a tensor
 *
 * Computes the trace of a square matrix by summing the diagonal
 *
 * @tparam OutType
 *   Output data type
 * @tparam InType
 *   Input data type
 *
 * @param dest
 *   Destination view of reduction
 * @param in
 *   Input data to reduce
 * @param stream
 *   CUDA stream ID
 */
template <typename OutType, typename InType>
void __MATX_INLINE__ trace_impl(OutType dest, const InType &in, int stream = 0)
{
  return trace(dest, in, cudaExecutor{stream});
}

} // end namespace matx
