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

#include <algorithm>
#include <cassert>
#include <initializer_list>

#include "matx_exec_kernel.h"
#include "matx_scalar_ops.h"
#include "matx_tensor.h"
#include "matx_transpose.cuh"
#include "matx_type_utils.h"

namespace matx {

template <typename T> class BaseOp {
public:
  using matxop = bool;

  // Launch work in the stream
  void run(cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), stream);
  }

  // Record an event after the work
  void run(cudaEvent_t ev, cudaStream_t stream = 0) noexcept
  {
    exec(*static_cast<T *>(this), stream);
    cudaEventRecord(ev, stream);    
  }  
};

/**
 * Make a deep copy of a view into another view
 *
 * Copies the data from a view into another view. Views should normally be
 * backed by different data objects, but it's not necessary if there is no
 * overlap between the soure and destination. If the source in destination
 * overlap in any way, it is a race condition and the result of the operation
 * is undefined.
 *
 * Both tensor views must be the same rank and size in every dimension
 *
 * @param out
 *   Tensor to copy into
 * @param in
 *   Tensor to copy from
 * @param stream
 *   CUDA stream to operate in
 */
template <class T, int Rank>
inline void copy(tensor_t<T, Rank> out, const tensor_t<T, Rank> &in,
                 const cudaStream_t stream)
{
  constexpr int rank = Rank;

  for (int i = 0; i < rank; i++) {
    MATX_ASSERT(out.Size(i) == in.Size(i), matxInvalidSize);
  }

  (out = self(in)).run(stream);
};

/**
 * Transpose the outer dimensions of a tensor view out-of-place
 *
 * Transposes the two fastest-changing dimensions of a tensor. Any higher
 * dimension is untouched. This has the same effect as permute with {1,0} as the
 * last two dims, but it is much faster for tensors that are already contiguous.
 * For tensors that are not a contiguous view, this function is not allowed.
 *
 * Both tensor views must be the same rank, and the dimensions that moved must
 * match their original size
 *
 * @param out
 *   Tensor to copy into
 * @param in
 *   Tensor to copy from
 * @param stream
 *   CUDA stream to operate in
 *
 */
template <class T, int RANK>
inline void transpose(tensor_t<T, RANK> &out,
                      const tensor_t<T, RANK> &in,
                      const cudaStream_t stream)
{

  if constexpr (RANK <= 1) {
    return;
  }

  if (!in.IsLinear()) {
    MATX_THROW(matxInvalidSize, "Must have a linear tensor view for transpose");
  }

  size_t shm = sizeof(T) * TILE_DIM * (TILE_DIM + 1);
  if constexpr (RANK == 2) {
    dim3 block(TILE_DIM, TILE_DIM);
    dim3 grid(static_cast<int>((in.Size(RANK - 1) + TILE_DIM - 1) / TILE_DIM),
              static_cast<int>((in.Size(RANK - 2) + TILE_DIM - 1) / TILE_DIM));
    transpose_kernel_oop<<<grid, block, shm, stream>>>(out, in);
  }
  else if constexpr (RANK >= 3) {
    index_t batch_dims =
        in.TotalSize() - (in.Size(RANK - 1) * in.Size(RANK - 2));

    dim3 block(batch_dims, TILE_DIM, TILE_DIM);
    dim3 grid(static_cast<int>((in.Size(RANK - 1) + TILE_DIM - 1) / TILE_DIM),
              static_cast<int>((in.Size(RANK - 2) + TILE_DIM - 1) / TILE_DIM));
    transpose_kernel_oop<<<grid, block, shm, stream>>>(out, in);
  }
};

/**
 * Permute a tensor view out-of-place
 *
 * Rearranges the dimensions of a tensor view without touching the data. This is
 * accomplished by changing the strides between dimensions to reflect the new
 * transposed order. This function can result in very in efficient memory
 * accesses, so it's recommended only to use in places performance is not
 * critical.
 *
 * Both tensor views must be the same rank, and the dimensions that moved must
 * match their original size
 *
 * @param out
 *   Tensor to copy into
 * @param in
 *   Tensor to copy from
 * @param dims
 *   Order of transposed tensor dimensions
 * @param stream
 *   CUDA stream to operate in
 *
 */
template <class T, int Rank>
inline void permute(tensor_t<T, Rank> &out, const tensor_t<T, Rank> &in,
                    const std::initializer_list<uint32_t> &dims,
                    const cudaStream_t stream)
{
  // This is very naive, we should make optimized versions for various swizzles
  auto in_t = in.Permute(dims.begin());

  copy(out, in_t, stream);
};

/**
 * Chain multiple operator statements
 *
 * Takes a variable list of operator statements to execute concurrently.
 * Chaining may improve performance over executing each operation separately.
 */
#ifdef DOXYGEN_ONLY
template <typename T1>
#else
template <typename... T1>
#endif
class CHAIN : public BaseOp<CHAIN<T1...>> {
public:
  using scalar_type = void;

  // Rank=0 accessor
  inline __device__ auto operator()() {}
  // Rank=1 accessor
  inline __device__ auto operator()(index_t i) {}
  // Rank=2 accessor
  inline __device__ auto operator()(index_t i, index_t j) {}
  // Rank=3 accessor
  inline __device__ auto operator()(index_t i, index_t j, index_t k) {}
  // Rank=4 accessor
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
  }

  // Rank of chain. Purely for type annotations and has no meaning
  static inline constexpr __host__ __device__ int32_t Rank() { return -2; }
  // Size of dimension. Purely for type annotations and has no meaning
  index_t inline __host__ __device__ Size(int) const { return 0; }
};

/**
 * Chain multiple operator statements
 *
 * Takes a variable list of operator statements to execute concurrently.
 * Chaining may improve performance over executing each operation separately.
 */
template <typename T1, typename... ARGS>
class CHAIN<T1, ARGS...> : public BaseOp<CHAIN<T1, ARGS...>> {
private:
  T1 op_;
  CHAIN<ARGS...> args_;

public:
  // Scalar type of operation
  using scalar_type = void;

  inline CHAIN(T1 op, ARGS... args) : op_(op), args_(args...)
  {
    static_assert((... && !is_tensor_view<decltype(args)>()),
                  "Only operator emmitters are allowed in CHAIN. Tensor views "
                  "are not allowed");
  }

  inline __device__ auto operator()()
  {
    get_value(op_);
    args_.operator()();
  }

  inline __device__ auto operator()(index_t i)
  {
    get_value(op_, i);
    args_.operator()(i);
  }

  inline __device__ auto operator()(index_t i, index_t j)
  {
    get_value(op_, i, j);
    args_.operator()(i, j);
  }

  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    get_value(op_, i, j, k);
    args_.operator()(i, j, k);
  }

  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    get_value(op_, i, j, k, l);
    args_.operator()(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank() noexcept
  {
    return std::max({T1::Rank(), ARGS::Rank()...});
  }

  index_t inline __host__ __device__ Size(int dim) const noexcept
  {
    index_t size1 = get_expanded_size<Rank()>(op_, dim);
    index_t size2 = get_expanded_size<Rank()>(args_, dim);
    return MAX(size1, size2);
  }
};

template <typename... Args> CHAIN(Args...)->CHAIN<Args...>;

/**
 * Conditionally execute an operator
 *
 * Compares two operators or views and conditionally executes the second
 * statement if the first is true. Values from an operator are executed
 * individually, and the only requirement for the conditional is the comparison
 * operator must be defined for the particular type. For example, operator< on
 * two integers is okay, but the same operator on two complex numbers will give
 * a compiler error.
 *
 */
template <typename T1, typename T2> class IF : public BaseOp<IF<T1, T2>> {
private:
  T1 cond_;
  T2 op_;

public:
  using scalar_type = void;
  inline IF(T1 cond, T2 op) : cond_(cond), op_(op)
  {
    static_assert((!is_tensor_view_t<T2>()),
                  "Only operator emmitters are allowed in IF. Tensor views are "
                  "not allowed");
    constexpr index_t rank1 = get_rank<T1>();
    constexpr index_t rank2 = get_rank<T2>();
    static_assert(rank1 == -1 || rank1 == Rank());
    static_assert(rank2 == -1 || rank2 == Rank());

    if constexpr (Rank() > 0) {
      for (int i = 0; i < Rank(); i++) {
        index_t size1 = get_expanded_size<Rank()>(cond_, i);
        index_t size2 = get_expanded_size<Rank()>(op_, i);
        MATX_ASSERT(size1 == 0 || size1 == Size(i), matxInvalidSize);
        MATX_ASSERT(size2 == 0 || size2 == Size(i), matxInvalidSize);
      }
    }
  }

  __device__ inline auto operator()()
  {
    if (get_value(cond_)) {
      get_value(op_);
    }
  }
  __device__ inline auto operator()(index_t i)
  {
    if (get_value(cond_, i))
      get_value(op_, i);
  }
  __device__ inline auto operator()(index_t i, index_t j)
  {
    if (get_value(cond_, i, j))
      get_value(op_, i, j);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k)
  {
    if (get_value(cond_, i, j, k))
      get_value(op_, i, j, k);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if (get_value(cond_, i, j, k, l))
      get_value(op_, i, j, k, l);
  }
  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return MAX(get_rank<T1>(), get_rank<T2>());
  }
  index_t inline __host__ __device__ Size(int dim) const
  {
    index_t size1 = get_expanded_size<Rank()>(op_, dim);
    index_t size2 = get_expanded_size<Rank()>(cond_, dim);
    return MAX(size1, size2);
  }
};

/**
 * Conditionally execute an operator, otherwise execute a different operator
 *
 * Compares two operators or views and conditionally executes the second
 * statement if the first is true, otherwise executes the third statement.
 * Values from an operator are executed individually, and the only requirement
 * for the conditional is the comparison operator must be defined for the
 * particular type. For example, operator< on two integers is okay, but the same
 * operator on two complex numbers will give a compiler error.
 *
 */
template <typename C1, typename T1, typename T2>
class IFELSE : public BaseOp<IFELSE<C1, T1, T2>> {
private:
  C1 cond_;
  T1 op1_;
  T2 op2_;

public:
  using scalar_type = void;
  inline IFELSE(C1 cond, T1 op1, T2 op2) : cond_(cond), op1_(op1), op2_(op2)
  {
    static_assert((!is_tensor_view_t<T1>() && !is_tensor_view_t<T2>()),
                  "Only operator emmitters are allowed in IFELSE. Tensor views "
                  "are not allowed");
    constexpr int32_t rank0 = get_rank<C1>();
    constexpr int32_t rank1 = get_rank<T1>();
    constexpr int32_t rank2 = get_rank<T2>();
    static_assert(rank0 == -1 || rank0 == Rank());
    static_assert(rank1 == -1 || rank1 == Rank());
    static_assert(rank2 == -1 || rank2 == Rank());

    if constexpr (Rank() > 0) {
      for (int i = 0; i < Rank(); i++) {
        index_t size0 = get_expanded_size<Rank()>(cond_, i);
        index_t size1 = get_expanded_size<Rank()>(op1, i);
        index_t size2 = get_expanded_size<Rank()>(op2, i);
        MATX_ASSERT(size0 == 0 || size0 == Size(i), matxInvalidSize);
        MATX_ASSERT(size1 == 0 || size1 == Size(i), matxInvalidSize);
        MATX_ASSERT(size2 == 0 || size2 == Size(i), matxInvalidSize);
      }
    }
  }

  __device__ inline auto operator()()
  {
    if (get_value(cond_))
      get_value(op1_);
    else
      get_value(op2_);
  }
  __device__ inline auto operator()(index_t i)
  {
    if (get_value(cond_, i))
      get_value(op1_, i);
    else
      get_value(op2_, i);
  }
  __device__ inline auto operator()(index_t i, index_t j)
  {
    if (get_value(cond_, i, j))
      get_value(op1_, i, j);
    else
      get_value(op2_, i, j);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k)
  {
    if (get_value(cond_, i, j, k))
      get_value(op1_, i, j, k);
    else
      get_value(op2_, i, j, k);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if (get_value(cond_, i, j, k, l))
      get_value(op1_, i, j, k, l);
    else
      get_value(op2_, i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return MAX(get_rank<C1>(), get_rank<T1>(), get_rank<T2>());
  }

  index_t inline __host__ __device__ Size(int dim) const
  {
    index_t size1 = get_expanded_size<Rank()>(op1_, dim);
    index_t size2 = get_expanded_size<Rank()>(op2_, dim);
    index_t size3 = get_expanded_size<Rank()>(cond_, dim);
    return MAX(size1, size2, size3);
  }
};

/**
 * Reverse the indexing of a View or operator on a single dimension
 *
 * Allows a view or operator to be indexed in reverse order. After applying the
 * operator, index 0 is the last element in the selected dimension, index 1 is
 * second to last, etc.
 *
 */
template <typename T1, int DIM> class ReverseOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline ReverseOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    if constexpr (DIM == 0)
      i = Size(0) - i - 1;
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    if constexpr (DIM == 0)
      i = Size(0) - i - 1;
    if constexpr (DIM == 1)
      j = Size(1) - j - 1;
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    if constexpr (DIM == 0)
      i = Size(0) - i - 1;
    if constexpr (DIM == 1)
      j = Size(1) - j - 1;
    if constexpr (DIM == 2)
      k = Size(2) - k - 1;
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if constexpr (DIM == 0)
      i = Size(0) - i - 1;
    if constexpr (DIM == 1)
      j = Size(1) - j - 1;
    if constexpr (DIM == 2)
      k = Size(2) - k - 1;
    if constexpr (DIM == 3)
      l = Size(3) - l - 1;
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Helper function to reverse the indexing of the last dimension of a tensor
 *
 * Requires a tensor of at least rank 1
 */
template <typename T1> auto reverseX(T1 t)
{
  MATX_ASSERT(T1::Rank() > 0, matxInvalidDim);
  return ReverseOp<T1, T1::Rank() - 1>(t);
};

/**
 * Helper function to reverse the indexing of the second-to-last
 * dimension of a tensor
 *
 * Requires a tensor of at least rank 2
 */
template <typename T1> auto reverseY(T1 t)
{
  MATX_ASSERT(T1::Rank() > 1, matxInvalidDim);
  return ReverseOp<T1, T1::Rank() - 2>(t);
};

/**
 * Helper function to reverse the indexing of the third-to-last
 * dimension of a tensor
 *
 * Requires a tensor of at least rank 3
 */
template <typename T1> auto reverseZ(T1 t)
{
  MATX_ASSERT(T1::Rank() > 2, matxInvalidDim);
  return ReverseOp<T1, T1::Rank() - 3>(t);
};

/**
 * Helper function to reverse the indexing of the first dimension of a tensor
 *
 * Requires a tensor of rank 4
 */
template <typename T1> auto reverseW(T1 t)
{
  MATX_ASSERT(T1::Rank() > 3, matxInvalidDim);
  return ReverseOp<T1, T1::Rank() - 4>(t);
};

/**
 * Flip the vertical axis of a tensor.
 */
template <typename T1> auto flipud(T1 t)
{
  if constexpr (T1::Rank() == 1) {
    return ReverseOp<T1, T1::Rank() - 1>(t);
  }

  return ReverseOp<T1, T1::Rank() - 2>(t);
};

/**
 * Flip the horizontal axis of a tensor.
 */
template <typename T1> auto fliplr(T1 t)
{
  if constexpr (T1::Rank() == 1) {
    return ReverseOp<T1, T1::Rank() - 1>(t);
  }

  return ReverseOp<T1, T1::Rank() - 1>(t);
};

/**
 * Performs a Hermitian transpose operator on a tensor
 *
 * This operation allows a user to perform a Hermitian operator using a
 * single operator instead of Permute followed by a conj() operator.
 */
template <typename T1, int DIM> class HermitianTransOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline HermitianTransOp(T1 op) : op_(op) {}

  inline __device__ auto operator()() { return conj(op_()); }
  inline __device__ auto operator()(index_t i) { return conj(op_(i)); }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    return conj(op_(j, i));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return conj(op_(k, j, i));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    return conj(op_(l, k, j, i));
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(Rank() - dim - 1);
  }
};

/**
 * Helper function for creating a hermitian transpose from an operator/View
 */
template <typename T1> auto hermitianT(T1 t)
{
  return HermitianTransOp<T1, T1::Rank()>(t);
}

/**
 * Returns elements on the diagonal
 *
 * Returns elements on the diagonal of a 2D tensor. Any dimensions above 2 will
 * be considered batch dimension and the size of those match the size of the
 * input operator. The last dimension is always sized to be the minimum of the
 * last two dimension of the input operator
 */
template <typename T1, int RANK> class DiagOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline DiagOp(T1 op) : op_(op) {}

  template <int M = RANK, std::enable_if_t<M == 1, bool> = true>
  inline __device__ auto operator()()
  {
    return op_(0);
  }

  template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
  inline __device__ auto operator()(index_t i)
  {
    return op_(i, i);
  }
  template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
  inline __device__ auto operator()(index_t i, index_t j)
  {
    return op_(i, j, j);
  }
  template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return op_(i, j, k, k);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return RANK - 1;
  }

  template <int M = RANK, std::enable_if_t<M == 2, bool> = true>
  inline __host__ __device__ index_t Size([[maybe_unused]] uint32_t dim) const
  {
    return std::min(op_.Size((uint32_t)(RANK - 1)),
                    op_.Size((uint32_t)(RANK - 2)));
  }

  template <int M = RANK, std::enable_if_t<M == 3, bool> = true>
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return dim == 0 ? op_.Size(dim)
                    : std::min(op_.Size((uint32_t)(RANK - 1)),
                               op_.Size((uint32_t)(RANK - 2)));
  }

  template <int M = RANK, std::enable_if_t<M == 4, bool> = true>
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return (dim <= 1) ? op_.Size(dim)
                      : std::min(op_.Size((uint32_t)(RANK - 1)),
                                 op_.Size((uint32_t)(RANK - 2)));
  }
};

/**
 * Get the elements on the diagonal
 *
 * @param t
 *   Input operator
 */
template <typename T1> auto diag(T1 t) { return DiagOp<T1, T1::Rank()>(t); }

/**
 * Kronecker tensor product
 *
 * Performs a Kronecker tensor product on two matrices. For input tensors A
 * (MxN) and B (PxQ), A is repeated and multiplied by each element in B to
 * create a new matrix of size M*P x N*Q.
 */
template <typename T1, typename T2, int DIM> class KronOp {
private:
  T1 op1_;
  T2 op2_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  template <int M = DIM, std::enable_if_t<M >= 2, bool> = true>
  inline KronOp(T1 op1, T2 op2) : op1_(op1), op2_(op2)
  {
  }

  inline __device__ auto operator()(index_t i, index_t j)
  {
    return op2_(i % op2_.Size(0), j % op2_.Size(1)) *
           op1_(i / op2_.Size(0), j / op2_.Size(1));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return op2_(i, j % op2_.Size(1), k % op2_.Size(2)) *
           op1_(i, j / op2_.Size(1), k / op2_.Size(2));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    return op2_(i, j, k % op2_.Size(2), l % op2_.Size(3)) *
           op1_(i, j, k / op2_.Size(2), l / op2_.Size(3));
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op1_.Size(dim) * op2_.Size(dim);
  }
};

/**
 * Kronecker tensor product
 *
 * The Kronecker tensor product is formed by the matrix b by ever element in the
 * matrix a. The resulting matrix has the number of rows and columns equal to
 * the product of the rows and columns of matrices a and b, respectively.
 *
 * @tparam T1
 *   Type of first input
 * @tparam T2
 *   Type of second input
 * @param a
 *   Operator or view for first input
 * @param b
 *   Operator or view for second input
 *
 * @returns
 *   New operator of the kronecker product
 */
template <typename T1, typename T2> auto kron(T1 a, T2 b)
{
  return KronOp<T1, T2, T1::Rank()>(a, b);
};

/**
 * Repeats a matrix the specified amount of times
 *
 * RepMatOp performs a "repmat" operation on a matrix where each dimension
 * specified in "reps" is repeated. Constructors for both scalars and arrays are
 * provided. The scalar version will repeat the matrix by the scalar amount in
 * every dimension, whereas the array version scales independently by each
 * dimension.
 */
template <typename T1, int DIM> class RepMatOp {
private:
  T1 op_;
  index_t reps_[MAX_TENSOR_DIM];

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline RepMatOp(T1 op, index_t reps) : op_(op)
  {
    for (int dim = 0; dim < DIM; dim++) {
      reps_[dim] = reps;
    }
  }

  inline RepMatOp(T1 op, const std::array<index_t, DIM> reps) : op_(op)
  {
    for (int dim = 0; dim < DIM; dim++) {
      reps_[dim] = reps[dim];
    }
  }

  inline RepMatOp(T1 op, const index_t *reps) : op_(op)
  {
    for (int dim = 0; dim < DIM; dim++) {
      reps_[dim] = reps[dim];
    }
  }

  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i) { return op_(i % op_.Size(0)); }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    return op_(i % op_.Size(0), j % op_.Size(1));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return op_(i % op_.Size(0), j % op_.Size(1), k % op_.Size(2));
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    return op_(i % op_.Size(0), j % op_.Size(1), k % op_.Size(2),
               l % op_.Size(3));
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim) * reps_[dim];
  }
};

/**
 * Repeat a matrix an equal number of times in each dimension
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to repeat
 * @param reps
 *   Amount to repeat
 *
 * @returns
 *   New operator with repeated data
 */
template <typename T1> auto repmat(T1 t, index_t reps)
{
  return RepMatOp<T1, T1::Rank()>(t, reps);
};

/**
 * Repeat a matrix a specific number of times in each direction
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to repeat
 * @param reps
 *   Array of times to repeat in each dimension
 *
 * @returns
 *   New operator with repeated data
 */
template <typename T1> auto repmat(T1 t, const index_t (&reps)[])
{
  return RepMatOp<T1, T1::Rank()>(t, reps);
};

/**
 * Repeat a matrix a specific number of times in each direction
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to repeat
 * @param reps
 *   Array of times to repeat in each dimension
 *
 * @returns
 *   New operator with repeated data
 */
template <typename T1> auto repmat(T1 t, const index_t *reps)
{
  return RepMatOp<T1, T1::Rank()>(t, reps);
};

/**
 * Self operator
 *
 * Returns the values of itself. This is useful when converting a type like a
 * tensor view into an operator
 */
template <typename T1, int DIM> class SelfOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline SelfOp(T1 op) : op_(op) {}

  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i) { return op_(i); }
  inline __device__ auto operator()(index_t i, index_t j) { return op_(i, j); }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Returns itself as an operator
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to access
 *
 * @returns
 *   Operator of input
 */
template <typename T1> auto self(T1 t) { return SelfOp<T1, T1::Rank()>(t); };

/**
 * Shifts the indexing of an operator or View by a given amount
 *
 * ShiftOp allows adjusting the relative view of a tensor to start at a
 * new offset. This may be useful to cut off part of a tensor that is
 * meaningless, while maintaining a 0-based offset from the new location. A
 * modulo is applied to the new index to allow wrapping around to the beginning.
 * Negative shifts are allowed, and have the effect of moving back from the end
 * of the tensor.
 */
template <typename T1, int DIM> class ShiftOp {
private:
  T1 op_;
  index_t shift_;
  index_t base_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline ShiftOp(T1 op, index_t shift) : op_(op), shift_(shift)
  {
    if (shift < 0) {
      while (-shift > Size(DIM)) {
        shift += Size(DIM);
      }

      base_ = Size(DIM) + shift;
    }
    else {
      while (shift > Size(DIM)) {
        shift -= Size(DIM);
      }

      base_ = shift;
    }
  }

  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    if constexpr (DIM == 0)
      i = (base_ + i) % Size(0);
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    if constexpr (DIM == 0)
      i = (base_ + i) % Size(0);
    if constexpr (DIM == 1)
      j = (base_ + j) % Size(1);
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    if constexpr (DIM == 0)
      i = (base_ + i) % Size(0);
    if constexpr (DIM == 1)
      j = (base_ + j) % Size(1);
    if constexpr (DIM == 2)
      k = (base_ + k) % Size(2);
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if constexpr (DIM == 0)
      i = (base_ + i) % Size(0);
    if constexpr (DIM == 1)
      j = (base_ + j) % Size(1);
    if constexpr (DIM == 2)
      k = (base_ + k) % Size(2);
    if constexpr (DIM == 3)
      l = (base_ + l) % Size(3);
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Helper function to shift dimension 0 by a given amount
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to shift
 * @param s
 *   Amount to shift forward
 *
 * @returns
 *   New operator with shifted indices
 */
template <typename T1> auto shift0(T1 t, index_t s)
{
  return ShiftOp<T1, 0>(t, s);
};

/**
 * Helper function to shift dimension 1 by a given amount
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to shift
 * @param s
 *   Amount to shift forward
 *
 * @returns
 *   New operator with shifted indices
 */
template <typename T1> auto shift1(T1 t, index_t s)
{
  return ShiftOp<T1, 1>(t, s);
};

/**
 * Helper function to shift  dimension 2 by a given amount
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to shift
 * @param s
 *   Amount to shift forward
 *
 * @returns
 *   New operator with shifted indices
 */
template <typename T1> auto shift2(T1 t, index_t s)
{
  return ShiftOp<T1, 2>(t, s);
};

/**
 * Helper function to shift dimension 3 by a given amount
 *
 * @tparam T1
 *   Type of operator or view
 * @param t
 *   Operator or view to shift
 * @param s
 *   Amount to shift forward
 *
 * @returns
 *   New operator with shifted indices
 */
template <typename T1> auto shift3(T1 t, index_t s)
{
  return ShiftOp<T1, 3>(t, s);
};

template <typename T1> class FFTShift1DOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline FFTShift1DOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    i = (i + (Size(0) + 1) / 2) % Size(0);
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    i = i;
    j = (j + (Size(1) + 1) / 2) % Size(1);
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    i = i;
    j = j;
    k = (k + (Size(2) + 1) / 2) % Size(2);
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    i = i;
    j = j;
    k = k;
    l = (l + (Size(3) + 1) / 2) % Size(3);
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Perform an FFTShift operation on the last dimension of a tensor
 *
 * Shifts the new indexing of the tensor's last dimension to begin at
 * Size()/2. MatX FFTs leave the sample order starting with DC, positive
 * frequencies, then negative frequencies last. FFTShift gives a shifted
 * view of a signal where the new order is negative frequencies, DC, then
 * positive frequencies.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <typename T1> auto fftshift1D(T1 t) { return FFTShift1DOp<T1>(t); }

template <typename T1> class FFTShift2DOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline FFTShift2DOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    i = (i + (Size(0) + 1) / 2) % Size(0);
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    i = (i + (Size(0) + 1) / 2) % Size(0);
    j = (j + (Size(1) + 1) / 2) % Size(1);
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    i = i;
    j = (j + (Size(1) + 1) / 2) % Size(1);
    k = (k + (Size(2) + 1) / 2) % Size(2);
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    i = i;
    j = j;
    k = (k + (Size(2) + 1) / 2) % Size(2);
    l = (l + (Size(3) + 1) / 2) % Size(3);
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Perform an IFFTShift operation on a 2D tensor swapping the first quadrant
 * with the third, and the second with the fourth.
 *
 * Shifts the new indexing of the tensor's last dimension to begin at
 * Size()/2. MatX FFTs leave the sample order starting with DC, positive
 * frequencies, then negative frequencies last. IFFTShift gives a shifted
 * view of a signal where the new order is negative frequencies, DC, then
 * positive frequencies.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <typename T1> auto fftshift2D(T1 t) { return FFTShift2DOp<T1>(t); }

template <typename T1> class IFFTShift1DOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline IFFTShift1DOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    i = (i + Size(0) / 2) % Size(0);
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    i = i;
    j = (j + Size(1) / 2) % Size(1);
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    i = i;
    j = j;
    k = (k + Size(2) / 2) % Size(2);
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    i = i;
    j = j;
    k = k;
    l = (l + Size(3) / 2) % Size(3);
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Perform an IFFTShift operation on the last dimension of a tensor
 *
 * Shifts the new indexing of the tensor's last dimension to begin at
 * Size()/2. MatX FFTs leave the sample order starting with DC, positive
 * frequencies, then negative frequencies last. IFFTShift gives a shifted
 * view of a signal where the new order is negative frequencies, DC, then
 * positive frequencies. Note that ifftshift is the same as fftshift if the
 * length of the signal is even.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <typename T1> auto ifftshift1D(T1 t) { return IFFTShift1DOp<T1>(t); }

template <typename T1> class IFFTShift2DOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline IFFTShift2DOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    i = (i + Size(0) / 2) % Size(0);
    return op_(i);
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    i = (i + Size(0) / 2) % Size(0);
    j = (j + Size(1) / 2) % Size(1);
    return op_(i, j);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    i = i;
    j = (j + Size(1) / 2) % Size(1);
    k = (k + Size(2) / 2) % Size(2);
    return op_(i, j, k);
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    i = i;
    j = j;
    k = (k + Size(2) / 2) % Size(2);
    l = (l + Size(3) / 2) % Size(3);
    return op_(i, j, k, l);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return op_.Size(dim);
  }
};

/**
 * Perform an IFFTShift operation on a 2D tensor swapping the first quadrant
 * with the third, and the second with the fourth.
 *
 * Shifts the new indexing of the tensor's last dimension to begin at
 * Size()/2. MatX FFTs leave the sample order starting with DC, positive
 * frequencies, then negative frequencies last. IFFTShift gives a shifted
 * view of a signal where the new order is negative frequencies, DC, then
 * positive frequencies. Note that ifftshift is the same as fftshift if the
 * length of the signal is even.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <typename T1> auto ifftshift2D(T1 t) { return IFFTShift2DOp<T1>(t); }

// Utility functions for converting scalar ops to tensor ops
// The op here has two inputs.
template <class I1, class Op> class matxUnaryOp {
private:
  I1 in1_;
  Op op_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = typename Op::scalar_type;

  inline matxUnaryOp(I1 in1, Op op) : in1_(in1), op_(op) {}

  __device__ inline auto operator()()
  {
    auto i1 = get_value(in1_);
    return op_(i1);
  }
  __device__ inline auto operator()(index_t i)
  {
    auto i1 = get_value(in1_, i);
    return op_(i1);
  }
  __device__ inline auto operator()(index_t i, index_t j)
  {
    auto i1 = get_value(in1_, i, j);
    return op_(i1);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k)
  {
    auto i1 = get_value(in1_, i, j, k);
    return op_(i1);
  }
  __device__ inline auto operator()(index_t i, uint32_t j, index_t k, index_t l)
  {
    auto i1 = get_value(in1_, i, j, k, l);
    return op_(i1);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<I1>();
  }

  index_t inline __host__ __device__ Size(int dim) const
  {
    return get_size(in1_, dim);
  }
};

template <typename T1, std::enable_if_t<is_complex_v<extract_scalar_type_t<T1>>,
                                        bool> = true>
class ComplexPlanarOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  inline ComplexPlanarOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    if (i >= op_.Size(0)) {
      return op_(i - op_.Size(0)).imag();
    }
    return op_(i).real();
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    if (i >= op_.Size(0)) {
      return op_(i - op_.Size(0), j).imag();
    }
    return op_(i, j).real();
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    if (j >= op_.Size(1)) {
      return op_(i, j - op_.Size(1), k).imag();
    }
    return op_(i, j, k).real();
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if (k >= op_.Size(2)) {
      return op_(i, j, k - op_.Size(2), l).imag();
    }
    return op_(i, j, k, l).real();
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    if constexpr (Rank() <= 1) {
      return op_.Size(dim) * 2;
    }

    return (dim == static_cast<uint32_t>(Rank()) - 2) ? op_.Size(dim) * 2
                                                      : op_.Size(dim);
  }
};

/**
 * Perform a planar layout shift on a complex interleaved input
 *
 * Takes an interleaved complex layout (real1, imag1, real2, ...) and transforms
 * it into planar format (real1, real2, ... realN, imag1, ... imagN). This is
 * mostly used for tensor core CGEMM which expects this layout. The indexing on
 * the new layout will be twice as many elements as complex elements since
 * real/imaginary are separated. If the rank is higher than 2, the conversion is
 * treated as a batched transform and only the inner two dims are converted.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <typename T1, std::enable_if_t<is_complex_v<extract_scalar_type_t<T1>>,
                                        bool> = true>
auto planar(T1 t)
{
  return ComplexPlanarOp<T1>(t);
}

template <typename T1,
          std::enable_if_t<!is_complex_v<extract_scalar_type_t<T1>>, bool> =
              true>
class ComplexInterleavedOp {
private:
  T1 op_;

public:
  using matxop = bool;
  using scalar_type = typename T1::scalar_type;

  using complex_type = std::conditional_t<is_matx_half_v<scalar_type>,
                                          matxHalfComplex<scalar_type>,
                                          cuda::std::complex<scalar_type>>;

  inline ComplexInterleavedOp(T1 op) : op_(op){};
  inline __device__ auto operator()() { return op_(); }
  inline __device__ auto operator()(index_t i)
  {
    return complex_type{op_(i), op_(op_.Size(0) / 2 + i)};
  }
  inline __device__ auto operator()(index_t i, index_t j)
  {
    return complex_type{op_(i, j), op_(op_.Size(0) / 2 + i, j)};
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k)
  {
    return {op_(i, j, k), op_(i, j + op_.Size(1) / 2, k)};
  }
  inline __device__ auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    return {op_(i, j, k, l), op_(i, j, k + op_.Size(2) / 2, l)};
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return get_rank<T1>();
  }
  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    if constexpr (Rank() <= 1) {
      return op_.Size(dim) / 2;
    }

    return (dim == static_cast<uint32_t>(Rank()) - 2) ? op_.Size(dim) / 2
                                                      : op_.Size(dim);
  }
};

/**
 * Perform an interleaved layout shift from a complex planar input
 *
 * Takes aplanar complex layout (real1, real2, ... realN, imag1, ... imagN). and
 * transforms it into interleaved format: (real1, imag1, real2, ...). This is
 * mostly used for tensor core CGEMM which expects planar inputs. The indexing
 * on the new layout will be half as many elements as complex elements since
 * real/imaginary are separated in planar. If the rank is higher than 2, the
 * conversion is treated as a batched transform and only the inner two dims are
 * converted.
 *
 * @tparam T1
 *   Type of View/Op
 * @param t
 *   View/Op to shift
 *
 */
template <
    typename T1,
    std::enable_if_t<!is_complex_v<extract_scalar_type_t<T1>>, bool> = true>
auto interleaved(T1 t)
{
  return ComplexInterleavedOp<T1>(t);
}

template <class I1, class I2, class Op> class matxBinaryOp {
private:
  I1 in1_;
  I2 in2_;
  Op op_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = typename Op::scalar_type;
  inline matxBinaryOp(I1 in1, I2 in2, Op op) : in1_(in1), in2_(in2), op_(op)
  {
    if constexpr (Rank() > 0) {
      for (int32_t i = 0; i < Rank(); i++) {
        index_t size1 = get_expanded_size<Rank()>(in1_, i);
        index_t size2 = get_expanded_size<Rank()>(in2_, i);

        MATX_ASSERT(size1 == 0 || size1 == Size(i), matxInvalidSize);
        MATX_ASSERT(size2 == 0 || size2 == Size(i), matxInvalidSize);
      }
    }
  }

  __device__ inline auto operator()()
  {
    // Rank 0
    auto i1 = get_value(in1_);
    auto i2 = get_value(in2_);
    return op_(i1, i2);
  }
  __device__ inline auto operator()(index_t i)
  {
    // Rank 1
    auto i1 = get_value(in1_, i);
    auto i2 = get_value(in2_, i);
    return op_(i1, i2);
  }
  __device__ inline auto operator()(index_t i, index_t j)
  {
    // Rank 2
    auto i1 = get_value(in1_, i, j);
    auto i2 = get_value(in2_, i, j);
    return op_(i1, i2);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k)
  {
    // Rank 3
    auto i1 = get_value(in1_, i, j, k);
    auto i2 = get_value(in2_, i, j, k);
    return op_(i1, i2);
  }
  __device__ inline auto operator()(index_t i, index_t j, index_t k, index_t l)
  {
    // Rank 4
    auto i1 = get_value(in1_, i, j, k, l);
    auto i2 = get_value(in2_, i, j, k, l);
    return op_(i1, i2);
  }

  static inline constexpr __host__ __device__ int32_t Rank()
  {
    return MAX(get_rank<I1>(), get_rank<I2>());
  }

  index_t inline __host__ __device__ Size(int dim) const
  {
    index_t size1 = get_expanded_size<Rank()>(in1_, dim);
    index_t size2 = get_expanded_size<Rank()>(in2_, dim);
    return MAX(size1, size2);
  }
};

#define DEFINE_UNARY_OP(FUNCTION, TENSOR_OP)                                   \
  template <typename I1,                                                       \
            typename = typename std::enable_if_t<is_matx_op<I1>()>>            \
  [[nodiscard]] inline auto FUNCTION(I1 i1)                                    \
  {                                                                            \
    using I1Type = extract_scalar_type_t<I1>;                                  \
    using Op = TENSOR_OP<I1Type>;                                              \
    return matxUnaryOp<I1, Op>(i1, Op());                                      \
  }

#define DEFINE_BINARY_OP(FUNCTION, TENSOR_OP)                                  \
  template <typename I1, typename I2,                                          \
            typename = typename std::enable_if_t<is_matx_op<I1>() or           \
                                                 is_matx_op<I2>()>>            \
  [[nodiscard]] inline auto FUNCTION(I1 i1, I2 i2)                             \
  {                                                                            \
    using I1Type = extract_scalar_type_t<I1>;                                  \
    using I2Type = extract_scalar_type_t<I2>;                                  \
    using Op = TENSOR_OP<I1Type, I2Type>;                                      \
    return matxBinaryOp<I1, I2, Op>(i1, i2, Op());                             \
  }

#ifdef DOXYGEN_ONLY
/**
 * Compute the square root of each value in a tensor.
 * @param t
 *   Tensor or operator input
 */
Op sqrt(Op t) {}

/**
 * Compute e^x of each value in a tensor.
 * @param t
 *   Tensor or operator input
 */
Op exp(Op t) {}

/**
 * Compute e^(jx) of each value in a tensor where j is sqrt(-1).
 * @param t
 *   Tensor or operator input
 */
Op expj(Op t) {}

/**
 * Compute log base 10 of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op log10(Op t) {}

/**
 * Compute log base 2 of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op log2(Op t) {}

/**
 * Compute log base e (natural log) of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op log(Op t) {}

/**
 * Compute log base e (natural log) of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op loge(Op t) {}

/**
 * Compute the complex conjugate of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op conj(Op t) {}

/**
 * Compute the squared magnitude of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op norm(Op t) {}

/**
 * Compute absolute value of every element in the tensor. For complex numbers
 * this returns the magnitude, or sqrt(x^2+y^2)
 * @param t
 *   Tensor or operator input
 */
Op abs(Op t) {}

/**
 * Compute the sine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op sin(Op t) {}

/**
 * Compute cosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op cos(Op t) {}

/**
 * Compute the tangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op tan(Op t) {}

/**
 * Compute the hyperbolic sine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op sinh(Op t) {}

/**
 * Compute hyperbolic cosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op cosh(Op t) {}

/**
 * Compute the hyperbolic tangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op tanh(Op t) {}

/**
 * Compute the arcsine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op asin(Op t) {}

/**
 * Compute arccosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op acos(Op t) {}

/**
 * Compute the arctangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op atan(Op t) {}

/**
 * Compute the hyperbolic arcsine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op asinh(Op t) {}

/**
 * Compute hyperbolic arccosine of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op acosh(Op t) {}

/**
 * Compute hyperbolic the arctangent of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op atanh(Op t) {}

/**
 * Compute the angle of a complex number.
 * @param t
 *   Tensor or operator input
 */
Op angle(Op t) {}

/**
 * Compute the principal value of the arctangent of y/x for complex numbers
 * @param t
 *   Tensor or operator input
 */
Op atan2(Op t) {}

/**
 * Compute the floor of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op floor(Op t) {}

/**
 * Compute the ceiling of every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op ceil(Op t) {}

/**
 * Round every element in the tensor
 * @param t
 *   Tensor or operator input
 */
Op round(Op t) {}

/**
 * Compute !t (logical NOT) of input tensor or operator
 * @param t
 *   LHS tensor or operator input
 */
Op operator!(Op t) {}

/***** Binary operators ********/

/**
 * Add two operators or tensors
 * @param t
 *   Tensor or operator input
 * @param t2
 *   RHS second tensor or operator input
 */
Op operator+(Op t, Op t2) {}

/**
 * Subtract two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS second tensor or operator input
 */
Op operator-(Op t, Op t2) {}

/**
 * Multiply two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS second tensor or operator input
 */
Op operator*(Op t, Op t2) {}

/**
 * Multiply two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS second tensor or operator input
 */
Op mul(Op t, Op t2) {}

/**
 * Divide two operators or tensors
 * @param t
 *   LHS tensor numerator
 * @param t2
 *   RHS tensor or operator denominator
 */
Op operator/(Op t, Op t2) {}

/**
 * Modulo two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS second tensor or operator modulus
 */
Op operator%(Op t, Op t2) {}

/**
 * Compute the t^t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator power
 */
Op pow(Op t, Op t2) {}

/**
 * Compute max(t, t2) of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op max(Op t, Op t2) {}

/**
 * Compute min(t, t2) of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op min(Op t, Op t2) {}

/**
 * Compute t < t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator<(Op t, Op t2) {}

/**
 * Compute t > t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator>(Op t, Op t2) {}

/**
 * Compute t <= t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator<=(Op t, Op t2) {}

/**
 * Compute t >= t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator>=(Op t, Op t2) {}

/**
 * Compute t == t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator==(Op t, Op t2) {}

/**
 * Compute t != t2 of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator!=(Op t, Op t2) {}

/**
 * Compute t && t2 (logical AND) of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator&&(Op t, Op t2) {}

/**
 * Compute t || t2 (logical OR) of two operators or tensors
 * @param t
 *   LHS tensor or operator input
 * @param t2
 *   RHS tensor or operator input
 */
Op operator||(Op t, Op t2) {}
#else
DEFINE_UNARY_OP(sqrt, SqrtOp);
DEFINE_UNARY_OP(exp, ExpOp);
DEFINE_UNARY_OP(expj, ExpjOp);
DEFINE_UNARY_OP(log10, Log10Op);
DEFINE_UNARY_OP(log2, Log2Op);
DEFINE_UNARY_OP(log, LogOp);
DEFINE_UNARY_OP(loge, LogOp);
DEFINE_UNARY_OP(conj, ConjOp);
DEFINE_UNARY_OP(norm, NormOp);
DEFINE_UNARY_OP(abs, AbsOp);
DEFINE_UNARY_OP(sin, SinOp);
DEFINE_UNARY_OP(cos, CosOp);
DEFINE_UNARY_OP(tan, TanOp);
DEFINE_UNARY_OP(asin, AsinOp);
DEFINE_UNARY_OP(acos, AcosOp);
DEFINE_UNARY_OP(atan, AtanOp);
DEFINE_UNARY_OP(sinh, SinhOp);
DEFINE_UNARY_OP(cosh, CoshOp);
DEFINE_UNARY_OP(tanh, TanhOp);
DEFINE_UNARY_OP(asinh, AsinhOp);
DEFINE_UNARY_OP(acosh, AcoshOp);
DEFINE_UNARY_OP(atanh, AtanhOp);
DEFINE_UNARY_OP(angle, AngleOp);
DEFINE_UNARY_OP(atan2, AngleOp);
DEFINE_UNARY_OP(floor, FloorOp);
DEFINE_UNARY_OP(ceil, CeilOp);
DEFINE_UNARY_OP(round, RoundOp);
// DEFINE_UNARY_OP( operator-, SubNegOp );

DEFINE_BINARY_OP(operator+, AddOp);
DEFINE_BINARY_OP(operator-, SubOp);
DEFINE_BINARY_OP(operator*, MulOp);
DEFINE_BINARY_OP(mul, MulOp);
DEFINE_BINARY_OP(operator/, DivOp);
DEFINE_BINARY_OP(operator%, ModOp);
DEFINE_BINARY_OP(operator|, OrOp);
DEFINE_BINARY_OP(operator&, AndOp);
DEFINE_BINARY_OP(operator^, XorOp);
DEFINE_BINARY_OP(pow, PowOp);
DEFINE_BINARY_OP(max, MaxOp);
DEFINE_BINARY_OP(min, MinOp);
DEFINE_BINARY_OP(operator<, LTOp);
DEFINE_BINARY_OP(operator>, GTOp);
DEFINE_BINARY_OP(operator<=, LTEOp);
DEFINE_BINARY_OP(operator>=, GTEOp);
DEFINE_BINARY_OP(operator==, EQOp);
DEFINE_BINARY_OP(operator!=, NEOp);
DEFINE_BINARY_OP(operator&&, AndAndOp);
DEFINE_BINARY_OP(operator||, OrOrOp);
DEFINE_UNARY_OP(operator!, NotOp);
#endif

// Doxygen doesn't recognize macros generating functions, so we need to fake
// each one here

} // end namespace matx
