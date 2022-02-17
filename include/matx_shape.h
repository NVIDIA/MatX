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
#include <iomanip>
#include <type_traits>

#include "matx_allocator.h"
#include "matx_error.h"
#include "matx_type_utils.h"

namespace matx {

/**
 * Class containing the shape and type of a tensor.
 *
 * tensorShape_t contains metadata about the shape of
 * a tensor but does not contain any data.
 */
template <int RANK> class tensorShape_t {
public:
  using matx_shape = bool; ///< Type trait to indicate shape

  tensorShape_t(){};

  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param sizes
   *   Sizes for each dimension. Length of sizes must match RANK
   */
  inline tensorShape_t(const index_t *const sizes) noexcept
  {
#pragma unroll
    for (int i = 0; i < RANK; i++) {
      n_[i] = sizes[i];
    }
  }

  /**
   * Constructor for a rank-1 and above tensor using initializer lists
   *
   * @param sizes
   *   Sizes for each dimension. Length of sizes must match RANK
   */
  template <int M = RANK, std::enable_if_t<M != 0, bool> = true>
  inline tensorShape_t(const index_t (&sizes)[M])
      : tensorShape_t(reinterpret_cast<const index_t *>(sizes))
  {
  }

  /**
   * Constructor for a rank-1 and above tensor using initializer lists
   *
   * @param sizes
   *   Sizes for each dimension. Length of sizes must match RANK
   */
  template <typename T, std::enable_if_t<!std::is_array_v<typename remove_cvref<T>::type> && !std::is_pointer_v<typename remove_cvref<T>::type>, bool> = true>
  inline tensorShape_t(const T &sizes)
      : tensorShape_t(reinterpret_cast<const index_t *>(sizes.data()))
  {
  }  

  /**
   * Get the size of a single dimension of the tensor
   *
   * @param dim
   *   Desired dimension
   *
   * @returns Number of elements in dimension
   *
   */
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const noexcept
  {
    if constexpr (Rank() > 0)
      return n_[dim];
    else
      return 0;

    // BUG WAR
    if constexpr (!(Rank() > 0))
      return 0;
    else
      return n_[dim];
  }

  /**
   * Get the total size of the shape
   *
   * @return
   *    The size of all dimensions combined. Note that this does not include the
   *    size of the data type itself, but only the product of the lengths of
   * each dimension
   */
  inline __MATX_HOST__ __MATX_DEVICE__ index_t TotalSize() const noexcept
  {
    index_t size = 1;
    for (int i = 0; i < RANK; i++) {
      size *= Size(i);
    }
    return size;
  }

  /**
   * Set the size of a dimension
   *
   * @param dim
   *   Dimension to set
   * @param size
   *   Set the size of a dimension
   */
  inline void SetSize(int dim, index_t size)
  {
    MATX_ASSERT(dim < RANK, matxInvalidDim);
    n_[dim] = size;
  }

  /**
   * Get shape as an array
   *
   */
  inline std::array<index_t, RANK> AsArray() const
  {
    return n_;
  }  

  /**
   * Get the rank of the tensor
   *
   * @returns Rank of the tensor
   *
   */
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

private:
  std::array<index_t, RANK> n_;
};

/**
 * @brief Shape equality
 * 
 * @tparam RANK1 Rank of first tensor
 * @tparam RANK2 Rank of second tensor
 * @param lhs Left shape
 * @param rhs Right shape
 * @return True if shapes are equal
 */
template <int RANK1, int RANK2>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator==(const tensorShape_t<RANK1> &lhs, const tensorShape_t<RANK2> &rhs)
{
  if constexpr (RANK1 != RANK2) {
    return false;
  }

  for (int i = 0; i < RANK1; i++) {
    if (lhs.Size(i) != rhs.Size(i)) {
      return false;
    }
  }

  return true;
}

/**
 * @brief Shape inequality
 * 
 * @tparam RANK1 Rank of first tensor
 * @tparam RANK2 Rank of second tensor
 * @param lhs Left shape
 * @param rhs Right shape
 * @return True if shapes are not equal
 */
template <int RANK1, int RANK2>
__MATX_HOST__ __MATX_DEVICE__ __forceinline__ bool
operator!=(const tensorShape_t<RANK1> &lhs, const tensorShape_t<RANK2> &rhs)
{
  return !(lhs == rhs);
}

}; // namespace matx
