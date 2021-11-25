////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
// this
//  list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <array>
#include <type_traits>
#include "matx_error.h"

namespace matx {


/**
 * @brief Type-erased generic tensor descriptor for strides and sizes
 *
 * @tparam ShapeType type of sizes
 * @tparam StrideType type of strides
 */
template <typename ShapeType, typename StrideType, int RANK> 
class tensor_desc_t {
public:
  using shape_type  = typename ShapeType::value_type;
  using stride_type = typename StrideType::value_type;
  using shape_container = ShapeType;
  using stride_container = StrideType;
  using matx_descriptor = bool;

  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t<ShapeType, StrideType, RANK>(const tensor_desc_t& ) = default;
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t<ShapeType, StrideType, RANK>(tensor_desc_t&&) = default;
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t& operator=(const tensor_desc_t&) = default;
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t& operator=(tensor_desc_t&&) = default;

  template <typename S = ShapeType, std::enable_if_t<!std::is_array_v<ShapeType> && !std::is_array_v<StrideType>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(ShapeType &&shape, StrideType &&stride)
      : shape_(std::forward<ShapeType>(shape)),
        stride_(std::forward<StrideType>(stride)) {
    MATX_ASSERT_STR(shape.size() == stride.size(), matxInvalidDim,
                       "Size and stride array sizes must match");
    MATX_ASSERT_STR(shape.size() == RANK, matxInvalidDim,
                       "Rank parameter must match array size");                       
  }

  // 0D tensors
  __MATX_INLINE__ __MATX_HOST__  tensor_desc_t() {
  } 

  /**
   * @brief Constructor with just shape for non-C-style arrays
   * 
   * @param shape 
   */
  template <typename S2, std::enable_if_t<!std::is_array_v<typename remove_cvref<S2>::type> && !is_matx_descriptor_v<typename remove_cvref<S2>::type>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__  tensor_desc_t(S2 &&shape) 
  {
    InitFromShape(std::forward<S2>(shape));
  }

  /**
   * @brief Constructor with just shape for C-style arrays
   * 
   * @param shape 
   */
  template <int M = RANK>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(const index_t (&shape)[M])
  {
    // Construct a new std::array. Slower, but saves duplication
    std::array<index_t, M> tshape;
    std::move(std::begin(shape), std::end(shape), tshape.begin());    
    InitFromShape(std::move(tshape));
  }  

  /**
   * @brief Constructor with perfect-forwarded shape and C array of strides
   * 
   * @param shape 
   */
  template <std::enable_if_t<!std::is_array_v<ShapeType>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(ShapeType &&shape, const stride_type (&strides)[RANK]) : 
      shape_(std::forward<ShapeType>(shape)) {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(*(shape + i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
      *(stride_.begin() + i) = strides[i];
    }
  }

  /**
   * @brief Constructor with perfect-forwarded shape and C array of strides
   * 
   * @param shape 
   */
  template <std::enable_if_t<!std::is_array_v<StrideType>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(const shape_type (&shape)[RANK], StrideType &&strides) : 
      stride_(std::forward<StrideType>(strides)) {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape[i] > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
      *(shape_.begin() + i) = shape[i];
    }
  }  

  /**
   * @brief Constructor with perfect-forwarded shape and C array of strides
   * 
   * @param shape 
   */
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(const shape_type (&shape)[RANK], const stride_type (&strides)[RANK]) {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape[i] > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
      *(stride_.begin() + i) = strides[i];
      *(shape_.begin() + i) = shape[i];
    }
  }    

  /**
   * Check if a descriptor is linear in memory for all elements in the view
   *
   * @return
   *    True is descriptor is linear, or false otherwise
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ bool IsLinear() const noexcept
  {
    stride_type ttl = 1;
    for (int i = RANK - 1; i >= 0; i--) {
      if (Stride(i) != ttl) {
        return false;
      }

      ttl *= Size(i);
    }

    return true;
  }

  /**
   * Get the total size of the shape
   *
   * @return
   *    The size of all dimensions combined. Note that this does not include the
   *    size of the data type itself, but only the product of the lengths of
   * each dimension
   */
  inline __MATX_HOST__ __MATX_DEVICE__ auto TotalSize() const noexcept
  {
    // The stride_type is expected to be able to hold this without overflowing
    stride_type size = 1; 
    for (int i = 0; i < RANK; i++) {
      size *= Size(i);
    }
    return size;
  }  

  template <typename S2>
  void __MATX_INLINE__ __MATX_HOST__ InitFromShape(S2 &&shape) {
    shape_ = std::forward<S2>(shape);

    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(*(shape.begin() + i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
    }

    if constexpr (RANK >= 1) {
      *(stride_.end() - 1) = 1;
    }

    #pragma unroll
    for (int i = RANK - 2; i >= 0; i--) {
      *(stride_.begin() + i) = Stride(i+1) * Size(i+1);
    } 
    
  }  

  void __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ SetSize(int dim, shape_type size) { *(shape_.begin() + dim) = size; }
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size([[maybe_unused]] int dim) const { 
    if constexpr (RANK == 0) {
      return 1;
    }
    else {
      return *(shape_.begin() + dim); 
    }
  }
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Stride([[maybe_unused]] int dim) const { 
    if constexpr (RANK == 0) {
      return 0;
    }
    else {
      return *(stride_.begin() + dim); 
    }
  }
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Shape() const { return shape_; }
  static auto constexpr Rank() { return RANK; }

private:
  ShapeType shape_;
  StrideType stride_;
};

/**
 * @brief Constant rank, dynamic size, dynamic strides
 *
 */
template <typename ShapeType, typename StrideType, int RANK>
using tensor_desc_cr_ds_t =
    tensor_desc_t<std::array<ShapeType, RANK>, std::array<StrideType, RANK>,
                  RANK>;

// 32-bit size and strides
template <int RANK>
using tensor_desc_cr_ds_32_32_t =
    tensor_desc_cr_ds_t<int32_t, int32_t, RANK>;

// 64-bit size and strides
template <int RANK>
using tensor_desc_cr_ds_64_64_t =
    tensor_desc_cr_ds_t<long long int, long long int, RANK>;

// 32-bit size and 64-bit strides
template <int RANK>
using tensor_desc_cr_ds_32_64_t =
    tensor_desc_cr_ds_t<int32_t, long long int, RANK>;

// index_t based size and stride
template <int RANK>
using tensor_desc_cr_disi_dist = tensor_desc_cr_ds_t<index_t, index_t, RANK>;

template <int RANK>
using DefaultDescriptor = tensor_desc_cr_ds_64_64_t<RANK>;

// template <typename ShapeIntType, typename StrideIntType, int RANK>
// struct DescriptorType {
//   using type = tensor_desc_cr_ds_t<ShapeType, StrideType, RANK>;
// }

// template <typename ShapeIntType, typename StrideIntType, int RANK>
// class DescriptorType<int32_t, StrideIntType, RANK>

}; // namespace matx