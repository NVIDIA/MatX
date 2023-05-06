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
#include "matx/core/error.h"

namespace matx {

/**
 * @brief Type-erased generic tensor descriptor for strides and sizes
 *
 * @tparam ShapeContainer type of sizes
 * @tparam StrideContainer type of strides
 */
template <typename ShapeContainer, typename StrideContainer, int RANK> 
class tensor_desc_t {
public:
  template <typename T1, typename T2, int R1>
  using self_type = tensor_desc_t<T1, T2, R1>;
  
  using shape_container = ShapeContainer;
  using stride_container = StrideContainer;
  using shape_type  = typename ShapeContainer::value_type; ///< Type trait of shape type
  using stride_type = typename StrideContainer::value_type; ///< Type trait of stride type
  using matx_descriptor = bool; ///< Type trait to indicate this is a tensor descriptor

  /**
   * @brief Default copy constructor
   */
  __MATX_INLINE__  tensor_desc_t<ShapeContainer, StrideContainer, RANK>(const tensor_desc_t &) = default;

  /**
   * @brief Default move constructor
   */  
  __MATX_INLINE__  tensor_desc_t<ShapeContainer, StrideContainer, RANK>(tensor_desc_t &&) = default;

  /**
   * @brief Default const copy assignment constructor
   */  
  __MATX_INLINE__  tensor_desc_t& operator=(const tensor_desc_t&) = default;

  /**
   * @brief Default copy assignment constructor
   */    
  __MATX_INLINE__  tensor_desc_t& operator=(tensor_desc_t&&) = default;

  /** Swaps two raw_pointer_buffers
   *
   * Swaps members of two raw_pointer_buffers
   *
   * @param lhs
   *   Left argument
   * @param rhs
   *   Right argument
   */
  friend void swap( tensor_desc_t<ShapeContainer, StrideContainer, RANK> &lhs, 
                    tensor_desc_t<ShapeContainer, StrideContainer, RANK> &rhs) noexcept
  {
    using std::swap;

    swap(lhs.shape_, rhs.shape_);
    swap(lhs.stride_, rhs.stride_);
  }   

  /**
   * @brief Construct a tensor_desc_t from a generic shape and stride
   * 
   * @tparam S Unused
   * @param shape Shape object
   * @param stride Stride object
   */
  template <typename S = ShapeContainer, std::enable_if_t<!std::is_array_v<ShapeContainer> && !std::is_array_v<StrideContainer>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(ShapeContainer &&shape, StrideContainer &&stride)
      : shape_(std::forward<ShapeContainer>(shape)),
        stride_(std::forward<StrideContainer>(stride)) {
    MATX_ASSERT_STR(shape.size() == stride.size(), matxInvalidDim,
                       "Size and stride array sizes must match");
    MATX_ASSERT_STR(shape.size() == RANK, matxInvalidDim,
                       "Rank parameter must match array size");                       
  }

  /**
   * @brief Construct a tensor_desc_t for a 0D tensor
   * 
   */
  __MATX_INLINE__ __MATX_HOST__  tensor_desc_t() {
  } 

  /**
   * @brief Constructor with just shape for non-C-style arrays
   * 
   * @tparam S2 Unused
   * @param shape 
   *   Shape of tensor
   */
  template <typename S2, std::enable_if_t<!std::is_array_v<typename remove_cvref<S2>::type> && !is_matx_descriptor_v<typename remove_cvref<S2>::type>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__  tensor_desc_t(S2 &&shape) 
  {
    InitFromShape(std::forward<S2>(shape));
  }

  /**
   * @brief Constructor with just shape for C-style arrays
   * 
   * @tparam M 
   *   Unused
   * @param shape 
   *   Shape of tensor
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
   *   Shape of tensor
   * @param strides
   *   Strides of tensor
   */
  template <typename S2, std::enable_if_t<!std::is_array_v<S2>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(S2 &&shape, const stride_type (&strides)[RANK]) : 
      shape_(std::forward<S2>(shape)) {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(*(shape.begin() + i) > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
      *(stride_.begin() + i) = strides[i];
    }
  }

  /**
   * @brief Constructor with perfect-forwarded shape and C array of strides
   * 
   * @param shape 
   *   Shape of tensor
   * @param strides
   *   Strides of tensor
   */
  template <std::enable_if_t<!std::is_array_v<StrideContainer>, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ tensor_desc_t(const shape_type (&shape)[RANK], StrideContainer &&strides) : 
      stride_(std::forward<StrideContainer>(strides)) {
    for (int i = 0; i < RANK; i++) {
      MATX_ASSERT_STR(shape[i] > 0, matxInvalidSize,
                      "Must specify size larger than 0 for each dimension");
      *(shape_.begin() + i) = shape[i];
    }
  }  

  /**
   * @brief Constructor with C-style array shape and strides
   * 
   * @param shape 
   *   Shape of tensor
   * @param strides
   *   Strides of tensor
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
   * Check if a descriptor is contiguous in memory for all elements in the view
   *
   * @return
   *    True is descriptor is contiguous, or false otherwise
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool IsContiguous() const noexcept
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
   *    size of the data type itself, but only the product of the lengths of each dimension
   */
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto TotalSize() const noexcept
  {
    // The stride_type is expected to be able to hold this without overflowing
    stride_type size = 1; 
    for (int i = 0; i < RANK; i++) {
      size *= Size(i);
    }
    return size;
  }  

  /**
   * @brief Initialize descriptor from existing shape
   * 
   * @tparam S2 Shape type
   * @param shape Shape object
   */
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

  /**
   * @brief Set the Size object
   * 
   * @param dim Dimension to size
   * @param size Size to set dimension to
   * 
   */
  void __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ SetSize(int dim, shape_type size) { *(shape_.begin() + dim) = size; }

  /**
   * @brief Return size of descriptor on a single dimension
   * 
   * @param dim Dimension to retrieve
   * @return Size of dimension
   */
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size([[maybe_unused]] int dim) const noexcept { 
    if constexpr (RANK == 0) {
      return static_cast<shape_type>(1);
    }

    return *(shape_.begin() + dim); 
  }

  /**
   * @brief Return strides contaienr of descriptor
   * 
   * @return Strides container
   */
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Strides() const { 
    return stride_;
  }
  /**
   * @brief Return stride of descriptor on a single dimension
   * 
   * @param dim Dimension to retrieve
   * @return Stride of dimension
   */
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Stride([[maybe_unused]] int dim) const { 
    if constexpr (RANK == 0) {
      return static_cast<stride_type>(0);
    }

    /*  In release mode with O3 on g++ seems to give incorrect warnings on this line from Clone()
        and clone(). It appears there's no valid code path that would cause this to be unitialized,
        so we're ignoring the warning in this one spot. */
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wmaybe-uninitialized"
    return *(stride_.begin() + dim); 
    #pragma GCC diagnostic pop
  }

  /**
   * @brief Return shape object
   * 
   * @return Shape object 
   */
  auto __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Shape() const { return shape_; }

  /**
   * @brief Get rank of descriptor
   * 
   * @return Rank of descriptor
   */
  static auto constexpr Rank() { return RANK; }

private:
  ShapeContainer shape_;
  StrideContainer stride_;
};

/**
 * @brief Tensor descriptor for compile-time descriptors
 * 
 * @tparam I First size
 * @tparam Is Parameter pack of sizes
 */
template <index_t I, index_t... Is> 
class static_tensor_desc_t {
public:
  using ShapeContainer = std::array<index_t, sizeof...(Is)>;  ///< Type trait of shape type
  using StrideContainer = std::array<index_t, sizeof...(Is)>; ///< Type trait of stride type
  using shape_type  = index_t; ///< Type trait of shape container
  using stride_type = index_t; ///< Type trait of stride container
  using matx_descriptor = bool; ///< Type trait to indicate this is a tensor descriptor

  /**
   * Check if a descriptor is linear in memory for all elements in the view
   *
   * @return
   *    True is descriptor is linear, or false otherwise
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool IsContiguous() const noexcept
  {
    return true;
  }

  /**
   * @brief Get size of dimension
   * 
   * @param dim Dimension to retrieve
   * @return Size of dimension
   */
  static constexpr auto Size(int dim) { return shape_[dim]; }

  /**
   * @brief Get stride of dimension
   * 
   * @param dim Dimension to retrieve
   * @return Stride of dimension
   */  
  static constexpr auto Stride(int dim) { return stride_[dim]; }

  /**
   * @brief Get rank of descriptor
   * 
   * @return Descriptor rank
   */  
  static constexpr int Rank() { return shape_.size(); }

  /**
   * @brief Get underlying shape object
   * 
   * @return Shape object
   */  
  static constexpr auto Shape() { return shape_; }

  /**
   * @brief Get total size of descriptor
   * 
   * @return Product of all sizes
   */  
  static constexpr auto TotalSize() {
      return std::accumulate(shape_.begin(), shape_.end(), 1, std::multiplies<index_t>());
  }  

private:
  static constexpr auto make_shape(){
      return std::array{I, Is...};
  }    

  static constexpr auto make_strides(){
      std::array<index_t, 1 + sizeof...(Is)> m{};
      m[m.size()-1] = 1;
      if constexpr (m.size() > 1) {
        for (int i = m.size()-2; i >= 0; i--) {
            m[i] = m[i+1] * Size(i + 1);
        }
      }
      return m;
  }

  static constexpr ShapeContainer shape_ = make_shape();
  static constexpr StrideContainer stride_ = make_strides();  
};

/**
 * @brief Constant rank, dynamic size, dynamic strides
 *
 * @tparam ShapeContainer Type of shape
 * @tparam StrideContainer Type of stride container
 * @tparam RANK Rank of shape
 */
template <typename ShapeContainer, typename StrideContainer, int RANK>
using tensor_desc_cr_ds_t =
    tensor_desc_t<std::array<ShapeContainer, RANK>, std::array<StrideContainer, RANK>,
                  RANK>;

/**
 * @brief 32-bit size and stride descriptor
 *
 * @tparam RANK Rank of shape
 */
template <int RANK>
using tensor_desc_cr_ds_32_32_t =
    tensor_desc_cr_ds_t<int32_t, int32_t, RANK>;

/**
 * @brief 64-bit size and stride descriptor
 *
 * @tparam RANK Rank of shape
 */
template <int RANK>
using tensor_desc_cr_ds_64_64_t =
    tensor_desc_cr_ds_t<long long int, long long int, RANK>;

/**
 * @brief 32-bit size and 64-bit stride descriptor
 *
 * @tparam RANK Rank of shape
 */
template <int RANK>
using tensor_desc_cr_ds_32_64_t =
    tensor_desc_cr_ds_t<int32_t, long long int, RANK>;

/**
 * @brief index_t size and stride descriptor
 *
 * @tparam RANK Rank of shape
 */
template <int RANK>
using tensor_desc_cr_disi_dist = tensor_desc_cr_ds_t<index_t, index_t, RANK>;

/**
 * @brief Default descriptor type
 *
 * @tparam RANK Rank of shape
 */
#ifdef INDEX_64_BIT 
  template <int RANK>
  using DefaultDescriptor = tensor_desc_cr_ds_64_64_t<RANK>;
#else
  template <int RANK>
  using DefaultDescriptor = tensor_desc_cr_ds_32_32_t<RANK>;
#endif

}; // namespace matx
