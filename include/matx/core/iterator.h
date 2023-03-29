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

#include "matx/core/defines.h"

namespace matx {
/**
 * @brief Iterator around operators for libraries that can take iterators as input (CUB).
 * 
 * @tparam T Data type
 * @tparam RANK Rank of tensor
 * @tparam Desc Descriptor for tensor
 * 
 */
template <typename OperatorType>
struct RandomOperatorIterator {
  using self_type = RandomOperatorIterator<OperatorType>;
  using value_type = detail::convert_matx_type_t<typename OperatorType::scalar_type>;
  using scalar_type = value_type;
  // using stride_type = std::conditional_t<is_tensor_view_v<OperatorType>, typename OperatorType::desc_type::stride_type,
  //                         index_t>;
  using stride_type = index_t;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = index_t;

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ RandomOperatorIterator(const OperatorType &t) : t_(t), offset_(0) { }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ RandomOperatorIterator(const OperatorType &t, stride_type offset) : t_(t), offset_(offset) {}

  /**
   * @brief Dereference value at a pre-computed offset
   * 
   * @return Value at offset 
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ value_type operator*() const
  {
    if constexpr (OperatorType::Rank() == 0) {
      return static_cast<value_type>(t_.operator()());
    }
    else {
      auto arrs = detail::GetIdxFromAbs(t_, offset_);
      return detail::mapply([&](auto &&...args) {
          return static_cast<value_type>(t_.operator()(args...));
        }, arrs);     
    }
  }  


  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator+(difference_type offset) const
  {
    return self_type{t_, offset_ + offset};
  }

  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ value_type operator[](difference_type offset) const
  {
    return *self_type{t_, offset_ + offset};
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  self_type operator++(int)
  {
      self_type retval = *this;
      offset_++;
      return retval;
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator++()
  {
      offset_++;
      return *this;
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type& operator+=(difference_type offset)
  {
      offset_ += offset;
      return *this;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator-(difference_type offset) const
  {
      return self_type{t_, offset_ - offset};
  }


  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type& operator-=(difference_type offset)
  {
      offset_ -= offset;
      return *this;
  }  

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
    return OperatorType::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
  {
    return t_.Size(dim);
  }  

  typename detail::base_type_t<OperatorType> t_;
  stride_type offset_;  
};



/**
 * @brief Iterator around operators for libraries that can take iterators as output (CUB).
 * 
 * @tparam T Data type
 * @tparam RANK Rank of tensor
 * @tparam Desc Descriptor for tensor
 * 
 */
template <typename OperatorType>
struct RandomOperatorOutputIterator {
  using self_type = RandomOperatorOutputIterator<OperatorType>;
  using value_type = detail::convert_matx_type_t<typename OperatorType::scalar_type>;
  using scalar_type = value_type;
  // using stride_type = std::conditional_t<is_tensor_view_v<OperatorType>, typename OperatorType::desc_type::stride_type,
  //                         index_t>;
  using stride_type = index_t;
  using pointer = value_type*;
  using reference = value_type&;
  using iterator_category = std::random_access_iterator_tag;
  using difference_type = index_t;

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ RandomOperatorOutputIterator(const OperatorType &t) : t_(t), offset_(0) { }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ RandomOperatorOutputIterator(const OperatorType &t, stride_type offset) : t_(t), offset_(offset) {}

  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ reference operator*()
  {
    if constexpr (OperatorType::Rank() == 0) {
      return (reference)(t_.operator()());
    }
    else {
      auto arrs = detail::GetIdxFromAbs(t_, offset_);

      return std::apply([&](auto &&...args) -> reference {
          return (reference)(t_.operator()(args...));
        }, arrs);    
    }
  }  

  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator+(difference_type offset) const
  {
    return self_type{t_, offset_ + offset};
  }
  

  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ reference operator[](difference_type offset) 
  {
    return *self_type{t_, offset_ + offset};
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  self_type operator++(int)
  {
      self_type retval = *this;
      offset_++;
      return retval;
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator++()
  {
      offset_++;
      return *this;
  }  

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type& operator+=(difference_type offset)
  {
      offset_ += offset;
      return *this;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type operator-(difference_type offset) const
  {
      return self_type{t_, offset_ - offset};
  }


  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ self_type& operator-=(difference_type offset)
  {
      offset_ -= offset;
      return *this;
  }  

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
    return OperatorType::Rank();
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
  {
    return t_.Size(dim);
  }    

  typename detail::base_type_t<OperatorType> t_;
  stride_type offset_;  
};

};