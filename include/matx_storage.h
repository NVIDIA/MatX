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

#include "matx_type_utils.h"
#include "matx_allocator.h"
#include "matx_error.h"

namespace matx
{
  /**
   * @brief Legacy storage method
   * 
   * Used to signal the old semantics where everything is stored as a shared_ptr internally
   * that may or may not have ownership
   * 
   */

  template <typename T, typename O, typename Allocator = matx_allocator<T>>
  class raw_pointer_buffer
  {
  public:
    using value_type = T;
    using iterator = T *;
    using citerator = T const *;
    using matx_storage_container = bool;

    raw_pointer_buffer(size_t size) : size_(size) {
      T *tmp = alloc_.allocate(size);
      ConfigureShared(tmp, size); 
    }

    raw_pointer_buffer(T *ptr, size_t size) : size_(size) { 
      ConfigureShared(ptr, size);  
    }

    raw_pointer_buffer(const raw_pointer_buffer &rhs) = default;
    raw_pointer_buffer& operator=(const raw_pointer_buffer &) = default;
    raw_pointer_buffer& operator=(raw_pointer_buffer &&) = default;
    raw_pointer_buffer(const raw_pointer_buffer &&rhs) noexcept {
      size_ = rhs.size_; 
      data_ = std::move(rhs.data_);
    } 
    ~raw_pointer_buffer() {}

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return data_.get();
    }

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + size_;
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + size_;
    }

    size_t size() const
    {
      return size_;
    }

    void SetData(T *const data) noexcept
    {
      data_.reset(data, [](auto ){});
    }   

    __MATX_INLINE__ T* allocate(size_t size)
    {
      alloc_.allocate();
    }

    auto capacity() const noexcept {
      return size_;
    }     

    /**
     * Get the reference count
     *
     * @returns Reference count or 0 if not tracked
     *
     */
    __MATX_INLINE__ __MATX_HOST__ auto use_count() const noexcept
    {
      return data_.use_count();
    }     

  private:
    Allocator alloc_ = {};
    size_t size_;
    std::shared_ptr<T> data_;

    void ConfigureShared(T *ptr, size_t size) {
      if constexpr (std::is_same_v<O, non_owning>) {
        data_ = std::shared_ptr<T>(ptr, [](auto){});
      }
      else {
        data_ = std::shared_ptr<T>(ptr, [&](auto p) { alloc_.deallocate(reinterpret_cast<void*>(p), size); });
      }        
    }
  };

  template <typename T>
  class smart_pointer_buffer
  {
  public:
    using value_type = T;
    using iterator = T *;
    using citerator = T const *;
    using matx_storage_container = bool;

    smart_pointer_buffer<T>() = delete;
    smart_pointer_buffer<T>(T ptr, size_t size) : data_(ptr), size_(size) {
      static_assert(is_smart_ptr_v<T>);
    }
    smart_pointer_buffer<T>(T &&ptr, size_t size) : data_(std::move(ptr)), size_(size) {
      static_assert(is_smart_ptr_v<T>);
    }

    smart_pointer_buffer<T>(const smart_pointer_buffer<T> &rhs) = default;
    smart_pointer_buffer& operator=(const smart_pointer_buffer &) = default;
    smart_pointer_buffer& operator=(smart_pointer_buffer &&) = default;
    smart_pointer_buffer<T>(const smart_pointer_buffer<T> &&rhs) {
      size_ = rhs.size_; 
      data_ = std::move(rhs.data_);
    }
    
    ~smart_pointer_buffer<T>() = default;

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return data_.get();
    }

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + size_;
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + size_;
    }

    size_t size() const
    {
      return size_;
    }

    void SetData(T *const data) noexcept
    {
      data_.reset(data_, [](auto){});
    }

    auto capacity() const noexcept {
      return size_;
    }

    __MATX_INLINE__ T* allocate(size_t size)
    {
      MATX_THROW(matxInvalidParameter, "Cannot call allocate on a smart pointer storage type");
    }

    /**
     * Get the reference count
     *
     * @returns Reference count or 0 if not tracked
     *
     */
    __MATX_INLINE__ __MATX_HOST__ auto use_count() const noexcept
    {
      return data_.use_count();
    }         

  private:
    size_t size_;
    T data_;
  };

  /**
   * @brief Primitive class to hold storage objects
   * 
   */
  template <typename C>
  class basic_storage
  {
  public:
    using value_type = typename C::value_type;
    using T = value_type;
    using iterator = value_type*;
    using citerator = value_type const *;
    using matx_storage = bool;
    using container = C;

    void SetData(T *const data) noexcept
    {
      static_assert(is_matx_storage_container_v<C>, "Must use MatX storage container type if trying to set data pointer");
      
      container_.SetData(data);
    }
    template <typename C2, std::enable_if_t<!is_matx_storage_v<C2>, bool> = true>
    __MATX_INLINE__ basic_storage(C2 &&obj) : container_(std::forward<C2>(obj))
    {
    }

    __MATX_INLINE__ basic_storage(const basic_storage &) = default;
    __MATX_INLINE__ basic_storage(basic_storage &&) = default;
    __MATX_INLINE__ basic_storage& operator=(const basic_storage &) = default;

    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return container_.data();
    }    

    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return container_.begin();
    }

    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return container_.end();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return container_.cbegin();
    }

    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return container_.cend();
    }

    __MATX_INLINE__ T* allocate(size_t size)
    {
      container_.allocate();
    }

    /**
     * Get the reference count
     *
     * @returns Reference count or 0 if not tracked
     *
     */
    __MATX_INLINE__ __MATX_HOST__ auto use_count() const noexcept
    {
      return container_.use_count();
    }

    /**
     * Get the total size of the underlying data, including the multiple of the
     * type
     *
     * @return
     *    The size (in bytes) of all dimensions combined
     */
    __MATX_INLINE__ size_t Bytes() const noexcept { return sizeof(T) * container_.capacity(); };    

  private:
    C container_;
  };

  template <typename T>
  using DefaultStorage = basic_storage<raw_pointer_buffer<T, owning, matx_allocator<T>>>;
};