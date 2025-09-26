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

#include "matx/core/type_utils.h"
#include "matx/core/allocator.h"
#include "matx/core/error.h"
#include <cstring>
#include <type_traits>
#include <utility>
#include <memory>

namespace matx
{
  /**
   * @brief SFINAE helper to detect duck-typed allocator interface
   */
  template<typename T, typename = void>
  struct has_allocator_interface : std::false_type {};

  /// @cond DOXYGEN_SHOULD_SKIP_THIS
  // For allocator objects
  template<typename T>
  struct has_allocator_interface<T, 
    decltype(
      std::declval<T>().allocate(std::declval<size_t>()),
      std::declval<T>().deallocate(std::declval<void*>(), std::declval<size_t>()),
      void()
    )
  > : std::true_type {};
  /// @endcond

  /// @cond DOXYGEN_SHOULD_SKIP_THIS
  // For allocator pointers
  template<typename T>
  struct has_allocator_interface<T*, 
    decltype(
      std::declval<T*>()->allocate(std::declval<size_t>()),
      std::declval<T*>()->deallocate(std::declval<void*>(), std::declval<size_t>()),
      void()
    )
  > : std::true_type {};
  /// @endcond

  /**
   * @brief Unified storage class for both owning and non-owning pointers
   * 
   * Supports duck-typed custom allocators while avoiding virtual functions
   * and pointer members in tensors.
   */
  template <typename T>
  class Storage {
  public:
    using value_type = T;
    using iterator = T*;
    using const_iterator = const T*;

    // Default constructor - creates empty storage
    Storage() : size_(0), data_() {}

    // Non-owning constructor - wraps existing pointer
    Storage(T* ptr, size_t size) 
      : size_(size), data_(ptr, [](T*){}) {}
    
    // Owning constructor with default allocator
    Storage(size_t size) 
      : size_(size) {
      if (size == 0) {
        // For zero size, create empty storage without allocation
        data_ = std::shared_ptr<T>();
      } else {
        T* ptr = matx_allocator<T>{}.allocate(size * sizeof(T));
        data_ = std::shared_ptr<T>(ptr, [size](T* p) {
          matx_allocator<T>{}.deallocate(p, size * sizeof(T));
        });
      }
    }
    
    // Owning constructor with any allocator that has allocate/deallocate methods
    template<typename Allocator>
    Storage(size_t size, Allocator&& alloc, 
           typename std::enable_if_t<has_allocator_interface<std::decay_t<Allocator>>::value>* = nullptr)
      : size_(size) {
      if (size == 0) {
        // For zero size, create empty storage without allocation
        data_ = std::shared_ptr<T>();
      } else if constexpr (std::is_pointer_v<std::decay_t<Allocator>>) {
        // Handle allocator pointers (any pointer to object with allocate/deallocate methods)
        T* ptr = static_cast<T*>(alloc->allocate(size * sizeof(T)));
        data_ = std::shared_ptr<T>(ptr, [size, alloc](T* p) {
          alloc->deallocate(p, size * sizeof(T));
        });
      } else {
        // Handle allocator objects with duck typing
        T* ptr = static_cast<T*>(alloc.allocate(size * sizeof(T)));
        data_ = std::shared_ptr<T>(ptr, [size, alloc](T* p) mutable {
          alloc.deallocate(p, size * sizeof(T));
        });
      }
    }
    
    // Constructor from existing shared_ptr
    Storage(std::shared_ptr<T> ptr, size_t size)
      : size_(size), data_(ptr) {}
    
    // Owning constructor with memory space
    Storage(size_t size, matxMemorySpace_t space, cudaStream_t stream = 0)
      : size_(size) {
      if (size == 0) {
        // For zero size, create empty storage without allocation
        data_ = std::shared_ptr<T>();
      } else {
        void* ptr;
        matxAlloc(&ptr, size * sizeof(T), space, stream);
        data_ = std::shared_ptr<T>(static_cast<T*>(ptr), [stream](T* p) {
          matxFree(p, stream);
        });
      }
    }

    // Core interface
    T* data() noexcept { return data_.get(); }
    const T* data() const noexcept { return data_.get(); }
    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return size_; }
    size_t use_count() const noexcept { return data_.use_count(); }
    
    // Iterator interface
    iterator begin() noexcept { return data(); }
    iterator end() noexcept { return data() + size(); }
    const_iterator begin() const noexcept { return data(); }
    const_iterator end() const noexcept { return data() + size(); }
    const_iterator cbegin() const noexcept { return data(); }
    const_iterator cend() const noexcept { return data() + size(); }

    // Utility functions
    size_t bytes() const noexcept { return sizeof(T) * capacity(); }

  private:
    size_t size_;
    std::shared_ptr<T> data_;
  };



  /**
   * @brief Factory function to create owning storage with default allocator
   */
  template <typename T>
  Storage<T> make_owning_storage(size_t size) {
    return Storage<T>(size);
  }

  /**
   * @brief Factory function to create owning storage with custom allocator (supports duck typing)
   */
  template <typename T, typename Allocator>
  Storage<T> make_owning_storage(size_t size, Allocator&& alloc) {
    return Storage<T>(size, std::forward<Allocator>(alloc));
  }

  /**
   * @brief Factory function to create owning storage with MatX memory space
   */
  template <typename T>
  Storage<T> make_owning_storage(size_t size, matxMemorySpace_t space, cudaStream_t stream = 0) {
    return Storage<T>(size, space, stream);
  }

  /**
   * @brief Factory function to create storage from existing shared_ptr
   */
  template <typename T>
  Storage<T> make_storage_from_shared_ptr(std::shared_ptr<T> ptr, size_t size) {
    return Storage<T>(ptr, size);
  }

  /**
   * @brief Factory function to create non-owning storage
   */
  template <typename T>
  Storage<T> make_non_owning_storage(T* ptr, size_t size) {
    return Storage<T>(ptr, size);
  }


};