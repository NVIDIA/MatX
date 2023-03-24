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

namespace matx
{
  /**
   * @brief Raw pointer buffer
   * 
   * Used to store a raw pointer backing a tensor. Supports ownership through a shared_ptr
   * internally.
   * 
   */
  template <typename T, typename Allocator = matx_allocator<T>>
  class raw_pointer_buffer
  {
  public:
    using value_type = T; ///< Type trait for value_type
    using iterator = T *; ///< Type trait for iterator value
    using citerator = T const *; ///< Type trait for const iterator value
    using matx_storage_container = bool; ///< Type trait to indicate this is a storage type

    /** 
     * @brief Default construct a raw_pointer_buffer. This should only be used when temporarily
     * creating an empty tensor for construction later.
    */
    raw_pointer_buffer() { }
    
    /**
     * @brief Construct a new raw pointer buffer object and allocate space
     * 
     * @param size Size of allocation
     */
    raw_pointer_buffer(size_t size) : size_(size), owning_(true) {
      T *tmp = alloc_.allocate(size);
      ConfigureShared(tmp, size); 
    }

    /**
     * @brief Construct a new raw pointer buffer object from an existing pointer and size
     * 
     * @param ptr Previously-allocated pointer
     * @param size Size of allocation
     * @param owning if this class owns memory
     */
    raw_pointer_buffer(T *ptr, size_t size, bool owning = false) : size_(size), owning_(owning) { 
      ConfigureShared(ptr, size);  
    }

    /**
     * @brief Default copy constructor
     * 
     */
    raw_pointer_buffer(const raw_pointer_buffer &rhs) = default;

    /**
     * @brief Default copy assignment constructor
     * 
     */      
    raw_pointer_buffer& operator=(const raw_pointer_buffer &) = default;

    /**
     * @brief Default move assignment constructor
     * 
     */      
    raw_pointer_buffer& operator=(raw_pointer_buffer &&) = default;

    /**
     * @brief Move constructor
     * 
     */      
    raw_pointer_buffer(raw_pointer_buffer &&rhs) noexcept {
      size_ = rhs.size_; 
      data_ = std::move(rhs.data_);
      owning_ = rhs.owning_;
    } 

    /**
     * @brief Destroy the raw pointer buffer object
     * 
     */
    ~raw_pointer_buffer() = default;

    /** Swaps two raw_pointer_buffers
     *
     * Swaps members of two raw_pointer_buffers
     *
     * @param lhs
     *   Left argument
     * @param rhs
     *   Right argument
     */
    friend void swap(raw_pointer_buffer<T> &lhs, raw_pointer_buffer<T> &rhs) noexcept
    {
      using std::swap;

      swap(lhs.size_, rhs.size_);
      swap(lhs.owning_, rhs.owning_);
      swap(lhs.alloc_, rhs.alloc_);
      swap(lhs.data_, rhs.data_);
    }          

    /**
     * @brief Get underlying data pointer
     * 
     * @return Pointer to start of data
     */
    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return data_.get();
    }

    /**
     * @brief Get begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    /**
     * @brief Get end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + size_;
    }

    /**
     * @brief Get constant begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    /**
     * @brief Get constant end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + size_;
    }

    /**
     * @brief Get size of allocation
     * 
     * @return Size of allocation 
     */
    size_t size() const
    {
      return size_;
    }

    /**
     * @brief Set the data pointer in an object
     * 
     * @param data New value of data pointer
     */
    void SetData(T *const data) noexcept
    {
      data_.reset(data, [](auto ){});
    }   

    /**
     * @brief Allocate storage from container
     * 
     * @param size Size in bytes to allocate
     */
    __MATX_INLINE__ T* allocate(size_t size)
    {
      alloc_.allocate();
    }

    /**
     * @brief Get storage capacity of container
     * 
     * @return auto 
     */
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
    bool owning_;

    std::shared_ptr<T> data_;
    
    void ConfigureShared(T *ptr, [[maybe_unused]] size_t size) {
      if constexpr (std::is_const_v<T>) {
        if (!owning_) {
          data_ = std::shared_ptr<T>(ptr, [](auto){});
        }
        else {
          MATX_ASSERT_STR(false, matxInvalidParameter, "Cannot use an owning tensor type with const data");
        }
      }
      else {
        if (!owning_) {
          data_ = std::shared_ptr<T>(ptr, [](auto){});
        }
        else {
          data_ = std::shared_ptr<T>(ptr, [=](auto p) { alloc_.deallocate(reinterpret_cast<void*>(p), size); });
        }
      }   
    }
  };

  /**
   * @brief Storage class for smart pointers (unique and shared)
   * 
   * Used when an existing shared or unique_ptr object exists and MatX needs to take either shared or full
   * ownership
   * 
   * @tparam T Type to store in smart pointer
   */
  template <typename T>
  class smart_pointer_buffer
  {
  public:
    using value_type = T; ///< Type trait for value_type
    using iterator = T *; ///< Type trait for iterator value
    using citerator = T const *; ///< Type trait for const iterator value
    using matx_storage_container = bool; ///< Type trait to indicate this is a container

    /** 
     * @brief Default construct a smart_pointer_buffer. This should only be used when temporarily
     * creating an empty tensor for construction later.
    */
    smart_pointer_buffer<T>() {};

    /**
     * @brief Construct a new smart pointer buffer from an existing object
     * 
     * @param ptr Smart poiner object
     * @param size Size of allocation
     */
    smart_pointer_buffer<T>(T &&ptr, size_t size) : data_(std::forward<T>(ptr)), size_(size) {
      static_assert(is_smart_ptr_v<T>);
    }

    /** Swaps two smart_pointer_buffer
     *
     * Swaps members of two smart_pointer_buffers
     *
     * @param lhs
     *   Left argument
     * @param rhs
     *   Right argument
     */
    friend void swap(smart_pointer_buffer<T> &lhs, smart_pointer_buffer<T> &rhs) noexcept
    {
      using std::swap;

      swap(lhs.size_, rhs.size_);
      swap(lhs.data_, rhs.data_);
    }      

    /**
     * @brief Default const copy constructor
     * 
     */      
    smart_pointer_buffer& operator=(const smart_pointer_buffer &) = default;

    /**
     * @brief Default move assignment constructor
     * 
     */      
    smart_pointer_buffer& operator=(smart_pointer_buffer &&) = default;

    /**
     * @brief Move a smart pointer buffer object
     * 
     * @param rhs Object to move from
     */
    smart_pointer_buffer<T>(smart_pointer_buffer<T> &&rhs) {
      size_ = rhs.size_; 
      data_ = std::move(rhs.data_);
    }    

    /**
     * @brief Default destructor
     * 
     */  
    ~smart_pointer_buffer<T>() = default;

    /**
     * @brief Get underlying data pointer
     * 
     * @return Pointer to start of data
     */
    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return data_.get();
    }

    /**
     * @brief Get begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return data();
    }

    /**
     * @brief Get end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return data() + size_;
    }

    /**
     * @brief Get constant begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return data();
    }

    /**
     * @brief Get constant end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return data() + size_;
    }

    /**
     * @brief Get size of container in bytes
     * 
     * @return Size in bytes 
     */
    size_t size() const
    {
      return size_;
    }

    /**
     * @brief Set the data pointer in an object
     * 
     * @param data New value of data pointer
     */
    void SetData(T *const data) noexcept
    {
      data_.reset(data_, [](auto){});
    }

    /**
     * @brief Get storage capacity of container
     * 
     * @return auto 
     */
    auto capacity() const noexcept {
      return size_;
    }

    /**
     * @brief Allocate storage from container
     * 
     * @param size Size in bytes to allocate
     */
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
    using value_type = typename C::value_type; ///< Type trait for value_type
    using T = value_type; ///< Type trait for type
    using iterator = value_type*; ///< Type trait for iterator value
    using citerator = value_type const *; ///< Type trait for const iterator value
    using matx_storage = bool; ///< Type trait to indicate this is a storage type
    using container = C; ///< Storage container type

    /** Swaps two basic_storages
     *
     * Swaps members of two basic_storages
     *
     * @param lhs
     *   Left argument
     * @param rhs
     *   Right argument
     */
    friend void swap(basic_storage<C> &lhs, basic_storage<C> &rhs) noexcept
    {
      using std::swap;

      swap(lhs.container_, rhs.container_);
    }      

    /**
     * @brief Set the data pointer in an object
     * 
     * @param data New value of data pointer
     */
    void SetData(T *const data) noexcept
    {
      static_assert(is_matx_storage_container_v<C>, "Must use MatX storage container type if trying to set data pointer");
      
      container_.SetData(data);
    }

    /**
     * @brief Construct an empty storage container
     * 
     */
    __MATX_INLINE__ basic_storage()
    {
    }    

    /**
     * @brief Construct a storage container
     * 
     * @tparam C2 Unused
     * @param obj Storage object
     */
    template <typename C2, std::enable_if_t<!is_matx_storage_v<C2>, bool> = true>
    __MATX_INLINE__ basic_storage(C2 &&obj) : container_(std::forward<C2>(obj))
    {
    }

    /**
     * @brief Default copy constructor
     * 
     */
    __MATX_INLINE__ basic_storage(const basic_storage &) = default;

    /**
     * @brief Default move constructor
     * 
     */    
    __MATX_INLINE__ basic_storage(basic_storage &&) = default;

    /**
     * @brief Default copy assignment constructor
     * 
     */    
    __MATX_INLINE__ basic_storage& operator=(const basic_storage &) = default;

    /**
     * @brief Default desctructor
     * 
     */    
    __MATX_INLINE__ ~basic_storage() = default;

    /**
     * @brief Get underlying data pointer
     * 
     * @return Pointer to start of data
     */
    [[nodiscard]] __MATX_INLINE__ T *data() noexcept
    {
      return container_.data();
    }    

    /**
     * @brief Get begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator begin() noexcept
    {
      return container_.begin();
    }

    /**
     * @brief Get end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ iterator end() noexcept
    {
      return container_.end();
    }

    /**
     * @brief Get constant begin iterator
     * 
     * @return Beginning iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cbegin() const noexcept
    {
      return container_.cbegin();
    }

    /**
     * @brief Get constant end iterator
     * 
     * @return Ending iterator
     */
    [[nodiscard]] __MATX_INLINE__ citerator cend() const noexcept
    {
      return container_.cend();
    }

    /**
     * @brief Allocate storage from container
     * 
     * @param size Size in bytes to allocate
     */
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
    __MATX_INLINE__ auto use_count() const noexcept
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

/**
 * @brief Default storage type using ownership semantics and the default MatX allocator
 * 
 * @tparam T Type of pointer
 */
  template <typename T>
  using DefaultStorage = basic_storage<raw_pointer_buffer<T, matx_allocator<T>>>;
};
