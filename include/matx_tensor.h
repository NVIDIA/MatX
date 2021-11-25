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

#include <cinttypes>
#include <cstdint>
#include <atomic>
#include <iomanip>
#include <numeric>
#include <memory>
#include <type_traits>

#include "matx_allocator.h"
#include "matx_error.h"
#include "matx_storage.h"
#include "matx_tensor_impl.h"
#include "matx_utility_kernels.cuh"
#include "matx_tensor_utils.h"

static constexpr int MAX_TENSOR_DIM = 4;
static constexpr bool PRINT_ON_DEVICE = false;

// forward declare
namespace matx {
template <typename T, int RANK, typename Storage, typename Desc> class tensor_t;
} // namespace matx

/* Special values used to indicate properties of tensors */
namespace matx {
#ifdef INDEX_64_BIT
enum {
  matxKeepDim = LLONG_MAX,
  matxDropDim = LLONG_MAX - 1,
  matxEnd = LLONG_MAX - 2,
  matxKeepStride = LLONG_MAX - 3
};
#else
enum {
  matxKeepDim = INT_MAX,
  matxDropDim = INT_MAX - 1,
  matxEnd = INT_MAX - 2,
  matxKeepStride = INT_MAX - 3
};
#endif


/**
 * View of an underlying tensor data object
 *
 * The tensor_t class provides multiple ways to view the data inside of a
 * matxTensorData_t object. Views do not modify the underlying data; they simply
 * present a different way to look at the data. This includes where the data
 * begins and ends, the stride, the rank, etc. Views are very lightweight, and
 * any number of views can be generated from the same data object. Since views
 * represent different ways of looking at the same data, it is the
 * responsibility of the user to ensure that proper synchronization is done when
 * using multiple views on the same data. Failure to do so can result in race
 * conditions on the device or host.
 */
template <typename T, 
          int RANK, 
          typename Storage = DefaultStorage<T>, 
          typename Desc = DefaultDescriptor<RANK>> 
class tensor_t : public tensor_impl_t<T,RANK,Desc> {
public:
  // Type specifier for reflection on class
  using type = T; // TODO is this necessary
  using scalar_type = T;

  // Type specifier for signaling this is a matx operation or tensor view
  using matxop = bool;
  using tensor_view = bool;
  using storage_type = Storage;
  using desc_type = Desc;


  // /**
  //  * @brief Conversion to basic tensor implementation
  //  * 
  //  */
  // operator tensor_t<T, RANK, Storage, Desc>() {
  //   return tensor_t<T, RANK, Storage, Desc>{ldata_, shape_, s_};
  // }

  tensor_t() :
    tensor_impl_t<T, RANK, Desc>{}, 
    storage_{typename Storage::container{sizeof(T)}}
  {
    this->SetLocalData(storage_.data());
    //static_assert(RANK == 0, "Default tensor constructor only works for rank-0 tensors.");
  }

  __MATX_HOST__ tensor_t(tensor_t const &rhs) noexcept
      : tensor_impl_t<T, RANK, Desc>{rhs.ldata_, rhs.desc_}, storage_(rhs.storage_)
      { }

  __MATX_HOST__ tensor_t(tensor_t &&rhs) noexcept
      : tensor_impl_t<T, RANK, Desc>{rhs.ldata_, std::move(rhs.desc_)}, storage_(std::move(rhs.storage_))
  { }

  /** Perform a shallow copy of a tensor view
   *
   * Alternative to operator= since it's used for lazy evaluation. This function
   * is used to perform a shallow copy of a tensor view where the data pointer
   * points to the same location as the right hand side's data. *
   *
   * @param rhs
   *   Tensor to copy from
   */
  __MATX_HOST__ void Shallow(const tensor_t<T, RANK, Storage, Desc> &rhs) noexcept
  {
    this->ldata_ = rhs.ldata_;
    storage_ = rhs.storage_;
    this->desc_ = rhs.desc_;
  }

  __MATX_INLINE__  ~tensor_t() = default;


  template <typename S2 = Storage, typename D2 = Desc, 
            std::enable_if_t<is_matx_storage_v<typename remove_cvref<S2>::type> && is_matx_descriptor_v<typename remove_cvref<D2>::type>, bool> = true>
  tensor_t(S2 &&s, D2 &&desc) :
    tensor_impl_t<T, RANK, Desc>{std::forward<D2>(desc)},
    storage_{std::forward<S2>(s)}
  {
    this->SetLocalData(storage_.data());
    // static_assert(std::is_same_v<Storage, DefaultStorage<RANK>>, 
    //   "Must use default storage if not providing")
  } 

  /**
   * @brief Construct a new tensor t object. Used to copy an existing storage object for proper reference counting
   * 
   * @param s 
   * @param desc 
   * @param ldata 
   */
  tensor_t(Storage s, Desc &&desc, T* ldata) :
    tensor_impl_t<T, RANK, Desc>{std::forward<Desc>(desc)},
    storage_{std::move(s)}
  {
    this->SetLocalData(ldata);
    // static_assert(std::is_same_v<Storage, DefaultStorage<RANK>>, 
    //   "Must use default storage if not providing")
  }  


  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param desc
   *   Tensor descriptor
   */
  template <typename D2 = Desc, typename = 
    typename std::enable_if_t<is_matx_descriptor_v<D2>>>
  __MATX_INLINE__ tensor_t(Desc &&desc) :
    tensor_impl_t<T, RANK, Desc>{std::forward<Desc>(desc)},
    storage_{typename Storage::container{this->desc_.TotalSize()*sizeof(T)}}
  {
    this->SetLocalData(storage_.data());
  }
  

  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param shape
   *   Tensor shape
   */
  __MATX_INLINE__ tensor_t(const typename Desc::shape_type (&shape)[RANK]) :
    tensor_impl_t<T, RANK, Desc>(shape),
    storage_{typename Storage::container{this->desc_.TotalSize()*sizeof(T)}}
  {    
    this->SetLocalData(storage_.data());
  }
   

  // Lazy operators
  /**
   * Lazy assignment operator=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, op);
  }
  /**
   * Lazy assignment operator=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, op_base);
  }

  /**
   * Lazy assignment operator+=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this + op);
  }

  /**
   * Lazy assignment operator+=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this + op_base);
  }

  /**
   * Lazy assignment operator-=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this - op);
  }

  /**
   * Lazy assignment operator-=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this - op_base);
  }

  /**
   * Lazy assignment operator*=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this * op);
  }

  /**
   * Lazy assignment operator*=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this * op_base);
  }

  /**
   * Lazy assignment operator/=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this / op);
  }

  /**
   * Lazy assignment operator/=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this / op_base);
  }

  /**
   * Lazy assignment operator<<=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this << op);
  }

  /**
   * Lazy assignment operator<<=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this << op_base);
  }

  /**
   * Lazy assignment operator>>=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this >> op);
  }

  /**
   * Lazy assignment operator>>=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this >> op_base);
  }

  /**
   * Lazy assignment operator|=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this | op);
  }

  /**
   * Lazy assignment operator|=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this | op_base);
  }

  /**
   * Lazy assignment operator&=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this & op);
  }

  /**
   * Lazy assignment operator&=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this & op_base);
  }

  /**
   * Lazy assignment operator^=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this ^ op);
  }

  /**
   * Lazy assignment operator^=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
    return set(*this, *this ^ op_base);
  }

  /**
   * Lazy assignment operator%=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Tensor view source
   *
   * @returns set object containing the destination view and source object
   *
   */
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const tensor_t<T, RANK, Storage, Desc> &op)
  {
      return set(*this, *this % op);
  }

  /**
   * Lazy assignment operator%=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const T2 &op)
  {
    const typename base_type<T2>::type &op_base = op;
      return set(*this, *this % op_base);
  }

  /**
   * Get a view of the tensor from the underlying data using a custom shape
   *
   * Returns a view based on the shape passed in. Both the rank and the
   * dimensions can be increased or decreased from the original data object as
   * long as they fit within the bounds of the memory allocation. This function
   * only allows a contiguous view of memory, regardless of the shape passed in.
   * For example, if the original shape is {8, 2} and a view of {2, 1} is
   * requested, the data in the new view would be the last two elements of the
   * last dimension of the original data.
   *
   * The function is similar to MATLAB and Python's reshape(), except it does
   * NOT make a copy of the data, whereas those languages may, depending on the
   * context. It is up to the user to understand any existing views on the
   * underlying data that may conflict with other views.
   *
   * While this function is similar to Slice(), it does not allow slicing a
   * particular start and end point as slicing does, and slicing also does not
   * allow increasing the rank of a tensor as View(shape) does.
   *
   * Note that the type of the data type of the tensor can also change from the
   * original data. This may be useful in situations where a union of data types
   * could be used in different ways. For example, a complex<float> could be
   * reshaped into a float tensor that has twice as many elements, and
   * operations can be done on floats instead of complex types.
   *
   * @tparam M
   *   New type of tensor
   * @tparam R
   *   New rank of tensor
   * @param shape
   *   New shape of tensor
   *
   * @return
   *    A view of the data with the appropriate strides and dimensions set
   */
  template <typename M = T, int R = RANK, typename Shape>
  __MATX_INLINE__ auto View(Shape &&shape)
  {
    // Ensure new shape's total size is not larger than the original
    MATX_ASSERT_STR(
        sizeof(M) * shape.TotalSize() <= storage_.Bytes(), matxInvalidSize,
        "Total size of new tensor must not be larger than the original");

    // Copy descriptor and call ctor with shape
    Desc new_desc{std::forward<Shape>(shape)};  
    return tensor_t<M, R, Storage, Desc>{storage_, std::move(new_desc), this->ldata_};
  }

  template <typename ShapeIntType, int NRANK>
  __MATX_INLINE__ auto View(const ShapeIntType (&shape)[NRANK])
  {
    // Change this to not rely on index_t
    std::array<index_t, NRANK> tshape;
    std::move(std::begin(shape), std::end(shape), tshape.begin()); 

    DefaultDescriptor<tshape.size()> desc{std::move(tshape)};   
    return tensor_t<T, NRANK, Storage, decltype(desc)>{storage_, std::move(desc), this->ldata_};
  }  

  /**
   * @brief Make a copy of a tensor and maintain all refcounts
   * 
   * @tparam M 
   * @tparam R 
   * @return __MATX_INLINE__ 
   */
  __MATX_INLINE__ auto View()
  {
    return *this;
  }

  /**
   * Prefetch the data asynchronously from the host to the device.
   *
   * All copies are done asynchronously in a stream. The order of the copy
   * is predictable within work in the same stream, but not when the transfer
   * will occur.
   *
   * @param stream
   *   The CUDA stream to prefetch within
   */
  __MATX_INLINE__ void PrefetchDevice(cudaStream_t const stream) const noexcept
  {
    int dev;
    cudaGetDevice(&dev);
    cudaMemPrefetchAsync(this->ldata_, this->desc_.TotalSize() * sizeof(T), dev, stream);
  }

  /**
   * Prefetch the data asynchronously from the device to the host.
   *
   * All copies are done asynchronously in a stream. The order of the copy
   * is predictable within work in the same stream, but not when the transfer
   * will occur.
   *
   * @param stream
   *   The CUDA stream to prefetch within
   */
  __MATX_INLINE__ void PrefetchHost(cudaStream_t const stream) const noexcept
  {
    cudaMemPrefetchAsync(this->ldata_, this->desc_.TotalSize() * sizeof(T), cudaCpuDeviceId,
                         stream);
  }

  /**
   * Create a view of only real-valued components of a complex array
   *
   * Only available on complex data types.
   *
   * @returns tensor view of only real-valued components
   *
   */
  template <typename U = T>
  __MATX_INLINE__ auto RealView() const noexcept
  {
    static_assert(is_complex_v<T>, "RealView() only works with complex types");

    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(this->ldata_);
    std::array<typename Desc::stride_type, RANK> strides;
    
#pragma unroll
    for (int i = 0; i < RANK; i++) {
      strides[i] = this->desc_.Stride(i);
    }

    if constexpr (RANK > 0) {
#pragma unroll
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    // Copy descriptor and call ctor with shape
    Desc new_desc{this->desc_.Shape(), std::move(strides)};  
    return tensor_t<Type, RANK, Storage, Desc>{storage_, std::move(new_desc), data};
  }

  /**
   * @brief Return the storage container from the tensor
   * 
   * @return storage container 
   */
  __MATX_INLINE__ auto GetStorage() noexcept {
    return storage_;
  }

  /**
   * Create a view of only imaginary-valued components of a complex array
   *
   * Only available on complex data types.
   *
   * @returns tensor view of only imaginary-valued components
   *
   */
  template <typename U = T>
  __MATX_INLINE__ auto ImagView() const noexcept
  {
    static_assert(is_complex_v<T>, "ImagView() only works with complex types");

    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(this->ldata_) + 1;
    std::array<typename Desc::stride_type, RANK> strides;
#pragma unroll
    for (int i = 0; i < RANK; i++) {
      strides[i] = this->Stride(i);
    }

    if constexpr (RANK > 0) {
#pragma unroll
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    Desc new_desc{this->desc_.Shape(), std::move(strides)};  
    return tensor_t<Type, RANK, Storage, Desc>{storage_, std::move(new_desc), data};
  }

  /**
   * Permute the dimensions of a tensor
   *
   * Accepts any order of permutation. Number of dimensions must match RANK of
   * tensor
   *
   * @tparam M
   *   Rank of tensor to permute. Should not be used directly
   *
   * @param dims
   *   Dimensions of tensor
   *
   * @returns tensor view of only imaginary-valued components
   *
   */
  __MATX_INLINE__ tensor_t Permute(const uint32_t (&dims)[RANK]) const
  {
    static_assert(RANK >= 2, "Only tensors of rank 2 and higher can be permuted.");
    std::array<typename Desc::shape_type, RANK> n;
    std::array<typename Desc::stride_type, RANK> s;
    [[maybe_unused]] bool done[RANK] = {0};

#pragma unroll
    for (int i = 0; i < RANK; i++) {
      int d = dims[i];
      MATX_ASSERT_STR(d < RANK, matxInvalidDim,
                      "Index to permute is larger than tensor rank");
      MATX_ASSERT_STR(done[d] == false, matxInvalidParameter,
                      "Cannot list the same dimension to permute twice");
      done[d] = true;
      n[i] = this->Size(d);
      s[i] = this->Stride(d);
    }

    Desc new_desc{std::move(n), std::move(s)};  
    return tensor_t<T, RANK, Storage, Desc>{storage_, std::move(new_desc), this->ldata_};
  }

  /**
   * Permute the last two dimensions of a matrix
   *
   * Utility function to permute the last two dimensions of a tensor. This is
   * useful in the numerous operations that take a permuted matrix as input, but
   * we don't want to permute the inner dimensions of a larger tensor.
   *
   * @tparam M
   *  Rank of tensor
   *
   * @param dims
   *  Dimensions of tensors
   *
   * @returns tensor view with last two dims permuted
   *
   */
  __MATX_INLINE__ auto PermuteMatrix() const
  {
    static_assert(RANK >= 2, "Only tensors of rank 2 and higher can be permuted.");
    uint32_t tdims[RANK];
    std::iota(std::begin(tdims), std::end(tdims), 0);
    std::swap(tdims[RANK - 2], tdims[RANK - 1]);
    return Permute(tdims);
  }

  /**
   * Get the underlying local data pointer from the view
   *
   * @returns Underlying data pointer of type T
   *
   */
  __MATX_HOST__ __MATX_INLINE__ T *Data() const noexcept { return this->ldata_; }

  /**
   * Set the underlying data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @param data
   *   Data pointer to set
   * @param refcnt
   *   Optional reference count for new memory or nullptr if not tracked
   *
   */
  template <typename ShapeType, 
      std::enable_if_t<!std::is_pointer_v<typename remove_cvref<ShapeType>::type>, bool> = true>  
  __MATX_HOST__ __MATX_INLINE__ void
  Reset(T *const data, ShapeType &&shape) noexcept
  {
    this->desc_.InitFromShape(std::forward<ShapeType>(shape));
    storage_.SetData(data);
    this->ldata_ = data;
  }  

  /**
   * Set the underlying data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @param data
   *   Data pointer to set
   * @param refcnt
   *   Optional reference count for new memory or nullptr if not tracked
   *
   */
  __MATX_HOST__ __MATX_INLINE__ void
  Reset(T *const data) noexcept
  {
    storage_.SetData(data);
    this->ldata_ = data;
  }

  /**
   * Set the underlying data and local data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @param data
   *   Allocated data pointer
   * @param ldata
   *   Local data pointer offset into allocated
   *
   */
  __MATX_HOST__ __MATX_INLINE__ void
  Reset(T *const data, T *const ldata) noexcept
  {
    storage_.SetData(data);
    this->ldata_ = ldata;
  }


  /**
   * Get the stride of a single dimension of the tensor
   *
   * @param dim
   *   Desired dimension
   *
   * @returns Stride (in elements) in dimension
   *
   */
  __MATX_INLINE__ __MATX_HOST__ typename Desc::stride_type Stride(uint32_t dim) const
  {
    static_assert(RANK >= 1, "Indexed strides are only available on tensors of rank 1 or higher.");
    return this->desc_.Stride(dim);
  }

  /**
   * Get the reference count
   *
   * @returns Reference count or 0 if not tracked
   *
   */
  __MATX_INLINE__ __MATX_HOST__ auto GetRefCount() const noexcept
  {
    return storage_.use_count();
  }  


  /**
   * Create an overlapping tensor view
   *
   * Creates and overlapping tensor view where an existing tensor can be
   * repeated into a higher rank with overlapping elements. For example, the
   * following 1D tensor [1 2 3 4 5] could be cloned into a 2d tensor with a
   * window size of 2 and overlap of 1, resulting in:
   *
   * [1 2
   *  2 3
   *  3 4
   *  4 5]
   *
   * Currently this only works on 1D tensors going to 2D, but may be expanded
   * for higher dimensions in the future. Note that if the window size does not
   * divide evenly into the existing column dimension, the view may chop off the
   * end of the data to make the tensor rectangular.
   *
   * @param windows
   *   Window size (columns in output)
   * @param strides
   *   Strides between data elements
   *
   * @returns Overlapping view of data
   *
   */
  __MATX_INLINE__ auto
  OverlapView(std::initializer_list<typename Desc::shape_type> const &windows,
              std::initializer_list<typename Desc::stride_type> const &strides) const
  {
    static_assert(RANK == 1, "Overlapped views only supported on 1D tensors.");

    std::array<typename Desc::shape_type, RANK+1> n;
    std::array<typename Desc::stride_type, RANK+1> s;

    // This only works for 1D tensors going to 2D at the moment. Generalize to
    // higher dims later
    typename Desc::stride_type window_size = *(windows.begin());
    typename Desc::stride_type stride_size = *(strides.begin());

    MATX_ASSERT(stride_size < window_size, matxInvalidSize);
    MATX_ASSERT(stride_size > 0, matxInvalidSize);

    // Figure out the actual length of the signal we can use. It might be
    // shorter than the original tensor if the window/stride doesn't line up
    // properly to make a rectangular matrix.
    typename Desc::shape_type adj_el = this->desc_.Size(0) - window_size;
    while ((adj_el % stride_size) != 0) {
      adj_el--;
    }

    n[1] = window_size;
    s[1] = 1;
    n[0] = adj_el / stride_size + 1;
    s[0] = stride_size;

    tensor_desc_t<decltype(n), decltype(s), RANK+1> new_desc{std::move(n), std::move(s)}; 
    printf("%lld %lld\n", n[0], n[1]);
    printf("%lld %lld\n", s[0], s[1]);
    return tensor_t<T, RANK + 1, Storage, decltype(new_desc)>{storage_, std::move(new_desc), this->ldata_};    
  }

  /**
   * Clone a tensor into a higher-dimension tensor
   *
   * Clone() allows a copy-less method to clone data into a higher dimension
   * tensor. The underlying data does not grow or copy, but instead the indices
   * of the higher-ranked tensor access the original data potentially multiple
   * times. Clone is similar to MATLAB's repmat() function where it's desired
   * to take a tensor of a lower dimension and apply an operation with it to a
   * tensor in a higher dimension by broadcasting the values.
   *
   * For example, in a
   * rank=2 tensor that's MxN, and a rank=1 tensor that's 1xN, Clone() can take
   * the rank=1 tensor and broadcast to an MxN rank=2 tensor, and operations
   * such as the Hadamard product can be performed. In this example, the final
   * operation will benefit heavily from device caching since the same 1xN
   * rank=1 tensor will be accessed M times.
   *
   * @param clones
   *   List of sizes of each dimension to clone. Parameter length must match
   * rank of tensor. A special sentinel value of matxKeepDim should be used when
   * the dimension from the original tensor is to be kept.
   *
   * @returns Cloned view representing the higher-dimension tensor
   *
   */
  template <int N>
  __MATX_INLINE__ auto Clone(const typename Desc::shape_type (&clones)[N]) const
  {
    static_assert(N <= 4 && N > RANK, "Clone() must increase tensor to a higher rank <= 4.");
    std::array<typename Desc::shape_type, N> n;
    std::array<typename Desc::stride_type, N> s;    

    int d = 0;

#pragma unroll
    for (int i = 0; i < N; i++) {
      typename Desc::shape_type size = clones[i];

      if (size == matxKeepDim) {
        n[i] = this->desc_.Size(d);
        if constexpr (RANK == 0) {
          s[i] = 1;
        }
        else {
          s[i] = this->desc_.Stride(d);
        }
        d++;
      }
      else {
        n[i] = size;
        s[i] = 0;
      }
    }
    MATX_ASSERT_STR(d == RANK, matxInvalidDim,
                    "Must keep as many dimension as the original tensor has");
    tensor_desc_t<decltype(n), decltype(s), N> new_desc{std::move(n), std::move(s)};  
    return tensor_t<T, N, Storage, decltype(new_desc)>{storage_, std::move(new_desc), this->ldata_};
  }

  /**
   * Rank-0 initializer list setting
   *
   * @param val
   *   0 initializer list value
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<M == 0, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void SetVals(T const &val) noexcept
  {
    this->operator()() = val;
  }

  /**
   * Rank-1 non-complex or rank-0 initializer list setting
   *
   * @param vals
   *   1D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 1) ||
                                               (is_cuda_complex_v<T> && M == 0),
                                           bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void SetVals(const std::initializer_list<T> &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      if constexpr (is_cuda_complex_v<T>) {
        typename T::value_type real = (vals.begin() + i)->real();
        typename T::value_type imag = (vals.begin() + i + 1)->real();
        this->operator()() = {real, imag};
      }
      else {
        this->operator()(static_cast<typename Desc::shape_type>(i)) = *(vals.begin() + i);
      }
    }
  }

  /**
   * Rank-2 non-complex or rank-1 initializer list setting
   *
   * @param vals
   *   1D/2D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 2) ||
                                               (is_cuda_complex_v<T> && M == 1),
                                           bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<T>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        if constexpr (is_cuda_complex_v<T>) {
          typename T::value_type real =
              ((vals.begin() + i)->begin() + j)->real();
          typename T::value_type imag =
              ((vals.begin() + i)->begin() + j + 1)->real();
          this->operator()(i) = {real, imag};
          j++;
        }
        else {
          this->operator()(static_cast<typename Desc::shape_type>(i), static_cast<typename Desc::shape_type>(j)) = *((vals.begin() + i)->begin() + j);
        }
      }
    }
  }

  /**
   * Rank-3 non-complex or rank-2 complex initializer list setting
   *
   * @param vals
   *   3D/2D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 3) ||
                                               (is_cuda_complex_v<T> && M == 2),
                                           bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<T>>>
              vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          if constexpr (is_cuda_complex_v<T>) {
            typename T::value_type real =
                (((vals.begin() + i)->begin() + j)->begin() + k)->real();
            typename T::value_type imag =
                (((vals.begin() + i)->begin() + j)->begin() + k + 1)->real();
            this->operator()(i, j) = {real, imag};
            k++;
          }
          else {
            this->operator()(static_cast<typename Desc::shape_type>(i), static_cast<typename Desc::shape_type>(j), static_cast<typename Desc::shape_type>(k)) =
                *(((vals.begin() + i)->begin() + j)->begin() + k);
          }
        }
      }
    }
  }

  /**
   * Rank-4 non-complex or rank-3 complex initializer list setting
   *
   * @param vals
   *   3D/4D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK, std::enable_if_t<(!is_cuda_complex_v<T> && M == 4) ||
                                               (is_cuda_complex_v<T> && M == 3),
                                           bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin + k)->size();
               l++) {
            if constexpr (is_cuda_complex_v<T>) {
              typename T::value_type real =
                  ((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                   l)
                      ->real();
              typename T::value_type imag =
                  ((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                   l + 1)
                      ->real();
              this->operator()(i, j, k) = {real, imag};
              l++;
            }
            else {
              this->operator()(static_cast<typename Desc::shape_type>(i), static_cast<typename Desc::shape_type>(j), static_cast<typename Desc::shape_type>(k), static_cast<typename Desc::shape_type>(l)) =
                  *((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l);
            }
          }
        }
      }
    }
  }

  /**
   * Rank-4 complex initializer list setting
   *
   * @param vals
   *   4D initializer list of values
   *
   * @returns reference to view
   *
   */
  template <int M = RANK,
            std::enable_if_t<is_cuda_complex_v<T> && M == 4, bool> = true>
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>>
              &vals) noexcept
  {
    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin + k)->size();
               l++) {
            for (size_t m = 0;
                 m < ((((vals.begin() + i)->begin() + j)->begin + k)->begin + l)
                         ->size();
                 m++) {
              typename T::value_type real =
                  (((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l)
                       ->begin() +
                   m)
                      ->real();
              typename T::value_type imag =
                  (((((vals.begin() + i)->begin() + j)->begin() + k)->begin() +
                    l)
                       ->begin() +
                   m + 1)
                      ->real();
              this->operator()(i, j, k, l) = {real, imag};
              m++;
            }
          }
        }
      }
    }
  }

  /**
   * Slice a tensor either within the same dimension or to a lower dimension
   *
   * Slice() allows a copy-less method to extract a subset of data from one or
   * more dimensions of a tensor. This includes completely dropping an unwanted
   * dimension, or simply taking a piece of a wanted dimension. Slice() is very
   * similar to indexing operations in both Python and MATLAB.
   *
   * @param firsts
   *   List of starting index into each dimension. Indexing is 0-based
   *
   * @param ends
   *   List of ending index into each dimension. Indexing is 0-based
   *   Two special sentinel values can be used:
   *     1) matxEnd is used to indicate the end of that particular
   *        dimension without specifying the size. This is similar to "end" in
   *        MATLAB and leaving off an end in Python "a[1:]"
   *
   *     2) matxDropDim is used to slice (drop) a dimension entirely. This
   * results in a tensor with a smaller rank than the original
   *
   * @param strides
   *   List of strides for each dimension.
   *   A special sentinel value of matxKeepStride is used to keep the existing
   * stride of the dimension
   *
   * @returns Sliced view of tensor
   *
   */
  template <int N = RANK>
  __MATX_INLINE__ auto Slice([[maybe_unused]] const typename Desc::shape_type (&firsts)[RANK],
                              [[maybe_unused]] const typename Desc::shape_type (&ends)[RANK],
                              [[maybe_unused]] const typename Desc::stride_type (&strides)[RANK]) const
  {
    static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");
    std::array<typename Desc::shape_type, N> n = {};
    std::array<typename Desc::stride_type, N> s = {};

    T *data = this->ldata_;
    int d = 0;
    bool def_stride = (strides[0] == -1);

#pragma unroll
    for (int i = 0; i < RANK; i++) {
      typename Desc::shape_type first = firsts[i];
      typename Desc::shape_type end = ends[i];

      MATX_ASSERT_STR(first < end, matxInvalidSize, "Slice must be at least one element long");

      [[maybe_unused]] typename Desc::stride_type stride_mult = (def_stride || strides[i] == matxKeepStride)
                                ? 1
                                : strides[i]; // custom stride

      MATX_ASSERT_STR(first < end, matxInvalidParameter,
                      "Starting slice must be less than end slice");
      MATX_ASSERT_STR(first < this->desc_.Size(i), matxInvalidParameter,
                      "Requested slice start index out of bounds");

      // offset by first
      data += first * Stride(i);

      if (end != matxDropDim) {
        if (end == matxEnd) {
          n[d] = this->Size(i) - first;
        }
        else {
          n[d] = end - first;
        }

        // New length is shorter if we have a non-1 stride
        n[d] = static_cast<typename Desc::shape_type>(std::ceil(
            static_cast<double>(n[d]) / static_cast<double>(stride_mult)));

        s[d] = Stride(i) * stride_mult;
        d++;
      }
    }

    MATX_ASSERT_STR(d == N, matxInvalidDim,
                    "Number of indices must match the target rank to slice to");

    tensor_desc_t<decltype(n), decltype(s), N> new_desc{std::move(n), std::move(s)};  
    return tensor_t<T, N, Storage, decltype(new_desc)>{storage_, std::move(new_desc), data};
  }

  /**
   * Slice a tensor either within the same dimension or to a lower dimension
   *
   * Slice() allows a copy-less method to extract a subset of data from one or
   * more dimensions of a tensor. This includes completely dropping an unwanted
   * dimension, or simply taking a piece of a wanted dimension. Slice() is very
   * similar to indexing operations in both Python and MATLAB.
   *
   * @param firsts
   *   List of starting index into each dimension. Indexing is 0-based
   *
   * @param ends
   *   List of ending index into each dimension. Indexing is 0-based
   *   Two special sentinel values can be used:
   *     1) matxEnd is used to indicate the end of that particular
   *        dimension without specifying the size. This is similar to "end" in
   *        MATLAB and leaving off an end in Python "a[1:]"
   *
   *     2) matxDropDim is used to slice (drop) a dimension entirely. This
   * results in a tensor with a smaller rank than the original
   *
   * @returns Sliced view of tensor
   *
   */
  template <int N = RANK>
  __MATX_INLINE__ auto Slice(const typename Desc::shape_type (&firsts)[RANK],
                              const typename Desc::shape_type (&ends)[RANK]) const
  {
    static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");
    const typename Desc::stride_type strides[RANK] = {-1};
    return Slice<N>(firsts, ends, strides);
  }


  /**
   * Print a value
   *
   * Type-agnostic function to print a value to stdout
   *
   * @param val
   */
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ void PrintVal(const T &val) const noexcept
  {
    if constexpr (is_complex_v<T>) {
      printf("%.4f%+.4fj ", static_cast<float>(val.real()),
             static_cast<float>(val.imag()));
    }
    else if constexpr (is_matx_half_v<T> || is_half_v<T>) {
      printf("%.4f ", static_cast<float>(val));
    }
    else if constexpr (std::is_floating_point_v<T>) {
      printf("%.4f ", val);
    }
    else if constexpr (std::is_same_v<T, long long int>) {
      printf("%lld ", val);
    }
    else if constexpr (std::is_same_v<T, int64_t>) {
      printf("%" PRId64 " ", val);
    }
    else if constexpr (std::is_same_v<T, int32_t>) {
      printf("%" PRId32 " ", val);
    }
    else if constexpr (std::is_same_v<T, int16_t>) {
      printf("%" PRId16 " ", val);
    }
    else if constexpr (std::is_same_v<T, int8_t>) {
      printf("%" PRId8 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint64_t>) {
      printf("%" PRIu64 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint32_t>) {
      printf("%" PRIu32 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint16_t>) {
      printf("%" PRIu16 " ", val);
    }
    else if constexpr (std::is_same_v<T, uint8_t>) {
      printf("%" PRIu8 " ", val);
    }
  }

  /**
   * Print a tensor
   *
   * Type-agnostic function to print a tensor to stdout
   *
   */
  template <typename ... Args>
  __MATX_HOST__ __MATX_DEVICE__ void InternalPrint(Args ...dims) const noexcept
  {
    MATX_STATIC_ASSERT(RANK == sizeof...(Args), "Number of dimensions to print must match tensor rank");

    if constexpr (sizeof...(Args) == 0) {
      PrintVal(this->operator()());
      printf("\n");
    }
    else if constexpr (sizeof...(Args) == 1) {
      auto& k = pp_get<0>(dims...);
      for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(0) : k); _k++) {
        printf("%06lld: ", _k);
        PrintVal(this->operator()(_k));
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 2) {
      auto& k = pp_get<0>(dims...);
      auto& l = pp_get<1>(dims...);
      for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(0) : k); _k++) {
        for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(1) : l); _l++) {
          if (_l == 0)
            printf("%06lld: ", _k);

          PrintVal(this->operator()(_k, _l));
        }
        printf("\n");
      }
    }
    else if constexpr (sizeof...(Args) == 3) {
      auto& j = pp_get<0>(dims...);
      auto& k = pp_get<1>(dims...);
      auto& l = pp_get<2>(dims...);
      for (typename Desc::shape_type _j = 0; _j < ((j == 0) ? this->Size(0) : j); _j++) {
        printf("[%06lld,:,:]\n", _j);
        for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(1) : k); _k++) {
          for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(2) : l); _l++) {
            if (_l == 0)
              printf("%06lld: ", _k);

            PrintVal(this->operator()(_j, _k, _l));
          }
          printf("\n");
        }
        printf("\n");
      }      
    }
    else if constexpr (sizeof...(Args) == 4) {
      auto& i = pp_get<0>(dims...);
      auto& j = pp_get<1>(dims...);
      auto& k = pp_get<2>(dims...);
      auto& l = pp_get<3>(dims...); 
      for (typename Desc::shape_type _i = 0; _i < ((i == 0) ? this->Size(0) : i); _i++) {
        for (typename Desc::shape_type _j = 0; _j < ((j == 0) ? this->Size(1) : j); _j++) {
          printf("[%06lld,%06lld,:,:]\n", _i, _j);
          for (typename Desc::shape_type _k = 0; _k < ((k == 0) ? this->Size(2) : k); _k++) {
            for (typename Desc::shape_type _l = 0; _l < ((l == 0) ? this->Size(3) : l); _l++) {
              if (_l == 0)
                printf("%06lld: ", _k);

              PrintVal(this->operator()(_i, _j, _k, _l));
            }
            printf("\n");
          }
          printf("\n");
        }
      }
    }
  }  

  /**
   * Print a tensor
   *
   * Type-agnostic function to print a tensor to stdout
   *
   */
  template <typename ... Args>
  void Print(Args ...dims) const
  {
#ifdef __CUDACC__    
    auto kind = GetPointerKind(this->ldata_);
    cudaDeviceSynchronize();
    if (HostPrintable(kind)) {
      InternalPrint(dims...);
    }
    else if (DevicePrintable(kind)) {
      if (PRINT_ON_DEVICE) {
        PrintKernel<<<1, 1>>>(*this, dims...);
      }
      else {
        tensor_t<T, RANK, Storage, Desc> tmpv(this->shape_, (const typename Desc::stride_type(&)[RANK])this->s_);
        cudaMemcpy(tmpv.Data(), this->Data(), tmpv.Bytes(),
                   cudaMemcpyDeviceToHost);
        tmpv.Print(dims...);
      }
    }
#else
    InternalPrint(dims...);
#endif    
  }

  /**
   * @brief Returns an N-D coordinate as an array corresponding to the absolute index abs
   * 
   * @param abs Absolute index
   * @return std::array of indices 
   */
  __MATX_INLINE__ auto GetIdxFromAbs(typename Desc::shape_type abs) {
    std::array<typename Desc::shape_type, RANK> indices;
    std::array<typename Desc::shape_type, RANK> sh = this->desc_.Shape();
    
    for (int idx = 0; idx < RANK; idx++) {
      if (idx == RANK-1) {
        indices[RANK-1] = abs;
      }
      else {
        typename Desc::stride_type prod = std::accumulate(sh.data() + idx + 1, sh.data() + RANK, 1, std::multiplies<typename Desc::stride_type>());
        indices[idx] = abs / prod;
        abs -= prod * indices[idx];
      }
    }

    return indices;
  }

private:
  Storage storage_;
};

template <typename Storage, typename Desc>
using tensor_impl_adv_t = tensor_t<typename Storage::T, Desc::Rank(), Storage, Desc>;

} // end namespace matx
