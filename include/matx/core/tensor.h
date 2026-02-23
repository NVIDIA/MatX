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
#include <cuda/std/numeric>

#include "matx/core/allocator.h"
#include "matx/core/error.h"
#include "matx/core/storage.h"
#include "matx/core/tensor_impl.h"
#include "matx/core/tensor_utils.h"
#include "matx/core/dlpack.h"
#include "matx/core/tie.h"
#include "matx/kernels/utility.cuh"



// forward declare
namespace matx {
template <typename T, int RANK, typename Desc> class tensor_t;
} // namespace matx

/* Special values used to indicate properties of tensors */
namespace matx {


/**
 * View of an underlying tensor data object
 *
 * Tensor views do not modify the underlying data; they simply
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
          typename Desc = DefaultDescriptor<RANK>>
class tensor_t : public detail::tensor_impl_t<T,RANK,Desc> {
public:
  // Type specifier for reflection on class
  using type = T; ///< Type of traits
  using value_type = T; ///< Type of traits
  // Type specifier for signaling this is a matx operation or tensor view
  using matxop = bool; ///< Indicate this is a MatX operator
  using matxoplvalue = bool; ///< Indicate this is a MatX operator that can be on the lhs of an equation
  using tensor_view = bool; ///< Indicate this is a MatX tensor view
  using tensor_t_type = bool; ///< This is a tensor_t (not a tensor_impl_t)
  using shape_type = typename Desc::shape_type;
  using stride_type = typename Desc::stride_type;
  using shape_container = typename Desc::shape_container;
  using stride_container = typename Desc::stride_container;
  using desc_type = Desc; ///< Descriptor type trait
  using self_type = tensor_t<T, RANK, Desc>;

  /**
   * @brief Construct a new 0-D tensor t object
   *
   */
  tensor_t() : detail::tensor_impl_t<T, RANK, Desc>{}
  {
    this->SetLocalData(nullptr);
  }

  /**
   * @brief Copy constructor
   *
   * @param rhs Object to copy from
   */
  __MATX_HOST__ tensor_t(tensor_t const &rhs) noexcept
      : detail::tensor_impl_t<T, RANK, Desc>{rhs.Data(), rhs.desc_}, storage_(rhs.storage_)
      { }

  /**
   * @brief Move constructor
   *
   * @param rhs Object to move from
   */
  __MATX_HOST__ tensor_t(tensor_t &&rhs) noexcept
      : detail::tensor_impl_t<T, RANK, Desc>{rhs.Data(), std::move(rhs.desc_)}, storage_(std::move(rhs.storage_))
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
  __MATX_HOST__ void Shallow(const self_type &rhs) noexcept
  {
    this->SetData(rhs.Data());
    storage_ = rhs.storage_;
    this->desc_ = rhs.desc_;
  }

  /** Swaps two tensors
   *
   * Swaps members of two tensors, including pointers, shapes, and descriptors
   *
   * @param lhs
   *   Left argument
   * @param rhs
   *   Right argument
   */
  friend void swap(self_type &lhs, self_type &rhs) noexcept
  {
    using std::swap;

    auto tmpdata = lhs.Data();
    lhs.SetData(rhs.Data());
    rhs.SetData(tmpdata);
    swap(lhs.storage_, rhs.storage_);
    swap(lhs.desc_, rhs.desc_);
  }

  __MATX_INLINE__  ~tensor_t() = default;

  const std::string str() const {
    return name_;
  }

  void set_name(std::string name)
  {
    name_ = name;
  }

  /**
   * @brief Construct a new tensor t object from an arbitrary shape and descriptor
   *
   * @tparam S2 Shape type
   * @tparam D2 Descriptor type
   * @param s Shape object
   * @param desc Descriptor object
   */
  template <typename D2 = Desc>
    requires is_matx_descriptor<remove_cvref_t<D2>>
  tensor_t(Storage<T> s, D2 &&desc) :
    detail::tensor_impl_t<T, RANK, Desc>{std::forward<D2>(desc)},
    storage_{std::move(s)}
  {
    this->SetLocalData(storage_.data());
  }

  /**
   * @brief Construct a new tensor t object. Used to copy an existing storage object for proper reference counting
   *
   * @param s
   * @param desc
   * @param ldata
   */
  template <typename D2 = Desc>
  tensor_t(Storage<T> s, D2 &&desc, T* ldata) :
    detail::tensor_impl_t<T, RANK, D2>{std::forward<D2>(desc)},
    storage_{std::move(s)}
  {
    this->SetLocalData(ldata);
  }


  /**
   * Constructor for a rank-1 and above tensor.
   *
   * @param desc
   *   Tensor descriptor
   */
  template <typename D2 = Desc>
    requires is_matx_descriptor<D2>
  __MATX_INLINE__ tensor_t(D2 &&desc) :
    detail::tensor_impl_t<T, RANK, D2>{std::forward<D2>(desc)},
    storage_{make_owning_storage<T>(this->desc_.TotalSize())}
  {
    this->SetLocalData(storage_.data());
  }

  /**
   * Constructor for a rank-0 tensor.
   *
   * NOTE: Use empty braces {} for the unused parameter.
   *
   */
  __MATX_INLINE__ tensor_t(const std::initializer_list<detail::no_size_t> /* unused */) :
    // The ctor argument is unused, but matches {} for rank-0 tensors. We do
    // not use [[maybe_unused]] due to https://gcc.gnu.org/bugzilla/show_bug.cgi?id=81429 in gcc < 9.3
    detail::tensor_impl_t<T, RANK, Desc>(cuda::std::array<index_t, 0>{}),
    storage_{make_owning_storage<T>(1)}
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
    detail::tensor_impl_t<T, RANK, Desc>(shape),
    storage_{make_owning_storage<T>(this->desc_.TotalSize())}
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const self_type &op)
  {
      return detail::set(*this, op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, op_base);
    //return detail::set(static_cast<detail::tensor_impl_t<T, RANK, Desc>&>(*this), op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const self_type &op)
  {
      return detail::set(*this, *this + op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this + op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const self_type &op)
  {
      return detail::set(*this, *this - op);
  }

  /**
   * Lazy assignment operator-=. Used to create a "set" object for deferred
   * execution on a device
   *
   * @tparam T2
   *   Type of operator
   * @param op
   *   Operator or scalar type to assign
   *
   * @returns set object containing the destination view and source object
   *
   */
  template <typename T2>
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const T2 &op)
  {
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this - op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const self_type &op)
  {
      return detail::set(*this, *this * op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this * op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const self_type &op)
  {
      return detail::set(*this, *this / op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this / op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const self_type &op)
  {
      return detail::set(*this, *this << op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this << op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const self_type &op)
  {
      return detail::set(*this, *this >> op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this >> op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const self_type &op)
  {
      return detail::set(*this, *this | op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this | op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const self_type &op)
  {
      return detail::set(*this, *this & op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this & op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const self_type &op)
  {
      return detail::set(*this, *this ^ op);
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
    const typename detail::base_type_t<T2> &op_base = op;
    return detail::set(*this, *this ^ op_base);
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
  [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const self_type &op)
  {
      return detail::set(*this, *this % op);
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
    const typename detail::base_type_t<T2> &op_base = op;
      return detail::set(*this, *this % op_base);
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
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    [[maybe_unused]] stride_type prod = cuda::std::accumulate(cuda::std::begin(shape), cuda::std::end(shape), static_cast<stride_type>(1), cuda::std::multiplies<stride_type>());
    // Ensure new shape's total size is not larger than the original
    MATX_ASSERT_STR(
        sizeof(M) * prod <= storage_.bytes(), matxInvalidSize,
        "Total size of new tensor must not be larger than the original");

    // This could be loosened up to make sure only the fastest changing dims
    // are compact
    MATX_ASSERT_STR(this->desc_.IsContiguous(), matxInvalidSize,
       "To get a reshaped view the tensor must be compact");

    // Copy descriptor and call ctor with shape
    Desc new_desc{std::forward<Shape>(shape)};
    // Multiple views can share the same storage
    return tensor_t<M, R, Desc>{storage_, std::move(new_desc), this->Data()};
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
   * @tparam ShapeIntType
   *   Type of integer shape array
   * @tparam NRANK
   *   New rank of tensor
   * @param shape
   *   New shape of tensor
   *
   * @return
   *    A view of the data with the appropriate strides and dimensions set
   */
  template <typename ShapeIntType, int NRANK>
  __MATX_INLINE__ auto View(const ShapeIntType (&shape)[NRANK])
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    // Change this to not rely on index_t
    cuda::std::array<index_t, NRANK> tshape;
    std::move(std::begin(shape), std::end(shape), tshape.begin());

    [[maybe_unused]] stride_type prod = cuda::std::accumulate(cuda::std::begin(shape), cuda::std::end(shape), static_cast<stride_type>(1), cuda::std::multiplies<stride_type>());
    MATX_ASSERT_STR(
        sizeof(T) * prod <= storage_.bytes(), matxInvalidSize,
        "Total size of new tensor must not be larger than the original");

    // This could be loosened up to make sure only the fastest changing dims
    // are compact
    MATX_ASSERT_STR(this->desc_.IsContiguous(), matxInvalidSize,
       "To get a reshaped view the tensor must be compact");

    DefaultDescriptor<tshape.size()> desc{std::move(tshape)};
    return tensor_t<T, NRANK, decltype(desc)>{storage_, std::move(desc), this->Data()};
  }

  /**
   * @brief Make a copy of a tensor and maintain all refcounts
   *
   * @return Copy of view
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
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    int dev;
    cudaGetDevice(&dev);
  #if CUDART_VERSION <= 12000
    cudaMemPrefetchAsync(this->Data(), this->desc_.TotalSize() * sizeof(T), dev, stream);
  #else
    cudaMemLocation loc;
    loc.id = dev;
    loc.type = cudaMemLocationTypeDevice;
    cudaMemPrefetchAsync(this->Data(), this->desc_.TotalSize() * sizeof(T), loc, 0, stream);
  #endif
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
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  #if CUDART_VERSION <= 12000
    cudaMemPrefetchAsync(this->Data(), this->desc_.TotalSize() * sizeof(T), cudaCpuDeviceId,
                         stream);
  #else
    cudaMemLocation loc;
    loc.id = cudaCpuDeviceId;
    loc.type = cudaMemLocationTypeHost;
    cudaMemPrefetchAsync(this->Data(), this->desc_.TotalSize() * sizeof(T), loc, 0, stream);
  #endif
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
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    static_assert(is_complex_v<T>, "RealView() only works with complex types");

    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(this->Data());
    cuda::std::array<typename Desc::stride_type, RANK> strides;

MATX_LOOP_UNROLL
    for (int i = 0; i < RANK; i++) {
      strides[i] = this->desc_.Stride(i);
    }

    if constexpr (RANK > 0) {
MATX_LOOP_UNROLL
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    // Copy descriptor and call ctor with shape
    Desc new_desc{this->desc_.Shape(), std::move(strides)};
    // Create non-owning storage with the correct type for the real view
    auto real_storage = make_non_owning_storage<Type>(data, storage_.size() * 2);
    return tensor_t<Type, RANK, Desc>{real_storage, std::move(new_desc), data};
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
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    static_assert(is_complex_v<T>, "ImagView() only works with complex types");

    using Type = typename U::value_type;
    Type *data = reinterpret_cast<Type *>(this->Data()) + 1;
    cuda::std::array<stride_type, RANK> strides;
MATX_LOOP_UNROLL
    for (int i = 0; i < RANK; i++) {
      strides[i] = this->Stride(i);
    }

    if constexpr (RANK > 0) {
MATX_LOOP_UNROLL
      for (int i = 0; i < RANK; i++) {
        strides[i] *= 2;
      }
    }

    Desc new_desc{this->desc_.Shape(), std::move(strides)};
    // Create non-owning storage with the correct type for the imaginary view  
    auto imag_storage = make_non_owning_storage<Type>(data, storage_.size() * 2);
    return tensor_t<Type, RANK, Desc>{imag_storage, std::move(new_desc), data};
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
   * @returns permuted tensor view
   *
   */
  __MATX_INLINE__ auto Permute(const cuda::std::array<int32_t, RANK> &dims) const
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    auto new_desc = this->PermuteImpl(dims);
    return tensor_t<T, RANK, Desc>{storage_, std::move(new_desc), this->Data()};
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
   * @returns permuted tensor view
   *
   */
  __MATX_INLINE__ auto Permute(const int32_t (&dims)[RANK]) const
  {
    return Permute(detail::to_array(dims));
  }

  /**
   * Permute the last two dimensions of a matrix
   *
   * Utility function to permute the last two dimensions of a tensor. This is
   * useful in the numerous operations that take a permuted matrix as input, but
   * we don't want to permute the inner dimensions of a larger tensor.
   *
   * @returns tensor view with last two dims permuted
   *
   */
  __MATX_INLINE__ auto PermuteMatrix() const
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    static_assert(RANK >= 2, "Only tensors of rank 2 and higher can be permuted.");
    int32_t tdims[RANK];
    cuda::std::iota(std::begin(tdims), std::end(tdims), 0);
    cuda::std::swap(tdims[RANK - 2], tdims[RANK - 1]);
    return Permute(tdims);
  }

  /**
   * Set the underlying data pointer from the view
   *
   * Decrements any reference-counted memory and potentially frees before
   * resetting the data pointer. If refcnt is not nullptr, the count is
   * incremented.
   *
   * @tparam ShapeType
   *   Shape type
   * @param data
   *   Data pointer to set
   * @param shape
   *   Shape of tensor
   */
  template <typename ShapeType>
    requires (!std::is_pointer_v<remove_cvref_t<ShapeType>>)
  __MATX_HOST__ __MATX_INLINE__ void
  Reset(T *const data, ShapeType &&shape) noexcept
  {
    this->desc_.InitFromShape(std::forward<ShapeType>(shape));
    // For non-owning storage, we need to recreate the storage object
    storage_ = make_non_owning_storage<T>(data, this->desc_.TotalSize());
    this->SetData(data);
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
   *
   */
  __MATX_HOST__ __MATX_INLINE__ void
  Reset(T *const data) noexcept
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    // For non-owning storage, we need to recreate the storage object
    storage_ = make_non_owning_storage<T>(data, this->desc_.TotalSize());
    this->SetData(data);
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
    // For non-owning storage, we need to recreate the storage object
    storage_ = make_non_owning_storage<T>(data, this->desc_.TotalSize());
    this->SetData(ldata);
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
   * Creates an overlapping tensor view where an existing tensor can be
   * repeated into a higher rank with overlapping elements. For example, the
   * following 1D tensor [1 2 3 4 5] could be cloned into a 2D tensor with a
   * window size of 2 and overlap of 1, resulting in:
   *
   \verbatim
     [1 2
      2 3
      3 4
      4 5]
   \endverbatim
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
  template <int N>
  __MATX_INLINE__ auto
  OverlapView(const cuda::std::array<typename Desc::shape_type, N> &windows,
              const cuda::std::array<typename Desc::stride_type, N> &strides) const {
    auto new_desc = this->template OverlapViewImpl<N>(windows, strides);
    return tensor_t<T, RANK + 1, decltype(new_desc)>{storage_, std::move(new_desc), this->Data()};
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
  __MATX_INLINE__ auto Clone(const cuda::std::array<index_t, N> &clones) const
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    auto new_desc = this->template CloneImpl<N>(clones);
    return tensor_t<T, N, decltype(new_desc)>{storage_, std::move(new_desc), this->Data()};
  }

  template <int N>
  __MATX_INLINE__ auto Clone(const index_t (&clones)[N]) const
  {
    return Clone<N>(detail::to_array(clones));
  }

  __MATX_INLINE__ __MATX_HOST__ bool IsHostAccessiblePointer() {
    void* hostPtr = nullptr;
    const CUresult retval =
        cuPointerGetAttribute(&hostPtr, CU_POINTER_ATTRIBUTE_HOST_POINTER, (CUdeviceptr)this->Data());
    MATX_ASSERT_STR_EXP(retval, CUDA_SUCCESS, matxNotSupported, "Pointer is not host-accessible");
    return hostPtr != nullptr;
  }

  /**
   * Rank-0 initializer list setting
   *
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param val
   *   0 initializer list value
   *
   */
  template <int M = RANK>
    requires (M == 0)
  __MATX_INLINE__ __MATX_HOST__ void SetVals(T const &val)
  {
    static_assert(RANK == 0, "Single value in SetVals must be applied only to rank-0 tensor");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");
    this->operator()() = val;
  }

  /**
   * Rank-1 non-complex or rank-0 initializer list setting
   *
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param vals
   *   1D initializer list of values
   *
   */
  template <int M = RANK>
    requires ((!is_cuda_complex<T> && M == 1) || (is_cuda_complex<T> && M == 0))
  __MATX_INLINE__ __MATX_HOST__ void SetVals(const std::initializer_list<T> &vals)
  {
    static_assert(((!is_cuda_complex_v<T> && RANK == 1) || (is_cuda_complex_v<T> && RANK == 0)),
      "Single initializer list on SetVals only for non-complex rank 1 tensor or complex rank 0 tensors");
    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

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
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param vals
   *   1D/2D initializer list of values
   *
   */
  template <int M = RANK>
    requires ((!is_cuda_complex<T> && M == 2) || (is_cuda_complex<T> && M == 1))
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<T>>
              &vals)
  {
    static_assert(((!is_cuda_complex_v<T> && RANK == 2) || (is_cuda_complex_v<T> && RANK == 1)),
      "Double initializer list on SetVals only for non-complex rank 2 tensor or complex rank 1 tensors");
    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    for (index_t i = 0; i < static_cast<index_t>(vals.size()); i++) {
      for (index_t j = 0; j < static_cast<index_t>((vals.begin() + i)->size()); j++) {
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
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param vals
   *   3D/2D initializer list of values
   *
   */
  template <int M = RANK>
    requires ((!is_cuda_complex<T> && M == 3) || (is_cuda_complex<T> && M == 2))
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<T>>>
              vals)
  {
    static_assert(((!is_cuda_complex_v<T> && RANK == 3) || (is_cuda_complex_v<T> && RANK == 2)),
      "Triple initializer list on SetVals only for non-complex rank 3 tensor or complex rank 2 tensors");
    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

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
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param vals
   *   3D/4D initializer list of values
   *
   */
  template <int M = RANK>
    requires ((!is_cuda_complex<T> && M == 4) || (is_cuda_complex<T> && M == 3))
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>
              &vals)
  {
    static_assert(((!is_cuda_complex_v<T> && RANK == 4) || (is_cuda_complex_v<T> && RANK == 3)),
      "Quad initializer list on SetVals only for non-complex rank 4 tensor or complex rank 3 tensors");
    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin() + k)->size();
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
   * Note that for performance reasons only host-accessible pointers are supported with SetVals
   * at the moment.
   *
   * @param vals
   *   4D initializer list of values
   *
   */
  template <int M = RANK>
    requires (is_cuda_complex<T> && M == 4)
  __MATX_INLINE__ __MATX_HOST__ void
  SetVals(const std::initializer_list<
          const std::initializer_list<const std::initializer_list<
              const std::initializer_list<const std::initializer_list<T>>>>>
              &vals)
  {
    static_assert((is_cuda_complex_v<T> && RANK == 4),
          "Quintuple initializer list on SetVals only for complex rank 3 tensors");
    MATX_ASSERT_STR(IsHostAccessiblePointer(), matxNotSupported, "SetVals only supports host-accessible pointers (managed, host-pinned, or ATS-mapped)");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    for (size_t i = 0; i < vals.size(); i++) {
      for (size_t j = 0; j < (vals.begin() + i)->size(); j++) {
        for (size_t k = 0; k < ((vals.begin() + i)->begin() + j)->size(); k++) {
          for (size_t l = 0;
               l < (((vals.begin() + i)->begin() + j)->begin() + k)->size();
               l++) {
            for (size_t m = 0;
                 m < ((((vals.begin() + i)->begin() + j)->begin() + k)->begin() + l)
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
   * *NOTE* Users should not call Slice() directly anymore. Use the slice() operator instead.
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
  template <int N = RANK, typename StrideType>
  __MATX_INLINE__ auto Slice([[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                            [[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &ends,
                            [[maybe_unused]] StrideType strides) const
  {
    auto [new_desc, data] = this->template SliceImpl<N, StrideType>(firsts, ends, strides);
    return tensor_t<T, N, decltype(new_desc)>{storage_, std::move(new_desc), data};
  }

  template <typename StrideType, int N = RANK>
  __MATX_INLINE__ auto Slice(const typename Desc::shape_type (&firsts)[RANK],
                            const typename Desc::shape_type (&ends)[RANK],
                            StrideType strides) const
  {
    return Slice<N>(detail::to_array(firsts), detail::to_array(ends), detail::to_array(strides));
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
  __MATX_INLINE__ auto Slice(const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                            const cuda::std::array<typename Desc::shape_type, RANK> &ends) const
  {
    static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");

    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    return Slice<N, detail::NoStride>(firsts, ends, detail::NoStride{});
  }

  template <int N = RANK>
  __MATX_INLINE__ auto Slice(const typename Desc::shape_type (&firsts)[RANK],
                              const typename Desc::shape_type (&ends)[RANK]) const
  {
    return Slice<N>(detail::to_array(firsts), detail::to_array(ends));
  }


  static void FreeDLPack(struct DLManagedTensor *mtv) {
      delete [] mtv->dl_tensor.shape;
      delete [] mtv->dl_tensor.strides;
      delete static_cast<self_type *>(mtv->manager_ctx);

      mtv->dl_tensor.shape    = nullptr;
      mtv->dl_tensor.strides  = nullptr;

      delete mtv;
      mtv                     = nullptr;
    };

  /**
   * @brief Get a DLPack v0.8 structure representing the tensor
   *
   * DLPack is a commonly-used tensor memory layout format for moving tensors between libraries. This function
   * returns a DLPack structure based on a tensor_t. The caller is responsible for freeing the memory
   * by calling ->deleter(self).
   *
   * **Note**: This function will increment the reference count of the tensor. It is expected that once a tensor
   * is converted to DLPack someone will eventually call deleter(). If that does not happen a memory leak
   * will occur.
   *
   * @returns Pointer to new DLManagedTensorVersioned pointer. The caller must call the deleter function when finished.
   */
  DLManagedTensor *ToDlPack() const {
    auto mt = new DLManagedTensor;
    DLTensor *t = &mt->dl_tensor;
    CUpointer_attribute attr[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE, CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
    CUmemorytype mem_type;
    int dev_ord;
    void *data[2]       = {&mem_type, &dev_ord};

    t->data             = static_cast<void*>(this->Data());
    t->device.device_id = 0;

    // Determine where this memory resides
    void *data_ptr = const_cast<tensor_t*>(this)->GetStorage().data();
    auto kind = GetPointerKind(data_ptr);
    [[maybe_unused]] auto mem_res = cuPointerGetAttributes(sizeof(attr)/sizeof(attr[0]), attr, data, reinterpret_cast<CUdeviceptr>(data_ptr));
    MATX_ASSERT_STR_EXP(mem_res, CUDA_SUCCESS, matxCudaError, "Error returned from cuPointerGetAttributes");
    if (kind == MATX_INVALID_MEMORY) {
      if (mem_type == CU_MEMORYTYPE_DEVICE) {
        t->device.device_type = kDLCUDA;
        t->device.device_id = dev_ord;
      }
      else {
        t->device.device_type = kDLCPU;
      }
    }
    else {
      // We have a record of this pointer and can map it from the record
      switch (kind) {
        case MATX_MANAGED_MEMORY:
        case MATX_DEVICE_MEMORY:
        case MATX_ASYNC_DEVICE_MEMORY:
          t->device.device_type = kDLCUDA;
          t->device.device_id = dev_ord;
          break;
        case MATX_HOST_MEMORY:
          t->device.device_type = kDLCUDAHost;
          t->device.device_id = dev_ord;
          break;
        case MATX_HOST_MALLOC_MEMORY:
          t->device.device_type = kDLCPU;
          break;
        default:
          MATX_ASSERT_STR(false, matxCudaError, "Cannot determine kind of memory");
          break;
      }
    }

    t->ndim         = RANK;
    t->dtype        = detail::TypeToDLPackType<T>();
    t->shape        = new int64_t[RANK];
    t->strides      = new int64_t[RANK];
    for (int r = 0; r < RANK; r++) {
      t->shape[r]   = this->Size(r);
      t->strides[r] = this->Stride(r);
    }
    t->byte_offset  = 0;

    // Increment reference count by making a copy of the shared_ptr by allocating on the heap and
    // setting it as the context
    auto t_copy = new self_type{*this};
    mt->manager_ctx = t_copy;
    //mt->flags = 0; // Only for v1.0

    auto deleter = &self_type::FreeDLPack;
    mt->deleter = deleter;

    return mt;
  }

private:
  Storage<T> storage_;
  std::string name_ = std::string("tensor_") + std::to_string(RANK) + "_" + detail::to_short_str<T>();
};


} // end namespace matx
