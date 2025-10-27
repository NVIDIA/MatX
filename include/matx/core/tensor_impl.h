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

#include <cassert>
#include <type_traits>
#include <cuda/std/functional>
#include "matx/core/vector.h"
#include "matx/core/error.h"
#include "matx/core/defines.h"
#include "matx/core/tensor_desc.h"
#include "matx/core/type_utils.h"
#include "matx/core/tensor_utils.h"
#include "matx/operators/set.h"
#include "matx/core/sparse_tensor_format.h"
#include "matx/core/utils.h"
//#include "matx_exec_kernel.h"
#include "iterator.h"
#include "matx/core/make_tensor.h"

namespace matx {

namespace detail {

template <typename T>
struct DenseTensorData {
  using dense_data = bool;
  T *ldata_;
};

template <typename T, typename CRD, typename POS, typename TF>
struct SparseTensorData {
  using sparse_data = bool;
  using value_type = T;
  using crd_type = CRD;
  using pos_type = POS;
  using Format = TF;
  static constexpr int LVL = TF::LVL;
  T *ldata_;
  CRD *crd_[LVL];
  POS *pos_[LVL];
};

/**
 * @brief Bare implementation of tensor class
 *
 * Defines the minimum operations needed for a tensor to be useful. This class
 * should be limited to primitive types. The bare class contains operations that
 * are necessary to function on a device (getters/setters), but does not contain
 * any ownership semantics.
 *
 * This class is designed to be inherited from more capable tensor classes like
 * tensor_t below. Because of the asynchronous nature of GPU devices, derived classes
 * are responsible for handling ownership and lifetime of the pointer in a
 * tensor_impl_t.
 *
 * @tparam T Type of tensor
 * @tparam RANK Rank of tensor
 */
template <typename T, int RANK, typename Desc = DefaultDescriptor<RANK>, typename TensorData = DenseTensorData<T>>
class tensor_impl_t {
  public:
    // Type specifier for reflection on class
    using type = T; // TODO is this necessary
    using value_type = T;
    using tensor_view = bool;
    using tensor_impl = bool;
    using desc_type = Desc;
    using data_type = TensorData;
    using shape_type = typename Desc::shape_type;
    using stride_type = typename Desc::stride_type;
    using matxoplvalue = bool;
    using self_type = tensor_impl_t<T, RANK, Desc, TensorData>;

    // Type specifier for signaling this is a matx operation
    using matxop = bool;

#ifdef MATX_EN_JIT
    // Tensors are considered a leaf type and there cannot be an inner storage type
    struct JIT_Storage {
      T *ldata_;
    };

    JIT_Storage ToJITStorage() const {
      return JIT_Storage{data_.ldata_};
    }

    __MATX_INLINE__ std::string get_jit_class_name() const {
      std::string symbol_name = "JITTensorImpl_" + detail::type_to_string_c_name<T>() + "_";

      symbol_name += "R" + std::to_string(RANK) + "_";

      symbol_name += "SI_";
      for (int i = 0; i < RANK; ++i) {
        symbol_name += std::to_string(desc_.Size(i));
        if (i != RANK - 1) {
          symbol_name += "_";
        }
      }

      symbol_name += "ST_";
      for (int i = 0; i < RANK; ++i) {
        symbol_name += std::to_string(desc_.Stride(i));
        if (i != RANK - 1) {
          symbol_name += "_";
        }
      }

      return symbol_name;
    }        

    __MATX_INLINE__ auto get_jit_op_str() const {
      // Note tensors are considered a leaf type, so we do not make JIT_Storage like we do for all other operators. This is because
      // the inner type is just a plain pointer and has no operator() defined on it.
      const std::string class_name = get_jit_class_name();
      return cuda::std::make_tuple(
         class_name, 
         std::string("struct " + class_name + "  {\n") + 
             "  static constexpr int RANK = " + std::to_string(Rank()) + ";\n" +
             "  using T = " + detail::type_to_string<T>() + ";\n" +
             "  using value_type = T;\n" +
             "  using matxop = bool;\n" +
             "  using stride_type = " +  detail::type_to_string<stride_type>() + ";\n" +
             "  T *ldata_;\n" +       
             "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> strides_ = { " + detail::array_to_string(desc_.Strides()) + " };\n" +
             "  constexpr static cuda::std::array<index_t, " + std::to_string(Rank()) + "> sizes_ = { " + detail::array_to_string(desc_.Shape()) + " };\n" +
             "  template <detail::ElementsPerThread EPT, int I = 0, typename ...Is>\n" +
             "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetVal([[maybe_unused]] cuda::std::tuple<Is...> tup)  {\n" +
             "    if constexpr (I < sizeof...(Is)) {\n" +
             "      if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {\n" +
             "        return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I] * static_cast<index_t>(EPT));\n" +
             "      }\n" +
             "      else {\n" +
             "        return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I]);\n" +
             "      }\n" +
             "    }\n" +
             "    else {\n" +
             "      return 0;\n" +
             "    }\n" +
             "  }\n" +
             "  template <detail::ElementsPerThread EPT, int I = 0, typename ...Is>\n" +
             "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetValC([[maybe_unused]] const cuda::std::tuple<Is...> tup) const {\n" +
             "    if constexpr (I < sizeof...(Is)) {\n" +
             "      if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {\n" +
             "        return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I] * static_cast<index_t>(EPT));\n" +
             "      }\n" +
             "      else {\n" +
             "        return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(strides_[I]);\n" +
             "      }\n" +
             "    }\n" +
             "    else {\n" +
             "      return 0;\n" +
             "    }\n" +
             "  }\n" + 
             "  template <detail::ElementsPerThread EPT, typename... Is>\n" +
             "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetOffsetOptimized(Is... indices) const {\n" +
             "     constexpr size_t rank = sizeof...(Is);\n" +
             "     constexpr int EPT_int = static_cast<int>(EPT);\n" +
             "     const cuda::std::array<index_t, rank> idx{indices...};\n" +
             "    \n" +
             "    if constexpr (rank == 1) {\n" +
             "        if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
             "          return idx[0] * (strides_[0] * EPT_int);\n" +
             "      } else {\n" +
             "        return idx[0] * strides_[0];\n" +
             "      }\n" +
             "    }\n" +
             "    else if constexpr (rank == 2) {\n" +
             "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
             "        return idx[0] * strides_[0] + idx[1] * (strides_[1] * EPT_int);\n" +
             "      } else {\n" +
             "        return idx[0] * strides_[0] + idx[1] * strides_[1];\n" +
             "      }\n" +
             "    }\n" +
             "    else if constexpr (rank == 3) {\n" +
             "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
             "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * (strides_[2] * EPT_int);\n" +
             "      } else {\n" +
             "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2];\n" +
             "      }\n" +
             "    }\n" +
             "    else if constexpr (rank == 4) {\n" +
             "      if constexpr (EPT != detail::ElementsPerThread::ONE) {\n" +
             "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2] + idx[3] * (strides_[3] * EPT_int);\n" +
             "      } else {\n" +
             "        return idx[0] * strides_[0] + idx[1] * strides_[1] + idx[2] * strides_[2] + idx[3] * strides_[3];\n" +
             "      }\n" +
             "    }\n" +
             "    else {\n" +
             "      // For rank > 4, fall back to the recursive implementation\n" +
             "      return GetValC<EPT, 0, Is...>(cuda::std::make_tuple(indices...));\n" +
             "    }\n" +
             "  }\n" +                         
             "  template <typename CapType, int M = RANK, typename... Is,\n" +
             "            cuda::std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>\n" +
             "  __MATX_INLINE__  __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const noexcept" + "{\n" +
             "    static_assert(sizeof...(Is) == M, \"Number of indices of operator() must match rank of tensor\");\n" +
             "     constexpr int EPT_int = static_cast<int>(CapType::ept);\n" +
             "     const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);\n" +
             "     if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {\n" +
             "       return ldata_[offset];\n" +
             "     } else if constexpr (EPT_int * sizeof(T) <= MAX_VEC_WIDTH_BYTES ) {\n" +
             "       return *reinterpret_cast<detail::Vector<T, EPT_int>*>(ldata_ + offset);\n" +
             "     } else {\n" +
             "       detail::Vector<T, EPT_int> vec;\n" +
             "       vec.template load<EPT_int>(ldata_ + offset);\n" +
             "       return vec;\n" +
             "     }\n" +
             "  }\n" +
             "  template <typename CapType, int M = RANK, typename... Is,\n" +
             "            cuda::std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>\n" +
             "  __MATX_INLINE__  __MATX_DEVICE__ decltype(auto) operator()(Is... indices) noexcept\n" +
             "  {\n" +
             "    static_assert(sizeof...(Is) == M, \"Number of indices of operator() must match rank of tensor\");\n" +
             "    constexpr int EPT_int = static_cast<int>(CapType::ept);\n" +
             "    const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);\n" +
             "    if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {\n" +
             "      return ldata_[offset];\n" +
             "    } else {\n" +
             "      return *reinterpret_cast<detail::Vector<T, EPT_int>*>(ldata_ + offset);\n" +
             "    }\n" +
             "  }\n" +
             "  template <typename CapType, int M = RANK, typename... Is>\n" +
             "  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* data_ptr(index_t block_idx, index_t ttl_threads) const noexcept\n" +
             "  {\n"
             "    //const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);\n" +
             "    //return ldata_ + offset;\n" +
             "    return ldata_ + block_idx * ttl_threads * static_cast<index_t>(CapType::ept);\n" +
             "  }\n" +
             "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank()\n" +
             "  {\n" +
             "    return " + std::to_string(Rank()) + ";\n" +
             "  }\n" +
             "  constexpr __MATX_INLINE__  __MATX_DEVICE__ index_t Size(int dim) const\n" +
             "  {\n" +
             "    return sizes_[dim];\n " +
             "  }\n" +
             "};\n"
      );
    }    
#endif     

    __MATX_INLINE__ tensor_impl_t(const tensor_impl_t &) = default;
    __MATX_INLINE__ tensor_impl_t(tensor_impl_t &&) = default;
    __MATX_INLINE__ tensor_impl_t& operator=(tensor_impl_t &&) = default;


    __MATX_INLINE__ ~tensor_impl_t() = default;


    const std::string str() const {
      return std::string("tensor_impl_") + std::to_string(RANK) + "_" + to_short_str<T>();
    }

    /** Swaps two tensor implementations
     *
     * Swaps members of two tensor implementations
     *
     * @param lhs
     *   Left argument
     * @param rhs
     *   Right argument
     */
    friend void swap(self_type &lhs, self_type &rhs) noexcept
    {
      using cuda::std::swap;

      swap(lhs.data_, rhs.data_);
      swap(lhs.desc_, rhs.desc_);
    }

    /**
     * Constructor for a rank-0 tensor (scalar).
     */
    tensor_impl_t() {

    }

    /**
     * Constructor for a rank-0 tensor (scalar).
     *
     * @param data
     *   Data pointer
     */
    tensor_impl_t(T *const data)  {
      data_.ldata_ = data;
      static_assert(RANK == 0, "tensor_impl_t with single pointer parameter must be a rank 0 tensor");
    }
    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     */
    template <typename ShapeType,
              std::enable_if_t<!is_tensor_view_v<remove_cvref_t<ShapeType>> && !is_matx_descriptor_v<remove_cvref_t<ShapeType>>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(ShapeType &&shape) : desc_(std::forward<ShapeType>(shape))
    {
    }

    /**
     * Constructor for a rank-1 and above tensor.
     *
     * @param shape
     *   Tensor shape
     * @param strides
     *   Tensor strides
     */
    template <typename ShapeType, typename StrideType>
    __MATX_INLINE__ tensor_impl_t(ShapeType &&shape, StrideType &&strides)
        : desc_(std::forward<ShapeType>(shape), std::forward<StrideType>(strides))
    {}

    /**
     * Constructor for a rank-1 and above tensor using a user pointer and shape
     * input
     *
     * @tparam ShapeType
     *   Type of shape
     * @param ldata
     *   Offset data pointer (start of view)
     * @param shape
     *   Sizes for each dimension. Length of sizes must match RANK
     */
    template <typename ShapeType, std::enable_if_t<!is_matx_descriptor_v<typename remove_cvref<ShapeType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(T *const ldata, ShapeType &&shape)
        : desc_(std::forward<ShapeType>(shape))
    {
      data_.ldata_ = ldata;
    }


    /**
     * Constructor for creating a view with a user-defined data pointer.
     *
     * @tparam ShapeType
     *   Type of shape
     * @tparam StrideType
     *   Type of stride
     * @param ldata
     *   Offset data pointer (start of view)
     * @param shape
     *   Sizes for each dimension.
     * @param strides
     *   Tensor strides
     */
    template <typename ShapeType, typename StrideType>
    __MATX_INLINE__ tensor_impl_t(T *const ldata,
                    ShapeType &&shape,
                    StrideType &&strides)
        : desc_(std::forward<ShapeType>(shape), std::forward<StrideType>(strides))
    {
      data_.ldata_ = ldata;
    }


    /**
     * Constructor for creating a view with a descriptor and user-provided pointer
     *
     * If not reference counted, it is the caller's responsibility to manage the
     * data pointer, including allocation and freeing.
     *
     * @tparam DescriptorType
     *   Descriptor type
     * @param desc
     *   Tensor descriptor
     * @param ldata
     *   Data type
     */
// gcc 14.1 incorrectly reports desc as uninitialized in some contexts
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
    template <typename DescriptorType, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<DescriptorType>::type>, bool> = true>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ tensor_impl_t(T *const ldata,
                    DescriptorType &&desc)
        : desc_{std::forward<DescriptorType>(desc)}
    {
      data_.ldata_ = ldata;
    }
MATX_IGNORE_WARNING_POP_GCC

    /**
     * Constructor for creating a view with only a descriptor
     *
     * Descriptor must confirm to all descriptor semantics. See documentation for details
     *
     * @tparam DescriptorType
     *   Descriptor type
     * @param desc
     *   Tensor descriptor
     */
    template <typename DescriptorType, std::enable_if_t<is_matx_descriptor_v<typename remove_cvref<DescriptorType>::type>, bool> = true>
    __MATX_INLINE__ tensor_impl_t(DescriptorType &&desc)
        : desc_{std::forward<DescriptorType>(desc)}
    {
    }

    __MATX_HOST__ void Shallow(const self_type &rhs) noexcept
    {
      data_.ldata_ = rhs.Data();
      desc_ = rhs.desc_;
    }

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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto copy(const tensor_impl_t<T, RANK> &op)
    {
      data_.ldata_ = op.Data();
      desc_ = op.desc_;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator+=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator-=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator*=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator/=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator<<=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator>>=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator|=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator&=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator^=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
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
    [[nodiscard]] __MATX_INLINE__ __MATX_HOST__ auto operator%=(const tensor_impl_t<T, RANK> &op)
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
      const typename detail::base_type_t<T2> &op_base = op;
        return set(*this, *this % op_base);
    }


    /**
     * Get the shape the tensor from the underlying data
     *
     * @return
     *    A shape of the data with the appropriate dimensions set
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__
    auto Shape() const noexcept { return this->desc_.Shape(); }

    /**
     * Get the strides the tensor from the underlying data
     *
     * @return
     *    A shape of the data with the appropriate strides set
     */
    __MATX_INLINE__ auto Strides() const noexcept { return this->desc_.Strides(); }

    template <int N = RANK, typename StrideType>
    __MATX_INLINE__ auto SliceImpl([[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                              [[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &ends,
                              [[maybe_unused]] StrideType strides) const
    {
      static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<typename Desc::shape_type, N> n = {};
      cuda::std::array<typename Desc::stride_type, N> s = {};

      T *tmpdata = data_.ldata_;
      int d = 0;

      [[maybe_unused]] int end_count = 0;
      for (int i = 0; i < RANK; i++) {
        if (ends[i] == matxDropDim) {
          end_count++;
        }
      }

      MATX_ASSERT_STR(((RANK - end_count) == N), matxInvalidSize,
              "Number of matxDropDim specifiers must match the output rank");

  MATX_LOOP_UNROLL
      for (int i = 0; i < RANK; i++) {
        typename Desc::shape_type first = firsts[i] < 0 ? this->Size(i) + firsts[i] : firsts[i];
        typename Desc::shape_type end = ends[i]   < 0 ? this->Size(i) + ends[i]   : ends[i];

        MATX_ASSERT_STR((end > matxIdxSentinel) || (end <= this->Size(i)), matxInvalidDim,
          "Slice end index out of range of operator");

        MATX_ASSERT_STR(first < end, matxInvalidSize, "Slice must be at least one element long");

        [[maybe_unused]] typename Desc::stride_type stride_mult;

        if constexpr (std::is_same_v<StrideType, detail::NoStride>) {
          stride_mult = 1;
        }
        else {
          stride_mult = (strides[i] == matxKeepStride) ? 1 : strides[i];
        }

        MATX_ASSERT_STR(first < end, matxInvalidParameter,
                        "Starting slice must be less than end slice");
        MATX_ASSERT_STR(first < this->desc_.Size(i), matxInvalidParameter,
                        "Requested slice start index out of bounds");

        // offset by first
        tmpdata += first * Stride(i);

        if constexpr (N > 0) {
          if (end != matxDropDim) {
            MATX_ASSERT_STR(end != matxKeepDim, matxInvalidParameter, "matxKeepDim only valid for clone(), not slice()");
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
      }

      MATX_ASSERT_STR(d == N, matxInvalidDim,
                      "Number of indices must match the target rank to slice to");

      return cuda::std::make_tuple(tensor_desc_t<decltype(n), decltype(s), N>{std::move(n), std::move(s)}, tmpdata);
    }


    template <int N = RANK, typename StrideType>
    __MATX_INLINE__ auto Slice([[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                               [[maybe_unused]] const cuda::std::array<typename Desc::shape_type, RANK> &ends,
                               [[maybe_unused]] StrideType strides) const
    {
      auto [new_desc, data] = this->SliceImpl<N, StrideType>(firsts, ends, strides);
      return tensor_impl_t<T, N, decltype(new_desc)>{data, std::move(new_desc)};
    }

    template <int N = RANK>
    __MATX_INLINE__ auto Slice(const cuda::std::array<typename Desc::shape_type, RANK> &firsts,
                              const cuda::std::array<typename Desc::shape_type, RANK> &ends) const
    {
      static_assert(N <= RANK && RANK > 0, "Must slice to a rank the same or less than current rank.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      return Slice<N, detail::NoStride>(firsts, ends, detail::NoStride{});
    }


    template <int N>
    __MATX_INLINE__ auto CloneImpl(const cuda::std::array<index_t, N> &clones) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<index_t, N> n;
      cuda::std::array<typename Desc::stride_type, N> s;

      int d = 0;

      MATX_LOOP_UNROLL
      for (int i = 0; i < N; i++) {
        index_t size = clones[i];

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
      return new_desc;
    }


    template <int N>
    __MATX_INLINE__ auto Clone(const cuda::std::array<index_t, N> &clones) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      auto new_desc = CloneImpl<N>(clones);

      return tensor_impl_t<T, N, decltype(new_desc)>{this->data_.ldata_, std::move(new_desc)};
    }

    __MATX_INLINE__ auto PermuteImpl(const cuda::std::array<int32_t, RANK> &dims) const
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      static_assert(RANK >= 1, "Only tensors of rank 1 and higher can be permuted.");
      cuda::std::array<shape_type, RANK> n;
      cuda::std::array<stride_type, RANK> s;
      [[maybe_unused]] bool done[RANK] = {0};

  MATX_LOOP_UNROLL
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

      return Desc{std::move(n), std::move(s)};
    }

    __MATX_INLINE__ auto Permute(const cuda::std::array<int32_t, RANK> &dims) const
    {
      auto new_desc = PermuteImpl(dims);
      return tensor_impl_t<T, RANK, decltype(new_desc)>{this->data_.ldata_, std::move(new_desc)};
    }

    template <int N>
    __MATX_INLINE__ auto
    OverlapViewImpl(const cuda::std::array<typename Desc::shape_type, N> &windows,
                const cuda::std::array<typename Desc::stride_type, N> &strides) const
    {
      static_assert(RANK == 1, "Overlapped views only supported on 1D tensors.");

      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      cuda::std::array<typename Desc::shape_type, RANK+1> n;
      cuda::std::array<typename Desc::stride_type, RANK+1> s;

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
      return new_desc;
    }

    template <int N>
    __MATX_INLINE__ auto
    OverlapView(const cuda::std::array<typename Desc::shape_type, N> &windows,
                const cuda::std::array<typename Desc::stride_type, N> &strides) const {
      auto new_desc = OverlapViewImpl<N>(windows, strides);
      return tensor_impl_t<T, RANK + 1, decltype(new_desc)>{this->data_.ldata_, std::move(new_desc)};
    }

    template <typename O>
    __MATX_INLINE__ bool isSameView(const O &o) {
      if constexpr (is_tensor_view_v<O> && RANK == O::Rank()) {
        return Data() == o.Data() &&
          this->Shape() == o.Shape() &&
          this->Strides() == o.Strides();
      } else {
        return false;
      }
    }

    /**
     * Set the size of a dimension
     *
     * @return
     *    A shape of the data with the appropriate dimensions set
     */
    __MATX_INLINE__ auto Descriptor() const noexcept { return this->desc_; }

    template <typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* GetPointer(Is... indices) const noexcept
    {
      return data_.ldata_ + GetOffsetOptimized<detail::ElementsPerThread::ONE>(indices...);
    }

    // Locates position of an element at given indices, or returns -1 when not
    // found.
    template <int L = 0>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t
    GetPos(index_t *lvlsz, index_t *lvl, index_t pos) const {
      static constexpr int LVL = TensorData::Format::LVL;
      if constexpr (L < LVL) {
        using lspec = std::tuple_element_t<L, typename TensorData::Format::LvlSpecs>;
        if constexpr (lspec::Type::isDense() || lspec::Type::isRange()) {
          // Dense level: pos * size + i.
          // TODO: see below, use a constexpr GetLvlSize(L) instead?
          const index_t dpos = pos * lvlsz[L] + lvl[L];
          if constexpr (L + 1 < LVL) {
            return GetPos<L + 1>(lvlsz, lvl, dpos);
          } else {
            return dpos;
          }
        } else if constexpr (lspec::Type::isSingleton()) {
          // Singleton level: pos if crd[pos] == i and next levels match.
          if (CRDData(L)[pos] == lvl[L]) {
            if constexpr (L + 1 < LVL) {
              return GetPos<L + 1>(lvlsz, lvl, pos);
            } else {
              return pos;
            }
          }
        } else if constexpr (lspec::Type::isCompressed() || lspec::Type::isCompressedNU()) {
          // Compressed level: scan for match on i and test next levels.
          const typename TensorData::crd_type *c = CRDData(L);
          const typename TensorData::pos_type *p = POSData(L);
          for (index_t pp = p[pos], hi = p[pos + 1]; pp < hi; pp++) {
            if (c[pp] == lvl[L]) {
              if constexpr (L + 1 < LVL) {
                const index_t cpos = GetPos<L + 1>(lvlsz, lvl, pp);
                if constexpr (lspec::Type::isCompressed()) {
                  return cpos; // always end scan (unique)
                } else if (cpos != -1) {
                  return cpos; // only end scan on success (non-unique)
                }
              } else {
                return pp;
              }
            }
          }
        } else {
#ifndef __CUDACC__
          MATX_THROW(matxNotSupported, "unimplemented case");
#endif
        }
      }
      return -1; // not found
    }

    // Element getter (viz. "lhs = Acoo(0,0);"). Note that due to the compact
    // nature of sparse data structures, these storage formats do not provide
    // cheap random access to their elements. Instead, the implementation will
    // search for a stored element at the given position (which involves a scan
    // at each compressed level). The implicit value zero is returned when the
    // element cannot be found. So, although functional for testing, clients
    // should avoid using getters inside performance critial regions, since
    // the implementation is far worse than O(1).
    template <typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T GetSparseValue(Is... indices) const noexcept {
      static constexpr int DIM = TensorData::Format::DIM;
      static constexpr int LVL = TensorData::Format::LVL;

      static_assert(sizeof...(Is) == DIM,
          "Number of indices of operator() must match rank of sparse tensor");
      cuda::std::array<index_t, DIM> dim{indices...};
      cuda::std::array<index_t, LVL> lvl;
      cuda::std::array<index_t, LVL> lvlsz;
      TensorData::Format::template dim2lvl<false>(dim.data(), lvl.data());
      // TODO: only compute once and provide a constexpr LvlSize(l) instead?
      TensorData::Format::template dim2lvl<true>(Shape().data(), lvlsz.data());
      const index_t pos = GetPos(lvlsz.data(), lvl.data(), 0);
      if (pos != -1) {
        const typename TensorData::value_type tmp = Data()[pos];
        return tmp;
      }

      return static_cast<typename TensorData::value_type>(0); // implicit zero
    }

    /**
     * Check if a tensor is linear in memory for all elements in the view
     *
     * @return
     *    The size of the dimension
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ constexpr bool IsContiguous() const noexcept
    {
      return desc_.IsContiguous();
    }

    template <typename detail::ElementsPerThread EPT, int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetVal([[maybe_unused]] cuda::std::tuple<Is...> tup)  {
      if constexpr (I < sizeof...(Is)) {
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
        if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {
          return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(this->desc_.Stride(I) * static_cast<index_t>(EPT));
        }
        else {
          return GetVal<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(this->desc_.Stride(I));
        }
MATX_IGNORE_WARNING_POP_GCC
      }
      else {
        return 0;
      }
    }

    // Optimized offset calculation for ranks 1-4 with explicit stride multiplications
    template <detail::ElementsPerThread EPT, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetOffsetOptimized(Is... indices) const {
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")      
      constexpr size_t rank = sizeof...(Is);
      constexpr int EPT_int = static_cast<int>(EPT);
      const cuda::std::array<index_t, rank> idx{indices...};
      
      if constexpr (rank == 1) {
        if constexpr (EPT != detail::ElementsPerThread::ONE) {
          return idx[0] * (this->desc_.Stride(0) * EPT_int);
        } else {
          return idx[0] * this->desc_.Stride(0);
        }
      }
      else if constexpr (rank == 2) {
        if constexpr (EPT != detail::ElementsPerThread::ONE) {
          return idx[0] * this->desc_.Stride(0) + idx[1] * (this->desc_.Stride(1) * EPT_int);
        } else {
          return idx[0] * this->desc_.Stride(0) + idx[1] * this->desc_.Stride(1);
        }
      }
      else if constexpr (rank == 3) {
        if constexpr (EPT != detail::ElementsPerThread::ONE) {
          return idx[0] * this->desc_.Stride(0) + idx[1] * this->desc_.Stride(1) + idx[2] * (this->desc_.Stride(2) * EPT_int);
        } else {
          return idx[0] * this->desc_.Stride(0) + idx[1] * this->desc_.Stride(1) + idx[2] * this->desc_.Stride(2);
        }
      }
      else if constexpr (rank == 4) {
        if constexpr (EPT != detail::ElementsPerThread::ONE) {
          return idx[0] * this->desc_.Stride(0) + idx[1] * this->desc_.Stride(1) + idx[2] * this->desc_.Stride(2) + idx[3] * (this->desc_.Stride(3) * EPT_int);
        } else {
          return idx[0] * this->desc_.Stride(0) + idx[1] * this->desc_.Stride(1) + idx[2] * this->desc_.Stride(2) + idx[3] * this->desc_.Stride(3);
        }
      }
      else {
        // For rank > 4, fall back to the recursive implementation
        return GetValC<EPT, 0, Is...>(cuda::std::make_tuple(indices...));
      }
MATX_IGNORE_WARNING_POP_GCC      
    }

    template <detail::ElementsPerThread EPT, int I = 0, typename ...Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ stride_type GetValC([[maybe_unused]] const cuda::std::tuple<Is...> tup) const {
      if constexpr (I < sizeof...(Is)) {
MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
        if constexpr (EPT != detail::ElementsPerThread::ONE && I == sizeof...(Is) - 1) {
          return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(this->desc_.Stride(I) * static_cast<index_t>(EPT));
        }
        else {
          return GetValC<EPT, I+1, Is...>(tup) + cuda::std::get<I>(tup)*(this->desc_.Stride(I));
        }
MATX_IGNORE_WARNING_POP_GCC
      }
      else {
        return 0;
      }
    }

    /**
     * operator() getter
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns value at given index
     *
     */
    template <typename CapType, int M = RANK, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const noexcept
    {
      static_assert(sizeof...(Is) == M, "Number of indices of operator() must match rank of tensor");
      if constexpr (!is_sparse_data_v<TensorData>) {
#ifndef NDEBUG
        assert(data_.ldata_ != nullptr);
#endif
        constexpr int EPT_int = static_cast<int>(CapType::ept);
        const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);

        if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {
          return data_.ldata_[offset];
        } else if constexpr (EPT_int * sizeof(T) <= MAX_VEC_WIDTH_BYTES ) {
          return *reinterpret_cast<detail::Vector<T, EPT_int>*>(data_.ldata_ + offset);
        } else {
          detail::Vector<T, EPT_int> vec;
          vec.template load<EPT_int>(data_.ldata_ + offset);
          return vec;
        }
      }
      else { // Sparse tensor getter
        return GetSparseValue(indices...);
      }
    }

    /**
     * Return a pointer to the tensor's data at the specified indices
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns pointer to value at given index
     */
    template <typename CapType, int M = RANK, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T* data_ptr(Is... indices) const noexcept
    {
      static_assert(sizeof...(Is) == M, "Number of indices of data_ptr must match rank of tensor");
      if constexpr (!is_sparse_data_v<TensorData>) {
        constexpr int EPT_int = static_cast<int>(CapType::ept);
        const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);
        return data_.ldata_ + offset;
      }
      else {
        // For sparse tensor, this is not supported
        return nullptr;
      }
    }

    template <int M = RANK, typename... Is>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const noexcept
    {
      return this->template operator()<DefaultCapabilities>(indices...);
    }    

    /**
     * operator() getter
     *
     * @param indices
     *   Indices of tensor
     *
     * @returns value at given index
     *
     */
    template <typename CapType, int M = RANK, typename... Is,
      std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) noexcept
    {
      if constexpr (!is_sparse_data_v<TensorData>) {
        static_assert(sizeof...(Is) == M, "Number of indices of operator() must match rank of tensor");
#ifndef NDEBUG
        assert(data_.ldata_ != nullptr);
#endif
        constexpr int EPT_int = static_cast<int>(CapType::ept);
        const index_t offset = GetOffsetOptimized<CapType::ept>(indices...);

        if constexpr (CapType::ept == detail::ElementsPerThread::ONE) {
          return data_.ldata_[offset];
        } else {
          return *reinterpret_cast<detail::Vector<T, EPT_int>*>(data_.ldata_ + offset);
        }
      }
      else {
        return GetSparseValue(indices...);
      }
    }

    template <int M = RANK, typename... Is,
      std::enable_if_t<cuda::std::conjunction_v<cuda::std::is_integral<Is>...>, bool> = true>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) noexcept
    {
      return this->template operator()<DefaultCapabilities>(indices...);
    }

    template <typename CapType>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T {
          return this->operator()<CapType>(args...);
        }, idx);
    }

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    template <typename CapType>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T& {
          return this->operator()<CapType>(args...);
        }, idx);
    }

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) const noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T {
          return this->operator()<DefaultCapabilities>(args...);
        }, idx);
    }

    /**
     * operator() getter with an array index
     *
     * @returns value in tensor
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  decltype(auto) operator()(const cuda::std::array<index_t, RANK> &idx) noexcept
    {
      return cuda::std::apply([&](auto &&...args) -> T& {
          return this->operator()<DefaultCapabilities>(args...);
        }, idx);
    }


    template <detail::OperatorCapability Cap, typename InType>
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
      // Since tensors are a "leaf" operator type, we will never have an operator passed to a tensor as the
      // type, but only POD types.
      if constexpr (Cap == detail::OperatorCapability::ELEMENTS_PER_THREAD) {
        if constexpr (Rank() == 0) {
          return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
        }
        else {
          if (Stride(Rank() - 1) != 1) {
            return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
          }
        }

        // Maybe relax this constraint later, but for now the tensor has to be contiguous. This should prevent clones
        // and strides from vectorizing.
        if (!IsContiguous()) {
          return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
        }

        if constexpr (sizeof(T) != alignment_by_type<T>()) {
          // If the alignment of the type does not match sizeof(T), then the logic
          // below will not necessarily work. It would generally work if alignment_by_type<T>() < sizeof(T),
          // except for cases where sizeof(T) > MAX_VEC_WIDTH_BYTES, which result in division by zero.
          // Curently, this is only true for vector types (e.g., double3) and we use one EPT for these cases.
          return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, detail::ElementsPerThread::ONE};
        } else {
          // Set to the max ILP if possible and let the dispatch routine reduce it if needed
          int width = in.jit ? 32 : MAX_VEC_WIDTH_BYTES / sizeof(T); 
          while (width > 1) {
            if (((Lsize() % width) == 0) &&                                       // Last dim is a multiple of vector load size
              ((reinterpret_cast<uintptr_t>(data_.ldata_) % (sizeof(T) * width)) == 0)) {
              break;
            }

            width /= 2;
          }

          return cuda::std::array<detail::ElementsPerThread, 2>{detail::ElementsPerThread::ONE, static_cast<detail::ElementsPerThread>(width)};
        }
      }
      else if constexpr (Cap == OperatorCapability::MAX_EPT_VEC_LOAD) {
        int vec = MAX_VEC_WIDTH_BYTES / sizeof(T);
        // Round down to next lower power of two
        // (if already a power of two, remains unchanged)
        // For vec = 0, will stay zero (avoid infinite loop)
        int power = 1;
        while (power * 2 <= vec) {
          power *= 2;
        }

        return power;
      }
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
        // No need to use combine_capabilities here since we're just returning a string.
        return get_jit_class_name();
#else
        return "";
#endif
      } 
      else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
        const auto [key, value] = get_jit_op_str();
      
        // Insert into the map if the key doesn't exist
        if (in.find(key) == in.end()) {
          in[key] = value;
        }

        return true;
#else
        return false;
#endif
      }
      else {
        return detail::capability_attributes<Cap>::default_value;
      }
    }


    /**
     * Get the rank of the tensor
     *
     * @returns Rank of the tensor
     *
     */
    static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }


    /**
     * Get the total number of elements in the tensor
     *
     *
     * @returns Total number of elements across all dimensions
     *
     */
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto TotalSize() const noexcept { return desc_.TotalSize(); }

    /**
     * Get the size of a single dimension of the tensor
     *
     * @param dim
     *   Desired dimension
     *
     * @returns Number of elements in dimension
     *
     */
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
    {
      return desc_.Size(dim);
    }

    /**
     * Get the stride of a single dimension of the tensor
     *
     * @param dim
     *   Desired dimension
     *
     * @returns Stride of dimension
     *
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Stride(uint32_t dim) const noexcept
    {
      return desc_.Stride(dim);
    }


    /**
     * Get the size of the last dimension
     *
     * @return
     *    The size of the dimension
     */
    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Lsize() const noexcept
    {
      return desc_.Size(Rank() - 1);
    }

    __MATX_INLINE__ __MATX_HOST__  auto Bytes() const noexcept
    {
      return TotalSize() * sizeof(*data_.ldata_);
    }

    /**
     * @brief Get data pointer
     *
     * @return data pointer
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto Data() const noexcept {
      return data_.ldata_;
    }

    /**
     * @brief Set data pointer
     *
     * @param data Pointer to new data pointer
     */
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  void SetData(T *data) noexcept {
      data_.ldata_ = data;
    }

    template <typename U = TensorData,
          std::enable_if_t<is_sparse_data_v<U>, int> = 0>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto CRDData(int l) const noexcept {
      return data_.crd_[l];
    }
    template <typename U = TensorData,
          std::enable_if_t<is_sparse_data_v<U>, int> = 0>
    __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__  auto POSData(int l) const noexcept{
      return data_.pos_[l];
    }

    /**
     * @brief Set local data pointer
     *
     * @param data
     *   Data pointer to set
     */
    void SetLocalData(T* data) {
      data_.ldata_ = data;
    }

    template <typename U = TensorData>
    auto SetSparseData(T* data,
                      typename U::crd_type* crd[U::LVL],
                      typename U::pos_type* pos[U::LVL])
        -> std::enable_if_t<is_sparse_data_v<U>, void> {
      data_.ldata_ = data;
      memcpy(data_.crd_, crd, U::LVL*sizeof(crd[0]));
      memcpy(data_.pos_, pos, U::LVL*sizeof(pos[0]));
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
    {
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
    {
    }

    __MATX_INLINE__ __MATX_HOST__ bool has_capability([[maybe_unused]] OperatorCapability cap) const {
      // Get the capability of the current operator node itself
      // The derived class's get_capability_impl will handle the specific logic for that operator type,
      // including querying children if it's a composite operator.
      return false; 
    } 


  protected:
    TensorData data_;
    Desc desc_;
};

}
};
