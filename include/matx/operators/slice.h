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


#include "matx/core/type_utils.h"
#include "matx/core/type_utils_both.h"
#include "matx/operators/base_operator.h"
#include "matx/core/operator_options.h"
namespace matx
{
  /**
   * Slices elements from an operator/tensor.
   */
  namespace detail {

    template <int DIM, typename T, typename StrideType>
      class SliceOp : public BaseOp<SliceOp<DIM, T, StrideType>>
    {
      public:
        using value_type = typename T::value_type;
        using self_type = SliceOp<DIM, T, StrideType>;

      private:
        using shape_type = index_t;
        mutable typename detail::base_type_t<T> op_;
        cuda::std::array<shape_type, DIM> sizes_;
        cuda::std::array<int32_t, DIM> dims_;
        cuda::std::array<shape_type, T::Rank()> starts_;
        StrideType strides_; // Add [[no_unique_address]] in c++20

      public:
        using matxop = bool;
        using matxoplvalue = bool;

        static_assert(T::Rank()>0, "SliceOp: Rank of operator must be greater than 0.");
        static_assert(DIM<=T::Rank(), "SliceOp: DIM must be less than or equal to operator rank.");

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string params_str;
          for (int i = 0; i < DIM; i++) {
            params_str += std::format("d{}_s{}_", dims_[i], sizes_[i]);
          }
          return std::format("JITSlice_{}", params_str);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T, typename StrideType> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int DIM_ = {};\n"
                "  constexpr static int OpRank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, DIM_> sizes_ = {{ {} }};\n"
                "  constexpr static cuda::std::array<int32_t, DIM_> dims_ = {{ {} }};\n"
                "  constexpr static cuda::std::array<index_t, OpRank_> starts_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  StrideType strides_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const {{ /* slice logic */ }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return DIM_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int32_t dim) const {{ return sizes_[dim]; }}\n"
                "}};\n",
                func_name, DIM, T::Rank(), detail::array_to_string(sizes_), detail::array_to_string(dims_), detail::array_to_string(starts_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "slice(" + op_.str() + ")"; }

        __MATX_INLINE__ SliceOp(const T &op, const cuda::std::array<shape_type, T::Rank()> &starts,
                                      const cuda::std::array<shape_type, T::Rank()> &ends,
                                      StrideType strides) : op_(op) {
          int32_t d = 0;
          for(int32_t i = 0; i < T::Rank(); i++) {
            shape_type start = starts[i] < 0 ? op.Size(i) + starts[i] : starts[i];
            shape_type end   = ends[i]   < 0 ? op.Size(i) + ends[i]   : ends[i];

            MATX_ASSERT_STR((start > matxIdxSentinel) || (start < op.Size(i)), matxInvalidDim,
              "Slice slice index out of range of operator");
            MATX_ASSERT_STR((end > matxIdxSentinel) || (end <= op.Size(i)), matxInvalidDim,
              "Slice end index out of range of operator");

            starts_[i] = start;

            if constexpr (!std::is_same_v<NoStride, StrideType>) {
              strides_[i] = strides[i];
            }

            // compute dims and sizes
            if(end != matxDropDim) {
              MATX_ASSERT_STR(end != matxKeepDim, matxInvalidParameter, "matxKeepDim only valid for clone(), not slice()");

              dims_[d] = i;

              if(end == matxEnd) {
                sizes_[d] = op.Size(i) - start;
              } else {
                sizes_[d] = end - start;
              }

              //adjust size by stride
              if constexpr (!std::is_same_v<NoStride, StrideType>) {
                sizes_[d] = (shape_type)std::ceil(static_cast<double>(sizes_[d])/ static_cast<double>(strides_[d]));
              }

              d++;
            }
          }
          MATX_ASSERT_STR(d==Rank(), matxInvalidDim, "SliceOp: Number of dimensions without matxDropDim must equal new rank.");
          MATX_LOG_TRACE("{} constructor: input_rank={}, output_rank={}", str(), T::Rank(), DIM);
        };

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(
            Op&& op,
            const decltype(starts_) &starts,
            const decltype(strides_) &strides,
            const decltype(dims_) &dims,
            Is... indices)
        {   
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            static_assert(sizeof...(Is)==Rank());
            static_assert((cuda::std::is_convertible_v<Is, index_t> && ... ));
        
            // convert variadic type to tuple so we can read/update
            cuda::std::array<index_t, T::Rank()> ind = starts;
            cuda::std::array<index_t, Rank()> inds{indices...};

            MATX_LOOP_UNROLL
            for (int32_t i = 0; i < T::Rank(); i++) {
              MATX_LOOP_UNROLL
              for(int32_t j = 0; j < Rank(); j++) {
                if(dims[j] == i) {
                  if constexpr (!cuda::std::is_same_v<NoStride, StrideType>) {
                    ind[i] = starts[j] + inds[j] * strides[i];
                  }
                  else {
                    ind[i] = starts[j] + inds[j];
                  }
                }
              }
            }       
                
            return get_value<CapType>(cuda::std::forward<Op>(op), ind);
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return std::format("{}<{}>", get_jit_class_name(), op_jit_name);
#else
            return "";
#endif
          }
          else if constexpr (Cap == OperatorCapability::JIT_CLASS_QUERY) {
#ifdef MATX_EN_JIT
            const auto [key, value] = get_jit_op_str();
            if (in.find(key) == in.end()) {
              in[key] = value;
            }
            detail::get_operator_capability<Cap>(op_, in);
            return true;
#else
            return false;
#endif
          }
          else if constexpr (Cap == OperatorCapability::DYN_SHM_SIZE) {
            return detail::get_operator_capability<Cap>(op_, in);
          }
          // Just support 1 EPT in slice for now to get this thing out the door. Later on the logic should be similar
          // to the tensor's EPT logic if the input is a tensor
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));          
          } else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op_), starts_, strides_, dims_, indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), starts_, strides_, dims_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }        

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return DIM;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ shape_type Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        ~SliceOp() = default;
        SliceOp(const SliceOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }
    };
  }

#ifndef DOXYGEN_ONLY
  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.Slice(starts, ends, strides);
    } else {
      return detail::SliceOp<OpType::Rank(),OpType,decltype(strides)>(op, starts, ends, strides);
    }
  }

  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      ::matx::detail::NoStride strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.Slice(starts, ends, strides);
    } else {
      return detail::SliceOp<OpType::Rank(),OpType,detail::NoStride>(op, starts, ends, detail::NoStride{});
    }
  }

  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()],
      const index_t (&strides)[OpType::Rank()])
  {
    return slice(op,
        detail::to_array(starts),
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    return slice(op, starts, ends, detail::NoStride{});
  }
  template <typename OpType>
  __MATX_INLINE__ auto slice( const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()])
  {
    return slice(op,
        detail::to_array(starts),
        detail::to_array(ends));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.
   *
   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @param strides Optional:  the stride between consecutive elements
   * @return sliced operator
   */
  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      const cuda::std::array<index_t, OpType::Rank()> &strides)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template Slice<N>(starts, ends, strides);
    } else {
      return detail::SliceOp<N,OpType,decltype(strides)>(op, starts, ends, strides);
    }
  }

  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends,
      [[maybe_unused]] ::matx::detail::NoStride no_stride)
  {
    if constexpr (is_tensor_view_v<OpType>) {
      return op.template Slice<N>(starts, ends);
    } else {
      return detail::SliceOp<N,OpType,detail::NoStride>(op, starts, ends, no_stride);
    }
  }


  template <int N, typename OpType>
    __MATX_INLINE__ auto slice( const OpType &op,
        const index_t (&starts)[OpType::Rank()],
        const index_t (&ends)[OpType::Rank()],
        const index_t (&strides)[OpType::Rank()])
  {
    return slice<N,OpType>(op,
        detail::to_array(starts),
        detail::to_array(ends),
        detail::to_array(strides));
  }

  /**
   * @brief Operator to logically slice a tensor or operator.
   *
   * The rank of the the operator must be greater than 0.

   * This operator can appear as an rvalue or lvalue.

   * The Rank template parameter N is optional when rank does not change
   *
   * @tparam N The Rank of the output operator - optional when slice produces same rank as input
   * @tparam OpType Input operator/tensor type
   * @param op Input operator
   * @param starts the first element (inclusive) of each dimension of the input operator.
   * @param ends the last element (exclusive) of each dimension of the input operator.  matxDrop Dim removes that dimension.  matxEnd deontes all remaining elements in that dimension.
   * @return sliced operator
   */
  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType &op,
      const cuda::std::array<index_t, OpType::Rank()> &starts,
      const cuda::std::array<index_t, OpType::Rank()> &ends)
  {
    return slice<N,OpType>(op, starts, ends, detail::NoStride{});
  }

  template <int N, typename OpType>
  __MATX_INLINE__ auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()])
  {
    return slice<N,OpType>(op,
        detail::to_array(starts),
        detail::to_array(ends));
  }

#else
   auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()]) { }

   auto slice (const OpType &op,
      const index_t (&starts)[OpType::Rank()],
      const index_t (&ends)[OpType::Rank()],
      const index_t (&strides)[OpType::Rank()]) { }
#endif
} // end namespace matx
