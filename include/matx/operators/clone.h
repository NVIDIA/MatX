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
#include "matx/operators/base_operator.h"

namespace matx
{
  namespace detail {
    template <std::size_t CRank, typename T>
      class CloneOp : public BaseOp<CloneOp<CRank, T>>
    {
      static_assert(CRank > T::Rank(), "Clone rank must be higher than input rank");
      private:
        mutable typename detail::base_type_t<T> op_;
        cuda::std::array<index_t, CRank> sizes_;         // size of each dimension after cloning
        cuda::std::array<index_t, T::Rank()> dims_;      // gather map for computing operator() indices
      public:
        using matxop = bool;

        using value_type = typename T::value_type;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string sizes_str, dims_str;
          for (size_t i = 0; i < CRank; i++) {
            sizes_str += std::to_string(sizes_[i]);
            if (i < CRank - 1) sizes_str += "_";
          }
          for (size_t i = 0; i < T::Rank(); i++) {
            dims_str += std::to_string(dims_[i]);
            if (i < T::Rank() - 1) dims_str += "_";
          }
          return std::format("JITClone_sizes{}_dims{}", sizes_str, dims_str);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static int OpRank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> sizes_ = {{ {} }};\n"
                "  constexpr static cuda::std::array<index_t, OpRank_> dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array<index_t, Rank_> sind{{indices...}};\n"
                "      cuda::std::array<index_t, OpRank_> gind;\n"
                "      for(int i = 0; i < OpRank_; i++) {{\n"
                "        auto idx = dims_[i];\n"
                "        gind[i] = sind[idx];\n"
                "      }}\n"
                "      return get_value<CapType>(op_, gind);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int dim) const {{ return sizes_[dim]; }}\n"
                "}};\n",
                func_name, CRank, T::Rank(), detail::array_to_string(sizes_), detail::array_to_string(dims_))
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "clone(" + op_.str() + ")"; }

        __MATX_INLINE__ CloneOp(const T &op, cuda::std::array<index_t, CRank> shape) : op_(op) {
          static_assert(T::Rank() < CRank, "Cloning rank must be higher than input operator rank");

          [[maybe_unused]] const index_t num_keep = static_cast<index_t>(
			  std::count_if(shape.begin(), shape.end(), [](index_t i) { return i == matxKeepDim; }));
          MATX_ASSERT_STR(num_keep == T::Rank(), matxInvalidParameter,
            "Number of matxKeepDim in a clone must match input operator rank");

          // create gather list
          int d = 0;
          for(int i = 0; i < Rank(); i++) {
            if constexpr (T::Rank() > 0) { // This is needed since the compiler can be fooled
              if(shape[i] == matxKeepDim) {
                sizes_[i] = op_.Size(d);
                // gcc incorrectly shows an invalid access to array element [1] in a unit test here. This is not
                // possible based on runtime checks we have. Disable the warning temporarily.
MATX_IGNORE_WARNING_PUSH_GCC("-Warray-bounds")
                dims_[d++] = i;
MATX_IGNORE_WARNING_POP_GCC
              } else {
                sizes_[i] = shape[i];
              }
            }
            else {
              MATX_ASSERT(shape[i] != matxKeepDim, matxInvalidDim);
              sizes_[i] = shape[i];
            }
          }
          MATX_ASSERT(d == T::Rank(), matxInvalidDim);
          MATX_LOG_TRACE("{} constructor: input_rank={}, output_rank={}", str(), T::Rank(), CRank);
        }

        template <typename CapType, typename Op, typename Dims, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Dims &dims, Is... indices)
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
  MATX_IGNORE_WARNING_PUSH_GCC("-Wmaybe-uninitialized")
            cuda::std::array<index_t, Rank()> sind{indices...};
            cuda::std::array<index_t, T::Rank()> gind;
  MATX_IGNORE_WARNING_POP_GCC

            // gather indices
            for(int i = 0; i < T::Rank(); i++) {
              auto idx = dims[i];
              gind[i] = sind[idx];
            }

            return get_value<CapType>(cuda::std::forward<Op>(op), gind);
          }
          else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename Op, typename Dims, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const Dims &dims, Is... indices)
        {
          return get_impl<detail::ElementsPerThread::ONE>(cuda::std::forward<Op>(op), dims, indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return get_impl<CapType>(cuda::std::as_const(op_), dims_, indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), dims_, indices...);
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
          return CRank;
        }
        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
        {
          return sizes_[dim];
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
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
          else if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }
    };
  }


  /**
   * @brief Operator to clone an operator or tensor across dimensions
   *
   * @tparam Rank the rank of the cloned operator
   * @tparam T source operator/tensor type
   * @param t source operator/tensor
   * @param shape the shape of the cloned operator/tensor.
   * Each element is either the size of the cloned dimension or `matxKeepDim` to be from the source tensor
   * @return operator to compute the cloned value
   */
  template <std::size_t Rank, typename Op>
  auto __MATX_INLINE__ clone(const Op &t, const cuda::std::array<index_t, Rank> &shape)
  {
    static_assert(Rank >= Op::Rank(), "Cloning rank must be >= input operator rank");

    if constexpr (Op::Rank() == Rank) {
      return t; // No-op to same rank
    }
    else if constexpr (is_tensor_view_v<Op>) {
      return t.template Clone<static_cast<int>(Rank)>(shape);
    } else {
      return detail::CloneOp<static_cast<int>(Rank), Op>(t, shape);

    }
  };

  template <int Rank, typename Op>
  auto __MATX_INLINE__ clone(const Op &t, const index_t (&shape)[Rank])
  {
    return clone<Rank, Op>(t, detail::to_array(shape));
  };


} // end namespace matx
