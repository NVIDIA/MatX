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
/**
   * logically reshapes dimensions of a tensor/operator
   * TotalSize for reshape and input operator must match
   */
  namespace detail {
    template <int RANK, typename T, typename ShapeType>
      class ReshapeOp : public BaseOp<ReshapeOp<RANK, T, ShapeType>>
    {
      public:
        using value_type = typename T::value_type;

      private:
        mutable typename detail::base_type_t<T> op_;
	      ShapeType sizes_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using self_type = ReshapeOp<RANK, T, ShapeType>;

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          std::string sizes_str;
          for (int i = 0; i < Rank(); i++) {
            sizes_str += std::to_string(sizes_[i]);
            if (i < Rank() - 1) sizes_str += "_";
          }
          return "JITReshape_sizes" + sizes_str;
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          
          std::string sizes_array = "{ ";
          for (int i = 0; i < Rank(); i++) {
            sizes_array += std::to_string(sizes_[i]);
            if (i < Rank() - 1) sizes_array += ", ";
          }
          sizes_array += " }";

          std::string op_sizes_array = "{ ";
          for (int i = 0; i < T::Rank(); i++) {
            op_sizes_array += std::to_string(op_.Size(i));
            if (i < T::Rank() - 1) op_sizes_array += ", ";
          }
          op_sizes_array += " }";
          
          return cuda::std::make_tuple(
            func_name,
            std::string("template <typename T> struct " + func_name + " {\n") +
                "  using value_type = typename T::value_type;\n" +
                "  using matxop = bool;\n" +
                "  constexpr static int Rank_ = " + std::to_string(Rank()) + ";\n" +
                "  constexpr static int OpRank_ = " + std::to_string(T::Rank()) + ";\n" +
                "  constexpr static cuda::std::array<index_t, Rank_> sizes_ = " + sizes_array + ";\n" +
                "  constexpr static cuda::std::array<index_t, OpRank_> op_sizes_ = " + op_sizes_array + ";\n" +
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n" +
                "  template <typename CapType, typename... Is>\n" +
                "  __MATX_INLINE__ __MATX_DEVICE__ decltype(auto) operator()(Is... indices) const\n" +
                "  {\n" +
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {\n" +
                "      cuda::std::array<index_t, Rank_> inds{indices...};\n" +
                "      cuda::std::array<index_t, OpRank_> ninds;\n" +
                "      index_t idx = 0;\n" +
                "      index_t stride = 1;\n" +
                "      for(int i = Rank_ - 1 ; i >= 0 ; i--) {\n" +
                "        idx += stride * inds[i];\n" +
                "        stride *= sizes_[i];\n" +
                "      }\n" +
                "      for(int i = OpRank_ - 1; i >= 0; i--) {\n" +
                "        ninds[i] = idx % op_sizes_[i];\n" +
                "        idx /= op_sizes_[i];\n" +
                "      }\n" +
                "      return get_value<CapType>(op_, ninds);\n" +
                "    } else {\n" +
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{};\n" +
                "    }\n" +
                "  }\n" +
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() { return Rank_; }\n" +
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int32_t dim) const { return sizes_[dim]; }\n" +
                "};\n"
          );
        }
#endif

        __MATX_INLINE__ std::string str() const { return "reshape(" + op_.str() + ")"; }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return RANK;
        }

        static_assert(Rank() > 0, "ReshapeOp: Rank of operator must be greater than 0.");
        static_assert(T::Rank() > 0, "ReshapeOp: Rank of input operator must be greater than 0.");

        __MATX_INLINE__ ReshapeOp(const T &op, ShapeType &&s) : op_(op), sizes_(s) {

          index_t size = 1;

          for(int32_t i = 0; i < Rank(); i++) {
            size *= sizes_[i];
          }

          MATX_ASSERT_STR(size == TotalSize(op_), matxInvalidSize, "ReshapeOp: TotalSize of reshape must match");
          MATX_LOG_TRACE("{} constructor: rank={}, total_size={}", str(), Rank(), size);
        };

        template <typename CapType, typename Op, typename... Is>
        static __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) get_impl(Op&& op, const decltype(sizes_) &sizes, Is... indices)
        {   
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            cuda::std::array<index_t, Rank()> inds{indices...};
            cuda::std::array<index_t, T::Rank()> ninds;

            index_t idx = 0;
            index_t stride = 1;

            // linearlize incoming index
MATX_LOOP_UNROLL
            for(int i = Rank() - 1 ; i >= 0 ; i--) {
              idx += stride * inds[i];
              stride *= sizes[i];
            }

            // extract new indices
  MATX_LOOP_UNROLL
            for(int i = T::Rank() - 1; i >= 0; i--) {
              ninds[i] = idx % op.Size(i);
              idx /= op.Size(i);
            }

            return get_value<CapType>(cuda::std::forward<Op>(op), ninds);       
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const 
        {
          return get_impl<CapType>(cuda::std::as_const(op_), sizes_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return get_impl<CapType>(cuda::std::forward<decltype(op_)>(op_), sizes_, indices...);
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices)
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
#ifdef MATX_EN_JIT
            const auto op_jit_name = detail::get_operator_capability<Cap>(op_, in);
            return get_jit_class_name() + "<" + op_jit_name + ">";
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
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          return sizes_[dim];
        }

        template <typename S2, typename Executor>
        __MATX_INLINE__ void PreRun(S2 &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PreRun(std::forward<S2>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename S2, typename Executor>
        __MATX_INLINE__ void PostRun(S2 &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T>()) {
            op_.PostRun(std::forward<S2>(shape), std::forward<Executor>(ex));
          }
        }

        ~ReshapeOp() = default;
        ReshapeOp(const ReshapeOp &rhs) = default;

        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R>
        __MATX_INLINE__ auto operator=(const R &rhs) {
          return set(*this, rhs);
        }
    };
  }

    /**
   * @brief Operator to reshape a tensor or operator.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam RANK the reshaped rank
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param s  the size of each reshaped dimension
   * @return reshaped operator
   */
  template <int RANK, typename T, typename ShapeType>
    requires (!cuda::std::is_array_v<remove_cvref_t<ShapeType>>)
  __MATX_INLINE__ auto reshape(const T &op, ShapeType &&s)
  {
    return detail::ReshapeOp<RANK, T, ShapeType>(op, std::forward<ShapeType>(s));
  }

    /**
   * @brief Operator to reshape a tensor or operator.
   *
   * This operator can appear as an rvalue or lvalue.
   *
   * @tparam RANK the reshaped rank
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param sizes the size of each reshaped dimension
   * @return reshaped operator
   */
  template <int RANK, typename T>
    __MATX_INLINE__ auto reshape( const T &op,
        const index_t (&sizes)[RANK]) {
      return reshape<RANK, T>(op, detail::to_array(sizes));
    }
} // end namespace matx
