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
   * Upsamples a signal by stuffing zeros
   */
  namespace detail {
    template <typename T>
      class UpsampleOp : public BaseOp<UpsampleOp<T>>
    {
      private:
        mutable typename detail::base_type_t<T> op_;
        int32_t dim_;
        index_t n_;

      public:
        using matxop = bool;
        using matxoplvalue = bool;
        using value_type = typename T::value_type;
        using self_type = UpsampleOp<T>;

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return T::Rank();
        }        

        static_assert(Rank() > 0, "UpsampleOp: Rank of operator must be greater than 0.");

#ifdef MATX_EN_JIT
        struct JIT_Storage {
          typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;
        };

        JIT_Storage ToJITStorage() const {
          return JIT_Storage{detail::to_jit_storage(op_)};
        }

        __MATX_INLINE__ std::string get_jit_class_name() const {
          return std::format("JITUpsample_dim{}_n{}", dim_, n_);
        }

        __MATX_INLINE__ auto get_jit_op_str() const {
          std::string func_name = get_jit_class_name();
          cuda::std::array<index_t, Rank()> out_dims_;
          for (int i = 0; i < Rank(); ++i) {
            out_dims_[i] = Size(i);
          }
          
          return cuda::std::make_tuple(
            func_name,
            std::format("template <typename T> struct {} {{\n"
                "  using value_type = typename T::value_type;\n"
                "  using matxop = bool;\n"
                "  constexpr static int32_t dim_ = {};\n"
                "  constexpr static index_t n_ = {};\n"
                "  constexpr static int Rank_ = {};\n"
                "  constexpr static cuda::std::array<index_t, Rank_> out_dims_ = {{ {} }};\n"
                "  typename detail::inner_storage_or_self_t<detail::base_type_t<T>> op_;\n"
                "  template <typename CapType, typename... Is>\n"
                "  __MATX_INLINE__ __MATX_DEVICE__ auto operator()(Is... indices) const\n"
                "  {{\n"
                "    if constexpr (CapType::ept == ElementsPerThread::ONE) {{\n"
                "      cuda::std::array<index_t, Rank_> ind{{indices...}};\n"
                "      if ((ind[dim_] % n_) == 0) {{\n"
                "        ind[dim_] /= n_;\n"
                "        return get_value<CapType>(op_, ind);\n"
                "      }}\n"
                "      return static_cast<value_type>(0);\n"
                "    }} else {{\n"
                "      return Vector<value_type, static_cast<index_t>(CapType::ept)>{{}};\n"
                "    }}\n"
                "  }}\n"
                "  static __MATX_INLINE__ constexpr __MATX_DEVICE__ int32_t Rank() {{ return Rank_; }}\n"
                "  constexpr __MATX_INLINE__ __MATX_DEVICE__ index_t Size(int32_t dim) const {{ return out_dims_[dim]; }}\n"
                "}};\n",
                func_name, dim_, n_, Rank(), detail::array_to_string(out_dims_))
          );
        }
#endif 

        __MATX_INLINE__ std::string str() const { return "upsample(" + op_.str() + ")"; }

        __MATX_INLINE__ UpsampleOp(const T &op, int32_t dim, index_t n) : op_(op), dim_(dim), n_(n) {
          MATX_LOG_TRACE("{} constructor: dim={}, n={}, rank={}", str(), dim, n, Rank());
        };

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
              static_assert(sizeof...(Is)==Rank());
              static_assert((std::is_convertible_v<Is, index_t> && ... ));

              // convert variadic type to tuple so we can read/update
              cuda::std::array<index_t, Rank()> ind{indices...};
              if ((ind[dim_] % n_) == 0) {
                ind[dim_] /= n_;
                return get_value<CapType>(op_, ind);
              }

            return static_cast<typename decltype(op_)::value_type>(0);
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
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
        } else {
          auto self_has_cap = capability_attributes<Cap>::default_value;
          return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
        }
      }

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int32_t dim) const
        {
          if (dim == dim_) {
            return op_.Size(dim) * n_;
          }
          else {
            return op_.Size(dim);
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
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

        ~UpsampleOp() = default;
        UpsampleOp(const UpsampleOp &rhs) = default;
        
        __MATX_INLINE__ auto operator=(const self_type &rhs) {
          return set(*this, rhs);
        }

        template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { 
          return set(*this, rhs); 
        }
    };
  }

  /**
   * @brief Operator to upsample an operator by inserting zeros between values
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dim the factor to upsample
   * @param n Upsample rate
   * @return Upsampled operator
   */
  template <typename T>
  __MATX_INLINE__ auto upsample( const T &op, int32_t dim, index_t n) {
    return detail::UpsampleOp<T>(op, dim, n);
  }

  /**
   * @brief Operator to downsample an operator by dropping samples
   *
   * @tparam T Input operator/tensor type
   * @param op Input operator
   * @param dim the factor to downsample
   * @param n Downsample rate
   * @return Downsample operator
   */
  template <typename T>
  __MATX_INLINE__ auto downsample( const T &op, int32_t dim, index_t n) {
    index_t starts[T::Rank()];
    index_t ends[T::Rank()];
    index_t strides[T::Rank()];

    for (int32_t r = 0; r < T::Rank(); r++) {
      starts[r]  = 0;
      ends[r]    = matxEnd;
      strides[r] = (r == dim) ? n : 1;
    }

    return slice(op, starts, ends, strides);
  }
} // end namespace matx
