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
    template <typename T1>
      class ComplexInterleavedOp : public BaseOp<ComplexInterleavedOp<T1>>
    {
      private:
        typename detail::base_type_t<T1> op_;

      public:
        using matxop = bool;
        using value_type = typename T1::value_type;

        using complex_type = std::conditional_t<is_matx_half_v<value_type>,
              matxHalfComplex<value_type>,
              cuda::std::complex<value_type>>;
        __MATX_INLINE__ std::string str() const { return "interleaved(" + op_.str() + ")"; }

        __MATX_INLINE__ ComplexInterleavedOp(const T1 &op) : op_(op) {
          MATX_LOG_TRACE("{} constructor: rank={}", str(), Rank());
          static_assert(!is_complex_v<extract_value_type_t<T1>>, "Complex interleaved op only works on scalar input types");
          static_assert(Rank() > 0);
        };
 

        template <typename CapType, typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            auto real = get_value<DefaultCapabilities>(op_, indices...);

            constexpr size_t rank_idx = (Rank() == 1) ? 0 : (Rank() - 2);
            cuda::std::array idx{indices...};
            idx[rank_idx] += op_.Size(rank_idx) / 2;

            auto imag = get_value<DefaultCapabilities>(op_, idx);
            return complex_type{real, imag};
          } else {
            return Vector<value_type, static_cast<index_t>(CapType::ept)>{};
          }
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const 
        {
          return this->operator()<DefaultCapabilities>(indices...);
        }

        static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
        {
          return detail::get_rank<T1>();
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }

        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
        {
          if constexpr (is_matx_op<T1>()) {
            op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
          }
        }          

        constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const noexcept
        {
          if constexpr (Rank() <= 1)
          {
            return op_.Size(dim) / 2;
          }

          return (dim == static_cast<int>(Rank()) - 2) ? op_.Size(dim) / 2
            : op_.Size(dim);
        }

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType& in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(op_, in));
          }
          else if constexpr (Cap == OperatorCapability::ALIASED_MEMORY) {
            auto in_copy = in;
            in_copy.permutes_input_output = true;
            return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(op_, in_copy));
          }
          else {
            auto self_has_cap = capability_attributes<Cap>::default_value;
            return combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(op_, in));
          }
        }
    };
  }

  /**
   * Perform an interleaved layout shift from a complex planar input
   *
   * Takes aplanar complex layout (real1, real2, ... realN, imag1, ... imagN). and
   * transforms it into interleaved format: (real1, imag1, real2, ...). This is
   * mostly used for tensor core CGEMM which expects planar inputs. The indexing
   * on the new layout will be half as many elements as complex elements since
   * real/imaginary are separated in planar. If the rank is higher than 2, the
   * conversion is treated as a batched transform and only the inner two dims are
   * converted.
   *
   * @tparam T1
   *   Type of View/Op
   * @param t
   *   View/Op to shift
   *
   */
  template <typename T1>
    auto interleaved(const T1 &t)
    {
      static_assert(!is_complex_v<extract_value_type_t<T1>>, "Input to interleaved operator must be real-valued");
      return detail::ComplexInterleavedOp<T1>(t);
    }
} // end namespace matx
