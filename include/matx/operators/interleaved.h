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
        typename base_type<T1>::type op_;

      public:
        using matxop = bool;
        using scalar_type = typename T1::scalar_type;

        using complex_type = std::conditional_t<is_matx_half_v<scalar_type>,
              matxHalfComplex<scalar_type>,
              cuda::std::complex<scalar_type>>;
        __MATX_INLINE__ std::string str() const { return "interleaved(" + op_.str() + ")"; }

        __MATX_INLINE__ ComplexInterleavedOp(T1 op) : op_(op) {
          static_assert(!is_complex_v<extract_scalar_type_t<T1>>, "Complex interleaved op only works on scalar input types");
          static_assert(Rank() > 0);
        };

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ complex_type operator()(Is... indices) const 
        {
          auto real = op_(indices...);

          constexpr size_t rank_idx = (Rank() == 1) ? 0 : (Rank() - 2);
          auto tup = cuda::std::make_tuple(indices...);
          cuda::std::get<rank_idx>(tup) += op_.Size(rank_idx) / 2;

          auto imag = cuda::std::apply(op_, tup);
          return complex_type{real, imag};
        }

        template <typename... Is>
        __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ complex_type operator()(Is... indices) 
        {
          auto real = op_(indices...);

          constexpr size_t rank_idx = (Rank() == 1) ? 0 : (Rank() - 2);
          auto tup = cuda::std::make_tuple(indices...);
          cuda::std::get<rank_idx>(tup) += op_.Size(rank_idx) / 2;

          auto imag = cuda::std::apply(op_, tup);
          return complex_type{real, imag};
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
    auto interleaved(T1 t)
    {
      static_assert(!is_complex_v<extract_scalar_type_t<T1>>, "Input to interleaved operator must be real-valued");
      return detail::ComplexInterleavedOp<T1>(t);
    }
} // end namespace matx
