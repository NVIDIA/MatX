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
#include "matx/operators/permute.h"

namespace matx
{


  namespace detail {

    template <typename T1, int RANK, int AXIS> 
      class MeshGridOp : public BaseOp<MeshGridOp<T1, RANK, AXIS>> {
        private:
          T1 t1_;
          cuda::std::array<index_t, RANK> shape_;
          int idx_;

        public:
          using matxop = bool;
          using value_type = typename T1::value_type;
          //typedef typename T1::value_type value_type;

          __MATX_INLINE__ std::string str() const { return "meshgrid"; }

          __MATX_INLINE__ MeshGridOp(T1 t1, cuda::std::array<index_t, RANK> shape, int idx) : t1_(t1), shape_(shape), idx_(idx) {
            static_assert(shape.size() == RANK );
            static_assert(is_matx_op<T1>());
          }



          template <typename CapType, typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
#ifdef __CUDA_ARCH__
            if constexpr (CapType::jit) {
              if ((threadIdx.x * CapType::ept) >= Size(Rank() - 1)) {
                return detail::GetJitSentinelValue<CapType, value_type>();
              }
            }
#endif
            cuda::std::array<index_t, RANK> idx = {indices...};
            return idx[idx_];
          }

          template <OperatorCapability Cap, typename InType>
          __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] const InType &in) const {
            if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
              const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
              return combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(t1_, in));
            } else {            
              auto self_has_cap = detail::capability_attributes<Cap>::default_value;
              return detail::combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(t1_, in));
            }
          }          

          template <typename... Is>
          __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
            return this->operator()<DefaultCapabilities>(indices...);
          }

          __MATX_INLINE__  __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
            return shape_[dim];
          }

          template <typename ShapeType, typename Executor>
          __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept
          {
            if constexpr (is_matx_op<T1>()) {
              t1_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            }
          }

          template <typename ShapeType, typename Executor>
          __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept
          {
            if constexpr (is_matx_op<T1>()) {
              t1_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
            }
          }            

          static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }
      };

    template <typename... Ts, int... I>
      __MATX_INLINE__  auto meshgrid_impl(std::integer_sequence<int, I...>, Ts&... ts) {
        auto constexpr RANK = (int)sizeof...(Ts);

        // construct shape from size of each rank1 tensor
        cuda::std::array<index_t, RANK> shape{ts.Size(0)...};

        // Python XY indexing is reverse from natural tensor indexing.  We permute here to match
        int perm[RANK];
        for(int i=0; i < RANK; i++) {
          perm[RANK-i-1] = i;
        }

        // return one meshgrid operator per rank
        return cuda::std::tuple{ permute(MeshGridOp<Ts, RANK, I>{ts, shape, I}, perm)...};
      }

  } // end namespace detail

  /**
   * Creates mesh grid operators
   *
   *
   * @tparam Ts...
   *   list of N Rank1 operators types defining mesh points
   * @param ts
   *   list of N Rank1 operators
   *
   * @returns N RankN operators of mesh grid.  Grids are returned in cartisian indexing 
   * (transpose of matrix indexing)
   *
   */
  template <typename... Ts>
    __MATX_INLINE__  auto meshgrid(Ts&&... ts) {

      // generating index sequence that we can convert to axis
      return detail::meshgrid_impl(std::make_integer_sequence<int, (int)sizeof...(Ts)>{}, ts...);
    }

} // end namespace matx
