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


namespace matx
{
  namespace detail {
    template <typename T, typename ShapeType> class ConstVal : public BaseOp<ConstVal<T,ShapeType>> {
      static constexpr int RANK = cuda::std::tuple_size<typename remove_cvref<ShapeType>::type>::value;

      private:
      ShapeType s_;
      T v_;

      public:
      // dummy type to signal this is a matxop
      using matxop = bool;
      using value_type = T;

      __MATX_INLINE__ std::string str() const { return  "constval"; }
      ConstVal(ShapeType &&s, T val) : s_(std::forward<ShapeType>(s)), v_(val){};

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ T operator()(Is...) const { 
        return v_; };

      constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const {
        if constexpr (!is_noshape_v<ShapeType>) {
          return *(s_.begin() + dim);
        }
        else {
          return index_t(0);
        }
      }
      static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { 
        if constexpr (!is_noshape_v<ShapeType>) {
          return RANK;
        }
        else {
          return matxNoRank;
        }
      }
    };
  }
} // end namespace matx
