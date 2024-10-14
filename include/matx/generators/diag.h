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
    template <typename T, typename ShapeType> 
      class Diag : public BaseOp<Diag<T, ShapeType>>{
      static constexpr int RANK = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;

      private:
      ShapeType s_;
      T val_;

      public:
      // dummy type to signal this is a matxop
      using matxop = bool;
      using value_type = T;

       __MATX_INLINE__ std::string str() const { return "diag"; }

      Diag(ShapeType &&s, T val) : s_(std::forward<ShapeType>(s)), val_(val)
      {
        if constexpr (!is_noshape_v<ShapeType>) {
          static_assert(Rank() > 1, "Diagonal generator must be used with an operator of rank 1 or higher");
        }
      };

      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
        if (((pp_get<0>(indices...) == indices) && ...)) {
          return T(val_);
        }
        else {
          return T(0.0f);
        }
      }

      constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
      {
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

  /**
   * Creates a diagonal tensor with a given value on the diagonals
   *
   * diag returns a given value on all elements on the diagonals of a tensor, and
   * 0 otherwise. In other words, if the index of every dimension is the same, the
   * value is returned, otherwise a zero is returned. This version of diag() is
   * shapeless and can be indexed using any input dimension and it will return a
   * valid value. It is preferred to use it over the shaped versions if the shape
   * can be deduced by other contexts.
   *
   * @tparam T Data type
   * 
   * @param val Value to return
   *
   */
  template <typename T = int, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
  inline auto diag(T val)
  {
    return detail::Diag<T, detail::NoShape>(detail::NoShape{}, T(val));
  }

  /**
   * Creates a diagonal tensor with a given value on the diagonals
   *
   * diag returns a given value on all elements on the diagonals of a tensor, and
   * 0 otherwise. In other words, if the index of every dimension is the same, the
   * value is returned, otherwise a zero is returned.
   *
   * @tparam T
   *   Data type
   *
   */
  template <typename T = int, typename ShapeType,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type> &&
                   !is_matx_op<ShapeType>(), bool> = true>
  inline auto diag(ShapeType &&s, T val)
  {
    return detail::Diag<T, ShapeType>(std::forward<ShapeType>(s), val);
  }

  /**
   * Creates a diagonal tensor with a given value on the diagonals
   *
   * diag returns a given value on all elements on the diagonals of a tensor, and
   * 0 otherwise. In other words, if the index of every dimension is the same, the
   * value is returned, otherwise a zero is returned.
   *
   * @tparam T Data type
   * @tparam RANK Rank of input
   * 
   * @param s Array of operator dimensions
   * @param val Value to return
   *
   */
  template <typename T = int, int RANK>
  inline auto diag(const index_t (&s)[RANK], T val)
  {
    return diag(detail::to_array(s), val);
  }

  /**
   * Creates an identity patterns on the tensor
   *
   * eye returns 1 on all elements on the diagonals of a tensor, and 0 otherwise.
   * In other words, if the index of every dimension is the same, a 1 is returned,
   * otherwise a zero is returned.
   *
   * @tparam T
   *   Data type
   *
   */
  template <typename T = int, typename ShapeType,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
  inline auto eye(ShapeType &&s)
  {
    return detail::Diag<T, ShapeType>(std::forward<ShapeType>(s), T(1));
  }

  /**
   * Creates an identity patterns on the tensor
   *
   * eye returns 1 on all elements on the diagonals of a tensor, and 0 otherwise.
   * In other words, if the index of every dimension is the same, a 1 is returned,
   * otherwise a zero is returned.
   *
   * @tparam T
   *   Data type
   *
   */
  template <typename T = int, int RANK>
  inline auto eye(const index_t (&s)[RANK])
  {
    return eye<T>(detail::to_array(s));
  }

  /**
   * Creates an identity generator
   *
   * eye() returns 1 on all elements on the diagonals of a tensor, and 0 otherwise.
   * In other words, if the index of every dimension is the same, a 1 is returned,
   * otherwise a zero is returned. This version of eye() is shapeless and can be indexed 
   * using any input dimension and it will return a valid value. It is preferred to use
   * it over the shaped versions if the shape can be deduced by other contexts.
   *
   * @tparam T Data type
   *
   */
  template <typename T = int>
  inline auto eye()
  {
    return detail::Diag<T, detail::NoShape>(detail::NoShape{}, T(1));
  }
} // end namespace matx
