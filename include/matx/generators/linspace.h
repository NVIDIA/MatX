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

#include "matx/generators/range.h"

namespace matx
{
  namespace detail {
    template <class T, int NUM_RC> class LinspaceOp : public BaseOp<LinspaceOp<T, NUM_RC>> {
      private:
        cuda::std::array<T, NUM_RC> steps_;
        cuda::std::array<T, NUM_RC> firsts_;
        int axis_;
        index_t count_;
      public:
        using value_type = T;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "linspace"; }

        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { 
          if constexpr (NUM_RC == 1) {
            return 1; 
          }
          else {
            return 2;
          }
        }  

        inline LinspaceOp(const T (&firsts)[NUM_RC], const T (&lasts)[NUM_RC], index_t count, int axis) 
        {
          axis_ = axis;
          count_ = count;
          for (int i = 0; i < NUM_RC; ++i) {
            firsts_[i] = firsts[i];
            steps_[i] = (lasts[i] - firsts[i]) / static_cast<T>(count - 1);
          }
        }

        template <typename... Is>
        __MATX_DEVICE__ __MATX_HOST__ __MATX_INLINE__ T operator()(Is... indices) const { 
          static_assert(sizeof...(indices) == NUM_RC, "Number of indices incorrect in linspace");
          cuda::std::array idx{indices...};
          if constexpr (sizeof...(indices) == 1) {
            return firsts_[0] + steps_[0] * static_cast<T>(idx[0]);
          } else {
            if (axis_ == 0) {
              return firsts_[idx[1]] + steps_[idx[1]] * static_cast<T>(idx[0]);
            } else {
              return firsts_[idx[0]] + steps_[idx[0]] * static_cast<T>(idx[1]);
            }
          }
        }

      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
      {
        if constexpr (NUM_RC == 1) {
          return count_;
        } else {
          if (dim != axis_) {
            return NUM_RC;
          } else {
            return count_;
          }
        }
      }        
    };
  }


  /**
   * @brief Create a matrix linearly-spaced range of values
   *
   * Creates a set of values using starts and stops that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the count parameter
   * 
   * @tparam NUM_RC Number of rows or columns, depending on the axis
   * @tparam T Type of the values
   * @param firsts First values
   * @param lasts Last values
   * @param count Number of values in a row or column, depending on the axis
   * @param axis Axis to operate over
   * @return Operator with linearly-spaced values 
   */
  template <int NUM_RC, typename T = float>
  inline auto linspace(const T (&firsts)[NUM_RC], const T (&lasts)[NUM_RC], index_t count, int axis = 0)
  {
    return detail::LinspaceOp<T, NUM_RC>(firsts, lasts, count, axis);
  }   

  /**
   * @brief Create a linearly-spaced vector of values
   *
   * Creates a set of values using startsand stop that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the count parameter
   * 
   * @tparam T Type of the values
   * @param first First value
   * @param last Last value
   * @param count Number of values in a row or column, depending on the axis
   * @param axis Axis to operate over
   * @return Operator with linearly-spaced values 
   */
  template <typename T = float>
  inline auto linspace(T first, T last, index_t count, int axis = 0)
  {
    const T firsts[] = {first};
    const T lasts[] = {last};
    return linspace(firsts, lasts, count, axis);
  }

  /**
   * @brief Create a linearly-spaced range of values
   *
   * Creates a set of values using a start and end that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the shape and selected dimension.
   * 
   * @tparam Dim Dimension to operate over
   * @tparam NUM_RC Rank of shape
   * @tparam T Operator type
   * @param s Array of sizes
   * @param first First value
   * @param last Last value
   * @return Operator with linearly-spaced values 
   */
  template <int Dim, int NUM_RC, typename T>
  [[deprecated("Use matx::linspace(T first, T last, index_t count, int axis = 0) instead.")]]  
  inline auto linspace([[maybe_unused]]const index_t (&s)[NUM_RC], T first, T last)
  {
    const T firsts[] = {first};
    const T lasts[] = {last};   
    return linspace(firsts, lasts, NUM_RC, 0);
  }  

  /**
   * @brief Create a linearly-spaced range of values
   *
   * Creates a set of values using a start and end that are linearly-
   * spaced apart over the set of values. Distance is determined
   * by the shape and selected dimension.
   * 
   * @tparam T Operator type
   * @tparam Dim Dimension to operate over
   * @tparam ShapeType Shape type
   * @param s Shape object
   * @param first First value
   * @param last Last value
   * @return Operator with linearly-spaced values 
   */
  template <int Dim, typename ShapeType, typename T,
           std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
  [[deprecated("Use matx::linspace(T first, T last, index_t count, int axis = 0) instead.")]]           
  inline auto linspace(ShapeType &&s, T first, T last)
  {
    constexpr int NUM_RC = cuda::std::tuple_size<std::decay_t<ShapeType>>::value;
    static_assert(NUM_RC > Dim);
    auto count =  *(s.begin() + Dim);
    const T firsts[] = {first};
    const T lasts[] = {last};       
    return linspace(firsts, lasts, count, 0);
  }  
} // end namespace matx
