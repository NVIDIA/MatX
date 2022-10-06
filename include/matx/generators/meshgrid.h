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
    template <typename T> class Meshgrid_X {
      private:
        std::array<T, 3> x_;
        std::array<T, 3> y_;

      public:
        // dummy type to signal this is a matxop
        using matxop = bool;
        using scalar_type = T;

        __MATX_INLINE__ std::string str() { return "meshgridx"; }	

        Meshgrid_X(std::array<T, 3> x, std::array<T, 3> y) : x_(x), y_(y) {}

        inline __MATX_DEVICE__ T operator()(index_t i, index_t j) const
        {
          return x_[0] + j * (x_[1] - x_[0]) / (x_[2] - 1);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
        {
          return (dim == 0) ? y_[2] : x_[2];
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 2; }
    };

    template <typename T> class Meshgrid_Y {
      private:
        std::array<T, 3> x_;
        std::array<T, 3> y_;

      public:
        // dummy type to signal this is a matxop
        using matxop = bool;
        using scalar_type = T;
        
	__MATX_INLINE__ std::string str() { return "meshgridy"; }	

        Meshgrid_Y(std::array<T, 3> x, std::array<T, 3> y) : x_(x), y_(y) {}

        inline __MATX_DEVICE__ T operator()(index_t i, index_t j) const
        {
          return y_[0] + i * (y_[1] - y_[0]) / (y_[2] - 1);
        };

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ index_t Size(uint32_t dim) const
        {
          return (dim == 0) ? y_[2] : x_[2];
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 2; }
    };
  }

  /**
   * Creates an mesh grid X matrix
   *
   *
   * @tparam T
   *   Data type
   *
   */
  template <typename T = int>
    inline auto meshgrid_x(const std::array<T, 3> &x, const std::array<T, 3> &y)
    {
      return detail::Meshgrid_X<T>(x, y);
    }

  /**
   * Creates an mesh grid Y matrix
   *
   *
   * @tparam T
   *   Data type
   *
   */
  template <typename T = int>
    inline auto meshgrid_y(const std::array<T, 3> &x, const std::array<T, 3> &y)
    {
      return detail::Meshgrid_Y<T>(x, y);
    } 
} // end namespace matx
