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

#include "matx/generators/linspace.h"

namespace matx
{
  enum class ChirpMethod {
    CHIRP_METHOD_LINEAR
  };

  enum class ChirpType {
    CHIRP_TYPE_REAL,
    CHIRP_TYPE_COMPLEX
  };

  namespace detail {
    template <typename SpaceOp, typename FreqType> 
      class Chirp : public BaseOp<Chirp<SpaceOp, FreqType>> {
        using space_type = typename SpaceOp::value_type;


        private:
        SpaceOp sop_;
        FreqType f0_;
        FreqType f1_;
        space_type t1_;
        ChirpMethod method_;

        public:
        using value_type = FreqType;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "chirp"; }

        inline __MATX_HOST__ __MATX_DEVICE__ Chirp(SpaceOp sop, FreqType f0, space_type t1, FreqType f1, ChirpMethod method) : 
          sop_(sop),
          f0_(f0),
          f1_(f1),          
          t1_(t1),
          method_(method)
        {}

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return detail::combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(sop_, in));
          } else {        
            auto self_has_cap = detail::capability_attributes<Cap>::default_value;
            return detail::combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(sop_, in));
          }
        }        

        template <typename CapType>
        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return detail::ApplyGeneratorVecFunc<CapType, FreqType>([this](index_t idx) { 
            if (method_ == ChirpMethod::CHIRP_METHOD_LINEAR) {
              return cuda::std::cos(2.0f * M_PI * (f0_ * sop_(idx) + 0.5f * ((f1_ - f0_) / t1_) * sop_(idx) * sop_(idx)));
            }

            return 0.0; 
          }, i);
        }

        inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
        {
          return this->operator()<DefaultCapabilities>(i);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return sop_.Size(0);
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }  
      };

    template <typename SpaceOp, typename FreqType> 
      class ComplexChirp  : public BaseOp<ComplexChirp<SpaceOp, FreqType>> {
        using space_type = typename SpaceOp::value_type;


        private:
        SpaceOp sop_;
        FreqType f0_;
        FreqType f1_;
        space_type t1_;
        ChirpMethod method_;

        public:
        using value_type = cuda::std::complex<FreqType>;
        using matxop = bool;
        
	__MATX_INLINE__ std::string str() const { return "cchirp"; }
        
	inline __MATX_HOST__ __MATX_DEVICE__ ComplexChirp(SpaceOp sop, FreqType f0, space_type t1, FreqType f1, ChirpMethod method) : 
          sop_(sop),
          f0_(f0),
          f1_(f1),
          t1_(t1),          
          method_(method)
        {}

        template <OperatorCapability Cap, typename InType>
        __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const {
          if constexpr (Cap == OperatorCapability::ELEMENTS_PER_THREAD) {
            const auto my_cap = cuda::std::array<ElementsPerThread, 2>{ElementsPerThread::ONE, ElementsPerThread::ONE};
            return detail::combine_capabilities<Cap>(my_cap, detail::get_operator_capability<Cap>(sop_, in));
          } else {        
            auto self_has_cap = detail::capability_attributes<Cap>::default_value;
            return detail::combine_capabilities<Cap>(self_has_cap, detail::get_operator_capability<Cap>(sop_, in));
          }
        }

        template <typename CapType>
        inline __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(index_t i) const
        {
          return detail::ApplyGeneratorVecFunc<CapType, value_type>([this](index_t idx) { 
            if (method_ == ChirpMethod::CHIRP_METHOD_LINEAR) {
              FreqType real = cuda::std::cos(2.0f * M_PI * (f0_ * sop_(idx) + 0.5f * ((f1_ - f0_) / t1_) * sop_(idx) * sop_(idx)));
              FreqType imag = -cuda::std::cos(2.0f * M_PI * (f0_ * sop_(idx) + 0.5f * ((f1_ - f0_) / t1_) * sop_(idx) * sop_(idx) + 90.0/360.0));
              return cuda::std::complex<FreqType>{real, imag};
            }

            return cuda::std::complex<FreqType>{0, 0};
          }, i);
        }

        inline __MATX_HOST__ __MATX_DEVICE__ decltype(auto) operator()(index_t i) const
        {
          return this->operator()<DefaultCapabilities>(i);
        }

        constexpr inline __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
        {
          return sop_.Size(0);
        }
        static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }  
      };
  }

  /**
   * Creates a real chirp signal (swept-frequency cosine)
   * 
   * SpaceOp provides the time vector with custom spacing.
   *
   * @tparam FreqType
   *   Frequency data type
   * @tparam SpaceOp
   *   Operator type of spacer
   *
   * @param t
   *   Vector representing values in time
   * @param f0
   *   Instantenous frequency at time 0
   * @param t1
   *   Time for f1
   * @param f1
   *   Frequency (Hz) at time t1
   * @param method
   *   Chirp method (CHIRP_METHOD_LINEAR)
   *
   * @returns The chirp operator
   */
  template <typename SpaceOp, typename FreqType>
    inline auto chirp(SpaceOp t, FreqType f0, typename SpaceOp::value_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
    {
      MATX_ASSERT_STR(method == ChirpMethod::CHIRP_METHOD_LINEAR, matxInvalidType, "Only linear chirps are supported")

        return detail::Chirp<SpaceOp, FreqType>(t, f0, t1, f1, method);       
    }

  /**
   * Creates a complex chirp signal (swept-frequency cosine)
   * 
   * SpaceOp provides the time vector with custom spacing.
   *
   * @tparam FreqType
   *   Frequency data type
   * @tparam SpaceOp
   *   Operator type of spacer
   *
   * @param t
   *   Vector representing values in time
   * @param f0
   *   Instantenous frequency at time 0
   * @param t1
   *   Time for f1
   * @param f1
   *   Frequency (Hz) at time t1
   * @param method
   *   Chirp method (CHIRP_METHOD_LINEAR)
   *
   * @returns The chirp operator
   */
  template <typename SpaceOp, typename FreqType>
    inline auto cchirp(SpaceOp t, FreqType f0, typename SpaceOp::value_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
    {
      MATX_ASSERT_STR(method == ChirpMethod::CHIRP_METHOD_LINEAR, matxInvalidType, "Only linear chirps are supported")

        return detail::ComplexChirp<SpaceOp, FreqType>(t, f0, t1, f1, method);       
    }

  /**
   * Creates a chirp signal (swept-frequency cosine)
   * 
   * Creates a linearly-spaced sequence from 0 to "last" with "num" elements in between. Each step is
   * of size 1/num.
   *
   * @tparam FreqType
   *   Frequency data type
   * @tparam TimeType
   *   Type of time vector
   * @tparam Method
   *   Chirp method (CHIRP_METHOD_LINEAR)
   *
   * @param num
   *   Number of time samples
   * @param last
   *   Last time sample value
   * @param f0
   *   Instantenous frequency at time 0
   * @param t1
   *   Time for f1
   * @param f1
   *   Frequency (Hz) at time t1
   * @param method
   *   Method to use to generate the chirp
   *
   * @returns The chirp operator
   */
  template <typename TimeType, typename FreqType>
    inline auto chirp(index_t num, TimeType last, FreqType f0, TimeType t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
    {
      auto space = linspace((TimeType)0, last, num);
      return chirp(space, f0, t1, f1, method);
    }
    
    
  /**
   *  Creates a complex chirp signal (swept-frequency cosine)
   * 
   * Creates a linearly-spaced sequence from 0 to "last" with "num" elements in between. Each step is
   * of size 1/num.
   *
   * @tparam FreqType
   *   Frequency data type
   * @tparam TimeType
   *   Type of time vector
   * @tparam Method
   *   Chirp method (CHIRP_METHOD_LINEAR)
   *
   * @param num
   *   Number of time samples
   * @param last
   *   Last time sample value
   * @param f0
   *   Instantenous frequency at time 0
   * @param t1
   *   Time for f1
   * @param f1
   *   Frequency (Hz) at time t1
   * @param method
   *   Method to use to generate the chirp
   *
   * @returns The chirp operator
   */
  template <typename TimeType, typename FreqType>
    inline auto cchirp(index_t num, TimeType last, FreqType f0, TimeType t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
    {
      auto space = linspace((TimeType)0, last, num);
      return cchirp(space, f0, t1, f1, method);
    }



} // end namespace matx
