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

#include "matx/core/error.h"
#include <cuda/std/complex>
#include <curand_kernel.h>
#include <type_traits>

namespace matx {

/**
 * Random number distribution
 */
enum Distribution_t { UNIFORM, NORMAL };

#ifdef __CUDACC__
template <typename Gen>
__global__ void curand_setup_kernel(Gen *states, uint64_t seed, index_t size)
{
  index_t idx = (index_t)blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size)
    curand_init(seed, idx, 0, &states[idx]);
};
#endif


/**
 * @brief Get a random number
 *
 * @tparam T inner type of data
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param min min of the range to generate
 * @param max max of the range to generate
 */
template <typename T, typename Gen>
__MATX_INLINE__ __MATX_DEVICE__ void get_randomi(T &val, Gen *state, double min, double max)
{
  
  double normFloat = curand_uniform(state);

  // Scale to the provided min and max range
  val = static_cast<T>(normFloat * (max - min)+ min);

};

/**
 * @brief Get a random number
 *
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param dist Distribution
 */
template <typename Gen>
__MATX_INLINE__ __MATX_DEVICE__ void get_random(float &val, Gen *state,
                                      Distribution_t dist)
{
  if (dist == UNIFORM) {
    val = curand_uniform(state);
  }
  else {
    val = curand_normal(state);
  }
};

/**
 * @brief Get a random number
 *
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param dist Distribution
 */
template <typename Gen>
__MATX_INLINE__ __MATX_DEVICE__ void get_random(double &val, Gen *state,
                                      Distribution_t dist)
{
  if (dist == UNIFORM) {
    val = curand_uniform_double(state);
  }
  else {
    val = curand_normal_double(state);
  }
};

/**
 * @brief Get a random number
 *
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param dist Distribution
 */
template <typename Gen>
__MATX_INLINE__ __MATX_DEVICE__ void get_random(cuda::std::complex<float> &val,
                                      Gen *state, Distribution_t dist)
{
  if (dist == UNIFORM) {
    val.real(curand_uniform(state));
    val.imag(curand_uniform(state));
  }
  else {
    float2 r = curand_normal2(state);
    val.real(r.x);
    val.imag(r.y);
  }
};

/**
 * @brief Get a random number
 *
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param dist Distribution
 */
template <typename Gen>
__MATX_INLINE__ __MATX_DEVICE__ void get_random(cuda::std::complex<double> &val,
                                      Gen *state, Distribution_t dist)
{
  if (dist == UNIFORM) {
    val.real(curand_uniform_double(state));
    val.imag(curand_uniform_double(state));
  }
  else {
    double2 r = curand_normal2_double(state);
    val.real(r.x);
    val.imag(r.y);
  }
};

/**
 * Generates random numbers
 *
 * @tparam
 *   Type of random number
 *
 * Generate random numbers based on a size and seed. Uses the Philox 4x32
 * generator with 10 rounds.
 */
template <typename T> class [[deprecated("Use random() operator instead of randomGenerator_t")]] randomGenerator_t {
private:
  index_t total_threads_;
  bool init_;
  curandStatePhilox4_32_10_t *states_;
  uint64_t seed_;

public:
  randomGenerator_t() = delete;

  /**
   * Constructs a random number generator
   *
   * This call will allocate memory sufficiently large enough to store state of
   * the RNG
   *
   * @param total_threads
   *   Number of random values to generate
   * @param seed
   *   Seed for the RNG
   */
  __MATX_INLINE__ randomGenerator_t(index_t total_threads, uint64_t seed)
      : total_threads_(total_threads)
  {
#ifdef __CUDACC__
    matxAlloc((void **)&states_,
              total_threads_ * sizeof(curandStatePhilox4_32_10_t),
              MATX_DEVICE_MEMORY);

    int threads = 128;
    int blocks = static_cast<int>((total_threads_ + threads - 1) / threads);
    curand_setup_kernel<<<blocks, threads>>>(states_, seed, total_threads);
#endif
  };

  /**
   * Destroy the RNG and free all memory
   */
  __MATX_INLINE__ ~randomGenerator_t() { matxFree(states_); }
};

namespace detail {

  template< typename T >
  struct randIntParams{
    T min_;
    T max_;
  };
  
  template< typename T >
  struct randFloatParams{
    Distribution_t dist_;
    T alpha_;
    T beta_;
  };  
  
  template <typename T, typename ShapeType>
  class RandomOp : public BaseOp<RandomOp<T, ShapeType>> {
    private:
      using inner_t = typename inner_op_type_t<T>::type;
      static constexpr int RANK = cuda::std::tuple_size<ShapeType>{};
      cuda::std::array<index_t, RANK> shape_;
      cuda::std::array<index_t, RANK> strides_;
      index_t total_size_;
      mutable curandStatePhilox4_32_10_t *states_;
      uint64_t seed_;     
      mutable bool init_ = false;
      mutable bool device_;
      
      union{
        randFloatParams<inner_t> fParams_;
        randIntParams<inner_t>   iParams_;
      };
      
      // Used by host operators only
      mutable curandGenerator_t gen_;
      //T val;


    public:
      using value_type = T;
      using matxop = bool;     

      __MATX_INLINE__ std::string str() const { return "random"; }

      // Shapeless constructor to be allocated at run invocation
      RandomOp() = delete;

      // base constructor, should never be called directly
      __MATX_INLINE__ RandomOp(ShapeType &&s, uint64_t seed) : seed_(seed)
      {
        total_size_ = std::accumulate(s.begin(), s.end(), 1, std::multiplies<index_t>());

        if constexpr (RANK >= 1) {
          strides_[RANK-1] = 1;
        }

        #pragma unroll
        for (int i = 0; i < RANK; ++i)
        {
          shape_[i] = s[i];
        }


        #pragma unroll
        for (int i = RANK - 2; i >= 0; i--) {
          strides_[i] = strides_[i+1] * s[i+1];
        }
      }

      // Constructor for randFloatParams
      __MATX_INLINE__ RandomOp(ShapeType &&s, uint64_t seed, randFloatParams<inner_t> params) :
          RandomOp(std::forward<ShapeType>(s), seed)
      {
          fParams_ = params;
      }

      // Constructor for randIntParams
      __MATX_INLINE__ RandomOp(ShapeType &&s, uint64_t seed, randIntParams<inner_t> params) :
          RandomOp(std::forward<ShapeType>(s), seed)
      {
          iParams_ = params;
      }

      template <typename ST, typename Executor>
      __MATX_INLINE__ void InnerPreRun([[maybe_unused]] ST &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {          
      }
           
      template <typename ST, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ST &&shape, Executor &&ex) const
      {
        InnerPreRun(std::forward<ST>(shape), std::forward<Executor>(ex));          
#ifdef __CUDACC__
        if constexpr (is_cuda_executor_v<Executor>) {
          if (!init_) {
            auto stream = ex.getStream();
            matxAlloc((void **)&states_,
                      total_size_ * sizeof(curandStatePhilox4_32_10_t),
                      MATX_ASYNC_DEVICE_MEMORY, stream);

            int threads = 128;
            int blocks = static_cast<int>((total_size_ + threads - 1) / threads);
            curand_setup_kernel<<<blocks, threads, 0, stream>>>(states_, seed_, total_size_);
            init_   = true;
            device_ = true;
          }
        }
        else if constexpr (is_host_executor_v<Executor>) {
          if (!init_) {
            [[maybe_unused]] curandStatus_t ret;

            ret = curandCreateGeneratorHost(&gen_, CURAND_RNG_PSEUDO_MT19937);
            MATX_ASSERT_STR_EXP(ret, CURAND_STATUS_SUCCESS, matxCudaError, "Failed to create random number generator");

            ret = curandSetPseudoRandomGeneratorSeed(gen_, seed_);
            MATX_ASSERT_STR_EXP(ret, CURAND_STATUS_SUCCESS, matxCudaError, "Error setting random seed");

            // In the future we may allocate a buffer, but for now we generate a single number at a time
            // matxAlloc((void **)&val, total_size_ * sizeof(T), MATX_HOST_MEMORY, stream);
            init_   = true;
            device_ = false;
          }
        }
#endif
      }

      template <typename ST, typename Executor>
      __MATX_INLINE__ void PostRun([[maybe_unused]] ST &&shape, [[maybe_unused]] Executor &&ex) const noexcept
      {
        if constexpr (is_cuda_executor_v<Executor>) {
          matxFree(states_);
        }
        else if constexpr (is_host_executor_v<Executor>) {
          curandDestroyGenerator(gen_);
          //matxFree(val);
        }
      }

      template <int I = 0, typename ...Is, std::enable_if_t<I == sizeof...(Is), bool> = true>
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const cuda::std::tuple<Is...>) const {
        return 0;
      }

      template <int I = 0, typename ...Is, std::enable_if_t<I < sizeof...(Is), bool> = true>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const cuda::std::tuple<Is...> tup) const {
        return GetValC<I+1, Is...>(tup) + cuda::std::get<I>(tup)*strides_[I];
      }

      /**
       * Retrieve a value from a random view
       *
       * @tparam Is Index type
       * @param indices Index values
       */
      template <VecWidth InWidth, VecWidth OutWidth, typename... Is>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T operator()([[maybe_unused]] Is... indices) const
      {
        T val;
#ifdef __CUDA_ARCH__
        if constexpr (
                     std::is_same_v<T, float>  ||
                     std::is_same_v<T, double> || 
                     std::is_same_v<T, cuda::std::complex<float>> ||
                     std::is_same_v<T, cuda::std::complex<double>>
                     ) 
        {
          if constexpr (sizeof...(indices) == 0) {
            get_random(val, &states_[0], fParams_.dist_);
          }
          else {
            get_random(val, &states_[GetValC<0, Is...>(cuda::std::make_tuple(indices...))], fParams_.dist_);
          }

          val = fParams_.alpha_ * val + fParams_.beta_;
        }
        else if constexpr(
                  std::is_same_v<T, uint32_t> || 
                  std::is_same_v<T,  int32_t> ||
                  std::is_same_v<T, uint64_t> ||
                  std::is_same_v<T,  int64_t>   
        )
        {
          if constexpr (sizeof...(indices) == 0) {
            get_randomi(val, &states_[0], iParams_.min_, iParams_.max_);
          }
          else {
            get_randomi(val, &states_[GetValC<0, Is...>(cuda::std::make_tuple(indices...))], iParams_.min_, iParams_.max_);
          }          
        }

#else
        if constexpr (
                     std::is_same_v<T, float>  ||
                     std::is_same_v<T, double> || 
                     std::is_same_v<T, cuda::std::complex<float>> ||
                     std::is_same_v<T, cuda::std::complex<double>>
                     ) 
        {        
          
          if (fParams_.dist_ == UNIFORM) {
            if constexpr (std::is_same_v<T, float>) {
              curandGenerateUniform(gen_, &val, 1);
            }
            else if constexpr (std::is_same_v<T, double>) {
              curandGenerateUniformDouble(gen_, &val, 1);
            }
            else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
              float *tmp = reinterpret_cast<float *>(&val);
              curandGenerateUniform(gen_, &tmp[0], 1);
              curandGenerateUniform(gen_, &tmp[1], 1);
            }
            else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
              double *tmp = reinterpret_cast<double *>(&val);
              curandGenerateUniformDouble(gen_, &tmp[0], 1);
              curandGenerateUniformDouble(gen_, &tmp[1], 1);
            }

            val = fParams_.alpha_ * val + fParams_.beta_;
          }
          else if (fParams_.dist_ == NORMAL) {
            if constexpr (std::is_same_v<T, float>) {
              curandGenerateNormal(gen_, &val, 1, fParams_.beta_, fParams_.alpha_);
            }
            else if constexpr (std::is_same_v<T, double>) {
              curandGenerateNormalDouble(gen_, &val, 1, fParams_.beta_, fParams_.alpha_);
            }
            else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
              float *tmp = reinterpret_cast<float *>(&val);
              curandGenerateNormal(gen_, &tmp[0], 1, fParams_.beta_, fParams_.alpha_);
              curandGenerateNormal(gen_, &tmp[1], 1, fParams_.beta_, fParams_.alpha_);
            }
            else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
              double *tmp = reinterpret_cast<double *>(&val);
              curandGenerateNormalDouble(gen_, &tmp[0], 1, fParams_.beta_, fParams_.alpha_);
              curandGenerateNormalDouble(gen_, &tmp[1], 1, fParams_.beta_, fParams_.alpha_);
            }
          }
          else {
            val = 0;
          }
        }
        else if constexpr(
                         std::is_same_v<T, uint32_t> || 
                         std::is_same_v<T,  int32_t> ||
                         std::is_same_v<T, uint64_t> ||
                         std::is_same_v<T,  int64_t>   
                         )
        {
          float fScale;
          curandGenerateUniform(gen_, &fScale, 1);
                    
          // Scale to the provided min and max range
          double fMax = static_cast<double>(iParams_.max_);
          double fMin = static_cast<double>(iParams_.min_);
          val = static_cast<T>(fScale * (fMax - fMin) + fMin);                                
        }
#endif

        return val;
      }


      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
      {
        return shape_[dim];
      }

      static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }
    };
  }

  /**
   * @brief Return a random number with a specified shape.
   * 
   * Supported Types: float, double, complex<float>, complex<double>
   *
   * @tparam ShapeType Shape type
   * @tparam T Type of output
   * @tparam LowerType Either T or the inner type of T if T is complex* 
   * @param s Shape of operator
   * @param dist Distribution (either NORMAL or UNIFORM)
   * @param seed Random number seed
   * @param alpha Value to multiply by each number
   * @param beta Value to add to each number
   * @return Random number operator
   */
  template <typename T, typename ShapeType, typename LowerType = typename inner_op_type_t<T>::type,
           std::enable_if_t<!std::is_array_v<remove_cvref_t<ShapeType>>, bool> = true>
  __MATX_INLINE__ auto random(ShapeType &&s, Distribution_t dist, uint64_t seed = 0, LowerType alpha = 1, LowerType beta = 0)
  {
    static_assert(
                  std::is_same_v<T, float> || 
                  std::is_same_v<T, double> ||
                  std::is_same_v<T, cuda::std::complex<float>> ||
                  std::is_same_v<T, cuda::std::complex<double>>, 
                  "random only supports floating point or complex floating point data types"

                 );   

    using shape_strip_t = remove_cvref_t<ShapeType>;
    matx::detail::randFloatParams<LowerType> params{dist, alpha, beta};

    return detail::RandomOp<T, shape_strip_t>(std::forward<shape_strip_t>(s), seed, params);
  }

  /**
   * @brief Return a random number with a specified shape.
   * 
   * Supported Types: float, double, complex<float>, complex<double>
   * 
   * @tparam RANK Rank of operator
   * @tparam T Type of output
   * @tparam LowerType Either T or the inner type of T if T is complex
   * @param s Array of dimensions
   * @param dist Distribution (either NORMAL or UNIFORM)
   * @param seed Random number seed
   * @param alpha Value to multiply by each number
   * @param beta Value to add to each number
   * @return Random number operator
   */
  template <typename T, int RANK, typename LowerType = typename inner_op_type_t<T>::type>
  __MATX_INLINE__ auto random(const index_t (&s)[RANK], Distribution_t dist, uint64_t seed = 0, LowerType alpha = 1, LowerType beta = 0)
  {
    auto sarray = detail::to_array(s);
    return random<T, decltype(sarray)>(std::move(sarray), dist, seed, alpha, beta);
  }


  /**
   * @brief Return a random number with a specified shape.
   * 
   *  Supported types: uint32_t, int32_t, uint64_t, int64_t
   * 
   * @tparam ShapeType Shape type
   * @tparam T Type of output
   * @tparam LowerType Either T or the inner type of T if T is complex* 
   * @param s Shape of operator
   * @param seed Random number seed
   * @param min min of generation range
   * @param max max of generation range
   * @return Random number operator
   */
  template <typename T, typename ShapeType, typename LowerType = typename inner_op_type_t<T>::type,
           std::enable_if_t<!std::is_array_v<remove_cvref_t<ShapeType>>, bool> = true>
  __MATX_INLINE__ auto randomi(ShapeType &&s, uint64_t seed = 0, LowerType min = 0, LowerType max = 100)
  {
    static_assert(
                  std::is_same_v<T, uint32_t> || 
                  std::is_same_v<T,  int32_t> ||
                  std::is_same_v<T, uint64_t> ||
                  std::is_same_v<T,  int64_t> ,
                  "randomi only supports signed and unsigned integral types"
                 );      
                  
    using shape_strip_t = remove_cvref_t<ShapeType>;
    matx::detail::randIntParams<T> params{min, max};
    
    return detail::RandomOp<T, shape_strip_t>(std::forward<shape_strip_t>(s), seed, params);
  }
  
  /**
   * @brief Return a random number with a specified shape.
   * 
   *  Supported types: uint32_t, int32_t, uint64_t, int64_t
   *
   * @tparam RANK Rank of operator
   * @tparam T Type of output
   * @tparam LowerType Either T or the inner type of T if T is complex
   * @param s Array of dimensions
   * @param seed Random number seed
   * @param min min of generation range 
   * @param max max of generation range
   * @return Random number operator
   */
  template <typename T, int RANK, typename LowerType = typename inner_op_type_t<T>::type>
  __MATX_INLINE__ auto randomi(const index_t (&s)[RANK], uint64_t seed = 0, LowerType min = 0, LowerType max = 100)
  {
    auto sarray = detail::to_array(s);
    return randomi<T, decltype(sarray)>(std::move(sarray), seed, min, max);
  }

} // end namespace matx
