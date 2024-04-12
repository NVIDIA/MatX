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
 * @tparam Gen Generator type
 * @param val Value to store in
 * @param state Generator state
 * @param dist Distribution
 */
template <typename Gen>
__inline__ __MATX_DEVICE__ void get_random(float &val, Gen *state,
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
__inline__ __MATX_DEVICE__ void get_random(double &val, Gen *state,
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
__inline__ __MATX_DEVICE__ void get_random(cuda::std::complex<float> &val,
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
__inline__ __MATX_DEVICE__ void get_random(cuda::std::complex<double> &val,
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

template <typename T, int RANK> class randomTensorView_t;

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
  inline randomGenerator_t(index_t total_threads, uint64_t seed)
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
   * Get a tensor view of the random numbers
   *
   * @param shape
   *   Dimensions of the view in the form of a Shape
   * @param dist
   *   Distribution to use
   * @param alpha
   *   Alpha value
   * @param beta
   *   Beta value
   * @returns
   *   A randomTensorView_t with given parameters
   */
  template <std::size_t RANK>
  inline auto GetTensorView(const std::array<index_t, RANK> &shape, Distribution_t dist,
                            T alpha = 1, T beta = 0)
{
  return randomTensorView_t<T, static_cast<int>(RANK)>(shape, states_, dist, alpha, beta);
}

  /**
   * Get a tensor view of the random numbers
   *
   * @param sizes
   *   Dimensions of the view in the form of an initializer list
   * @param dist
   *   Distribution to use
   * @param alpha
   *   Alpha value
   * @param beta
   *   Beta value
   * @returns
   *   A randomTensorView_t with given parameters
   */
  template <int RANK>
  inline auto GetTensorView(const index_t (&sizes)[RANK], Distribution_t dist,
                            T alpha = 1, T beta = 0)
  {
    return GetTensorView(detail::to_array(sizes), dist, alpha, beta);
  }

  /**
   * Destroy the RNG and free all memory
   */
  inline ~randomGenerator_t() { matxFree(states_); }
};

/**
 * Random number generator view
 *
 * @tparam
 *   Type of random number
 * @tparam
 *   Rank of view
 *
 * Provides a view into a previously-allocated randomGenerator_t
 */
template <typename T, int RANK> class randomTensorView_t {
private:
  std::array<index_t, RANK> shape_;
  std::array<index_t, RANK> strides_;
  curandStatePhilox4_32_10_t *states_;
  Distribution_t dist_;
  T alpha_, beta_;

public:
  using type = T; ///< Type trait to get type
  using scalar_type = T; ///< Type trait to get type
  // dummy type to signal this is a matxop
  using matxop = bool; ///< Type trait to indicate this is an operator

  __MATX_INLINE__ std::string str() const { return "rand"; }

  /**
   * @brief Construct a new randomTensorView t object
   *
   * @param shape Shape of view
   * @param states States of RNG
   * @param dist RNG distribution
   * @param alpha Alpha value
   * @param beta Beta value
   */

  template < typename ShapeType, std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
  randomTensorView_t(    ShapeType &&shape,
                         curandStatePhilox4_32_10_t *states,
                         Distribution_t dist, T alpha, T beta)
      : alpha_(alpha), beta_(beta)
  {
    dist_ = dist;
    states_ = states;

    if constexpr (RANK >= 1) {
      strides_[RANK-1] = 1;
    }

    #pragma unroll
    for (int i = 0; i < RANK; ++i)
    {
      shape_[i] = shape[i];
    }


    #pragma unroll
    for (int i = RANK - 2; i >= 0; i--) {
      strides_[i] = Stride(i+1) * Size(i+1);
    }
  }

  __MATX_INLINE__ auto Shape() const noexcept { return shape_; }
  /**
   * Get the random number at an index
   *
   * @tparam I Unused
   * @tparam Is Index types
   * @return Value at index
   */
  template <int I = 0, typename ...Is, std::enable_if_t<I == sizeof...(Is), bool> = true>
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const std::tuple<Is...>) const {
    return 0;
  }

  template <int I = 0, typename ...Is, std::enable_if_t<I < sizeof...(Is), bool> = true>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const std::tuple<Is...> tup) const {
    return GetValC<I+1, Is...>(tup) + std::get<I>(tup)*Stride(I);
  }

  /**
   * Retrieve a value from a random view
   *
   * @tparam Is Index type
   * @param indices Index values
   */
  template <typename... Is>
  inline __MATX_DEVICE__ T operator()(Is... indices) const
  {
    T val;
    if constexpr (RANK == 0) {
      get_random(val, &states_[0], dist_);
    }
    else {
      get_random(val, &states_[GetValC<0, Is...>(std::make_tuple(indices...))], dist_);
    }

    return alpha_ * val + beta_;
  };

  /**
   * Get rank of random view
   *
   * @return Rank of view
   */
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

  /**
   *  Get size of a dimension
   *
   * @param dim Dimension to retrieve
   * @return Size of dimension
   */
  constexpr index_t inline __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const
  {
    return shape_[dim];
  }

  /**
   * Get stride of a dimension
   *
   * @param dim Dimension to retrieve
   * @return Stride of dimension
   */
  index_t inline __MATX_HOST__ __MATX_DEVICE__ Stride(int dim) const
  {
    return strides_[dim];
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept
  {
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) noexcept  
  {
  }  
};

  namespace detail {
    template <typename T, typename ShapeType>
    class RandomOp : public BaseOp<RandomOp<T, ShapeType>> {
      private:
        using inner_t = typename inner_op_type_t<T>::type;
        static constexpr int RANK = std::tuple_size<ShapeType>{};
        Distribution_t dist_;
        std::array<index_t, RANK> shape_;
        std::array<index_t, RANK> strides_;
        index_t total_size_;
        curandStatePhilox4_32_10_t *states_;
        uint64_t seed_;
        inner_t alpha_;
        inner_t beta_;
        bool init_ = false;
        bool device_;

        // Used by host operators only
        curandGenerator_t gen_;
        //T val;


      public:
        using scalar_type = T;
        using matxop = bool;

        __MATX_INLINE__ std::string str() const { return "random"; }

        // Shapeless constructor to be allocated at run invocation
        RandomOp() = delete;

        inline RandomOp(ShapeType &&s, Distribution_t dist, uint64_t seed, inner_t alpha, inner_t beta) :
          dist_(dist), seed_(seed), alpha_(alpha), beta_(beta)
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


      template <typename ST, typename Executor>
      __MATX_INLINE__ void PreRun([[maybe_unused]] ST &&shape, Executor &&ex)
      {
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
      __MATX_INLINE__ void PostRun([[maybe_unused]] ST &&shape, [[maybe_unused]] Executor &&ex) noexcept
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
      constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const std::tuple<Is...>) const {
        return 0;
      }

      template <int I = 0, typename ...Is, std::enable_if_t<I < sizeof...(Is), bool> = true>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t GetValC(const std::tuple<Is...> tup) const {
        return GetValC<I+1, Is...>(tup) + std::get<I>(tup)*strides_[I];
      }

      /**
       * Retrieve a value from a random view
       *
       * @tparam Is Index type
       * @param indices Index values
       */
      template <typename... Is>
      __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T operator()([[maybe_unused]] Is... indices) const
      {
        T val;
#ifdef __CUDA_ARCH__
        if constexpr (sizeof...(indices) == 0) {
          get_random(val, &states_[0], dist_);
        }
        else {
          get_random(val, &states_[GetValC<0, Is...>(std::make_tuple(indices...))], dist_);
        }

        val = alpha_ * val + beta_;
#else
        if (dist_ == UNIFORM) {
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

          val = alpha_ * val + beta_;
        }
        else if (dist_ == NORMAL) {
          if constexpr (std::is_same_v<T, float>) {
            curandGenerateNormal(gen_, &val, 1, beta_, alpha_);
          }
          else if constexpr (std::is_same_v<T, double>) {
            curandGenerateNormalDouble(gen_, &val, 1, beta_, alpha_);
          }
          else if constexpr (std::is_same_v<T, cuda::std::complex<float>>) {
            float *tmp = reinterpret_cast<float *>(&val);
            curandGenerateNormal(gen_, &tmp[0], 1, beta_, alpha_);
            curandGenerateNormal(gen_, &tmp[1], 1, beta_, alpha_);
          }
          else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
            double *tmp = reinterpret_cast<double *>(&val);
            curandGenerateNormalDouble(gen_, &tmp[0], 1, beta_, alpha_);
            curandGenerateNormalDouble(gen_, &tmp[1], 1, beta_, alpha_);
          }
        }
#endif

        return val;
      }


      constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
      {
        return shape_[dim];
      }

      static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }
    };
  }

  /**
   * @brief Return a random number with a specified shape
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
  inline auto random(ShapeType &&s, Distribution_t dist, uint64_t seed = 0, LowerType alpha = 1, LowerType beta = 0)
  {
    using shape_strip_t = remove_cvref_t<ShapeType>;
    return detail::RandomOp<T, shape_strip_t>(std::forward<shape_strip_t>(s), dist, seed, alpha, beta);
  }

  /**
   * @brief Return a random number with a specified shape
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
  inline auto random(const index_t (&s)[RANK], Distribution_t dist, uint64_t seed = 0, LowerType alpha = 1, LowerType beta = 0)
  {
    auto sarray = detail::to_array(s);
    return random<T, decltype(sarray)>(std::move(sarray), dist, seed, alpha, beta);
  }

} // end namespace matx
