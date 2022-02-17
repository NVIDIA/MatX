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

#include "matx_error.h"
#include "matx_shape.h"
#include "matx_tensor_ops.h"
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
 * @return __inline__ 
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
 * @return __inline__ 
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
 * @return __inline__ 
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
 * @return __inline__ 
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
template <typename T> class randomGenerator_t {
private:
  index_t total_threads_;
  curandStatePhilox4_32_10_t *states_;

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
   *   Dimensions of the view in the form of an tensorShape_t
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
  inline auto GetTensorView(const tensorShape_t<RANK> shape,
                            Distribution_t dist, T alpha = 1, T beta = 0)
  {
    return randomTensorView_t<T, RANK>(shape, states_, dist, alpha, beta);
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
    tensorShape_t<RANK> shape((const index_t *)sizes);
    return randomTensorView_t<T, RANK>(shape, states_, dist, alpha, beta);
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
  tensorShape_t<RANK> shape_;
  std::array<index_t, RANK> strides_;
  curandStatePhilox4_32_10_t *states_;
  Distribution_t dist_;
  T alpha_, beta_;

public:
  using type = T; ///< Type trait to get type
  using scalar_type = T; ///< Type trait to get type
  // dummy type to signal this is a matxop
  using matxop = bool; ///< Type trait to indicate this is an operator

  /**
   * @brief Construct a new randomTensorView t object
   * 
   * @param shape Shape of view
   * @param states States of RNG
   * @param dist RNG distribution
   * @param alpha Alpha value
   * @param beta Beta value
   */
  randomTensorView_t(const tensorShape_t<RANK> shape,
                         curandStatePhilox4_32_10_t *states,
                         Distribution_t dist, T alpha, T beta)
      : alpha_(alpha), beta_(beta), shape_(shape)
  {
    dist_ = dist;
    states_ = states;

    if constexpr (RANK >= 1) {
      strides_[RANK-1] = 1;
    }

    #pragma unroll
    for (int i = RANK - 2; i >= 0; i--) {
      strides_[i] = Stride(i+1) * Size(i+1);
    }
  }

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
    return shape_.Size(dim);
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
};

/**
 * Populate a tensor with random values
 *
 * @param t
 *   Output tensor view
 * @param dist
 *   Type of random distribution
 * @param seed
 *   Random seed
 * @param stream
 *   Stream to execute
 * Note: leave this function out for now since it may be confusing to users with
 * all the allocations it does
 */
// template<class T, int RANK>
// inline void rand(tensor_t<T,RANK> t, Distribution_t dist = UNIFORM,
// uint64_t seed = 0, cudaStream_t stream = 0) {
//   randomGenerator_t<T> r(t.TotalSize(), seed);
//   auto tmp = r.template GetTensorView<RANK> (t.Shape(), dist);
//   exec(set(t, tmp), stream);
// }

} // end namespace matx
