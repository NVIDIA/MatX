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

#include <array>
#include <cmath>

#include "matx_tensor.h"

namespace matx {

template <typename T, int RANK> class ConstVal {
private:
  tensorShape_t<RANK> s_;
  T v_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  ConstVal(tensorShape_t<RANK> s, T val) : s_(s), v_(val){};

  inline __device__ T operator()() { return v_; };
  inline __device__ T operator()(index_t) { return v_; };
  inline __device__ T operator()(index_t, index_t) { return v_; };
  inline __device__ T operator()(index_t, index_t, index_t) { return v_; };
  inline __device__ T operator()(index_t, index_t, index_t, index_t)
  {
    return v_;
  };

  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return s_.Size(dim);
  }
  static inline constexpr __host__ __device__ int32_t Rank() { return RANK; }
};

/**
 * Return zero for all elements
 *
 * Zeros is used as an operator that always returns a 0 type for all
 * elements. It can be used in place of memset to zero a block of memory.
 *
 * @tparam T
 *   Data type
 *
 * @param s
 *   Shape of tensor
 */
template <typename T = int, int RANK>
inline auto zeros(const tensorShape_t<RANK> &s)
{
  return ConstVal<T, RANK>(s, T(0));
}

template <typename T = int, int RANK>
inline auto zeros(const index_t (&s)[RANK])
{
  return zeros(tensorShape_t<RANK>{(const index_t *)s});
}

/**
 * Return one for all elements
 *
 * Ones is used as an operator that always returns a 1 type for all
 * elements. It can be used in place of memset to set all values to 1.
 *
 * @tparam T
 *   Data type
 *
 * @param s
 *   Shape of tensor
 */
template <typename T = int, int RANK>
inline auto ones(const tensorShape_t<RANK> &s)
{
  return ConstVal<T, RANK>(s, T(1));
}

template <typename T = int, int RANK> inline auto ones(const index_t (&s)[RANK])
{
  return ones(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T, int RANK> class Diag {
private:
  tensorShape_t<RANK> s_;
  T val_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  Diag(tensorShape_t<RANK> s, T val) : s_(s), val_(val){};

  inline __device__ T operator()() { return T(val_); };
  inline __device__ T operator()(index_t i)
  {
    if (i == 0)
      return val_;
    else
      return T(0.0f);
  };
  inline __device__ T operator()(index_t i, index_t j)
  {
    if (i == j)
      return T(val_);
    else
      return T(0.0f);
  };
  inline __device__ T operator()(index_t i, index_t j, index_t k)
  {
    if (i == j && i == k)
      return T(val_);
    else
      return T(0.0f);
  };
  inline __device__ T operator()(index_t i, index_t j, index_t k, index_t l)
  {
    if (i == j && k == l && i == k)
      return T(val_);
    else
      return T(0.0f);
  };

  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return s_.Size(dim);
  }
  static inline constexpr __host__ __device__ int32_t Rank() { return RANK; }
};

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
template <typename T = int, int RANK>
inline auto diag(const tensorShape_t<RANK> &s, T val)
{
  return Diag<T, RANK>(s, val);
}

template <typename T = int, int RANK>
inline auto diag(const index_t (&s)[RANK], T val)
{
  return diag(tensorShape_t<RANK>{(const index_t *)s}, val);
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
inline auto eye(const tensorShape_t<RANK> &s)
{
  return Diag<T, RANK>(s, T(1));
}

template <typename T = int, int RANK> inline auto eye(const index_t (&s)[RANK])
{
  return eye(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename Generator1D, int Dim, int RANK> class matxGenerator1D_t {
public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = typename Generator1D::scalar_type;

  matxGenerator1D_t(tensorShape_t<RANK> s, Generator1D f) : f_(f), s_(s) {}
  inline __device__ auto operator()(int i) { return f_(i); };
  inline __device__ auto operator()(int i, int j)
  {
    if constexpr (Dim == 0) {
      return f_(i);
    }
    else {
      return f_(j);
    }
    // BUG WAR
    return scalar_type(0);
  };
  inline __device__ auto operator()(int i, int j, int k)
  {
    if constexpr (Dim == 0) {
      return f_(i);
    }
    else if constexpr (Dim == 1) {
      return f_(j);
    }
    else {
      return f_(k);
    }
    // BUG WAR
    return scalar_type(0);
  };
  inline __device__ auto operator()(int i, int j, int k, int l)
  {
    if constexpr (Dim == 0) {
      return f_(i);
    }
    else if constexpr (Dim == 1) {
      return f_(j);
    }
    else if constexpr (Dim == 2) {
      return f_(k);
    }
    else {
      return f_(l);
    }
    // BUG WAR
    return scalar_type(0);
  };

  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return s_.Size(dim);
  }
  static inline constexpr __host__ __device__ int32_t Rank() { return RANK; }

private:
  Generator1D f_;
  tensorShape_t<RANK> s_;
};

template <typename T> class Hamming {
private:
  index_t size_;

public:
  using scalar_type = T;

  inline __host__ __device__ Hamming(index_t size) : size_(size){};

  inline __host__ __device__ T operator()(index_t i)
  {
    return T(.54) - T(.46) * cuda::std::cos(T(2 * M_PI) * T(i) / T(size_ - 1));
  }
};

/// @name HammingWindows
/// @{
/**
 * Creates a Hamming window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 *
 * @tparam RANK
 *   The RANK of the shape, can be deduced from shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Hamming window across the selected dimension.
 */
template <typename T = float, int RANK>
inline auto hamming_x(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 1);
  Hamming<T> h(s.Size(RANK - 1));
  return matxGenerator1D_t<Hamming<T>, RANK - 1, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hamming_x(const index_t (&s)[RANK])
{
  return hamming_x(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hamming_y(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 2);
  Hamming<T> h(s.Size(RANK - 2));
  return matxGenerator1D_t<Hamming<T>, RANK - 2, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hamming_y(const index_t (&s)[RANK])
{
  return hamming_y(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hamming_z(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 3);
  Hamming<T> h(s.Size(RANK - 3));
  return matxGenerator1D_t<Hamming<T>, RANK - 3, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hamming_z(const index_t (&s)[RANK])
{
  return hamming_z(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hamming_w(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 4);
  Hamming<T> h(s.Size(RANK - 4));
  return matxGenerator1D_t<Hamming<T>, RANK - 4, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hamming_w(const index_t (&s)[RANK])
{
  return hamming_w(tensorShape_t<RANK>{(const index_t *)s});
}

/// @}
template <typename T> class Hanning {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __host__ __device__ Hanning(index_t size) : size_(size){};

  inline __host__ __device__ T operator()(index_t i)
  {
    return T(0.5) * (1 - cuda::std::cos(T(2 * M_PI) * T(i) / T(size_ - 1)));
  }
};

/// @name HanningWindows
/// @{
/**
 * Creates a Hanning window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 *
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Hanning window across the selected dimension.
 */
template <typename T = float, int RANK>
inline auto hanning_x(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 1);
  Hanning<T> h(s.Size(RANK - 1));
  return matxGenerator1D_t<Hanning<T>, RANK - 1, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hanning_x(const index_t (&s)[RANK])
{
  return hanning_x(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hanning_y(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 2);
  Hanning<T> h(s.Size(RANK - 2));
  return matxGenerator1D_t<Hanning<T>, RANK - 2, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hanning_y(const index_t (&s)[RANK])
{
  return hanning_y(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hanning_z(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 3);
  Hanning<T> h(s.Size(RANK - 3));
  return matxGenerator1D_t<Hanning<T>, RANK - 3, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hanning_z(const index_t (&s)[RANK])
{
  return hanning_z(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto hanning_w(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 4);
  Hanning<T> h(s.Size(RANK - 4));
  return matxGenerator1D_t<Hanning<T>, RANK - 4, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto hanning_w(const index_t (&s)[RANK])
{
  return hanning_w(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T> class Blackman {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __host__ __device__ Blackman(index_t size) : size_(size){};

  inline __host__ __device__ T operator()(index_t i)
  {
    return T(0.42) +
           ((T)0.5 *
            cuda::std::cos(T(M_PI) * (1 - size_ + 2 * T(i)) / T(size_ - 1))) +
           ((T)0.08 * cuda::std::cos(T(2 * M_PI) * (1 - size_ + 2 * T(i)) /
                                     T(size_ - 1)));
  }
};

/**
 * Creates a Blackman window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 *
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Blackman window across the selected dimension.
 */
template <typename T = float, int RANK>
inline auto blackman_x(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 1);
  Blackman<T> h(s.Size(RANK - 1));
  return matxGenerator1D_t<Blackman<T>, RANK - 1, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto blackman_x(const index_t (&s)[RANK])
{
  return blackman_x(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto blackman_y(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 2);
  Blackman<T> h(s.Size(RANK - 2));
  return matxGenerator1D_t<Blackman<T>, RANK - 2, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto blackman_y(const index_t (&s)[RANK])
{
  return blackman_y(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto blackman_z(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 3);
  Blackman<T> h(s.Size(RANK - 3));
  return matxGenerator1D_t<Blackman<T>, RANK - 3, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto blackman_z(const index_t (&s)[RANK])
{
  return blackman_z(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto blackman_w(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 4);
  Blackman<T> h(s.Size(RANK - 4));
  return matxGenerator1D_t<Blackman<T>, RANK - 4, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto blackman_w(const index_t (&s)[RANK])
{
  return blackman_w(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T> class Bartlett {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __host__ __device__ Bartlett(index_t size) : size_(size){};

  inline __host__ __device__ T operator()(index_t i)
  {
    return (T(2) / (T(size_) - 1)) *
           (((T(size_) - 1) / T(2)) -
            cuda::std::abs(T(i) - ((T(size_) - 1) / T(2))));
  }
};

/**
 * Creates a Bartlett window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 *
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Bartlett window across the selected dimension.
 */
template <typename T = float, int RANK>
inline auto bartlett_x(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 1);
  Bartlett<T> h(s.Size(RANK - 1));
  return matxGenerator1D_t<Bartlett<T>, RANK - 1, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto bartlett_x(const index_t (&s)[RANK])
{
  return bartlett_x(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto bartlett_y(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 2);
  Bartlett<T> h(s.Size(RANK - 2));
  return matxGenerator1D_t<Bartlett<T>, RANK - 2, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto bartlett_y(const index_t (&s)[RANK])
{
  return bartlett_y(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto bartlett_z(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 3);
  Bartlett<T> h(s.Size(RANK - 3));
  return matxGenerator1D_t<Bartlett<T>, RANK - 3, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto bartlett_z(const index_t (&s)[RANK])
{
  return bartlett_z(tensorShape_t<RANK>{(const index_t *)s});
}

template <typename T = float, int RANK>
inline auto bartlett_w(const tensorShape_t<RANK> &s)
{
  static_assert(RANK >= 4);
  Bartlett<T> h(s.Size(RANK - 4));
  return matxGenerator1D_t<Bartlett<T>, RANK - 4, RANK>(s, h);
}
template <typename T = float, int RANK>
inline auto bartlett_w(const index_t (&s)[RANK])
{
  return bartlett_w(tensorShape_t<RANK>{(const index_t *)s});
}

template <class T> class Range {
private:
  T first_;
  T step_;

public:
  using scalar_type = T;

  Range() = default;

  Range(T first, T step) : first_(first), step_(step) {}

  __device__ inline T operator()(index_t idx)
  {
    if constexpr (is_matx_half_v<T>) {
      return first_ + T(static_cast<T>((float)idx) * step_);
    }
    else {
      return first_ + T(static_cast<T>(idx) * step_);
    }

    if constexpr (!is_matx_half_v<T>) {
      return first_ + T(static_cast<T>(idx) * step_);
    }
    else {
      return first_ + T(static_cast<T>((float)idx) * step_);
    }
  }
};

/**
 * Create a range of values along the x dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_x(const tensorShape_t<RANK> &s, T first, T step)
{
  static_assert(RANK >= 1);
  Range<T> r(first, step);
  return matxGenerator1D_t<Range<T>, RANK - 1, RANK>(s, r);
}

/**
 * Create a range of values along the x dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_x(const index_t (&s)[RANK], T first, T step)
{
  return range_x(tensorShape_t<RANK>{(const index_t *)s}, first, step);
}

/**
 * Create a range of values along the y dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_y(const tensorShape_t<RANK> &s, T first, T step)
{
  static_assert(RANK >= 2);
  Range<T> r(first, step);
  return matxGenerator1D_t<Range<T>, RANK - 2, RANK>(s, r);
}

/**
 * Create a range of values along the y dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_y(const index_t (&s)[RANK], T first, T step)
{
  return range_y(tensorShape_t<RANK>{(const index_t *)s}, first, step);
}

/**
 * Create a range of values along the z dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_z(const tensorShape_t<RANK> &s, T first, T step)
{
  static_assert(RANK >= 3);
  Range<T> r(first, step);
  return matxGenerator1D_t<Range<T>, RANK - 3, RANK>(s, r);
}

/**
 * Create a range of values along the z dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_z(const index_t (&s)[RANK], T first, T step)
{
  return range_z(tensorShape_t<RANK>{(const index_t *)s}, first, step);
}

/**
 * Create a range of values along the w dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_w(const tensorShape_t<RANK> &s, T first, T step)
{
  static_assert(RANK >= 4);
  Range<T> r(first, step);
  return matxGenerator1D_t<Range<T>, RANK - 4, RANK>(s, r);
}

/**
 * Create a range of values along the w dimension
 *
 * Creates a range of values of type T with a start and step size.
 * Value is determined by the index in operator()
 *
 * @param s
 *   Tensor shape
 * @param first
 *   Starting value
 * @param step
 *   Step size
 *
 */
template <typename T = float, int RANK>
inline auto range_w(const index_t (&s)[RANK], T first, T step)
{
  return range_w(tensorShape_t<RANK>{(const index_t *)s}, first, step);
}

template <class T> class Linspace {
private:
  Range<T> range_;

public:
  using scalar_type = T;

  inline Linspace(T first, T last, index_t count)
  {
#ifdef __CUDA_ARCH__
    range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
#else
    // Host has no support for most half precision operators/intrinsics
    if constexpr (is_matx_half_v<T>) {
      range_ = Range<T>{static_cast<float>(first),
                        (static_cast<float>(last) - static_cast<float>(first)) /
                            static_cast<float>(count - 1)};
    }
    else {
      range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
    }
#endif
  }

  __device__ inline T operator()(index_t idx) { return range_(idx); }
};

/// @name Linspace
/// @{
/**
 * Create a linearly-spaced range of values
 *
 * Creates a set of values using a start and end that are linearly-
 * spaced apart over the set of values. Distance is determined
 * by the shape and selected dimension.
 *
 */
template <typename T = float, int RANK>
inline auto linspace_x(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 1);
  index_t count = s.Size(RANK - 1);
  Linspace<T> l(first, last, count);
  return matxGenerator1D_t<Linspace<T>, RANK - 1, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto linspace_x(const index_t (&s)[RANK], T first, T last)
{
  return linspace_x(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto linspace_y(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 2);
  index_t count = s.Size(RANK - 2);
  Linspace<T> l(first, last, count);
  return matxGenerator1D_t<Linspace<T>, RANK - 2, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto linspace_y(const index_t (&s)[RANK], T first, T last)
{
  return linspace_y(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto linspace_z(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 3);
  index_t count = s.Size(RANK - 3);
  Linspace<T> l(first, last, count);
  return matxGenerator1D_t<Linspace<T>, RANK - 3, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto linspace_z(const index_t (&s)[RANK], T first, T last)
{
  return linspace_z(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto linspace_w(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 4);
  index_t count = s.Size(RANK - 4);
  Linspace<T> l(first, last, count);
  return matxGenerator1D_t<Linspace<T>, RANK - 4, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto linspace_w(const index_t (&s)[RANK], T first, T last)
{
  return linspace_w(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <class T> class Logspace {
private:
  Range<T> range_;

public:
  using scalar_type = T;

  inline Logspace(T first, T last, index_t count)
  {
#ifdef __CUDA_ARCH__
    if constexpr (is_matx_half_v<T>) {
      range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1.0f)};
    }
    else {
      range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
    }
#else
    // Host has no support for most half precision operators/intrinsics
    if constexpr (is_matx_half_v<T>) {
      range_ = Range<T>{static_cast<float>(first),
                        (static_cast<float>(last) - static_cast<float>(first)) /
                            static_cast<float>(count - 1)};
    }
    else {
      range_ = Range<T>{first, (last - first) / static_cast<T>(count - 1)};
    }
#endif
  }

  __device__ inline T operator()(index_t idx)
  {
    if constexpr (is_matx_half_v<T>) {
      return static_cast<T>(
          cuda::std::pow(10, static_cast<float>(range_(idx))));
    }
    else {
      return cuda::std::pow(10, range_(idx));
    }

    // WAR for compiler bug.
    if constexpr (!is_matx_half_v<T>) {
      return cuda::std::pow(10, range_(idx));
    }
    else {
      return static_cast<T>(
          cuda::std::pow(10, static_cast<float>(range_(idx))));
    }
  }
};

/// @name Linspace
/// @{
/**
 * Create a log-10-spaced range of values
 *
 * Creates a set of values using a start and end that are log-10-
 * spaced apart over the set of values. Distance is determined
 * by the shape and selected dimension.
 *
 */
template <typename T = float, int RANK>
inline auto logspace_x(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 1);
  index_t count = s.Size(RANK - 1);
  Logspace<T> l(first, last, count);
  return matxGenerator1D_t<Logspace<T>, RANK - 1, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto logspace_x(const index_t (&s)[RANK], T first, T last)
{
  return logspace_x(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto logspace_y(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 2);
  index_t count = s.Size(RANK - 2);
  Logspace<T> l(first, last, count);
  return matxGenerator1D_t<Logspace<T>, RANK - 2, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto logspace_y(const index_t (&s)[RANK], T first, T last)
{
  return logspace_y(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto logspace_z(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 3);
  index_t count = s.Size(RANK - 3);
  Logspace<T> l(first, last, count);
  return matxGenerator1D_t<Logspace<T>, RANK - 3, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto logspace_z(const index_t (&s)[RANK], T first, T last)
{
  return logspace_z(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T = float, int RANK>
inline auto logspace_w(const tensorShape_t<RANK> &s, T first, T last)
{
  static_assert(RANK >= 4);
  index_t count = s.Size(RANK - 4);
  Logspace<T> l(first, last, count);
  return matxGenerator1D_t<Logspace<T>, RANK - 4, RANK>(s, l);
}
template <typename T = float, int RANK>
inline auto logspace_w(const index_t (&s)[RANK], T first, T last)
{
  return logspace_w(tensorShape_t<RANK>{(const index_t *)s}, first, last);
}

template <typename T> class Meshgrid_X {
private:
  std::array<T, 3> x_;
  std::array<T, 3> y_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  Meshgrid_X(std::array<T, 3> x, std::array<T, 3> y) : x_(x), y_(y) {}

  inline __device__ T operator()(index_t i, index_t j)
  {
    return x_[0] + j * (x_[1] - x_[0]) / (x_[2] - 1);
  }

  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return (dim == 0) ? y_[2] : x_[2];
  }
  static inline constexpr __host__ __device__ int32_t Rank() { return 2; }
};

template <typename T> class Meshgrid_Y {
private:
  std::array<T, 3> x_;
  std::array<T, 3> y_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  Meshgrid_Y(std::array<T, 3> x, std::array<T, 3> y) : x_(x), y_(y) {}

  inline __device__ T operator()(index_t i, index_t j)
  {
    return y_[0] + i * (y_[1] - y_[0]) / (y_[2] - 1);
  };

  inline __host__ __device__ index_t Size(uint32_t dim) const
  {
    return (dim == 0) ? y_[2] : x_[2];
  }
  static inline constexpr __host__ __device__ int32_t Rank() { return 2; }
};
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
  return Meshgrid_X<T>(x, y);
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
  return Meshgrid_Y<T>(x, y);
}
/// @}
} // end namespace matx
