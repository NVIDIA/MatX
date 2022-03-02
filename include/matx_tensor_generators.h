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

namespace detail {
template <typename T, typename ShapeType> class ConstVal {
  static constexpr int RANK = std::tuple_size<typename remove_cvref<ShapeType>::type>::value;

private:
  ShapeType s_;
  T v_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  ConstVal(ShapeType &&s, T val) : s_(std::forward<ShapeType>(s)), v_(val){};

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is...) const { 
    return v_; };

  constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
  {
    return *(s_.begin() + dim);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }
};
}

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
template <typename T = int, typename ShapeType,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto zeros(ShapeType &&s)
{
  return detail::ConstVal<T, ShapeType>(std::forward<ShapeType>(s), T(0));
}

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
inline auto zeros(const index_t (&s)[RANK])
{
  return zeros<T>(detail::to_array(s));
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
template <typename T = int, typename ShapeType,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto ones(ShapeType &&s)
{
  return detail::ConstVal<T, ShapeType>(std::forward<ShapeType>(s), T(1));
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
inline auto ones(const index_t (&s)[RANK])
{
  return ones<T>(detail::to_array(s));
}

namespace detail {
template <typename T, typename ShapeType> class Diag {
  static constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;

private:
  ShapeType s_;
  T val_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

  Diag(ShapeType &&s, T val) : s_(std::forward<ShapeType>(s)), val_(val)
  {
    static_assert(Rank() > 1, "Diagonal generator must be used with an operator of rank 1 or higher");
  };

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
    if (((pp_get<0>(indices...) == indices) && ...)) {
      return T(val_);
    }
    else {
      return T(0.0f);
    }
  }

  constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
  {
    return *(s_.begin() + dim);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }
};
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
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
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
 * @tparam T
 *   Data type
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
template <typename T = int, int RANK> inline auto eye(const index_t (&s)[RANK])
{
  return eye(detail::to_array(s));
}

namespace detail {
template <typename Generator1D, int Dim, typename ShapeType> class matxGenerator1D_t {
public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = typename Generator1D::scalar_type;
  static constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;

  template <typename S>
  matxGenerator1D_t(S &&s, Generator1D f) : f_(f), s_(std::forward<S>(s)) {}

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const {
    return f_(pp_get<Dim>(indices...));
  }

  constexpr inline __MATX_HOST__ __MATX_DEVICE__ auto Size(int dim) const
  {
    return *(s_.begin() + dim);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return RANK; }

private:
  Generator1D f_;
  ShapeType s_;
};
}


namespace detail {
template <typename T> class Alternating {
private:
  index_t size_;

public:
  using scalar_type = T;

  inline __MATX_HOST__ __MATX_DEVICE__ Alternating(index_t size) : size_(size) {};
  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const 
  {
    return (-2 * (i & 1)) + 1;
  }
};
}


/**
 * Creates an alternating +1/-1 sequence
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape, can be deduced from shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for alternating sequence
 */
template <int Dim, typename ShapeType, typename T = float, 
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto alternate(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Alternating<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::Alternating<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates an alternating +1/-1 sequence
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape, can be deduced from shape
 *
 * @param s
 *   C array representing shape of the tensor
 *
 */
template <int Dim, int RANK, typename T = float>
inline auto alternate(const index_t (&s)[RANK])
{
  return alternate<Dim>(detail::to_array(s));
}

namespace detail {
template <typename T> class Hamming {
private:
  index_t size_;

public:
  using scalar_type = T;

  inline __MATX_HOST__ __MATX_DEVICE__ Hamming(index_t size) : size_(size){};

  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const 
  {
    return T(.54) - T(.46) * cuda::std::cos(T(2 * M_PI) * T(i) / T(size_ - 1));
  }
};
}


/**
 * Creates a Hamming window operator of shape s with the
 * window applies along the specified dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape, can be deduced from shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Hamming window across the selected dimension.
 */
template <int Dim, typename ShapeType, typename T = float, 
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto hamming(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Hamming<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::Hamming<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates a Hamming window operator of C-array shape s with the
 * window applies along the specified dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape, can be deduced from shape
 *
 * @param s
 *   C array representing shape of the tensor
 *
 * Returns values for a Hamming window across the selected dimension.
 */
template <int Dim, int RANK, typename T = float>
inline auto hamming(const index_t (&s)[RANK])
{
  return hamming<Dim>(detail::to_array(s));
}


namespace detail {
template <typename T> class Hanning {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __MATX_HOST__ __MATX_DEVICE__ Hanning(index_t size) : size_(size){};

  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const
  {
    return T(0.5) * (1 - cuda::std::cos(T(2 * M_PI) * T(i) / T(size_ - 1)));
  }
};
}

/**
 * Creates a Hanning window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Hanning window across the selected dimension.
 */
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto hanning(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Hanning<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::Hanning<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates a Hanning window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The C-array shape of the tensor
 *
 * Returns values for a Hanning window across the selected dimension.
 */
template <int Dim, int RANK, typename T = float>
inline auto hanning(const index_t (&s)[RANK])
{
  return hanning<Dim>(detail::to_array(s));
}


namespace detail {
template <typename T> class FlatTop {
private:
  index_t size_;

  static constexpr T a0 = 0.21557895;
  static constexpr T a1 = 0.41663158;
  static constexpr T a2 = 0.277263158;
  static constexpr T a3 = 0.083578947;
  static constexpr T a4 = 0.006947368;  

public:

  using scalar_type = T;
  inline __MATX_HOST__ __MATX_DEVICE__ FlatTop(index_t size) : size_(size){};

  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const
  {
    return  a0  
            - a1 * cuda::std::cos(2*M_PI*i / (size_ - 1))
            + a2 * cuda::std::cos(4*M_PI*i / (size_ - 1))
            - a3 * cuda::std::cos(6*M_PI*i / (size_ - 1))
            + a4 * cuda::std::cos(8*M_PI*i / (size_ - 1));
  }
};
}

/**
 * Creates a Flattop window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Flattop window across the selected dimension.
 */
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto flattop(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::FlatTop<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::FlatTop<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates a Flattop window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Flattop window across the selected dimension.
 */
template <int Dim, int RANK, typename T = float>
inline auto flattop(const index_t (&s)[RANK])
{
  return flattop<Dim>(detail::to_array(s));
}


namespace detail {
template <typename T> class Blackman {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __MATX_HOST__ __MATX_DEVICE__ Blackman(index_t size) : size_(size){};

  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const
  {
    return T(0.42) +
           ((T)0.5 *
            cuda::std::cos(T(M_PI) * (1 - size_ + 2 * T(i)) / T(size_ - 1))) +
           ((T)0.08 * cuda::std::cos(T(2 * M_PI) * (1 - size_ + 2 * T(i)) /
                                     T(size_ - 1)));
  }
};
}

/**
 * Creates a Blackman window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Blackman window across the selected dimension.
 */
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto blackman(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Blackman<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::Blackman<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates a Blackman window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Blackman window across the selected dimension.
 */
template <int Dim, int RANK, typename T = float>
inline auto blackman(const index_t (&s)[RANK])
{
  return blackman<Dim>(detail::to_array(s));
}

namespace detail {
template <typename T> class Bartlett {
private:
  index_t size_;

public:
  using scalar_type = T;
  inline __MATX_HOST__ __MATX_DEVICE__ Bartlett(index_t size) : size_(size){};

  inline __MATX_HOST__ __MATX_DEVICE__ T operator()(index_t i) const 
  {
    return (T(2) / (T(size_) - 1)) *
           (((T(size_) - 1) / T(2)) -
            cuda::std::abs(T(i) - ((T(size_) - 1) / T(2))));
  }
};
}

/**
 * Creates a Bartlett window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The shape of the tensor
 *
 * Returns values for a Bartlett window across the selected dimension.
 */
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto bartlett(ShapeType &&s)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Bartlett<T> h( *(s.begin() + Dim));
  return detail::matxGenerator1D_t<detail::Bartlett<T>, Dim, ShapeType>(std::forward<ShapeType>(s), h);
}

/**
 * Creates a Bartlett window operator of shape s with the
 * window applies along the x, y, z, or w dimension
 *
 * @tparam T
 *   Data type
 * @tparam Dim
 *   Dimension to create window over
 * @tparam RANK
 *   The RANK of the shape
 *
 * @param s
 *   The C array shape of the tensor
 *
 * Returns values for a Bartlett window across the selected dimension.
 */
template <int Dim, int RANK, typename T = float>
inline auto bartlett(const index_t (&s)[RANK])
{
  return bartlett<Dim>(detail::to_array(s));
}



namespace detail {
template <class T> class Range {
private:
  T first_;
  T step_;

public:
  using scalar_type = T;

  Range() = default;

  Range(T first, T step) : first_(first), step_(step) {}

  __MATX_DEVICE__ inline T operator()(index_t idx) const
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
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto range(ShapeType &&s, T first, T step)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  detail::Range<T> r(first, step);
  return detail::matxGenerator1D_t<detail::Range<T>, Dim, ShapeType>(std::forward<ShapeType>(s), r);
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
template <int Dim, int RANK, typename T = float>
inline auto range(const index_t (&s)[RANK], T first, T step)
{
  return range<Dim>(detail::to_array(s), first, step);
}


namespace detail {
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

  __MATX_DEVICE__ inline T operator()(index_t idx) const { return range_(idx); }
};
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
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto linspace(ShapeType &&s, T first, T last)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  auto count =  *(s.begin() + Dim);
  detail::Linspace<T> l(first, last, count);
  return detail::matxGenerator1D_t<detail::Linspace<T>, Dim, ShapeType>(std::forward<ShapeType>(s), l);
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
template <int Dim, int RANK, typename T = float>
inline auto linspace(const index_t (&s)[RANK], T first, T last)
{
  return linspace<Dim>(detail::to_array(s), first, last);
}

namespace detail {
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

  __MATX_DEVICE__ inline T operator()(index_t idx) const
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
}


/**
 * @brief Create a log10-spaced range of values
 *
 * Creates a set of values using a start and end that are log10-
 * spaced apart over the set of values. Distance is determined
 * by the shape and selected dimension.
 * 
 * @tparam T Operator type
 * @tparam Dim Dimension to operate over
 * @tparam ShapeType Shape type
 * @param s Shape object
 * @param first First value
 * @param last Last value
 * @return Operator with log10-spaced values 
 */
template <int Dim, typename ShapeType, typename T = float,
  std::enable_if_t<!std::is_array_v<typename remove_cvref<ShapeType>::type>, bool> = true>
inline auto logspace(ShapeType &&s, T first, T last)
{
  constexpr int RANK = std::tuple_size<std::decay_t<ShapeType>>::value;
  static_assert(RANK > Dim);
  auto count = *(s.begin() + Dim);
  detail::Logspace<T> l(first, last, count);
  return detail::matxGenerator1D_t<detail::Logspace<T>, Dim, ShapeType>(std::forward<ShapeType>(s), l);
}

/**
 * @brief Create a log10-spaced range of values
 *
 * Creates a set of values using a start and end that are log10-
 * spaced apart over the set of values. Distance is determined
 * by the shape and selected dimension.
 * 
 * @tparam T Operator type
 * @tparam Dim Dimension to operate over
 * @tparam ShapeType Shape type
 * @param s Shape object
 * @param first First value
 * @param last Last value
 * @return Operator with log10-spaced values 
 */
template <int Dim, int RANK, typename T = float>
inline auto logspace(const index_t (&s)[RANK], T first, T last)
{
  return logspace<Dim>(detail::to_array(s), first, last);
}


enum class ChirpMethod {
  CHIRP_METHOD_LINEAR
};

enum class ChirpType {
  CHIRP_TYPE_REAL,
  CHIRP_TYPE_COMPLEX
};

namespace detail {
template <typename SpaceOp, typename FreqType> 
class Chirp {
  using space_type = typename SpaceOp::scalar_type;

  
private:
  SpaceOp sop_;
  FreqType f0_;
  FreqType f1_;
  space_type t1_;
  ChirpMethod method_;

public:
  using scalar_type = FreqType;
  using matxop = bool;
  inline __MATX_HOST__ __MATX_DEVICE__ Chirp(SpaceOp sop, FreqType f0, space_type t1, FreqType f1, ChirpMethod method) : 
      sop_(sop),
      f0_(f0),
      t1_(t1),
      f1_(f1),
      method_(method)
        {}

  inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
  {
    if (method_ == ChirpMethod::CHIRP_METHOD_LINEAR) {
      return cuda::std::cos(2.0f * M_PI * (f0_ * sop_(i) + 0.5f * ((f1_ - f0_) / t1_) * sop_(i) * sop_(i)));
    }
  }

  constexpr inline __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
  {
    return sop_.Size(0);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }  
};

template <typename SpaceOp, typename FreqType> 
class ComplexChirp {
  using space_type = typename SpaceOp::scalar_type;

  
private:
  SpaceOp sop_;
  FreqType f0_;
  FreqType f1_;
  space_type t1_;
  ChirpMethod method_;

public:
  using scalar_type = cuda::std::complex<FreqType>;
  using matxop = bool;
  inline __MATX_HOST__ __MATX_DEVICE__ ComplexChirp(SpaceOp sop, FreqType f0, space_type t1, FreqType f1, ChirpMethod method) : 
      sop_(sop),
      f0_(f0),
      t1_(t1),
      f1_(f1),
      method_(method)
        {}

  inline __MATX_HOST__ __MATX_DEVICE__ auto operator()(index_t i) const
  {
    if (method_ == ChirpMethod::CHIRP_METHOD_LINEAR) {
      FreqType real = cuda::std::cos(2.0f * M_PI * (f0_ * sop_(i) + 0.5f * ((f1_ - f0_) / t1_) * sop_(i) * sop_(i)));
      FreqType imag = -cuda::std::cos(2.0f * M_PI * (f0_ * sop_(i) + 0.5f * ((f1_ - f0_) / t1_) * sop_(i) * sop_(i) + 90.0/360.0));
      return cuda::std::complex<FreqType>{real, imag};
    }
  }

  constexpr inline __MATX_HOST__ __MATX_DEVICE__ index_t Size([[maybe_unused]] int dim) const
  {
    return sop_.Size(0);
  }
  static inline constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() { return 1; }  
};
}

namespace signal {
/**
 * Creates a chirp signal (swept-frequency cosine)
 * 
 * SpaceOp provides the time vector with custom spacing.
 *
 * @tparam FreqType
 *   Frequency data type
 * @tparam SpaceOp
 *   Operator type of spacer
 * @tparam Method
 *   Chirp method (CHIRP_METHOD_LINEAR)
 *
 * @param t
 *   Vector representing values in time
 * @param f0
 *   Instantenous frequency at time 0
 * @param t1
 *   Time for f1
 * @param f1
 *   Frequency (Hz) at time t1
 *
 * @returns The chirp operator
 */
template <typename SpaceOp, typename FreqType>
inline auto chirp(SpaceOp t, FreqType f0, typename SpaceOp::scalar_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
{
  MATX_ASSERT_STR(method == ChirpMethod::CHIRP_METHOD_LINEAR, matxInvalidType, "Only linear chirps are supported")

  return detail::Chirp<SpaceOp, FreqType>(t, f0, t1, f1, method);       
}

template <typename SpaceOp, typename FreqType>
inline auto cchirp(SpaceOp t, FreqType f0, typename SpaceOp::scalar_type t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
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
  std::array<index_t, 1> shape = {num};
  auto space = linspace<0>(std::move(shape), (TimeType)0, last);
  return chirp(space, f0, t1, f1, method);
}

template <typename TimeType, typename FreqType>
inline auto cchirp(index_t num, TimeType last, FreqType f0, TimeType t1, FreqType f1, ChirpMethod method = ChirpMethod::CHIRP_METHOD_LINEAR)
{
  std::array<index_t, 1> shape = {num};
  auto space = linspace<0>(std::move(shape), (TimeType)0, last);
  return cchirp(space, f0, t1, f1, method);
}
}



namespace detail {
template <typename T> class Meshgrid_X {
private:
  std::array<T, 3> x_;
  std::array<T, 3> y_;

public:
  // dummy type to signal this is a matxop
  using matxop = bool;
  using scalar_type = T;

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
