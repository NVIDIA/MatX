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
#include "matx_matmul.h"
#include "matx_tensor.h"

#include <cstdio>
#include <numeric>

namespace matx {
namespace detail {
/**
 * Parameters needed to compute a covariance matrix. For the most part, these
 * are very similar to that of a standard cov call
 */
struct CovParams_t {
  void *A;
  MatXDataType_t dtype;
  cudaStream_t stream;
};

template <typename TensorTypeC, typename TensorTypeA> class matxCovHandle_t {
public:
  static constexpr int RANK = TensorTypeA::Rank();
  using T1 = typename TensorTypeA::scalar_type;
  /**
   * Construct a handle for computing a covariance matrix
   *
   * Creates a covariance matrix handle for computing the covariance of a matrix
   * A, where the rows of A are the observations of data and columns are sets of
   * data. For complex matrices the output will match Python with the E[XX']
   * convention.
   *
   * @tparam T1
   *    Data type of A and C matrices
   * @tparam RANK
   *    Rank of A/B/C matrices
   *
   * @param a
   *   A input matrix view
   * @param c
   *   C output covariance matrix view
   */
  matxCovHandle_t(TensorTypeC &c, const TensorTypeA &a)
  {
    static_assert(RANK >= 2);
    MATX_ASSERT(c.Size(RANK - 1) == c.Size(RANK - 2), matxInvalidSize);
    MATX_ASSERT(a.Size(RANK - 1) == c.Size(RANK - 1), matxInvalidSize);

    // Ensure batch dimensions are equal
    for (int i = 2; i < RANK - 2; i++) {
      MATX_ASSERT(a.Size(i) == c.Size(i), matxInvalidSize);
    }

    // This must come before the things below to properly set class parameters
    params_ = GetCovParams(c, a);

    onesM = make_tensor_p<T1>({a.Size(RANK - 2), a.Size(RANK - 2)});
    means = make_tensor_p<T1>(a.Shape());
    devs = make_tensor_p<T1>(a.Shape());

    // Transposed view of deviations
    tensorShape_t<RANK> tmp;
    for (int i = 0; i < RANK; i++) {
      tmp.SetSize(i, a.Size(RANK - i - 1));
    }

    devsT = make_tensor_p<T1>(tmp);

    // Populate our ones matrix
    exec(set(*onesM,
             ones({a.Size(RANK - 2), a.Size(RANK - 2)})),
         0);
  }

  static CovParams_t GetCovParams([[maybe_unused]] TensorTypeC &c, const TensorTypeA &a)
  {
    CovParams_t params;
    params.dtype = TypeToInt<T1>();
    params.A = a.Data();

    return params;
  }

  /**
   * Cov handle destructor
   *
   * Destroys any helper data used for provider type and any workspace memory
   * created
   *
   */
  ~matxCovHandle_t()
  {
    delete onesM;
    delete means;
    delete devs;
  }

/**
 * Compute a covariance matrix
 *
 * Computes a covariance matrix using columns as data sets and rows as
 * observations. The resultant matrix C is a symmetric positive semi-definite
 * matrix where the diagonals are the variances, and off-diagonals are
 * covariances.
 *
 * Passing a tensor of rank > 2 acts as batching dimensions
 *
 *
 * @tparam T1
 *   Type of beta
 * @param c
 *   Output covariance matrix
 * @param a
 *   Input tensor A
 * @param stream
 *   CUDA stream
 *
 */
  inline void Exec(TensorTypeC &c, const TensorTypeA &a,
                   cudaStream_t stream)
  {
    // Calculate a matrix of means
    matmul(*means, *onesM, a, stream,
                 1.0f / static_cast<float>(a.Size(RANK - 2)));

    // Subtract the means from the observations to get the deviations
    exec(set(*devs, a - *means), stream);

    if constexpr (is_complex_v<T1>) {
      // This step is not really necessary since BLAS can do it for us, but
      // until we have a way to detect a Hermitian property on a matrix, we need
      // to have this in a temporary variable. Note that we use the Python
      // convention of E[XX'] instead of MATLAB's E[X'X]. Both are "correct",
      // but we need to match python output
      exec(set(*devsT, hermitianT(*devs)), stream);
    }
    else {
      transpose(*devsT, *devs, stream);
    }

    // Multiply by itself and scale by N-1 for the final covariance
    matmul(c, *devsT, *devs, stream,
                1.0f / static_cast<float>(a.Size(RANK - 2) - 1));
  }    

  private:
    // Member variables
    matx::tensor_t<T1, 2> *onesM;
    matx::tensor_t<T1, RANK> *means;
    matx::tensor_t<T1, RANK> *devs;
    matx::tensor_t<T1, RANK> *devsT;  
    CovParams_t params_;
};

/**
 * Crude hash on cpv to get a reasonably good delta for collisions. This doesn't
 * need to be perfect, but fast enough to not slow down lookups, and different
 * enough so the common covariance parameters change
 */
struct CovParamsKeyHash {
  std::size_t operator()(const CovParams_t &k) const noexcept
  {
    return std::hash<uint64_t>()((uint64_t)k.A) +
           std::hash<uint64_t>()((size_t)k.stream);
  }
};

/**
 * Test covariance parameters for equality. Unlike the hash, all parameters must
 * match.
 */
struct CovParamsKeyEq {
  bool operator()(const CovParams_t &l, const CovParams_t &t) const noexcept
  {
    return l.A == t.A && l.stream == t.stream && l.dtype == t.dtype;
  }
};

// Static caches of covariance matrices
static matxCache_t<CovParams_t, CovParamsKeyHash, CovParamsKeyEq> cov_cache;

} // end namespace detail
/**
 * Compute a covariance matrix without a plan
 *
 * Creates a new cov plan in the cache if none exists, and uses that to execute
 * the covariance calculation. This function is preferred over creating a plan
 * directly for both efficiency and simpler code. Since it only uses the
 * signature of the covariance to decide if a plan is cached, it may be able to
 * reused plans for different A matrices
 *
 * @tparam T1
 *    Data type of A matrix
 * @tparam RANK
 *    Rank of A matrix
 *
 * @param c
 *   Covariance matrix output view
 * @param a
 *   Covariance matrix input view
 * @param stream
 *   CUDA stream
 */
template <typename TensorTypeC, typename TensorTypeA>
void cov(TensorTypeC &c, const TensorTypeA &a,
         cudaStream_t stream = 0)
{
  // Get parameters required by these tensors
  auto params = detail::matxCovHandle_t<TensorTypeC, TensorTypeA>::GetCovParams(c, a);
  params.stream = stream;

  // Get cache or new cov plan if it doesn't exist
  auto ret = detail::cov_cache.Lookup(params);
  if (ret == std::nullopt) {
    auto tmp = new detail::matxCovHandle_t<TensorTypeC, TensorTypeA>{c, a};
    detail::cov_cache.Insert(params, static_cast<void *>(tmp));

    tmp->Exec(c, a, stream);
  }
  else {
    auto cov_type = static_cast<detail::matxCovHandle_t<TensorTypeC, TensorTypeA> *>(ret.value());
    cov_type->Exec(c, a, stream);
  }
}

} // end namespace matx
