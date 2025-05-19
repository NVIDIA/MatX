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

// This file contains the options for the operators both public-facing and private-facing. Only
// trivial types should be put in this file since it's used by the RTC compiler as well.
#pragma once

namespace matx {

typedef enum {
  AMBGFUN_CUT_TYPE_2D,
  AMBGFUN_CUT_TYPE_DELAY,
  AMBGFUN_CUT_TYPE_DOPPLER,
} AMBGFunCutType_t;

enum class FFTNorm {
  BACKWARD, /// fft is unscaled, ifft is 1/N
  FORWARD, /// fft is scaled 1/N, ifft is not scaled
  ORTHO /// fft is scaled 1/sqrt(N), ifft is scaled 1/sqrt(N)
};

typedef enum {
  MATX_C_MODE_FULL, // Default. Keep all elements of ramp up/down
  MATX_C_MODE_SAME, // Only keep elements where entire filter was present
  MATX_C_MODE_VALID
} matxConvCorrMode_t;

typedef enum {
  MATX_C_METHOD_DIRECT,
  MATX_C_METHOD_FFT
} matxConvCorrMethod_t;

  enum class PercentileMethod {
    LINEAR,
    LOWER,
    HIGHER,
    HAZEN,
    WEIBULL,
    MEDIAN_UNBIASED,
    NORMAL_UNBIASED,
    MIDPOINT,
    NEAREST
  }; 

/**
 * @brief Direction for sorting
 *
 */
typedef enum { SORT_DIR_ASC, SORT_DIR_DESC } SortDirection_t;

/* Solver parameter enums */

/**
 * @brief Algorithm to use for matrix inverse
 *
 */
typedef enum {
  MAT_INVERSE_ALGO_LU,
} MatInverseAlgo_t;

/**
 * @enum SolverFillMode
 *   Indicates which part (lower or upper) of the dense matrix was filled
 *   and should be used by the function.
 */
enum class SolverFillMode {
  UPPER,  /**< Use the upper part of the matrix */
  LOWER   /**< Use the lower part of the matrix */
};

/**
 * @enum EigenMode
 *   Specifies whether or not eigenvectors should be computed.
 */
enum class EigenMode {
  NO_VECTOR,  /**< Only eigenvalues are computed */
  VECTOR      /**< Both eigenvalues and eigenvectors are computed */
};

/**
 * @enum SVDMode
 *   Modes for computing columns of *U* and rows of *VT* in Singular Value Decomposition (SVD).
 *   Corresponds to the LAPACK/cuSolver parameters jobu and jobvt. The same option is used
 *   for both jobu and jobvt in MatX.
 */
enum class SVDMode {
  ALL,     /**< Compute all columns of *U* and all rows of *VT* (Equivalent to jobu = jobvt = 'A') */
  REDUCED, /**< Compute only the first `min(m,n` columns of *U* and rows of *VT* (Equivalent to jobu = jobvt = 'S') */
  NONE     /**< Compute no columns of *U* or rows of *VT* (Equivalent to jobu = jobvt = 'N') */
};

/**
 * @enum SVDHostAlgo
 *   Controls the LAPACK driver used for SVD on host.
 */
enum class SVDHostAlgo {
  QR,  /**< QR-based method (corresponds to `gesvd`) */
  DC   /**< Divide and Conquer method (corresponds to `gesdd`) */
};


namespace detail {
  static constexpr int MAX_FFT_RANK = 2;

  enum class FFTType {
    C2C,
    R2C,
    C2R,
    Z2Z,
    D2Z,
    Z2D
  };

  enum class FFTDirection {
    FORWARD,
    BACKWARD
  };

  struct NoShape{};
  struct EmptyOp{};
  struct NoStride{};  
}

};