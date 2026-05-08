////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDERS OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/error.h"
#include "matx/core/type_utils.h"
#include "matx/transforms/solver_common.h"

#include <string>
#include <vector>

namespace matx {
namespace detail {

template <typename T>
inline constexpr bool is_dense_solve_supported_type_v =
    std::is_same_v<T, float> ||
    std::is_same_v<T, double> ||
    std::is_same_v<T, cuda::std::complex<float>> ||
    std::is_same_v<T, cuda::std::complex<double>>;

template <typename ATensor, typename BTensor>
static constexpr bool IsDenseSolveVectorRHS()
{
  return remove_cvref_t<BTensor>::Rank() == remove_cvref_t<ATensor>::Rank() - 1;
}

template <typename OutputTensor, typename ATensor, typename BTensor>
__MATX_INLINE__ void ValidateDenseSolve(OutputTensor &&out,
                                        const ATensor &a,
                                        const BTensor &b)
{
  using OutTensor_t = remove_cvref_t<OutputTensor>;
  using ATensor_t = remove_cvref_t<ATensor>;
  using BTensor_t = remove_cvref_t<BTensor>;
  using T = typename ATensor_t::value_type;

  constexpr int ARANK = ATensor_t::Rank();
  constexpr int BRANK = BTensor_t::Rank();
  constexpr int ORANK = OutTensor_t::Rank();

  static_assert(ARANK >= 2, "Dense solve A must have rank 2 or higher");
  static_assert(BRANK == ARANK || BRANK == ARANK - 1,
                "Dense solve B must have rank A.Rank() or A.Rank() - 1");
  static_assert(ORANK == BRANK,
                "Dense solve output rank must match B rank");
  static_assert(std::is_same_v<T, typename BTensor_t::value_type>,
                "Dense solve A and B value types must match");
  static_assert(std::is_same_v<T, typename OutTensor_t::value_type>,
                "Dense solve output value type must match A and B");
  static_assert(is_dense_solve_supported_type_v<T>,
                "Dense solve supports float, double, complex<float>, and complex<double>");

  MATX_ASSERT_STR(a.Size(ARANK - 1) == a.Size(ARANK - 2),
                  matxInvalidSize,
                  "Dense solve A must be square");

  for (int i = 0; i < ORANK; i++) {
    MATX_ASSERT_STR(out.Size(i) == b.Size(i), matxInvalidSize,
                    "Dense solve output shape must match B");
  }

  if constexpr (IsDenseSolveVectorRHS<ATensor_t, BTensor_t>()) {
    for (int i = 0; i < BRANK - 1; i++) {
      MATX_ASSERT_STR(a.Size(i) == b.Size(i), matxInvalidSize,
                      "Dense solve A and B batch dimensions must match");
    }
    MATX_ASSERT_STR(b.Size(BRANK - 1) == a.Size(ARANK - 1), matxInvalidSize,
                    "Dense solve vector RHS length must match A");
  }
  else {
    for (int i = 0; i < ARANK - 2; i++) {
      MATX_ASSERT_STR(a.Size(i) == b.Size(i), matxInvalidSize,
                      "Dense solve A and B batch dimensions must match");
    }
    MATX_ASSERT_STR(b.Size(BRANK - 2) == a.Size(ARANK - 1), matxInvalidSize,
                    "Dense solve matrix RHS row count must match A");
  }
}

template <typename ATensor, typename BTensor>
__MATX_INLINE__ index_t GetDenseSolveNumRhs(const BTensor &b)
{
  if constexpr (IsDenseSolveVectorRHS<ATensor, BTensor>()) {
    return 1;
  }
  else {
    return b.Size(remove_cvref_t<BTensor>::Rank() - 1);
  }
}

__MATX_INLINE__ void CheckDenseSolveInfo(const int info,
                                         const char *provider,
                                         const char *routine)
{
  if (info < 0) {
    MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
      ("Parameter " + std::to_string(-info) + " had an illegal value in " +
       provider + " " + routine).c_str());
  }
  else {
    MATX_ASSERT_STR_EXP(info, 0, matxSolverError,
      ("Singular matrix in " + std::string(provider) + " " + routine +
       ": U(" + std::to_string(info) + "," + std::to_string(info) +
       ") = 0").c_str());
  }
}

__MATX_INLINE__ void CheckDenseSolveInfos(const std::vector<int> &infos,
                                          const char *provider,
                                          const char *routine)
{
  for (const auto info : infos) {
    CheckDenseSolveInfo(info, provider, routine);
  }
}

} // end namespace detail
} // end namespace matx
