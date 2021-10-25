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

#include "matx_half.h"
#include "matx_half_complex.h"
#include "matx_type_utils.h"
#include "gtest/gtest.h"
#include <cuda/std/ccomplex>

using testing::Types;

template <typename T> auto inline GenerateData();
template <> auto inline GenerateData<bool>() { return true; }
template <> auto inline GenerateData<matx::matxFp16>()
{
  return matx::matxFp16{1.0};
}
template <> auto inline GenerateData<matx::matxBf16>()
{
  return matx::matxBf16{1.0f};
}
template <> auto inline GenerateData<int32_t>() { return -1; }
template <> auto inline GenerateData<uint32_t>() { return 1; }
template <> auto inline GenerateData<int64_t>() { return -1; }
template <> auto inline GenerateData<uint64_t>() { return 1; }
template <> auto inline GenerateData<float>() { return 1.0f; }
template <> auto inline GenerateData<double>() { return 1.5; }
template <> auto inline GenerateData<matx::matxFp16Complex>()
{
  return matx::matxFp16Complex(1.5, -2.5);
}
template <> auto inline GenerateData<matx::matxBf16Complex>()
{
  return matx::matxBf16Complex(1.5, -2.5);
}
template <> auto inline GenerateData<cuda::std::complex<float>>()
{
  return cuda::std::complex<float>(1.5, -2.5);
}
template <> auto inline GenerateData<cuda::std::complex<double>>()
{
  return cuda::std::complex<double>(1.5, -2.5);
}

// Define the types to test for each group. If a type is put into a list that
// isn't compatible with a test type, a compiler error will occur

typedef Types<matx::matxFp16, matx::matxBf16, bool, uint32_t, int32_t, uint64_t,
              int64_t, float, double, cuda::std::complex<float>,
              cuda::std::complex<double>, matx::matxFp16Complex,
              matx::matxBf16Complex>
    MatXAllTypes;
typedef Types<matx::matxFp16, matx::matxBf16, float, double,
              cuda::std::complex<float>, cuda::std::complex<double>,
              matx::matxFp16Complex, matx::matxBf16Complex>
    MatXFloatTypes;
typedef Types<matx::matxFp16, matx::matxBf16, float, double>
    MatXFloatNonComplexTypes;
typedef Types<matx::matxFp16, matx::matxBf16> MatXFloatHalfTypes;
typedef Types<matx::matxFp16, matx::matxBf16, uint32_t, int32_t, uint64_t,
              int64_t, float, double, cuda::std::complex<float>,
              cuda::std::complex<double>, matx::matxFp16Complex,
              matx::matxBf16Complex>
    MatXNumericTypes;

typedef Types<uint32_t, int32_t, uint64_t, int64_t, float, double,
              cuda::std::complex<float>, cuda::std::complex<double>>
    MatXNumericNoHalfTypes;
typedef Types<bool> MatXBoolTypes;
typedef Types<float, double> MatXFloatNonComplexNonHalfTypes;
typedef Types<cuda::std::complex<float>, cuda::std::complex<double>,
              matx::matxFp16Complex, matx::matxBf16Complex>
    MatXComplexTypes;
typedef Types<cuda::std::complex<float>, cuda::std::complex<double>>
    MatXComplexNonHalfTypes;
typedef Types<uint32_t, int32_t, uint64_t, int64_t, float, double>
    MatXNumericNonComplexTypes;
typedef Types<uint32_t, int32_t, uint64_t, int64_t> MatXAllIntegralTypes;
typedef Types<int32_t, int64_t> MatXSignedIntegralTypes;
typedef Types<uint32_t, uint64_t> MatXUnsignedIntegralTypes;
