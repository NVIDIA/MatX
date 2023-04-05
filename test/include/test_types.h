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

#include <cuda/std/ccomplex>
#include "matx/executors/device.h"
#include "matx/executors/host.h"
#include "gtest/gtest.h"
#include "matx.h"

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
template <> auto inline GenerateData<int32_t>() { return (int32_t)-1; }
template <> auto inline GenerateData<uint32_t>() { return (uint32_t)1; }
template <> auto inline GenerateData<int64_t>() { return (int64_t)-1; }
template <> auto inline GenerateData<uint64_t>() { return(uint64_t) 1; }
template <> auto inline GenerateData<float>() { return (float)1.0f; }
template <> auto inline GenerateData<double>() { return (double)1.5; }
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

using ExecutorTypesAll = std::tuple<matx::cudaExecutor, matx::SingleThreadHostExecutor>;

// Define the types to test for each group. If a type is put into a list that
// isn't compatible with a test type, a compiler error will occur

using MatXAllTypes = Types<matx::matxFp16, matx::matxBf16, bool, uint32_t, int32_t, uint64_t,
              int64_t, float, double, cuda::std::complex<float>,
              cuda::std::complex<double>, matx::matxFp16Complex,
              matx::matxBf16Complex>;
using MatXFloatTypes = Types<matx::matxFp16, matx::matxBf16, float, double,
              cuda::std::complex<float>, cuda::std::complex<double>,
              matx::matxFp16Complex, matx::matxBf16Complex>;
using MatXFloatNonHalfTypes = Types<float, double,
              cuda::std::complex<float>, cuda::std::complex<double>>;
using MatXFloatNonComplexTypes = Types<matx::matxFp16, matx::matxBf16, float, double>;
  
using MatXFloatHalfTypes = Types<matx::matxFp16, matx::matxBf16>;
using MatXNumericTypes = Types<matx::matxFp16, matx::matxBf16, uint32_t, int32_t, uint64_t,
              int64_t, float, double, cuda::std::complex<float>,
              cuda::std::complex<double>, matx::matxFp16Complex,
              matx::matxBf16Complex>;

using MatXNumericNoHalfTypes = Types<uint32_t, int32_t, uint64_t, int64_t, float, double,
              cuda::std::complex<float>, cuda::std::complex<double>>;
using MatXBoolTypes =  Types<bool>;
using MatXFloatNonComplexNonHalfTypes = Types<float, double>;
using MatXComplexTypes = Types<cuda::std::complex<float>, cuda::std::complex<double>,
              matx::matxFp16Complex, matx::matxBf16Complex>;
using MatXComplexNonHalfTypes = Types<cuda::std::complex<float>, cuda::std::complex<double>>;
using MatXNumericNonComplexTypes = Types<uint32_t, int32_t, uint64_t, int64_t, float, double>;
using MatXAllIntegralTypes = Types<uint32_t, int32_t, uint64_t, int64_t>;
using MatXSignedIntegralTypes = Types<int32_t, int64_t>;
using MatXUnsignedIntegralTypes = Types<uint32_t, uint64_t>;


/* Taken from https://stackoverflow.com/questions/70404549/cartesian-product-of-stdtuple */
template<typename T1, typename T2>
class TypedCartesianProduct {
  template<typename T, typename... Ts>
  static auto innerHelper(T&&, std::tuple<Ts...>&&) 
  -> decltype(
       std::make_tuple(
         std::make_tuple(std::declval<T>(), std::declval<Ts>())...));

  template <typename... Ts, typename T>
  static auto outerHelper(std::tuple<Ts...>&&, T&&) 
  -> decltype(
       std::tuple_cat(innerHelper(std::declval<Ts>(), std::declval<T>())...));

 public:
  using type = decltype(outerHelper(std::declval<T1>(), std::declval<T2>()));
};


template <typename Tuple>
struct TupleToTypes {};

template <typename... T>
struct TupleToTypes<std::tuple<T...>>
{
  using type = ::Types<T...>;
};


using MatXFloatNonComplexNonHalfTuple        = std::tuple<float, double>;
using MatXNumericNoHalfTuple                 = std::tuple<uint32_t, int32_t, uint64_t, int64_t, float, double,
                                                          cuda::std::complex<float>, cuda::std::complex<double>>;
using MatXComplexNonHalfTuple                = std::tuple<cuda::std::complex<float>, cuda::std::complex<double>>;
using MatXNumericNonComplexTuple             = std::tuple<uint32_t, int32_t, uint64_t, int64_t, float, double>;
using MatXComplexTuple                       = std::tuple<cuda::std::complex<float>, cuda::std::complex<double>,
                                                          matx::matxFp16Complex, matx::matxBf16Complex>;
                                                          
using MatXAllTuple                           = std::tuple<matx::matxFp16, matx::matxBf16, bool, uint32_t, int32_t, uint64_t,
                                                      int64_t, float, double, cuda::std::complex<float>,
                                                      cuda::std::complex<double>, matx::matxFp16Complex,
                                                      matx::matxBf16Complex>;
using MatXFloatTuple                         = std::tuple< matx::matxFp16, matx::matxBf16, float, double,
                                                           cuda::std::complex<float>, cuda::std::complex<double>,
                                                           matx::matxFp16Complex, matx::matxBf16Complex>;    

using MatXNumericTuple                       = std::tuple<matx::matxFp16, matx::matxBf16, uint32_t, int32_t, uint64_t,
                                              int64_t, float, double, cuda::std::complex<float>,
                                              cuda::std::complex<double>, matx::matxFp16Complex,
                                              matx::matxBf16Complex>;     
using MatXIntegralTuple                      = std::tuple<uint32_t, int32_t, uint64_t, int64_t>;                                                                                                                                                         

using MatXFloatNonComplexTuple               = std::tuple<matx::matxFp16, matx::matxBf16, float, double>;                                                      
using MatXBooleanTuple                       = std::tuple<bool>;   

using MatXNumericNonComplexTypesAllExecs      = TupleToTypes<TypedCartesianProduct<MatXNumericNonComplexTuple, ExecutorTypesAll>::type>::type;
using MatXFloatNonComplexNonHalfTypesAllExecs = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexNonHalfTuple, ExecutorTypesAll>::type>::type;
using MatXNumericNoHalfTypesAllExecs          = TupleToTypes<TypedCartesianProduct<MatXNumericNoHalfTuple, ExecutorTypesAll>::type>::type;
using MatXComplexNonHalfTypesAllExecs         = TupleToTypes<TypedCartesianProduct<MatXComplexNonHalfTuple, ExecutorTypesAll>::type>::type;
using MatXComplexTypesAllExecs                = TupleToTypes<TypedCartesianProduct<MatXComplexTuple, ExecutorTypesAll>::type>::type;

using MatXTypesAllExecs                       = TupleToTypes<TypedCartesianProduct<MatXAllTuple, ExecutorTypesAll>::type>::type;
using MatXTypesFloatNonComplexAllExecs        = TupleToTypes<TypedCartesianProduct<MatXFloatNonComplexTuple, ExecutorTypesAll>::type>::type;
using MatXTypesFloatAllExecs                  = TupleToTypes<TypedCartesianProduct<MatXFloatTuple, ExecutorTypesAll>::type>::type;
using MatXTypesNumericAllExecs                  = TupleToTypes<TypedCartesianProduct<MatXNumericTuple, ExecutorTypesAll>::type>::type;
using MatXTypesIntegralAllExecs                  = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAll>::type>::type;
using MatXTypesBooleanAllExecs                  = TupleToTypes<TypedCartesianProduct<MatXIntegralTuple, ExecutorTypesAll>::type>::type;
