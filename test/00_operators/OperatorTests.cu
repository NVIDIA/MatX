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

#include "assert.h"
#include "matx.h"
#include "matx_pybind.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"
#include <type_traits>

using namespace matx;

template <typename TensorType>
class OperatorTestsComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloat : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumeric : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumericNonComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatNonComplex : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatNonComplexNonHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsIntegral : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsBoolean : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsFloatHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsNumericNoHalf : public ::testing::Test {
};
template <typename TensorType>
class OperatorTestsAll : public ::testing::Test {
};

TYPED_TEST_SUITE(OperatorTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(OperatorTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(OperatorTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(OperatorTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(OperatorTestsNumericNonComplex,
                 MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(OperatorTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypes);
TYPED_TEST_SUITE(OperatorTestsBoolean, MatXBoolTypes);
TYPED_TEST_SUITE(OperatorTestsFloatHalf, MatXFloatHalfTypes);
TYPED_TEST_SUITE(OperatorTestsNumericNoHalf, MatXNumericNoHalfTypes);

TYPED_TEST(OperatorTestsFloat, TrigFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;
  (tov0 = sin(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_sin(c)));

  (tov0 = cos(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_cos(c)));

  (tov0 = tan(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_tan(c)));

  (tov0 = asin(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_asin(c)));

  (tov0 = acos(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_acos(c)));

  (tov0 = atan(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_atan(c)));

  (tov0 = sinh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_sinh(c)));

  (tov0 = cosh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_cosh(c)));

  (tov0 = tanh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_tanh(c)));

  (tov0 = asinh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_asinh(c)));

  (tov0 = acosh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_acosh(c)));

  (tov0 = atanh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_atanh(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplex, AngleOp)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<typename TypeParam::value_type, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = angle(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_angle(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAll, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  TypeParam d = c;
  TypeParam z = 0;
  tiv0() = c;

  tensor_t<TypeParam, 0> tov00;

  IFELSE(tiv0 == d, tov0 = z, tov0 = d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  IFELSE(tiv0 == d, tov0 = tiv0, tov0 = d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), tiv0()));

  IFELSE(tiv0 != d, tov0 = d, tov0 = z).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), z));

  CHAIN(tov0 = c, tov00 = c).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov00(), c));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexNonHalf, OperatorFuncs)
{
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<cuda::std::complex<TypeParam>, 0> tov0;
  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = expj(tiv0)).run();
  cudaStreamSynchronize(0);

  EXPECT_TRUE(MatXUtils::MatXTypeCompare(
      tov0(),
      cuda::std::complex(cuda::std::cos(tiv0()), cuda::std::sin(tiv0()))));
}

TYPED_TEST(OperatorTestsFloatNonComplex, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = log10(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_log10(c)));

  (tov0 = log(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_log(c)));

  (tov0 = log2(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_log2(c)));

  (tov0 = floor(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_floor(c)));

  (tov0 = ceil(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_ceil(c)));

  (tov0 = round(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_round(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNonComplex, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;
  TypeParam d = c + 1;

  (tov0 = max(tiv0, d)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), max(c, d)));

  (tov0 = min(tiv0, d)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), min(c, d)));

  // These operators convert type T into bool
  tensor_t<bool, 0> tob;

  (tob = tiv0 < d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c < d));

  (tob = tiv0 > d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c > d));

  (tob = tiv0 <= d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c <= d));

  (tob = tiv0 >= d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c >= d));

  (tob = tiv0 == d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c == d));

  (tob = tiv0 != d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tob(), c != d));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = tiv0 + tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c + c));

  (tov0 = tiv0 - tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c - c));

  (tov0 = tiv0 * tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c * c));

  (tov0 = tiv0 / tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c / c));

  IF(tiv0 == tiv0, set(tov0, c)).run();

  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));

  TypeParam p = 2.0f;
  (tov0 = pow(tiv0, p)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_pow(c, p)));

  TypeParam three = 3.0f;

  (tov0 = tiv0 * tiv0 * (tiv0 + tiv0) / tiv0 + three).run();
  cudaStreamSynchronize(0);

  TypeParam res;
  res = c * c * (c + c) / c + three;
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), res, 0.07));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsIntegral, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;
  TypeParam mod = 2;

  (tov0 = tiv0 % mod).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c % mod));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsBoolean, OperatorFuncs)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  TypeParam d = false;
  tiv0() = c;

  (tov0 = tiv0 && d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c && d));

  (tov0 = tiv0 || d).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c || d));

  (tov0 = !tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), !c));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplex, OperatorFuncs)
{
  MATX_ENTER_HANDLER();

  tensor_t<TypeParam, 0> tiv0;
  tensor_t<TypeParam, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = exp(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_exp(c)));

  (tov0 = conj(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), _internal_conj(c)));

  // abs and norm take a complex and output a floating point value
  tensor_t<typename TypeParam::value_type, 0> tdd0;
  auto tvd0 = tdd0.View();
  (tvd0 = norm(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tvd0(), _internal_norm(c)));

  (tvd0 = abs(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tvd0(), _internal_abs(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNoHalf, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;

  tensor_t<TypeParam, 1> a({count});
  tensor_t<TypeParam, 1> b({count});
  tensor_t<TypeParam, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = static_cast<value_promote_t<TypeParam>>(i);
    b(i) = static_cast<value_promote_t<TypeParam>>(i + 100);
  }

  {
    (c = a + b).run();

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt + (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b).run();

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt * (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b + a).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TypeParam)100) + tcnt));
    }
  }

  {

    (c = a * b + a * (TypeParam)4.0f).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TypeParam)100.0f) + tcnt * (TypeParam)4));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatHalf, AdvancedOperators)
{
  MATX_ENTER_HANDLER();
  index_t count = 10;

  tensor_t<TypeParam, 1> a({count});
  tensor_t<TypeParam, 1> b({count});
  tensor_t<TypeParam, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = (double)i;
    b(i) = (double)(i + 2);
  }

  {
    (c = a + b).run();

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = (double)i;
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), (float)tcnt + ((float)tcnt + 2.0f)));
    }
  }

  {
    (c = a * b).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = (double)i;
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), (float)tcnt * ((float)tcnt + 2.0f)));
    }
  }

  {
    (c = a * b + a).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = (double)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt));
    }
  }

  {

    (c = a * b + a * (TypeParam)2.0f).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = (double)i;
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), (float)tcnt * ((float)tcnt + 2.0f) + (float)tcnt * 2.0f));
    }
  }
  MATX_EXIT_HANDLER();
}


// Testing 4 basic arithmetic operations with complex numbers and non-complex
TYPED_TEST(OperatorTestsComplex, ComplexTypeCompatibility)
{
  MATX_ENTER_HANDLER();
  index_t count = 10;

  tensor_t<float, 1> fview({count});
  tensor_t<TypeParam, 1> dview({count});

  using data_type =
      typename std::conditional_t<is_complex_half_v<TypeParam>, float,
                                  typename TypeParam::value_type>;

  // Multiply by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<value_promote_t<TypeParam>>(i),
                static_cast<value_promote_t<TypeParam>>(i)};
  }

  (dview = dview * fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<value_promote_t<TypeParam>>(i * i));
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<value_promote_t<TypeParam>>(i * i));
  }

  // Divide by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = i == 0 ? static_cast<float>(1) : static_cast<float>(i);
    dview(i) = {static_cast<value_promote_t<TypeParam>>(i),
                static_cast<value_promote_t<TypeParam>>(i)};
  }

  (dview = dview / fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).real()),
              i == 0 ? static_cast<value_promote_t<TypeParam>>(0)
                     : static_cast<value_promote_t<TypeParam>>(1));
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).imag()),
              i == 0 ? static_cast<value_promote_t<TypeParam>>(0)
                     : static_cast<value_promote_t<TypeParam>>(1));
  }

  // Add scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<value_promote_t<TypeParam>>(i),
                static_cast<value_promote_t<TypeParam>>(i)};
  }

  (dview = dview + fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<value_promote_t<TypeParam>>(i + i));
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<value_promote_t<TypeParam>>(i + i));
  }

  // Subtract scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<value_promote_t<TypeParam>>(i),
                static_cast<value_promote_t<TypeParam>>(i)};
  }

  (dview = dview - fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<value_promote_t<TypeParam>>(-1));
    ASSERT_EQ(static_cast<value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<value_promote_t<TypeParam>>(-1));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, SquareCopyTranspose)
{
  MATX_ENTER_HANDLER();
  index_t count = 512;
  tensor_t<TypeParam, 2> t2({count, count});
  tensor_t<TypeParam, 2> t2t({count, count});

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      t2(i, j) = static_cast<value_promote_t<TypeParam>>(i * count + j);
    }
  }

  t2.PrefetchDevice(0);
  t2t.PrefetchDevice(0);
  copy(t2t, t2, 0);

  t2t.PrefetchHost(0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2t(i, j),
                                             TypeParam(i * count + (double)j)));
    }
  }

  t2t.PrefetchDevice(0);
  transpose(t2t, t2, 0);

  t2t.PrefetchHost(0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j),
                                             TypeParam(i * count + (double)j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2t(j, i), TypeParam(i * count + j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, NonSquareTranspose)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  index_t count1 = 200, count2 = 100;
  tensor_t<TypeParam, 2> t2({count1, count2});
  tensor_t<TypeParam, 2> t2t({count2, count1});

  for (index_t i = 0; i < count1; i++) {
    for (index_t j = 0; j < count2; j++) {
      t2(i, j) = static_cast<value_promote_t<TypeParam>>(i * count + j);
    }
  }

  t2.PrefetchDevice(0);
  t2t.PrefetchDevice(0);
  transpose(t2t, t2, 0);

  t2t.PrefetchHost(0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count1; i++) {
    for (index_t j = 0; j < count2; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j),
                                             TypeParam(i * count + (double)j)));
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2t(j, i),
                                             TypeParam(i * count + (double)j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, CloneAndAdd)
{
  MATX_ENTER_HANDLER();
  index_t numSamples = 8;
  index_t numPulses = 4;
  index_t numPairs = 2;
  index_t numBeams = 2;

  tensor_t<float, 4> beamwiseRangeDoppler(
      {numBeams, numPulses, numPairs, numSamples});
  tensor_t<float, 2> steeredMx({numBeams, numSamples});
  tensor_t<float, 3> velAccelHypoth({numPulses, numPairs, numSamples});

  for (index_t i = 0; i < numBeams; i++) {
    for (index_t j = 0; j < numSamples; j++) {
      steeredMx(i, j) = static_cast<float>((i + 1) * 10 + (j + 1));
    }
  }

  for (index_t i = 0; i < numPulses; i++) {
    for (index_t j = 0; j < numPairs; j++) {
      for (index_t k = 0; k < numSamples; k++) {
        velAccelHypoth(i, j, k) = static_cast<float>(
            (i + 1) * 10000 + (j + 1) * 1000 + (k + 1) * 100);
      }
    }
  }

  auto smx =
      steeredMx.Clone<4>({matxKeepDim, numPulses, numPairs, matxKeepDim});
  auto vah = velAccelHypoth.Clone<4>(
      {numBeams, matxKeepDim, matxKeepDim, matxKeepDim});

  (beamwiseRangeDoppler = smx + vah).run();

  cudaStreamSynchronize(0);
  for (index_t i = 0; i < numBeams; i++) {
    for (index_t j = 0; j < numPulses; j++) {
      for (index_t k = 0; k < numPairs; k++) {
        for (index_t l = 0; l < numSamples; l++) {
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(
              beamwiseRangeDoppler(i, j, k, l),
              steeredMx(i, l) + velAccelHypoth(j, k, l)));
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(
              beamwiseRangeDoppler(i, j, k, l),
              ((i + 1) * 10 + (l + 1)) // steeredMx
                  + ((j + 1) * 10000 + (k + 1) * 1000 +
                     (l + 1) * 100) // velAccelHypoth
              ));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, Reshape)
{
  MATX_ENTER_HANDLER();
  index_t count = 10;
  tensor_t<TypeParam, 4> t4({count, count, count, count});
  tensor_t<TypeParam, 1> t1({count * count * count * count});

  for (index_t i = 0; i < t4.Size(0); i++) {
    for (index_t j = 0; j < t4.Size(1); j++) {
      for (index_t k = 0; k < t4.Size(2); k++) {
        for (index_t l = 0; l < t4.Size(3); l++) {
          t4(i, j, k, l) =
              static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          t1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
             i * t4.Size(3) * t4.Size(2) * t4.Size(1)) =
              static_cast<value_promote_t<TypeParam>>(i + j + k + l);
        }
      }
    }
  }

  // Drop to a single dimension of same original total size
  auto rsv1 = t4.View({count * count * count * count});
  for (index_t i = 0; i < t4.Size(0); i++) {
    for (index_t j = 0; j < t4.Size(1); j++) {
      for (index_t k = 0; k < t4.Size(2); k++) {
        for (index_t l = 0; l < t4.Size(3); l++) {
          MATX_ASSERT_EQ(rsv1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
                              i * t4.Size(3) * t4.Size(2) * t4.Size(1)),
                         (TypeParam)(i + j + k + (double)l));
        }
      }
    }
  }

  // Drop to 2D with a subset of the original size
  auto rsv2 = t4.View({2, 2});
  for (index_t i = 0; i < rsv2.Size(0); i++) {
    for (index_t j = 0; j < rsv2.Size(1); j++) {
      MATX_ASSERT_EQ(rsv2(i, j), t4(0, 0, 0, i * rsv2.Size(1) + j));
    }
  }

  // Create a 4D tensor from the 1D
  auto rsv4 = t1.View({count, count, count, count});
  for (index_t i = 0; i < rsv4.Size(0); i++) {
    for (index_t j = 0; j < rsv4.Size(1); j++) {
      for (index_t k = 0; k < rsv4.Size(2); k++) {
        for (index_t l = 0; l < rsv4.Size(3); l++) {
          MATX_ASSERT_EQ(rsv4(i, j, k, l),
                         t1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
                            i * t4.Size(3) * t4.Size(2) * t4.Size(1)));
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsFloatNonComplexNonHalf, VarianceStd)
{
  MATX_ENTER_HANDLER();
  auto pb = std::make_unique<MatXPybind>();
  constexpr index_t size = 100;
  pb->InitAndRunTVGenerator<TypeParam>("00_operators", "stats", "run", {size});

  tensor_t<TypeParam, 0> t0;
  tensor_t<TypeParam, 1> t1({size});
  pb->NumpyToTensorView(t1, "x");

  var(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "var", 0.01);

  stdd(t0, t1, 0);
  MATX_TEST_ASSERT_COMPARE(pb, t0, "std", 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexNonHalf, Reduce)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 0> t0;

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});
    auto t2 = ones<float>({30, 40});
    auto t1 = ones<float>({30});

    sum(t0, t4, 0);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(
        t0(), (TypeParam)(t4.Size(0) * t4.Size(1) * t4.Size(2) * t4.Size(3))));

    sum(t0, t3, 0);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(
        t0(), (TypeParam)(t3.Size(0) * t3.Size(1) * t3.Size(2))));

    sum(t0, t2, 0);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(
        MatXUtils::MatXTypeCompare(t0(), (TypeParam)(t2.Size(0) * t2.Size(1))));

    sum(t0, t1, 0);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(t1.Size(0))));
  }
  {
    tensor_t<TypeParam, 1> t1({30});

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});
    auto t2 = ones<float>({30, 40});
    // t4.Print();
    sum(t1, t4, 0);

    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TypeParam)(t4.Size(1) * t4.Size(2) * t4.Size(3))));
    }

    sum(t1, t3, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i), (TypeParam)(t3.Size(1) * t3.Size(2))));
    }

    sum(t1, t2, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(t2.Size(1))));
    }
  }

  {
    tensor_t<TypeParam, 2> t2({30, 40});

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});

    sum(t2, t4, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2(i, j), (TypeParam)(t4.Size(2) * t4.Size(3))));
      }
    }

    sum(t2, t3, 0);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(t3.Size(2))));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Any)
{
  MATX_ENTER_HANDLER();
  using TypeParam = float;
  {
    tensor_t<TypeParam, 0> t0;

    tensor_t<float, 1> t1({30});
    tensor_t<float, 2> t2({30, 40});
    tensor_t<float, 3> t3({30, 40, 50});
    tensor_t<float, 4> t4({30, 40, 50, 60});

    (t1 = zeros<float>(t1.Shape())).run();
    (t2 = zeros<float>(t2.Shape())).run();
    (t3 = zeros<float>(t3.Shape())).run();
    (t4 = zeros<float>(t4.Shape())).run();
    cudaStreamSynchronize(0);

    t1(5) = 5.0;
    t3(1, 1, 1) = 6.0;

    any(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    any(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    any(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    any(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, All)
{
  MATX_ENTER_HANDLER();
  using TypeParam = float;
  {
    tensor_t<TypeParam, 0> t0;

    tensor_t<float, 1> t1({30});
    tensor_t<float, 2> t2({30, 40});
    tensor_t<float, 3> t3({30, 40, 50});
    tensor_t<float, 4> t4({30, 40, 50, 60});

    (t1 = ones<float>(t1.Shape())).run();
    (t2 = ones<float>(t2.Shape())).run();
    (t3 = ones<float>(t3.Shape())).run();
    (t4 = ones<float>(t4.Shape())).run();
    cudaStreamSynchronize(0);

    t1(5) = 0.0;
    t3(1, 1, 1) = 0.0;

    all(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    all(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));

    all(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    all(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(0)));
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Median)
{
  MATX_ENTER_HANDLER();
  using TypeParam = float;
  {
    tensor_t<TypeParam, 0> t0{};
    tensor_t<TypeParam, 1> t1e{{10}};
    tensor_t<TypeParam, 1> t1o{{11}};
    tensor_t<TypeParam, 2> t2e{{2, 4}};
    tensor_t<TypeParam, 2> t2o{{2, 5}};
    tensor_t<TypeParam, 1> t1out{{2}};

    t1e.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0});
    t1o.SetVals({1, 3, 8, 2, 9, 6, 7, 4, 5, 0, 10});
    t2e.SetVals({{2, 4, 1, 3}, {3, 1, 2, 4}});
    t2o.SetVals({{2, 4, 1, 3, 5}, {3, 1, 5, 2, 4}});

    median(t0, t1e);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(4.5f)));

    median(t0, t1o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(5)));

    median(t1out, t2e);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TypeParam)(2.5f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TypeParam)(2.5f)));

    median(t1out, t2o);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(0), (TypeParam)(3.0f)));
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1out(1), (TypeParam)(3.0f)));
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Mean)
{
  MATX_ENTER_HANDLER();
  using TypeParam = float;
  {
    tensor_t<TypeParam, 0> t0;

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});
    auto t2 = ones<float>({30, 40});
    auto t1 = ones<float>({30});

    mean(t0, t4);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t3);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));

    mean(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(1)));
  }
  {
    tensor_t<TypeParam, 1> t1({30});

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});
    auto t2 = ones<float>({30, 40});

    mean(t1, t4);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }

    mean(t1, t3);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }

    mean(t1, t2);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t1.Size(0); i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)(1)));
    }
  }

  {
    tensor_t<TypeParam, 2> t2({30, 40});

    auto t4 = ones<float>({30, 40, 50, 60});
    auto t3 = ones<float>({30, 40, 50});

    mean(t2, t4);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(1)));
      }
    }

    mean(t2, t3);
    cudaStreamSynchronize(0);
    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), (TypeParam)(1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNonComplex, Prod)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 0> t0;

    tensorShape_t<2> s2({3, 4});
    tensorShape_t<1> s1({3});

    tensor_t<TypeParam, 1> t1{s1};
    tensor_t<TypeParam, 2> t2{s2};
    TypeParam t1p = (TypeParam)1;
    for (int i = 0; i < s1.Size(0); i++) {
      t1(i) = static_cast<value_promote_t<TypeParam>>((float)rand() /
                                                      (float)INT_MAX * 2.0f);
      t1p *= t1(i);
    }

    TypeParam t2p = (TypeParam)1;
    for (int i = 0; i < s2.Size(0); i++) {
      for (int j = 0; j < s2.Size(1); j++) {
        t2(i, j) = static_cast<value_promote_t<TypeParam>>(
            (float)rand() / (float)INT_MAX * 2.0f);
        t2p *= t2(i, j);
      }
    }

    prod(t0, t2);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t2p));

    prod(t0, t1);
    cudaStreamSynchronize(0);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), t1p));
  }

  MATX_EXIT_HANDLER();
}

// TYPED_TEST(OperatorTestsNumericNonComplex, Reduce)
// {
//   MATX_ENTER_HANDLER();
//   {
//     tensor_t<TypeParam, 0> t0data;
//     tensor_t<TypeParam, 4> t4data({30, 40, 50, 60});

//     auto t0 = t0data.View();
//     auto t4 = t4data.View();
//     for(index_t i = 0 ; i < t4.Size(0); i++) {
//       for(index_t j = 0 ; j < t4.Size(1); j++) {
//         for(index_t k = 0 ; k < t4.Size(2); k++) {
//           for(index_t l = 0 ; l < t4.Size(3); l++) {
//             t4(i,j,k,l) = (TypeParam) (i + j + k + l - 20);
//           }
//         }
//       }
//     }

//     reduce(t0, t4, reduceOpMax<TypeParam>(), 0);
//     cudaStreamSynchronize(0);
//     EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam) (t4.Size(0) +
//     t4.Size(1) + t4.Size(2) + t4.Size(3) - 20 - 4) ));

//     reduce(t0, t4, reduceOpMin<TypeParam>(), 0);
//     cudaStreamSynchronize(0);
//     EXPECT_TRUE(MatXUtils::MatXTypeCompare(t0(), (TypeParam)(-20) ));
//   }

//   MATX_EXIT_HANDLER();
// }

TYPED_TEST(OperatorTestsNumeric, Broadcast)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 0> t0;
    tensor_t<TypeParam, 4> t4i({10, 20, 30, 40});
    tensor_t<TypeParam, 4> t4o({10, 20, 30, 40});

    t0() = (TypeParam)2.0f;
    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t0).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t4i(i, j, k, l) * (double)t0());
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t0());
            }
          }
        }
      }
    }
    (t4o = t0 * t4i).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t0() * (double)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t0() * t4i(i, j, k, l));
            }
          }
        }
      }
    }
  }
  {
    tensor_t<TypeParam, 1> t1({4});
    tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
    tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

    for (index_t i = 0; i < t1.Size(0); i++) {
      t1(i) = static_cast<value_promote_t<TypeParam>>(i);
    }

    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t4i(i, j, k, l) * (double)t1(l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t1(l));
            }
          }
        }
      }
    }

    (t4o = t1 * t4i).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t1(l) * (double)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t1(l) * t4i(i, j, k, l));
            }
          }
        }
      }
    }
  }

  {
    tensor_t<TypeParam, 2> t2({3, 4});
    tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
    tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        t2(i, j) = static_cast<value_promote_t<TypeParam>>(i + j);
      }
    }

    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t2).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t4i(i, j, k, l) * (double)t2(k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t2(k, l));
            }
          }
        }
      }
    }

    (t4o = t2 * t4i).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t2(k, l) * (double)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t2(k, l) * t4i(i, j, k, l));
            }
          }
        }
      }
    }
  }

  {
    tensor_t<TypeParam, 3> t3({2, 3, 4});
    tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
    tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

    for (index_t i = 0; i < t3.Size(0); i++) {
      for (index_t j = 0; j < t3.Size(1); j++) {
        for (index_t k = 0; k < t3.Size(2); k++) {
          t3(i, j, k) = static_cast<value_promote_t<TypeParam>>(i + j + k);
        }
      }
    }

    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i * t3).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t4i(i, j, k, l) * (double)t3(j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t3(j, k, l));
            }
          }
        }
      }
    }

    (t4o = t3 * t4i).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t3(j, k, l) * (double)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t3(j, k, l) * t4i(i, j, k, l));
            }
          }
        }
      }
    }
  }

  {
    tensor_t<TypeParam, 0> t0;
    tensor_t<TypeParam, 1> t1({4});
    tensor_t<TypeParam, 2> t2({3, 4});
    tensor_t<TypeParam, 3> t3({2, 3, 4});
    tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
    tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

    t0() = (TypeParam)200.0f;

    for (index_t i = 0; i < t2.Size(0); i++) {
      t1(i) = static_cast<value_promote_t<TypeParam>>(i);
    }

    for (index_t i = 0; i < t2.Size(0); i++) {
      for (index_t j = 0; j < t2.Size(1); j++) {
        t2(i, j) = static_cast<value_promote_t<TypeParam>>(i + j);
      }
    }

    for (index_t i = 0; i < t3.Size(0); i++) {
      for (index_t j = 0; j < t3.Size(1); j++) {
        for (index_t k = 0; k < t3.Size(2); k++) {
          t3(i, j, k) = static_cast<value_promote_t<TypeParam>>(i + j + k);
        }
      }
    }

    for (index_t i = 0; i < t4i.Size(0); i++) {
      for (index_t j = 0; j < t4i.Size(1); j++) {
        for (index_t k = 0; k < t4i.Size(2); k++) {
          for (index_t l = 0; l < t4i.Size(3); l++) {
            t4i(i, j, k, l) =
                static_cast<value_promote_t<TypeParam>>(i + j + k + l);
          }
        }
      }
    }

    (t4o = t4i + t3 + t2 + t1 + t0).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t4i(i, j, k, l) + (double)t3(j, k, l) +
                                 (double)t2(k, l) + (double)t1(l) +
                                 (double)(double)t0());
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) + t3(j, k, l) +
                                                  t2(k, l) + t1(l) + t0());
            }
          }
        }
      }
    }

    (t4o = t0 + t1 + t2 + t3 + t4i).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < t4o.Size(0); i++) {
      for (index_t j = 0; j < t4o.Size(1); j++) {
        for (index_t k = 0; k < t4o.Size(2); k++) {
          for (index_t l = 0; l < t4o.Size(3); l++) {
            if constexpr (IsHalfType<TypeParam>()) {
              MATX_ASSERT_EQ(t4o(i, j, k, l),
                             (double)t0() + (double)t1(l) + (double)t2(k, l) +
                                 (double)t3(j, k, l) + (double)t4i(i, j, k, l));
            }
            else {
              MATX_ASSERT_EQ(t4o(i, j, k, l), t0() + t1(l) + t2(k, l) +
                                                  t3(j, k, l) +
                                                  t4i(i, j, k, l));
            }
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}
