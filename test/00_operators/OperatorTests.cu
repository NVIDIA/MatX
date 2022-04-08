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
  return;
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sin(c)));

  (tov0 = cos(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cos(c)));

  (tov0 = tan(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tan(c)));

  (tov0 = asin(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asin(c)));

  (tov0 = acos(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acos(c)));

  (tov0 = atan(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atan(c)));

  (tov0 = sinh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_sinh(c)));

  (tov0 = cosh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_cosh(c)));

  (tov0 = tanh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_tanh(c)));

  (tov0 = asinh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_asinh(c)));

  (tov0 = acosh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_acosh(c)));

  (tov0 = atanh(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_atanh(c)));

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
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_angle(c)));  

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplex, RealImagOp)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 0> tiv0;
  tensor_t<typename TypeParam::value_type, 0> tov0;

  TypeParam c = GenerateData<TypeParam>();
  tiv0() = c;

  (tov0 = real(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.real()));  

  (tov0 = imag(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c.imag()));   

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

  (tov0 = c, tov00 = c).run();
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
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log10(c)));

  (tov0 = log(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log(c)));

  (tov0 = log2(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_log2(c)));

  (tov0 = floor(tiv0)).run();   
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_floor(c)));

  (tov0 = ceil(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_ceil(c)));

  (tov0 = round(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_round(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsFloatNonComplexNonHalf, NDOperatorFuncs)
{
  MATX_ENTER_HANDLER();
  auto a = make_tensor<TypeParam>({1,2,3,4,5,6,7,8});
  auto b = make_tensor<TypeParam>({1,2,3,4,5,6,7,8});
  (a = ones(a.Shape())).run();
  cudaDeviceSynchronize();
  (b = ones(b.Shape())).run();
  cudaDeviceSynchronize();
  (a = a + b).run();

  auto t0 = make_tensor<TypeParam>();
  sum(t0, a);
  cudaStreamSynchronize(0);
  ASSERT_EQ(t0(), 2 * a.TotalSize());
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

  (tov0 = -tiv0).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), -c));

  IF(tiv0 == tiv0, tov0 = c).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), c));

  TypeParam p = 2.0f;
  (tov0 = pow(tiv0, p)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_pow(c, p)));

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
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_exp(c)));

  (tov0 = conj(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tov0(), detail::_internal_conj(c)));

  // abs and norm take a complex and output a floating point value
  tensor_t<typename TypeParam::value_type, 0> tdd0;
  (tdd0 = norm(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::_internal_norm(c)));

  (tdd0 = abs(tiv0)).run();
  cudaStreamSynchronize(0);
  EXPECT_TRUE(MatXUtils::MatXTypeCompare(tdd0(), detail::_internal_abs(c)));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAll, Flatten)
{
  MATX_ENTER_HANDLER();

  auto t2 = make_tensor<TypeParam>({10, 2});
  auto val = GenerateData<TypeParam>();

  for (index_t i = 0; i < t2.Size(0); i++) {
    for (index_t j = 0; j < t2.Size(1); j++) {
      t2(i,j) = val;
    }
  }

  auto t1 = make_tensor<TypeParam>({t2.Size(0)*t2.Size(1)});
  (t1 = flatten(t2)).run();
  cudaStreamSynchronize(0);
  
  for (index_t i = 0; i < t2.Size(0)*t2.Size(1); i++) {
    ASSERT_EQ(t1(i), val);
  }

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
    a(i) = static_cast<detail::value_promote_t<TypeParam>>(i);
    b(i) = static_cast<detail::value_promote_t<TypeParam>>(i + 100);
  }

  {
    (c = a + b).run();

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt + (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b).run();

    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt * (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b + a).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TypeParam)100) + tcnt));
    }
  }

  {

    (c = a * b + a * (TypeParam)4.0f).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TypeParam)100.0f) + tcnt * (TypeParam)4));
    }
  }
  MATX_EXIT_HANDLER();
}

// TODO: add more host testing on everything that supports it
TYPED_TEST(OperatorTestsNumericNoHalf, HostAdvancedOperators)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;

  tensor_t<TypeParam, 1> a({count});
  tensor_t<TypeParam, 1> b({count});
  tensor_t<TypeParam, 1> c({count});

  for (index_t i = 0; i < count; i++) {
    a(i) = static_cast<detail::value_promote_t<TypeParam>>(i);
    b(i) = static_cast<detail::value_promote_t<TypeParam>>(i + 100);
  }

  {
    (c = a + b).run(SingleThreadHostExecutor{});
    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt + (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b).run(SingleThreadHostExecutor{});

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(c(i), tcnt * (tcnt + (TypeParam)100)));
    }
  }

  {
    (c = a * b + a).run(SingleThreadHostExecutor{});

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          c(i), tcnt * (tcnt + (TypeParam)100) + tcnt));
    }
  }

  {

    (c = a * b + a * (TypeParam)4.0f).run(SingleThreadHostExecutor{});

    for (index_t i = 0; i < count; i++) {
      TypeParam tcnt = static_cast<detail::value_promote_t<TypeParam>>(i);
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
    dview(i) = {static_cast<detail::value_promote_t<TypeParam>>(i),
                static_cast<detail::value_promote_t<TypeParam>>(i)};
  }

  (dview = dview * fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<detail::value_promote_t<TypeParam>>(i * i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TypeParam>>(i * i));
  }

  // Divide by scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = i == 0 ? static_cast<float>(1) : static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TypeParam>>(i),
                static_cast<detail::value_promote_t<TypeParam>>(i)};
  }

  (dview = dview / fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).real()),
              i == 0 ? static_cast<detail::value_promote_t<TypeParam>>(0)
                     : static_cast<detail::value_promote_t<TypeParam>>(1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).imag()),
              i == 0 ? static_cast<detail::value_promote_t<TypeParam>>(0)
                     : static_cast<detail::value_promote_t<TypeParam>>(1));
  }

  // Add scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i);
    dview(i) = {static_cast<detail::value_promote_t<TypeParam>>(i),
                static_cast<detail::value_promote_t<TypeParam>>(i)};
  }

  (dview = dview + fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<detail::value_promote_t<TypeParam>>(i + i));
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TypeParam>>(i + i));
  }

  // Subtract scalar
  for (index_t i = 0; i < count; i++) {
    fview(i) = static_cast<float>(i + 1);
    dview(i) = {static_cast<detail::value_promote_t<TypeParam>>(i),
                static_cast<detail::value_promote_t<TypeParam>>(i)};
  }

  (dview = dview - fview).run();
  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).real()),
              static_cast<detail::value_promote_t<TypeParam>>(-1));
    ASSERT_EQ(static_cast<detail::value_promote_t<TypeParam>>(dview(i).imag()),
              static_cast<detail::value_promote_t<TypeParam>>(-1));
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
      t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i * count + j);
    }
  }

  t2.PrefetchDevice(0);
  t2t.PrefetchDevice(0);
  matx::copy(t2t, t2, 0);

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
      t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i * count + j);
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
              static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
          t1(l + k * t4.Size(3) + j * t4.Size(3) * t4.Size(2) +
             i * t4.Size(3) * t4.Size(2) * t4.Size(1)) =
              static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
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


  // Test if oversized views throw
  try {
    t4.View({1000, 1000, 100});
    FAIL() << "Oversized views not throwing";
  } catch (detail::matxException &e) {}


  MATX_EXIT_HANDLER();
}


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
                static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
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
  // }
  // {
  //   tensor_t<TypeParam, 1> t1({4});
  //   tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t1.Size(0); i++) {
  //     t1(i) = static_cast<detail::value_promote_t<TypeParam>>(i);
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t1).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t1(l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t1(l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t1 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t1(l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t1(l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TypeParam, 2> t2({3, 4});
  //   tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     for (index_t j = 0; j < t2.Size(1); j++) {
  //       t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i + j);
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t2).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t2(k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t2(k, l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t2 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t2(k, l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t2(k, l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TypeParam, 3> t3({2, 3, 4});
  //   tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

  //   for (index_t i = 0; i < t3.Size(0); i++) {
  //     for (index_t j = 0; j < t3.Size(1); j++) {
  //       for (index_t k = 0; k < t3.Size(2); k++) {
  //         t3(i, j, k) = static_cast<detail::value_promote_t<TypeParam>>(i + j + k);
  //       }
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i * t3).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) * (double)t3(j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) * t3(j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t3 * t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t3(j, k, l) * (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t3(j, k, l) * t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

  // {
  //   tensor_t<TypeParam, 0> t0;
  //   tensor_t<TypeParam, 1> t1({4});
  //   tensor_t<TypeParam, 2> t2({3, 4});
  //   tensor_t<TypeParam, 3> t3({2, 3, 4});
  //   tensor_t<TypeParam, 4> t4i({1, 2, 3, 4});
  //   tensor_t<TypeParam, 4> t4o({1, 2, 3, 4});

  //   t0() = (TypeParam)200.0f;

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     t1(i) = static_cast<detail::value_promote_t<TypeParam>>(i);
  //   }

  //   for (index_t i = 0; i < t2.Size(0); i++) {
  //     for (index_t j = 0; j < t2.Size(1); j++) {
  //       t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i + j);
  //     }
  //   }

  //   for (index_t i = 0; i < t3.Size(0); i++) {
  //     for (index_t j = 0; j < t3.Size(1); j++) {
  //       for (index_t k = 0; k < t3.Size(2); k++) {
  //         t3(i, j, k) = static_cast<detail::value_promote_t<TypeParam>>(i + j + k);
  //       }
  //     }
  //   }

  //   for (index_t i = 0; i < t4i.Size(0); i++) {
  //     for (index_t j = 0; j < t4i.Size(1); j++) {
  //       for (index_t k = 0; k < t4i.Size(2); k++) {
  //         for (index_t l = 0; l < t4i.Size(3); l++) {
  //           t4i(i, j, k, l) =
  //               static_cast<detail::value_promote_t<TypeParam>>(i + j + k + l);
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t4i + t3 + t2 + t1 + t0).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t4i(i, j, k, l) + (double)t3(j, k, l) +
  //                                (double)t2(k, l) + (double)t1(l) +
  //                                (double)(double)t0());
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t4i(i, j, k, l) + t3(j, k, l) +
  //                                                 t2(k, l) + t1(l) + t0());
  //           }
  //         }
  //       }
  //     }
  //   }

  //   (t4o = t0 + t1 + t2 + t3 + t4i).run();
  //   cudaStreamSynchronize(0);

  //   for (index_t i = 0; i < t4o.Size(0); i++) {
  //     for (index_t j = 0; j < t4o.Size(1); j++) {
  //       for (index_t k = 0; k < t4o.Size(2); k++) {
  //         for (index_t l = 0; l < t4o.Size(3); l++) {
  //           if constexpr (IsHalfType<TypeParam>()) {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l),
  //                            (double)t0() + (double)t1(l) + (double)t2(k, l) +
  //                                (double)t3(j, k, l) + (double)t4i(i, j, k, l));
  //           }
  //           else {
  //             MATX_ASSERT_EQ(t4o(i, j, k, l), t0() + t1(l) + t2(k, l) +
  //                                                 t3(j, k, l) +
  //                                                 t4i(i, j, k, l));
  //           }
  //         }
  //       }
  //     }
  //   }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumericNonComplex, Concatenate)
{
  MATX_ENTER_HANDLER();

  index_t i, j;

  auto t11 = make_tensor<TypeParam>({10});
  auto t12 = make_tensor<TypeParam>({5});
  auto t1o = make_tensor<TypeParam>({15});
  auto t1o1 = make_tensor<TypeParam>({30});

  t11.SetVals({0,1,2,3,4,5,6,7,8,9});
  t12.SetVals({0,1,2,3,4});

  (t1o = concat<0>(t11, t12)).run();
  cudaStreamSynchronize(0);

  for (i = 0; i < t11.Size(0) + t12.Size(0); i++) {
    if (i < t11.Size(0)) {
      ASSERT_EQ(t11(i), t1o(i));
    }
    else {
      ASSERT_EQ(t12(i - t11.Size(0)), t1o(i));
    }
  }

  // 2D tensors
  auto t21 = make_tensor<TypeParam>({4, 4});
  auto t22 = make_tensor<TypeParam>({3, 4});
  auto t23 = make_tensor<TypeParam>({4, 3});

  auto t2o1 = make_tensor<TypeParam>({7,4});  
  auto t2o2 = make_tensor<TypeParam>({4,7});  
  t21.SetVals({{1,2,3,4},
               {2,3,4,5},
               {3,4,5,6},
               {4,5,6,7}} );
  t22.SetVals({{5,6,7,8},
               {6,7,8,9},
               {9,10,11,12}});
  t23.SetVals({{5,6,7},
               {6,7,8},
               {9,10,11},
               {10,11,12}});

  (t2o1 = concat<0>(t21, t22)).run();
  cudaStreamSynchronize(0);

  for (i = 0; i < t21.Size(0) + t22.Size(0); i++) {
    for (j = 0; j < t21.Size(1); j++) {
      if (i < t21.Size(0)) {
        ASSERT_EQ(t21(i,j), t2o1(i,j));
      }
      else {
        ASSERT_EQ(t22(i - t21.Size(0), j), t2o1(i,j));
      }
    }
  }

  (t2o2 = concat<1>(t21, t23)).run(); 
  cudaStreamSynchronize(0);
  
  for (j = 0; j < t21.Size(1) + t23.Size(1); j++) {
    for (i = 0; i < t21.Size(0); i++) {
      if (j < t21.Size(1)) {
        ASSERT_EQ(t21(i,j), t2o2(i,j));
      }
      else {
        ASSERT_EQ(t23(i, j - t21.Size(1)), t2o2(i,j));
      }
    }
  }  

  // Concatenating 3 tensors
  (t1o1 = concat<0>(t11, t11, t11)).run();
  cudaStreamSynchronize(0);

  for (i = 0; i < t1o1.Size(0); i++) {
    ASSERT_EQ(t1o1(i), t11(i % t11.Size(0)));
  }
  
  MATX_EXIT_HANDLER();
}


TYPED_TEST(OperatorTestsComplex, HermitianTranspose)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2s({count1, count0});
  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      TypeParam tmp = {(float)i, (float)-j};
      t2(i, j) = tmp;
    }
  }

  (t2s = hermitianT(t2)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(static_cast<double>(t2s(j, i).real()),
                                     static_cast<double>(t2(i, j).real())));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(-static_cast<double>(t2s(j, i).imag()),
                                     static_cast<double>(t2(i, j).imag())));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplex, PlanarTransform)
{
  MATX_ENTER_HANDLER();
  index_t m = 10;
  index_t k = 20;
  tensor_t<TypeParam, 2> t2({m, k});
  tensor_t<typename TypeParam::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      TypeParam tmp = {(float)i, (float)-j};
      t2(i, j) = tmp;
    }
  }

  (t2p = planar(t2)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j))) << i << " " << j << "\n";
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsComplex, InterleavedTransform)
{
  MATX_ENTER_HANDLER();
  index_t m = 10;
  index_t k = 20;
  tensor_t<TypeParam, 2> t2({m, k});
  tensor_t<typename TypeParam::value_type, 2> t2p({m * 2, k});
  for (index_t i = 0; i < 2 * m; i++) {
    for (index_t j = 0; j < k; j++) {
      if (i >= m) {
        t2p(i, j) = 2.0f;
      }
      else {
        t2p(i, j) = -1.0f;
      }
    }
  }

  (t2 = interleaved(t2p)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < m; i++) {
    for (index_t j = 0; j < k; j++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j).real(), t2p(i, j)));
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsAll, RepMat)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  index_t count1 = 4;
  index_t same_reps = 10;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2s({count0 * same_reps, count1 * same_reps});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i);
    }
  }

  auto repop = repmat(t2, same_reps);
  ASSERT_TRUE(repop.Size(0) == same_reps * t2.Size(0));
  ASSERT_TRUE(repop.Size(1) == same_reps * t2.Size(1));

  (t2s = repop).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count0 * same_reps; i++) {
    for (index_t j = 0; j < count1 * same_reps; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2s(i, j), t2(i % count0, j % count1)));
    }
  }

  // Now a rectangular repmat
  tensor_t<TypeParam, 2> t2r({count0 * same_reps, count1 * same_reps * 2});

  auto rrepop = repmat(t2, {same_reps, same_reps * 2});
  ASSERT_TRUE(rrepop.Size(0) == same_reps * t2.Size(0));
  ASSERT_TRUE(rrepop.Size(1) == same_reps * 2 * t2.Size(1));

  (t2r = rrepop).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count0 * same_reps; i++) {
    for (index_t j = 0; j < count1 * same_reps * 2; j++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t2r(i, j), t2(i % count0, j % count1)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, Shift)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 100;
  index_t count1 = 201;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2s({count0, count1});
  tensor_t<TypeParam, 2> t2s2({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i * count1 + j);
    }
  }

  {
    (t2s = shift0(t2, 5)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2((i + 5) % count0, j)));
      }
    }
  }

  {
    (t2s = shift1(t2, 5)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, (j + 5) % count1)));
      }
    }
  }

  {
    (t2s = shift0(shift1(t2, 5), 6)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + 6) % count0, (j + 5) % count1)));
      }
    }
  }

  {
    (t2s = fftshift2D(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j), t2((i + (count0 + 1) / 2) % count0,
                          (j + (count1 + 1) / 2) % count1)));
      }
    }
  }

  {
    (t2s = ifftshift2D(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2s(i, j),
            t2((i + (count0) / 2) % count0, (j + (count1) / 2) % count1)));
      }
    }
  }

  // Negative shifts
  {
    (t2s = shift0(t2, -5)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t idim = i < 5 ? (t2.Size(0) - 5 + i) : (i - 5);
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(idim, j)));
      }
    }
  }

  {
    (t2s = shift1(t2, -5)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        index_t jdim = j < 5 ? (t2.Size(1) - 5 + j) : (j - 5);
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, jdim)));
      }
    }
  }

  // Large shifts
  {
    (t2s = shift0(t2, t2.Size(0) * 4)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2(i, j)));
      }
    }
  }

  {
    // Shift 4 times the size back, minus one. This should be equivalent to
    // simply shifting by -1
    (t2s = shift0(t2, -t2.Size(0) * 4 - 1)).run();
    (t2s2 = shift0(t2, -1)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2s(i, j), t2s2(i, j)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(OperatorTestsNumeric, Reverse)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2r({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<detail::value_promote_t<TypeParam>>(i * count1 + j);
    }
  }

  {
    (t2r = reverseY(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t2r = reverseX(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  {
    (t2r = reverseX(reverseY(t2))).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(
            t2r(i, j), t2(count0 - i - 1, count1 - j - 1)));
      }
    }
  }

  // Flip versions
  {
    (t2r = flipud(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(count0 - i - 1, j)));
      }
    }
  }

  {
    (t2r = fliplr(t2)).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count0; i++) {
      for (index_t j = 0; j < count1; j++) {
        EXPECT_TRUE(
            MatXUtils::MatXTypeCompare(t2r(i, j), t2(i, count1 - j - 1)));
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Cast)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  auto t = make_tensor<int8_t>({count0});
  auto t2 = make_tensor<int8_t>({count0});
  auto to = make_tensor<float>({count0});

  t.SetVals({126, 126, 126, 126});
  t2.SetVals({126, 126, 126, 126});
  
  (to = as_type<int8_t>(t + t2)).run();
  cudaStreamSynchronize(0);

  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }

  (to = as_int8(t + t2)).run();
  cudaStreamSynchronize(0);
  
  for (int i = 0; i < t.Size(0); i++) {
    ASSERT_EQ(to(i), -4); // -4 from 126 + 126 wrap-around
  }  

  MATX_EXIT_HANDLER();
}

