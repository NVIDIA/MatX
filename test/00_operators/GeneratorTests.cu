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
class BasicGeneratorTestsComplex : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsFloat : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsNumeric : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsNumericNonComplex : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsFloatNonComplex : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsFloatNonComplexNonHalf : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsIntegral : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsBoolean : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsFloatHalf : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsNumericNoHalf : public ::testing::Test {
};
template <typename TensorType>
class BasicGeneratorTestsAll : public ::testing::Test {
};

TYPED_TEST_SUITE(BasicGeneratorTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsNumericNonComplex,
                 MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsBoolean, MatXBoolTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatHalf, MatXFloatHalfTypes);
TYPED_TEST_SUITE(BasicGeneratorTestsNumericNoHalf, MatXNumericNoHalfTypes);

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, Windows)
{
  MATX_ENTER_HANDLER();

  auto pb = std::make_unique<MatXPybind>();
  const index_t win_size = 100;
  pb->InitAndRunTVGenerator<TypeParam>("00_operators", "window", "run",
                                       {win_size});
  std::array<index_t, 1> shape({win_size});
  auto ov = make_tensor<TypeParam>(shape);

  (ov = hanning_x(shape)).run();
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hanning", 0.01);

  (ov = hamming_x(shape)).run();
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hamming", 0.01);

  (ov = bartlett_x(shape)).run();
  MATX_TEST_ASSERT_COMPARE(pb, ov, "bartlett", 0.01);

  (ov = blackman_x(shape)).run();
  MATX_TEST_ASSERT_COMPARE(pb, ov, "blackman", 0.01);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsAll, Diag)
{
  MATX_ENTER_HANDLER();
  {
    tensor_t<TypeParam, 2> tc({10, 10});
    tensor_t<TypeParam, 1> td({10});

    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {

        // The half precision headers define competing constructors for
        // double/float, so we need to cast
        TypeParam val(static_cast<value_promote_t<TypeParam>>(i * 10 + j));
        tc(i, j) = val;
      }
    }

    (td = diag(tc)).run();
    cudaStreamSynchronize(0);

    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        if (i == j) {
          MATX_ASSERT_EQ(td(i), tc(i, j));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Kron)
{
  MATX_ENTER_HANDLER();
  using dtype = int;
  auto pb = std::make_unique<MatXPybind>();
  pb->InitTVGenerator<dtype>("00_operators", "kron_operator", {});
  pb->RunTVGenerator("run");

  tensor_t<dtype, 2> bv({2, 2});
  tensor_t<dtype, 2> ov({8, 8});
  bv.SetVals({{1, -1}, {-1, 1}});

  (ov = kron(eye({4, 4}), bv)).run();
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, ov, "square", 0);

  tensor_t<dtype, 2> av({2, 3});
  tensor_t<dtype, 2> ov2({4, 6});
  av.SetVals({{1, 2, 3}, {4, 5, 6}});

  (ov2 = kron(av, ones({2, 2}))).run();
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, ov2, "rect", 0);

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, MeshGrid)
{
  MATX_ENTER_HANDLER();
  using dtype = int;
  auto pb = std::make_unique<MatXPybind>();
  constexpr index_t xd = 3;
  constexpr index_t yd = 5;
  pb->InitAndRunTVGenerator<dtype>("00_operators", "meshgrid_operator", "run",
                                   {yd, xd});

  tensor_t<dtype, 2> xv({yd, xd});
  tensor_t<dtype, 2> yv({yd, xd});

  (xv = meshgrid_x({1, xd, xd}, {1, yd, yd})).run();
  (yv = meshgrid_y({1, xd, xd}, {1, yd, yd})).run();
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, xv, "X", 0);
  MATX_TEST_ASSERT_COMPARE(pb, yv, "Y", 0);

  MATX_EXIT_HANDLER();
}



TYPED_TEST(BasicGeneratorTestsAll, Zeros)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  std::array<index_t, 1> s({count});

  auto t1 = make_tensor<TypeParam>(s);

  (t1 = zeros(s)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TypeParam>()) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (float)0));
    }
    else {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)0));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsAll, Ones)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  std::array<index_t, 1> s({count});
  auto t1 = make_tensor<TypeParam>(s);

  (t1 = ones(s)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TypeParam>()) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (float)1));
    }
    else {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TypeParam)1));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumericNonComplex, Range)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  tensor_t<TypeParam, 1> t1{{count}};

  (t1 = range_x(t1.Shape(), 1, 1)).run();
  cudaStreamSynchronize(0);

  TypeParam one = 1;
  TypeParam two = 1;
  TypeParam three = 1;

  for (index_t i = 0; i < count; i++) {
    TypeParam it = static_cast<value_promote_t<TypeParam>>(i);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), it + one));
  }

  {
    (t1 = t1 * t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (it + one) * (it + one)));
    }
  }

  {
    (t1 = t1 * two).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) * two));
    }
  }

  {
    (t1 = three * t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) *
                                                        two * three));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumericNonComplex, Linspace)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  tensor_t<TypeParam, 1> t1{{count}};
  auto s = t1.Shape();
  (t1 = linspace_x(s, (TypeParam)1, (TypeParam)100)).run();
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < count; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), i + 1));
  }

  {
    (t1 = t1 + t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (i + 1) + (i + 1)));
    }
  }

  {
    (t1 = (TypeParam)1 + t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t1(i), (i + 1.0f) + (i + 1.0f) + 1.0f));
    }
  }

  {
    (t1 = t1 + (TypeParam)2).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (i + 1) + (i + 1) + 1 + 2));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, Logspace)
{
  MATX_ENTER_HANDLER();
  index_t count = 20;
  tensor_t<TypeParam, 1> t1{{count}};
  TypeParam start = 1.0f;
  TypeParam stop = 2.0f;
  auto s = t1.Shape();
  (t1 = logspace_x(s, start, stop)).run();

  cudaStreamSynchronize(0);

  // Use doubles for verification since half operators have no equivalent host
  // types
  double step = (static_cast<double>(stop) - static_cast<double>(start)) /
                static_cast<double>(s[s.size() - 1] - 1);

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TypeParam>()) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i),
          cuda::std::powf(10, static_cast<double>(start) +
                                  static_cast<double>(step) *
                                      static_cast<double>(i)),
          2));
    }
    else {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(
          t1(i),
          cuda::std::powf(10, static_cast<double>(start) +
                                  static_cast<double>(step) *
                                      static_cast<double>(i)),
          0.01));
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(BasicGeneratorTestsNumeric, Eye)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;

  tensor_t<TypeParam, 1> t1({count});
  tensor_t<TypeParam, 2> t2({count, count});
  tensor_t<TypeParam, 3> t3({count, count, count});
  tensor_t<TypeParam, 4> t4({count, count, count, count});

  t1.PrefetchDevice(0);
  t2.PrefetchDevice(0);
  t3.PrefetchDevice(0);
  t4.PrefetchDevice(0);

  auto eye1 = eye<TypeParam>({count});
  auto eye2 = eye<TypeParam>({count, count});
  auto eye3 = eye<TypeParam>({count, count, count});
  auto eye4 = eye<TypeParam>({count, count, count, count});

  (t1 = eye1).run();
  (t2 = eye2).run();
  (t3 = eye3).run();
  (t4 = eye4).run();

  t1.PrefetchHost(0);
  t2.PrefetchHost(0);
  t3.PrefetchHost(0);
  t4.PrefetchHost(0);

  TypeParam one = 1.0f;
  TypeParam zero = 0.0f;

  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    if (i == 0)
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), one));
    else
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), zero));
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      if (i == j)
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), one));
      else
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), zero));
    }
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      for (index_t k = 0; k < count; k++) {
        if (i == j && j == k)
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(t3(i, j, k), one));
        else
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(t3(i, j, k), zero));
      }
    }
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      for (index_t k = 0; k < count; k++) {
        for (index_t l = 0; l < count; l++) {
          if (i == j && j == k && k == l)
            EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4(i, j, k, l), one));
          else
            EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4(i, j, k, l), zero));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumeric, Diag)
{
  MATX_ENTER_HANDLER();
  index_t count = 100;
  TypeParam c = GenerateData<TypeParam>();

  tensor_t<TypeParam, 1> t1({count});
  tensor_t<TypeParam, 2> t2({count, count});
  tensor_t<TypeParam, 3> t3({count, count, count});
  tensor_t<TypeParam, 4> t4({count, count, count, count});

  t1.PrefetchDevice(0);
  t2.PrefetchDevice(0);
  t3.PrefetchDevice(0);
  t4.PrefetchDevice(0);

  auto diag1 = diag<TypeParam>({count}, c);
  auto diag2 = diag<TypeParam>({count, count}, c);
  auto diag3 = diag<TypeParam>({count, count, count}, c);
  auto diag4 = diag<TypeParam>({count, count, count, count}, c);

  (t1 = diag1).run();
  (t2 = diag2).run();
  (t3 = diag3).run();
  (t4 = diag4).run();

  t1.PrefetchHost(0);
  t2.PrefetchHost(0);
  t3.PrefetchHost(0);
  t4.PrefetchHost(0);

  TypeParam zero = 0.0f;

  cudaDeviceSynchronize();

  for (index_t i = 0; i < count; i++) {
    if (i == 0)
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), c));
    else
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), zero));
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      if (i == j)
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), c));
      else
        EXPECT_TRUE(MatXUtils::MatXTypeCompare(t2(i, j), zero));
    }
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      for (index_t k = 0; k < count; k++) {
        if (i == j && j == k)
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(t3(i, j, k), c));
        else
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(t3(i, j, k), zero));
      }
    }
  }

  for (index_t i = 0; i < count; i++) {
    for (index_t j = 0; j < count; j++) {
      for (index_t k = 0; k < count; k++) {
        for (index_t l = 0; l < count; l++) {
          if (i == j && j == k && k == l)
            EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4(i, j, k, l), c));
          else
            EXPECT_TRUE(MatXUtils::MatXTypeCompare(t4(i, j, k, l), zero));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}


TYPED_TEST(BasicGeneratorTestsComplex, HermitianTranspose)
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

TYPED_TEST(BasicGeneratorTestsComplex, PlanarTransform)
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
          MatXUtils::MatXTypeCompare(t2(i, j).imag(), t2p(i + t2.Size(0), j)));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsComplex, InterleavedTransform)
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

TYPED_TEST(BasicGeneratorTestsAll, RepMat)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 4;
  index_t count1 = 4;
  index_t same_reps = 10;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2s({count0 * same_reps, count1 * same_reps});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<value_promote_t<TypeParam>>(i);
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

TYPED_TEST(BasicGeneratorTestsNumeric, Shift)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 100;
  index_t count1 = 201;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2s({count0, count1});
  tensor_t<TypeParam, 2> t2s2({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<value_promote_t<TypeParam>>(i * count1 + j);
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

TYPED_TEST(BasicGeneratorTestsNumeric, Reverse)
{
  MATX_ENTER_HANDLER();
  index_t count0 = 100;
  index_t count1 = 200;
  tensor_t<TypeParam, 2> t2({count0, count1});
  tensor_t<TypeParam, 2> t2r({count0, count1});

  for (index_t i = 0; i < count0; i++) {
    for (index_t j = 0; j < count1; j++) {
      t2(i, j) = static_cast<value_promote_t<TypeParam>>(i * count1 + j);
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

