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

  auto pb = std::make_unique<detail::MatXPybind>();
  const index_t win_size = 100;
  pb->InitAndRunTVGenerator<TypeParam>("00_operators", "window", "run",
                                       {win_size});
  std::array<index_t, 1> shape({win_size});
  auto ov = make_tensor<TypeParam>(shape);

  // example-begin hanning-gen-test-1
  // Assign a Hanning window of size `win_size` to `ov`
  (ov = hanning<0>({win_size})).run();
  // example-end hanning-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hanning", 0.01);

  // example-begin hamming-gen-test-1
  // Assign a Hamming window of size `win_size` to `ov`
  (ov = hamming<0>({win_size})).run();
  // example-end hamming-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hamming", 0.01);

  // example-begin bartlett-gen-test-1
  // Assign a bartlett window of size `win_size` to `ov`
  (ov = bartlett<0>({win_size})).run();
  // example-end bartlett-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "bartlett", 0.01);

  // example-begin blackman-gen-test-1
  // Assign a blackman window of size `win_size` to `ov`
  (ov = blackman<0>({win_size})).run();
  // example-end blackman-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "blackman", 0.01);

  // example-begin flattop-gen-test-1
  // Assign a flattop window of size `win_size` to `ov`
  (ov = flattop<0>({win_size})).run();
  // example-end flattop-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "flattop", 0.01);


  MATX_EXIT_HANDLER();  
}

TYPED_TEST(BasicGeneratorTestsAll, Diag)
{
  MATX_ENTER_HANDLER();
  {
    // example-begin diag-op-test-1
    // The generator form of `diag()` takes an operator input and returns only
    // the diagonal elements as output
    auto tc = make_tensor<TypeParam>({10, 10});
    auto td = make_tensor<TypeParam>({10});

    // Initialize the diagonal elements of `tc`
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        TypeParam val(static_cast<detail::value_promote_t<TypeParam>>(i * 10 + j));
        tc(i, j) = val;
      }
    }

    // Assign the diagonal elements of `tc` to `td`.
    (td = diag(tc)).run();
    // example-end diag-op-test-1
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

TYPED_TEST(BasicGeneratorTestsFloat, Alternate)
{
  MATX_ENTER_HANDLER();

  // example-begin alternate-gen-test-1
  auto td = make_tensor<TypeParam>({10});

  // td contains the sequence 1, -1, 1, -1, 1, -1, 1, -1, 1, -1
  (td = alternate(10)).run();
  // example-end alternate-gen-test-1

  cudaStreamSynchronize(0);

  for (int i = 0; i < 10; i++) {
    MATX_ASSERT_EQ(td(i), (TypeParam)-2* (TypeParam)(i&1) + (TypeParam)1)
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Kron)
{
  MATX_ENTER_HANDLER();
  using dtype = int;
  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitTVGenerator<dtype>("00_operators", "kron_operator", {});
  pb->RunTVGenerator("run");

  // example-begin kron-gen-test-1
  auto bv = make_tensor<dtype>({2, 2});
  auto ov = make_tensor<dtype>({8, 8});
  bv.SetVals({{1, -1}, {-1, 1}});

  (ov = kron(eye({4, 4}), bv)).run();
  // example-end kron-gen-test-1
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
  auto pb = std::make_unique<detail::MatXPybind>();
  constexpr dtype xd = 3;
  constexpr dtype yd = 5;
  pb->InitAndRunTVGenerator<dtype>("00_operators", "meshgrid_operator", "run", {xd, yd});

  // example-begin meshgrid-gen-test-1
  auto xv = make_tensor<dtype>({yd, xd});
  auto yv = make_tensor<dtype>({yd, xd});

  auto x = linspace<0>({xd}, 1, xd);
  auto y = linspace<0>({yd}, 1, yd);

  // Create a mesh grid with "x" as x extents and "y" as y extents and assign it to "xv"/"yv"
  auto [xx, yy] = meshgrid(x, y);

  (xv = xx).run();
  (yv = yy).run();
  // example-end meshgrid-gen-test-1

  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, xv, "X", 0);
  MATX_TEST_ASSERT_COMPARE(pb, yv, "Y", 0);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, FFTFreq)
{
  MATX_ENTER_HANDLER();
  auto pb = std::make_unique<detail::MatXPybind>();
  pb->template InitAndRunTVGenerator<TypeParam>(
      "01_signal", "fftfreq", "run", {100});


  auto t1 = make_tensor<TypeParam>({100});
  auto t2 = make_tensor<TypeParam>({101});

  // example-begin fftfreq-gen-test-1
  // Generate FFT frequencies using the length of the "t1" tensor and assign to t1
  (t1 = fftfreq(t1.Size(0))).run();
  // example-end fftfreq-gen-test-1
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, t1, "F1", 0.1);

  (t2 = fftfreq(t2.Size(0))).run();
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, t2, "F2", 0.1);

  // example-begin fftfreq-gen-test-2
  // Generate FFT frequencies using the length of the "t1" tensor and a sample spacing of 0.5 and assign to t1
  (t1 = fftfreq(t1.Size(0), 0.5)).run();
  // example-end fftfreq-gen-test-2
  cudaStreamSynchronize(0);
  MATX_TEST_ASSERT_COMPARE(pb, t1, "F3", 0.1);  

  MATX_EXIT_HANDLER();
}


TYPED_TEST(BasicGeneratorTestsAll, Zeros)
{
  MATX_ENTER_HANDLER();
  // example-begin zeros-gen-test-1    
  index_t count = 100;

  std::array<index_t, 1> s({count});
  auto t1 = make_tensor<TypeParam>(s);

  (t1 = zeros(s)).run();
  // example-end zeros-gen-test-1

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
  // example-begin ones-gen-test-1    
  index_t count = 100;
  std::array<index_t, 1> s({count});
  auto t1 = make_tensor<TypeParam>(s);

  (t1 = ones(s)).run();
  // example-end ones-gen-test-1    
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
  // example-begin range-gen-test-1
  index_t count = 100;
  tensor_t<TypeParam, 1> t1{{count}};

  // Generate a sequence of 100 numbers starting at 1 and spaced by 1
  (t1 = range<0>(t1.Shape(), 1, 1)).run();
  // example-end range-gen-test-1  
  cudaStreamSynchronize(0);

  TypeParam one = 1;
  TypeParam two = 1;
  TypeParam three = 1;

  for (index_t i = 0; i < count; i++) {
    TypeParam it = static_cast<detail::value_promote_t<TypeParam>>(i);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), it + one));
  }

  {
    (t1 = t1 * t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (it + one) * (it + one)));
    }
  }

  {
    (t1 = t1 * two).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) * two));
    }
  }

  {
    (t1 = three * t1).run();
    cudaStreamSynchronize(0);

    for (index_t i = 0; i < count; i++) {
      TypeParam it = static_cast<detail::value_promote_t<TypeParam>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) *
                                                        two * three));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumericNonComplex, Linspace)
{
  MATX_ENTER_HANDLER();
  // example-begin linspace-gen-test-1
  index_t count = 100;
  auto t1 = make_tensor<TypeParam>({count});

  // Create a set of linearly-spaced numbers starting at 1, ending at 100, and 
  // with `count` points in between
  (t1 = linspace<0>(t1.Shape(), (TypeParam)1, (TypeParam)100)).run();
  // example-end linspace-gen-test-1
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
  // example-begin logspace-gen-test-1
  index_t count = 20;
  tensor_t<TypeParam, 1> t1{{count}};
  TypeParam start = 1.0f;
  TypeParam stop = 2.0f;
  auto s = t1.Shape();

  // Create a logarithmically-spaced sequence of numbers and assign to tensor "t1"
  (t1 = logspace<0>(s, start, stop)).run();
  // example-end logspace-gen-test-1

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
  // example-begin eye-gen-test-1
  index_t count = 10;

  auto t2 = make_tensor<TypeParam>({count, count});
  auto t3 = make_tensor<TypeParam>({count, count, count});
  auto t4 = make_tensor<TypeParam>({count, count, count, count});

  auto eye2 = eye<TypeParam>({count, count});
  auto eye3 = eye<TypeParam>({count, count, count});
  auto eye4 = eye<TypeParam>({count, count, count, count});

  // For each of t2, t3, and t4 the values on the diagonal where each index is the same
  // is 1, and 0 everywhere else
  (t2 = eye2).run();
  (t3 = eye3).run();
  (t4 = eye4).run();
  // example-end eye-gen-test-1

  TypeParam one = 1.0f;
  TypeParam zero = 0.0f;

  cudaDeviceSynchronize();

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
  index_t count = 10;
  TypeParam c = GenerateData<TypeParam>();

  // example-begin diag-gen-test-1
  // Create a 2D, 3D, and 4D tensor and make only their diagonal elements set to `c`
  auto t2 = make_tensor<TypeParam>({count, count});
  auto t3 = make_tensor<TypeParam>({count, count, count});
  auto t4 = make_tensor<TypeParam>({count, count, count, count});

  auto diag2 = diag<TypeParam>({count, count}, c);
  auto diag3 = diag<TypeParam>({count, count, count}, c);
  auto diag4 = diag<TypeParam>({count, count, count, count}, c);

  (t2 = diag2).run();
  (t3 = diag3).run();
  (t4 = diag4).run();
  // example-end diag-gen-test-1

  TypeParam zero = 0.0f;

  cudaDeviceSynchronize();

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

TYPED_TEST(BasicGeneratorTestsFloatNonComplexNonHalf, Chirp)
{
  MATX_ENTER_HANDLER();
  index_t count = 1500;
  TypeParam end = 10;
  TypeParam f0 = -200;
  TypeParam f1 = 300;
  
  auto pb = std::make_unique<detail::MatXPybind>();
  pb->template InitAndRunTVGenerator<TypeParam>(
      "01_signal", "chirp", "run", {count, static_cast<index_t>(end), static_cast<index_t>(f0), static_cast<index_t>(f1)});  

  // example-begin chirp-gen-test-1
  auto t1 = make_tensor<TypeParam>({count});
  // Create a chirp of length "count" and assign it to tensor "t1"
  (t1 = chirp(count, end, f0, end, f1)).run();
  // example-end chirp-gen-test-1

  MATX_TEST_ASSERT_COMPARE(pb, t1, "Y", 0.01);

  // example-begin cchirp-gen-test-1
  auto t1c = make_tensor<cuda::std::complex<TypeParam>>({count});
  // Create a complex chirp of length "count" and assign it to tensor "t1"
  (t1c = cchirp(count, end, f0, end, f1, ChirpMethod::CHIRP_METHOD_LINEAR)).run();
  // example-end cchirp-gen-test-1

  pb.reset();
  MATX_EXIT_HANDLER();
}


