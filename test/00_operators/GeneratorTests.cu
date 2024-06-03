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

TYPED_TEST_SUITE(BasicGeneratorTestsAll, MatXAllTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsComplex, MatXComplexTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsFloat, MatXFloatTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsNumeric, MatXNumericTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsIntegral, MatXAllIntegralTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsNumericNonComplex,
                 MatXNumericNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatNonComplex, MatXFloatNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatNonComplexNonHalf,
                 MatXFloatNonComplexNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsBoolean, MatXBoolTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsFloatHalf, MatXFloatHalfTypesCUDAExec);
TYPED_TEST_SUITE(BasicGeneratorTestsNumericNoHalf, MatXNumericNonHalfTypesCUDAExec);

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, Windows)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  const index_t win_size = 100;
  pb->InitAndRunTVGenerator<TestType>("00_operators", "window", "run",
                                       {win_size});
  cuda::std::array<index_t, 1> shape({win_size});
  auto ov = make_tensor<TestType>(shape);

  // example-begin hanning-gen-test-1
  // Assign a Hanning window of size `win_size` to `ov`
  (ov = hanning<0>({win_size})).run(exec);
  // example-end hanning-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hanning", 0.01);

  // example-begin hamming-gen-test-1
  // Assign a Hamming window of size `win_size` to `ov`
  (ov = hamming<0>({win_size})).run(exec);
  // example-end hamming-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "hamming", 0.01);

  // example-begin bartlett-gen-test-1
  // Assign a bartlett window of size `win_size` to `ov`
  (ov = bartlett<0>({win_size})).run(exec);
  // example-end bartlett-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "bartlett", 0.01);

  // example-begin blackman-gen-test-1
  // Assign a blackman window of size `win_size` to `ov`
  (ov = blackman<0>({win_size})).run(exec);
  // example-end blackman-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "blackman", 0.01);

  // example-begin flattop-gen-test-1
  // Assign a flattop window of size `win_size` to `ov`
  (ov = flattop<0>({win_size})).run(exec);
  // example-end flattop-gen-test-1
  MATX_TEST_ASSERT_COMPARE(pb, ov, "flattop", 0.01);


  MATX_EXIT_HANDLER();  
}

TYPED_TEST(BasicGeneratorTestsAll, Diag)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};

  MATX_ENTER_HANDLER();
  {
    // example-begin diag-op-test-1
    // The generator form of `diag()` takes an operator input and returns only
    // the diagonal elements as output
    auto tc = make_tensor<TestType>({10, 10});
    auto td = make_tensor<TestType>({10});

    // Initialize the diagonal elements of `tc`
    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        TestType val(static_cast<detail::value_promote_t<TestType>>(i * 10 + j));
        tc(i, j) = val;
      }
    }

    // Assign the diagonal elements of `tc` to `td`.
    (td = diag(tc)).run(exec);
    // example-end diag-op-test-1
    exec.sync();

    for (int i = 0; i < 10; i++) {
      for (int j = 0; j < 10; j++) {
        if (i == j) {
          MATX_ASSERT_EQ(td(i), tc(i, j));
        }
      }
    }

    // Test with a nested transform. Restrict to floating point types for
    // the convolution
    if constexpr (std::is_same_v<TestType,float> || std::is_same_v<TestType,double>)
    {
      auto delta = make_tensor<TestType>({1});
      delta(0) = static_cast<TestType>(1.0);
      exec.sync();

      (td = 0).run(exec);
      (td = diag(conv1d(tc, delta, MATX_C_MODE_SAME))).run(exec);
      exec.sync();

      for (int i = 0; i < 10; i++) {
        for (int j = 0; j < 10; j++) {
          if (i == j) {
            MATX_ASSERT_EQ(td(i), tc(i, j));
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsFloat, Alternate)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};  

  // example-begin alternate-gen-test-1
  auto td = make_tensor<TestType>({10});

  // td contains the sequence 1, -1, 1, -1, 1, -1, 1, -1, 1, -1
  (td = alternate(10)).run(exec);
  // example-end alternate-gen-test-1

  exec.sync();

  for (int i = 0; i < 10; i++) {
    MATX_ASSERT_EQ(td(i), (TestType)-2* (TestType)(i&1) + (TestType)1)
  }

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, Kron)
{
  MATX_ENTER_HANDLER();

  cudaExecutor exec{};    
  using dtype = int;
  auto pb = std::make_unique<detail::MatXPybind>();
  pb->InitTVGenerator<dtype>("00_operators", "kron_operator", {});
  pb->RunTVGenerator("run");

  // example-begin kron-gen-test-1
  auto bv = make_tensor<dtype>({2, 2});
  auto ov = make_tensor<dtype>({8, 8});
  bv.SetVals({{1, -1}, {-1, 1}});

  (ov = kron(eye({4, 4}), bv)).run(exec);
  // example-end kron-gen-test-1
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, ov, "square", 0);

  tensor_t<dtype, 2> av({2, 3});
  tensor_t<dtype, 2> ov2({4, 6});
  av.SetVals({{1, 2, 3}, {4, 5, 6}});

  // example-begin ones-gen-test-2 
  // Explicit shape specified in ones()
  (ov2 = kron(av, ones({2, 2}))).run(exec);
  // example-end ones-gen-test-2  
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, ov2, "rect", 0);

  MATX_EXIT_HANDLER();
}

TEST(OperatorTests, MeshGrid)
{
  MATX_ENTER_HANDLER();
  
  cudaExecutor exec{};  
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

  (xv = xx).run(exec);
  (yv = yy).run(exec);
  // example-end meshgrid-gen-test-1

  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, xv, "X", 0);
  MATX_TEST_ASSERT_COMPARE(pb, yv, "Y", 0);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, FFTFreq)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};

  auto pb = std::make_unique<detail::MatXPybind>();
  pb->template InitAndRunTVGenerator<TestType>(
      "01_signal", "fftfreq", "run", {100});


  auto t1 = make_tensor<TestType>({100});
  auto t2 = make_tensor<TestType>({101});

  // example-begin fftfreq-gen-test-1
  // Generate FFT frequencies using the length of the "t1" tensor and assign to t1
  (t1 = fftfreq(t1.Size(0))).run(exec);
  // example-end fftfreq-gen-test-1
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, t1, "F1", 0.1);

  (t2 = fftfreq(t2.Size(0))).run(exec);
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, t2, "F2", 0.1);

  // example-begin fftfreq-gen-test-2
  // Generate FFT frequencies using the length of the "t1" tensor and a sample spacing of 0.5 and assign to t1
  (t1 = fftfreq(t1.Size(0), 0.5)).run(exec);
  // example-end fftfreq-gen-test-2
  exec.sync();
  MATX_TEST_ASSERT_COMPARE(pb, t1, "F3", 0.1);  

  MATX_EXIT_HANDLER();
}


TYPED_TEST(BasicGeneratorTestsAll, Zeros)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};    
  // example-begin zeros-gen-test-1    
  index_t count = 100;

  cuda::std::array<index_t, 1> s({count});
  auto t1 = make_tensor<TestType>(s);

  (t1 = zeros()).run(exec);
  // example-end zeros-gen-test-1

  exec.sync();

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TestType>()) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (float)0));
    }
    else {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)0));
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsAll, Ones)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};    
  // example-begin ones-gen-test-1    
  index_t count = 100;
  cuda::std::array<index_t, 1> s({count});
  auto t1 = make_tensor<TestType>(s);

  (t1 = ones()).run(exec);
  // example-end ones-gen-test-1    
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TestType>()) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (float)1));
    }
    else {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (TestType)1));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumericNonComplex, Range)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};   

  // example-begin range-gen-test-1
  index_t count = 100;
  tensor_t<TestType, 1> t1{{count}};

  // Generate a sequence of 100 numbers starting at 1 and spaced by 1
  (t1 = range<0>(t1.Shape(), 1, 1)).run(exec);
  // example-end range-gen-test-1  
  exec.sync();

  TestType one = 1;
  TestType two = 1;
  TestType three = 1;

  for (index_t i = 0; i < count; i++) {
    TestType it = static_cast<detail::value_promote_t<TestType>>(i);
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), it + one));
  }

  {
    (t1 = t1 * t1).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType it = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (it + one) * (it + one)));
    }
  }

  {
    (t1 = t1 * two).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType it = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) * two));
    }
  }

  {
    (t1 = three * t1).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      TestType it = static_cast<detail::value_promote_t<TestType>>(i);
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), ((it + one) * (it + one)) *
                                                        two * three));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsNumericNonComplex, Linspace)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};   

  // example-begin linspace-gen-test-1
  index_t count = 100;
  auto t1 = make_tensor<TestType>({count});

  // Create a set of linearly-spaced numbers starting at 1, ending at 100, and 
  // with `count` points in between
  (t1 = linspace<0>(t1.Shape(), (TestType)1, (TestType)100)).run(exec);
  // example-end linspace-gen-test-1
  exec.sync();

  for (index_t i = 0; i < count; i++) {
    EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), i + 1));
  }

  {
    (t1 = t1 + t1).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (i + 1) + (i + 1)));
    }
  }

  {
    (t1 = (TestType)1 + t1).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(
          MatXUtils::MatXTypeCompare(t1(i), (i + 1.0f) + (i + 1.0f) + 1.0f));
    }
  }

  {
    (t1 = t1 + (TestType)2).run(exec);
    exec.sync();

    for (index_t i = 0; i < count; i++) {
      EXPECT_TRUE(MatXUtils::MatXTypeCompare(t1(i), (i + 1) + (i + 1) + 1 + 2));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicGeneratorTestsFloatNonComplex, Logspace)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};   

  // example-begin logspace-gen-test-1
  index_t count = 20;
  tensor_t<TestType, 1> t1{{count}};
  TestType start = 1.0f;
  TestType stop = 2.0f;
  auto s = t1.Shape();

  // Create a logarithmically-spaced sequence of numbers and assign to tensor "t1"
  (t1 = logspace<0>(s, start, stop)).run(exec);
  // example-end logspace-gen-test-1

  exec.sync();

  // Use doubles for verification since half operators have no equivalent host
  // types
  double step = (static_cast<double>(stop) - static_cast<double>(start)) /
                static_cast<double>(s[s.size() - 1] - 1);

  for (index_t i = 0; i < count; i++) {
    if constexpr (IsHalfType<TestType>()) {
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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};   

  // example-begin eye-gen-test-1
  index_t count = 10;

  auto t2 = make_tensor<TestType>({count, count});
  auto t3 = make_tensor<TestType>({count, count, count});
  auto t4 = make_tensor<TestType>({count, count, count, count});

  auto eye_op = eye<TestType>();


  // For each of t2, t3, and t4 the values on the diagonal where each index is the same
  // is 1, and 0 everywhere else
  (t2 = eye_op).run(exec);
  (t3 = eye_op).run(exec);
  (t4 = eye_op).run(exec);
  // example-end eye-gen-test-1

  TestType one = 1.0f;
  TestType zero = 0.0f;

  exec.sync();

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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};     
  index_t count = 10;
  TestType c = GenerateData<TestType>();

  // example-begin diag-gen-test-1
  // Create a 2D, 3D, and 4D tensor and make only their diagonal elements set to `c`
  auto t2 = make_tensor<TestType>({count, count});
  auto t3 = make_tensor<TestType>({count, count, count});
  auto t4 = make_tensor<TestType>({count, count, count, count});

  auto diag2 = diag<TestType>({count, count}, c);
  auto diag3 = diag<TestType>({count, count, count}, c);
  auto diag4 = diag<TestType>({count, count, count, count}, c);

  (t2 = diag2).run(exec);
  (t3 = diag3).run(exec);
  (t4 = diag4).run(exec);
  // example-end diag-gen-test-1

  TestType zero = 0.0f;

  exec.sync();

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
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;  
  ExecType exec{};   
    
  index_t count = 1500;
  TestType end = 10;
  TestType f0 = -200;
  TestType f1 = 300;
  
  auto pb = std::make_unique<detail::MatXPybind>();
  pb->template InitAndRunTVGenerator<TestType>(
      "01_signal", "chirp", "run", {count, static_cast<index_t>(end), static_cast<index_t>(f0), static_cast<index_t>(f1)});  

  // example-begin chirp-gen-test-1
  auto t1 = make_tensor<TestType>({count});
  // Create a chirp of length "count" and assign it to tensor "t1"
  (t1 = chirp(count, end, f0, end, f1)).run(exec);
  // example-end chirp-gen-test-1

  MATX_TEST_ASSERT_COMPARE(pb, t1, "Y", 0.01);

  // example-begin cchirp-gen-test-1
  auto t1c = make_tensor<cuda::std::complex<TestType>>({count});
  // Create a complex chirp of length "count" and assign it to tensor "t1"
  (t1c = cchirp(count, end, f0, end, f1, ChirpMethod::CHIRP_METHOD_LINEAR)).run(exec);
  // example-end cchirp-gen-test-1

  pb.reset();
  MATX_EXIT_HANDLER();
}


