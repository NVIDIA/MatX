////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include "operator_test_types.hpp"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"

using namespace matx;
using namespace matx::test;

#ifdef MATX_EN_UNSAFE_ALIAS_DETECTION

// Test that simple element-wise aliasing (a = a + a) does NOT throw
TYPED_TEST(OperatorTestsNumericAllExecs, SimpleAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({10, 20});

  // This should NOT throw because it's a simple element-wise operation
  EXPECT_NO_THROW({
    (a = a + a).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}

// Test that non-aliasing (b = a + c) does NOT throw an error
TYPED_TEST(OperatorTestsNumericAllExecs, NoAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({10, 20});
  auto b = make_tensor<TestType>({10, 20});
  auto c = make_tensor<TestType>({10, 20});

  // This should NOT throw because a, b, and c are all different tensors
  EXPECT_NO_THROW({
    (b = a + c).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}


// Test that non-overlapping slices do NOT throw
TYPED_TEST(OperatorTestsNumericAllExecs, NonOverlappingSlices)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({20, 20});

  // Create non-overlapping slices
  auto a_slice1 = slice(a, {0, 0}, {10, 10});
  auto a_slice2 = slice(a, {10, 10}, {20, 20});
  
  // This should NOT throw because the slices don't overlap
  EXPECT_NO_THROW({
    (a_slice1 = a_slice2).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}

// Test overlapping slices with element-wise operations do NOT throw
TYPED_TEST(OperatorTestsNumericAllExecs, OverlappingSlices)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({20, 20});

  // Create overlapping slices
  auto a_slice1 = slice(a, {0, 0}, {15, 15});
  auto a_slice2 = slice(a, {5, 5}, {20, 20});
  
  // This should throw because it's an overlapping slice operation
  EXPECT_THROW({
    (a_slice1 = a_slice2).run(exec);
    exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}


// Test that cloned tensors do NOT alias
TYPED_TEST(OperatorTestsNumericAllExecs, CloneNoAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({10});
  auto b = make_tensor<TestType>({10, 20});
  auto c = make_tensor<TestType>({10, 20});

  // Clone creates a lazy operator, not actual memory, so no aliasing
  auto a_cloned = clone<2>(a, {matxKeepDim, 20});
  
  // This should NOT throw because clone doesn't actually share memory
  EXPECT_NO_THROW({
    (b = a_cloned + c).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}


// Test that different tensors in expression do NOT alias
TYPED_TEST(OperatorTestsNumericAllExecs, ComplexExpressionNoAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({10, 10});
  auto b = make_tensor<TestType>({10, 10});
  auto c = make_tensor<TestType>({10, 10});
  auto d = make_tensor<TestType>({10, 10});

  // This should NOT throw - all different tensors
  EXPECT_NO_THROW({
    (d = a + b + c).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}


// Test nested transform operations with aliasing (FFT example)
TYPED_TEST(OperatorTestsComplexNonHalfTypesAllExecs, NestedTransformAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  }    

  ExecType exec{};

  auto a = make_tensor<TestType>({128});
  auto b = make_tensor<TestType>({128});

  // This should not throw because PreRun will allocate temporary memory for the FFTs
  EXPECT_NO_THROW({
    (a = ifft(fft(a) * fft(b))).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}

// Test nested transform operations WITHOUT aliasing
TYPED_TEST(OperatorTestsComplexNonHalfTypesAllExecs, NestedTransformNoAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  }    

  ExecType exec{};

  auto a = make_tensor<TestType>({128});
  auto b = make_tensor<TestType>({128});
  auto c = make_tensor<TestType>({128});

  // This should NOT throw because c, a, and b are all different
  // c = ifft(fft(a) * fft(b))
  EXPECT_NO_THROW({
    (c = ifft(fft(a) * fft(b))).run(exec);
    exec.sync();
  });

  MATX_EXIT_HANDLER();
}

// Test that shift() with aliasing DOES throw (permutation transform)
TYPED_TEST(OperatorTestsNumericAllExecs, ShiftAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({100});

  // This SHOULD throw because shift() is a permutation transform on RHS
  EXPECT_THROW({
    (a = shift<0>(a, 5)).run(exec);
    exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}



// Test that permute() with slice aliasing DOES throw
TYPED_TEST(OperatorTestsNumericAllExecs, PermuteSliceAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({20, 20});
  auto a_slice = slice(a, {0, 0}, {10, 10});

  // This SHOULD throw - permute() transforms how a_slice is accessed
  EXPECT_THROW({
    (a_slice = permute(a_slice, {1, 0})).run(exec);
    exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

// Test that reverse() with aliasing DOES throw (permutation transform)
TYPED_TEST(OperatorTestsNumericAllExecs, ReverseAliasing)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  ExecType exec{};

  auto a = make_tensor<TestType>({100});

  // This SHOULD throw - reverse() is a permutation transform
  EXPECT_THROW({
    (a = reverse<0>(a)).run(exec);
    exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

// Test that matmul() with aliasing DOES throw (transform operation)
TYPED_TEST(OperatorTestsFloatAllExecs, MatmulAliasing)
{
  MATX_ENTER_HANDLER();
  

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  }

  ExecType exec{};

  auto a = make_tensor<TestType>({10, 10});
  auto b = make_tensor<TestType>({10, 10});

  // This SHOULD throw - a appears on LHS and in matmul on RHS
  EXPECT_THROW({
    (a = matmul(a, b)).run(exec);
    exec.sync();
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}


#endif