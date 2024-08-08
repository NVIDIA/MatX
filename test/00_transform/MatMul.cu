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

using namespace matx;

/* NOTE: CUTLASS tests are disabled for now. The compile times are too long at
 * the moment */
template <typename T> class MatMulTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
  void SetUp() override
  {
    CheckTestTensorCoreTypeSupport<GTestType>();

    if constexpr (!detail::CheckMatMulSupport<GExecType, GTestType>()) {
      GTEST_SKIP();
    }

    // Use an arbitrary number of threads for the select threads host exec.
    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>(); // Half precision needs a bit more
                                         // tolerance when compared to fp32
    if constexpr (is_complex_half_v<GTestType> || is_matx_half_v<GTestType>) {
      thresh = 0.5f;
    }
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
};

template <typename TensorType>
class MatMulTestFloatTypes : public MatMulTest<TensorType> {
};
template <typename TensorType>
class MatMulTestFloatNonHalfTypes : public MatMulTest<TensorType> {
};

template <typename TensorType>
class MatMulTestFloatNonComplexTypes : public MatMulTest<TensorType> {
};

TYPED_TEST_SUITE(MatMulTestFloatTypes, MatXTypesFloatAllExecs);
TYPED_TEST_SUITE(MatMulTestFloatNonHalfTypes, MatXFloatNonHalfTypesAllExecs);
TYPED_TEST_SUITE(MatMulTestFloatNonComplexTypes, MatXTypesFloatNonComplexAllExecs);

template <typename T>
struct float_to_complex
{
  using type = cuda::std::complex<T>;
};

template <>
struct float_to_complex<matxFp16>
{
  using type = matxFp16Complex;
};

template <>
struct float_to_complex<matxBf16>
{
  using type = matxBf16Complex;
};

template <typename T>
using float_to_complex_t = typename float_to_complex<T>::type;

TYPED_TEST(MatMulTestFloatTypes, SmallRect)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    // example-begin matmul-test-1
    // Perform the GEMM C = A*B
    (c = matmul(a, b)).run(this->exec);
    // example-end matmul-test-1
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectATranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    tensor_t<TestType, 2> a{{k, m}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run_a_transpose", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    // example-begin matmul-test-2
    // Perform the GEMM C = A^T * B
    auto at = a.PermuteMatrix();
    (c = matmul(at, b)).run(this->exec);
    // example-end matmul-test-2
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectBTranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{n, k}};
    tensor_t<TestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run_b_transpose", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    // example-begin matmul-test-3
    // Perform the GEMM C = A * B^T
    auto bt = b.PermuteMatrix();
    (c = matmul(a, bt)).run(this->exec);
    // example-end matmul-test-3
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);\
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes, SmallRectCTranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{n, m}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    auto ct = transpose_matrix(c);

    (ct = matmul(a, b)).run(this->exec);
    MATX_TEST_ASSERT_COMPARE(this->pb, ct, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectUserPointer)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    TestType *ap, *bp, *cp;
    cudaMallocManaged(&ap, m*k*sizeof(TestType));
    cudaMallocManaged(&bp, k*n*sizeof(TestType));
    cudaMallocManaged(&cp, m*n*sizeof(TestType));

    auto a = make_tensor<TestType, 2>(ap, {m, k},false);
    auto b = make_tensor<TestType, 2>(bp, {k, n},false);
    auto c = make_tensor<TestType, 2>(cp, {m, n},false);

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (c = matmul(a, b)).run(this->exec);
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    cudaFree(ap);
    cudaFree(bp);
    cudaFree(cp);
  }
  MATX_EXIT_HANDLER();
}


TYPED_TEST(MatMulTestFloatTypes, DISABLED_SmallRectTranspose)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};

    auto at = a.Permute({1,0});
    auto bt = b.Permute({1,0});
    auto ct = c.Permute({1,0});

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run_transpose", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (ct = matmul(bt, at)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, ct, "c", 0.01);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallSquare)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 4;
    constexpr index_t n = 4;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (c = matmul(a, b)).run(this->exec);
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    // matmul<TestType, TestType, TestType, 2, PROVIDER_TYPE_CUTLASS>(c, a,
    //                                                                    b);
    // MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRect)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 128;
    constexpr index_t k = 256;
    constexpr index_t n = 512;
    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (c = matmul(a, b)).run(this->exec);
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    // matmul<TestType, TestType, TestType, 2, PROVIDER_TYPE_CUTLASS>(c, a,
    //                                                                    b);
    // MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
  // example-begin matmul-test-4
  constexpr index_t batches = 5;
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 512;
  
  tensor_t<TestType, 3> a{{batches, m, k}};
  tensor_t<TestType, 3> b{{batches, k, n}};
  tensor_t<TestType, 3> c{{batches, m, n}};  

  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "matmul_operators", "run", {batches, m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  // Perform a batched gemm with "batches" GEMMs
  (c = matmul(a, b)).run(this->exec);

  // example-end matmul-test-4
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched0StrideA)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t batches = 2;
    constexpr index_t m = 3;
    constexpr index_t k = 4;
    constexpr index_t n = 5;
    
    tensor_t<TestType, 2> a0{{m, k}};
    tensor_t<TestType, 3> b{{batches, k, n}};
    tensor_t<TestType, 2> b0{{k, n}};
    tensor_t<TestType, 3> c{{batches, m, n}};  
    tensor_t<TestType, 2> c0{{m, n}};  

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a0, "a");
    this->pb->NumpyToTensorView(b0, "b");
    this->pb->NumpyToTensorView(c0, "c");
    (b = b0).run(this->exec);

    // Perform a batched gemm with "batches" GEMMs
    (c = matmul(a0, b)).run(this->exec);

    this->exec.sync();

    for (int i = 0; i < c.Size(0); i++) {
      for (int j = 0; j < c.Size(1); j++) {
        for (int p = 0; p < c.Size(2); p++) {
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(c0(j, p), c(i, j, p), this->thresh));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched0StrideB)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t batches = 2;
    constexpr index_t m = 3;
    constexpr index_t k = 4;
    constexpr index_t n = 5;
    
    tensor_t<TestType, 3> a{{batches, m, k}};
    tensor_t<TestType, 2> a0{{m, k}};
    tensor_t<TestType, 2> b0{{k, n}};
    tensor_t<TestType, 3> c{{batches, m, n}};  
    tensor_t<TestType, 2> c0{{m, n}};  

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a0, "a");
    this->pb->NumpyToTensorView(b0, "b");
    this->pb->NumpyToTensorView(c0, "c");
    (a = a0).run(this->exec);

    // Perform a batched gemm with "batches" GEMMs
    (c = matmul(a, b0)).run(this->exec);

    this->exec.sync();

    for (int i = 0; i < c.Size(0); i++) {
      for (int j = 0; j < c.Size(1); j++) {
        for (int p = 0; p < c.Size(2); p++) {
          EXPECT_TRUE(MatXUtils::MatXTypeCompare(c0(j, p), c(i, j, p), this->thresh));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}


TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched3DStridedBatch)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    // example-begin matmul-test-5
    constexpr index_t batches = 16;
    constexpr index_t m = 128;
    constexpr index_t k = 256;
    constexpr index_t n = 512;
    
    tensor_t<TestType, 3> a{{batches, m, k}};
    tensor_t<TestType, 3> b{{batches, k, n}};
    tensor_t<TestType, 3> c{{batches, m, n}};  

    auto as = a.Slice({0, 0, 0}, {matxEnd, matxEnd, matxEnd}, {2, 1, 1});
    auto bs = b.Slice({0, 0, 0}, {matxEnd, matxEnd, matxEnd}, {2, 1, 1});
    tensor_t<TestType, 3> cs{{batches/2, m, n}};


    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {batches, m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    // Perform a strided and batched GEMM where "as" and "bs" have a stride of 2 in their inner-most dimension
    (cs = matmul(as, bs)).run(this->exec);
    // example-end matmul-test-5

    MATX_TEST_ASSERT_COMPARE(this->pb, cs, "cs", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonComplexTypes, MixedTypes)
{
  // a -> complex, b -> real, c -> complex
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 4;
    constexpr index_t k = 8;
    constexpr index_t n = 16;

    using ComplexTestType = float_to_complex_t<TestType>;

    tensor_t<ComplexTestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<ComplexTestType, 2> c{{m, n}};

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run_mixed", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (c = matmul(a, b)).run(this->exec);
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched4D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    // constexpr index_t batches = 5;
    // constexpr index_t m = 128;
    // constexpr index_t k = 256;
    // constexpr index_t n = 512;
    
    auto a = make_tensor<TestType>({5, 5, 128, 256});
    auto b = make_tensor<TestType>({5, 5, 256, 512});
    auto c = make_tensor<TestType>({5, 5, 128, 512});  

    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {5, 5, 128, 256, 512});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    (c = matmul(a, b)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes,  MatMulAxis)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 16;
    constexpr index_t k = 32;
    constexpr index_t n = 64;
    constexpr index_t b = 8;
      
    tensor_t<TestType, 3> a3{{b, m, k}};
    tensor_t<TestType, 3> b3{{b, k, n}};
    tensor_t<TestType, 3> c3{{b, m, n}};
      
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "matmul_operators", "run", {b, m, k, n});

    this->pb->NumpyToTensorView(a3, "a");
    this->pb->NumpyToTensorView(b3, "b");
    
    { // identity permute
      const int axis[2] = {1, 2};
      cuda::std::array<int, 3> perm({0, 1, 2});

      auto ai = make_tensor<TestType>({b, m, k});
      auto bi = make_tensor<TestType>({b, k, n});
      auto ci = make_tensor<TestType>({b, m, n});

      auto ap = permute(ai, perm);
      auto bp = permute(bi, perm);
      auto cp = permute(ci, perm);

      (ap = a3).run(this->exec);
      (bp = b3).run(this->exec);

      (ci = matmul(ai, bi, axis)).run(this->exec);
      
      (c3 = cp).run(this->exec);

      this->exec.sync();

      MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
    }

    { // transposing inner dims
      // example-begin matmul-test-6  
      const int axis[2] = {2, 1};
      cuda::std::array<int, 3> perm({0, 2, 1});

      auto ai = make_tensor<TestType>({b, k, m});
      auto bi = make_tensor<TestType>({b, n, k});
      auto ci = make_tensor<TestType>({b, n, m});

      auto ap = permute(ai, perm);
      auto bp = permute(bi, perm);
      auto cp = permute(ci, perm);

      // copy data into permuted inputs
      (ap = a3).run(this->exec);
      (bp = b3).run(this->exec);

      // Perform a GEMM with the last two dimensions permuted
      (ci = matmul(ai, bi, axis)).run(this->exec);
      // example-end matmul-test-6    
      
      // copy result from permuted output
      (c3 = cp).run(this->exec);

      this->exec.sync();

      MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
    }
    
    { // first and last
      const int axis[2] = {0 ,2};
      cuda::std::array<int, 3> perm({1, 0, 2});

      tensor_t<TestType, 3> ai{{m, b, k}};
      tensor_t<TestType, 3> bi{{k, b, n}};
      tensor_t<TestType, 3> ci{{m, b, n}};

      auto ap = permute(ai, perm);
      auto bp = permute(bi, perm);
      auto cp = permute(ci, perm);

      // copy data into permuted inputs
      (ap = a3).run(this->exec);
      (bp = b3).run(this->exec);

      (ci = matmul(ai, bi, axis)).run(this->exec);
      
      // copy result from permuted output
      (c3 = cp).run(this->exec);

      this->exec.sync();

      MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
    }
  
    {  // affine not supported
      const int axis[2] = {0, 1};
      cuda::std::array<int, 3> perm({2, 0, 1});

      tensor_t<TestType, 3> ai{{m, k, b}};
      tensor_t<TestType, 3> bi{{k, n, b}};
      tensor_t<TestType, 3> ci{{m, n, b}};

      auto ap = permute(ai, perm);
      auto bp = permute(bi, perm);
      auto cp = permute(ci, perm);

      // copy data into permuted inputs
      (ap = a3).run(this->exec);
      (bp = b3).run(this->exec);

      (ci = matmul(ai, bi, axis)).run(this->exec);
      
      // copy result from permuted output
      (c3 = cp).run(this->exec);

      this->exec.sync();

      MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes,  MatMulOp)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 16;
    constexpr index_t k = 32;
    constexpr index_t n = 64;
    constexpr index_t b = 8;
      
    tensor_t<TestType, 3> a3{{b, m, k}};
    tensor_t<TestType, 3> b3{{b, k, n}};
    tensor_t<TestType, 3> c3{{b, m, n}};
      
    this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "matmul_operators", "run", {b, m, k, n});

    this->pb->NumpyToTensorView(a3, "a");
    this->pb->NumpyToTensorView(b3, "b");
    
    { // simple identity remaps

      auto rb = range<0>({b},0, 1);

      auto ar = remap<0>(a3, rb);
      auto br = remap<0>(b3, rb);
      auto cr = remap<0>(c3, rb);

      (cr = matmul(ar, br)).run(this->exec);
      
      MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes,  MatMulBroadcast)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t n = 16;
    constexpr index_t b = 8;
    constexpr index_t x = 3;
    constexpr index_t y = 4;

    tensor_t<TestType, 2> eye2{{n, n}};
    tensor_t<TestType, 5> a5{{x, y, b, n, n}};
    tensor_t<TestType, 5> c5{{x, y, b, n, n}};

    const TestType two { 2.0 };
    const TestType three { 3.0 };

    (eye2 = two*eye<TestType>({n,n})).run(this->exec);
    (a5 = three).run(this->exec);

    (c5 = TestType(0)).run(this->exec);
    // Broadcast eye2, scaling each entry in a5 by 2
    (c5 = matmul(eye2, a5)).run(this->exec);

    this->exec.sync();

    for (index_t i0 = 0; i0 < x; i0++)
      for (index_t i1 = 0; i1 < y; i1++)
        for (index_t i2 = 0; i2 < b; i2++)
          for (index_t i3 = 0; i3 < n; i3++)
            for (index_t i4 = 0; i4 < n; i4++) {
              if constexpr (is_complex_v<TestType>) {
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4).real(), 2.0*a5(i0,i1,i2,i3,i4).real(), this->thresh);
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4).imag(), 2.0*a5(i0,i1,i2,i3,i4).imag(), this->thresh);
              } else {
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4), two*a5(i0,i1,i2,i3,i4), this->thresh);
              }
            }

    (c5 = TestType(0)).run(this->exec);
    // Broadcast eye2, scaling each entry in a5 by 2
    (c5 = matmul(a5, eye2)).run(this->exec);

    this->exec.sync();

    for (index_t i0 = 0; i0 < x; i0++)
      for (index_t i1 = 0; i1 < y; i1++)
        for (index_t i2 = 0; i2 < b; i2++)
          for (index_t i3 = 0; i3 < n; i3++)
            for (index_t i4 = 0; i4 < n; i4++) {
              if constexpr (is_complex_v<TestType>) {
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4).real(), 2.0*a5(i0,i1,i2,i3,i4).real(), this->thresh);
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4).imag(), 2.0*a5(i0,i1,i2,i3,i4).imag(), this->thresh);
              } else {
                ASSERT_NEAR(c5(i0,i1,i2,i3,i4), two*a5(i0,i1,i2,i3,i4), this->thresh);
              }
            }

  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumMatVec)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 128;
    constexpr index_t k = 256;
    constexpr index_t n = 1;

    tensor_t<TestType, 2> a{{m, k}};
    tensor_t<TestType, 2> b{{k, n}};
    tensor_t<TestType, 2> c{{m, n}};
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    auto cs = slice<1>(c, {0,0}, {matxEnd, matxDropDim});
    auto bs = slice<1>(b, {0,0}, {matxEnd, matxDropDim});
    // example-begin matvec-test-1
    // "a" is a matrix and "bs" is a vector
    (cs = matvec(a, bs)).run(this->exec);
    // example-end matvec-test-1

    // Test the rank/size of the matvec operator
    auto a_times_bs = matvec(a, bs);
    ASSERT_EQ(a_times_bs.Rank(), 1);
    ASSERT_EQ(a_times_bs.Size(0), m);
    ASSERT_EQ(cs.Size(0), m);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    // Test also with rank-1 tensors rather than just slices
    tensor_t<TestType, 1> bv{{k}};
    tensor_t<TestType, 1> cv{{m}};
    (bv = bs).run(this->exec);
    (cv = cs).run(this->exec);
    (cv = matvec(a, bv)).run(this->exec);;

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    // Test with batching
    constexpr index_t batch1 = 5;
    constexpr index_t batch2 = 9;
    auto a_batch = clone<4>(a, {batch1, batch2, matxKeepDim, matxKeepDim});
    auto b_batch = clone<3>(bs, {batch1, batch2, matxKeepDim});
    auto batched_matvec = matvec(a_batch, b_batch);
    ASSERT_EQ(batched_matvec.Rank(), 3);
    ASSERT_EQ(batched_matvec.Size(0), batch1);
    ASSERT_EQ(batched_matvec.Size(1), batch2);
    ASSERT_EQ(batched_matvec.Size(2), m);
    auto result = make_tensor<TestType>(batched_matvec.Shape());
    (result = batched_matvec).run(this->exec);
    for (index_t i = 0; i < batch1; i++) {
      for (index_t j = 0; j < batch2; j++) {
        auto rs = slice<1>(result, {i,j,0}, {matxDropDim,matxDropDim,matxEnd});
        auto rsc = clone<2>(rs, {matxKeepDim,1});
        MATX_TEST_ASSERT_COMPARE(this->pb, rsc, "c", this->thresh);
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumMatVecBatch)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    constexpr index_t m = 128;
    constexpr index_t k = 256;
    constexpr index_t n = 1;
    constexpr index_t blocks = 8;

    tensor_t<TestType, 3> a{{blocks, m, k}};
    tensor_t<TestType, 3> b{{blocks, k, n}};
    tensor_t<TestType, 3> c{{blocks, m, n}};
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {blocks, m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    auto cs = slice<2>(c, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
    auto bs = slice<2>(b, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
    (cs = matvec(a, bs)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    tensor_t<TestType, 2> bv{{blocks, k}};
    tensor_t<TestType, 2> cv{{blocks, m}};
    (bv = bs).run(this->exec);
    (cv = cs).run(this->exec);
    (cv = matvec(a, bv)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MatVecRowVector)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    // Test that the second-to-last dimension of A can be 1 (i.e. A can be a row
    // vector). In the case of matvec, this means that A*b is effectively a dot product.
    constexpr index_t m = 1;
    constexpr index_t k = 256;
    constexpr index_t n = 1;
    constexpr index_t blocks = 8;

    tensor_t<TestType, 3> a{{blocks, m, k}};
    tensor_t<TestType, 3> b{{blocks, k, n}};
    tensor_t<TestType, 3> c{{blocks, m, n}};
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "matmul_operators", "run", {blocks, m, k, n});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    auto cs = slice<2>(c, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
    auto bs = slice<2>(b, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
    (cs = matvec(a, bs)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    tensor_t<TestType, 2> bv{{blocks, k}};
    tensor_t<TestType, 2> cv{{blocks, m}};
    (bv = bs).run(this->exec);
    (cv = cs).run(this->exec);
    (cv = matvec(a, bv)).run(this->exec);

    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  }
  MATX_EXIT_HANDLER();
}



TYPED_TEST(MatMulTestFloatTypes, OuterProduct)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckMatMulSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    [[maybe_unused]] constexpr index_t an = 10;
    [[maybe_unused]] constexpr index_t bn = 4;
    [[maybe_unused]] constexpr index_t batches = 5;

    auto a = make_tensor<TestType>({an});
    auto b = make_tensor<TestType>({bn});
    auto c = make_tensor<TestType>({an, bn});
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "outer_operators", "run", {batches, an, bn});

    this->pb->NumpyToTensorView(a, "a");
    this->pb->NumpyToTensorView(b, "b");

    // example-begin outer-test-1
    (c = outer(a, b)).run(this->exec);
    // example-end outer-test-1

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

    auto ba = make_tensor<TestType>({batches, an});
    auto bb = make_tensor<TestType>({batches, bn});
    this->pb->NumpyToTensorView(ba, "ba");
    this->pb->NumpyToTensorView(bb, "bb");

    auto bc = make_tensor<TestType>({batches, an, bn});  
    (bc = outer(ba, bb)).run(this->exec);

    this->exec.sync();
    MATX_TEST_ASSERT_COMPARE(this->pb, bc, "bc", this->thresh);
  }
  MATX_EXIT_HANDLER();
}
