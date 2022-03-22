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
#include "matx_matmul.h"
#include "matx_pybind.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

using namespace matx;

/* NOTE: CUTLASS tests are disabled for now. The compile times are too long at
 * the moment */
template <typename T> class MatMulTest : public ::testing::Test {
protected:
  void SetUp() override
  {
    CheckTestTensorCoreTypeSupport<T>();

    pb = std::make_unique<detail::MatXPybind>(); // Half precision needs a bit more
                                         // tolerance when compared to fp32
    if constexpr (is_complex_half_v<T> || is_matx_half_v<T>) {
      thresh = 0.5f;
    }
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
};

template <typename TensorType>
class MatMulTestFloatTypes : public MatMulTest<TensorType> {
};

TYPED_TEST_SUITE(MatMulTestFloatTypes, MatXFloatTypes);

TYPED_TEST(MatMulTestFloatTypes, SmallRect)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectATranspose)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  tensor_t<TypeParam, 2> a{{k, m}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run_a_transpose", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto at = a.PermuteMatrix();
  matmul(c, at, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectBTranspose)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{n, k}};
  tensor_t<TypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run_b_transpose", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto bt = b.PermuteMatrix();
  matmul(c, a, bt);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallRectUserPointer)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  TypeParam *ap, *bp, *cp;
  cudaMallocManaged(&ap, m*k*sizeof(TypeParam));
  cudaMallocManaged(&bp, k*n*sizeof(TypeParam));
  cudaMallocManaged(&cp, m*n*sizeof(TypeParam));

  auto a = make_tensor<TypeParam, 2, non_owning>(ap, {m, k});
  auto b = make_tensor<TypeParam, 2, non_owning>(bp, {k, n});
  auto c = make_tensor<TypeParam, 2, non_owning>(cp, {m, n});

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  cudaFree(ap);
  cudaFree(bp);
  cudaFree(cp);

  MATX_EXIT_HANDLER();
}


TYPED_TEST(MatMulTestFloatTypes, DISABLED_SmallRectTranspose)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};

  auto at = a.Permute({1,0});
  auto bt = b.Permute({1,0});
  auto ct = c.Permute({1,0});

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run_transpose", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(ct), decltype(bt), decltype(at), PROVIDER_TYPE_CUBLASLT>(ct, bt, at);

  MATX_TEST_ASSERT_COMPARE(this->pb, ct, "c", 0.01);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, SmallSquare)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 4;
  constexpr index_t n = 4;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  // matmul<TypeParam, TypeParam, TypeParam, 2, PROVIDER_TYPE_CUTLASS>(c, a,
  //                                                                    b);
  // MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRect)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 512;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  // matmul<TypeParam, TypeParam, TypeParam, 2, PROVIDER_TYPE_CUTLASS>(c, a,
  //                                                                    b);
  // MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}