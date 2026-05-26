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
#include <algorithm>
#include <cmath>

using namespace matx;

namespace {
template <typename T>
__MATX_INLINE__ T InvJITValue(double real, double imag = 0.0)
{
  if constexpr (cuda::std::is_same_v<T, cuda::std::complex<float>>) {
    return T{static_cast<float>(real), static_cast<float>(imag)};
  }
  else if constexpr (cuda::std::is_same_v<T, cuda::std::complex<double>>) {
    return T{real, imag};
  }
  else {
    return static_cast<T>(real);
  }
}

template <typename T>
double InvJITAbsDiff(const T &a, const T &b)
{
  if constexpr (is_complex_v<T>) {
    return std::max(std::abs(static_cast<double>(a.real() - b.real())),
                    std::abs(static_cast<double>(a.imag() - b.imag())));
  }
  else {
    return std::abs(static_cast<double>(a - b));
  }
}

template <typename T, typename TensorType>
void FillInvJITMatrix2x2(TensorType &A)
{
  A(0, 0) = InvJITValue<T>(4.0, 0.25);
  A(0, 1) = InvJITValue<T>(7.0, -0.5);
  A(1, 0) = InvJITValue<T>(2.0, 0.125);
  A(1, 1) = InvJITValue<T>(6.0, 0.375);
}

template <typename T, typename TensorType>
void FillInvJITSPDMatrix3x3(TensorType &A, index_t batch = -1)
{
  auto set = [&](index_t i, index_t j, double value) {
    if constexpr (TensorType::Rank() == 3) {
      A(batch, i, j) = InvJITValue<T>(value);
    }
    else {
      A(i, j) = InvJITValue<T>(value);
    }
  };

  set(0, 0, 6.0);
  set(0, 1, 1.0);
  set(0, 2, 0.5);
  set(1, 0, 1.0);
  set(1, 1, 5.0);
  set(1, 2, 1.5);
  set(2, 0, 0.5);
  set(2, 1, 1.5);
  set(2, 2, 4.0);
}

template <typename T, typename TensorType>
void FillInvJITTallFullRankMatrix(TensorType &H)
{
  H(0, 0) = InvJITValue<T>(2.0);
  H(0, 1) = InvJITValue<T>(0.1);
  H(0, 2) = InvJITValue<T>(0.2);
  H(1, 0) = InvJITValue<T>(0.2);
  H(1, 1) = InvJITValue<T>(1.8);
  H(1, 2) = InvJITValue<T>(0.3);
  H(2, 0) = InvJITValue<T>(0.1);
  H(2, 1) = InvJITValue<T>(0.2);
  H(2, 2) = InvJITValue<T>(1.7);
  H(3, 0) = InvJITValue<T>(0.3);
  H(3, 1) = InvJITValue<T>(0.4);
  H(3, 2) = InvJITValue<T>(0.5);
}
}

template <typename T> class InvSolverTest : public ::testing::Test {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();

  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.001f;
};

template <typename TensorType>
class InvSolverTestFloatTypes : public InvSolverTest<TensorType> {
};

TYPED_TEST_SUITE(InvSolverTestFloatTypes,
  MatXFloatNonHalfTypesCUDAExec);

#if defined(MATX_EN_MATHDX) && defined(MATX_EN_JIT)
template <typename TensorType>
class InvSolverJITTestFloatTypes : public ::testing::Test {
};

TYPED_TEST_SUITE(InvSolverJITTestFloatTypes,
  MatXFloatNonHalfTypesCUDAExec);

TEST(InvSolverJITRegression, SelectsPassThroughBlockDimFromIntersection)
{
  EXPECT_EQ(detail::SelectJITPassThroughBlockDim(cuda::std::array<int, 2>{64, 64}), 64);
  EXPECT_EQ(detail::SelectJITPassThroughBlockDim(cuda::std::array<int, 2>{32, 1024}), 256);
  EXPECT_EQ(detail::SelectJITPassThroughBlockDim(cuda::std::array<int, 2>{512, 1024}), 512);
  EXPECT_THROW({ detail::SelectJITPassThroughBlockDim(cuda::std::array<int, 2>{128, 64}); },
               matx::detail::matxException);
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxRuntimeQueries)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({2, 2});
  auto op = inv(A);

  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_GE(detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op),
            static_cast<int>(2 * 4 * sizeof(TestType) + 3 * sizeof(int)));

  const auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
  EXPECT_EQ(block_dim[0], 32);
  EXPECT_EQ(block_dim[1], 1024);
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxMatchesCudaPath)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A_jit = make_tensor<TestType>({2, 2});
  auto A_cuda = make_tensor<TestType>({2, 2});
  auto O_jit = make_tensor<TestType>({2, 2});
  auto O_cuda = make_tensor<TestType>({2, 2});

  FillInvJITMatrix2x2<TestType>(A_jit);
  (A_cuda = A_jit).run(cudaExecutor{});

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (O_jit = inv(A_jit)).run(jit_exec);
  (O_cuda = inv(A_cuda)).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t i = 0; i < 2; i++) {
    for (index_t j = 0; j < 2; j++) {
      ASSERT_NEAR(InvJITAbsDiff(O_jit(i, j), O_cuda(i, j)), 0.0, 1e-4);
    }
  }
}

TEST(InvSolverJITRegression, CuSolverDxMultipleSizesInOneJITExpression)
{
  MATX_ENTER_HANDLER();
  using TestType = float;

  auto A2 = make_tensor<TestType>({2, 2});
  auto A3 = make_tensor<TestType>({3, 3});
  auto Out = make_tensor<TestType>({2, 2});
  auto Ref = make_tensor<TestType>({2, 2});
  auto mdiff = make_tensor<TestType>({});

  A2(0, 0) = TestType{4};
  A2(0, 1) = TestType{1};
  A2(1, 0) = TestType{2};
  A2(1, 1) = TestType{3};
  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      A3(i, j) = i == j ? TestType(i + 5) : TestType{};
    }
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (Out = inv(A2) + slice<2>(inv(A3), {0, 0}, {2, 2})).run(jit_exec);
  (Ref = inv(A2) + slice<2>(inv(A3), {0, 0}, {2, 2})).run(cuda_exec);
  (mdiff = max(abs(Out - Ref))).run(cuda_exec);
  cuda_exec.sync();

  ASSERT_NEAR(mdiff(), TestType(0), TestType(0.001));

  MATX_EXIT_HANDLER();
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxBatchedMatchesCudaPath)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A_jit = make_tensor<TestType>({2, 2, 2});
  auto A_cuda = make_tensor<TestType>({2, 2, 2});
  auto O_jit = make_tensor<TestType>({2, 2, 2});
  auto O_cuda = make_tensor<TestType>({2, 2, 2});

  A_jit(0, 0, 0) = InvJITValue<TestType>(4.0, 0.25);
  A_jit(0, 0, 1) = InvJITValue<TestType>(7.0, -0.5);
  A_jit(0, 1, 0) = InvJITValue<TestType>(2.0, 0.125);
  A_jit(0, 1, 1) = InvJITValue<TestType>(6.0, 0.375);
  A_jit(1, 0, 0) = InvJITValue<TestType>(5.0, -0.125);
  A_jit(1, 0, 1) = InvJITValue<TestType>(3.0, 0.25);
  A_jit(1, 1, 0) = InvJITValue<TestType>(1.0, -0.375);
  A_jit(1, 1, 1) = InvJITValue<TestType>(4.0, 0.5);
  (A_cuda = A_jit).run(cudaExecutor{});

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (O_jit = inv(A_jit)).run(jit_exec);
  (O_cuda = inv(A_cuda)).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t b = 0; b < 2; b++) {
    for (index_t i = 0; i < 2; i++) {
      for (index_t j = 0; j < 2; j++) {
        ASSERT_NEAR(InvJITAbsDiff(O_jit(b, i, j), O_cuda(b, i, j)), 0.0, 1e-4);
      }
    }
  }
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxPosvInverseMatchesCudaPath)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A_jit = make_tensor<TestType>({3, 3});
  auto A_cuda = make_tensor<TestType>({3, 3});
  auto O_jit = make_tensor<TestType>({3, 3});
  auto O_cuda = make_tensor<TestType>({3, 3});

  FillInvJITSPDMatrix3x3<TestType>(A_jit);
  (A_cuda = A_jit).run(cudaExecutor{});

  auto op = inv<MAT_INVERSE_ALGO_POSV>(A_jit);
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_GE(detail::get_operator_capability<detail::OperatorCapability::DYN_SHM_SIZE>(op),
            static_cast<int>(2 * 9 * sizeof(TestType) + sizeof(int)));

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (O_jit = op).run(jit_exec);
  (O_cuda = inv(A_cuda)).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_NEAR(InvJITAbsDiff(O_jit(i, j), O_cuda(i, j)), 0.0, 1e-4);
    }
  }
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxPosvBatchedMatchesCudaPath)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A_jit = make_tensor<TestType>({2, 3, 3});
  auto A_cuda = make_tensor<TestType>({2, 3, 3});
  auto O_jit = make_tensor<TestType>({2, 3, 3});
  auto O_cuda = make_tensor<TestType>({2, 3, 3});

  FillInvJITSPDMatrix3x3<TestType>(A_jit, 0);
  FillInvJITSPDMatrix3x3<TestType>(A_jit, 1);
  A_jit(1, 0, 0) = InvJITValue<TestType>(7.0);
  A_jit(1, 1, 1) = InvJITValue<TestType>(6.0);
  A_jit(1, 2, 2) = InvJITValue<TestType>(5.0);
  (A_cuda = A_jit).run(cudaExecutor{});

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (O_jit = inv<MAT_INVERSE_ALGO_POSV>(A_jit)).run(jit_exec);
  (O_cuda = inv(A_cuda)).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t b = 0; b < 2; b++) {
    for (index_t i = 0; i < 3; i++) {
      for (index_t j = 0; j < 3; j++) {
        ASSERT_NEAR(InvJITAbsDiff(O_jit(b, i, j), O_cuda(b, i, j)), 0.0, 1e-4);
      }
    }
  }
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxRejectsUnsupportedShape)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({2, 3});
  auto O = make_tensor<TestType>({2, 3});
  auto op = inv(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (O = op).run(exec); }, matx::detail::matxException);
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxRejectsUnsupportedRank)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({1, 1, 1, 2, 2});
  auto O = make_tensor<TestType>({1, 1, 1, 2, 2});
  auto op = inv(A);

  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));

  CUDAJITExecutor exec{};
  EXPECT_THROW({ (O = op).run(exec); }, matx::detail::matxException);
}

TYPED_TEST(InvSolverJITTestFloatTypes, CuSolverDxFusedMatmulPosvInverseMatchesCudaPath)
{
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto H_jit = make_tensor<TestType>({4, 3});
  auto H_cuda = make_tensor<TestType>({4, 3});
  auto O_jit = make_tensor<TestType>({3, 3});
  auto O_cuda = make_tensor<TestType>({3, 3});

  FillInvJITTallFullRankMatrix<TestType>(H_jit);
  (H_cuda = H_jit).run(cudaExecutor{});

  auto op = inv<MAT_INVERSE_ALGO_POSV>(matmul(permute(H_jit, {1, 0}), H_jit));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));

  const auto block_dim = detail::get_operator_capability<detail::OperatorCapability::BLOCK_DIM>(op);
  EXPECT_NE(block_dim[0], detail::capability_attributes<detail::OperatorCapability::BLOCK_DIM>::invalid);
  EXPECT_NE(block_dim[1], detail::capability_attributes<detail::OperatorCapability::BLOCK_DIM>::invalid);

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (O_jit = op).run(jit_exec);
  (O_cuda = inv(matmul(permute(H_cuda, {1, 0}), H_cuda))).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t i = 0; i < 3; i++) {
    for (index_t j = 0; j < 3; j++) {
      ASSERT_NEAR(InvJITAbsDiff(O_jit(i, j), O_cuda(i, j)), 0.0, 1e-3);
    }
  }
}
#endif

TYPED_TEST(InvSolverTestFloatTypes, Inv4x4)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({4, 4});
  auto Ainv = make_tensor<TestType>({4, 4});
  auto Ainv_ref = make_tensor<TestType>({4, 4});  

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {4});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  // example-begin inv-test-1
  // Perform an inverse on matrix "A" and store the output in "Ainv"
  (Ainv = inv(A)).run(this->exec);
  // example-end inv-test-1  
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
      }
    }
  }

  // Repeat the test using in-place transforms
  (A = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), A(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), A(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), A(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}        

TYPED_TEST(InvSolverTestFloatTypes, Inv4x4Batched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({100, 4, 4});
  auto Ainv = make_tensor<TestType>({100, 4, 4});
  auto Ainv_ref = make_tensor<TestType>({100, 4, 4});  

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {100, 4});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), this->thresh);
        }
      }
    }
  }

  // Repeat the test using in-place transforms
  (A = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), A(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), A(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), A(b, i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}    

TYPED_TEST(InvSolverTestFloatTypes, Inv8x8)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({8, 8});
  auto Ainv = make_tensor<TestType>({8, 8});
  auto Ainv_ref = make_tensor<TestType>({8, 8});  

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {8});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
      }
    }
  }

  // Repeat the test using in-place transforms
  (A = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), A(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), A(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), A(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(InvSolverTestFloatTypes, Inv8x8Batched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto A = make_tensor<TestType>({100, 8, 8});
  auto Ainv = make_tensor<TestType>({100, 8, 8});
  auto Ainv_ref = make_tensor<TestType>({100, 8, 8});  

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {100, 8});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), this->thresh);
        }
      }
    }
  }

  // Repeat the test using in-place transforms
  (A = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), A(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), A(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), A(b, i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}    

TYPED_TEST(InvSolverTestFloatTypes, Inv256x256)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  //int dim_size = 8;
  auto A = make_tensor<TestType>({256, 256});
  auto Ainv = make_tensor<TestType>({256, 256});
  auto Ainv_ref = make_tensor<TestType>({256, 256});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {256});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
      }
    }
  }

  // Repeat the test using in-place transforms
  (A = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j < A.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), A(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), A(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), A(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(InvSolverTestFloatTypes, InvOperatorInputSmall)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  const int num_batches = 3;
  const int N = 16;
  auto Afull = make_tensor<TestType>({num_batches, 1, 1, N, N});
  auto A = lcollapse<3>(Afull);
  auto Ainv = make_tensor<TestType>({num_batches, N, N});
  auto Ainv_ref = make_tensor<TestType>({num_batches, N, N});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {num_batches, N});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(InvSolverTestFloatTypes, InvOperatorInputLarge)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  const int num_batches = 8;
  const int N = 200;
  auto Afull = make_tensor<TestType>({2, 2, 2, N, N});
  auto A = lcollapse<3>(Afull);
  auto Ainv = make_tensor<TestType>({num_batches, N, N});
  auto Ainv_ref = make_tensor<TestType>({num_batches, N, N});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "inv", "run", {num_batches, N});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");

  (Ainv = inv(A)).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j < A.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), this->thresh);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}
