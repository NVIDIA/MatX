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
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
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
      for (index_t j = 0; j <= i; j++) {
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
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
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
      for (index_t j = 0; j <= i; j++) {
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
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), this->thresh);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

