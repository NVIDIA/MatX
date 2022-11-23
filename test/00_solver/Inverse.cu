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
protected:
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();

  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class InvSolverTestFloatTypes : public InvSolverTest<TensorType> {
};

TYPED_TEST_SUITE(InvSolverTestFloatTypes,
                 MatXFloatNonHalfTypes);

TYPED_TEST(InvSolverTestFloatTypes, Inv4x4)
{
  MATX_ENTER_HANDLER();

  auto A = make_tensor<TypeParam>({4, 4});
  auto Ainv = make_tensor<TypeParam>({4, 4});
  auto Ainv_ref = make_tensor<TypeParam>({4, 4});  

  this->pb->template InitAndRunTVGenerator<TypeParam>("00_solver", "inv", "run", {4, 1});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  inv(Ainv, A, 0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TypeParam>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), 0.001);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), 0.001);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), 0.001);
      }
    }
  }

  MATX_EXIT_HANDLER();
}        

TYPED_TEST(InvSolverTestFloatTypes, Inv4x4Batched)
{
  MATX_ENTER_HANDLER();

  auto A = make_tensor<TypeParam>({100, 4, 4});
  auto Ainv = make_tensor<TypeParam>({100, 4, 4});
  auto Ainv_ref = make_tensor<TypeParam>({100, 4, 4});  

  this->pb->template InitAndRunTVGenerator<TypeParam>("00_solver", "inv", "run", {4, 100});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  inv(Ainv, A, 0);
  cudaStreamSynchronize(0);

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j <= i; j++) {
        if constexpr (is_complex_v<TypeParam>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), 0.001);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), 0.001);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), 0.001);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}    

TYPED_TEST(InvSolverTestFloatTypes, Inv8x8)
{
  MATX_ENTER_HANDLER();

  auto A = make_tensor<TypeParam>({8, 8});
  auto Ainv = make_tensor<TypeParam>({8, 8});
  auto Ainv_ref = make_tensor<TypeParam>({8, 8});  

  this->pb->template InitAndRunTVGenerator<TypeParam>("00_solver", "inv", "run", {8, 1});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  inv(Ainv, A, 0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TypeParam>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), 0.001);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), 0.001);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), 0.001);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(InvSolverTestFloatTypes, Inv8x8Batched)
{
  MATX_ENTER_HANDLER();

  auto A = make_tensor<TypeParam>({100, 8, 8});
  auto Ainv = make_tensor<TypeParam>({100, 8, 8});
  auto Ainv_ref = make_tensor<TypeParam>({100, 8, 8});  

  this->pb->template InitAndRunTVGenerator<TypeParam>("00_solver", "inv", "run", {8, 100});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  inv(Ainv, A, 0);
  cudaStreamSynchronize(0);

  for (index_t b = 0; b < A.Size(0); b++) {
    for (index_t i = 0; i < A.Size(1); i++) {
      for (index_t j = 0; j <= i; j++) {
        if constexpr (is_complex_v<TypeParam>) {
          ASSERT_NEAR(Ainv_ref(b, i, j).real(), Ainv(b, i, j).real(), 0.001);
          ASSERT_NEAR(Ainv_ref(b, i, j).imag(), Ainv(b, i, j).imag(), 0.001);
        }
        else {
          ASSERT_NEAR(Ainv_ref(b, i, j), Ainv(b, i, j), 0.001);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}    

TYPED_TEST(InvSolverTestFloatTypes, Inv256x256)
{
  MATX_ENTER_HANDLER();

  //int dim_size = 8;
  auto A = make_tensor<TypeParam>({256, 256});
  auto Ainv = make_tensor<TypeParam>({256, 256});
  auto Ainv_ref = make_tensor<TypeParam>({256, 256});  

  this->pb->template InitAndRunTVGenerator<TypeParam>("00_solver", "inv", "run", {256, 1});
  this->pb->NumpyToTensorView(A, "A");
  this->pb->NumpyToTensorView(Ainv_ref, "A_inv");  

  inv(Ainv, A, 0);
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < A.Size(0); i++) {
    for (index_t j = 0; j <= i; j++) {
      if constexpr (is_complex_v<TypeParam>) {
        ASSERT_NEAR(Ainv_ref(i, j).real(), Ainv(i, j).real(), 0.001);
        ASSERT_NEAR(Ainv_ref(i, j).imag(), Ainv(i, j).imag(), 0.001);
      }
      else {
        ASSERT_NEAR(Ainv_ref(i, j), Ainv(i, j), 0.001);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

