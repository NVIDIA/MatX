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
constexpr int dim_size = 100;

template <typename T> class EigenSolverTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;      
  void SetUp() override
  {
    if constexpr (!detail::CheckSolverSupport<GExecType>()) {
      GTEST_SKIP();
    }

    // Use an arbitrary number of threads for the select threads host exec.
    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() override { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
  GExecType exec{};
  float thresh = 0.001f;
};

template <typename TensorType>
class EigenSolverTestFloatTypes : public EigenSolverTest<TensorType> {
};

TYPED_TEST_SUITE(EigenSolverTestFloatTypes,
                 MatXFloatNonHalfTypesAllExecs);

TYPED_TEST(EigenSolverTestFloatTypes, EigenBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  auto Bv = make_tensor<TestType>({dim_size, dim_size});
  auto Evv = make_tensor<TestType>({dim_size, dim_size});
  auto Wov = make_tensor<value_type>({dim_size});

  auto Gv = make_tensor<TestType>({dim_size, 1});
  auto Lvv = make_tensor<TestType>({dim_size, 1});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "eig", "run", {dim_size});
  this->pb->NumpyToTensorView(Bv, "B");

  // example-begin eig-test-1
  // Note that eigenvalue/vector solutions are not necessarily ordered in the same way other libraries
  // may order them. When comparing against other libraries it's best to check A*v = lambda * v
  (mtie(Evv, Wov) = eig(Bv)).run(this->exec);
  // example-end eig-test-1

  // Now we need to go through all the eigenvectors and eigenvalues and make
  // sure the results match the equation A*v = lambda*v, where v are the
  // eigenvectors corresponding to the eigenvalue lambda.
  for (index_t i = 0; i < dim_size; i++) {
    auto v = slice<2>(Evv, {0, i}, {matxEnd, i + 1});

    // Compute lambda*v
    (Lvv = Wov(i) * v).run(this->exec);

    // Compute A*v
    (Gv = matmul(Bv, v)).run(this->exec);
    this->exec.sync();

    // Compare
    for (index_t j = 0; j < dim_size; j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Gv(j, 0).real(), Lvv(j, 0).real(), this->thresh);
        ASSERT_NEAR(Gv(j, 0).imag(), Lvv(j, 0).imag(), this->thresh);
      }
      else {
        ASSERT_NEAR(Gv(j, 0), Lvv(j, 0), this->thresh);
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(EigenSolverTestFloatTypes, EigenBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr int batches = 10;
  auto Bv = make_tensor<TestType>({batches, dim_size, dim_size});
  auto Evv = make_tensor<TestType>({batches, dim_size, dim_size});
  auto Wov = make_tensor<value_type>({batches, dim_size});

  auto Gv = make_tensor<TestType>({dim_size, 1});
  auto Lvv = make_tensor<TestType>({dim_size, 1});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "eig", "run", {batches, dim_size});
  this->pb->NumpyToTensorView(Bv, "B");

  // Note that eigenvalue/vector solutions are not necessarily ordered in the same way other libraries
  // may order them. When comparing against other libraries it's best to check A*v = lambda * v
  (mtie(Evv, Wov) = eig(Bv)).run(this->exec);

  // Now we need to go through all the eigenvectors and eigenvalues and make
  // sure the results match the equation A*v = lambda*v, where v are the
  // eigenvectors corresponding to the eigenvalue lambda.
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < dim_size; i++) {
      // ith col vector from bth batch
      auto v = slice<2>(Evv, {b, 0, i}, {matxDropDim, matxEnd, i + 1});

      // Compute lambda*v
      (Lvv = Wov(b, i) * v).run(this->exec);

      // Compute A*v
      auto Bv_b = slice<2>(Bv, {b, 0, 0}, {matxDropDim, matxEnd, matxEnd});
      (Gv = matmul(Bv_b, v)).run(this->exec);
      this->exec.sync();

      // Compare
      for (index_t j = 0; j < dim_size; j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Gv(j, 0).real(), Lvv(j, 0).real(), this->thresh);
          ASSERT_NEAR(Gv(j, 0).imag(), Lvv(j, 0).imag(), this->thresh);
        }
        else {
          ASSERT_NEAR(Gv(j, 0), Lvv(j, 0), this->thresh);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}