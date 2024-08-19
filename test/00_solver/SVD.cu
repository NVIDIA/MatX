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

template <typename T> class SVDTest : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;     
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.001f;
};

template <typename T> class SVDSolverTest : public SVDTest<T> {
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
      this->exec = SelectThreadsHostExecutor{params};
    }

    this->pb = std::make_unique<detail::MatXPybind>();
  }
};

template <typename TensorType>
class SVDSolverTestNonHalfTypes : public SVDSolverTest<TensorType> {
};

template <typename TensorType>
class SVDPISolverTestNonHalfTypes : public SVDTest<TensorType> {
};

TYPED_TEST_SUITE(SVDSolverTestNonHalfTypes, MatXFloatNonHalfTypesAllExecs);
TYPED_TEST_SUITE(SVDPISolverTestNonHalfTypes, MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;      
  using value_type = typename inner_op_type_t<TestType>::type;
  constexpr index_t m = 100;
  constexpr index_t n = 50;

  tensor_t<TestType, 2> Av{{m, n}};
  tensor_t<value_type, 1> Sv{{std::min(m, n)}};
  tensor_t<TestType, 2> Uv{{m, m}};
  tensor_t<TestType, 2> VTv{{n, n}};

  tensor_t<value_type, 2> Dv{{m, n}};
  tensor_t<TestType, 2> UDVTv{{m, n}};

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "svd", "run", {m, n});
  this->pb->NumpyToTensorView(Av, "A");

  // example-begin svd-test-1
  (mtie(Uv, Sv, VTv) = svd(Av)).run(this->exec);
  // example-end svd-test-1

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.

  // Construct diagonal matrix D from the vector of singular values S
  (Dv = zeros<value_type>({m, n})).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < Sv.Size(0); i++) {
    Dv(i, i) = Sv(i);
  }

  (UDVTv = matmul(matmul(Uv, Dv), VTv)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Av(i, j).real(), UDVTv(i, j).real(), this->thresh) << i << " " << j;
        ASSERT_NEAR(Av(i, j).imag(), UDVTv(i, j).imag(), this->thresh) << i << " " << j;
      }
      else {
        ASSERT_NEAR(Av(i, j), UDVTv(i, j), this->thresh) << i << " " << j;
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDMLeqN)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;      
  using value_type = typename inner_op_type_t<TestType>::type;
  constexpr index_t m = 50;
  constexpr index_t n = 100;

  tensor_t<TestType, 2> Av{{m, n}};
  tensor_t<value_type, 1> Sv{{std::min(m, n)}};
  tensor_t<TestType, 2> Uv{{m, m}};
  tensor_t<TestType, 2> VTv{{n, n}};

  tensor_t<value_type, 2> Dv{{m, n}};
  tensor_t<TestType, 2> UDVTv{{m, n}};

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "svd", "run", {m, n});
  this->pb->NumpyToTensorView(Av, "A");

  (mtie(Uv, Sv, VTv) = svd(Av)).run(this->exec);

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.

  // Construct diagonal matrix D from the vector of singular values S
  (Dv = zeros<value_type>({m, n})).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < Sv.Size(0); i++) {
    Dv(i, i) = Sv(i);
  }

  (UDVTv = matmul(matmul(Uv, Dv), VTv)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Av(i, j).real(), UDVTv(i, j).real(), this->thresh) << i << " " << j;
        ASSERT_NEAR(Av(i, j).imag(), UDVTv(i, j).imag(), this->thresh) << i << " " << j;
      }
      else {
        ASSERT_NEAR(Av(i, j), UDVTv(i, j), this->thresh) << i << " " << j;
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDHostAlgoQR)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;      
  using value_type = typename inner_op_type_t<TestType>::type;
  constexpr index_t m = 100;
  constexpr index_t n = 50;

  tensor_t<TestType, 2> Av{{m, n}};
  tensor_t<value_type, 1> Sv{{std::min(m, n)}};
  tensor_t<TestType, 2> Uv{{m, m}};
  tensor_t<TestType, 2> VTv{{n, n}};

  tensor_t<value_type, 2> Dv{{m, n}};
  tensor_t<TestType, 2> UDVTv{{m, n}};

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "svd", "run", {m, n});
  this->pb->NumpyToTensorView(Av, "A");

  (mtie(Uv, Sv, VTv) = svd(Av, SVDMode::ALL, SVDHostAlgo::QR)).run(this->exec);

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.

  // Construct diagonal matrix D from the vector of singular values S
  (Dv = zeros<value_type>({m, n})).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < Sv.Size(0); i++) {
    Dv(i, i) = Sv(i);
  }

  (UDVTv = matmul(matmul(Uv, Dv), VTv)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Av(i, j).real(), UDVTv(i, j).real(), this->thresh) << i << " " << j;
        ASSERT_NEAR(Av(i, j).imag(), UDVTv(i, j).imag(), this->thresh) << i << " " << j;
      }
      else {
        ASSERT_NEAR(Av(i, j), UDVTv(i, j), this->thresh) << i << " " << j;
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;      
  using value_type = typename inner_op_type_t<TestType>::type;

  constexpr index_t batches = 10;
  constexpr index_t m = 100;
  constexpr index_t n = 50;

  auto Av = make_tensor<TestType>({batches, m, n});
  auto Sv = make_tensor<value_type>({batches, std::min(m, n)});
  auto Uv = make_tensor<TestType>({batches, m, m});
  auto VTv = make_tensor<TestType>({batches, n, n});

  auto Dv = make_tensor<value_type>({batches, m, n});
  auto UDVTv = make_tensor<TestType>({batches, m, n});

  this->pb->template InitAndRunTVGenerator<TestType>("00_solver", "svd", "run", {batches, m, n});
  this->pb->NumpyToTensorView(Av, "A");

  (mtie(Uv, Sv, VTv) = svd(Av)).run(this->exec);

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.

  // Construct batched diagonal matrix D from the vector of singular values S
  (Dv = zeros<value_type>({m, n})).run(this->exec);
  this->exec.sync();

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < Sv.Size(1); i++) {
      Dv(b, i, i) = Sv(b, i);
    }
  }

  (UDVTv = matmul(matmul(Uv, Dv), VTv)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < Av.Size(1); i++) {
      for (index_t j = 0; j < Av.Size(2); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Av(b, i, j).real(), UDVTv(b, i, j).real(), this->thresh) << i << " " << j;
          ASSERT_NEAR(Av(b, i, j).imag(), UDVTv(b, i, j).imag(), this->thresh) << i << " " << j;
        }
        else {
          ASSERT_NEAR(Av(b, i, j), UDVTv(b, i, j), this->thresh) << i << " " << j;
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

template <typename TypeParam, int RANK, typename Executor>
void svdpi_test( const index_t (&AshapeA)[RANK], Executor exec) {
  using AType = TypeParam;
  using SType = typename inner_op_type_t<AType>::type;

  cuda::std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  index_t mm = Ashape[RANK-2];
  index_t nn = Ashape[RANK-1];
  index_t r = std::min(nn,mm);

  auto Ushape = Ashape;
  Ushape[RANK-1] = r;

  auto VTshape = Ashape;
  VTshape[RANK-2] = r;

  cuda::std::array<index_t, RANK-1> Sshape;
  for(index_t i = 0; i < RANK-2; i++) {
    Sshape[i] = Ashape[i];
  }
  Sshape[RANK-2] = r;

  // example-begin svdpi-test-1
  auto A = make_tensor<AType>(Ashape);
  auto U = make_tensor<AType>(Ushape);
  auto VT = make_tensor<AType>(VTshape);
  auto S = make_tensor<SType>(Sshape);

  int iterations = 100;

  (A = random<AType>(AshapeA, NORMAL)).run(exec);
  auto x0 = random<SType>(std::move(Sshape), NORMAL);

  (U = 0).run(exec);
  (S = 0).run(exec);
  (VT = 0).run(exec);

  (mtie(U, S, VT) = svdpi(A, x0, iterations, r)).run(exec);
  // example-end svdpi-test-1

  auto Rshape = Ushape;
  Rshape[RANK-1] = r;
  Rshape[RANK-2] = r;

  auto UD = make_tensor<AType>(Ushape);
  auto UDVT = make_tensor<AType>(Ashape);
  auto UTU = make_tensor<AType>(Rshape);
  auto VTV = make_tensor<AType>(Rshape);

  auto UTUd = make_tensor<SType>(Rshape);
  auto VTVd = make_tensor<SType>(Rshape);
  auto Ad = make_tensor<SType>(Ashape);

  (UTU = matmul(conj(transpose_matrix(U)) , U)).run(exec);
  (VTV = matmul(VT, conj(transpose_matrix(VT)))).run(exec); 

  cuda::std::array<index_t, RANK> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[RANK-2] = mm;

  // cloning D across matrix
  auto D = clone<RANK>(S, Dshape);
  // scale U by eigen values (equivalent to matmul of the diagonal matrix)
  (UD = U * D).run(exec);

  (UDVT = matmul(UD, VT)).run(exec);

  auto e = eye<SType>({r,r});
  auto eShape = Rshape;
  eShape[RANK-1] = matxKeepDim;
  eShape[RANK-2] = matxKeepDim;

  auto mdiffU = make_tensor<SType>({});
  auto mdiffV = make_tensor<SType>({});
  auto mdiffA = make_tensor<SType>({});

  auto I = clone<RANK>(e, eShape);

  (UTUd = abs(UTU - I)).run(exec);
  (VTVd = abs(VTV - I)).run(exec);
  (Ad = abs(A - UDVT)).run(exec);

  (mdiffU = max(UTUd)).run(exec);
  (mdiffV = max(VTVd)).run(exec);
  (mdiffA = max(Ad)).run(exec);

  exec.sync();

#if 0
  printf("A\n"); print(A);
  printf("U\n"); print(U);
  printf("VT\n"); print(VT);
  printf("S\n"); print(S);

  printf("UTU\n"); print(UTU);
  printf("VTV\n"); print(VTV);
  printf("A\n"); print(A);
  printf("UDVT\n"); print(UDVT);

  printf("mdiffU: %f\n", (float)mdiffU());
  printf("mdiffV: %f\n", (float)mdiffV());
  printf("mdiffA: %f\n", (float)mdiffA());
#endif

  ASSERT_NEAR( mdiffU(), SType(0), .1);
  ASSERT_NEAR( mdiffV(), SType(0), .1);
  ASSERT_NEAR( mdiffA(), SType(0), .00001);
}

TYPED_TEST(SVDPISolverTestNonHalfTypes, SVDPI)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
    
  svdpi_test<TestType>({4,4}, this->exec);
  svdpi_test<TestType>({4,16}, this->exec);
  svdpi_test<TestType>({16,4}, this->exec);

  svdpi_test<TestType>({25,4,4}, this->exec);
  svdpi_test<TestType>({25,4,16}, this->exec);
  svdpi_test<TestType>({25,16,4}, this->exec);

  svdpi_test<TestType>({5,5,4,4}, this->exec);
  svdpi_test<TestType>({5,5,4,16}, this->exec);
  svdpi_test<TestType>({5,5,16,4}, this->exec);
  
  MATX_EXIT_HANDLER();
}

template <typename TypeParam, int RANK, typename Executor>
void svdbpi_test( const index_t (&AshapeA)[RANK], Executor exec) {
  using AType = TypeParam;
  using SType = typename inner_op_type_t<AType>::type;

  cuda::std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  exec.sync();

  index_t mm = Ashape[RANK-2];
  index_t nn = Ashape[RANK-1];
  index_t r = std::min(nn,mm);

  auto Ushape = Ashape;
  Ushape[RANK-1] = r;

  auto VTshape = Ashape;
  VTshape[RANK-2] = r;

  cuda::std::array<index_t, RANK-1> Sshape;
  for(index_t i = 0; i < RANK-2; i++) {
    Sshape[i] = Ashape[i];
  }
  Sshape[RANK-2] = r;

  // example-begin svdbpi-test-1
  auto A = make_tensor<AType>(Ashape);
  auto U = make_tensor<AType>(Ushape);
  auto VT = make_tensor<AType>(VTshape);
  auto S = make_tensor<SType>(Sshape);

  int iterations = 100;

  (A = random<AType>(std::move(Ashape), NORMAL)).run(exec);


  (U = 0).run(exec);
  (S = 0).run(exec);
  (VT = 0).run(exec);

  (mtie(U, S, VT) = svdbpi(A, iterations)).run(exec);
  // example-end svdbpi-test-1

  auto Rshape = Ushape;
  Rshape[RANK-1] = r;
  Rshape[RANK-2] = r;

  auto UD = make_tensor<AType>(Ushape);
  auto UDVT = make_tensor<AType>(Ashape);
  auto UTU = make_tensor<AType>(Rshape);
  auto VTV = make_tensor<AType>(Rshape);

  auto UTUd = make_tensor<SType>(Rshape);
  auto VTVd = make_tensor<SType>(Rshape);
  auto Ad = make_tensor<SType>(Ashape);

  (UTU = matmul(conj(transpose_matrix(U)), U)).run(exec);
  (VTV = matmul(VT, conj(transpose_matrix(VT)))).run(exec); 

  cuda::std::array<index_t, RANK> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[RANK-2] = mm;

  // cloning D across matrix
  auto D = clone<RANK>(S, Dshape);
  // scale U by eigen values (equivalent to matmul of the diagonal matrix)
  (UD = U * D).run(exec);

  (UDVT = matmul(UD, VT)).run(exec);

  auto e = eye<SType>({r,r});
  auto eShape = Rshape;
  eShape[RANK-1] = matxKeepDim;
  eShape[RANK-2] = matxKeepDim;

  auto mdiffU = make_tensor<SType>({});
  auto mdiffV = make_tensor<SType>({});
  auto mdiffA = make_tensor<SType>({});

  auto I = clone<RANK>(e, eShape);

  (UTUd = abs(UTU - I)).run(exec);
  (VTVd = abs(VTV - I)).run(exec);
  (Ad = abs(A - UDVT)).run(exec);

  (mdiffU = max(UTUd)).run(exec);
  (mdiffV = max(VTVd)).run(exec);
  (mdiffA = max(Ad)).run(exec);

  exec.sync();

#if 0
  printf("A\n"); print(A);
  printf("U\n"); print(U);
  printf("VT\n"); print(VT);
  printf("S\n"); print(S);

  printf("UTU\n"); print(UTU);
  printf("VTV\n"); print(VTV);
  printf("A\n"); print(A);
  printf("UDVT\n"); print(UDVT);

  printf("mdiffU: %f\n", (float)mdiffU());
  printf("mdiffV: %f\n", (float)mdiffV());
  printf("mdiffA: %f\n", (float)mdiffA());
#endif

  ASSERT_NEAR( mdiffU(), SType(0), .1);
  ASSERT_NEAR( mdiffV(), SType(0), .1);
  ASSERT_NEAR( mdiffA(), SType(0), .00001);
  
  exec.sync();
}

TYPED_TEST(SVDPISolverTestNonHalfTypes, SVDBPI)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;  

  svdbpi_test<TestType>({4,4}, this->exec);
  svdbpi_test<TestType>({4,16}, this->exec);
  svdbpi_test<TestType>({16,4}, this->exec);

  svdbpi_test<TestType>({25,4,4}, this->exec);
  svdbpi_test<TestType>({25,4,16}, this->exec);
  svdbpi_test<TestType>({25,16,4}, this->exec);

  svdbpi_test<TestType>({5,5,4,4}, this->exec);
  svdbpi_test<TestType>({5,5,4,16}, this->exec);
  svdbpi_test<TestType>({5,5,16,4}, this->exec);

  MATX_EXIT_HANDLER();
}
