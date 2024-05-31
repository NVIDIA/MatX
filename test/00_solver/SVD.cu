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
constexpr index_t m = 100;
constexpr index_t n = 50;

template <typename T> class SVDSolverTest : public ::testing::Test {
protected:
  using GTestType = std::tuple_element_t<0, T>;
  using GExecType = std::tuple_element_t<1, T>;     
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<GTestType>("00_solver", "svd", "run", {m, n});
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class SVDSolverTestNonHalfTypes : public SVDSolverTest<TensorType> {
};

TYPED_TEST_SUITE(SVDSolverTestNonHalfTypes,
  MatXFloatNonHalfTypesCUDAExec);

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBasic)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;      

  using scalar_type = typename inner_op_type_t<TestType>::type;
  tensor_t<TestType, 2> Av{{m, n}};
  tensor_t<TestType, 2> Atv{{n, m}};
  tensor_t<scalar_type, 1> Sv{{std::min(m, n)}};
  tensor_t<TestType, 2> Uv{{m, m}};
  tensor_t<TestType, 2> Vv{{n, n}};

  tensor_t<scalar_type, 2> Sav{{m, n}};
  tensor_t<TestType, 2> SSolav{{m, n}};
  tensor_t<TestType, 2> Uav{{m, m}};
  tensor_t<TestType, 2> Vav{{n, n}};

  this->pb->NumpyToTensorView(Av, "A");

  // Used only for validation
  auto tmpV = make_tensor<TestType>({m, n});

  // example-begin svd-test-1
  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  (Atv = transpose(Av)).run(this->exec);

  auto Atv2 = Atv.View({m, n});
  (mtie(Uv, Sv, Vv) = svd(Atv2)).run(this->exec);
  // example-end svd-test-1

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.
  // However, U and V are in column-major format, so we have to transpose them
  // back to verify the identity.
  (Uav = transpose(Uv)).run(this->exec);
  (Vav = transpose(Vv)).run(this->exec);

  // Zero out s
  (Sav = zeros<typename inner_op_type_t<TestType>::type>({m, n})).run(this->exec);
  this->exec.sync();

  // Construct S matrix since it's just a vector from cuSolver
  for (index_t i = 0; i < n; i++) {
    Sav(i, i) = Sv(i);
  }

  this->exec.sync();

  (SSolav = 0).run(this->exec);
  if constexpr (is_complex_v<TestType>) {
    (SSolav.RealView() = Sav).run(this->exec);
  }
  else {
    (SSolav = Sav).run(this->exec);
  }

  (tmpV = matmul(Uav, SSolav)).run(this->exec); // U * S
  (SSolav = matmul(tmpV, Vav)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      if constexpr (is_complex_v<TestType>) {
        ASSERT_NEAR(Av(i, j).real(), SSolav(i, j).real(), 0.001) << i << " " << j;
        ASSERT_NEAR(Av(i, j).imag(), SSolav(i, j).imag(), 0.001) << i << " " << j;
      }
      else {
        ASSERT_NEAR(Av(i, j), SSolav(i, j), 0.001) << i << " " << j;
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBasicBatched)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
  using ExecType = std::tuple_element_t<1, TypeParam>;      

  constexpr index_t batches = 10;

  using scalar_type = typename inner_op_type_t<TestType>::type;
  auto Av1 = make_tensor<TestType>({m, n});
  this->pb->NumpyToTensorView(Av1, "A");
  auto Av = make_tensor<TestType>({batches, m, n});
  auto Atv = make_tensor<TestType>({batches, n, m});
  (Av = Av1).run(this->exec);  

  auto Sv = make_tensor<scalar_type>({batches, std::min(m, n)});
  auto Uv = make_tensor<TestType>({batches, m, m});
  auto Vv = make_tensor<TestType>({batches, n, n});

  auto Sav = make_tensor<scalar_type>({batches, m, n});
  auto SSolav = make_tensor<TestType>({batches, m, n});
  auto Uav = make_tensor<TestType>({batches, m, m});
  auto Vav = make_tensor<TestType>({batches, n, n});

  // Used only for validation
  auto tmpV = make_tensor<TestType>({batches, m, n});

  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  (Atv = transpose_matrix(Av)).run(this->exec);

  auto Atv2 = Atv.View({batches, m, n});
  (mtie(Uv, Sv, Vv) = svd(Atv2)).run(this->exec);

  this->exec.sync();

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.
  // However, U and V are in column-major format, so we have to transpose them
  // back to verify the identity.
  (Uav = transpose_matrix(Uv)).run(this->exec);
  (Vav = transpose_matrix(Vv)).run(this->exec);

  // Zero out s
  (Sav = zeros<typename inner_op_type_t<TestType>::type>({batches, m, n})).run(this->exec);
  this->exec.sync();

  // Construct S matrix since it's just a vector from cuSolver
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < n; i++) {
      Sav(b, i, i) = Sv(b, i);
    }
  }

  this->exec.sync();

  (SSolav = 0).run(this->exec);
  if constexpr (is_complex_v<TestType>) {
    (SSolav.RealView() = Sav).run(this->exec);
  }
  else {
    (SSolav = Sav).run(this->exec);
  }

  (tmpV = matmul(Uav, SSolav)).run(this->exec); // U * S
  (SSolav = matmul(tmpV, Vav)).run(this->exec); // (U * S) * V'
  this->exec.sync();

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < Av.Size(0); i++) {
      for (index_t j = 0; j < Av.Size(1); j++) {
        if constexpr (is_complex_v<TestType>) {
          ASSERT_NEAR(Av(b, i, j).real(), SSolav(b, i, j).real(), 0.001) << i << " " << j;
          ASSERT_NEAR(Av(b, i, j).imag(), SSolav(b, i, j).imag(), 0.001) << i << " " << j;
        }
        else {
          ASSERT_NEAR(Av(b, i, j), SSolav(b, i, j), 0.001) << i << " " << j;
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

  std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  index_t mm = Ashape[RANK-2];
  index_t nn = Ashape[RANK-1];
  index_t r = std::min(nn,mm);

  auto Ushape = Ashape;
  Ushape[RANK-1] = r;

  auto VTshape = Ashape;
  VTshape[RANK-2] = r;

  std::array<index_t, RANK-1> Sshape;
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

  std::array<index_t, RANK> Dshape;
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

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDPI)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;
    
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

  std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  exec.sync();

  index_t mm = Ashape[RANK-2];
  index_t nn = Ashape[RANK-1];
  index_t r = std::min(nn,mm);

  auto Ushape = Ashape;
  Ushape[RANK-1] = r;

  auto VTshape = Ashape;
  VTshape[RANK-2] = r;

  std::array<index_t, RANK-1> Sshape;
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

  std::array<index_t, RANK> Dshape;
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

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBPI)
{
  MATX_ENTER_HANDLER();
  using TestType = std::tuple_element_t<0, TypeParam>;  

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
