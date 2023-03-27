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
  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<T>("00_solver", "svd", "run", {m, n});
  }

  void TearDown() { pb.reset(); }

  std::unique_ptr<detail::MatXPybind> pb;
};

template <typename TensorType>
class SVDSolverTestNonHalfTypes : public SVDSolverTest<TensorType> {
};

TYPED_TEST_SUITE(SVDSolverTestNonHalfTypes,
                 MatXFloatNonHalfTypes);

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBasic)
{
  MATX_ENTER_HANDLER();

  using scalar_type = typename inner_op_type_t<TypeParam>::type;
  tensor_t<TypeParam, 2> Av{{m, n}};
  tensor_t<TypeParam, 2> Atv{{n, m}};
  tensor_t<scalar_type, 1> Sv{{std::min(m, n)}};
  tensor_t<TypeParam, 2> Uv{{m, m}};
  tensor_t<TypeParam, 2> Vv{{n, n}};

  tensor_t<scalar_type, 2> Sav{{m, n}};
  tensor_t<TypeParam, 2> SSolav{{m, n}};
  tensor_t<TypeParam, 2> Uav{{m, m}};
  tensor_t<TypeParam, 2> Vav{{n, n}};

  this->pb->NumpyToTensorView(Av, "A");

  // Used only for validation
  tensor_t<TypeParam, 2> tmpV{{m, n}};

  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  transpose(Atv, Av, 0);

  auto Atv2 = Atv.View({m, n});
  svd(Uv, Sv, Vv, Atv2);

  cudaStreamSynchronize(0);

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.
  // However, U and V are in column-major format, so we have to transpose them
  // back to verify the identity.
  transpose(Uav, Uv, 0);
  transpose(Vav, Vv, 0);

  // Zero out s
  (Sav = zeros<typename inner_op_type_t<TypeParam>::type>({m, n})).run();
  cudaStreamSynchronize(0);

  // Construct S matrix since it's just a vector from cuSolver
  for (index_t i = 0; i < n; i++) {
    Sav(i, i) = Sv(i);
  }

  cudaStreamSynchronize(0);

  (SSolav = 0).run();
  if constexpr (is_complex_v<TypeParam>) {
    (SSolav.RealView() = Sav).run();
  }
  else {
    (SSolav = Sav).run();
  }

  matmul(tmpV, Uav, SSolav); // U * S
  matmul(SSolav, tmpV, Vav); // (U * S) * V'
  cudaStreamSynchronize(0);

  for (index_t i = 0; i < Av.Size(0); i++) {
    for (index_t j = 0; j < Av.Size(1); j++) {
      if constexpr (is_complex_v<TypeParam>) {
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

  constexpr index_t batches = 10;

  using scalar_type = typename inner_op_type_t<TypeParam>::type;
  auto Av1 = make_tensor<TypeParam>({m, n});
  this->pb->NumpyToTensorView(Av1, "A");
  auto Av = make_tensor<TypeParam>({batches, m, n});
  auto Atv = make_tensor<TypeParam>({batches, n, m});
  (Av = Av1).run();  

  auto Sv = make_tensor<scalar_type>({batches, std::min(m, n)});
  auto Uv = make_tensor<TypeParam>({batches, m, m});
  auto Vv = make_tensor<TypeParam>({batches, n, n});

  auto Sav = make_tensor<scalar_type>({batches, m, n});
  auto SSolav = make_tensor<TypeParam>({batches, m, n});
  auto Uav = make_tensor<TypeParam>({batches, m, m});
  auto Vav = make_tensor<TypeParam>({batches, n, n});

  // Used only for validation
  auto tmpV = make_tensor<TypeParam>({batches, m, n});

  // cuSolver only supports col-major solving today, so we need to transpose,
  // solve, then transpose again to compare to Python
  transpose(Atv, Av, 0);

  auto Atv2 = Atv.View({batches, m, n});
  svd(Uv, Sv, Vv, Atv2);

  cudaStreamSynchronize(0);

  // Since SVD produces a solution that's not necessarily unique, we cannot
  // compare against Python output. Instead, we just make sure that A = U*S*V'.
  // However, U and V are in column-major format, so we have to transpose them
  // back to verify the identity.
  transpose(Uav, Uv, 0);
  transpose(Vav, Vv, 0);

  // Zero out s
  (Sav = zeros<typename inner_op_type_t<TypeParam>::type>({batches, m, n})).run();
  cudaStreamSynchronize(0);

  // Construct S matrix since it's just a vector from cuSolver
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < n; i++) {
      Sav(b, i, i) = Sv(b, i);
    }
  }

  cudaStreamSynchronize(0);

  (SSolav = 0).run();
  if constexpr (is_complex_v<TypeParam>) {
    (SSolav.RealView() = Sav).run();
  }
  else {
    (SSolav = Sav).run();
  }

  matmul(tmpV, Uav, SSolav); // U * S
  matmul(SSolav, tmpV, Vav); // (U * S) * V'
  cudaStreamSynchronize(0);

  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < Av.Size(0); i++) {
      for (index_t j = 0; j < Av.Size(1); j++) {
        if constexpr (is_complex_v<TypeParam>) {
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

template <typename TypeParam, int RANK>
void svdpi_test( const index_t (&AshapeA)[RANK]) {
  using AType = TypeParam;
  using SType = typename inner_op_type_t<AType>::type;

  std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  cudaStream_t stream = 0;

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

  auto A = make_tensor<AType>(Ashape);
  auto U = make_tensor<AType>(Ushape);
  auto VT = make_tensor<AType>(VTshape);
  auto S = make_tensor<SType>(Sshape);

  int iterations = 100;

  randomGenerator_t<AType> gen(A.TotalSize(),0);
  auto random = gen.GetTensorView(Ashape, NORMAL);
  (A = random).run(stream);

  auto x0 = gen.GetTensorView(Sshape, NORMAL);

  A.PrefetchDevice(stream);
  U.PrefetchDevice(stream);
  VT.PrefetchDevice(stream);
  S.PrefetchDevice(stream);

  (U = 0).run(stream);
  (S = 0).run(stream);
  (VT = 0).run(stream);

  svdpi(U, S, VT, A, x0, iterations, stream, r);

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

  matmul(UTU, conj(transpose(U)) , U, stream);
  matmul(VTV, VT, conj(transpose(VT)), stream); 

  std::array<index_t, RANK> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[RANK-2] = mm;

  // cloning D across matrix
  auto D = clone<RANK>(S, Dshape);
  // scale U by eigen values (equivalent to matmul of the diagonal matrix)
  (UD = U * D).run(stream);

  matmul(UDVT, UD, VT, stream);

  auto e = eye<SType>({r,r});
  auto eShape = Rshape;
  eShape[RANK-1] = matxKeepDim;
  eShape[RANK-2] = matxKeepDim;

  auto mdiffU = make_tensor<SType>();
  auto mdiffV = make_tensor<SType>();
  auto mdiffA = make_tensor<SType>();

  auto I = clone<RANK>(e, eShape);

  (UTUd = abs(UTU - I)).run(stream);
  (VTVd = abs(VTV - I)).run(stream);
  (Ad = abs(A - UDVT)).run(stream);

  rmax(mdiffU, UTUd, stream);
  rmax(mdiffV, VTVd, stream);
  rmax(mdiffA, Ad, stream);

  cudaDeviceSynchronize();

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
  
  svdpi_test<TypeParam>({4,4});
  svdpi_test<TypeParam>({4,16});
  svdpi_test<TypeParam>({16,4});

  svdpi_test<TypeParam>({25,4,4});
  svdpi_test<TypeParam>({25,4,16});
  svdpi_test<TypeParam>({25,16,4});

  svdpi_test<TypeParam>({5,5,4,4});
  svdpi_test<TypeParam>({5,5,4,16});
  svdpi_test<TypeParam>({5,5,16,4});
  
  MATX_EXIT_HANDLER();
}

template <typename TypeParam, int RANK>
void svdbpi_test( const index_t (&AshapeA)[RANK]) {
  using AType = TypeParam;
  using SType = typename inner_op_type_t<AType>::type;

  std::array<index_t, RANK> Ashape = detail::to_array(AshapeA);

  cudaStream_t stream = 0;
  cudaDeviceSynchronize();

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

  auto A = make_tensor<AType>(Ashape);
  auto U = make_tensor<AType>(Ushape);
  auto VT = make_tensor<AType>(VTshape);
  auto S = make_tensor<SType>(Sshape);
  
  A.PrefetchDevice(stream);
  U.PrefetchDevice(stream);
  VT.PrefetchDevice(stream);
  S.PrefetchDevice(stream);

  int iterations = 100;

  randomGenerator_t<AType> gen(A.TotalSize(),0);
  auto random = gen.GetTensorView(Ashape, NORMAL);
  (A = random).run(stream);


  (U = 0).run(stream);
  (S = 0).run(stream);
  (VT = 0).run(stream);

  svdbpi(U, S, VT, A, iterations, stream);

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

  matmul(UTU, conj(transpose(U)) , U, stream);
  matmul(VTV, VT, conj(transpose(VT)), stream); 

  std::array<index_t, RANK> Dshape;
  Dshape.fill(matxKeepDim);
  Dshape[RANK-2] = mm;

  // cloning D across matrix
  auto D = clone<RANK>(S, Dshape);
  // scale U by eigen values (equivalent to matmul of the diagonal matrix)
  (UD = U * D).run(stream);

  matmul(UDVT, UD, VT, stream);

  auto e = eye<SType>({r,r});
  auto eShape = Rshape;
  eShape[RANK-1] = matxKeepDim;
  eShape[RANK-2] = matxKeepDim;

  auto mdiffU = make_tensor<SType>();
  auto mdiffV = make_tensor<SType>();
  auto mdiffA = make_tensor<SType>();

  auto I = clone<RANK>(e, eShape);

  (UTUd = abs(UTU - I)).run(stream);
  (VTVd = abs(VTV - I)).run(stream);
  (Ad = abs(A - UDVT)).run(stream);

  rmax(mdiffU, UTUd, stream);
  rmax(mdiffV, VTVd, stream);
  rmax(mdiffA, Ad, stream);

  cudaDeviceSynchronize();

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
  
  cudaDeviceSynchronize();
}

TYPED_TEST(SVDSolverTestNonHalfTypes, SVDBPI)
{
  MATX_ENTER_HANDLER();

  svdbpi_test<TypeParam>({4,4});
  svdbpi_test<TypeParam>({4,16});
  svdbpi_test<TypeParam>({16,4});

  svdbpi_test<TypeParam>({25,4,4});
  svdbpi_test<TypeParam>({25,4,16});
  svdbpi_test<TypeParam>({25,16,4});

  svdbpi_test<TypeParam>({5,5,4,4});
  svdbpi_test<TypeParam>({5,5,4,16});
  svdbpi_test<TypeParam>({5,5,16,4});

  MATX_EXIT_HANDLER();
}
