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
template <typename TensorType>
class MatMulTestFloatNonHalfTypes : public MatMulTest<TensorType> {
};

template <typename TensorType>
class MatMulTestFloatNonComplexTypes : public MatMulTest<TensorType> {
};

TYPED_TEST_SUITE(MatMulTestFloatTypes, MatXFloatTypes);
TYPED_TEST_SUITE(MatMulTestFloatNonHalfTypes, MatXFloatNonHalfTypes);
TYPED_TEST_SUITE(MatMulTestFloatNonComplexTypes, MatXFloatNonComplexTypes);

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

TYPED_TEST(MatMulTestFloatNonHalfTypes, SmallRectCTranspose)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;
  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{n, m}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto ct = transpose(c);

  matmul(ct, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, ct, "c", this->thresh);

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

  auto a = make_tensor<TypeParam, 2>(ap, {m, k},false);
  auto b = make_tensor<TypeParam, 2>(bp, {k, n},false);
  auto c = make_tensor<TypeParam, 2>(cp, {m, n},false);

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

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched)
{
  MATX_ENTER_HANDLER();
  constexpr index_t batches = 5;
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 512;
  
  tensor_t<TypeParam, 3> a{{batches, m, k}};
  tensor_t<TypeParam, 3> b{{batches, k, n}};
  tensor_t<TypeParam, 3> c{{batches, m, n}};  

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {batches, m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched3DStridedBatch)
{
  MATX_ENTER_HANDLER();
  constexpr index_t batches = 16;
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 512;
  
  tensor_t<TypeParam, 3> a{{batches, m, k}};
  tensor_t<TypeParam, 3> b{{batches, k, n}};
  tensor_t<TypeParam, 3> c{{batches, m, n}};  

  auto as = a.Slice({0, 0, 0}, {matxEnd, matxEnd, matxEnd}, {2, 1, 1});
  auto bs = b.Slice({0, 0, 0}, {matxEnd, matxEnd, matxEnd}, {2, 1, 1});
  tensor_t<TypeParam, 3> cs{{batches/2, m, n}};


  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {batches, m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul(cs, as, bs);

  MATX_TEST_ASSERT_COMPARE(this->pb, cs, "cs", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonComplexTypes, MixedTypes)
{
  // a -> complex, b -> real, c -> complex
  MATX_ENTER_HANDLER();

  constexpr index_t m = 4;
  constexpr index_t k = 8;
  constexpr index_t n = 16;

  using ComplexTypeParam = float_to_complex_t<TypeParam>;

  tensor_t<ComplexTypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<ComplexTypeParam, 2> c{{m, n}};

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run_mixed", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul(c, a, b);
  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumRectBatched4D)
{
  MATX_ENTER_HANDLER();
  // constexpr index_t batches = 5;
  // constexpr index_t m = 128;
  // constexpr index_t k = 256;
  // constexpr index_t n = 512;
  
  auto a = make_tensor<TypeParam>({5, 5, 128, 256});
  auto b = make_tensor<TypeParam>({5, 5, 256, 512});
  auto c = make_tensor<TypeParam>({5, 5, 128, 512});  

  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {5, 5, 128, 256, 512});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  matmul<decltype(c), decltype(a), decltype(b), PROVIDER_TYPE_CUBLASLT>(c, a, b);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes,  MatMulAxis)
{
  MATX_ENTER_HANDLER();
   
  constexpr index_t m = 16;
  constexpr index_t k = 32;
  constexpr index_t n = 64;
  constexpr index_t b = 8;
    
  tensor_t<TypeParam, 3> a3{{b, m, k}};
  tensor_t<TypeParam, 3> b3{{b, k, n}};
  tensor_t<TypeParam, 3> c3{{b, m, n}};
    
  this->pb->template InitAndRunTVGenerator<TypeParam>(
    "00_transforms", "matmul_operators", "run", {b, m, k, n});

  this->pb->NumpyToTensorView(a3, "a");
  this->pb->NumpyToTensorView(b3, "b");
  
  { // identity permute
    const int axis[2] = {1, 2};
    std::array<int, 3> perm({0, 1, 2});

    tensor_t<TypeParam, 3> ai{{b, m, k}};
    tensor_t<TypeParam, 3> bi{{b, k, n}};
    tensor_t<TypeParam, 3> ci{{b, m, n}};

    auto ap = permute(ai, perm);
    auto bp = permute(bi, perm);
    auto cp = permute(ci, perm);

    (ap = a3).run();
    (bp = b3).run();

    matmul(ci, ai, bi, axis);
    
    (c3 = cp).run();

    cudaStreamSynchronize(0);

    MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
  }

  { // transposing inner dims
    const int axis[2] = {2, 1};
    std::array<int, 3> perm({0, 2, 1});

    tensor_t<TypeParam, 3> ai{{b, k, m}};
    tensor_t<TypeParam, 3> bi{{b, n, k}};
    tensor_t<TypeParam, 3> ci{{b, n, m}};

    auto ap = permute(ai, perm);
    auto bp = permute(bi, perm);
    auto cp = permute(ci, perm);

    // copy data into permuted inputs
    (ap = a3).run();
    (bp = b3).run();

    matmul(ci, ai, bi, axis);
    
    // copy result from permuted output
    (c3 = cp).run();

    cudaStreamSynchronize(0);

    MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
  }
  
  { // first and last
    const int axis[2] = {0 ,2};
    std::array<int, 3> perm({1, 0, 2});

    tensor_t<TypeParam, 3> ai{{m, b, k}};
    tensor_t<TypeParam, 3> bi{{k, b, n}};
    tensor_t<TypeParam, 3> ci{{m, b, n}};

    auto ap = permute(ai, perm);
    auto bp = permute(bi, perm);
    auto cp = permute(ci, perm);

    // copy data into permuted inputs
    (ap = a3).run();
    (bp = b3).run();

    matmul(ci, ai, bi, axis);
    
    // copy result from permuted output
    (c3 = cp).run();

    cudaStreamSynchronize(0);

    MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
  }
 
  {  // affine not supported
    const int axis[2] = {0, 1};
    std::array<int, 3> perm({2, 0, 1});

    tensor_t<TypeParam, 3> ai{{m, k, b}};
    tensor_t<TypeParam, 3> bi{{k, n, b}};
    tensor_t<TypeParam, 3> ci{{m, n, b}};

    auto ap = permute(ai, perm);
    auto bp = permute(bi, perm);
    auto cp = permute(ci, perm);

    // copy data into permuted inputs
    (ap = a3).run();
    (bp = b3).run();

    matmul(ci, ai, bi, axis);
    
    // copy result from permuted output
    (c3 = cp).run();

    cudaStreamSynchronize(0);

    MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatNonHalfTypes,  MatMulOp)
{
  MATX_ENTER_HANDLER();
   
  constexpr index_t m = 16;
  constexpr index_t k = 32;
  constexpr index_t n = 64;
  constexpr index_t b = 8;
    
  tensor_t<TypeParam, 3> a3{{b, m, k}};
  tensor_t<TypeParam, 3> b3{{b, k, n}};
  tensor_t<TypeParam, 3> c3{{b, m, n}};
    
  this->pb->template InitAndRunTVGenerator<TypeParam>(
    "00_transforms", "matmul_operators", "run", {b, m, k, n});

  this->pb->NumpyToTensorView(a3, "a");
  this->pb->NumpyToTensorView(b3, "b");
  
  { // simple identity remaps

    auto rb = range<0>({b},0, 1);

    auto ar = remap<0>(a3, rb);
    auto br = remap<0>(b3, rb);
    auto cr = remap<0>(c3, rb);

    matmul(cr, ar, br);
    
    MATX_TEST_ASSERT_COMPARE(this->pb, c3, "c", this->thresh);
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumMatVec)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 1;

  tensor_t<TypeParam, 2> a{{m, k}};
  tensor_t<TypeParam, 2> b{{k, n}};
  tensor_t<TypeParam, 2> c{{m, n}};
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto cs = slice<1>(c, {0,0}, {matxEnd, matxDropDim});
  auto bs = slice<1>(b, {0,0}, {matxEnd, matxDropDim});
  matvec<decltype(cs), decltype(a), decltype(bs), PROVIDER_TYPE_CUBLASLT>(cs, a, bs);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  // Test also with rank-1 tensors rather than just slices
  tensor_t<TypeParam, 1> bv{{k}};
  tensor_t<TypeParam, 1> cv{{m}};
  (bv = bs).run();
  (cv = cs).run();
  matvec<decltype(cv), decltype(a), decltype(bv), PROVIDER_TYPE_CUBLASLT>(cv, a, bv);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MediumMatVecBatch)
{
  MATX_ENTER_HANDLER();
  constexpr index_t m = 128;
  constexpr index_t k = 256;
  constexpr index_t n = 1;
  constexpr index_t blocks = 8;

  tensor_t<TypeParam, 3> a{{blocks, m, k}};
  tensor_t<TypeParam, 3> b{{blocks, k, n}};
  tensor_t<TypeParam, 3> c{{blocks, m, n}};
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {blocks, m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto cs = slice<2>(c, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
  auto bs = slice<2>(b, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
  matvec<decltype(cs), decltype(a), decltype(bs), PROVIDER_TYPE_CUBLASLT>(cs, a, bs);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  tensor_t<TypeParam, 2> bv{{blocks, k}};
  tensor_t<TypeParam, 2> cv{{blocks, m}};
  (bv = bs).run();
  (cv = cs).run();
  matvec<decltype(cv), decltype(a), decltype(bv), PROVIDER_TYPE_CUBLASLT>(cv, a, bv);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(MatMulTestFloatTypes, MatVecRowVector)
{
  MATX_ENTER_HANDLER();
  // Test that the second-to-last dimension of A can be 1 (i.e. A can be a row
  // vector). In the case of matvec, this means that A*b is effectively a dot product.
  constexpr index_t m = 1;
  constexpr index_t k = 256;
  constexpr index_t n = 1;
  constexpr index_t blocks = 8;

  tensor_t<TypeParam, 3> a{{blocks, m, k}};
  tensor_t<TypeParam, 3> b{{blocks, k, n}};
  tensor_t<TypeParam, 3> c{{blocks, m, n}};
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "matmul_operators", "run", {blocks, m, k, n});

  this->pb->NumpyToTensorView(a, "a");
  this->pb->NumpyToTensorView(b, "b");

  auto cs = slice<2>(c, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
  auto bs = slice<2>(b, {0,0,0}, {matxEnd, matxEnd, matxDropDim});
  matvec<decltype(cs), decltype(a), decltype(bs), PROVIDER_TYPE_CUBLASLT>(cs, a, bs);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  tensor_t<TypeParam, 2> bv{{blocks, k}};
  tensor_t<TypeParam, 2> cv{{blocks, m}};
  (bv = bs).run();
  (cv = cs).run();
  matvec<decltype(cv), decltype(a), decltype(bv), PROVIDER_TYPE_CUBLASLT>(cv, a, bv);

  MATX_TEST_ASSERT_COMPARE(this->pb, c, "c", this->thresh);

  MATX_EXIT_HANDLER();
}
