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

#ifdef MATX_EN_JIT
struct EmptyLastDimJitOp {
  using matxop = bool;
  using value_type = float;

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
  {
    return 2;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
  {
    return dim == 0 ? 2 : 0;
  }

  __MATX_INLINE__ std::string str() const
  {
    return "EmptyLastDimJitOp";
  }

  template <typename CapType, typename... Is>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ value_type operator()(Is...) const
  {
    return value_type{};
  }

  template <detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const
  {
    if constexpr (Cap == detail::OperatorCapability::SUPPORTS_JIT) {
      return true;
    }
    else if constexpr (Cap == detail::OperatorCapability::JIT_TYPE_QUERY) {
      return std::string{"EmptyLastDimJitOp"};
    }
    else if constexpr (Cap == detail::OperatorCapability::JIT_CLASS_QUERY) {
      return true;
    }
    else {
      return detail::capability_attributes<Cap>::default_value;
    }
  }
};
#endif

template <typename T> struct CUBTestsData {
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;   
  GExecType exec{};   

  tensor_t<GTestType, 0> t0{{}};
  tensor_t<GTestType, 1> t1{{10}};
  tensor_t<GTestType, 2> t2{{20, 10}};
  tensor_t<GTestType, 3> t3{{30, 20, 10}};
  tensor_t<GTestType, 4> t4{{40, 30, 20, 10}};

  tensor_t<GTestType, 2> t2s = t2.Permute({1, 0});
  tensor_t<GTestType, 3> t3s = t3.Permute({2, 1, 0});
  tensor_t<GTestType, 4> t4s = t4.Permute({3, 2, 1, 0});
};

template <typename TensorType>
class CUBTestsComplex : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsFloat : public ::testing::Test,
                              public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsFloatNonComplex
    : public ::testing::Test,
      public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsNumeric : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsNumericNonComplex
    : public ::testing::Test,
      public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsIntegral : public ::testing::Test,
                                 public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsBoolean : public ::testing::Test,
                                public CUBTestsData<TensorType> {
};
template <typename TensorType>
class CUBTestsAll : public ::testing::Test,
                            public CUBTestsData<TensorType> {
};

template <typename TensorType>
class CUBTestsNumericNonComplexAllExecs : public ::testing::Test, public CUBTestsData<TensorType> {
};


TYPED_TEST_SUITE(CUBTestsAll, MatXAllTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsComplex, MatXComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsFloat, MatXFloatTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsFloatNonComplex, MatXFloatNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsNumeric, MatXNumericTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsIntegral, MatXAllIntegralTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsNumericNonComplex, MatXNumericNonComplexTypesCUDAExec);
TYPED_TEST_SUITE(CUBTestsBoolean, MatXBoolTypesCUDAExec);

TYPED_TEST_SUITE(CUBTestsNumericNonComplexAllExecs,
                 MatXFloatNonComplexNonHalfTypesAllExecs);  

TEST(TensorStats, Hist)
{
  MATX_ENTER_HANDLER(); 

  constexpr int levels = 7;
  tensor_t<float, 1> inv({10});
  tensor_t<int, 1> outv({levels - 1});

  inv.SetVals({2.2, 6.0, 7.1, 2.9, 3.5, 0.3, 2.9, 2.0, 6.1, 999.5});

  cudaExecutor exec{};

  // example-begin hist-test-1
  (outv = hist(inv, 0.0f, 12.0f, levels)).run(exec);
  // example-end hist-test-1
  exec.sync();

  cuda::std::array<int, levels - 1> sol = {1, 5, 0, 3, 0, 0};
  for (index_t i = 0; i < outv.Lsize(); i++) {
    ASSERT_NEAR(outv(i), sol[i], 0.001);
  }

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, HistNonContiguousInput)
{
  MATX_ENTER_HANDLER();

  constexpr int levels = 4;
  tensor_t<float, 1> inv_storage({12});
  tensor_t<int, 1> outv({levels - 1});

  inv_storage.SetVals({0, 99, 1, 99, 2, 99, 0, 99, 1, 99, 2, 99});

  cudaExecutor exec{};
  const index_t hist_shape[1] = {6};
  const index_t hist_strides[1] = {2};
  auto inv_strided = make_tensor(inv_storage.Data(), hist_shape, hist_strides);
  (outv = hist(inv_strided, 0.0f, 3.0f, levels)).run(exec);
  exec.sync();

  for (index_t i = 0; i < outv.Size(0); i++) {
    ASSERT_EQ(outv(i), 2);
  }

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, CumsumNonContiguousInput)
{
  MATX_ENTER_HANDLER();

  tensor_t<int, 2> inv({3, 4});
  tensor_t<int, 2> outv({4, 3});

  inv.SetVals({{1, 2, 3, 4},
               {10, 20, 30, 40},
               {100, 200, 300, 400}});

  cudaExecutor exec{};
  auto invp = inv.Permute({1, 0});
  (outv = cumsum(invp)).run(exec);
  exec.sync();

  for (index_t i = 0; i < outv.Size(0); i++) {
    int running = 0;
    for (index_t j = 0; j < outv.Size(1); j++) {
      running += invp(i, j);
      ASSERT_EQ(outv(i, j), running);
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, SortLegacyRank2AndOversizedGuards)
{
  MATX_ENTER_HANDLER();

  cudaExecutor exec{};
  tensor_t<int, 2> inv({2, cubSegmentCuttoff});
  auto outv = make_tensor<int>(inv.Shape());

  for (index_t i = 0; i < inv.Size(0); i++) {
    for (index_t j = 0; j < inv.Size(1); j++) {
      inv(i, j) = static_cast<int>(inv.Size(1) - j + i);
    }
  }

  (outv = matx::sort(inv, SORT_DIR_ASC)).run(exec);
  exec.sync();
  for (index_t i = 0; i < outv.Size(0); i++) {
    ASSERT_EQ(outv(i, 0), static_cast<int>(i + 1));
    ASSERT_EQ(outv(i, outv.Size(1) - 1), static_cast<int>(outv.Size(1) + i));
  }

  (outv = matx::sort(inv, SORT_DIR_DESC)).run(exec);
  exec.sync();
  for (index_t i = 0; i < outv.Size(0); i++) {
    ASSERT_EQ(outv(i, 0), static_cast<int>(outv.Size(1) + i));
    ASSERT_EQ(outv(i, outv.Size(1) - 1), static_cast<int>(i + 1));
  }

  int *ptr = nullptr;
  index_t *idx_ptr = nullptr;
  auto huge_in = make_tensor<int>(ptr, {2048, 2048, 1024}, false);
  auto huge_out = make_tensor<int>(ptr, {2048, 2048, 1024}, false);
  auto huge_idx_in = make_tensor<index_t>(idx_ptr, {2048, 2048, 1024}, false);
  auto huge_idx_out = make_tensor<index_t>(idx_ptr, {2048, 2048, 1024}, false);

  EXPECT_THROW({ detail::sort_impl_inner(huge_out, huge_in, SORT_DIR_ASC, exec); },
               detail::matxException);
  EXPECT_THROW({ detail::sort_pairs_impl_inner(huge_idx_out, huge_idx_in, huge_out, huge_in, SORT_DIR_ASC, exec); },
               detail::matxException);

  tensor_t<int, 2> noncontig_base({2, 3});
  tensor_t<int, 2> noncontig_out({3, 2});
  auto noncontig_view = noncontig_base.Permute({1, 0});
  detail::SortParams_t sort_params{SORT_DIR_ASC};
  using NonContigSortPlan = detail::matxCubPlan_t<decltype(noncontig_out),
                                                  decltype(noncontig_view),
                                                  detail::CUB_OP_RADIX_SORT,
                                                  detail::SortParams_t>;
  auto create_noncontig_sort_plan = [&]() {
    NonContigSortPlan plan(noncontig_out, noncontig_view, sort_params, exec.getStream());
    (void)plan;
  };
  EXPECT_THROW(create_noncontig_sort_plan(), detail::matxException);

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, NonContiguousFindUniqueAndArgReduceOutputs)
{
  MATX_ENTER_HANDLER();

  cudaExecutor exec{};
  tensor_t<float, 2> inv({2, 4});
  inv.SetVals({{1, 1, 2, 2},
               {1, 1, 2, 2}});
  auto invp = inv.Permute({1, 0});

  tensor_t<float, 1> found({8});
  tensor_t<int, 1> found_idx({8});
  tensor_t<int, 0> num_found{{}};

  (mtie(found, num_found) = find(invp, GT{1.5f})).run(exec);
  exec.sync();
  ASSERT_EQ(num_found(), 4);
  for (index_t i = 0; i < num_found(); i++) {
    ASSERT_EQ(found(i), 2.0f);
  }

  (mtie(found_idx, num_found) = find_idx(invp, GT{1.5f})).run(exec);
  exec.sync();
  ASSERT_EQ(num_found(), 4);
  for (index_t i = 0; i < num_found(); i++) {
    ASSERT_EQ(found_idx(i), i + 4);
  }

  tensor_t<float, 1> unique_out({8});
  auto unique_params = detail::UniqueParams_t<decltype(num_found)>{num_found};
  auto unique_plan = detail::matxCubPlan_t<decltype(unique_out),
                                           decltype(invp),
                                           detail::CUB_OP_UNIQUE,
                                           decltype(unique_params)>{unique_out, invp, unique_params, exec.getStream()};
  unique_plan.ExecUnique(unique_out, invp, exec.getStream());
  exec.sync();
  ASSERT_EQ(num_found(), 2);
  ASSERT_EQ(unique_out(0), 1.0f);
  ASSERT_EQ(unique_out(1), 2.0f);

  tensor_t<float, 2> arg_in({2, 3});
  tensor_t<float, 1> arg_out({2});
  tensor_t<index_t, 1> arg_idx({2});
  arg_in.SetVals({{1, 5, 2},
                  {4, 3, 6}});

  (mtie(arg_out, arg_idx) = argmax(arg_in, {1})).run(exec);
  exec.sync();
  ASSERT_EQ(arg_out(0), 5.0f);
  ASSERT_EQ(arg_idx(0), 1);
  ASSERT_EQ(arg_out(1), 6.0f);
  ASSERT_EQ(arg_idx(1), 5);

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, InternalCubPlansRejectInvalidOperations)
{
  int *ptr = nullptr;
  index_t *idx_ptr = nullptr;
  auto in = make_tensor<int>(ptr, {4}, false);
  auto out = make_tensor<int>(ptr, {}, false);
  auto idx = make_tensor<index_t>(idx_ptr, {}, false);
  using SingleParams = detail::ReduceParams_t<detail::CustomArgMaxCmp, cuda::std::tuple<index_t, int>>;
  using DualParams = detail::ReduceParams_t<detail::CustomArgMinMaxCmp, cuda::std::tuple<index_t, int, index_t, int>>;
  SingleParams single_params{detail::CustomArgMaxCmp{}, cuda::std::make_tuple(index_t{0}, int{})};
  DualParams dual_params{detail::CustomArgMinMaxCmp{}, cuda::std::make_tuple(index_t{0}, int{}, index_t{0}, int{})};

  using SinglePlan = detail::matxCubSingleArgPlan_t<decltype(out), decltype(idx), decltype(in), SingleParams>;
  using DualPlan = detail::matxCubDualArgPlan_t<decltype(out), decltype(idx), decltype(in), DualParams>;
  auto create_single_plan = [&]() {
    SinglePlan plan(out, idx, in, detail::CUB_OP_REDUCE, single_params, 0);
    (void)plan;
  };
  auto create_dual_plan = [&]() {
    DualPlan plan(out, idx, out, idx, in, detail::CUB_OP_REDUCE, dual_params, 0);
    (void)plan;
  };

  EXPECT_THROW(create_single_plan(), detail::matxException);
  EXPECT_THROW(create_dual_plan(), detail::matxException);
}

#ifdef MATX_EN_JIT
TEST(TensorStats, CubBlockJIT)
{
  MATX_ENTER_HANDLER();

  CUDAJITExecutor exec{};
  constexpr index_t rows = 2;
  constexpr index_t batches = 4;
  constexpr index_t cols = 16;
  constexpr index_t ndiv_cols = 17;

  tensor_t<float, 3> in({rows, batches, cols});
  tensor_t<float, 3> ndiv_in({rows, batches, ndiv_cols});
  tensor_t<float, 2> reduce_out({rows, batches});
  tensor_t<float, 2> prod_out({rows, batches});
  tensor_t<float, 2> min_out({rows, batches});
  tensor_t<float, 2> max_out({rows, batches});
  tensor_t<float, 2> ndiv_reduce_out({rows, batches});
  tensor_t<float, 2> ndiv_prod_out({rows, batches});
  tensor_t<float, 2> ndiv_min_out({rows, batches});
  tensor_t<float, 2> ndiv_max_out({rows, batches});
  tensor_t<float, 3> ndiv_scan_out({rows, batches, ndiv_cols});
  tensor_t<float, 3> ndiv_sort_out({rows, batches, ndiv_cols});
  tensor_t<index_t, 3> ndiv_argsort_out({rows, batches, ndiv_cols});
  tensor_t<float, 2> shift_in({rows, batches});
  tensor_t<float, 2> mixed_out({rows, batches});
  tensor_t<float, 3> scan_out({rows, batches, cols});
  tensor_t<float, 3> sort_out({rows, batches, cols});
  tensor_t<float, 3> sort_desc_out({rows, batches, cols});
  tensor_t<index_t, 3> argsort_out({rows, batches, cols});
  tensor_t<index_t, 3> argsort_desc_out({rows, batches, cols});
  tensor_t<float, 0> scalar_sum{{}};
  tensor_t<float, 0> scalar_prod{{}};
  tensor_t<float, 1> vector_in({cols});

  for (index_t i = 0; i < rows; i++) {
    for (index_t j = 0; j < batches; j++) {
      shift_in(i, j) = static_cast<float>(100 * i + 10 * j);
      for (index_t k = 0; k < cols; k++) {
        in(i, j, k) = static_cast<float>((k % 5) + 1 + j);
      }
      for (index_t k = 0; k < ndiv_cols; k++) {
        ndiv_in(i, j, k) = 1.0f + static_cast<float>((i + j + k) % 5) * 0.125f;
      }
    }
  }
  for (index_t k = 0; k < cols; k++) {
    vector_in(k) = static_cast<float>((k % 7) + 1);
  }

  (reduce_out = sum(in, {2})).run(exec);
  (prod_out = prod(in, {2})).run(exec);
  (min_out = min(in, {2})).run(exec);
  (max_out = max(in, {2})).run(exec);
  (ndiv_reduce_out = sum(ndiv_in, {2})).run(exec);
  (ndiv_prod_out = prod(ndiv_in, {2})).run(exec);
  (ndiv_min_out = min(ndiv_in, {2})).run(exec);
  (ndiv_max_out = max(ndiv_in, {2})).run(exec);
  (ndiv_scan_out = cumsum(ndiv_in)).run(exec);
  (ndiv_sort_out = matx::sort(ndiv_in, SORT_DIR_ASC)).run(exec);
  (ndiv_argsort_out = argsort(ndiv_in, SORT_DIR_ASC)).run(exec);
  (mixed_out = sum(in, {2}) + fftshift1D(shift_in) + 5.0f).run(exec);
  (scan_out = cumsum(in)).run(exec);
  (sort_out = matx::sort(in, SORT_DIR_ASC)).run(exec);
  (sort_desc_out = matx::sort(in, SORT_DIR_DESC)).run(exec);
  (argsort_out = argsort(in, SORT_DIR_ASC)).run(exec);
  (argsort_desc_out = argsort(in, SORT_DIR_DESC)).run(exec);
  (scalar_sum = sum(vector_in)).run(exec);
  (scalar_prod = prod(vector_in)).run(exec);
  exec.sync();

  float expected_scalar_sum = 0.0f;
  float expected_scalar_prod = 1.0f;
  for (index_t k = 0; k < cols; k++) {
    expected_scalar_sum += vector_in(k);
    expected_scalar_prod *= vector_in(k);
  }
  ASSERT_NEAR(scalar_sum(), expected_scalar_sum, 0.001f);
  const auto scalar_prod_tol = cuda::std::max(0.001f, expected_scalar_prod * 1e-5f);
  ASSERT_NEAR(scalar_prod(), expected_scalar_prod, scalar_prod_tol);

  for (index_t i = 0; i < rows; i++) {
    for (index_t j = 0; j < batches; j++) {
      float expected_sum = 0.0f;
      float expected_prod = 1.0f;
      float expected_min = in(i, j, 0);
      float expected_max = in(i, j, 0);
      float running = 0.0f;
      for (index_t k = 0; k < cols; k++) {
        expected_sum += in(i, j, k);
        expected_prod *= in(i, j, k);
        expected_min = cuda::std::min(expected_min, in(i, j, k));
        expected_max = cuda::std::max(expected_max, in(i, j, k));
        running += in(i, j, k);
        ASSERT_NEAR(scan_out(i, j, k), running, 0.001f);
        if (k > 0) {
          ASSERT_LE(sort_out(i, j, k - 1), sort_out(i, j, k));
          ASSERT_GE(sort_desc_out(i, j, k - 1), sort_desc_out(i, j, k));
          ASSERT_LE(in(i, j, argsort_out(i, j, k - 1)), in(i, j, argsort_out(i, j, k)));
          ASSERT_GE(in(i, j, argsort_desc_out(i, j, k - 1)), in(i, j, argsort_desc_out(i, j, k)));
        }
        ASSERT_GE(argsort_out(i, j, k), static_cast<index_t>(0));
        ASSERT_LT(argsort_out(i, j, k), cols);
        ASSERT_GE(argsort_desc_out(i, j, k), static_cast<index_t>(0));
        ASSERT_LT(argsort_desc_out(i, j, k), cols);
      }
      ASSERT_NEAR(reduce_out(i, j), expected_sum, 0.001f);
      const auto prod_tol = cuda::std::max(0.001f, expected_prod * 1e-5f);
      ASSERT_NEAR(prod_out(i, j), expected_prod, prod_tol);
      ASSERT_NEAR(min_out(i, j), expected_min, 0.001f);
      ASSERT_NEAR(max_out(i, j), expected_max, 0.001f);
      const index_t shifted_j = (j + (batches + 1) / 2) % batches;
      ASSERT_NEAR(mixed_out(i, j), expected_sum + shift_in(i, shifted_j) + 5.0f, 0.001f);

      float ndiv_expected_sum = 0.0f;
      float ndiv_expected_prod = 1.0f;
      float ndiv_expected_min = ndiv_in(i, j, 0);
      float ndiv_expected_max = ndiv_in(i, j, 0);
      float ndiv_running = 0.0f;
      for (index_t k = 0; k < ndiv_cols; k++) {
        ndiv_expected_sum += ndiv_in(i, j, k);
        ndiv_expected_prod *= ndiv_in(i, j, k);
        ndiv_expected_min = cuda::std::min(ndiv_expected_min, ndiv_in(i, j, k));
        ndiv_expected_max = cuda::std::max(ndiv_expected_max, ndiv_in(i, j, k));
        ndiv_running += ndiv_in(i, j, k);
        ASSERT_NEAR(ndiv_scan_out(i, j, k), ndiv_running, 0.001f);
        if (k > 0) {
          ASSERT_LE(ndiv_sort_out(i, j, k - 1), ndiv_sort_out(i, j, k));
          ASSERT_LE(ndiv_in(i, j, ndiv_argsort_out(i, j, k - 1)), ndiv_in(i, j, ndiv_argsort_out(i, j, k)));
        }
        ASSERT_GE(ndiv_argsort_out(i, j, k), static_cast<index_t>(0));
        ASSERT_LT(ndiv_argsort_out(i, j, k), ndiv_cols);
      }
      ASSERT_NEAR(ndiv_reduce_out(i, j), ndiv_expected_sum, 0.001f);
      const auto ndiv_prod_tol = cuda::std::max(0.001f, ndiv_expected_prod * 1e-5f);
      ASSERT_NEAR(ndiv_prod_out(i, j), ndiv_expected_prod, ndiv_prod_tol);
      ASSERT_NEAR(ndiv_min_out(i, j), ndiv_expected_min, 0.001f);
      ASSERT_NEAR(ndiv_max_out(i, j), ndiv_expected_max, 0.001f);
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, CubBlockJITRejectsEmptyCriticalDim)
{
  MATX_ENTER_HANDLER();

  EmptyLastDimJitOp empty{};

  auto sum_op = sum<EmptyLastDimJitOp, 1>(empty);
  auto prod_op = prod<EmptyLastDimJitOp, 1>(empty);
  auto min_op = matx::min<EmptyLastDimJitOp, 1>(empty);
  auto max_op = matx::max<EmptyLastDimJitOp, 1>(empty);
  auto cumsum_op = cumsum(empty);
  auto sort_op = matx::sort(empty, SORT_DIR_ASC);
  auto argsort_op = argsort(empty, SORT_DIR_ASC);

  EXPECT_FALSE(jit_supported(sum_op));
  EXPECT_FALSE(jit_supported(prod_op));
  EXPECT_FALSE(jit_supported(min_op));
  EXPECT_FALSE(jit_supported(max_op));
  EXPECT_FALSE(jit_supported(cumsum_op));
  EXPECT_FALSE(jit_supported(sort_op));
  EXPECT_FALSE(jit_supported(argsort_op));

  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(sum_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(prod_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(min_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(max_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(cumsum_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(sort_op), 0);
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::STATIC_SHM_SIZE>(argsort_op), 0);

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, CubBlockJITWithOperatorInput)
{
  MATX_ENTER_HANDLER();

  CUDAJITExecutor exec{};
  constexpr index_t rows = 3;
  constexpr index_t cols = 16;

  const float starts[rows] = {1.0f, 2.0f, -3.0f};
  const float stops[rows] = {16.0f, 17.0f, 12.0f};
  tensor_t<float, 1> shift_in({rows});
  tensor_t<float, 1> out({rows});

  for (index_t i = 0; i < rows; i++) {
    shift_in(i) = static_cast<float>(10 * i);
  }

  auto generated = linspace(starts, stops, cols, 1);
  auto expr = sum(generated, {1}) + fftshift1D(shift_in) + 5.0f;
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  (out = expr).run(exec);
  exec.sync();

  for (index_t i = 0; i < rows; i++) {
    const index_t shifted_i = (i + (rows + 1) / 2) % rows;
    const float expected_sum = static_cast<float>(cols) * (starts[i] + stops[i]) * 0.5f;
    ASSERT_NEAR(out(i), expected_sum + shift_in(shifted_i) + 5.0f, 0.001f);
  }

  MATX_EXIT_HANDLER();
}

TEST(TensorStats, CubBlockJITWithFFT)
{
  MATX_ENTER_HANDLER();

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  constexpr index_t batches = 2;
  constexpr index_t cols = 16;

  tensor_t<cuda::std::complex<float>, 2> in({batches, cols});
  tensor_t<cuda::std::complex<float>, 1> jit_out({batches});
  tensor_t<cuda::std::complex<float>, 1> ref_out({batches});

  for (index_t i = 0; i < batches; i++) {
    for (index_t j = 0; j < cols; j++) {
      in(i, j) = cuda::std::complex<float>{static_cast<float>(i + j), static_cast<float>(j % 3)};
    }
  }

  auto expr = sum(fft(in), {1});
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  (jit_out = expr).run(jit_exec);
  (ref_out = sum(fft(in), {1})).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t i = 0; i < batches; i++) {
    ASSERT_NEAR(jit_out(i).real(), ref_out(i).real(), 0.01f);
    ASSERT_NEAR(jit_out(i).imag(), ref_out(i).imag(), 0.01f);
  }

  MATX_EXIT_HANDLER();
}
#endif

TEST(TensorStats, NoMatchesFindAndUniqueHost)
{
  MATX_ENTER_HANDLER();

  HostExecutor exec{};
  tensor_t<float, 1> in({3});
  tensor_t<float, 1> out({3});
  tensor_t<int, 1> idx({3});
  tensor_t<int, 0> num_found{{}};

  in.SetVals({1, 2, 3});

  (mtie(out, num_found) = find(in, GT{10.0f})).run(exec);
  ASSERT_EQ(num_found(), 0);

  (mtie(idx, num_found) = find_idx(in, GT{10.0f})).run(exec);
  ASSERT_EQ(num_found(), 0);

  in.SetVals({1, 1, 2});

  (mtie(out, num_found) = unique(in)).run(exec);
  ASSERT_EQ(num_found(), 2);
  ASSERT_EQ(out(0), 1.0f);
  ASSERT_EQ(out(1), 2.0f);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CUBTestsNumericNonComplexAllExecs, CumSum)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  for (index_t i = 0; i < this->t1.Lsize(); i++) {
    this->t1(i) = static_cast<TestType>((2 * (i % 2) - 1) * i);
  }

  tensor_t<TestType, 1> tmpv({this->t1.Lsize()});

  // Ascending
  // example-begin cumsum-test-1
  // Compute the cumulative sum/exclusive scan across "t1"
  (tmpv = cumsum(this->t1)).run(this->exec);
  // example-end cumsum-test-1
  this->exec.sync();

  TestType ttl = 0;
  for (index_t i = 0; i < tmpv.Lsize(); i++) {
    ttl += this->t1(i);
    ASSERT_NEAR(tmpv(i), ttl, 0.001);
  }

  // 2D tests
  auto tmpv2 = make_tensor<TestType>(this->t2.Shape());

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TestType>((2 * (j % 2) - 1) * j + i);
    }
  }

  (tmpv2 = cumsum(this->t2)).run(this->exec);
  this->exec.sync();
  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    ttl = 0;
    for (index_t j = 0; j < tmpv2.Size(1); j++) {
      ttl += this->t2(i, j);
      ASSERT_NEAR(tmpv2(i, j), ttl, 0.001) << i << j;
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(CUBTestsNumericNonComplexAllExecs, Sort)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
 

  for (index_t i = 0; i < this->t1.Lsize(); i++) {
    this->t1(i) = static_cast<TestType>((2 * (i % 2) - 1) * i);
  }

  auto tmpv = make_tensor<TestType>({this->t1.Lsize()});

  // example-begin sort-test-1
  // Ascending sort of 1D input
  (tmpv = matx::sort(this->t1, SORT_DIR_ASC)).run(this->exec);
  // example-end sort-test-1
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_TRUE(tmpv(i) > tmpv(i - 1));
  }

  // example-begin sort-test-2
  // Descending sort of 1D input
  (tmpv = matx::sort(this->t1, SORT_DIR_DESC)).run(this->exec);
  // example-end sort-test-2
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_TRUE(tmpv(i) < tmpv(i - 1));
  }

  // operator input test
  const auto L = tmpv.Lsize();
  (tmpv = matx::sort(matx::concat(0,
    static_cast<TestType>(2.0) * matx::ones<TestType>({1}), matx::ones<TestType>({L-1})), SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();
  ASSERT_TRUE(tmpv(L-1) == static_cast<TestType>(2.0));
  for (index_t i = 0; i < L-1; i++) {
    ASSERT_TRUE(tmpv(i) == static_cast<TestType>(1.0));
  }

  // 2D tests
  auto tmpv2 = make_tensor<TestType>(this->t2.Shape());

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TestType>((2 * (j % 2) - 1) * j + i);
    }
  }

  (tmpv2 = matx::sort(this->t2, SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_TRUE(tmpv2(i, j) > tmpv2(i, j - 1));
    }
  }

  // Sort the first column of t2
  auto tmpslice = make_tensor<TestType>({this->t2.Size(0)});
  (tmpslice = matx::sort(matx::slice<1>(this->t2, {0, 0}, {matx::matxEnd, matx::matxDropDim}), SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 1; i < this->t2.Size(0); i++) {
    ASSERT_TRUE(tmpslice(i) > tmpslice(i-1));
  }

  // Descending
  (tmpv2 = matx::sort(this->t2, SORT_DIR_DESC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_TRUE(tmpv2(i, j) < tmpv2(i, j - 1));
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(CUBTestsNumericNonComplexAllExecs, Argsort)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
 

  for (index_t i = 0; i < this->t1.Lsize(); i++) {
    this->t1(i) = static_cast<TestType>((2 * (i % 2) - 1) * i);
  }

  auto tmpv = make_tensor<index_t>({this->t1.Lsize()});

  // example-begin argsort-test-1
  // Ascending argsort of 1D input
  (tmpv = matx::argsort(this->t1, SORT_DIR_ASC)).run(this->exec);
  // example-end argsort-test-1
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_GT(this->t1(tmpv(i)), this->t1(tmpv(i - 1)));
  }

  // example-begin argsort-test-2
  // Descending argsort of 1D input
  (tmpv = matx::argsort(this->t1, SORT_DIR_DESC)).run(this->exec);
  // example-end argsort-test-2
  this->exec.sync();

  for (index_t i = 1; i < tmpv.Lsize(); i++) {
    ASSERT_LT(this->t1(tmpv(i)), this->t1(tmpv(i - 1)));
  }

  // operator input test
  const auto L = tmpv.Lsize();
  (tmpv = matx::argsort(matx::concat(0,
    static_cast<TestType>(2.0) * matx::ones<TestType>({1}), matx::ones<TestType>({L-1})), SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();
  ASSERT_EQ(tmpv(L-1), 0);
  for (index_t i = 0; i < L-1; i++) {
    ASSERT_EQ(tmpv(i), i+1);
  }

  // 2D tests
  auto tmpv2 = make_tensor<index_t>(this->t2.Shape());

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TestType>((2 * (j % 2) - 1) * j + i);
    }
  }

  (tmpv2 = matx::argsort(this->t2, SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_GT(this->t2(i, tmpv2(i, j)), this->t2(i, tmpv2(i, j - 1)));
    }
  }

  // Sort the first column of t2
  auto tmpslice = make_tensor<index_t>({this->t2.Size(0)});
  (tmpslice = matx::argsort(matx::slice<1>(this->t2, {0, 0}, {matx::matxEnd, matx::matxDropDim}), SORT_DIR_ASC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 1; i < this->t2.Size(0); i++) {
    ASSERT_GT(this->t2(tmpslice(i), 0), this->t2(tmpslice(i-1), 0));
  }

  // Descending
  (tmpv2 = matx::argsort(this->t2, SORT_DIR_DESC)).run(this->exec);
  this->exec.sync();

  for (index_t i = 0; i < tmpv2.Size(0); i++) {
    for (index_t j = 1; j < tmpv2.Size(1); j++) {
      ASSERT_LT(this->t2(i, tmpv2(i, j)), this->t2(i, tmpv2(i, j - 1)));
    }
  }

  MATX_EXIT_HANDLER();
}
