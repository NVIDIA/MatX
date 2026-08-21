////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include "matx/distributed.h"
#include "gtest/gtest.h"

using namespace matx;
using namespace matx::experimental;

namespace {

using distributed_matrix =
    distributed_tensor_t<float, 2, block_distribution_t<2>>;

static_assert(
    decltype(sum(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(prod(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(min(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(max(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(mean(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(all(std::declval<const distributed_matrix &>()))::Rank() == 0);
static_assert(
    decltype(any(std::declval<const distributed_matrix &>()))::Rank() == 0);

template <typename Tensor>
concept CanAssignDistributedSum = requires(Tensor &output,
                                           const distributed_matrix &input,
                                           distributedCUDAExecutor &executor) {
  (output = sum(input)).run(executor);
};

static_assert(CanAssignDistributedSum<
              distributed_tensor_t<float, 0, replicated_distribution_t<0>>>);

} // namespace

TEST(DistributedMpApi, RankZeroReplicatedTensor) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    GTEST_SKIP() << "This API validation test requires a CUDA device";
  }
  distributed_context context{{0}};
  replicated_distribution_t<0> distribution{distributed_index_t<0>{}, {{0, 0}}};
  auto scalar = make_distributed_tensor<float>(distribution, context);
  EXPECT_EQ(scalar.Rank(), 0);
  EXPECT_EQ(scalar.TotalSize(), 1);
  EXPECT_EQ(scalar.LocalFragmentCount(), 1U);
  EXPECT_EQ(scalar.LocalView(0).TotalSize(), 1);
}

#if defined(MATX_EN_CUBLASMP) || defined(MATX_EN_CUSOLVERMP)
TEST(DistributedMpApi, RejectsNullCommunicatorBeforeBackendSetup) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0) {
    GTEST_SKIP() << "This API validation test requires a CUDA device";
  }

  distributed_context context{{0}};
  EXPECT_THROW((distributedCUDAExecutor{context, nullptr, 1, 1}),
               matx::detail::matxException);
}
#endif
