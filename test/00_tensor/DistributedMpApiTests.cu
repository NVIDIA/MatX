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
