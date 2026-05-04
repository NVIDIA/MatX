////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include "matx.h"

#include "gtest/gtest.h"

#include <string>

using namespace matx;

namespace {

struct ShapeChecker {
  cuda::std::array<index_t, 2> dims;

  static constexpr int Rank()
  {
    return 2;
  }

  index_t Size(int dim) const
  {
    return dims[dim];
  }

  template <typename Op>
  void Check(Op &op) const
  {
    MATX_ASSERT_COMPATIBLE_OP_SIZES(op);
  }
};

void ExpectErrorString(matxError_t error, const std::string &expected)
{
  EXPECT_EQ(std::string(matxErrorString(error)), expected);
}

} // namespace

TEST(ErrorTests, ErrorStringCoversAllCodes)
{
  ExpectErrorString(matxSuccess, "matxSuccess");
  ExpectErrorString(matxIOError, "matxIOError");
  ExpectErrorString(matxOutOfMemory, "matxOutOfMemory");
  ExpectErrorString(matxNotSupported, "matxNotSupported");
  ExpectErrorString(matxInvalidParameter, "matxInvalidParameter");
  ExpectErrorString(matxInvalidDim, "matxInvalidDim");
  ExpectErrorString(matxInvalidSize, "matxInvalidSize");
  ExpectErrorString(matxCudaError, "matxCudaError");
  ExpectErrorString(matxCufftError, "matxCufftError");
  ExpectErrorString(matxLibMathdxError, "matxLibMathdxError");
  ExpectErrorString(matxMatMulError, "matxMatMulError");
  ExpectErrorString(matxAssertError, "matxAssertError");
  ExpectErrorString(matxInvalidType, "matxInvalidType");
  ExpectErrorString(matxLUError, "matxLUError");
  ExpectErrorString(matxInverseError, "matxInverseError");
  ExpectErrorString(matxSolverError, "matxSolverError");
  ExpectErrorString(matxcuTensorError, "matxcuTensorError");
  ExpectErrorString(matxInvalidExecutor, "matxInvalidExecutor");
  volatile int unknown_error_code = 999;
  ExpectErrorString(static_cast<matxError_t>(unknown_error_code), "Unknown");
}

TEST(ErrorTests, ExceptionFormatsCharAndStringMessages)
{
  detail::matxException char_msg(matxInvalidParameter, "char message", "char_file.cu", 12);
  EXPECT_EQ(char_msg.e, matxInvalidParameter);
  EXPECT_TRUE(std::string(char_msg.what()).find("matxInvalidParameter") != std::string::npos);
  EXPECT_TRUE(std::string(char_msg.what()).find("char message") != std::string::npos);
  EXPECT_TRUE(std::string(char_msg.what()).find("char_file.cu:12") != std::string::npos);

  detail::matxException string_msg(matxNotSupported, std::string("string message"), "string_file.cu", 34);
  EXPECT_EQ(string_msg.e, matxNotSupported);
  EXPECT_TRUE(std::string(string_msg.what()).find("matxNotSupported") != std::string::npos);
  EXPECT_TRUE(std::string(string_msg.what()).find("string message") != std::string::npos);
  EXPECT_TRUE(std::string(string_msg.what()).find("string_file.cu:34") != std::string::npos);
}

TEST(ErrorTests, ThrowAndAssertMacrosRaise)
{
  EXPECT_THROW({ MATX_THROW(matxInvalidType, "manual throw"); }, detail::matxException);

#ifndef NDEBUG
  EXPECT_THROW({ MATX_ASSERT(false, matxAssertError); }, detail::matxException);
  EXPECT_THROW({ MATX_ASSERT_STR(false, matxInvalidParameter, "extra context"); }, detail::matxException);
  EXPECT_THROW({ MATX_ASSERT_STR_EXP(1, 2, matxInvalidSize, "mismatch"); }, detail::matxException);
#endif

  MATX_ASSERT(true, matxAssertError);
  MATX_ASSERT_STR(true, matxInvalidParameter, "unused");
  MATX_ASSERT_STR_EXP(2, 2, matxInvalidSize, "unused");
}

TEST(ErrorTests, CudaCheckHandlesSuccessAndFailure)
{
  MATX_CUDA_CHECK(cudaSuccess);
  cudaGetLastError();
  MATX_CUDA_CHECK_LAST_ERROR();
  EXPECT_THROW({ MATX_CUDA_CHECK(cudaErrorInvalidValue); }, detail::matxException);
}

TEST(ErrorTests, CompatibleOperatorSizesAreChecked)
{
  ShapeChecker checker{{2, 3}};
  auto compatible = make_tensor<int>({2, 3}, MATX_HOST_MALLOC_MEMORY);
  auto incompatible = make_tensor<int>({2, 4}, MATX_HOST_MALLOC_MEMORY);

  checker.Check(compatible);
  EXPECT_THROW({ checker.Check(incompatible); }, detail::matxException);
}
