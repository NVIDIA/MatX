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

template <typename T> class FFTTest : public ::testing::Test {

protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;
  GExecType exec{};   
  void SetUp() override
  {
    CheckTestTypeSupport<GTestType>();

    if constexpr (!detail::CheckFFTSupport<GExecType, GTestType>()) {
      GTEST_SKIP();
    }

    // Use an arbitrary number of threads for the select threads host exec.
    if constexpr (is_select_threads_host_executor_v<GExecType>) {
      HostExecParams params{4};
      exec = SelectThreadsHostExecutor{params};
    }

    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to fp32
    if constexpr (is_complex_half_v<GTestType>) {
      thresh = 0.4f;
    }
  }

  void TearDown() override { pb.reset(); }
  std::unique_ptr<detail::MatXPybind> pb;
  float thresh = 0.01f;
};

template <typename TensorType>
class FFTTestComplexTypes : public FFTTest<TensorType> {
};

template <typename TensorType>
class FFTTestComplexNonHalfTypes : public FFTTest<TensorType> {
};

template <typename TensorType>
class FFTTestComplexNonHalfTypesAllExecs : public FFTTest<TensorType> {
};

TYPED_TEST_SUITE(FFTTestComplexTypes, MatXComplexTypesAllExecs);
TYPED_TEST_SUITE(FFTTestComplexNonHalfTypes, MatXComplexNonHalfTypesAllExecs);
TYPED_TEST_SUITE(FFTTestComplexNonHalfTypesAllExecs, MatXComplexNonHalfTypesAllExecs);

TYPED_TEST(FFTTestComplexTypes, FFT1D1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim});

    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    // example-begin fft-1
    // Perform a 1D FFT from input av into output avo. Input and output sizes will be deduced by the
    // type of the tensors and output size.
    (avo = fft(av)).run(this->exec);
    // example-end fft-1
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1DFWD1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_1d_fwd", {fft_dim, fft_dim});

    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    // example-begin fft-1-fwd
    // Perform a 1D FFT from input av into output avo with FORWARD scaling (1/N). Input and output sizes will be deduced by the
    // type of the tensors and output size.
    (avo = fft(av, fft_dim, FFTNorm::FORWARD)).run(this->exec);
    // example-end fft-1
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1DORTHO1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_1d_ortho", {fft_dim, fft_dim});

    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    // example-begin fft-1
    // Perform a 1D FFT from input av into output avo with ORTHO scaling (1/sqrt(N)). Input and output sizes will be deduced by the
    // type of the tensors and output size.
    (avo = fft(av, fft_dim, FFTNorm::ORTHO)).run(this->exec);
    // example-end fft-1
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT1Axis)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  const int d1 = 8;
  const int d2 = 32;
  const int d3 = 64;

  // example-begin fft-2
  auto in = make_tensor<TestType>({d1, d2, d3});
  auto out1 = make_tensor<TestType>({d1, d2, d3});
  auto out2 = make_tensor<TestType>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        in(i,j,k) = static_cast<TestType>((float)(i+j+k));
      }
    }
  }

  // Perform a batched 1D FFT on a 3D tensor across axis 2. Since axis 2 is the last dimension,
  // this is equivalent to not specifying the axis
  (out1 = fft(in)).run(this->exec);
  (out2 = fft(in, {2})).run(this->exec);

  // example-end fft-2
  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  // example-begin fft-3
  // Perform a batched 1D FFT on a 3D tensor across axis 1. This is equivalent to permuting the last
  // two axes before input and after the output
  (out1.Permute({0,2,1}) = fft(in.Permute({0,2,1}))).run(this->exec);
  (out2 = fft(in, {1})).run(this->exec);  

  // example-end fft-3
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  // example-begin ifft-1
  // Perform a batched 1D IFFT on a 3D tensor across axis 2. Since axis 2 is the last dimension,
  // this is equivalent to not specifying the axis
  (out1 = ifft(in)).run(this->exec);
  (out2 = ifft(in, {2})).run(this->exec);    
  // example-end ifft-1
  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  // example-begin ifft-2
  // Perform a batched 1D IFFT on a 3D tensor across axis 1. This is equivalent to permuting the last
  // two axes before input and after the output
  (out1.Permute({0,2,1}) = ifft(in.Permute({0,2,1}))).run(this->exec);
  (out2 = ifft(in, {1})).run(this->exec);    
  // example-end ifft-2
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  {
    auto in1 =  ones<TestType>(in.Shape());
    (out1.Permute({0,2,1}) = fft(permute(in1, {0,2,1}))).run(this->exec);
    (out2 = fft(in1, {1})).run(this->exec);

    this->exec.sync();

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }


    (out1.Permute({1,2,0}) = ifft(permute(in1, {1,2,0}))).run(this->exec);
    (out2 = ifft(in1, {0})).run(this->exec);
    this->exec.sync();

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT2Axis)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  const int d1 = 8;
  const int d2 = 16;
  const int d3 = 32;

  auto in = make_tensor<TestType>({d1, d2, d3});
  auto out1 = make_tensor<TestType>({d1, d2, d3});
  auto out2 = make_tensor<TestType>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        in(i,j,k) = sin(float(i+1)) + sin(float(j+1)) + sin(float(k+1));
      }
    }
  }

  // example-begin fft2-1
  // Perform a 2D FFT from 3D tensor "in" into "out1". This is equivalent to performing the FFT
  // on the last two dimension unpermuted.
  (out1 = fft2(in)).run(this->exec);
  (out2 = fft2(in, {1,2})).run(this->exec);
  // example-end fft2-1
  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  // example-begin fft2-2
  // Perform a 2D FFT from 3D tensor "in" into "out2" across dimensions 2, 0. This is equivalent
  // to permuting the tensor before input and after output
  (out1.Permute({1,2,0}) = fft2(in.Permute({1,2,0}))).run(this->exec);
  (out2 = fft2(in, {2,0})).run(this->exec);
  // example-end fft2-2
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  (out1.Permute({1,0,2}) = fft2(in.Permute({1,0,2}))).run(this->exec);
  (out2 = fft2(in, {0,2})).run(this->exec);
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  // example-begin ifft2-1
  // Perform a 2D FFT from 3D tensor "in" into "out1". This is equivalent to performing the FFT
  // on the last two dimension unpermuted.
  (out1 = ifft2(in)).run(this->exec);
  (out2 = ifft2(in, {1,2})).run(this->exec);
  // example-end ifft2-1
  this->exec.sync();

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  // example-begin ifft2-2
  // Perform a 2D FFT from 3D tensor "in" into "out2" across dimensions 2, 0. This is equivalent
  // to permuting the tensor before input and after output
  (out1.Permute({1,2,0}) = ifft2(in.Permute({1,2,0}))).run(this->exec);
  (out2 = ifft2(in, {2,0})).run(this->exec);
  // example-end ifft2-2
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  (out1.Permute({1,0,2}) = ifft2(in.Permute({1,0,2}))).run(this->exec);
  (out2 = ifft2(in, {0,2})).run(this->exec);
  this->exec.sync();
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
 
  {
    auto in1 =  ones<TestType>(in.Shape());

    (out1.Permute({1,0,2}) = fft2(permute(in1, {1,0,2}))).run(this->exec);
    (out2 = fft2(in1, {0,2})).run(this->exec);
    this->exec.sync();

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }
    

    (out1.Permute({1,0,2}) = ifft2(permute(in1, {1,0,2}))).run(this->exec);
    (out2 = ifft2(in1, {0,2})).run(this->exec);
    this->exec.sync();

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(FFTTestComplexTypes, IFFT1D1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim});
    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = ifft(av)).run(this->exec);

    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}


TYPED_TEST(FFTTestComplexNonHalfTypes, IRFFT1D1024C2R)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "irfft_1d", {fft_dim, 2 * (fft_dim - 1)});
    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<typename TestType::value_type, 1> avo{{2 * (fft_dim - 1)}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = irfft(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1DORTHO1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_1d_ortho", {fft_dim, fft_dim});
    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = ifft(av, fft_dim, FFTNorm::ORTHO)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1DFWD1024C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_1d_fwd", {fft_dim, fft_dim});
    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = ifft(av, fft_dim, FFTNorm::FORWARD)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024PadC2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim * 2});
    // example-begin fft-4
    auto av = make_tensor<TestType>({fft_dim});
    auto avo = make_tensor<TestType>({fft_dim * 2});
    this->pb->NumpyToTensorView(av, "a_in");

    // Specify the FFT size as bigger than av. Thus, av will be zero-padded to the appropriate size
    (avo = fft(av, fft_dim * 2)).run(this->exec);
    // example-end fft-4
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024PadBatchedC2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 4;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_1d_batched", {fft_dim+1, fft_dim+2, fft_dim*2});
    tensor_t<TestType, 2> av{{fft_dim + 1, fft_dim + 2}};
    tensor_t<TestType, 2> avo{{fft_dim + 1, fft_dim * 2}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft(av, fft_dim*2)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);

    // example-begin fft-5
    // Perform an FFT but force the size to be fft_dim * 2 instead of the output size
    (avo = fft(av, fft_dim * 2)).run(this->exec); // Force the FFT size
    // example-end fft-5
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);  
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1D1024PadC2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim * 2});
    tensor_t<TestType, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim * 2}};
    this->pb->NumpyToTensorView(av, "a_in");

    // Specify the IFFT size as bigger than av. Thus, av will be zero-padded to the appropriate size
    (avo = ifft(av, fft_dim * 2)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024R2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 1024;
    using rtype = typename TestType::value_type;
    this->pb->template InitAndRunTVGenerator<rtype>(
        "00_transforms", "fft_operators", "rfft_1d", {fft_dim, fft_dim});

    tensor_t<typename TestType::value_type, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim / 2 + 1}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024PadR2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>; 
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 4;
    using rtype = typename TestType::value_type;
    this->pb->template InitAndRunTVGenerator<rtype>(
        "00_transforms", "fft_operators", "rfft_1d", {fft_dim, fft_dim*2});

    tensor_t<typename TestType::value_type, 1> av{{fft_dim}};
    tensor_t<TestType, 1> avo{{fft_dim + 1}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft(av, fft_dim*2)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT1DSizeChecks)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ComplexType = TestType;
  //using RealType = typename TestType::value_type;

  const index_t N = 16;
  auto tc = make_tensor<ComplexType>({N});
  auto tr = make_tensor<ComplexType>({N});

  // C2C, output size larger than input size
  ASSERT_THROW({
    auto t2 = make_tensor<ComplexType>({2*N});
    // We do not implicitly zero-pad to a larger transform size
    (t2 = fft(tc)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  // C2C, output size smaller than input size
  ASSERT_THROW({
    auto t2 = make_tensor<ComplexType>({(N/2)+1});
    (t2 = fft(tc)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  // R2C, output size smaller than N/2 + 1
  ASSERT_THROW({
    auto t2 = make_tensor<ComplexType>({N/2});
    (t2 = fft(tr)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  // R2C, output size larger than N/2 + 1
  ASSERT_THROW({
    auto t2 = make_tensor<ComplexType>({N/2+2});
    (t2 = fft(tr)).run(this->exec);
    this->exec.sync();
  }, matx::detail::matxException);

  // Enable once C2R is supported
//   // C2R, output size smaller than N
//   ASSERT_THROW({
//     auto tcs = slice(tc, {0}, {N/2+1});
//     auto t2 = make_tensor<RealType>({N-1});
//     (t2 = fft(tcs)).run(this->exec);
//     this->exec.sync();
//   }, matx::detail::matxException);

//   // C2R, output size too large
//  ASSERT_THROW({
//     auto tcs = slice(tc, {0}, {N/2+1});
//     auto t2 = make_tensor<RealType>({N+2});
//     (t2 = fft(tcs)).run(this->exec);
//     this->exec.sync();
//  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT1D1024PadBatchedR2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {  
    const index_t fft_dim = 4;
    using rtype = typename TestType::value_type;
    this->pb->template InitAndRunTVGenerator<rtype>(
        "00_transforms", "fft_operators", "rfft_1d_batched", {fft_dim, fft_dim, fft_dim*2});

    tensor_t<typename TestType::value_type, 2> av{{fft_dim, fft_dim}};
    tensor_t<TestType, 2> avo{{fft_dim, fft_dim + 1}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft(av, fft_dim*2)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_2d", {fft_dim, fft_dim});

    tensor_t<TestType, 2> av{{fft_dim, fft_dim}};
    tensor_t<TestType, 2> avo{{fft_dim, fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16x32C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim[] = {16, 32};
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_2d", {fft_dim[0], fft_dim[1]});

    tensor_t<TestType, 2> av{{fft_dim[0], fft_dim[1]}};
    tensor_t<TestType, 2> avo{{fft_dim[0], fft_dim[1]}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16BatchedC2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t batch_size = 10;
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_2d_batched",
        {batch_size, fft_dim, fft_dim});

    tensor_t<TestType, 3> av{{batch_size, fft_dim, fft_dim}};
    tensor_t<TestType, 3> avo{{batch_size, fft_dim, fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16BatchedStridedC2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t batch_size = 10;
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "fft_2d_batched_strided",
        {fft_dim, batch_size, fft_dim});

    tensor_t<TestType, 3> av{{fft_dim, batch_size, fft_dim}};
    tensor_t<TestType, 3> avo{{fft_dim, batch_size, fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    const int32_t axes[] = {0, 2};
    (avo = fft2(av, axes)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT2D16C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim = 16;
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_2d", {fft_dim, fft_dim});

    tensor_t<TestType, 2> av{{fft_dim, fft_dim}};
    tensor_t<TestType, 2> avo{{fft_dim, fft_dim}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = ifft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT2D16x32C2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;
  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {
    const index_t fft_dim[] = {16, 32};
    this->pb->template InitAndRunTVGenerator<TestType>(
        "00_transforms", "fft_operators", "ifft_2d", {fft_dim[0], fft_dim[1]});

    tensor_t<TestType, 2> av{{fft_dim[0], fft_dim[1]}};
    tensor_t<TestType, 2> avo{{fft_dim[0], fft_dim[1]}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = ifft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16R2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {  
    const index_t fft_dim = 16;
    using rtype = typename TestType::value_type;
    this->pb->template InitAndRunTVGenerator<rtype>(
        "00_transforms", "fft_operators", "rfft_2d", {fft_dim, fft_dim});

    tensor_t<rtype, 2> av{{fft_dim, fft_dim}};
    tensor_t<TestType, 2> avo{{fft_dim, fft_dim / 2 + 1}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

#if defined(MATX_EN_JIT) && defined(MATX_EN_MATHDX)
namespace {
using FFTJITComplex = cuda::std::complex<float>;

void FillFFT2JITInput2D(auto &in, index_t fft_dim)
{
  for (index_t row = 0; row < fft_dim; row++) {
    for (index_t col = 0; col < fft_dim; col++) {
      in(row, col) = FFTJITComplex{
        static_cast<float>((row + 1) * (col + 2) % 17),
        static_cast<float>((2 * row + col) % 11)};
    }
  }
}

void FillFFT2JITInput3D(auto &in, index_t batches, index_t fft_dim)
{
  for (index_t batch = 0; batch < batches; batch++) {
    for (index_t row = 0; row < fft_dim; row++) {
      for (index_t col = 0; col < fft_dim; col++) {
        in(batch, row, col) = FFTJITComplex{
          static_cast<float>((batch + 1) * (row + 2) + col),
          static_cast<float>((batch + row + 2 * col) % 13)};
      }
    }
  }
}

void AssertFFT2JITClose2D(auto &jit_out, auto &ref_out, index_t fft_dim, float tol)
{
  for (index_t row = 0; row < fft_dim; row++) {
    for (index_t col = 0; col < fft_dim; col++) {
      ASSERT_NEAR(jit_out(row, col).real(), ref_out(row, col).real(), tol);
      ASSERT_NEAR(jit_out(row, col).imag(), ref_out(row, col).imag(), tol);
    }
  }
}

void AssertFFT2JITClose3D(auto &jit_out, auto &ref_out, index_t batches, index_t fft_dim, float tol)
{
  for (index_t batch = 0; batch < batches; batch++) {
    for (index_t row = 0; row < fft_dim; row++) {
      for (index_t col = 0; col < fft_dim; col++) {
        ASSERT_NEAR(jit_out(batch, row, col).real(), ref_out(batch, row, col).real(), tol);
        ASSERT_NEAR(jit_out(batch, row, col).imag(), ref_out(batch, row, col).imag(), tol);
      }
    }
  }
}
} // namespace

TEST(FFTJIT, CuFFTDx2DFFT2Fusion)
{
  MATX_ENTER_HANDLER();

  constexpr index_t fft_dim = 4;

  auto in = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto jit_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto ref_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});

  FillFFT2JITInput2D(in, fft_dim);

  auto expr = fft2(in) + FFTJITComplex{1.0f, -1.0f};
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (jit_out = expr).run(jit_exec);
  (ref_out = fft2(in) + FFTJITComplex{1.0f, -1.0f}).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  AssertFFT2JITClose2D(jit_out, ref_out, fft_dim, 0.01f);

  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDx2DIFFT2Fusion)
{
  MATX_ENTER_HANDLER();

  constexpr index_t fft_dim = 4;

  auto in = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto jit_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto ref_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});

  FillFFT2JITInput2D(in, fft_dim);

  auto expr = ifft2(in, FFTNorm::BACKWARD) * FFTJITComplex{0.5f, 0.0f};
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (jit_out = expr).run(jit_exec);
  (ref_out = ifft2(in, FFTNorm::BACKWARD) * FFTJITComplex{0.5f, 0.0f}).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  AssertFFT2JITClose2D(jit_out, ref_out, fft_dim, 0.01f);

  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDx2DFFT2OrthoBoundary)
{
  MATX_ENTER_HANDLER();

  constexpr index_t fft_dim = 32;

  auto in = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto jit_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});
  auto ref_out = make_tensor<FFTJITComplex>({fft_dim, fft_dim});

  FillFFT2JITInput2D(in, fft_dim);

  auto expr = fft2(in, FFTNorm::ORTHO) - FFTJITComplex{0.25f, 0.75f};
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (jit_out = expr).run(jit_exec);
  (ref_out = fft2(in, FFTNorm::ORTHO) - FFTJITComplex{0.25f, 0.75f}).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  AssertFFT2JITClose2D(jit_out, ref_out, fft_dim, 0.05f);

  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDx2DFFT2BatchedForwardNorm)
{
  MATX_ENTER_HANDLER();

  constexpr index_t batches = 3;
  constexpr index_t fft_dim = 8;

  auto in = make_tensor<FFTJITComplex>({batches, fft_dim, fft_dim});
  auto jit_out = make_tensor<FFTJITComplex>({batches, fft_dim, fft_dim});
  auto ref_out = make_tensor<FFTJITComplex>({batches, fft_dim, fft_dim});

  FillFFT2JITInput3D(in, batches, fft_dim);

  auto expr = fft2(in, FFTNorm::FORWARD) + FFTJITComplex{0.125f, -0.5f};
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (jit_out = expr).run(jit_exec);
  (ref_out = fft2(in, FFTNorm::FORWARD) + FFTJITComplex{0.125f, -0.5f}).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  AssertFFT2JITClose3D(jit_out, ref_out, batches, fft_dim, 0.01f);

  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDxTensorAndOperatorInputsInOneExpression)
{
  MATX_ENTER_HANDLER();

  // FFTs of a tensor and of an operator expression with the same size can appear in the same expression
  constexpr index_t batches = 2;
  constexpr index_t fft_dim = 16;

  auto in = make_tensor<FFTJITComplex>({batches, fft_dim});
  auto jit_out = make_tensor<FFTJITComplex>({batches});
  auto ref_out = make_tensor<FFTJITComplex>({batches});

  for (index_t i = 0; i < batches; i++) {
    for (index_t j = 0; j < fft_dim; j++) {
      in(i, j) = FFTJITComplex{static_cast<float>(i + j), static_cast<float>(j % 3)};
    }
  }

  auto expr = sum(fft(in) + fft(in * FFTJITComplex{2.0f, 0.0f}), {1});
  if (!jit_supported(expr)) {
    GTEST_SKIP();
  }

  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  (jit_out = expr).run(jit_exec);
  (ref_out = sum(fft(in) + fft(in * FFTJITComplex{2.0f, 0.0f}), {1})).run(cuda_exec);
  jit_exec.sync();
  cuda_exec.sync();

  for (index_t i = 0; i < batches; i++) {
    ASSERT_NEAR(jit_out(i).real(), ref_out(i).real(), 0.01f);
    ASSERT_NEAR(jit_out(i).imag(), ref_out(i).imag(), 0.01f);
  }

  MATX_EXIT_HANDLER();
}
#endif

TYPED_TEST(FFTTestComplexTypes, FFT2D16x32R2C)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;

  if constexpr (!detail::CheckFFTSupport<ExecType, TestType>()) {
    GTEST_SKIP();
  } else {    
    const index_t fft_dim[] = {16, 32};
    using rtype = typename TestType::value_type;
    this->pb->template InitAndRunTVGenerator<rtype>(
        "00_transforms", "fft_operators", "rfft_2d", {fft_dim[0], fft_dim[1]});

    tensor_t<rtype, 2> av{{fft_dim[0], fft_dim[1]}};
    tensor_t<TestType, 2> avo{{fft_dim[0], fft_dim[1] / 2 + 1}};
    this->pb->NumpyToTensorView(av, "a_in");

    (avo = fft2(av)).run(this->exec);
    this->exec.sync();

    MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  }
  MATX_EXIT_HANDLER();
}

// Enable once C2R is supported
TYPED_TEST(FFTTestComplexNonHalfTypes, IRFFT2D16C2R)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  const index_t fft_dim = 16;
  using rtype = typename TestType::value_type;
  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "fft_operators", "irfft_2d", {fft_dim, 2 * (fft_dim - 1)});

  tensor_t<TestType, 2> av{{fft_dim, fft_dim}};
  tensor_t<rtype, 2> avo{{fft_dim, 2 * (fft_dim - 1)}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = irfft2(av)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, IRFFT2D16x32C2R)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  const index_t fft_dim[] = {16, 32};
  using rtype = typename TestType::value_type;
  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "fft_operators", "irfft_2d", {fft_dim[0], fft_dim[1]});

  tensor_t<TestType, 2> av{{fft_dim[0], fft_dim[1] / 2 + 1}};
  tensor_t<rtype, 2> avo{{fft_dim[0], fft_dim[1]}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = irfft2(av)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT1D1024C2CShort)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim - 16});

  tensor_t<TestType, 1> av{{fft_dim}};
  tensor_t<TestType, 1> avo{{fft_dim - 16}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av, fft_dim - 16)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

// A cached 1D plan is shared by every call with the same cuFFT parameters,
// including a contiguous call that batches all outer dims in one cuFFT call and
// a strided call that batches only dim R-2 and loops over the rest. Each case
// clears the plan caches, runs a contiguous call to cache the plan, and then
// runs a strided call that reuses it.
TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, FFT1DContiguousThenStridedBatch)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using RealType = typename inner_op_type_t<TestType>::type;

  const index_t fft_dim = 8;
  const index_t half_dim = fft_dim / 2 + 1;
  const auto n = static_cast<RealType>(fft_dim);

  // Checks every element along the last dim against expected(k), where k is the
  // index in the last dim. Complex outputs must have a zero imaginary part.
  auto check = [&](const auto &out, auto expected) {
    using OutType = typename std::decay_t<decltype(out)>::value_type;
    auto c = make_tensor<OutType>(out.Shape());
    (c = out).run(this->exec);
    this->exec.sync();
    const index_t len = c.Size(c.Rank() - 1);
    for (index_t idx = 0; idx < c.TotalSize(); idx++) {
      const double want = expected(idx % len);
      if constexpr (is_complex_v<OutType>) {
        ASSERT_NEAR(static_cast<double>(c.Data()[idx].real()), want, this->thresh) << "flat index " << idx;
        ASSERT_NEAR(static_cast<double>(c.Data()[idx].imag()), 0.0, this->thresh) << "flat index " << idx;
      }
      else {
        ASSERT_NEAR(static_cast<double>(c.Data()[idx]), want, this->thresh) << "flat index " << idx;
      }
    }
  };
  // Forward transform of all ones: bin 0 is fft_dim, every other bin is 0
  const auto fwd_ones = [&](index_t k) { return k == 0 ? static_cast<double>(fft_dim) : 0.0; };
  // Inverse transform of all ones: bin 0 is 1, every other bin is 0
  const auto inv_ones = [&](index_t k) { return k == 0 ? 1.0 : 0.0; };
  // Inverse transform of a spectrum with only bin 0 set to fft_dim: all ones
  const auto all_ones = [&](index_t) { return 1.0; };

  // Fills a complex half-spectrum tensor with fft_dim in bin 0 and 0 elsewhere
  auto make_dc_spectrum = [&](auto shape) {
    auto t = make_tensor<TestType>(shape);
    (t = zeros<TestType>(t.Shape())).run(this->exec);
    auto bin0 = slice(t, {0, 0, 0}, {matxEnd, matxEnd, 1});
    (bin0 = ones<TestType>(bin0.Shape()) * n).run(this->exec);
    return t;
  };

  // C2C forward, rank 3, strided input: {4,2,8} view of a {4,3,8} tensor
  {
    ClearCaches();
    auto ac = make_tensor<TestType>({1, 2, fft_dim});
    auto oc = make_tensor<TestType>({1, 2, fft_dim});
    (ac = ones<TestType>(ac.Shape())).run(this->exec);
    (oc = fft(ac)).run(this->exec);
    check(oc, fwd_ones);

    auto a_full = make_tensor<TestType>({4, 3, fft_dim});
    (a_full = ones<TestType>(a_full.Shape())).run(this->exec);
    auto a = slice(a_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
    ASSERT_FALSE(a.IsContiguous());
    auto o = make_tensor<TestType>({4, 2, fft_dim});
    (o = zeros<TestType>(o.Shape())).run(this->exec);
    (o = fft(a)).run(this->exec);
    check(o, fwd_ones);
  }

  // C2C forward, rank 3, strided output: {4,2,8} view of a {4,3,8} tensor
  {
    ClearCaches();
    auto ac = make_tensor<TestType>({1, 2, fft_dim});
    auto oc = make_tensor<TestType>({1, 2, fft_dim});
    (ac = ones<TestType>(ac.Shape())).run(this->exec);
    (oc = fft(ac)).run(this->exec);
    check(oc, fwd_ones);

    auto a = make_tensor<TestType>({4, 2, fft_dim});
    (a = ones<TestType>(a.Shape())).run(this->exec);
    auto o_full = make_tensor<TestType>({4, 3, fft_dim});
    (o_full = zeros<TestType>(o_full.Shape())).run(this->exec);
    auto o = slice(o_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
    ASSERT_FALSE(o.IsContiguous());
    (o = fft(a)).run(this->exec);
    check(o, fwd_ones);
  }

  // C2C forward, rank 4, strided input with two looped outer dims: {3,2,2,8} view of a {3,2,3,8} tensor
  {
    ClearCaches();
    auto ac = make_tensor<TestType>({1, 1, 2, fft_dim});
    auto oc = make_tensor<TestType>({1, 1, 2, fft_dim});
    (ac = ones<TestType>(ac.Shape())).run(this->exec);
    (oc = fft(ac)).run(this->exec);
    check(oc, fwd_ones);

    auto a_full = make_tensor<TestType>({3, 2, 3, fft_dim});
    (a_full = ones<TestType>(a_full.Shape())).run(this->exec);
    auto a = slice(a_full, {0, 0, 0, 0}, {matxEnd, matxEnd, 2, matxEnd});
    ASSERT_FALSE(a.IsContiguous());
    auto o = make_tensor<TestType>({3, 2, 2, fft_dim});
    (o = zeros<TestType>(o.Shape())).run(this->exec);
    (o = fft(a)).run(this->exec);
    check(o, fwd_ones);
  }

  // C2C inverse, rank 3, strided input: {4,2,8} view of a {4,3,8} tensor
  {
    ClearCaches();
    auto ac = make_tensor<TestType>({1, 2, fft_dim});
    auto oc = make_tensor<TestType>({1, 2, fft_dim});
    (ac = ones<TestType>(ac.Shape())).run(this->exec);
    (oc = ifft(ac)).run(this->exec);
    check(oc, inv_ones);

    auto a_full = make_tensor<TestType>({4, 3, fft_dim});
    (a_full = ones<TestType>(a_full.Shape())).run(this->exec);
    auto a = slice(a_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
    ASSERT_FALSE(a.IsContiguous());
    auto o = make_tensor<TestType>({4, 2, fft_dim});
    (o = zeros<TestType>(o.Shape())).run(this->exec);
    (o = ifft(a)).run(this->exec);
    check(o, inv_ones);
  }

  // R2C, rank 3, strided input: {4,2,8} view of a real {4,3,8} tensor
  {
    ClearCaches();
    auto ac = make_tensor<RealType>({1, 2, fft_dim});
    auto oc = make_tensor<TestType>({1, 2, half_dim});
    (ac = ones<RealType>(ac.Shape())).run(this->exec);
    (oc = fft(ac)).run(this->exec);
    check(oc, fwd_ones);

    auto a_full = make_tensor<RealType>({4, 3, fft_dim});
    (a_full = ones<RealType>(a_full.Shape())).run(this->exec);
    auto a = slice(a_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
    ASSERT_FALSE(a.IsContiguous());
    auto o = make_tensor<TestType>({4, 2, half_dim});
    (o = zeros<TestType>(o.Shape())).run(this->exec);
    (o = fft(a)).run(this->exec);
    check(o, fwd_ones);
  }

  // C2R, rank 3, strided input: {4,2,5} view of a complex {4,3,5} tensor
  {
    ClearCaches();
    auto ac = make_dc_spectrum(cuda::std::array<index_t, 3>{1, 2, half_dim});
    auto oc = make_tensor<RealType>({1, 2, fft_dim});
    (oc = irfft(ac)).run(this->exec);
    check(oc, all_ones);

    auto a_full = make_dc_spectrum(cuda::std::array<index_t, 3>{4, 3, half_dim});
    auto a = slice(a_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
    ASSERT_FALSE(a.IsContiguous());
    auto o = make_tensor<RealType>({4, 2, fft_dim});
    (o = zeros<RealType>(o.Shape())).run(this->exec);
    (o = irfft(a)).run(this->exec);
    check(o, all_ones);
  }
  MATX_EXIT_HANDLER();
}

// The cuFFT plan for a strided input describes only dimension R-2, so the
// output's other batch dimensions are indexed with the input's. Mismatched
// batch dimensions must be rejected on every call, including when the plan is
// already cached. The public fft() operator checks shapes before reaching the
// plan, so this calls the transform directly.
TEST(FFTCUDA, FFT1DMismatchedBatchDimsThrow)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::complex<float>;
  const index_t fft_dim = 8;
  cudaExecutor exec{};

  ClearCaches();
  auto a_full = make_tensor<TestType>({4, 3, fft_dim});
  (a_full = ones<TestType>(a_full.Shape())).run(exec);
  auto a = slice(a_full, {0, 0, 0}, {matxEnd, 2, matxEnd});
  auto o = make_tensor<TestType>({2, 2, fft_dim});

  // Caches a plan with the same cuFFT parameters as the mismatched call below
  auto a_ok = slice(a_full, {0, 0, 0}, {2, 2, matxEnd});
  detail::fft_impl(o, a_ok, 0, FFTNorm::BACKWARD, exec);
  exec.sync();

  EXPECT_THROW(detail::fft_impl(o, a, 0, FFTNorm::BACKWARD, exec), matx::detail::matxException);

  // Contiguous input with more batch rows than the output
  auto a_contig = make_tensor<TestType>({4, 2, fft_dim});
  EXPECT_THROW(detail::fft_impl(o, a_contig, 0, FFTNorm::BACKWARD, exec), matx::detail::matxException);
  exec.sync();
  MATX_EXIT_HANDLER();
}

// 2D variant of FFT1DMismatchedBatchDimsThrow. The 2D plan batches dimension
// R-3 using the input's size, so a smaller output batch must be rejected even
// when the plan is already cached.
TEST(FFTCUDA, FFT2DMismatchedBatchDimsThrow)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::complex<float>;
  const index_t fft_dim = 8;
  cudaExecutor exec{};

  ClearCaches();
  auto a = make_tensor<TestType>({4, fft_dim, fft_dim});
  (a = ones<TestType>(a.Shape())).run(exec);

  // Caches a plan with the same cuFFT parameters as the mismatched call below
  auto o_ok = make_tensor<TestType>({4, fft_dim, fft_dim});
  detail::fft2_impl(o_ok, a, FFTNorm::BACKWARD, exec);
  exec.sync();

  auto o = make_tensor<TestType>({2, fft_dim, fft_dim});
  EXPECT_THROW(detail::fft2_impl(o, a, FFTNorm::BACKWARD, exec), matx::detail::matxException);
  exec.sync();
  MATX_EXIT_HANDLER();
}

// The 2D size checks used to compare element counts only, and only in Debug
// builds. These outputs have the right element count but the wrong shape.
TEST(FFTCUDA, FFT2DMismatchedSizesThrow)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 8;
  cudaExecutor exec{};

  ClearCaches();
  auto c = make_tensor<cuda::std::complex<float>>({fft_dim, fft_dim});
  auto r = make_tensor<float>({fft_dim, fft_dim});
  (c = ones<cuda::std::complex<float>>(c.Shape())).run(exec);
  (r = ones<float>(r.Shape())).run(exec);

  // C2C: 16x4 output for an 8x8 input
  auto c_out = make_tensor<cuda::std::complex<float>>({2 * fft_dim, fft_dim / 2});
  EXPECT_THROW(detail::fft2_impl(c_out, c, FFTNorm::BACKWARD, exec), matx::detail::matxException);

  // R2C: 4x10 output for an 8x8 input (expected 8x5)
  auto r2c_out = make_tensor<cuda::std::complex<float>>({fft_dim / 2, 10});
  EXPECT_THROW(detail::fft2_impl(r2c_out, r, FFTNorm::BACKWARD, exec), matx::detail::matxException);

  // C2R: 8x8 output from a 4x10 input (expected 8x5)
  auto c2r_in = make_tensor<cuda::std::complex<float>>({fft_dim / 2, 10});
  EXPECT_THROW(detail::ifft2_impl(r, c2r_in, FFTNorm::BACKWARD, exec), matx::detail::matxException);
  exec.sync();
  MATX_EXIT_HANDLER();
}

// Cloned views give a batch distance of 0, which cuFFT rejects with a generic
// error. MatX should throw its own error in every build type, for both inputs
// and outputs.
TEST(FFTCUDA, FFT1DClonedBatchThrows)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::complex<float>;
  const index_t fft_dim = 8;
  const index_t batch = 4;
  cudaExecutor exec{};

  auto row = make_tensor<TestType>({fft_dim});
  (row = ones<TestType>(row.Shape())).run(exec);
  auto dense = make_tensor<TestType>({batch, fft_dim});
  (dense = ones<TestType>(dense.Shape())).run(exec);

  auto cloned_in = row.Clone<2>({batch, matxKeepDim});
  ASSERT_EQ(cloned_in.Stride(0), 0);
  auto out = make_tensor<TestType>({batch, fft_dim});
  EXPECT_THROW(detail::fft_impl(out, cloned_in, 0, FFTNorm::BACKWARD, exec), matx::detail::matxException);

  auto cloned_out = row.Clone<2>({batch, matxKeepDim});
  EXPECT_THROW(detail::fft_impl(cloned_out, dense, 0, FFTNorm::BACKWARD, exec), matx::detail::matxException);
  exec.sync();
  MATX_EXIT_HANDLER();
}

// Runs the same FFT plan on two streams and checks every result against a
// single-stream reference.
TEST(FFTCUDA, MultiStreamFFT1D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::complex<float>;

  // Prime length so the FFT uses cuFFT's non-power-of-2 (Bluestein) path
  const index_t fft_dim = 4099;
  const index_t batch = 64;
  const int iters = 8;

  cudaStream_t s1, s2;
  ASSERT_EQ(cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking), cudaSuccess);
  cudaExecutor e1{s1};
  cudaExecutor e2{s2};

  auto a1 = make_tensor<TestType>({batch, fft_dim});
  auto a2 = make_tensor<TestType>({batch, fft_dim});
  auto r1 = make_tensor<TestType>({batch, fft_dim});
  auto r2 = make_tensor<TestType>({batch, fft_dim});
  (a1 = random<TestType>({batch, fft_dim}, NORMAL, 1)).run(e1);
  (a2 = random<TestType>({batch, fft_dim}, NORMAL, 2)).run(e1);

  // References computed sequentially on a single stream
  (r1 = fft(a1)).run(e1);
  (r2 = fft(a2)).run(e1);
  e1.sync();

  auto o1 = make_tensor<TestType>({iters, batch, fft_dim});
  auto o2 = make_tensor<TestType>({iters, batch, fft_dim});
  for (int it = 0; it < iters; it++) {
    auto o1_it = slice<2>(o1, {it, 0, 0}, {matxDropDim, matxEnd, matxEnd});
    auto o2_it = slice<2>(o2, {it, 0, 0}, {matxDropDim, matxEnd, matxEnd});
    (o1_it = fft(a1)).run(e1);
    (o2_it = fft(a2)).run(e2);
  }
  e1.sync();
  e2.sync();

  // Check on the default stream so that no other cached transform holds memory
  // allocated on s1 or s2
  cudaExecutor e0{};
  auto err = make_tensor<float>({});
  (err = max(abs(o1 - clone<3>(r1, {iters, matxKeepDim, matxKeepDim})))).run(e0);
  e0.sync();
  EXPECT_LT(err(), 1e-2f) << "stream 1 results differ from the single-stream reference";
  (err = max(abs(o2 - clone<3>(r2, {iters, matxKeepDim, matxKeepDim})))).run(e0);
  e0.sync();
  EXPECT_LT(err(), 1e-2f) << "stream 2 results differ from the single-stream reference";

  // The cache holds the cuFFT workspaces allocated on s1 and s2. ClearCaches()
  // must not free them on the destroyed streams.
  ASSERT_EQ(cudaStreamDestroy(s1), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(s2), cudaSuccess);
  ClearCaches();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  MATX_EXIT_HANDLER();
}

#if CUDA_VERSION >= 12050
// Runs an FFT on a green-context stream. The plan cache records the stream's
// context for its cuFFT workspace, which cuStreamGetCtx rejects for
// green-context streams on 12.x drivers.
TEST(FFTCUDA, GreenContextStreamFFT1D)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::complex<float>;
  int driver_version = 0;
  ASSERT_EQ(cuDriverGetVersion(&driver_version), CUDA_SUCCESS);
  if (driver_version < 12050) {
    GTEST_SKIP() << "cuGreenCtxStreamCreate requires a 12.5 or newer driver";
  }

  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  // Green contexts are unavailable in some configurations, such as under MPS
  CUdevResource sm_resource;
  CUdevResourceDesc desc;
  if (cuDeviceGetDevResource(device, &sm_resource, CU_DEV_RESOURCE_TYPE_SM) != CUDA_SUCCESS ||
      cuDevResourceGenerateDesc(&desc, &sm_resource, 1) != CUDA_SUCCESS) {
    GTEST_SKIP() << "Green context resources are not supported";
  }
  CUgreenCtx green_ctx;
  if (cuGreenCtxCreate(&green_ctx, desc, device, CU_GREEN_CTX_DEFAULT_STREAM) != CUDA_SUCCESS) {
    GTEST_SKIP() << "Green contexts are not supported";
  }
  CUstream green_stream;
  if (cuGreenCtxStreamCreate(&green_stream, green_ctx, CU_STREAM_NON_BLOCKING, 0) != CUDA_SUCCESS) {
    EXPECT_EQ(cuGreenCtxDestroy(green_ctx), CUDA_SUCCESS);
    GTEST_SKIP() << "Green context streams are not supported";
  }
  // Destroy the stream and context however the test exits
  struct GreenContextGuard {
    CUstream stream;
    CUgreenCtx ctx;
    ~GreenContextGuard() {
      ClearCaches();
      EXPECT_EQ(cuStreamDestroy(stream), CUDA_SUCCESS);
      EXPECT_EQ(cuGreenCtxDestroy(ctx), CUDA_SUCCESS);
    }
  } guard{green_stream, green_ctx};

  const index_t fft_dim = 4099;
  const index_t batch = 16;
  cudaExecutor e0{};
  cudaExecutor eg{reinterpret_cast<cudaStream_t>(green_stream)};
  auto a = make_tensor<TestType>({batch, fft_dim});
  auto ref = make_tensor<TestType>({batch, fft_dim});
  auto out = make_tensor<TestType>({batch, fft_dim});
  (a = random<TestType>({batch, fft_dim}, NORMAL, 1)).run(e0);
  (ref = fft(a)).run(e0);
  e0.sync();

  (out = fft(a)).run(eg);
  eg.sync();
  auto err = make_tensor<float>({});
  (err = max(abs(out - ref))).run(e0);
  e0.sync();
  EXPECT_LT(err(), 1e-2f) << "green-context stream results differ from the default-stream reference";

  // Free the workspace allocated on the green-context stream
  ClearCaches();
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  MATX_EXIT_HANDLER();
}
#endif

TYPED_TEST(FFTTestComplexNonHalfTypesAllExecs, IFFT1D1024C2CShort)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
 
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TestType>(
      "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim - 16});
  tensor_t<TestType, 1> av{{fft_dim}};
  tensor_t<TestType, 1> avo{{fft_dim - 16}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av, fft_dim - 16)).run(this->exec);
  this->exec.sync();

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}
