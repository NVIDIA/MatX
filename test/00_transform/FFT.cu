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
void AssertFFTJITClose(const auto &actual, const auto &reference, double tolerance)
{
  for (index_t i = 0; i < TotalSize(reference); ++i) {
    SCOPED_TRACE(i);
    const auto expected = reference.Data()[i];
    ASSERT_NEAR(actual.Data()[i].real(), expected.real(), tolerance * (1.0 + cuda::std::abs(expected.real())));
    ASSERT_NEAR(actual.Data()[i].imag(), expected.imag(), tolerance * (1.0 + cuda::std::abs(expected.imag())));
  }
}

template <typename Complex, int Rank>
void CheckFusedFFTConjugatedModel()
{
  const detail::DeviceAttributes *attrs = nullptr;
  if (detail::TryGetCurrentDeviceAttributes(attrs) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device required";
  }
  using Scalar = typename Complex::value_type;
  const double tolerance = cuda::std::is_same_v<Scalar, float> ? 5e-4 : 1e-10;
  CUDAJITExecutor jit_exec{};
  cudaExecutor cuda_exec{};
  for (index_t length : {32, 512}) {
    for (index_t batches : {1, 3}) {
      SCOPED_TRACE(::testing::Message() << "rank " << Rank << " batches " << batches << " length " << length);
      cuda::std::array<index_t, Rank> shape{};
      shape.fill(2);
      shape[Rank - 2] = batches;
      shape[Rank - 1] = length;
      auto input = make_tensor<Complex>(shape);
      auto model = make_tensor<Complex>({length});
      auto output = make_tensor<Complex>(shape);
      auto reference = make_tensor<Complex>(shape);
      for (index_t i = 0; i < TotalSize(input); ++i) {
        input.Data()[i] = Complex{static_cast<Scalar>((i * 7) % 23) / Scalar{23},
                                  static_cast<Scalar>((i * 3) % 17) / Scalar{17}};
      }
      for (index_t i = 0; i < length; ++i) {
        model(i) = Complex{static_cast<Scalar>((i * 5) % 19) / Scalar{19},
                           static_cast<Scalar>((i * 11) % 13) / Scalar{13}};
      }

      // Only an unsupported backend may skip the test. A false result for the
      // complete assignment is the regression this test must catch.
      if (!detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(fft(input)) ||
          !detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(ifft(input))) {
        GTEST_SKIP() << "cuFFTDx does not support this FFT configuration";
      }
      auto model_shape = shape;
      model_shape[Rank - 1] = matxKeepDim;
      auto model_matrix = clone<Rank>(model, model_shape);
      auto assignment = (output = ifft(fft(input) * conj(model_matrix)));
      for (int launch = 0; launch < 2; ++launch) {
        SCOPED_TRACE(launch);
        ASSERT_TRUE(jit_supported(assignment));
        assignment.run(jit_exec);
        // Use a separate expression so JIT launch-parameter updates cannot
        // affect the trusted non-JIT reference path.
        (reference = ifft(fft(input) * conj(model_matrix))).run(cuda_exec);
        jit_exec.sync();
        cuda_exec.sync();
        AssertFFTJITClose(output, reference, tolerance);
        for (index_t i = 0; i < TotalSize(input); ++i) {
          input.Data()[i] += Complex{Scalar{0.125}, Scalar{-0.0625}};
        }
      }
    }
  }
}
} // namespace

TEST(FFTJIT, CuFFTDxFusedFFTConjugatedModelFloat)
{
  MATX_ENTER_HANDLER();
  CheckFusedFFTConjugatedModel<cuda::std::complex<float>, 2>();
  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDxFusedFFTConjugatedModelDouble)
{
  MATX_ENTER_HANDLER();
  CheckFusedFFTConjugatedModel<cuda::std::complex<double>, 2>();
  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDxFusedFFTConjugatedModelRank3Float)
{
  MATX_ENTER_HANDLER();
  CheckFusedFFTConjugatedModel<cuda::std::complex<float>, 3>();
  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, CuFFTDxFusedFFTConjugatedModelRank3Double)
{
  MATX_ENTER_HANDLER();
  CheckFusedFFTConjugatedModel<cuda::std::complex<double>, 3>();
  MATX_EXIT_HANDLER();
}

TEST(FFTJIT, IncompatibleEPTUsesExplicitCUDAFallback)
{
  MATX_ENTER_HANDLER();
  const detail::DeviceAttributes *attrs = nullptr;
  if (detail::TryGetCurrentDeviceAttributes(attrs) != cudaSuccess) {
    cudaGetLastError();
    GTEST_SKIP() << "CUDA device required";
  }
  // More than 1024 elements guarantees the block FFT needs multiple EPT.
  constexpr index_t length = 2048;
  auto input = make_tensor<FFTJITComplex>({1, length});
  auto storage = make_tensor<FFTJITComplex>({2 * length});
  auto model = make_tensor<FFTJITComplex>(storage.Data(), {length}, {2});
  const index_t model_shape[] = {1, matxKeepDim};
  auto model_matrix = clone<2>(model, model_shape);
  auto output = make_tensor<FFTJITComplex>({1, length});
  auto reference = make_tensor<FFTJITComplex>({1, length});
  auto spectrum = make_tensor<FFTJITComplex>({1, length});
  for (index_t i = 0; i < length; ++i) {
    input(0, i) = FFTJITComplex{static_cast<float>(i % 23) / 23.0f, static_cast<float>(i % 17) / 17.0f};
    model(i) = FFTJITComplex{static_cast<float>(i % 19) / 19.0f, static_cast<float>(i % 13) / 13.0f};
  }
  if (!detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(fft(input)) ||
      !detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(ifft(input))) {
    GTEST_SKIP() << "cuFFTDx does not support this FFT configuration";
  }
  cudaExecutor cuda_exec{};
  auto assignment = (output = ifft(fft(input) * conj(model_matrix)));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(assignment));
  ASSERT_FALSE(jit_supported(assignment));
  assignment.run(cuda_exec);
  (spectrum = fft(input)).run(cuda_exec);
  (spectrum = spectrum * conj(model_matrix)).run(cuda_exec);
  (reference = ifft(spectrum)).run(cuda_exec);
  cuda_exec.sync();
  AssertFFTJITClose(output, reference, 5e-4);

  auto shifted = (output = fftshift1D(fft(input)));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(shifted));
  ASSERT_FALSE(jit_supported(shifted));
  shifted.run(cuda_exec);
  (spectrum = fft(input)).run(cuda_exec);
  (reference = fftshift1D(spectrum)).run(cuda_exec);
  cuda_exec.sync();
  AssertFFTJITClose(output, reference, 5e-4);
  MATX_EXIT_HANDLER();
}

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
