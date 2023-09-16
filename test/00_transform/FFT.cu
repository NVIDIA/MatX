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
  void SetUp() override
  {
    CheckTestTypeSupport<T>();

    pb = std::make_unique<detail::MatXPybind>();

    // Half precision needs a bit more tolerance when compared to fp32
    if constexpr (is_complex_half_v<T>) {
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

TYPED_TEST_SUITE(FFTTestComplexTypes, MatXComplexTypes);
TYPED_TEST_SUITE(FFTTestComplexNonHalfTypes, MatXComplexNonHalfTypes);

TYPED_TEST(FFTTestComplexTypes, FFT1D1024C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim});

  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  // example-begin fft-1
  // Perform a 1D FFT from input av into output avo. Input and output sizes will be deduced by the
  // type of the tensors and output size.
  (avo = fft(av)).run();
  // example-end fft-1
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1DFWD1024C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d_fwd", {fft_dim, fft_dim});

  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  // example-begin fft-1-fwd
  // Perform a 1D FFT from input av into output avo with FORWARD scaling (1/N). Input and output sizes will be deduced by the
  // type of the tensors and output size.
  (avo = fft(av, fft_dim, FFTNorm::FORWARD)).run();
  // example-end fft-1
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1DORTHO1024C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d_ortho", {fft_dim, fft_dim});

  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  // example-begin fft-1
  // Perform a 1D FFT from input av into output avo with ORTHO scaling (1/sqrt(N)). Input and output sizes will be deduced by the
  // type of the tensors and output size.
  (avo = fft(av, fft_dim, FFTNorm::ORTHO)).run();
  // example-end fft-1
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, FFT1Axis)
{
  MATX_ENTER_HANDLER();
  const int d1 = 8;
  const int d2 = 512;
  const int d3 = 1024;

  // example-begin fft-2
  auto in = make_tensor<TypeParam>({d1, d2, d3});
  auto out1 = make_tensor<TypeParam>({d1, d2, d3});
  auto out2 = make_tensor<TypeParam>({d1, d2, d3});

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        in(i,j,k) = static_cast<TypeParam>((float)(i+j+k));
      }
    }
  }

  // Perform a batched 1D FFT on a 3D tensor across axis 2. Since axis 2 is the last dimension,
  // this is equivalent to not specifying the axis
  (out1 = fft(in)).run();
  (out2 = fft(in, {2})).run();

  // example-end fft-2
  cudaStreamSynchronize(0);

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
  (out1.Permute({0,2,1}) = fft(in.Permute({0,2,1}))).run();
  (out2 = fft(in, {1})).run();  

  // example-end fft-3
  cudaStreamSynchronize(0);
  
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
  (out1 = ifft(in)).run();
  (out2 = ifft(in, {2})).run();    
  // example-end ifft-1
  cudaStreamSynchronize(0);

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
  (out1.Permute({0,2,1}) = ifft(in.Permute({0,2,1}))).run();
  (out2 = ifft(in, {1})).run();    
  // example-end ifft-2
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  {
    auto in1 =  ones<TypeParam>(in.Shape());
    (out1.Permute({0,2,1}) = fft(permute(in1, {0,2,1}))).run();
    (out2 = fft(in1, {1})).run();

    cudaStreamSynchronize(0);

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }


    (out1.Permute({1,2,0}) = ifft(permute(in1, {1,2,0}))).run();
    (out2 = ifft(in1, {0})).run();
    cudaStreamSynchronize(0);

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

TYPED_TEST(FFTTestComplexNonHalfTypes, FFT2Axis)
{
  MATX_ENTER_HANDLER();
  const int d1 = 128;
  const int d2 = 256;
  const int d3 = 512;

  auto in = make_tensor<TypeParam>({d1, d2, d3});
  auto out1 = make_tensor<TypeParam>({d1, d2, d3});
  auto out2 = make_tensor<TypeParam>({d1, d2, d3});

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
  (out1 = fft2(in)).run();
  (out2 = fft2(in, {1,2})).run();
  // example-end fft2-1
  cudaStreamSynchronize(0);

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
  (out1.Permute({1,2,0}) = fft2(in.Permute({1,2,0}))).run();
  (out2 = fft2(in, {2,0})).run();
  // example-end fft2-2
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  (out1.Permute({1,0,2}) = fft2(in.Permute({1,0,2}))).run();
  (out2 = fft2(in, {0,2})).run();
  cudaStreamSynchronize(0);
  
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
  (out1 = ifft2(in)).run();
  (out2 = ifft2(in, {1,2})).run();
  // example-end ifft2-1
  cudaStreamSynchronize(0);

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
  (out1.Permute({1,2,0}) = ifft2(in.Permute({1,2,0}))).run();
  (out2 = ifft2(in, {2,0})).run();
  // example-end ifft2-2
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  (out1.Permute({1,0,2}) = ifft2(in.Permute({1,0,2}))).run();
  (out2 = ifft2(in, {0,2})).run();
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
 
  {
    auto in1 =  ones<TypeParam>(in.Shape());

    (out1.Permute({1,0,2}) = fft2(permute(in1, {1,0,2}))).run();
    (out2 = fft2(in1, {0,2})).run();
    cudaStreamSynchronize(0);

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }
    

    (out1.Permute({1,0,2}) = ifft2(permute(in1, {1,0,2}))).run();
    (out2 = ifft2(in1, {0,2})).run();
    cudaStreamSynchronize(0);

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
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim});
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1DORTHO1024C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_1d_ortho", {fft_dim, fft_dim});
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av, fft_dim, FFTNorm::ORTHO)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1DFWD1024C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_1d_fwd", {fft_dim, fft_dim});
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av, fft_dim, FFTNorm::FORWARD)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024PadC2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim * 2});
  // example-begin fft-4
  auto av = make_tensor<TypeParam>({fft_dim});
  auto avo = make_tensor<TypeParam>({fft_dim * 2});
  this->pb->NumpyToTensorView(av, "a_in");

  // Perform an FFT on input av into output avo. Since avo is bigger than av, av will be zero-padded
  // to the appropriate size
  (avo = fft(av)).run();
  // example-end fft-4
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT1D1024PadBatchedC2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 4;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d_batched", {fft_dim+1, fft_dim+2, fft_dim*2});
  tensor_t<TypeParam, 2> av{{fft_dim + 1, fft_dim + 2}};
  tensor_t<TypeParam, 2> avo{{fft_dim + 1, fft_dim * 2}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);

  // example-begin fft-5
  // Perform an FFT but force the size to be fft_dim * 2 instead of the output size
  (avo = fft(av, fft_dim * 2)).run(); // Force the FFT size
  // example-end fft-5
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);  
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT1D1024PadC2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim * 2});
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim * 2}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, FFT1D1024R2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  using rtype = typename TypeParam::value_type;
  this->pb->template InitAndRunTVGenerator<rtype>(
      "00_transforms", "fft_operators", "rfft_1d", {fft_dim, fft_dim});

  tensor_t<typename TypeParam::value_type, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim / 2 + 1}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, FFT1D1024PadR2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 4;
  using rtype = typename TypeParam::value_type;
  this->pb->template InitAndRunTVGenerator<rtype>(
      "00_transforms", "fft_operators", "rfft_1d", {fft_dim, fft_dim*2});

  tensor_t<typename TypeParam::value_type, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim + 1}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, FFT1D1024PadBatchedR2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 4;
  using rtype = typename TypeParam::value_type;
  this->pb->template InitAndRunTVGenerator<rtype>(
      "00_transforms", "fft_operators", "rfft_1d_batched", {fft_dim, fft_dim, fft_dim*2});

  tensor_t<typename TypeParam::value_type, 2> av{{fft_dim, fft_dim}};
  tensor_t<TypeParam, 2> avo{{fft_dim, fft_dim + 1}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, FFT2D16C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 16;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_2d", {fft_dim, fft_dim});

  tensor_t<TypeParam, 2> av{{fft_dim, fft_dim}};
  tensor_t<TypeParam, 2> avo{{fft_dim, fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft2(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexTypes, IFFT2D16C2C)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 16;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_2d", {fft_dim, fft_dim});

  tensor_t<TypeParam, 2> av{{fft_dim, fft_dim}};
  tensor_t<TypeParam, 2> avo{{fft_dim, fft_dim}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft2(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}


TYPED_TEST(FFTTestComplexNonHalfTypes, FFT1D1024C2CShort)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "fft_1d", {fft_dim, fft_dim - 16});

  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim - 16}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = fft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FFTTestComplexNonHalfTypes, IFFT1D1024C2CShort)
{
  MATX_ENTER_HANDLER();
  const index_t fft_dim = 1024;
  this->pb->template InitAndRunTVGenerator<TypeParam>(
      "00_transforms", "fft_operators", "ifft_1d", {fft_dim, fft_dim - 16});
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim - 16}};
  this->pb->NumpyToTensorView(av, "a_in");

  (avo = ifft(av)).run();
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}
