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

  void TearDown() { pb.reset(); }

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

  fft(avo, av);
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

  fft(out1, in);
  fft(out2, in, {2});
  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  fft(out1.Permute({0,2,1}), in.Permute({0,2,1}));
  fft(out2, in, {1});
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  ifft(out1, in);
  ifft(out2, in, {2});
  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  ifft(out1.Permute({0,2,1}), in.Permute({0,2,1}));
  ifft(out2, in, {1});
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
    fft(out1.Permute({0,2,1}), permute(in1, {0,2,1}));
    fft(out2, in1, {1});

    cudaStreamSynchronize(0);

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }

    ifft(out1.Permute({1,2,0}), permute(in1, {1,2,0}));
    ifft(out2, in1, {0});
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

  fft2(out1, in);
  fft2(out2, in, {1,2});
  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  fft2(out1.Permute({1,2,0}), in.Permute({1,2,0}));
  fft2(out2, in, {2,0});
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  fft2(out1.Permute({1,0,2}), in.Permute({1,0,2}));
  fft2(out2, in, {0,2});
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }
  
  ifft2(out1, in);
  ifft2(out2, in, {1,2});
  cudaStreamSynchronize(0);

  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  ifft2(out1.Permute({1,2,0}), in.Permute({1,2,0}));
  ifft2(out2, in, {2,0});
  cudaStreamSynchronize(0);
  
  for(int i = 0; i < d1; i++) {
    for(int j = 0; j < d2; j++) {
      for(int k = 0; k < d3; k++) {
        ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
      }
    }
  }

  ifft2(out1.Permute({1,0,2}), in.Permute({1,0,2}));
  ifft2(out2, in, {0,2});
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

    fft2(out1.Permute({1,0,2}), permute(in1, {1,0,2}));
    fft2(out2, in1, {0,2});
    cudaStreamSynchronize(0);

    for(int i = 0; i < d1; i++) {
      for(int j = 0; j < d2; j++) {
        for(int k = 0; k < d3; k++) {
          ASSERT_EQ(out1(i,j,k), out2(i,j,k)); 
        }
      }
    }
    

    ifft2(out1.Permute({1,0,2}), permute(in1, {1,0,2}));
    ifft2(out2, in1, {0,2});
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

  ifft(avo, av);
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
  tensor_t<TypeParam, 1> av{{fft_dim}};
  tensor_t<TypeParam, 1> avo{{fft_dim * 2}};
  this->pb->NumpyToTensorView(av, "a_in");

  fft(avo, av);
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

  fft(avo, av);
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);

  fft(avo, av, fft_dim * 2); // Force the FFT size
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

  ifft(avo, av);
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

  fft(avo, av);
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

  fft(avo, av);
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

  fft(avo, av);
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

  fft2(avo, av);
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

  ifft2(avo, av);
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

  fft(avo, av);
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

  ifft(avo, av);
  cudaStreamSynchronize(0);

  MATX_TEST_ASSERT_COMPARE(this->pb, avo, "a_out", this->thresh);
  MATX_EXIT_HANDLER();
}
