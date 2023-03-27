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

template <typename TensorType> struct BasicTensorTestsData {
  tensor_t<TensorType, 0> t0{};
  tensor_t<TensorType, 1> t1{{10}};
  tensor_t<TensorType, 2> t2{{20, 10}};
  tensor_t<TensorType, 3> t3{{30, 20, 10}};
  tensor_t<TensorType, 4> t4{{40, 30, 20, 10}};

  tensor_t<TensorType, 2> t2s = t2.Permute({1, 0});
  tensor_t<TensorType, 3> t3s = t3.Permute({2, 1, 0});
  tensor_t<TensorType, 4> t4s = t4.Permute({3, 2, 1, 0});
};

template <typename TensorType>
class BasicTensorTestsComplex : public ::testing::Test,
                                public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsFloat : public ::testing::Test,
                              public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsFloatNonComplex
    : public ::testing::Test,
      public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsNumeric : public ::testing::Test,
                                public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsNumericNonComplex
    : public ::testing::Test,
      public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsIntegral : public ::testing::Test,
                                 public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsBoolean : public ::testing::Test,
                                public BasicTensorTestsData<TensorType> {
};
template <typename TensorType>
class BasicTensorTestsAll : public ::testing::Test,
                            public BasicTensorTestsData<TensorType> {
};

TYPED_TEST_SUITE(BasicTensorTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(BasicTensorTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(BasicTensorTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(BasicTensorTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(BasicTensorTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(BasicTensorTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(BasicTensorTestsNumericNonComplex, MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(BasicTensorTestsBoolean, MatXBoolTypes);

TYPED_TEST(BasicTensorTestsAll, DataSize)
{
  MATX_ENTER_HANDLER();
  ASSERT_EQ(10, this->t1.Size(0));
  ASSERT_EQ(20, this->t2.Size(0));
  ASSERT_EQ(30, this->t3.Size(0));
  ASSERT_EQ(40, this->t4.Size(0));

  ASSERT_EQ(10, this->t2.Size(1));
  ASSERT_EQ(20, this->t3.Size(1));
  ASSERT_EQ(30, this->t4.Size(1));

  ASSERT_EQ(10, this->t3.Size(2));
  ASSERT_EQ(20, this->t4.Size(2));

  ASSERT_EQ(10, this->t4.Size(3));
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, DataPointer)
{
  MATX_ENTER_HANDLER();

  ASSERT_EQ(&this->t0(), this->t0.Data());
  ASSERT_EQ(&this->t1(0), this->t1.Data());
  ASSERT_EQ(&this->t2(0, 0), this->t2.Data());
  ASSERT_EQ(&this->t3(0, 0, 0), this->t3.Data());
  ASSERT_EQ(&this->t4(0, 0, 0, 0), this->t4.Data());

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, Swap)
{
  MATX_ENTER_HANDLER();

  tensor_t<TypeParam, 2> tmp{{10,4}};
  tensor_t<TypeParam, 2> tmp2{{100,40}};

  ASSERT_EQ(tmp.Rank(), 2);
  ASSERT_EQ(tmp2.Rank(), 2);
  ASSERT_EQ(tmp.Size(0), 10);
  ASSERT_EQ(tmp.Size(1), 4);
  ASSERT_EQ(tmp2.Size(0), 100);
  ASSERT_EQ(tmp2.Size(1), 40);

  auto ptr = tmp.Data();
  auto ptr2 = tmp2.Data();
  swap(tmp, tmp2);

  ASSERT_EQ(tmp.Rank(), 2);
  ASSERT_EQ(tmp2.Rank(), 2);
  ASSERT_EQ(tmp2.Size(0), 10);
  ASSERT_EQ(tmp2.Size(1), 4);
  ASSERT_EQ(tmp.Size(0), 100);
  ASSERT_EQ(tmp.Size(1), 40);  
  ASSERT_EQ(tmp.Data(), ptr2);
  ASSERT_EQ(tmp2.Data(), ptr);
  ASSERT_EQ(tmp.GetRefCount(), 1);
  ASSERT_EQ(tmp2.GetRefCount(), 1);    

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, RefCnt)
{
  MATX_ENTER_HANDLER();

  tensor_t<TypeParam, 2> tmp{{10,4}};
  ASSERT_EQ(tmp.GetRefCount(), 1);

  tensor_t<TypeParam, 2> tmp2{tmp};
  ASSERT_EQ(tmp.GetRefCount(), 2);
  ASSERT_EQ(tmp2.GetRefCount(), 2);  

  tensor_t<TypeParam, 2> tmp3{{10,4}};
  tmp3.Shallow(tmp2);  
  ASSERT_EQ(tmp.GetRefCount(), 3);
  ASSERT_EQ(tmp2.GetRefCount(), 3);  
  ASSERT_EQ(tmp3.GetRefCount(), 3);

  TypeParam *data = tmp.Data();

  tmp3.Reset(reinterpret_cast<TypeParam*>(0x1234567));
  ASSERT_EQ(tmp3.GetRefCount(), 1);   

  tmp2.Reset(reinterpret_cast<TypeParam*>(0x1234567));
  ASSERT_EQ(tmp.GetRefCount(), 1);  

  tmp.Reset(reinterpret_cast<TypeParam*>(0x1234567)); 

  // Check if pointer was freed
  ASSERT_EQ(IsAllocated(data), false);  

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, ViewSize)
{
  MATX_ENTER_HANDLER();

  ASSERT_EQ(this->t1.Size(0), this->t1.Size(0));

  ASSERT_EQ(this->t2.Size(0), this->t2.Size(0));
  ASSERT_EQ(this->t2.Size(1), this->t2.Size(1));

  ASSERT_EQ(this->t3.Size(0), this->t3.Size(0));
  ASSERT_EQ(this->t3.Size(1), this->t3.Size(1));
  ASSERT_EQ(this->t3.Size(2), this->t3.Size(2));

  ASSERT_EQ(this->t4.Size(0), this->t4.Size(0));
  ASSERT_EQ(this->t4.Size(1), this->t4.Size(1));
  ASSERT_EQ(this->t4.Size(2), this->t4.Size(2));
  ASSERT_EQ(this->t4.Size(3), this->t4.Size(3));

  ASSERT_EQ(this->t0.TotalSize(), this->t0.TotalSize());
  ASSERT_EQ(this->t1.TotalSize(), this->t1.TotalSize());
  ASSERT_EQ(this->t2.TotalSize(), this->t2.TotalSize());
  ASSERT_EQ(this->t3.TotalSize(), this->t3.TotalSize());
  ASSERT_EQ(this->t4.TotalSize(), this->t4.TotalSize());

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, AssignmentOps)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 2> t2c{{20, 10}};
  tensor_t<TypeParam, 2> t2c2{{20, 10}};    

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i,j) = static_cast<TypeParam>(1);
      t2c2(i,j) = static_cast<TypeParam>(2);
    }
  }  

  (t2c = this->t2).run();

  cudaStreamSynchronize(0);
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), t2c(i,j));
    }
  }

  (this->t2 = t2c = t2c2).run();
  cudaStreamSynchronize(0);  
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), t2c2(i,j));
      ASSERT_EQ(t2c(i,j), t2c2(i,j));
    }
  }  

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsNumeric, AssignmentOps)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 2> t2c{{20, 10}};
  tensor_t<TypeParam, 2> t2c2{{20, 10}};    

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i,j) = static_cast<TypeParam>(1);
      t2c(i,j) = static_cast<TypeParam>(1);
    }
  }  

  (this->t2 += t2c).run();
  cudaStreamSynchronize(0);  
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), t2c(i,j) + t2c(i,j));
    }
  }

  (this->t2 -= t2c).run();
  cudaStreamSynchronize(0);  
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), t2c(i,j));
    }
  }  

  (this->t2 *= t2c).run();
  cudaStreamSynchronize(0);  
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), static_cast<TypeParam>(1));
    }
  }

  (t2c = this->t2).run();
  (this->t2 /= static_cast<TypeParam>(1)).run();
  cudaStreamSynchronize(0);  
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(t2c(i,j) , this->t2(i,j));
    }
  }    
    
  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsIntegral, AssignmentOps)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 2> t2c{{20, 10}};
  tensor_t<TypeParam, 2> t2c2{{20, 10}};

  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i,j) = static_cast<TypeParam>(1);
      t2c(i,j) = static_cast<TypeParam>(2);
    }
  }

  (this->t2 |= t2c).run();
  cudaStreamSynchronize(0);
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), 3);
    }
  }

  (this->t2 &= t2c).run();
  cudaStreamSynchronize(0);
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), 2);
    }
  }       

  (this->t2 ^= t2c).run();
  cudaStreamSynchronize(0);
  for (index_t i = 0; i < t2c.Size(0); i++) {
    for (index_t j = 0; j < t2c.Size(1); j++) {
      ASSERT_EQ(this->t2(i,j), 0);
    }
  }   

  MATX_EXIT_HANDLER();
}  

TYPED_TEST(BasicTensorTestsIntegral, Swizzle)
{
  MATX_ENTER_HANDLER();

  ASSERT_EQ(this->t2.Size(0), this->t2s.Size(1));
  ASSERT_EQ(this->t2.Size(1), this->t2s.Size(0));

  ASSERT_EQ(this->t2.Stride(1), this->t2s.Stride(0));
  ASSERT_EQ(this->t2.Stride(0), this->t2s.Stride(1));

  ASSERT_EQ(this->t3.Size(0), this->t3s.Size(2));
  ASSERT_EQ(this->t3.Size(1), this->t3s.Size(1));
  ASSERT_EQ(this->t3.Size(2), this->t3s.Size(0));

  ASSERT_EQ(this->t3.Stride(0), this->t3s.Stride(2));
  ASSERT_EQ(this->t3.Stride(1), this->t3s.Stride(1));
  ASSERT_EQ(this->t3.Stride(2), this->t3s.Stride(0));

  // Rank 2 swizzle
  for (index_t i = 0; i < this->t2.Size(0); i++) {
    for (index_t j = 0; j < this->t2.Size(1); j++) {
      this->t2(i, j) = static_cast<TypeParam>(i * 100 + j);
    }
  }

  for (index_t i = 0; i < this->t2s.Size(0); i++) {
    for (index_t j = 0; j < this->t2s.Size(1); j++) {
      ASSERT_EQ(this->t2s(i, j), j * 100 + i);
    }
  }

  // Rank 3 swizzle
  for (index_t i = 0; i < this->t3.Size(0); i++) {
    for (index_t j = 0; j < this->t3.Size(1); j++) {
      for (index_t k = 0; k < this->t3.Size(2); k++) {
        this->t3(i, j, k) = static_cast<TypeParam>(i * 10000 + j * 100 + k);
      }
    }
  }

  for (index_t i = 0; i < this->t3s.Size(0); i++) {
    for (index_t j = 0; j < this->t3s.Size(1); j++) {
      for (index_t k = 0; k < this->t3s.Size(2); k++) {
        ASSERT_EQ(this->t3s(i, j, k), k * 10000 + j * 100 + i);
      }
    }
  }

  // Rank 4 swizzle
  for (index_t i = 0; i < this->t4.Size(0); i++) {
    for (index_t j = 0; j < this->t4.Size(1); j++) {
      for (index_t k = 0; k < this->t4.Size(2); k++) {
        for (index_t l = 0; l < this->t4.Size(3); l++) {
          this->t4(i, j, k, l) =
              static_cast<TypeParam>(i * 1000000 + j * 10000 + k * 100 + l);
        }
      }
    }
  }

  for (index_t i = 0; i < this->t4s.Size(0); i++) {
    for (index_t j = 0; j < this->t4s.Size(1); j++) {
      for (index_t k = 0; k < this->t4s.Size(2); k++) {
        for (index_t l = 0; l < this->t4s.Size(3); l++) {
          ASSERT_EQ(this->t4s(i, j, k, l),
                    l * 1000000 + k * 10000 + j * 100 + i);
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsIntegral, InitAssign)
{
  MATX_ENTER_HANDLER();

  tensor_t<TypeParam, 1> t1v_small{{4}};
  t1v_small.SetVals({1, 2, 3, 4});
  for (index_t i = 0; i < 4; i++) {
    ASSERT_EQ(t1v_small(i), i + 1);
  }

  tensor_t<TypeParam, 2> t2v_small{{4, 4}};
  t2v_small.SetVals(
      {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}});

  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 4; j++) {
      ASSERT_EQ(t2v_small(i, j), i * 4 + j + 1);
    }
  }

  MATX_EXIT_HANDLER();
}


TYPED_TEST(BasicTensorTestsAll, Print)
{
  MATX_ENTER_HANDLER();

  auto t = make_tensor<TypeParam>({3});
  (t = ones(t.Shape())).run();
  print(t);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(BasicTensorTestsAll, DLPack)
{
  MATX_ENTER_HANDLER();

  auto t = make_tensor<TypeParam>({5,10,20});
  auto dl = t.GetDLPackTensor();

  ASSERT_EQ(dl->dl_tensor.ndim, 3);
  ASSERT_EQ(dl->dl_tensor.data, t.Data());
  ASSERT_EQ(dl->dl_tensor.device.device_id, 0);
  ASSERT_EQ(dl->dl_tensor.device.device_type, kDLCUDA);
  auto dlt = detail::TypeToDLPackType<TypeParam>();
  ASSERT_EQ(dl->dl_tensor.dtype.code, dlt.code);
  ASSERT_EQ(dl->dl_tensor.dtype.bits, dlt.bits);
  ASSERT_EQ(dl->dl_tensor.dtype.lanes, dlt.lanes);
  ASSERT_EQ(dl->dl_tensor.shape[0], t.Size(0));
  ASSERT_EQ(dl->dl_tensor.shape[1], t.Size(1));
  ASSERT_EQ(dl->dl_tensor.shape[2], t.Size(2));
  ASSERT_EQ(dl->dl_tensor.strides[0], t.Stride(0));
  ASSERT_EQ(dl->dl_tensor.strides[1], t.Stride(1));
  ASSERT_EQ(dl->dl_tensor.strides[2], t.Stride(2));
  ASSERT_EQ(t.GetRefCount(), 2);
  dl->deleter(dl);
  ASSERT_EQ(dl->dl_tensor.shape, nullptr);
  ASSERT_EQ(dl->dl_tensor.strides, nullptr);
  ASSERT_EQ(t.GetRefCount(), 1);

  MATX_EXIT_HANDLER();
}

