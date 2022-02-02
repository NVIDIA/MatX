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

template <typename TensorType> struct ViewTestsData {
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
class ViewTestsComplex : public ::testing::Test,
                                public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsFloat : public ::testing::Test,
                              public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsFloatNonComplex
    : public ::testing::Test,
      public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsNumeric : public ::testing::Test,
                                public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsNumericNonComplex
    : public ::testing::Test,
      public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsIntegral : public ::testing::Test,
                                 public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsBoolean : public ::testing::Test,
                                public ViewTestsData<TensorType> {
};
template <typename TensorType>
class ViewTestsAll : public ::testing::Test,
                            public ViewTestsData<TensorType> {
};

TYPED_TEST_SUITE(ViewTestsAll, MatXAllTypes);
TYPED_TEST_SUITE(ViewTestsComplex, MatXComplexTypes);
TYPED_TEST_SUITE(ViewTestsFloat, MatXFloatTypes);
TYPED_TEST_SUITE(ViewTestsFloatNonComplex, MatXFloatNonComplexTypes);
TYPED_TEST_SUITE(ViewTestsNumeric, MatXNumericTypes);
TYPED_TEST_SUITE(ViewTestsIntegral, MatXAllIntegralTypes);
TYPED_TEST_SUITE(ViewTestsNumericNonComplex, MatXNumericNonComplexTypes);
TYPED_TEST_SUITE(ViewTestsBoolean, MatXBoolTypes);


TYPED_TEST(ViewTestsNumericNonComplex, OverlapView)
{
  MATX_ENTER_HANDLER();

  tensor_t<TypeParam, 1> a{{10}};
  a.SetVals({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  auto ao = a.OverlapView({4}, {2});

  tensor_t<TypeParam, 2> b{{4, 4}};
  b.SetVals({{0, 1, 2, 3}, {2, 3, 4, 5}, {4, 5, 6, 7}, {6, 7, 8, 9}});
  for (index_t i = 0; i < b.Size(0); i++) {
    for (index_t j = 0; j < b.Size(1); j++) {
      ASSERT_EQ(ao(i, j), b(i, j));
    }
  }

  auto ao2 = a.OverlapView({4}, {1});

  tensor_t<TypeParam, 2> b2{{7, 4}};
  b2.SetVals({{0, 1, 2, 3},
              {1, 2, 3, 4},
              {2, 3, 4, 5},
              {3, 4, 5, 6},
              {4, 5, 6, 7},
              {5, 6, 7, 8},
              {6, 7, 8, 9}});
  for (index_t i = 0; i < b2.Size(0); i++) {
    for (index_t j = 0; j < b2.Size(1); j++) {
      ASSERT_EQ(ao2(i, j), b2(i, j));
    }
  }

  auto ao3 = a.OverlapView({4}, {3});
  tensor_t<TypeParam, 2> b3{{3, 4}};
  b3.SetVals({{0, 1, 2, 3}, {3, 4, 5, 6}, {6, 7, 8, 9}});
  for (index_t i = 0; i < b3.Size(0); i++) {
    for (index_t j = 0; j < b3.Size(1); j++) {
      ASSERT_EQ(ao3(i, j), b3(i, j));
    }
  }

  auto ao4 = a.OverlapView({3}, {2});
  tensor_t<TypeParam, 2> b4{{4, 3}};
  b4.SetVals({{0, 1, 2}, {2, 3, 4}, {4, 5, 6}, {6, 7, 8}});
  for (index_t i = 0; i < b4.Size(0); i++) {
    for (index_t j = 0; j < b4.Size(1); j++) {
      ASSERT_EQ(ao4(i, j), b4(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ViewTestsAll, Stride)
{
  MATX_ENTER_HANDLER();

  ASSERT_EQ(this->t1.Stride(0), 1);
  ASSERT_EQ(this->t2.Stride(1), 1);
  ASSERT_EQ(this->t3.Stride(2), 1);
  ASSERT_EQ(this->t4.Stride(3), 1);

  ASSERT_EQ(this->t2.Stride(0), this->t2.Size(1));
  ASSERT_EQ(this->t3.Stride(1), this->t3.Size(2));
  ASSERT_EQ(this->t4.Stride(2), this->t4.Size(3));

  ASSERT_EQ(this->t3.Stride(0), this->t3.Size(2) * this->t3.Size(1));
  ASSERT_EQ(this->t4.Stride(1), this->t4.Size(3) * this->t4.Size(2));

  ASSERT_EQ(this->t4.Stride(0),
            this->t4.Size(3) * this->t4.Size(2) * this->t4.Size(1));

  MATX_EXIT_HANDLER();
}


TYPED_TEST(ViewTestsIntegral, SliceStride)
{
  MATX_ENTER_HANDLER();
  this->t1.SetVals({10, 20, 30, 40, 50, 60, 70, 80, 90, 100});
  auto t1t = this->t1.Slice({0}, {matxEnd}, {2});

  for (index_t i = 0; i < this->t1.Size(0); i += 2) {
    ASSERT_EQ(this->t1(i), t1t(i / 2));
  }

  auto t1t2 = this->t1.Slice({2}, {matxEnd}, {2});

  for (index_t i = 0; i < t1t2.Size(0); i++) {
    ASSERT_EQ(30 + 20 * i, t1t2(i));
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(ViewTestsIntegral, Slice)
{
  MATX_ENTER_HANDLER();
  auto t2t = this->t2.Slice({1, 2}, {3, 5});
  auto t3t = this->t3.Slice({1, 2, 3}, {3, 5, 7});
  auto t4t = this->t4.Slice({1, 2, 3, 4}, {3, 5, 7, 9});

#ifndef NDEBUG
  // Negative slice test
  try {
    auto t2e = this->t2.Slice({1, 2}, {1, 2});
    ASSERT_EQ(true, false);
  }
  catch (...) {
    ASSERT_EQ(true, true);
  }
#endif  

  ASSERT_EQ(t2t.Size(0), 2);
  ASSERT_EQ(t2t.Size(1), 3);

  ASSERT_EQ(t3t.Size(0), 2);
  ASSERT_EQ(t3t.Size(1), 3);
  ASSERT_EQ(t3t.Size(2), 4);

  ASSERT_EQ(t4t.Size(0), 2);
  ASSERT_EQ(t4t.Size(1), 3);
  ASSERT_EQ(t4t.Size(2), 4);
  ASSERT_EQ(t4t.Size(3), 5);

  for (index_t i = 0; i < t2t.Size(0); i++) {
    for (index_t j = 0; j < t2t.Size(1); j++) {
      ASSERT_EQ(t2t(i, j), this->t2(i + 1, j + 2));
    }
  }

  for (index_t i = 0; i < t3t.Size(0); i++) {
    for (index_t j = 0; j < t3t.Size(1); j++) {
      for (index_t k = 0; k < t3t.Size(2); k++) {
        ASSERT_EQ(t3t(i, j, k), this->t3(i + 1, j + 2, k + 3));
      }
    }
  }

  for (index_t i = 0; i < t4t.Size(0); i++) {
    for (index_t j = 0; j < t4t.Size(1); j++) {
      for (index_t k = 0; k < t4t.Size(2); k++) {
        for (index_t l = 0; l < t4t.Size(3); l++) {
          ASSERT_EQ(t4t(i, j, k, l), this->t4(i + 1, j + 2, k + 3, l + 4));
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(ViewTestsAll, SliceAndReduce)
{
  tensor_t<float, 2> t2t{{20, 10}};
  tensor_t<float, 3> t3t{{30, 20, 10}};

  MATX_ENTER_HANDLER();
  {
    index_t j = 0;
    auto t2sly = t2t.Slice<1>({0, j}, {matxEnd, matxDropDim});
    for (index_t i = 0; i < t2sly.Size(0); i++) {
      ASSERT_EQ(t2sly(i), t2t(i, j));
    }
  }

  {
    index_t i = 0;
    auto t2slx = t2t.Slice<1>({i, 0}, {matxDropDim, matxEnd});
    for (index_t j = 0; j < t2slx.Size(0); j++) {
      ASSERT_EQ(t2slx(j), t2t(i, j));
    }
  }

  {
    index_t j = 0;
    index_t k = 0;
    auto t3slz = t3t.Slice<1>({0, j, k}, {matxEnd, matxDropDim, matxDropDim});
    for (index_t i = 0; i < t3slz.Size(0); i++) {
      ASSERT_EQ(t3slz(i), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t k = 0;
    auto t3sly = t3t.Slice<1>({i, 0, k}, {matxDropDim, matxEnd, matxDropDim});
    for (index_t j = 0; j < t3sly.Size(0); j++) {
      ASSERT_EQ(t3sly(j), t3t(i, j, k));
    }
  }

  {
    index_t i = 0;
    index_t j = 0;
    auto t3slx = t3t.Slice<1>({i, j, 0}, {matxDropDim, matxDropDim, matxEnd});
    for (index_t k = 0; k < t3slx.Size(0); k++) {
      ASSERT_EQ(t3slx(k), t3t(i, j, k));
    }
  }

  {
    index_t k = 0;
    auto t3slzy = t3t.Slice<2>({0, 0, k}, {matxEnd, matxEnd, matxDropDim});
    for (index_t i = 0; i < t3slzy.Size(0); i++) {
      for (index_t j = 0; j < t3slzy.Size(1); j++) {
        ASSERT_EQ(t3slzy(i, j), t3t(i, j, k));
      }
    }
  }

  {
    index_t j = 0;
    auto t3slzx = t3t.Slice<2>({0, j, 0}, {matxEnd, matxDropDim, matxEnd});
    for (index_t i = 0; i < t3slzx.Size(0); i++) {
      for (index_t k = 0; k < t3slzx.Size(1); k++) {
        ASSERT_EQ(t3slzx(i, k), t3t(i, j, k));
      }
    }
  }

  {
    index_t i = 0;
    auto t3slyx = t3t.Slice<2>({i, 0, 0}, {matxDropDim, matxEnd, matxEnd});
    for (index_t j = 0; j < t3slyx.Size(0); j++) {
      for (index_t k = 0; k < t3slyx.Size(1); k++) {
        ASSERT_EQ(t3slyx(j, k), t3t(i, j, k));
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TEST(BasicTensorTest, Clone)
{
  tensor_t<float, 0> t0{};
  tensor_t<float, 1> t1{{10}};
  tensor_t<float, 2> t2{{20, 10}};
  tensor_t<float, 3> t3{{30, 20, 10}};

  MATX_ENTER_HANDLER();
  // clone t0 across 0/1/2/3 dim
  auto t0c1 = t0.Clone<1>({5});
  ASSERT_EQ(t0c1.Size(0), 5);
  for (index_t i = 0; i < t0c1.Size(0); i++) {
    ASSERT_EQ(t0c1(i), t0());
  }

  auto t0c2 = t0.Clone<2>({5, 6});
  ASSERT_EQ(t0c2.Size(0), 5);
  ASSERT_EQ(t0c2.Size(1), 6);
  for (index_t i = 0; i < t0c2.Size(0); i++) {
    for (index_t j = 0; j < t0c2.Size(1); j++) {
      ASSERT_EQ(t0c2(i, j), t0());
    }
  }

  auto t0c3 = t0.Clone<3>({5, 6, 7});
  ASSERT_EQ(t0c3.Size(0), 5);
  ASSERT_EQ(t0c3.Size(1), 6);
  ASSERT_EQ(t0c3.Size(2), 7);
  for (index_t i = 0; i < t0c3.Size(0); i++) {
    for (index_t j = 0; j < t0c3.Size(1); j++) {
      for (index_t k = 0; k < t0c3.Size(2); k++) {
        ASSERT_EQ(t0c3(i, j, k), t0());
      }
    }
  }

  auto t0c4 = t0.Clone<4>({5, 6, 7, 8});
  ASSERT_EQ(t0c4.Size(0), 5);
  ASSERT_EQ(t0c4.Size(1), 6);
  ASSERT_EQ(t0c4.Size(2), 7);
  ASSERT_EQ(t0c4.Size(3), 8);
  for (index_t i = 0; i < t0c4.Size(0); i++) {
    for (index_t j = 0; j < t0c4.Size(1); j++) {
      for (index_t k = 0; k < t0c4.Size(2); k++) {
        for (index_t l = 0; l < t0c4.Size(3); l++) {
          ASSERT_EQ(t0c4(i, j, k, l), t0());
        }
      }
    }
  }

  auto t1c1 = t1.Clone<2>({5, matxKeepDim});
  ASSERT_EQ(t1c1.Size(0), 5);
  for (index_t i = 0; i < t1c1.Size(0); i++) {
    for (index_t j = 0; j < t1c1.Size(1); j++) {
      ASSERT_EQ(t1c1(i, j), t1(j));
    }
  }

  auto t1c2 = t1.Clone<3>({5, 6, matxKeepDim});
  ASSERT_EQ(t1c2.Size(0), 5);
  ASSERT_EQ(t1c2.Size(1), 6);
  ASSERT_EQ(t1c2.Size(2), t1.Size(0));
  for (index_t i = 0; i < t1c2.Size(0); i++) {
    for (index_t j = 0; j < t1c2.Size(1); j++) {
      for (index_t k = 0; k < t1c2.Size(2); k++) {
        ASSERT_EQ(t1c2(i, j, k), t1(k));
      }
    }
  }

  auto t1c3 = t1.Clone<4>({5, 6, 7, matxKeepDim});
  ASSERT_EQ(t1c3.Size(0), 5);
  ASSERT_EQ(t1c3.Size(1), 6);
  ASSERT_EQ(t1c3.Size(2), 7);
  ASSERT_EQ(t1c3.Size(3), t1.Size(0));
  for (index_t i = 0; i < t1c3.Size(0); i++) {
    for (index_t j = 0; j < t1c3.Size(1); j++) {
      for (index_t k = 0; k < t1c3.Size(2); k++) {
        for (index_t l = 0; l < t1c3.Size(3); l++) {
          ASSERT_EQ(t1c3(i, j, k, l), t1(l));
        }
      }
    }
  }

  // clone t2 across 0/1 dim
  auto t2c1 = t2.Clone<3>({5, matxKeepDim, matxKeepDim});
  ASSERT_EQ(t2c1.Size(0), 5);
  for (index_t i = 0; i < t2c1.Size(0); i++) {
    for (index_t j = 0; j < t2c1.Size(1); j++) {
      for (index_t k = 0; k < t2c1.Size(2); k++) {
        ASSERT_EQ(t2c1(i, j, k), t2(j, k));
      }
    }
  }

  auto t2c2 = t2.Clone<4>({5, 6, matxKeepDim, matxKeepDim});
  ASSERT_EQ(t2c2.Size(0), 5);
  ASSERT_EQ(t2c2.Size(1), 6);
  for (index_t i = 0; i < t2c2.Size(0); i++) {
    for (index_t j = 0; j < t2c2.Size(1); j++) {
      for (index_t k = 0; k < t2c2.Size(2); k++) {
        for (index_t l = 0; l < t2c2.Size(3); l++) {
          ASSERT_EQ(t2c2(i, j, k, l), t2(k, l));
        }
      }
    }
  }

  // clone t3 across 0 dim
  auto t3c1 = t3.Clone<4>({5, matxKeepDim, matxKeepDim, matxKeepDim});
  ASSERT_EQ(t3c1.Size(0), 5);
  for (index_t i = 0; i < t3c1.Size(0); i++) {
    for (index_t j = 0; j < t3c1.Size(1); j++) {
      for (index_t k = 0; k < t3c1.Size(2); k++) {
        for (index_t l = 0; l < t3c1.Size(3); l++) {
          ASSERT_EQ(t3c1(i, j, k, l), t3(j, k, l));
        }
      }
    }
  }

  MATX_EXIT_HANDLER();
}

TEST(ViewTests, Random)
{
  MATX_ENTER_HANDLER();
  {
    index_t count = 100;
    randomGenerator_t<float> rfloat(count * count * count, 0);
    auto t3fu = rfloat.GetTensorView<3>({count, count, count}, UNIFORM);
    auto t3fn = rfloat.GetTensorView<3>({count, count, count}, NORMAL);

    tensor_t<float, 3> t3f({count, count, count});

    (t3f = -1000000).run();
    (t3f = t3fu).run();
    t3f.PrefetchHost(0);
    cudaDeviceSynchronize();

    float total = 0;
    for (index_t i = 0; i < count; i++) {
      for (index_t j = 0; j < count; j++) {
        for (index_t k = 0; k < count; k++) {
          float val = t3f(i, j, k) - 0.5f; // mean centered at zero
          ASSERT_NE(val, -1000000);
          total += val;
          ASSERT_LE(val, 0.5f);
          ASSERT_LE(-0.5f, val);
        }
      }
    }

    ASSERT_LT(fabs(total / (count * count * count)), .05);

    (t3f = -1000000).run();
    (t3f = t3fn).run();
    t3f.PrefetchHost(0);
    cudaDeviceSynchronize();

    total = 0;

    for (index_t i = 0; i < count; i++) {
      for (index_t j = 0; j < count; j++) {
        for (index_t k = 0; k < count; k++) {
          float val = t3f(i, j, k);
          ASSERT_NE(val, -1000000);
          total += val;
        }
      }
    }

    ASSERT_LT(fabs(total / (count * count * count)), .15);
  }
  MATX_EXIT_HANDLER();
}


TYPED_TEST(ViewTestsComplex, RealComplexView)
{
  MATX_ENTER_HANDLER();
  tensor_t<TypeParam, 1> tc({10});
  auto tr = tc.RealView();
  auto ti = tc.ImagView();

  for (int i = 0; i < 10; i++) {
    TypeParam val(
        static_cast<promote_half_t<typename TypeParam::value_type>>(i),
        static_cast<promote_half_t<typename TypeParam::value_type>>(i + 10));
    tc(i) = val;
  }

  for (int i = 0; i < 10; i++) {
    ASSERT_EQ((float)tc(i).real(), (float)tr(i));
    ASSERT_EQ((float)tc(i).imag(), (float)ti(i));
  }
  MATX_EXIT_HANDLER();
}
