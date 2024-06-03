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

template <typename T> class FileIoTests : public ::testing::Test {
protected:
  using GTestType = cuda::std::tuple_element_t<0, T>;
  using GExecType = cuda::std::tuple_element_t<1, T>;     

  void SetUp() override
  {
    pb = std::make_unique<detail::MatXPybind>();
    pb->InitAndRunTVGenerator<GTestType>("00_file_io", "csv", "run", {});
  }

  void TearDown() override { pb.reset(); }
  GExecType exec{};
  std::unique_ptr<detail::MatXPybind> pb;
  const std::string small_csv = "../test/00_io/small_csv_comma_nh.csv";
  const std::string small_complex_csv = "../test/00_io/small_csv_complex_comma_nh.csv";
  tensor_t<GTestType, 2> Av{{10, 2}};
};

template <typename TensorType> class FileIoTestsNonComplexFloatTypes : public FileIoTests<TensorType> {};
template <typename TensorType> class FileIoTestsComplexFloatTypes : public FileIoTests<TensorType> {};

TYPED_TEST_SUITE(FileIoTestsNonComplexFloatTypes, MatXFloatNonComplexNonHalfTypesCUDAExec);
TYPED_TEST_SUITE(FileIoTestsComplexFloatTypes, MatXComplexNonHalfTypesCUDAExec);

TYPED_TEST(FileIoTestsNonComplexFloatTypes, SmallCSVRead)
{
  MATX_ENTER_HANDLER();

  // example-begin read_csv-test-1
  io::read_csv(this->Av, this->small_csv, ",");
  // example-end read_csv-test-1
  MATX_TEST_ASSERT_COMPARE(this->pb, this->Av, this->small_csv.c_str(), 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsComplexFloatTypes, SmallCSVRead)
{
  MATX_ENTER_HANDLER();

  io::read_csv(this->Av, this->small_complex_csv, ",");
  MATX_TEST_ASSERT_COMPARE(this->pb, this->Av, this->small_complex_csv.c_str(), 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, SmallCSVWrite)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;     
  tensor_t<TestType, 2> Avs{{10, 2}};

  this->pb->NumpyToTensorView(this->Av, this->small_csv.c_str());
  // example-begin write_csv-test-1
  io::write_csv(this->Av, "temp.csv", ",");
  // example-end write_csv-test-1
  io::read_csv(Avs, "temp.csv", ",", false);
  MATX_TEST_ASSERT_COMPARE(this->pb, Avs, this->small_csv.c_str(), 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATRead)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;     
  auto t = make_tensor<TestType>({1,10});

  // example-begin read_mat-test-1
  // Read "myvar" from mat file
  io::read_mat(t, "../test/00_io/test.mat", "myvar");
  // example-end read_mat-test-1
  ASSERT_NEAR(t(0,0), 1.456, 0.001);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATWrite)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  using ExecType = cuda::std::tuple_element_t<1, TypeParam>;     
  auto t = make_tensor<TestType>({2,3});
  auto t2 = make_tensor<TestType>({2,3});
  t.SetVals({{1,2,3},{4,5,6}});

  // Write "myvar" to mat file
  // example-begin write_mat-test-1
  io::write_mat(t, "test_write.mat", "myvar");
  // example-end write_mat-test-1
  io::read_mat(t2, "test_write.mat", "myvar");
  for (index_t i = 0; i < t.Size(0); i++) {
    for (index_t j = 0; j < t.Size(1); j++) {
      ASSERT_EQ(t(i,j), t2(i,j));
    }
  }

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATWriteRank5)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2,3,1,2,3});
  auto t2 = make_tensor<TestType>({2,3,1,2,3});

  (t = random<float>(t.Shape(), UNIFORM)).run(this->exec);

  this->exec.sync();

  // Read "myvar" from mat file
  io::write_mat(t, "test_write.mat", "myvar");
  io::read_mat(t2, "test_write.mat", "myvar");
  for (index_t i = 0; i < t.Size(0); i++) {
    for (index_t j = 0; j < t.Size(1); j++) {
      for (index_t k = 0; k < t.Size(2); k++) {
        for (index_t l = 0; l < t.Size(3); l++) {
          for (index_t m = 0; m < t.Size(4); m++) {
            ASSERT_EQ(t(i,j,k,l,m), t2(i,j,k,l,m));
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATWriteRank5GetShape)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2,3,1,2,3});
  tensor_t<TestType,5> t2;

  (t = random<float>(t.Shape(), UNIFORM)).run(this->exec);

  this->exec.sync();

  // Read "myvar" from mat file
  io::write_mat(t, "test_write.mat", "myvar");
  t2.Shallow(io::read_mat<decltype(t2)>("test_write.mat", "myvar"));

  for (index_t i = 0; i < t.Size(0); i++) {
    for (index_t j = 0; j < t.Size(1); j++) {
      for (index_t k = 0; k < t.Size(2); k++) {
        for (index_t l = 0; l < t.Size(3); l++) {
          for (index_t m = 0; m < t.Size(4); m++) {
            ASSERT_EQ(t(i,j,k,l,m), t2(i,j,k,l,m));
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsComplexFloatTypes, MATWriteRank5GetShape)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2,3,1,2,3});
  tensor_t<TestType,5> t2;

  (t = random<float>(t.Shape(), UNIFORM)).run(this->exec);

  this->exec.sync();

  // Read "myvar" from mat file
  io::write_mat(t, "test_write.mat", "myvar");
  t2.Shallow(io::read_mat<decltype(t2)>("test_write.mat", "myvar"));

  for (index_t i = 0; i < t.Size(0); i++) {
    for (index_t j = 0; j < t.Size(1); j++) {
      for (index_t k = 0; k < t.Size(2); k++) {
        for (index_t l = 0; l < t.Size(3); l++) {
          for (index_t m = 0; m < t.Size(4); m++) {
            ASSERT_EQ(t(i,j,k,l,m), t2(i,j,k,l,m));
          }
        }
      }
    }
  }
  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, NPYRead)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2, 3});

  // example-begin read_npy-test-1
  io::read_npy(t, "../test/00_io/test.npy");
  // example-end read_npy-test-1

  // The test npy is [[1.5, 2.5, 3.5], [4.5, 5.5, 6.5]]
  ASSERT_NEAR(t(0, 0), 1.5, 0.001);
  ASSERT_NEAR(t(0, 1), 2.5, 0.001);
  ASSERT_NEAR(t(0, 2), 3.5, 0.001);
  ASSERT_NEAR(t(1, 0), 4.5, 0.001);
  ASSERT_NEAR(t(1, 1), 5.5, 0.001);
  ASSERT_NEAR(t(1, 2), 6.5, 0.001);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, NPYWrite)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2, 3});
  auto t2 = make_tensor<TestType>({2, 3});
  t.SetVals({{1, 2, 3}, {4, 5, 6}});

  // example-begin write_npy-test-1
  io::write_npy(t, "test_write.npy");
  // example-end write_npy-test-1

  io::read_npy(t2, "test_write.npy");
  for (index_t i = 0; i < t.Size(0); i++) {
    for (index_t j = 0; j < t.Size(1); j++) {
      ASSERT_EQ(t(i, j), t2(i, j));
    }
  }

  MATX_EXIT_HANDLER();
}