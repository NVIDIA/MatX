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

#include <chrono>
#include <cstdint>
#include <filesystem>

using namespace matx;

class FileIoMatHost : public ::testing::Test {
protected:
  void SetUp() override
  {
    const auto stamp = std::chrono::steady_clock::now().time_since_epoch().count();
    const auto name = ::testing::UnitTest::GetInstance()->current_test_info()->name();
    const auto path = std::filesystem::temp_directory_path() /
                      (std::string("matx_") + name + "_" + std::to_string(stamp));
    ASSERT_TRUE(std::filesystem::create_directory(path));
    directory = path;
  }

  void TearDown() override
  {
    if (!directory.empty()) {
      std::filesystem::remove_all(directory);
    }
  }

  static pybind11::dict LoadVariables(const std::string &filename)
  {
    auto scipy = pybind11::module_::import("scipy.io");
    auto variables = scipy.attr("loadmat")(filename).cast<pybind11::dict>();
    variables.attr("pop")("__header__", pybind11::none());
    variables.attr("pop")("__version__", pybind11::none());
    variables.attr("pop")("__globals__", pybind11::none());
    return variables;
  }

  std::filesystem::path directory;
};

TEST_F(FileIoMatHost, LegacyWriteRoundTrip)
{
  auto input = make_tensor<float>({2, 3}, MATX_HOST_MALLOC_MEMORY);
  for (index_t row = 0; row < 2; ++row) {
    for (index_t col = 0; col < 3; ++col) {
      input(row, col) = static_cast<float>(10 * row + col);
    }
  }

  const auto filename = (directory / "legacy.mat").string();
  io::write_mat(input, filename, "samples");

  const auto variables = LoadVariables(filename);
  ASSERT_EQ(variables.size(), 1U);
  ASSERT_TRUE(variables.contains("samples"));
  const auto samples = variables["samples"].cast<pybind11::array>();
  ASSERT_EQ(samples.ndim(), 2);
  ASSERT_EQ(samples.shape(0), 2);
  ASSERT_EQ(samples.shape(1), 3);
  ASSERT_TRUE(samples.dtype().equal(pybind11::dtype::of<float>()));

  const auto values = samples.unchecked<float, 2>();
  auto readback = make_tensor<float>({2, 3}, MATX_HOST_MALLOC_MEMORY);
  io::read_mat(readback, filename, "samples");
  for (index_t row = 0; row < 2; ++row) {
    for (index_t col = 0; col < 3; ++col) {
      EXPECT_EQ(values(row, col), input(row, col));
      EXPECT_EQ(readback(row, col), input(row, col));
    }
  }
}

TEST_F(FileIoMatHost, LegacyWriteReplacesFile)
{
  auto input = make_tensor<float>({1, 1}, MATX_HOST_MALLOC_MEMORY);
  const auto filename = (directory / "replace.mat").string();
  input(0, 0) = 1.0f;
  io::write_mat(input, filename, "first");
  input(0, 0) = 2.0f;
  io::write_mat(input, filename, "second");

  const auto variables = LoadVariables(filename);
  ASSERT_EQ(variables.size(), 1U);
  EXPECT_TRUE(variables.contains("second"));
}

TEST_F(FileIoMatHost, WriteDictionaryRoundTrip)
{
  detail::MatXPybind python;
  const auto filename = (directory / "dictionary.mat").string();
  pybind11::dict variables;
  {
    auto matrix = make_tensor<float>({2, 3}, MATX_HOST_MALLOC_MEMORY);
    auto storage = make_tensor<int16_t>({2, 3, 6}, MATX_HOST_MALLOC_MEMORY);
    for (index_t i = 0; i < 2; ++i) {
      for (index_t j = 0; j < 3; ++j) {
        matrix(i, j) = static_cast<float>(10 * i + j);
        for (index_t k = 0; k < 6; ++k) {
          storage(i, j, k) = static_cast<int16_t>(100 * i + 10 * j + k);
        }
      }
    }
    auto view = slice(storage, {0, 0, 1}, {2, 3, 6}, {1, 1, 2});

    // example-begin write_mat-dictionary-add
    io::add_mat_variable(variables, "matrix", matrix);
    io::add_mat_variable(variables, "view", view);
    // example-end write_mat-dictionary-add

    matrix(0, 0) = -1.0f;
    storage(0, 0, 1) = -1;
  }

  // example-begin write_mat-dictionary-save
  io::write_mat(variables, filename);
  // example-end write_mat-dictionary-save

  const auto saved = LoadVariables(filename);
  ASSERT_EQ(saved.size(), 2U);
  ASSERT_TRUE(saved.contains("matrix"));
  ASSERT_TRUE(saved.contains("view"));
  const auto matrix = saved["matrix"].cast<pybind11::array>();
  const auto view = saved["view"].cast<pybind11::array>();
  ASSERT_EQ(matrix.ndim(), 2);
  ASSERT_EQ(matrix.shape(0), 2);
  ASSERT_EQ(matrix.shape(1), 3);
  ASSERT_TRUE(matrix.dtype().equal(pybind11::dtype::of<float>()));
  ASSERT_EQ(view.ndim(), 3);
  ASSERT_EQ(view.shape(0), 2);
  ASSERT_EQ(view.shape(1), 3);
  ASSERT_EQ(view.shape(2), 3);
  ASSERT_TRUE(view.dtype().equal(pybind11::dtype::of<int16_t>()));

  const auto matrix_values = matrix.unchecked<float, 2>();
  const auto view_values = view.unchecked<int16_t, 3>();
  for (index_t i = 0; i < 2; ++i) {
    for (index_t j = 0; j < 3; ++j) {
      EXPECT_EQ(matrix_values(i, j), static_cast<float>(10 * i + j));
      for (index_t k = 0; k < 3; ++k) {
        EXPECT_EQ(view_values(i, j, k), static_cast<int16_t>(100 * i + 10 * j + 2 * k + 1));
      }
    }
  }
}

TEST_F(FileIoMatHost, WriteDictionaryInvalidPath)
{
  detail::MatXPybind python;
  pybind11::dict variables;
  auto input = make_tensor<float>({1, 1}, MATX_HOST_MALLOC_MEMORY);
  input(0, 0) = 1.0f;
  io::add_mat_variable(variables, "samples", input);

  const auto filename = (directory / "missing" / "output.mat").string();
  EXPECT_THROW(io::write_mat(variables, filename), pybind11::error_already_set);
}

TEST_F(FileIoMatHost, WriteDictionaryRejectsInvalidNamesBeforeReplacingFile)
{
  auto input = make_tensor<float>({1, 1}, MATX_HOST_MALLOC_MEMORY);
  input(0, 0) = 1.0f;
  const auto filename = (directory / "existing.mat").string();
  io::write_mat(input, filename, "original");

  pybind11::dict variables;
  io::add_mat_variable(variables, "replacement", input);
  const auto replacement = variables["replacement"].cast<pybind11::array>();
  variables["_hidden"] = replacement;
  EXPECT_THROW(io::write_mat(variables, filename), pybind11::value_error);
  variables.attr("pop")("_hidden");
  variables[""] = replacement;
  EXPECT_THROW(io::write_mat(variables, filename), pybind11::value_error);
  variables.attr("pop")("");
  variables[pybind11::int_(1)] = replacement;
  EXPECT_THROW(io::write_mat(variables, filename), pybind11::type_error);

  const auto saved = LoadVariables(filename);
  ASSERT_EQ(saved.size(), 1U);
  ASSERT_TRUE(saved.contains("original"));
  const auto original = saved["original"].cast<pybind11::array>();
  EXPECT_EQ((original.unchecked<float, 2>()(0, 0)), 1.0f);
}

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

TYPED_TEST(FileIoTestsNonComplexFloatTypes, SmallCSVReadUninitialized)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  tensor_t<TestType, 2> t;

  io::read_csv(t, this->small_csv, ",");

  ASSERT_EQ(t.Size(0), 10);
  ASSERT_EQ(t.Size(1), 2);
  MATX_TEST_ASSERT_COMPARE(this->pb, t, this->small_csv.c_str(), 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, CSVReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  tensor_t<TestType, 2> t{{10, 2}};
  
  // Test reading a non-existent CSV file
  const std::string non_existent_file = "../test/00_io/non_existent_file.csv";
  ASSERT_THROW({
    io::read_csv(t, non_existent_file, ",");
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsComplexFloatTypes, SmallCSVRead)
{
  MATX_ENTER_HANDLER();

  io::read_csv(this->Av, this->small_complex_csv, ",");
  MATX_TEST_ASSERT_COMPARE(this->pb, this->Av, this->small_complex_csv.c_str(), 0.01);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsComplexFloatTypes, CSVReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  tensor_t<TestType, 2> t{{10, 2}};
  
  // Test reading a non-existent CSV file with complex types
  const std::string non_existent_file = "../test/00_io/non_existent_complex_file.csv";
  ASSERT_THROW({
    io::read_csv(t, non_existent_file, ",");
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, SmallCSVWrite)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
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
  auto t = make_tensor<TestType>({1,10});

  // example-begin read_mat-test-1
  // Read "myvar" from mat file
  io::read_mat(t, "../test/00_io/test.mat", "myvar");
  // example-end read_mat-test-1
  ASSERT_NEAR(t(0,0), 1.456, 0.001);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATReadUninitialized)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  tensor_t<TestType, 2> t;

  io::read_mat(t, "../test/00_io/test.mat", "myvar");

  ASSERT_EQ(t.Size(0), 1);
  ASSERT_EQ(t.Size(1), 10);
  ASSERT_NEAR(t(0,0), 1.456, 0.001);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto t = make_tensor<TestType>({1,10});

  // Test reading a non-existent MAT file
  const std::string non_existent_file = "../test/00_io/non_existent_file.mat";
  ASSERT_THROW({
    io::read_mat(t, non_existent_file, "myvar");
  }, matx::detail::matxException);

  // Test the overloaded read_mat that returns a tensor  
  ASSERT_THROW({
    auto t2 = io::read_mat<decltype(t)>(non_existent_file, "myvar");
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, MATWrite)
{
  MATX_ENTER_HANDLER();

  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
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

TYPED_TEST(FileIoTestsComplexFloatTypes, MATReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;
  auto t = make_tensor<TestType>({1,10});

  // Test reading a non-existent MAT file with complex types
  const std::string non_existent_file = "../test/00_io/non_existent_complex_file.mat";
  ASSERT_THROW({
    io::read_mat(t, non_existent_file, "myvar");
  }, matx::detail::matxException);

  // Test the overloaded read_mat that returns a tensor  
  ASSERT_THROW({
    auto t2 = io::read_mat<decltype(t)>(non_existent_file, "myvar");
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsComplexFloatTypes, NPYReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2, 3});

  // Test reading a non-existent NPY file with complex types
  const std::string non_existent_file = "../test/00_io/non_existent_complex_file.npy";
  ASSERT_THROW({
    io::read_npy(t, non_existent_file);
  }, matx::detail::matxException);

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

TYPED_TEST(FileIoTestsNonComplexFloatTypes, NPYReadUninitialized)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  tensor_t<TestType, 2> t;

  io::read_npy(t, "../test/00_io/test.npy");

  ASSERT_EQ(t.Size(0), 2);
  ASSERT_EQ(t.Size(1), 3);
  ASSERT_NEAR(t(0, 0), 1.5, 0.001);
  ASSERT_NEAR(t(1, 2), 6.5, 0.001);

  MATX_EXIT_HANDLER();
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, NPYReadInitializedShapeMismatch)
{
#ifdef NDEBUG
  GTEST_SKIP() << "NPY shape checks require assertions (NDEBUG must be undefined)";
#else
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({1, 3});

  ASSERT_THROW({
    io::read_npy(t, "../test/00_io/test.npy");
  }, matx::detail::matxException);

  MATX_EXIT_HANDLER();
#endif
}

TYPED_TEST(FileIoTestsNonComplexFloatTypes, NPYReadFileNotFound)
{
  MATX_ENTER_HANDLER();
  using TestType = cuda::std::tuple_element_t<0, TypeParam>;

  auto t = make_tensor<TestType>({2, 3});

  // Test reading a non-existent NPY file
  const std::string non_existent_file = "../test/00_io/non_existent_file.npy";
  ASSERT_THROW({
    io::read_npy(t, non_existent_file);
  }, matx::detail::matxException);

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
