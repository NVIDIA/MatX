////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2024, NVIDIA Corporation
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

#include <cstdio>
#include <cuda/std/ccomplex>
#include "matx.h"
#include "gtest/gtest.h"

using namespace matx;

class PrintTest : public ::testing::Test
{
public:

  tensor_t<cuda::std::complex<double>, 1> A1 {{16}};

  void SetUp() override
  {
    set_print_format_type(MATX_PRINT_FORMAT_DEFAULT);
    A1.SetVals({
        {-9.2466e-01, +9.9114e-01}, {-4.2534e-01, +1.0676e+00}, {-2.6438e+00, -6.2723e-01}, { 1.4518e-01, +3.2016e-01},
        {-1.2087e-01, -3.1101e-01}, {-5.7973e-01, -3.4408e-01}, {-6.2285e-01, -1.1709e+00}, {-3.2839e-01, -5.3706e-01},
        {-1.0745e+00, +1.3390e+00}, {-3.6314e-01, -2.4011e-01}, {-1.6711e+00, +1.2149e+00}, { 2.2655e+00, -2.0518e-01},
        { 3.1168e-01, +1.2999e+00}, {-1.8419e-01, +2.1812e-01}, { 1.2866e+00, -1.2135e+00}, { 1.1820e+00, -1.3723e+00}
    });
  }

  void TearDown() override
  {
    set_print_format_type(MATX_PRINT_FORMAT_DEFAULT);
  }

  template <typename Op>
  void print_checker(const Op& op, const char* expected_text)
  {
    FILE *fp = fopen("/tmp/matx_test_output.txt","w");
    ASSERT_FALSE(fp == NULL) << "Unable to open /tmp/matx_test_output.txt for reading.";

    fprint(fp, op);
    fclose(fp);

    fp = fopen("/tmp/matx_test_output.txt","r");
    ASSERT_FALSE(fp == NULL) << "Unable to open /tmp/matx_test_output.txt for writing.";

    char actual_text[2048] {0};
    fseek(fp, 0, SEEK_END);
    size_t file_size = ftell(fp);
    ASSERT_TRUE(file_size < 2047) << "/tmp/matx_test_output.txt larger than expected.";

    rewind(fp);
    size_t fread_result = fread(actual_text, file_size, 1, fp);
    ASSERT_TRUE(fread_result == 1) << "Unable to read /tmp/matx_test_output.txt.";

    int cmp_result = strncmp(expected_text,actual_text,2047);
    if (cmp_result != 0)
    {
        printf("Actual text (length %lu):\n%s\n",strnlen(actual_text,2047),actual_text);
        printf("Expected text (length %lu):\n%s\n",strnlen(expected_text,2047),expected_text);
    }
    ASSERT_EQ(cmp_result,0) << "Strcmp difference between actual and expected text";
    fclose(fp);
  }
};

TEST_F(PrintTest, DefaultTest1)
{
  MATX_ENTER_HANDLER();
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_DEFAULT, pft);

  print_checker(A1,
      "Tensor{complex<double>} Rank: 1, Sizes:[16], Strides:[1]\n"
      "000000: -9.2466e-01+9.9114e-01j \n"
      "000001: -4.2534e-01+1.0676e+00j \n"
      "000002: -2.6438e+00-6.2723e-01j \n"
      "000003:  1.4518e-01+3.2016e-01j \n"
      "000004: -1.2087e-01-3.1101e-01j \n"
      "000005: -5.7973e-01-3.4408e-01j \n"
      "000006: -6.2285e-01-1.1709e+00j \n"
      "000007: -3.2839e-01-5.3706e-01j \n"
      "000008: -1.0745e+00+1.3390e+00j \n"
      "000009: -3.6314e-01-2.4011e-01j \n"
      "000010: -1.6711e+00+1.2149e+00j \n"
      "000011:  2.2655e+00-2.0518e-01j \n"
      "000012:  3.1168e-01+1.2999e+00j \n"
      "000013: -1.8419e-01+2.1812e-01j \n"
      "000014:  1.2866e+00-1.2135e+00j \n"
      "000015:  1.1820e+00-1.3723e+00j \n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, DefaultTest2)
{
  MATX_ENTER_HANDLER();
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_DEFAULT, pft);

  auto A2 = reshape(A1, {4,4});

  print_checker(A2,
      "Operator{complex<double>} Rank: 2, Sizes:[4, 4]\n"
      "000000: -9.2466e-01+9.9114e-01j -4.2534e-01+1.0676e+00j -2.6438e+00-6.2723e-01j  1.4518e-01+3.2016e-01j \n"
      "000001: -1.2087e-01-3.1101e-01j -5.7973e-01-3.4408e-01j -6.2285e-01-1.1709e+00j -3.2839e-01-5.3706e-01j \n"
      "000002: -1.0745e+00+1.3390e+00j -3.6314e-01-2.4011e-01j -1.6711e+00+1.2149e+00j  2.2655e+00-2.0518e-01j \n"
      "000003:  3.1168e-01+1.2999e+00j -1.8419e-01+2.1812e-01j  1.2866e+00-1.2135e+00j  1.1820e+00-1.3723e+00j \n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, DefaultTest3)
{
  MATX_ENTER_HANDLER();
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_DEFAULT, pft);

  auto A3 = reshape(A1, {2,2,4});

  print_checker(A3,
      "Operator{complex<double>} Rank: 3, Sizes:[2, 2, 4]\n"
      "[000000,:,:]\n"
      "000000: -9.2466e-01+9.9114e-01j -4.2534e-01+1.0676e+00j -2.6438e+00-6.2723e-01j  1.4518e-01+3.2016e-01j \n"
      "000001: -1.2087e-01-3.1101e-01j -5.7973e-01-3.4408e-01j -6.2285e-01-1.1709e+00j -3.2839e-01-5.3706e-01j \n"
      "\n"
      "[000001,:,:]\n"
      "000000: -1.0745e+00+1.3390e+00j -3.6314e-01-2.4011e-01j -1.6711e+00+1.2149e+00j  2.2655e+00-2.0518e-01j \n"
      "000001:  3.1168e-01+1.2999e+00j -1.8419e-01+2.1812e-01j  1.2866e+00-1.2135e+00j  1.1820e+00-1.3723e+00j \n\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, DefaultTest4)
{
  MATX_ENTER_HANDLER();
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_DEFAULT, pft);

  auto A4 = reshape(A1, {2,2,2,2});

  print_checker(A4,
      "Operator{complex<double>} Rank: 4, Sizes:[2, 2, 2, 2]\n"
      "[000000,000000,:,:]\n"
      "000000: -9.2466e-01+9.9114e-01j -4.2534e-01+1.0676e+00j \n"
      "000001: -2.6438e+00-6.2723e-01j  1.4518e-01+3.2016e-01j \n"
      "\n"
      "[000000,000001,:,:]\n"
      "000000: -1.2087e-01-3.1101e-01j -5.7973e-01-3.4408e-01j \n"
      "000001: -6.2285e-01-1.1709e+00j -3.2839e-01-5.3706e-01j \n"
      "\n"
      "[000001,000000,:,:]\n"
      "000000: -1.0745e+00+1.3390e+00j -3.6314e-01-2.4011e-01j \n"
      "000001: -1.6711e+00+1.2149e+00j  2.2655e+00-2.0518e-01j \n"
      "\n"
      "[000001,000001,:,:]\n"
      "000000:  3.1168e-01+1.2999e+00j -1.8419e-01+2.1812e-01j \n"
      "000001:  1.2866e+00-1.2135e+00j  1.1820e+00-1.3723e+00j \n\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, DefaultTest5)
{
  MATX_ENTER_HANDLER();
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_DEFAULT, pft);
  
  auto testSlice = matx::slice<0>(A1, {0}, {matx::matxDropDim});

  print_checker(testSlice,
      "Tensor{complex<double>} Rank: 0, Sizes:[], Strides:[]\n"
      "-9.2466e-01+9.9114e-01j \n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, MlabTest1)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_MLAB, pft);

  print_checker(A1,
      "Tensor{complex<double>} Rank: 1, Sizes:[16], Strides:[1]\n"
      "[-9.2466e-01+9.9114e-01j ,\n"
      " -4.2534e-01+1.0676e+00j ,\n"
      " -2.6438e+00-6.2723e-01j ,\n"
      "  1.4518e-01+3.2016e-01j ,\n"
      " -1.2087e-01-3.1101e-01j ,\n"
      " -5.7973e-01-3.4408e-01j ,\n"
      " -6.2285e-01-1.1709e+00j ,\n"
      " -3.2839e-01-5.3706e-01j ,\n"
      " -1.0745e+00+1.3390e+00j ,\n"
      " -3.6314e-01-2.4011e-01j ,\n"
      " -1.6711e+00+1.2149e+00j ,\n"
      "  2.2655e+00-2.0518e-01j ,\n"
      "  3.1168e-01+1.2999e+00j ,\n"
      " -1.8419e-01+2.1812e-01j ,\n"
      "  1.2866e+00-1.2135e+00j ,\n"
      "  1.1820e+00-1.3723e+00j ]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, MlabTest2)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_MLAB, pft);

  auto A2 = reshape(A1, {4,4});

  print_checker(A2,
      "Operator{complex<double>} Rank: 2, Sizes:[4, 4]\n"
      "[-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j , -2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ; ...\n"
      " -1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j , -6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ; ...\n"
      " -1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j , -1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ; ...\n"
      "  3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ,  1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, MlabTest3)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_MLAB, pft);

  auto A3 = reshape(A1, {2,2,4});

  print_checker(A3,
      "Operator{complex<double>} Rank: 3, Sizes:[2, 2, 4]\n"
      "cat(3, ...\n"
      "       [-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j , -2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ; ...\n"
      "        -1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j , -6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ], ...\n"
      "       [-1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j , -1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ; ...\n"
      "         3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ,  1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ])\n\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, MlabTest4)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_MLAB, pft);

  auto A4 = reshape(A1, {2,2,2,2});

  print_checker(A4,
      "Operator{complex<double>} Rank: 4, Sizes:[2, 2, 2, 2]\n"
      "cat(4, ...\n"
      "       cat(3, ...\n"
      "              [-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j ; ...\n"
      "               -2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ], ...\n"
      "              [-1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j ; ...\n"
      "               -6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ]), ...\n"
      "       cat(3, ...\n"
      "              [-1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j ; ...\n"
      "               -1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ], ...\n"
      "              [ 3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ; ...\n"
      "                1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ]))\n\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, MlabTest5)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_MLAB, pft);

  auto testSlice = matx::slice<0>(A1, {0}, {matx::matxDropDim});

  print_checker(testSlice,
      "Tensor{complex<double>} Rank: 0, Sizes:[], Strides:[]\n"
      "-9.2466e-01+9.9114e-01j \n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, PythonTest1)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_PYTHON, pft);

  print_checker(A1,
      "Tensor{complex<double>} Rank: 1, Sizes:[16], Strides:[1]\n"
      "[-9.2466e-01+9.9114e-01j ,\n"
      " -4.2534e-01+1.0676e+00j ,\n"
      " -2.6438e+00-6.2723e-01j ,\n"
      "  1.4518e-01+3.2016e-01j ,\n"
      " -1.2087e-01-3.1101e-01j ,\n"
      " -5.7973e-01-3.4408e-01j ,\n"
      " -6.2285e-01-1.1709e+00j ,\n"
      " -3.2839e-01-5.3706e-01j ,\n"
      " -1.0745e+00+1.3390e+00j ,\n"
      " -3.6314e-01-2.4011e-01j ,\n"
      " -1.6711e+00+1.2149e+00j ,\n"
      "  2.2655e+00-2.0518e-01j ,\n"
      "  3.1168e-01+1.2999e+00j ,\n"
      " -1.8419e-01+2.1812e-01j ,\n"
      "  1.2866e+00-1.2135e+00j ,\n"
      "  1.1820e+00-1.3723e+00j ]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, PythonTest2)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_PYTHON, pft);

  auto A2 = reshape(A1, {4,4});

  print_checker(A2,
      "Operator{complex<double>} Rank: 2, Sizes:[4, 4]\n"
      "[[-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j , -2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ],\n"
      " [-1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j , -6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ],\n"
      " [-1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j , -1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ],\n"
      " [ 3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ,  1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ]]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, PythonTest3)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_PYTHON, pft);

  auto A3 = reshape(A1, {2,2,4});

  print_checker(A3,
      "Operator{complex<double>} Rank: 3, Sizes:[2, 2, 4]\n"
      "[[[-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j , -2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ],\n"
      "  [-1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j , -6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ]],\n"
      " [[-1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j , -1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ],\n"
      "  [ 3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ,  1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ]]]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, PythonTest4)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_PYTHON, pft);

  auto A4 = reshape(A1, {2,2,2,2});

  print_checker(A4,
      "Operator{complex<double>} Rank: 4, Sizes:[2, 2, 2, 2]\n"
      "[[[[-9.2466e-01+9.9114e-01j , -4.2534e-01+1.0676e+00j ],\n"
      "   [-2.6438e+00-6.2723e-01j ,  1.4518e-01+3.2016e-01j ]],\n"
      "  [[-1.2087e-01-3.1101e-01j , -5.7973e-01-3.4408e-01j ],\n"
      "   [-6.2285e-01-1.1709e+00j , -3.2839e-01-5.3706e-01j ]]],\n"
      " [[[-1.0745e+00+1.3390e+00j , -3.6314e-01-2.4011e-01j ],\n"
      "   [-1.6711e+00+1.2149e+00j ,  2.2655e+00-2.0518e-01j ]],\n"
      "  [[ 3.1168e-01+1.2999e+00j , -1.8419e-01+2.1812e-01j ],\n"
      "   [ 1.2866e+00-1.2135e+00j ,  1.1820e+00-1.3723e+00j ]]]]\n");

  MATX_EXIT_HANDLER();
}

TEST_F(PrintTest, PythonTest5)
{
  MATX_ENTER_HANDLER();
  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  auto pft = get_print_format_type();
  ASSERT_EQ(MATX_PRINT_FORMAT_PYTHON, pft);

  auto testSlice = matx::slice<0>(A1, {0}, {matx::matxDropDim});

  print_checker(testSlice,
      "Tensor{complex<double>} Rank: 0, Sizes:[], Strides:[]\n"
      "-9.2466e-01+9.9114e-01j \n");

  MATX_EXIT_HANDLER();
}


