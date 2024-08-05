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

#pragma once

#include <cstdio>
#include <exception>
#include <sstream>
#ifdef __CUDACC__
#include <cuda.h>
#endif

#include "matx/core/stacktrace.h"

namespace matx
{

  /**
   * @brief MatX error codes
   * 
   */
  enum matxError_t
  {
    matxSuccess,
    matxIOError,
    matxOutOfMemory,
    matxNotSupported,
    matxInvalidParameter,
    matxInvalidDim,
    matxInvalidSize,
    matxCudaError,
    matxCufftError,
    matxMatMulError,
    matxAssertError,
    matxInvalidType,
    matxLUError,
    matxInverseError,
    matxSolverError,
    matxcuTensorError,
    matxInvalidExecutor
  };

  static constexpr const char *matxErrorString(matxError_t e)
  {
    switch (e)
    {
    case matxSuccess:
      return "matxSuccess";
    case matxNotSupported:
      return "matxNotSupported";
    case matxInvalidParameter:
      return "matxInvalidParameter";
    case matxInvalidDim:
      return "matxInvalidDim";
    case matxInvalidSize:
      return "matxInvalidSize";
    case matxCudaError:
      return "matxCudaError";
    case matxCufftError:
      return "matxCufftError";
    case matxMatMulError:
      return "matxMatMulError";
    case matxOutOfMemory:
      return "matxOutOfMemory";
    case matxIOError:
      return "matxIOError";
    case matxAssertError:
      return "matxAssertError";
    case matxInvalidType:
      return "matxInvalidType";
    case matxLUError:
      return "matxLUError";
    case matxInverseError:
      return "matxInverseError";
    case matxSolverError:
      return "matxSolverError";
    case matxcuTensorError:
      break;
    default:
      return "Unknown";
    };

    return "Unknown";
  }

  namespace detail {
  struct matxException : public std::exception
  {
    matxError_t e;
    char str[400];
    std::stringstream stack;

    /**
     * @brief Throw an exception and print a stack trace
     * 
     * @param error 
     * @param s 
     * @param file 
     * @param line 
     */
    matxException(matxError_t error, const char *s, const char *file, int line)
        : e(error)
    {
      snprintf(str, sizeof(str), "matxException (%s: %s) - %s:%d\n",
               matxErrorString(error), s, file, line);
      detail::printStackTrace(stack);
    }

    matxException(matxError_t error, const std::string &s, const char *file, int line)
        : e(error)
    {
      snprintf(str, s.size(), "matxException (%s: %s) - %s:%d\n",
               matxErrorString(error), s.c_str(), file, line);
      detail::printStackTrace(stack);
    }

    const char *what() const throw() { return str; }
  };
  }

#define MATX_ENTER_HANDLER() \
  try                        \
  {

#define MATX_EXIT_HANDLER()                                     \
  }                                                             \
  catch (matx::detail::matxException & e)                                     \
  {                                                             \
    fprintf(stderr, "%s\n", e.what());                          \
    fprintf(stderr, "Stack Trace:\n%s", e.stack.str().c_str()); \
    exit(1);                                                    \
  }

#define MATX_THROW(e, str)                           \
  {                                                  \
    throw matx::detail::matxException(e, str, __FILE__, __LINE__); \
  }

#ifndef NDEBUG
  #define MATX_ASSERT(a, error) \
  {                           \
    if ((a) != true)          \
    {                         \
      MATX_THROW(error, #a);  \
    }                         \
  }

  #define MATX_ASSERT_STR(a, error, str) \
  {                                    \
    if ((a) != true)                   \
    {                                  \
      MATX_THROW(error, #a ": " str);  \
    }                                  \
  }

  #define MATX_ASSERT_STR_EXP(a, expected, error, str) \
  {                                    \
    auto tmp = a;                      \
    if ((tmp != expected))                   \
    {                                  \
      std::cout << #a ": " << str << "(" << tmp << " != " << expected << ")\n";\
      MATX_THROW(error, "");  \
    }                                  \
  }  
  
#else
  #define MATX_ASSERT(a, error) {}
  #define MATX_ASSERT_STR(a, error, str) {}
  #define MATX_ASSERT_STR_EXP(a, expected, error, str) {}
#endif

#define MATX_STATIC_ASSERT(a, error)    \
  {                                     \
    static_assert((a), #error ": " #a); \
  }

#define MATX_STATIC_ASSERT_STR(a, error, str) \
  {                                           \
    static_assert((a), #error ": " #str);       \
  }

#define MATX_CUDA_CHECK(e)                                      \
  if (e != cudaSuccess)                                         \
  {                                                             \
    fprintf(stderr, "%s:%d CUDA Error: %s\n", __FILE__,__LINE__, cudaGetErrorString(e)); \
    MATX_THROW(matx::matxCudaError, cudaGetErrorString(e));           \
  }

// Macro for checking cuda errors following a cuda launch or api call
#define CUDA_CHECK_LAST_ERROR()        \
  {                                    \
    const auto e = cudaGetLastError(); \
    MATX_CUDA_CHECK(e);                \
  }

// This macro asserts compatible dimensions of current class to an operator.
#define ASSERT_COMPATIBLE_OP_SIZES(op)                               \
  if constexpr (Rank() > 0) {                                        \
    bool compatible = true;                                          \
    _Pragma("unroll")                                                \
    for (int32_t i = 0; i < Rank(); i++) {                           \
      [[maybe_unused]] index_t size = matx::detail::get_expanded_size<Rank()>(op, i);       \
      compatible = (size == 0 || size == Size(i));                   \
    }                                                                \
    if (!compatible) { \
      std::cerr << "Incompatible operator sizes: ("; \
      for (int32_t i = 0; i < Rank(); i++) { \
        std::cerr << Size(i); \
        if (i != Rank() - 1) { \
          std::cerr << ","; \
        } \
      } \
      std::cerr << ") not compatible with ("; \
      for (int32_t i = 0; i < Rank(); i++) { \
        std::cerr << matx::detail::get_expanded_size<Rank()>(op, i); \
        if (i != Rank() - 1) { \
          std::cerr << ","; \
        } \
      } \
      std::cerr << ")" << std::endl; \
      MATX_THROW(matxInvalidSize, "Incompatible operator sizes"); \
    }                   \
  }    

} // end namespace matx
