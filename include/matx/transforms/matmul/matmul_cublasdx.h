////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include "matx/core/operator_options.h"
#include "matx/core/capabilities.h"
#include "matx/core/log.h"

#include "matx/core/half_complex.h"
#include "matx/core/half.h"

#include <limits>
#include <format>
#define GEMM_DX_FUNC_PREFIX "gemm_cublasdx_func"

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
#include <libcublasdx.h>

#define LIBCUBLASDX_CHECK(ans)                                                                                           \
  do {                                                                                                               \
    commondxStatusType result = (ans);                                                                             \
    MATX_ASSERT_STR_EXP(result, commondxStatusType::COMMONDX_SUCCESS, matxLibMathdxError, "cuBLASDx failed");\
  } while (0)

namespace matx {
  namespace detail {

// Returns true if the matrix size and data type are supported by cuBLASDx for the given compute capability.
  // Based on table from cuBLASDx documentation:
  // https://docs.nvidia.com/cuda/cublasdx/requirements_func.html#supported-maximal-sizes-with-non-overlapping-a-and-b
  template <typename T>
  __MATX_INLINE__ bool IscuBLASDxSupported(index_t m, index_t n, index_t k, int compute_capability)
  {
    // Using "Restricted AB with C in Shared" column from documentation
  int max_size = 0;

  // Real, half or bfloat16
  if constexpr (std::is_same_v<T, matxFp16> || std::is_same_v<T, matxBf16>) {
    if (compute_capability == 700 || compute_capability == 720) max_size = 128;
    else if (compute_capability == 750) max_size = 104;
    else if (compute_capability == 800 || compute_capability == 870) max_size = 166;
    else if (compute_capability == 860 || compute_capability == 890 || compute_capability == 1200 || compute_capability == 1210) max_size = 129;
      else if (compute_capability == 900 || compute_capability == 1000 || compute_capability == 1010 || compute_capability == 1030 || compute_capability == 1100) max_size = 196;
  }
    // Real, float OR Complex, half/bf16
  else if constexpr (std::is_same_v<T, float> || std::is_same_v<T, matxFp16Complex> || std::is_same_v<T, cuda::std::complex<matxBf16>>) {
    if (compute_capability == 700 || compute_capability == 720) max_size = 90;
    else if (compute_capability == 750) max_size = 73;
    else if (compute_capability == 800 || compute_capability == 870) max_size = 117;
    else if (compute_capability == 860 || compute_capability == 890 || compute_capability == 1200 || compute_capability == 1210) max_size = 91;
      else if (compute_capability == 900 || compute_capability == 1000 || compute_capability == 1010 || compute_capability == 1030 || compute_capability == 1100) max_size = 139;
  }
  // Real, double OR Complex, float
  else if constexpr (std::is_same_v<T, double> || std::is_same_v<T, cuda::std::complex<float>>) {
    if (compute_capability == 700 || compute_capability == 720) max_size = 64;
    else if (compute_capability == 750) max_size = 52;
    else if (compute_capability == 800 || compute_capability == 870) max_size = 83;
    else if (compute_capability == 860 || compute_capability == 890 || compute_capability == 1200 || compute_capability == 1210) max_size = 64;
      else if (compute_capability == 900 || compute_capability == 1000 || compute_capability == 1010 || compute_capability == 1030 || compute_capability == 1100) max_size = 98;
  }
  // Complex, double
  else if constexpr (std::is_same_v<T, cuda::std::complex<double>>) {
    if (compute_capability == 700 || compute_capability == 720) max_size = 45;
    else if (compute_capability == 750) max_size = 36;
    else if (compute_capability == 800 || compute_capability == 870) max_size = 58;
    else if (compute_capability == 860 || compute_capability == 890 || compute_capability == 1200 || compute_capability == 1210) max_size = 45;
      else if (compute_capability == 900 || compute_capability == 1000 || compute_capability == 1010 || compute_capability == 1030 || compute_capability == 1100) max_size = 69;
  }

  if (max_size == 0) {
    return false;
  }

  const auto max_shm = static_cast<size_t>(max_size) * static_cast<size_t>(max_size) * sizeof(T) * 2; // Most SHM both A and B can use
  const auto req_shm = sizeof(T) * (static_cast<size_t>(m) * static_cast<size_t>(k) + static_cast<size_t>(k) * static_cast<size_t>(n));

  // All dimensions must fit in shared memory
  return req_shm <= max_shm;
}

  template <typename InputType>
  class cuBLASDxHelper {
    private:
      index_t m_;       // Output rows (A rows)
      index_t n_;       // Output cols (B cols)
      index_t k_;       // Inner dimension (A cols = B rows)
      int cc_;          // Compute capability
      bool is_complex_; // Whether the type is complex

      template <typename Scalar>
      static std::string FormatScalarLiteral(Scalar value) {
        return std::format(
            "{:.{}g}",
            value,
            std::numeric_limits<Scalar>::max_digits10);
      }

    public:
      // Constructor
      cuBLASDxHelper() = default;

      // Getters
      index_t get_m() const { return m_; }
      index_t get_n() const { return n_; }
      index_t get_k() const { return k_; }
      int get_cc() const { return cc_; }
      bool get_is_complex() const { return is_complex_; }

      // Setters
      void set_m(index_t m) { m_ = m; }
      void set_n(index_t n) { n_ = n; }
      void set_k(index_t k) { k_ = k; }
      void set_cc(int cc) { cc_ = cc; }
      void set_is_complex(bool is_complex) { is_complex_ = is_complex; }

      cublasdxDescriptor GeneratePlan() const {
        cublasdxDescriptor h_;
        LIBCUBLASDX_CHECK(cublasdxCreateDescriptor(&h_));

        // Set the GEMM function
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64(h_, CUBLASDX_OPERATOR_FUNCTION, CUBLASDX_FUNCTION_MM));

        // Set problem size (M, N, K)
        long long int sizes[3] = {static_cast<long long int>(m_), static_cast<long long int>(n_), static_cast<long long int>(k_)};
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64s(h_, CUBLASDX_OPERATOR_SIZE, 3, sizes));

        // Set API type - use shared memory API
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64(h_, CUBLASDX_OPERATOR_API, CUBLASDX_API_SMEM));

        // Set execution mode - block level execution
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64(h_, CUBLASDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_BLOCK));

        // Set precision for A, B, C matrices (all same precision for now)
        commondxPrecision precision;
        if constexpr (std::is_same_v<InputType, matxFp16> || std::is_same_v<InputType, matxFp16Complex>) {
          precision = COMMONDX_PRECISION_F16;
        } else if constexpr (std::is_same_v<InputType, matxBf16> || std::is_same_v<InputType, cuda::std::complex<matxBf16>>) {
          precision = COMMONDX_PRECISION_BF16;
        } else if constexpr (std::is_same_v<InputType, float> || std::is_same_v<InputType, cuda::std::complex<float>>) {
          precision = COMMONDX_PRECISION_F32;
        } else if constexpr (std::is_same_v<InputType, double> || std::is_same_v<InputType, cuda::std::complex<double>>) {
          precision = COMMONDX_PRECISION_F64;
        } else {
          MATX_THROW(matxInvalidParameter, "Unsupported input type for cuBLASDx");
        }

        long long int precisions[3] = {static_cast<long long int>(precision), 
                                       static_cast<long long int>(precision), 
                                       static_cast<long long int>(precision)};
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64s(h_, CUBLASDX_OPERATOR_PRECISION, 3, precisions));

        // Set type (real or complex)
        cublasdxType type = is_complex_ ? CUBLASDX_TYPE_COMPLEX : CUBLASDX_TYPE_REAL;
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64(h_, CUBLASDX_OPERATOR_TYPE, type));

        // Set target compute capability
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64(h_, CUBLASDX_OPERATOR_SM, cc_));

        // Set arrangement - row major for all matrices (MatX default)
        long long int arrangements[3] = {CUBLASDX_ARRANGEMENT_ROW_MAJOR, 
                                         CUBLASDX_ARRANGEMENT_ROW_MAJOR, 
                                         CUBLASDX_ARRANGEMENT_ROW_MAJOR};
        LIBCUBLASDX_CHECK(cublasdxSetOperatorInt64s(h_, CUBLASDX_OPERATOR_ARRANGEMENT, 3, arrangements));

        return h_;
      }

      std::string GetSymbolName() const {
        std::string symbol_name;
        symbol_name += std::to_string(m_);
        symbol_name += "_";
        symbol_name += std::to_string(n_);
        symbol_name += "_";
        symbol_name += std::to_string(k_);
        symbol_name += "_T";
        symbol_name += is_complex_ ? "C" : "R";
        symbol_name += "_CC";
        symbol_name += std::to_string(cc_);

        // Add precision identifier
        if constexpr (std::is_same_v<InputType, matxFp16> || std::is_same_v<InputType, matxFp16Complex>) {
          symbol_name += "_F16";
        } else if constexpr (std::is_same_v<InputType, matxBf16> || std::is_same_v<InputType, cuda::std::complex<matxBf16>>) {
          symbol_name += "_BF16";
        } else if constexpr (std::is_same_v<InputType, float> || std::is_same_v<InputType, cuda::std::complex<float>>) {
          symbol_name += "_F32";
        } else if constexpr (std::is_same_v<InputType, double> || std::is_same_v<InputType, cuda::std::complex<double>>) {
          symbol_name += "_F64";
        }

        // Add CUDA version to the symbol name
#if defined(CUDART_VERSION)
        symbol_name += "_CUDA";
        symbol_name += std::to_string(CUDART_VERSION);
#else
        symbol_name += "_CUDAUNKNOWN";
#endif
        
        return symbol_name;
      }

      void PrintMembers() const {
        std::cout << "m_ = " << m_ << std::endl;
        std::cout << "n_ = " << n_ << std::endl;
        std::cout << "k_ = " << k_ << std::endl;
        std::cout << "cc_ = " << cc_ << std::endl;
        std::cout << "is_complex_ = " << is_complex_ << std::endl;
      }

      bool IsSupported() const {
        // Check basic size requirements
        if (!IscuBLASDxSupported<InputType>(m_, n_, k_, cc_)) {
          MATX_LOG_DEBUG("cuBLASDx not supported: matrix size too large for shared memory");
          return false;
        }

        // For now, only support float and double (and their complex variants)
        // Half and bf16 support can be added later
        if constexpr (std::is_same_v<InputType, float> || 
                      std::is_same_v<InputType, double> ||
                      std::is_same_v<InputType, cuda::std::complex<float>> ||
                      std::is_same_v<InputType, cuda::std::complex<double>>) {
          return true;
        }
        
        return false;
      }

      template <typename OpA, typename OpB>
      bool CheckJITSizeAndTypeRequirements() const {
        using OpAType = typename OpA::value_type;
        using OpBType = typename OpB::value_type;
        
        // A and B must have same type
        if constexpr (!std::is_same_v<OpAType, OpBType>) {
          return false;
        }
        
        // Check supported types for JIT
        if constexpr (!(std::is_same_v<OpAType, float> || 
                        std::is_same_v<OpAType, double> ||
                        std::is_same_v<OpAType, cuda::std::complex<float>> ||
                        std::is_same_v<OpAType, cuda::std::complex<double>>)) {
          return false;
        }
        
        // Check size constraints
        return IscuBLASDxSupported<OpAType>(m_, n_, k_, cc_);
      }

      int GetShmRequired() const {
        // For GEMM, we need shared memory for A (m x k), B (k x n), and C (m x n)
        size_t a_size = static_cast<size_t>(m_) * static_cast<size_t>(k_) * sizeof(InputType);
        size_t b_size = static_cast<size_t>(k_) * static_cast<size_t>(n_) * sizeof(InputType);
        size_t c_size = static_cast<size_t>(m_) * static_cast<size_t>(n_) * sizeof(InputType);
        
        // Total shared memory requirement
        size_t total_shm = a_size + b_size + c_size;
        
        MATX_LOG_DEBUG("cuBLASDx shared memory: A={}, B={}, C={}, Total={}", a_size, b_size, c_size, total_shm);
        return static_cast<int>(total_shm);
      }

      cuda::std::array<int, 3> GetBlockDim() const {
        auto handle = GeneratePlan();
        cuda::std::array<long long int, 3> block_dim = {0, 0, 0};

        LIBCUBLASDX_CHECK(
            cublasdxGetTraitInt64s(handle, CUBLASDX_TRAIT_SUGGESTED_BLOCK_DIM, 3, block_dim.data()));
        MATX_LOG_DEBUG("cuBLASDx suggested block dim: {} {} {}", block_dim[0], block_dim[1], block_dim[2]);
        
        cublasdxDestroyDescriptor(handle);
        
        return cuda::std::array<int, 3>{static_cast<int>(block_dim[0]), 
                                         static_cast<int>(block_dim[1]), 
                                         static_cast<int>(block_dim[2])};
      }

      cuda::std::array<int, 3> GetLeadingDimensions() const {
        auto handle = GeneratePlan();
        cuda::std::array<long long int, 3> ld = {0, 0, 0};

        LIBCUBLASDX_CHECK(
            cublasdxGetTraitInt64s(handle, CUBLASDX_TRAIT_SUGGESTED_LEADING_DIMENSION, 3, ld.data()));
        MATX_LOG_DEBUG("cuBLASDx suggested leading dimensions: {} {} {}", ld[0], ld[1], ld[2]);
        
        cublasdxDestroyDescriptor(handle);
        
        return cuda::std::array<int, 3>{static_cast<int>(ld[0]), 
                                         static_cast<int>(ld[1]), 
                                         static_cast<int>(ld[2])};
      }

      bool GenerateLTOIR(std::set<std::string> &ltoir_symbols) {
        LTOIRData ltoir;
        const auto symbol_name = std::string(GEMM_DX_FUNC_PREFIX) + "_" + GetSymbolName();
        ltoir_symbols.insert(symbol_name);        

        if (detail::GetCache().GetLTOIRCachedBytes(symbol_name) != nullptr) {
          MATX_LOG_DEBUG("cuBLASDx LTOIR found in cache with size {}", detail::GetCache().GetLTOIRCachedBytes(symbol_name)->length);
          return true;
        }

        auto handle = GeneratePlan();

        LIBCUBLASDX_CHECK(cublasdxSetOptionStr(handle, COMMONDX_OPTION_SYMBOL_NAME, symbol_name.c_str())); 

        commondxCode code;
        LIBCUBLASDX_CHECK(commondxCreateCode(&code));

        LIBCUBLASDX_CHECK(commondxSetCodeOptionInt64(code, COMMONDX_OPTION_TARGET_SM, cc_));
        LIBCUBLASDX_CHECK(cublasdxFinalizeCode(code, handle));

        LIBCUBLASDX_CHECK(commondxGetCodeLTOIRSize(code, &ltoir.length));
        ltoir.data = static_cast<char*>(malloc(ltoir.length));
        MATX_ASSERT_STR(ltoir.data != nullptr, matxInvalidParameter, "Failed to allocate LTO IR data");

        LIBCUBLASDX_CHECK(commondxGetCodeLTOIR(code, ltoir.length, ltoir.data));

        MATX_LOG_DEBUG("cuBLASDx Function {}", symbol_name);        
        MATX_LOG_DEBUG("cuBLASDx LTOIR size {}", ltoir.length);
        
        if (!detail::GetCache().StoreLTOIRCachedBytes(symbol_name, ltoir.data, ltoir.length)) {
          free(ltoir.data);
          MATX_LOG_ERROR("Failed to store cuBLASDx LTOIR cached bytes for: {}", symbol_name);
          return false;
        }
        
        // CRITICAL: Set to nullptr after transferring ownership to cache to prevent double-free
        ltoir.data = nullptr;
        ltoir.length = 0;
        
        LIBCUBLASDX_CHECK(commondxDestroyCode(code));
        LIBCUBLASDX_CHECK(cublasdxDestroyDescriptor(handle));
    
        return true;
      }

      std::string GetFuncStr(const std::string &gemm_func_name, float alpha, float beta) const {
        std::string result = R"(
          using value_type = )";
        result += detail::type_to_string<InputType>();
        result += R"(;
          
          // cuBLASDx requires block-level cooperation, so all threads in the block
          // must participate in loading data and executing the GEMM
          extern __shared__ __align__(16) char smem[];
          
          // Partition shared memory for A, B, C matrices
          constexpr size_t a_size = )";
        result += std::to_string(static_cast<int>(m_ * k_));
        result += R"( * sizeof(value_type);
          constexpr size_t b_size = )";
        result += std::to_string(static_cast<int>(k_ * n_));

        result += R"( * sizeof(value_type);
          
          value_type* smem_a = reinterpret_cast<value_type*>(smem);
          value_type* smem_b = reinterpret_cast<value_type*>(smem + a_size);
          value_type* smem_c = reinterpret_cast<value_type*>(smem + a_size + b_size);
          
          // Cooperatively load A and B from global to shared memory using operator()
          // Batch indices are already preset in the operators, so we only need 2D matrix indices
          const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
          const int total_threads = blockDim.x * blockDim.y * blockDim.z;
          // Load A matrix (m x k) - each thread loads multiple elements strided by total_threads
          constexpr index_t a_cols = )";
        result += std::to_string(static_cast<int>(k_));
        result += R"(;
          for (int i = tid; i < )";
        result += std::to_string(static_cast<int>(m_ * k_));
        result += R"(; i += total_threads) {
            const index_t row = i / a_cols;
            const index_t col = i % a_cols;
            smem_a[row * a_cols + col] = a_.template operator()<CapType>(row, col);
          }
          
          // Load B matrix (k x n) - each thread loads multiple elements strided by total_threads
          constexpr index_t b_cols = )";
        result += std::to_string(static_cast<int>(n_));
        result += R"(;
          for (int i = tid; i < )";
        result += std::to_string(static_cast<int>(k_ * n_));
        result += R"(; i += total_threads) {
            const index_t row = i / b_cols;
            const index_t col = i % b_cols;
            smem_b[row * b_cols + col] = b_.template operator()<CapType>(row, col);
          }
          
          __syncthreads();
          
          // Call the cuBLASDx generated GEMM function
          // Signature: void func(value_type* alpha, value_type* a, value_type* b, value_type* beta, value_type* c)
        )";
        using literal_type = cuda::std::conditional_t<
            std::is_same_v<InputType, double> || std::is_same_v<InputType, cuda::std::complex<double>>,
            double,
            float>;
        result += "value_type alpha_val = static_cast<value_type>(" + FormatScalarLiteral(static_cast<literal_type>(alpha)) + ");\n";
        result += "value_type beta_val = static_cast<value_type>(" + FormatScalarLiteral(static_cast<literal_type>(beta)) + ");\n";
        result += gemm_func_name;
        result += R"((&alpha_val, smem_a, smem_b, &beta_val, smem_c);
          
          __syncthreads();
          
          // Each thread returns its portion of the result
          // For vectorized execution, return a Vector; for scalar, return scalar
          if constexpr (CapType::ept == ElementsPerThread::ONE) {
            const int output_idx = threadIdx.x;
            return smem_c[output_idx];
          } else {
            constexpr int EPT = static_cast<int>(CapType::ept);
            const int output_idx = threadIdx.x * EPT;
            Vector<value_type, EPT> result;
            MATX_LOOP_UNROLL
            for (int i = 0; i < EPT; ++i) {
              result.data[i] = smem_c[output_idx + i];
            }
            return result;
          }
        )";

        return result;
      }
  };

  } // namespace detail
} // namespace matx

#else // !MATX_EN_MATHDX || !__CUDACC__

namespace matx {
  namespace detail {

  // Stub implementation when MathDx is not enabled
  template <typename T>
  __MATX_INLINE__ bool IscuBLASDxSupported([[maybe_unused]] index_t m, [[maybe_unused]] index_t n, 
                                           [[maybe_unused]] index_t k, [[maybe_unused]] int compute_capability)
  {
    return false;
  }

  } // namespace detail
} // namespace matx

#endif // MATX_EN_MATHDX && __CUDACC__
