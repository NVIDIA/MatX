////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/cache.h"
#include "matx/core/capabilities.h"
#include "matx/core/log.h"
#include "matx/core/operator_options.h"

#define SOLVER_DX_FUNC_PREFIX "solver_cusolverdx_func"

#if defined(MATX_EN_MATHDX) && defined(__CUDACC__)
#include <libcusolverdx.h>

#define LIBCUSOLVERDX_CHECK(ans)                                                                                      \
  do {                                                                                                                \
    commondxStatusType result = (ans);                                                                                \
    MATX_ASSERT_STR_EXP(result, commondxStatusType::COMMONDX_SUCCESS, matxLibMathdxError, "cuSolverDx failed");      \
  } while (0)

namespace matx {
namespace detail {

template <typename InputType>
class cuSolverDxHelper {
private:
  index_t m_ = 0;
  index_t n_ = 0;
  index_t k_ = 0;
  int cc_ = 0;
  cusolverdxFunction function_ = CUSOLVERDX_FUNCTION_POTRF;
  SolverFillMode fill_mode_ = SolverFillMode::UPPER;
  cusolverdxJob job_ = CUSOLVERDX_JOB_NO_VECTORS;

  static constexpr bool is_supported_value_type =
      cuda::std::is_same_v<InputType, float> ||
      cuda::std::is_same_v<InputType, double> ||
      cuda::std::is_same_v<InputType, cuda::std::complex<float>> ||
      cuda::std::is_same_v<InputType, cuda::std::complex<double>>;

  static constexpr commondxPrecision GetPrecision()
  {
    if constexpr (cuda::std::is_same_v<InputType, float> ||
                  cuda::std::is_same_v<InputType, cuda::std::complex<float>>) {
      return COMMONDX_PRECISION_F32;
    }
    else if constexpr (cuda::std::is_same_v<InputType, double> ||
                       cuda::std::is_same_v<InputType, cuda::std::complex<double>>) {
      return COMMONDX_PRECISION_F64;
    }
    else {
      return COMMONDX_PRECISION_F32;
    }
  }

  static constexpr cusolverdxType GetType()
  {
    if constexpr (is_complex_v<InputType>) {
      return CUSOLVERDX_TYPE_COMPLEX;
    }
    else {
      return CUSOLVERDX_TYPE_REAL;
    }
  }

  size_t GetSizeCount() const
  {
    switch (function_) {
      case CUSOLVERDX_FUNCTION_POTRF:
      case CUSOLVERDX_FUNCTION_POTRS:
      case CUSOLVERDX_FUNCTION_HEEV:
      case CUSOLVERDX_FUNCTION_HTEV:
        return 1;
      case CUSOLVERDX_FUNCTION_GESV_NO_PIVOT:
      case CUSOLVERDX_FUNCTION_GESV_PARTIAL_PIVOT:
      case CUSOLVERDX_FUNCTION_GELS:
        return 3;
      default:
        return 2;
    }
  }

  bool UsesFillMode() const
  {
    switch (function_) {
      case CUSOLVERDX_FUNCTION_POTRF:
      case CUSOLVERDX_FUNCTION_POTRS:
      case CUSOLVERDX_FUNCTION_POSV:
      case CUSOLVERDX_FUNCTION_HEEV:
      case CUSOLVERDX_FUNCTION_HTEV:
        return true;
      default:
        return false;
    }
  }

  bool UsesJob() const
  {
    switch (function_) {
      case CUSOLVERDX_FUNCTION_HEEV:
      case CUSOLVERDX_FUNCTION_HTEV:
        return true;
      default:
        return false;
    }
  }

  class DescriptorHandle {
  private:
    cusolverdxDescriptor handle_{};

  public:
    DescriptorHandle() = default;
    DescriptorHandle(const DescriptorHandle&) = delete;
    DescriptorHandle& operator=(const DescriptorHandle&) = delete;
    DescriptorHandle(DescriptorHandle&& other) noexcept : handle_(other.handle_)
    {
      other.handle_ = {};
    }
    DescriptorHandle& operator=(DescriptorHandle&& other) noexcept
    {
      if (this != &other) {
        reset();
        handle_ = other.handle_;
        other.handle_ = {};
      }
      return *this;
    }
    ~DescriptorHandle()
    {
      reset();
    }

    cusolverdxDescriptor get() const { return handle_; }
    cusolverdxDescriptor* put() { return &handle_; }

  private:
    void reset()
    {
      if (handle_ != 0) {
        cusolverdxDestroyDescriptor(handle_);
        handle_ = {};
      }
    }
  };

  class CodeHandle {
  private:
    commondxCode code_{};

  public:
    CodeHandle() = default;
    CodeHandle(const CodeHandle&) = delete;
    CodeHandle& operator=(const CodeHandle&) = delete;
    CodeHandle(CodeHandle&& other) noexcept : code_(other.code_)
    {
      other.code_ = {};
    }
    CodeHandle& operator=(CodeHandle&& other) noexcept
    {
      if (this != &other) {
        reset();
        code_ = other.code_;
        other.code_ = {};
      }
      return *this;
    }
    ~CodeHandle()
    {
      reset();
    }

    commondxCode get() const { return code_; }
    commondxCode* put() { return &code_; }

  private:
    void reset()
    {
      if (code_ != 0) {
        commondxDestroyCode(code_);
        code_ = {};
      }
    }
  };

  long long int GetKernelSharedMemoryFloor() const
  {
    const auto elem_size = static_cast<long long int>(sizeof(InputType));
    switch (function_) {
      case CUSOLVERDX_FUNCTION_POTRF: {
        const auto elems = static_cast<long long int>(n_) * static_cast<long long int>(n_);
        return elems * elem_size + static_cast<long long int>(sizeof(int));
      }
      case CUSOLVERDX_FUNCTION_GESV_NO_PIVOT:
      case CUSOLVERDX_FUNCTION_GESV_PARTIAL_PIVOT: {
        const auto elems = static_cast<long long int>(n_) * static_cast<long long int>(n_);
        return 2 * elems * elem_size + (static_cast<long long int>(n_) + 1) * static_cast<long long int>(sizeof(int));
      }
      default:
        return static_cast<long long int>(m_) * static_cast<long long int>(n_) * elem_size +
               static_cast<long long int>(sizeof(int));
    }
  }

  std::string GetTraitSymbolName(cusolverdxDescriptor handle) const
  {
    size_t symbol_name_size = 0;
    LIBCUSOLVERDX_CHECK(cusolverdxGetTraitStrSize(handle, CUSOLVERDX_TRAIT_SYMBOL_NAME, &symbol_name_size));

    std::string symbol_name(symbol_name_size, '\0');
    LIBCUSOLVERDX_CHECK(cusolverdxGetTraitStr(handle,
                                              CUSOLVERDX_TRAIT_SYMBOL_NAME,
                                              symbol_name.size(),
                                              symbol_name.data()));
    if (!symbol_name.empty() && symbol_name.back() == '\0') {
      symbol_name.pop_back();
    }

    return symbol_name;
  }

public:
  cuSolverDxHelper() = default;

  void set_m(index_t m) { m_ = m; }
  void set_n(index_t n) { n_ = n; }
  void set_k(index_t k) { k_ = k; }
  void set_cc(int cc) { cc_ = cc; }
  void set_function(cusolverdxFunction function) { function_ = function; }
  void set_fill_mode(SolverFillMode fill_mode) { fill_mode_ = fill_mode; }
  void set_job(cusolverdxJob job) { job_ = job; }

  index_t get_m() const { return m_; }
  index_t get_n() const { return n_; }
  index_t get_k() const { return k_; }
  int get_cc() const { return cc_; }
  cusolverdxFunction get_function() const { return function_; }
  SolverFillMode get_fill_mode() const { return fill_mode_; }

  DescriptorHandle GeneratePlan() const
  {
    MATX_ASSERT_STR(is_supported_value_type, matxInvalidParameter, "Unsupported input type for cuSolverDx");

    DescriptorHandle h_;
    LIBCUSOLVERDX_CHECK(cusolverdxCreateDescriptor(h_.put()));

    long long int sizes[3] = {
        static_cast<long long int>(m_),
        static_cast<long long int>(n_),
        static_cast<long long int>(k_)};
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64s(h_.get(), CUSOLVERDX_OPERATOR_SIZE, GetSizeCount(), sizes));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_TYPE, GetType()));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_PRECISION, GetPrecision()));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_SM, cc_));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_EXECUTION, COMMONDX_EXECUTION_BLOCK));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_API, CUSOLVERDX_API_SMEM));
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_FUNCTION, function_));

    long long int arrangements[2] = {CUSOLVERDX_ARRANGEMENT_ROW_MAJOR, CUSOLVERDX_ARRANGEMENT_ROW_MAJOR};
    LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64s(h_.get(), CUSOLVERDX_OPERATOR_ARRANGEMENT, 2, arrangements));

    if (UsesFillMode()) {
      const auto fill = fill_mode_ == SolverFillMode::LOWER ? CUSOLVERDX_FILL_MODE_LOWER : CUSOLVERDX_FILL_MODE_UPPER;
      LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_FILL_MODE, fill));
    }

    if (UsesJob()) {
      LIBCUSOLVERDX_CHECK(cusolverdxSetOperatorInt64(h_.get(), CUSOLVERDX_OPERATOR_JOB, job_));
    }

    return h_;
  }

  std::string GetSymbolName() const
  {
    std::string symbol_name;
    symbol_name += std::to_string(static_cast<int>(function_));
    symbol_name += "_";
    symbol_name += std::to_string(m_);
    symbol_name += "_";
    symbol_name += std::to_string(n_);
    symbol_name += "_";
    symbol_name += std::to_string(k_);
    symbol_name += "_F";
    symbol_name += std::to_string(static_cast<int>(fill_mode_));
    symbol_name += "_J";
    symbol_name += std::to_string(static_cast<int>(job_));
    symbol_name += "_T";
    symbol_name += is_complex_v<InputType> ? "C" : "R";
    symbol_name += "_CC";
    symbol_name += std::to_string(cc_);

    if constexpr (cuda::std::is_same_v<InputType, float> || cuda::std::is_same_v<InputType, cuda::std::complex<float>>) {
      symbol_name += "_F32";
    }
    else if constexpr (cuda::std::is_same_v<InputType, double> || cuda::std::is_same_v<InputType, cuda::std::complex<double>>) {
      symbol_name += "_F64";
    }

#if defined(CUDART_VERSION)
    symbol_name += "_CUDA";
    symbol_name += std::to_string(CUDART_VERSION);
#else
    symbol_name += "_CUDAUNKNOWN";
#endif

    return symbol_name;
  }

  bool IsSupported() const
  {
    if constexpr (!is_supported_value_type) {
      return false;
    }
    else {
      if (m_ <= 0 || n_ <= 0) {
        return false;
      }
      auto handle = GeneratePlan();
      long long int shm = 0;
      const bool supported =
          cusolverdxGetTraitInt64(handle.get(), CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE, &shm) == COMMONDX_SUCCESS;
      return supported;
    }
  }

  int GetShmRequired() const
  {
    auto handle = GeneratePlan();
    long long int shared_memory_size = 0;
    LIBCUSOLVERDX_CHECK(cusolverdxGetTraitInt64(handle.get(), CUSOLVERDX_TRAIT_SHARED_MEMORY_SIZE, &shared_memory_size));

    return static_cast<int>(cuda::std::max(shared_memory_size, GetKernelSharedMemoryFloor()));
  }

  cuda::std::array<int, 3> GetBlockDim() const
  {
    auto handle = GeneratePlan();
    cuda::std::array<long long int, 3> block_dim = {0, 0, 0};
    LIBCUSOLVERDX_CHECK(cusolverdxGetTraitInt64s(handle.get(), CUSOLVERDX_TRAIT_BLOCK_DIM, 3, block_dim.data()));
    return cuda::std::array<int, 3>{static_cast<int>(block_dim[0]),
                                    static_cast<int>(block_dim[1]),
                                    static_cast<int>(block_dim[2])};
  }

  cuda::std::array<int, 2> GetBlockDimRange() const
  {
    return cuda::std::array<int, 2>{32, 1024};
  }

  int GetWorkspaceSize() const
  {
    auto handle = GeneratePlan();
    long long int workspace_size = 0;
    LIBCUSOLVERDX_CHECK(cusolverdxGetTraitInt64(handle.get(), CUSOLVERDX_TRAIT_WORKSPACE_SIZE, &workspace_size));
    return static_cast<int>(workspace_size);
  }

  bool GenerateLTOIR(std::set<std::string> &ltoir_symbols)
  {
    const auto symbol_name = std::string(SOLVER_DX_FUNC_PREFIX) + "_" + GetSymbolName();
    ltoir_symbols.insert(symbol_name);

    if (detail::GetCache().GetLTOIRCachedBytes(symbol_name) != nullptr) {
      return true;
    }

    auto handle = GeneratePlan();
    LIBCUSOLVERDX_CHECK(cusolverdxSetOptionStr(handle.get(), COMMONDX_OPTION_SYMBOL_NAME, symbol_name.c_str()));
    const auto trait_symbol_name = GetTraitSymbolName(handle.get());
    MATX_ASSERT_STR(trait_symbol_name == symbol_name,
                    matxInvalidParameter,
                    "cuSolverDx returned an unexpected symbol name");

    CodeHandle code;
    LIBCUSOLVERDX_CHECK(commondxCreateCode(code.put()));
    LIBCUSOLVERDX_CHECK(commondxSetCodeOptionInt64(code.get(), COMMONDX_OPTION_TARGET_SM, cc_));
    LIBCUSOLVERDX_CHECK(cusolverdxFinalizeCode(code.get(), handle.get()));

    size_t ltoir_length = 0;
    LIBCUSOLVERDX_CHECK(commondxGetCodeLTOIRSize(code.get(), &ltoir_length));
    if (ltoir_length == 0) {
      MATX_LOG_ERROR("cuSolverDx returned empty LTOIR for: {}", symbol_name);
      return false;
    }
    std::unique_ptr<char, decltype(&free)> ltoir(static_cast<char*>(malloc(ltoir_length)), &free);
    if (ltoir == nullptr) {
      MATX_LOG_ERROR("Failed to allocate cuSolverDx LTOIR data for: {}", symbol_name);
      return false;
    }
    LIBCUSOLVERDX_CHECK(commondxGetCodeLTOIR(code.get(), ltoir_length, ltoir.get()));

    char *ltoir_data = ltoir.release();
    if (!detail::GetCache().StoreLTOIRCachedBytes(symbol_name, ltoir_data, ltoir_length)) {
      MATX_LOG_ERROR("Failed to store cuSolverDx LTOIR cached bytes for: {}", symbol_name);
      return false;
    }

    return true;
  }

  std::string GetPotrfFuncStr(const std::string &solver_func_name) const
  {
    std::string result = R"(
      using value_type = )";
    result += detail::type_to_string<InputType>();
    result += R"(;
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t elems = n * n;
      extern __shared__ __align__(16) char smem[];
      value_type* smem_a = reinterpret_cast<value_type*>(smem);
      int* info = reinterpret_cast<int*>(smem + elems * sizeof(value_type));
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        value_type a_value{};
        if constexpr (Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (Rank() == 4) {
          a_value = a_.template operator()<CapType>(idx[0], idx[1], row, col);
        }
        if (valid) {
          smem_a[linear] = a_value;
        }
      }

      if (tid == 0) {
        *info = 0;
      }
      __syncthreads();
    )";
    result += solver_func_name;
    result += R"((smem_a, info);
      __syncthreads();

      if (tid < elems) {
        return smem_a[tid];
      }
      return value_type{};
    )";
    return result;
  }

  std::string GetGesvInverseFuncStr(const std::string &solver_func_name) const
  {
    std::string result = R"(
      using value_type = )";
    result += detail::type_to_string<InputType>();
    result += R"(;
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t elems = n * n;
      extern __shared__ __align__(16) char smem[];
      value_type* smem_a = reinterpret_cast<value_type*>(smem);
      value_type* smem_b = reinterpret_cast<value_type*>(smem + elems * sizeof(value_type));
      int* ipiv = reinterpret_cast<int*>(smem + (2 * elems * sizeof(value_type)));
      int* info = reinterpret_cast<int*>(smem + (2 * elems * sizeof(value_type)) + n * sizeof(int));
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        value_type a_value{};
        if constexpr (Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (Rank() == 4) {
          a_value = a_.template operator()<CapType>(idx[0], idx[1], row, col);
        }
        if (valid) {
          smem_a[linear] = a_value;
          smem_b[linear] = row == col ? value_type{1} : value_type{0};
        }
      }

      if (tid == 0) {
        *info = 0;
      }
      __syncthreads();
    )";
    result += solver_func_name;
    result += R"((smem_a, ipiv, smem_b, info);
      __syncthreads();

      if (tid < elems) {
        return smem_b[tid];
      }
      return value_type{};
    )";
    return result;
  }
};

} // namespace detail
} // namespace matx

#endif
