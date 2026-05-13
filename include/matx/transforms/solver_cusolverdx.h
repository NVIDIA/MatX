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
      case CUSOLVERDX_FUNCTION_UNGQR:
      case CUSOLVERDX_FUNCTION_UNGLQ:
      case CUSOLVERDX_FUNCTION_UNMQR:
      case CUSOLVERDX_FUNCTION_UNMLQ:
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
    const auto minmn = static_cast<long long int>(cuda::std::min(m_, n_));
    switch (function_) {
      case CUSOLVERDX_FUNCTION_POTRF: {
        const auto elems = static_cast<long long int>(n_) * static_cast<long long int>(n_);
        return elems * elem_size + static_cast<long long int>(sizeof(int));
      }
      case CUSOLVERDX_FUNCTION_GETRF_PARTIAL_PIVOT: {
        const auto elems = static_cast<long long int>(m_) * static_cast<long long int>(n_);
        return elems * elem_size + minmn * static_cast<long long int>(sizeof(int)) + static_cast<long long int>(sizeof(int));
      }
      case CUSOLVERDX_FUNCTION_GEQRF:
      case CUSOLVERDX_FUNCTION_UNGQR: {
        const auto elems = static_cast<long long int>(m_) * static_cast<long long int>(n_);
        return elems * elem_size + static_cast<long long int>(k_ == 0 ? cuda::std::min(m_, n_) : k_) * elem_size + static_cast<long long int>(sizeof(int));
      }
      case CUSOLVERDX_FUNCTION_HEEV:
      case CUSOLVERDX_FUNCTION_HTEV: {
        using precision_type = typename inner_op_type_t<InputType>::type;
        const auto elems = static_cast<long long int>(n_) * static_cast<long long int>(n_);
        const auto matrix_bytes = elems * elem_size;
        const auto lambda_offset = AlignUp(matrix_bytes, static_cast<long long int>(alignof(precision_type)));
        const auto workspace_offset = AlignUp(lambda_offset + static_cast<long long int>(n_) * static_cast<long long int>(sizeof(precision_type)),
                                              static_cast<long long int>(alignof(InputType)));
        const auto info_offset = AlignUp(workspace_offset + static_cast<long long int>(GetWorkspaceSize()),
                                         static_cast<long long int>(alignof(int)));
        return info_offset + static_cast<long long int>(sizeof(int));
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

  static long long int AlignUp(long long int value, long long int alignment)
  {
    return ((value + alignment - 1) / alignment) * alignment;
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

    if (!detail::GetCache().StoreLTOIRCachedBytes(symbol_name, static_cast<const char*>(ltoir.get()), ltoir_length)) {
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
      static_assert(Rank() >= 2 && Rank() <= 4, "cuSolverDx JIT supports matrix operator ranks 2 through 4");
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

      const index_t out_row = idx[Rank() - 2];
      const index_t out_col = idx[Rank() - 1];
      if (out_row < n && out_col < n) {
        return smem_a[out_row * n + out_col];
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
      static_assert(Rank() >= 2 && Rank() <= 4, "cuSolverDx JIT supports matrix operator ranks 2 through 4");
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

      const index_t out_row = idx[Rank() - 2];
      const index_t out_col = idx[Rank() - 1];
      if (out_row < n && out_col < n) {
        return smem_b[out_row * n + out_col];
      }
      return value_type{};
    )";
    return result;
  }

  std::string GetLuProjectionFuncStr(const std::string &solver_func_name, int factors_component, int piv_component) const
  {
    std::string result = R"(
      using solver_value_type = typename OpA::value_type;
      static_assert(OpA::Rank() >= 2 && OpA::Rank() <= 4, "cuSolverDx JIT projections support input ranks 2 through 4");
      static constexpr index_t m = )";
    result += std::to_string(static_cast<int>(m_));
    result += R"(;
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t elems = m * n;
      static constexpr index_t piv_elems = (m < n ? m : n);
      extern __shared__ __align__(16) char smem[];
      solver_value_type* smem_a = reinterpret_cast<solver_value_type*>(smem);
      int* ipiv = reinterpret_cast<int*>(smem + elems * sizeof(solver_value_type));
      int* info = reinterpret_cast<int*>(smem + elems * sizeof(solver_value_type) + piv_elems * sizeof(int));
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        solver_value_type a_value{};
        if constexpr (OpA::Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (OpA::Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (OpA::Rank() == 4) {
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
    result += R"((smem_a, ipiv, info);
      __syncthreads();

      if constexpr (Component == )";
    result += std::to_string(factors_component);
    result += R"() {
        const index_t row = idx[Rank() - 2];
        const index_t col = idx[Rank() - 1];
        if (row < m && col < n) {
          return smem_a[row * n + col];
        }
      }
      else if constexpr (Component == )";
    result += std::to_string(piv_component);
    result += R"() {
        const index_t vec_idx = idx[Rank() - 1];
        if (vec_idx < piv_elems) {
          const int pivot = ipiv[vec_idx] == 0 ?
            static_cast<int>(vec_idx + 1) :
            ipiv[vec_idx];
          return static_cast<value_type>(pivot);
        }
      }
      return value_type{};
    )";
    return result;
  }

  std::string GetGeqrfProjectionFuncStr(const std::string &solver_func_name, int out_component, int tau_component) const
  {
    std::string result = R"(
      using solver_value_type = typename OpA::value_type;
      static_assert(OpA::Rank() >= 2 && OpA::Rank() <= 4, "cuSolverDx JIT projections support input ranks 2 through 4");
      static constexpr index_t m = )";
    result += std::to_string(static_cast<int>(m_));
    result += R"(;
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t elems = m * n;
      static constexpr index_t tau_elems = (m < n ? m : n);
      extern __shared__ __align__(16) char smem[];
      solver_value_type* smem_a = reinterpret_cast<solver_value_type*>(smem);
      solver_value_type* tau = reinterpret_cast<solver_value_type*>(smem + elems * sizeof(solver_value_type));
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        solver_value_type a_value{};
        if constexpr (OpA::Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (OpA::Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (OpA::Rank() == 4) {
          a_value = a_.template operator()<CapType>(idx[0], idx[1], row, col);
        }
        if (valid) {
          smem_a[linear] = a_value;
        }
      }

      __syncthreads();
    )";
    result += solver_func_name;
    result += R"((smem_a, tau);
      __syncthreads();

      if constexpr (Component == )";
    result += std::to_string(out_component);
    result += R"() {
        const index_t row = idx[Rank() - 2];
        const index_t col = idx[Rank() - 1];
        if (row < m && col < n) {
          return smem_a[row * n + col];
        }
      }
      else if constexpr (Component == )";
    result += std::to_string(tau_component);
    result += R"() {
        const index_t vec_idx = idx[Rank() - 1];
        if (vec_idx < tau_elems) {
          return tau[vec_idx];
        }
      }
      return value_type{};
    )";
    return result;
  }

  std::string GetQrProjectionFuncStr(const std::string &geqrf_func_name,
                                     const std::string &ungqr_func_name,
                                     int q_component,
                                     int r_component,
                                     index_t q_cols,
                                     index_t r_rows) const
  {
    std::string result = R"(
      using solver_value_type = typename OpA::value_type;
      static_assert(OpA::Rank() >= 2 && OpA::Rank() <= 4, "cuSolverDx JIT projections support input ranks 2 through 4");
      static constexpr index_t m = )";
    result += std::to_string(static_cast<int>(m_));
    result += R"(;
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t q_cols = )";
    result += std::to_string(static_cast<int>(q_cols));
    result += R"(;
      static constexpr index_t r_rows = )";
    result += std::to_string(static_cast<int>(r_rows));
    result += R"(;
      static constexpr index_t elems = m * n;
      static constexpr index_t q_elems = m * q_cols;
      static constexpr index_t tau_elems = q_cols;
      extern __shared__ __align__(16) char smem[];
      solver_value_type* smem_a = reinterpret_cast<solver_value_type*>(smem);
      solver_value_type* tau = reinterpret_cast<solver_value_type*>(smem + elems * sizeof(solver_value_type));
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        solver_value_type a_value{};
        if constexpr (OpA::Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (OpA::Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (OpA::Rank() == 4) {
          a_value = a_.template operator()<CapType>(idx[0], idx[1], row, col);
        }
        if (valid) {
          smem_a[linear] = a_value;
        }
      }

      __syncthreads();
    )";
    result += geqrf_func_name;
    result += R"((smem_a, tau);
      __syncthreads();

      if constexpr (Component == )";
    result += std::to_string(q_component);
    result += R"() {
        for (index_t base = 0; base < q_elems; base += total_threads) {
          const index_t linear = base + tid;
          if (linear < q_elems) {
            const index_t row = linear / q_cols;
            const index_t col = linear % q_cols;
            smem_a[linear] = smem_a[row * n + col];
          }
        }
        __syncthreads();
    )";
    result += ungqr_func_name;
    result += R"((smem_a, tau);
        __syncthreads();

        const index_t row = idx[Rank() - 2];
        const index_t col = idx[Rank() - 1];
        if (row < m && col < q_cols) {
          return smem_a[row * q_cols + col];
        }
      }
      else if constexpr (Component == )";
    result += std::to_string(r_component);
    result += R"() {
        const index_t row = idx[Rank() - 2];
        const index_t col = idx[Rank() - 1];
        if (row < r_rows && col < n && row <= col) {
          return smem_a[row * n + col];
        }
      }
      return value_type{};
    )";
    return result;
  }

  std::string GetHeevProjectionFuncStr(const std::string &solver_func_name, int vectors_component, int values_component) const
  {
    std::string result = R"(
      using solver_value_type = typename OpA::value_type;
      using precision_type = typename inner_op_type_t<solver_value_type>::type;
      static_assert(OpA::Rank() >= 2 && OpA::Rank() <= 4, "cuSolverDx JIT projections support input ranks 2 through 4");
      static constexpr index_t n = )";
    result += std::to_string(static_cast<int>(n_));
    result += R"(;
      static constexpr index_t elems = n * n;
      static constexpr size_t matrix_bytes = elems * sizeof(solver_value_type);
      static constexpr size_t lambda_offset = ((matrix_bytes + alignof(precision_type) - 1) / alignof(precision_type)) * alignof(precision_type);
      static constexpr size_t lambda_bytes = n * sizeof(precision_type);
      static constexpr size_t workspace_offset = ((lambda_offset + lambda_bytes + alignof(solver_value_type) - 1) / alignof(solver_value_type)) * alignof(solver_value_type);
      static constexpr size_t workspace_bytes = )";
    result += std::to_string(static_cast<int>(GetWorkspaceSize()));
    result += R"(;
      static constexpr size_t info_offset = ((workspace_offset + workspace_bytes + alignof(int) - 1) / alignof(int)) * alignof(int);
      extern __shared__ __align__(16) char smem[];
      solver_value_type* smem_a = reinterpret_cast<solver_value_type*>(smem);
      precision_type* lambda = reinterpret_cast<precision_type*>(smem + lambda_offset);
      solver_value_type* workspace = reinterpret_cast<solver_value_type*>(smem + workspace_offset);
      int* info = reinterpret_cast<int*>(smem + info_offset);
      const int tid = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y;
      const int total_threads = blockDim.x * blockDim.y * blockDim.z;
      cuda::std::array<index_t, Rank()> idx = { static_cast<index_t>(indices)... };

      for (index_t base = 0; base < elems; base += total_threads) {
        const index_t linear = base + tid;
        const bool valid = linear < elems;
        const index_t load_linear = valid ? linear : index_t{0};
        const index_t row = load_linear / n;
        const index_t col = load_linear % n;
        solver_value_type a_value{};
        if constexpr (OpA::Rank() == 2) {
          a_value = a_.template operator()<CapType>(row, col);
        }
        else if constexpr (OpA::Rank() == 3) {
          a_value = a_.template operator()<CapType>(idx[0], row, col);
        }
        else if constexpr (OpA::Rank() == 4) {
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
    result += R"((smem_a, lambda, workspace, info);
      __syncthreads();

      if constexpr (Component == )";
    result += std::to_string(vectors_component);
    result += R"() {
        const index_t row = idx[Rank() - 2];
        const index_t col = idx[Rank() - 1];
        if (row < n && col < n) {
          return smem_a[row * n + col];
        }
      }
      else if constexpr (Component == )";
    result += std::to_string(values_component);
    result += R"() {
        const index_t vec_idx = idx[Rank() - 1];
        if (vec_idx < n) {
          return static_cast<value_type>(lambda[vec_idx]);
        }
      }
      return value_type{};
    )";
    return result;
  }
};

} // namespace detail
} // namespace matx

#endif
