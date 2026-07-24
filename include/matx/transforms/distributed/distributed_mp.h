////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#pragma once

#if !defined(MATX_EN_CUBLASMP) && !defined(MATX_EN_CUSOLVERMP)
#error "distributed_mp.h requires MATX_EN_CUBLASMP or MATX_EN_CUSOLVERMP"
#endif

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <vector>

#include "matx/core/distributed_tensor.h"
#include "matx/core/make_tensor.h"
#include "matx/core/operator_options.h"
#include "matx/core/type_utils.h"
#include "matx/executors/distributed.h"

namespace matx::experimental {
namespace detail {

template <typename Tensor>
inline constexpr bool is_mp_matrix_v =
    is_distributed_tensor_v<Tensor> &&remove_cvref_t<Tensor>::Rank() == 2 &&
    std::is_same_v<typename remove_cvref_t<Tensor>::distribution_type,
                   block_cyclic_distribution_t>;

template <typename Tensor>
void ValidateMpMatrix(const Tensor &tensor,
                      const distributedCUDAExecutor &executor,
                      const char *name) {
  static_assert(is_mp_matrix_v<Tensor>,
                "MP matrices must use block_cyclic_distribution_t");
  matx::detail::DistributedCheck(
      executor.HasCollectiveResources(), matxInvalidExecutor,
      "Block-cyclic execution requires collective resources on its "
      "distributedCUDAExecutor");
  matx::detail::DistributedCheck(
      tensor.ContextId() == executor.ContextId(), matxInvalidExecutor,
      std::string(name) + " belongs to a different distributed context");
  matx::detail::DistributedCheck(
      tensor.LocalFragmentCount() == 1, matxNotSupported,
      std::string(name) + " must have one local fragment per process");
  const auto &distribution = tensor.DistributionDescriptor();
  matx::detail::DistributedCheck(
      distribution.ProcessGrid()[0] == executor.ProcessRows() &&
          distribution.ProcessGrid()[1] == executor.ProcessColumns() &&
          distribution.GridLayout() == executor.GridLayout(),
      matxInvalidSize,
      std::string(name) +
          " process grid or rank ordering does not match its MP executor");
  for (size_t rank = 0; rank < distribution.FragmentCount(); ++rank) {
    matx::detail::DistributedCheck(
        distribution.FragmentEndpoint(rank).process_rank ==
            static_cast<int>(rank),
        matxInvalidParameter,
        std::string(name) +
            " endpoints must be ordered by NCCL communicator rank");
  }
}

template <typename A, typename B>
void ValidateSameMpGrid(const A &a, const B &b) {
  const auto &ad = a.DistributionDescriptor();
  const auto &bd = b.DistributionDescriptor();
  matx::detail::DistributedCheck(
      ad.ProcessGrid() == bd.ProcessGrid() &&
          ad.GridLayout() == bd.GridLayout(),
      matxInvalidSize, "MP matrix process grids and rank ordering must match");
  for (size_t rank = 0; rank < ad.FragmentCount(); ++rank) {
    matx::detail::DistributedCheck(
        ad.FragmentEndpoint(rank) == bd.FragmentEndpoint(rank),
        matxInvalidParameter, "MP matrix rank mappings must match");
  }
}

template <typename T> struct column_major_buffer {
  T *data = nullptr;
  index_t rows = 0;
  index_t columns = 0;
};

template <typename Tensor>
auto PackColumnMajor(const Tensor &tensor, cudaExecutor &executor) {
  using value_type = typename remove_cvref_t<Tensor>::value_type;
  const auto &local = tensor.LocalView(0);
  column_major_buffer<value_type> packed{nullptr, local.Size(0), local.Size(1)};
  const size_t bytes =
      static_cast<size_t>(packed.rows * packed.columns) * sizeof(value_type);
  MATX_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&packed.data),
                                  bytes, executor.getStream()));
  index_t shape[2]{packed.rows, packed.columns};
  index_t strides[2]{1, packed.rows};
  auto packed_view = make_tensor(packed.data, shape, strides, false);
  (packed_view = local).run(executor);
  return packed;
}

template <typename Tensor, typename T>
void UnpackColumnMajor(Tensor &tensor, const column_major_buffer<T> &packed,
                       cudaExecutor &executor) {
  index_t shape[2]{packed.rows, packed.columns};
  index_t strides[2]{1, packed.rows};
  auto packed_view = make_tensor(packed.data, shape, strides, false);
  (tensor.LocalView(0) = packed_view).run(executor);
}

template <typename T>
void FreeColumnMajor(column_major_buffer<T> &buffer, cudaExecutor &executor) {
  if (buffer.data != nullptr) {
    MATX_CUDA_CHECK(cudaFreeAsync(buffer.data, executor.getStream()));
    buffer.data = nullptr;
  }
}

#ifdef MATX_EN_CUBLASMP
class cublas_mp_descriptor {
public:
  cublas_mp_descriptor(const block_cyclic_distribution_t &distribution,
                       index_t leading_dimension, cudaDataType_t type,
                       cublasMpGrid_t grid) {
    const auto shape = distribution.GlobalShape();
    const auto block = distribution.BlockShape();
    matx::detail::CublasMpCheck(cublasMpMatrixDescriptorCreate(
                                    shape[0], shape[1], block[0], block[1], 0,
                                    0, std::max<index_t>(1, leading_dimension),
                                    type, grid, &descriptor_),
                                "cublasMpMatrixDescriptorCreate");
  }
  cublas_mp_descriptor(const cublas_mp_descriptor &) = delete;
  ~cublas_mp_descriptor() {
    if (descriptor_ != nullptr) {
      (void)cublasMpMatrixDescriptorDestroy(descriptor_);
    }
  }
  operator cublasMpMatrixDescriptor_t() const noexcept { return descriptor_; }

private:
  cublasMpMatrixDescriptor_t descriptor_ = nullptr;
};
#endif

#ifdef MATX_EN_CUSOLVERMP
class cusolver_mp_descriptor {
public:
  cusolver_mp_descriptor(const block_cyclic_distribution_t &distribution,
                         index_t leading_dimension, cudaDataType_t type,
                         cusolverMpGrid_t grid) {
    const auto shape = distribution.GlobalShape();
    const auto block = distribution.BlockShape();
    matx::detail::CusolverMpCheck(
        cusolverMpCreateMatrixDesc(&descriptor_, grid, type, shape[0], shape[1],
                                   block[0], block[1], 0, 0,
                                   std::max<index_t>(1, leading_dimension)),
        "cusolverMpCreateMatrixDesc");
  }
  cusolver_mp_descriptor(const cusolver_mp_descriptor &) = delete;
  ~cusolver_mp_descriptor() {
    if (descriptor_ != nullptr) {
      (void)cusolverMpDestroyMatrixDesc(descriptor_);
    }
  }
  operator cusolverMpMatrixDescriptor_t() const noexcept { return descriptor_; }

private:
  cusolverMpMatrixDescriptor_t descriptor_ = nullptr;
};
#endif

} // namespace detail

#ifdef MATX_EN_CUBLASMP
/**
 * Collective block-cyclic matrix multiplication using cuBLASMp.
 *
 * Every process in the executor grid must enter this function in the same
 * order. MatX local fragments remain ordinary row-major tensor views; this
 * adapter packs them into the column-major local layout required by cuBLASMp.
 */
namespace detail {

template <typename Out, typename A, typename B>
void MatmulMpImpl(Out &out, const A &a, const B &b,
                  distributedCUDAExecutor &executor,
                  typename remove_cvref_t<Out>::value_type alpha =
                      typename remove_cvref_t<Out>::value_type{1},
                  typename remove_cvref_t<Out>::value_type beta =
                      typename remove_cvref_t<Out>::value_type{0}) {
  using value_type = typename remove_cvref_t<Out>::value_type;
  static_assert(detail::is_mp_matrix_v<Out> && detail::is_mp_matrix_v<A> &&
                    detail::is_mp_matrix_v<B>,
                "Block-cyclic matmul requires rank-2 distributed tensors");
  static_assert(
      std::is_same_v<value_type, typename remove_cvref_t<A>::value_type> &&
          std::is_same_v<value_type, typename remove_cvref_t<B>::value_type>,
      "Block-cyclic matmul requires identical input and output types");
  static_assert(
      std::is_same_v<value_type, float> || std::is_same_v<value_type, double> ||
          std::is_same_v<value_type, cuda::std::complex<float>> ||
          std::is_same_v<value_type, cuda::std::complex<double>>,
      "Block-cyclic matmul supports float, double, complex<float>, and "
      "complex<double>");

  detail::ValidateMpMatrix(out, executor, "block-cyclic matmul output");
  detail::ValidateMpMatrix(a, executor, "block-cyclic matmul left input");
  detail::ValidateMpMatrix(b, executor, "block-cyclic matmul right input");
  detail::ValidateSameMpGrid(out, a);
  detail::ValidateSameMpGrid(out, b);
  matx::detail::DistributedCheck(
      a.Size(1) == b.Size(0) && out.Size(0) == a.Size(0) &&
          out.Size(1) == b.Size(1),
      matxInvalidSize,
      "Block-cyclic matmul matrix dimensions are incompatible");

  const auto endpoint = out.LocalFragment(0).endpoint;
  executor.ForEndpoint(endpoint, [&](cudaExecutor &local_executor) {
    auto packed_a = detail::PackColumnMajor(a, local_executor);
    auto packed_b = detail::PackColumnMajor(b, local_executor);
    auto packed_c = detail::PackColumnMajor(out, local_executor);

    detail::cublas_mp_descriptor desc_a{
        a.DistributionDescriptor(), packed_a.rows,
        MatXTypeToCudaType<value_type>(), executor.CublasGrid()};
    detail::cublas_mp_descriptor desc_b{
        b.DistributionDescriptor(), packed_b.rows,
        MatXTypeToCudaType<value_type>(), executor.CublasGrid()};
    detail::cublas_mp_descriptor desc_c{
        out.DistributionDescriptor(), packed_c.rows,
        MatXTypeToCudaType<value_type>(), executor.CublasGrid()};

    size_t device_workspace_size = 0;
    size_t host_workspace_size = 0;
    matx::detail::CublasMpCheck(
        cublasMpGemm_bufferSize(
            executor.CublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, out.Size(0),
            out.Size(1), a.Size(1), &alpha, packed_a.data, 1, 1, desc_a,
            packed_b.data, 1, 1, desc_b, &beta, packed_c.data, 1, 1, desc_c,
            MatXTypeToCudaComputeType<value_type>(), &device_workspace_size,
            &host_workspace_size),
        "cublasMpGemm_bufferSize");

    void *device_workspace = nullptr;
    if (device_workspace_size != 0) {
      MATX_CUDA_CHECK(cudaMallocAsync(&device_workspace, device_workspace_size,
                                      local_executor.getStream()));
    }
    std::vector<std::byte> host_workspace(host_workspace_size);
    matx::detail::CublasMpCheck(
        cublasMpGemm(
            executor.CublasHandle(), CUBLAS_OP_N, CUBLAS_OP_N, out.Size(0),
            out.Size(1), a.Size(1), &alpha, packed_a.data, 1, 1, desc_a,
            packed_b.data, 1, 1, desc_b, &beta, packed_c.data, 1, 1, desc_c,
            MatXTypeToCudaComputeType<value_type>(), device_workspace,
            device_workspace_size, host_workspace.data(), host_workspace_size),
        "cublasMpGemm");

    detail::UnpackColumnMajor(out, packed_c, local_executor);
    detail::FreeColumnMajor(packed_a, local_executor);
    detail::FreeColumnMajor(packed_b, local_executor);
    detail::FreeColumnMajor(packed_c, local_executor);
    if (device_workspace != nullptr) {
      MATX_CUDA_CHECK(
          cudaFreeAsync(device_workspace, local_executor.getStream()));
    }
    // cuBLASMp may retain the host workspace until its stream work completes.
    MATX_CUDA_CHECK(cudaStreamSynchronize(local_executor.getStream()));
  });
}

} // namespace detail
#endif

#ifdef MATX_EN_CUSOLVERMP
/**
 * Collective block-cyclic Cholesky factorization using cuSOLVERMp.
 */
namespace detail {

template <typename Out, typename A>
void CholMpImpl(Out &out, const A &a, distributedCUDAExecutor &executor,
                SolverFillMode uplo = SolverFillMode::UPPER) {
  using value_type = typename remove_cvref_t<Out>::value_type;
  static_assert(detail::is_mp_matrix_v<Out> && detail::is_mp_matrix_v<A>,
                "Block-cyclic Cholesky requires rank-2 distributed tensors");
  static_assert(
      std::is_same_v<value_type, typename remove_cvref_t<A>::value_type>,
      "Block-cyclic Cholesky input and output types must match");
  static_assert(
      std::is_same_v<value_type, float> || std::is_same_v<value_type, double> ||
          std::is_same_v<value_type, cuda::std::complex<float>> ||
          std::is_same_v<value_type, cuda::std::complex<double>>,
      "Block-cyclic Cholesky supports float, double, complex<float>, and "
      "complex<double>");

  detail::ValidateMpMatrix(out, executor, "block-cyclic Cholesky output");
  detail::ValidateMpMatrix(a, executor, "block-cyclic Cholesky input");
  detail::ValidateSameMpGrid(out, a);
  matx::detail::DistributedCheck(
      a.Size(0) == a.Size(1) && out.Size(0) == a.Size(0) &&
          out.Size(1) == a.Size(1),
      matxInvalidSize,
      "Block-cyclic Cholesky requires equal square input and output");
  matx::detail::DistributedCheck(
      out.DistributionDescriptor().Compatible(a.DistributionDescriptor()),
      matxInvalidParameter,
      "Block-cyclic Cholesky input and output layouts must match");
  matx::detail::DistributedCheck(a.DistributionDescriptor().BlockShape()[0] ==
                                     a.DistributionDescriptor().BlockShape()[1],
                                 matxInvalidSize,
                                 "cuSOLVERMp Cholesky requires square blocks");

  const auto endpoint = out.LocalFragment(0).endpoint;
  executor.ForEndpoint(endpoint, [&](cudaExecutor &local_executor) {
    auto packed = detail::PackColumnMajor(a, local_executor);
    detail::cusolver_mp_descriptor descriptor{
        a.DistributionDescriptor(), packed.rows,
        MatXTypeToCudaType<value_type>(), executor.CusolverGrid()};
    const cublasFillMode_t fill = uplo == SolverFillMode::UPPER
                                      ? CUBLAS_FILL_MODE_UPPER
                                      : CUBLAS_FILL_MODE_LOWER;

    size_t device_workspace_size = 0;
    size_t host_workspace_size = 0;
    matx::detail::CusolverMpCheck(
        cusolverMpPotrf_bufferSize(
            executor.CusolverHandle(), fill, a.Size(0), packed.data, 1, 1,
            descriptor, MatXTypeToCudaType<value_type>(),
            &device_workspace_size, &host_workspace_size),
        "cusolverMpPotrf_bufferSize");

    void *device_workspace = nullptr;
    int *info = nullptr;
    if (device_workspace_size != 0) {
      MATX_CUDA_CHECK(cudaMallocAsync(&device_workspace, device_workspace_size,
                                      local_executor.getStream()));
    }
    MATX_CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void **>(&info),
                                    sizeof(int), local_executor.getStream()));
    std::vector<std::byte> host_workspace(host_workspace_size);
    matx::detail::CusolverMpCheck(
        cusolverMpPotrf(executor.CusolverHandle(), fill, a.Size(0), packed.data,
                        1, 1, descriptor, MatXTypeToCudaType<value_type>(),
                        device_workspace, device_workspace_size,
                        host_workspace.data(), host_workspace_size, info),
        "cusolverMpPotrf");

    detail::UnpackColumnMajor(out, packed, local_executor);
    int host_info = 0;
    MATX_CUDA_CHECK(cudaMemcpyAsync(&host_info, info, sizeof(int),
                                    cudaMemcpyDeviceToHost,
                                    local_executor.getStream()));
    detail::FreeColumnMajor(packed, local_executor);
    if (device_workspace != nullptr) {
      MATX_CUDA_CHECK(
          cudaFreeAsync(device_workspace, local_executor.getStream()));
    }
    MATX_CUDA_CHECK(cudaFreeAsync(info, local_executor.getStream()));
    // cuSOLVERMp may retain the host workspace until its stream work completes.
    MATX_CUDA_CHECK(cudaStreamSynchronize(local_executor.getStream()));
    matx::detail::DistributedCheck(
        host_info == 0, matxSolverError,
        "cusolverMpPotrf reported a non-positive-definite matrix or invalid "
        "argument (info=" +
            std::to_string(host_info) + ")");
  });
}

} // namespace detail
#endif

} // namespace matx::experimental
