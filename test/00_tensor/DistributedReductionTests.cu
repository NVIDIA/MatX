////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
// Copyright (c) 2026, NVIDIA Corporation. All rights reserved.
////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include "gtest/gtest.h"

#include <vector>

using namespace matx;
using namespace matx::experimental;

#ifdef MATX_EN_NCCL
namespace {

class LocalNcclCommunicators {
public:
  explicit LocalNcclCommunicators(const std::vector<int> &devices)
      : communicators_(devices.size(), nullptr) {
    matx::detail::NcclCheck(ncclCommInitAll(communicators_.data(),
                                            static_cast<int>(devices.size()),
                                            devices.data()),
                            "ncclCommInitAll");
  }
  ~LocalNcclCommunicators() {
    for (auto communicator : communicators_)
      if (communicator != nullptr)
        (void)ncclCommDestroy(communicator);
  }
  ncclComm_t operator[](size_t index) const { return communicators_[index]; }

private:
  std::vector<ncclComm_t> communicators_;
};

template <typename Tensor> void FillByGlobalIndex(Tensor &tensor) {
  const auto &distribution = tensor.DistributionDescriptor();
  for (size_t local = 0; local < tensor.LocalFragmentCount(); ++local) {
    const auto &fragment = tensor.LocalFragment(local);
    const auto shape = distribution.LocalShape(fragment.distribution_index);
    std::vector<float> host(
        static_cast<size_t>(tensor.LocalView(local).TotalSize()));
    for (index_t linear = 0; linear < tensor.LocalView(local).TotalSize();
         ++linear) {
      const auto local_index =
          experimental::detail::DistributedUnflatten<Tensor::Rank()>(linear,
                                                                     shape);
      const auto global =
          distribution.LocalToGlobal(fragment.distribution_index, local_index);
      host[static_cast<size_t>(linear)] =
          static_cast<float>(global[0] * tensor.Size(1) + global[1] + 1);
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(tensor.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(float),
                               cudaMemcpyHostToDevice));
  }
}

template <typename Tensor>
std::vector<float> CopyReplica(const Tensor &tensor, size_t local) {
  const auto &fragment = tensor.LocalFragment(local);
  matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
  std::vector<float> host(
      static_cast<size_t>(tensor.LocalView(local).TotalSize()));
  MATX_CUDA_CHECK(cudaMemcpy(host.data(), tensor.LocalView(local).Data(),
                             host.size() * sizeof(float),
                             cudaMemcpyDeviceToHost));
  return host;
}

} // namespace

TEST(DistributedReduction, SingleNodeMultiGpuStandardReductions) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count < 2)
    GTEST_SKIP() << "Requires at least two CUDA devices";

  std::vector<int> devices{0, 1};
  std::vector<distributed_endpoint_t> endpoints{{0, 0}, {0, 1}};
  LocalNcclCommunicators communicators{devices};
  distributed_nccl_topology_t topology{
      endpoints, {{{0, 0}, communicators[0]}, {{0, 1}, communicators[1]}}};
  distributed_context context{devices};
  distributedCUDAExecutor executor{context, std::move(topology)};

  auto input_distribution = block_distribution_t<2>::Slab({5, 4}, endpoints, 0);
  auto input = make_distributed_tensor<float>(input_distribution, context);
  FillByGlobalIndex(input);

  replicated_distribution_t<1> vector_distribution{{4}, endpoints};
  auto column_sum =
      make_distributed_tensor<float>(vector_distribution, context);
  int rows[]{0};
  (column_sum = sum(input, rows)).run(executor);

  replicated_distribution_t<0> scalar_distribution{distributed_index_t<0>{},
                                                   endpoints};
  auto total = make_distributed_tensor<float>(scalar_distribution, context);
  auto product = make_distributed_tensor<float>(scalar_distribution, context);
  auto minimum = make_distributed_tensor<float>(scalar_distribution, context);
  auto maximum = make_distributed_tensor<float>(scalar_distribution, context);
  auto average = make_distributed_tensor<float>(scalar_distribution, context);
  auto every = make_distributed_tensor<float>(scalar_distribution, context);
  auto some = make_distributed_tensor<float>(scalar_distribution, context);
  (total = sum(input)).run(executor);
  (product = prod(input)).run(executor);
  (minimum = min(input)).run(executor);
  (maximum = max(input)).run(executor);
  (average = mean(input)).run(executor);
  (every = all(input)).run(executor);
  (some = any(input)).run(executor);
  executor.sync();

  for (size_t local = 0; local < 2; ++local) {
    const auto columns = CopyReplica(column_sum, local);
    for (int column = 0; column < 4; ++column)
      EXPECT_FLOAT_EQ(columns[static_cast<size_t>(column)],
                      static_cast<float>(45 + 5 * column));
    EXPECT_FLOAT_EQ(CopyReplica(total, local)[0], 210.0F);
    EXPECT_FLOAT_EQ(CopyReplica(minimum, local)[0], 1.0F);
    EXPECT_FLOAT_EQ(CopyReplica(maximum, local)[0], 20.0F);
    EXPECT_FLOAT_EQ(CopyReplica(average, local)[0], 10.5F);
    EXPECT_FLOAT_EQ(CopyReplica(every, local)[0], 1.0F);
    EXPECT_FLOAT_EQ(CopyReplica(some, local)[0], 1.0F);
    EXPECT_NEAR(CopyReplica(product, local)[0], 2.432902e18F, 3.0e12F);
  }

  replicated_distribution_t<2> replicated_input_distribution{{2, 2}, endpoints};
  auto replicated_input =
      make_distributed_tensor<float>(replicated_input_distribution, context);
  FillByGlobalIndex(replicated_input);
  auto replicated_total =
      make_distributed_tensor<float>(scalar_distribution, context);
  (replicated_total = sum(replicated_input)).run(executor);
  executor.sync();
  for (size_t local = 0; local < 2; ++local)
    EXPECT_FLOAT_EQ(CopyReplica(replicated_total, local)[0], 10.0F);

  block_cyclic_distribution_t block_cyclic_distribution{
      {4, 4}, {2, 2}, {1, 2}, endpoints};
  auto block_cyclic =
      make_distributed_tensor<float>(block_cyclic_distribution, context);
  FillByGlobalIndex(block_cyclic);
  auto block_cyclic_total =
      make_distributed_tensor<float>(scalar_distribution, context);
  (block_cyclic_total = sum(block_cyclic)).run(executor);
  executor.sync();
  for (size_t local = 0; local < 2; ++local)
    EXPECT_FLOAT_EQ(CopyReplica(block_cyclic_total, local)[0], 136.0F);

  using descriptor = block_fragment_descriptor_t<2>;
  block_distribution_t<2> multi_fragment_distribution{
      {4, 4},
      std::vector<descriptor>{{{0, 0}, {0, 0}, {2, 2}},
                              {{0, 1}, {0, 2}, {2, 2}},
                              {{0, 0}, {2, 0}, {2, 2}},
                              {{0, 1}, {2, 2}, {2, 2}}}};
  auto multi_fragment =
      make_distributed_tensor<float>(multi_fragment_distribution, context);
  FillByGlobalIndex(multi_fragment);
  auto multi_fragment_total =
      make_distributed_tensor<float>(scalar_distribution, context);
  (multi_fragment_total = sum(multi_fragment)).run(executor);
  executor.sync();
  for (size_t local = 0; local < 2; ++local)
    EXPECT_FLOAT_EQ(CopyReplica(multi_fragment_total, local)[0], 136.0F);

  replicated_distribution_t<0> reversed_output_distribution{
      distributed_index_t<0>{}, {{0, 1}, {0, 0}}};
  auto reversed_output =
      make_distributed_tensor<float>(reversed_output_distribution, context);
  EXPECT_THROW((reversed_output = sum(input)).run(executor),
               matx::detail::matxException);
}

TEST(DistributedReduction, RejectsExecutorWithoutNcclTopology) {
  int device_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&device_count));
  if (device_count == 0)
    GTEST_SKIP() << "Requires a CUDA device";
  distributed_context context{{0}};
  distributedCUDAExecutor executor{context};
  auto input_distribution = block_distribution_t<1>::Slab({4}, {{0, 0}});
  auto input = make_distributed_tensor<float>(input_distribution, context);
  replicated_distribution_t<0> output_distribution{distributed_index_t<0>{},
                                                   {{0, 0}}};
  auto output = make_distributed_tensor<float>(output_distribution, context);
  EXPECT_THROW((output = sum(input)).run(executor),
               matx::detail::matxException);
}
#endif
