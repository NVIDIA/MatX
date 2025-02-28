#include "matx.h"

int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();

  cudaStream_t stream = 0;
  matx::cudaExecutor exec{stream};

  // // manually set to log all NVTX levels
  // MATX_NVTX_SET_LOG_LEVEL( matx::matx_nvxtLogLevels::MATX_NVTX_LOG_ALL );

  matx::index_t size_x = 128;
  matx::index_t size_y = 256;

  auto A      = matx::make_tensor<float>({size_x, size_y});
  auto B      = matx::make_tensor<float>({size_x, size_y});
  auto C      = matx::make_tensor<float>({size_x, size_y});
  auto D      = matx::make_tensor<float>({size_x, size_y});
  auto result = matx::make_tensor<float>({size_x, size_y});

  // run once to warm-up
  (result = cos(C)).run(exec);
  (result = result / D).run(exec);
  (result = result * B).run(exec);
  (A = B * cos(C)/D).run(exec);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < 10; i++) {

    // first individual, independent kernels
    int unfused_range = MATX_NVTX_START_RANGE("Unfused Kernels");
    (result = cos(C)).run(exec);
    (result = result / D).run(exec);
    (result = result * B).run(exec);
    MATX_NVTX_END_RANGE(unfused_range);

    // now, as a fused operation
    int fused_range = MATX_NVTX_START_RANGE("Fused Operation");
    (A = B * cos(C)/D).run(exec);
    MATX_NVTX_END_RANGE(fused_range);
  }

  MATX_EXIT_HANDLER();
}