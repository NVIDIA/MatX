////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
////////////////////////////////////////////////////////////////////////////////

#include "matx.h"
#include "matx/distributed.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

using namespace matx;
using namespace matx::experimental;

namespace {

index_t ParsePositive(const char *text, const char *name) {
  size_t parsed = 0;
  const long long value = std::stoll(text, &parsed);
  if (text[parsed] != '\0' || value <= 0) {
    throw std::invalid_argument(std::string{name} +
                                " must be a positive integer");
  }
  return static_cast<index_t>(value);
}

int RunExample(int argc, char **argv) {
  if (argc > 3) {
    std::cerr << "Usage: distributed_fft_multi_gpu [gpu_count "
                 "[element_count]]\n";
    return 2;
  }

  const index_t requested_gpu_count =
      argc > 1 ? ParsePositive(argv[1], "gpu_count") : 2;
  const index_t element_count =
      argc > 2 ? ParsePositive(argv[2], "element_count") : 1 << 20;
  if (requested_gpu_count > std::numeric_limits<int>::max()) {
    throw std::invalid_argument("gpu_count exceeds the supported range");
  }
  const int gpu_count = static_cast<int>(requested_gpu_count);
  if (gpu_count < 2) {
    throw std::invalid_argument(
        "The cuFFT multi-GPU backend requires at least two GPUs");
  }
  if (element_count < gpu_count) {
    throw std::invalid_argument(
        "element_count must be at least the requested GPU count");
  }

  int visible_gpu_count = 0;
  MATX_CUDA_CHECK(cudaGetDeviceCount(&visible_gpu_count));
  if (gpu_count > visible_gpu_count) {
    throw std::invalid_argument("Requested " + std::to_string(gpu_count) +
                                " GPUs, but only " +
                                std::to_string(visible_gpu_count) +
                                " are visible");
  }

  std::vector<int> devices;
  std::vector<distributed_endpoint_t> endpoints;
  devices.reserve(static_cast<size_t>(gpu_count));
  endpoints.reserve(static_cast<size_t>(gpu_count));
  for (int device = 0; device < gpu_count; ++device) {
    devices.push_back(device);
    endpoints.push_back({0, device});
  }

  using complex_type = cuda::std::complex<float>;
  distributed_context context{devices};
  distributedCUDAExecutor executor{context};
  auto distribution =
      block_distribution_t<1>::Slab({element_count}, endpoints, 0);
  auto input = make_distributed_tensor<complex_type>(distribution, context);
  auto output = make_distributed_tensor<complex_type>(distribution, context);

  // An impulse at global index zero has a constant Fourier transform.
  for (size_t local = 0; local < input.LocalFragmentCount(); ++local) {
    const auto &fragment = input.LocalFragment(local);
    const index_t local_count = input.LocalView(local).Size(0);
    std::vector<complex_type> host(static_cast<size_t>(local_count),
                                   complex_type{0.0F, 0.0F});
    const auto first_global =
        distribution.LocalToGlobal(fragment.distribution_index, {0});
    if (first_global[0] == 0) {
      host[0] = complex_type{1.0F, 0.0F};
    }
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(input.LocalView(local).Data(), host.data(),
                               host.size() * sizeof(complex_type),
                               cudaMemcpyHostToDevice));
  }

  // This is one process coordinating several GPUs on one node through cuFFT
  // Xt/Mg. No MPI communicator is involved.
  (output = fft(input)).run(executor);
  executor.sync();

  float maximum_error = 0.0F;
  for (size_t local = 0; local < output.LocalFragmentCount(); ++local) {
    const auto &fragment = output.LocalFragment(local);
    const index_t local_count = output.LocalView(local).Size(0);
    std::vector<complex_type> host(static_cast<size_t>(local_count));
    matx::detail::distributed_device_guard guard{fragment.endpoint.device_id};
    MATX_CUDA_CHECK(cudaMemcpy(host.data(), output.LocalView(local).Data(),
                               host.size() * sizeof(complex_type),
                               cudaMemcpyDeviceToHost));
    for (const auto &value : host) {
      maximum_error =
          std::max(maximum_error, std::abs(value.real() - 1.0F));
      maximum_error = std::max(maximum_error, std::abs(value.imag()));
    }
  }

  std::cout << "Single-node cuFFT transform of " << element_count
            << " elements across " << gpu_count
            << " GPUs; maximum error " << maximum_error << '\n';
  return maximum_error <= 1.0e-5F ? 0 : 1;
}

} // namespace

int main(int argc, char **argv) {
  try {
    return RunExample(argc, argv);
  } catch (const std::exception &error) {
    std::cerr << "distributed_fft_multi_gpu failed: " << error.what() << '\n';
    return 1;
  }
}
