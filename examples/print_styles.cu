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

#include "matx.h"
#include <cassert>
#include <cstdio>
#include <cuda/std/ccomplex>

using namespace matx;

/**
 * Print Styles Example
 *
 * This example shows how to change the print() formatting for different styles.
 *
 */
template < typename T>
void concatOp(T & output, const std::vector<T> & matx_tensors, cudaStream_t stream = 0) {
  assert(matx_tensors.size() > 0);
  switch (matx_tensors.size()) {
    case 3:
      (output = matx::concat(0, matx_tensors[0], matx_tensors[1], matx_tensors[2])).run(stream);
      break;
    case 2:
      (output = matx::concat(0, matx_tensors[0], matx_tensors[1])).run(stream);
      break;
    case 1:
      (output = matx_tensors[0]).run(stream);
      break;
  }
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char **argv)
{
  MATX_ENTER_HANDLER();
  // example-begin print-example-1
  using complex = cuda::std::complex<double>;

  auto A1 = make_tensor<complex>({16});
  A1.SetVals({
    {  1,  -1}, {  2,   2}, {  3,  -3}, {  4,  4},
    { -5,   5}, { -6,  -6}, { -7,   7}, { -8, -8},
    {  9,  -9}, { 10,  10}, { 11, -11}, { 12, 12},
    {-13,  13}, {-14,  14}, {-15,  15}, {-16, 16}
  });

  auto A2 = reshape(A1, {4,4});
  auto A3 = reshape(A1, {2,2,4});
  auto A4 = reshape(A1, {2,2,2,2});

  printf("MATX_PRINT_FORMAT_DEFAULT:\n");
  print(A1);
  print(A2);
  print(A3);
  print(A4);

  set_print_format_type(MATX_PRINT_FORMAT_MLAB);
  printf("MATX_PRINT_FORMAT_MLAB:\n");
  print(A1);
  print(A2);
  print(A3);
  print(A4);

  set_print_format_type(MATX_PRINT_FORMAT_PYTHON);
  printf("MATX_PRINT_FORMAT_PYTHON:\n");
  print(A1);
  print(A2);
  print(A3);
  print(A4);
  // example-end print-example-1

  // create stream
  cudaStream_t stream;
  cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking);

  std::vector<float3> geo_mesh_vertices(10);
  std::vector<float3> edge_mesh_vertices(10);

  auto matx_mesh_vertices = matx::make_tensor<float3>({static_cast<matx::index_t>(geo_mesh_vertices.size()  +
                        edge_mesh_vertices.size())}, matx::MATX_ASYNC_DEVICE_MEMORY);

  // move CPU vectors to device streams asynchronously
  std::vector<matx::tensor_t<float3, 1>> matx_meshes;
  if (geo_mesh_vertices.size()) {
    auto matx_geo_mesh_vertices = matx::make_tensor<float3>({static_cast<matx::index_t>(geo_mesh_vertices.size())}, matx::MATX_ASYNC_DEVICE_MEMORY);
    cudaMemcpyAsync(matx_geo_mesh_vertices.Data(), geo_mesh_vertices.data(), sizeof(float3) * geo_mesh_vertices.size(), cudaMemcpyHostToDevice, stream);
    matx_meshes.push_back(matx_geo_mesh_vertices);
  }
  if (edge_mesh_vertices.size()) {
    auto matx_edge_mesh_vertices = matx::make_tensor<float3>({static_cast<matx::index_t>(edge_mesh_vertices.size())}, matx::MATX_ASYNC_DEVICE_MEMORY);
    cudaMemcpyAsync(matx_edge_mesh_vertices.Data(), edge_mesh_vertices.data(), sizeof(float3) * edge_mesh_vertices.size(), cudaMemcpyHostToDevice, stream);
    matx_meshes.push_back(matx_edge_mesh_vertices);
  }

  concatOp(matx_mesh_vertices, matx_meshes, stream);
  cudaStreamSynchronize(stream);

  CUDA_CHECK_LAST_ERROR();
  MATX_EXIT_HANDLER();
  return 0;
}
