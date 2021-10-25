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

namespace matx {

inline void get_grid_dims(dim3 &blocks, dim3 &threads, index_t size0,
                          int max_cta_size = 1024)
{
  int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  while (nt < max_cta_size) {
    if (static_cast<index_t>(threads.x) < size0) {
      threads.x *= 2;
    }
    nt *= 2;
  }
  // launch as many blocks as necessary
  blocks.x = static_cast<int>((size0 + threads.x - 1) / threads.x);
  blocks.y = 1;
  blocks.z = 1;
}

inline void get_grid_dims(dim3 &blocks, dim3 &threads, index_t size0,
                          index_t size1, int max_cta_size = 1024)
{
  int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  while (nt < max_cta_size) {
    if (static_cast<index_t>(threads.x) < size1) {
      threads.x *= 2;
    }
    else if (static_cast<index_t>(threads.y) < size0) {
      threads.y *= 2;
    }
    nt *= 2;
  }
  // launch as many blocks as necessary
  blocks.x = static_cast<int>((size1 + threads.x - 1) / threads.x);
  blocks.y = static_cast<int>((size0 + threads.y - 1) / threads.y);
  blocks.z = 1;
}

inline void get_grid_dims(dim3 &blocks, dim3 &threads, index_t size0,
                          index_t size1, index_t size2, int max_cta_size = 1024)
{
  int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  while (nt < max_cta_size) {
    if (static_cast<index_t>(threads.x) < size2) {
      threads.x *= 2;
    }
    else if (static_cast<index_t>(threads.y) < size1) {
      threads.y *= 2;
    }
    else if (static_cast<index_t>(threads.z) < size0) {
      threads.z *= 2;
    }
    nt *= 2;
  }
  // launch as many blocks as necessary
  blocks.x = static_cast<int>((size2 + threads.x - 1) / threads.x);
  blocks.y = static_cast<int>((size1 + threads.y - 1) / threads.y);
  blocks.z = static_cast<int>((size0 + threads.z - 1) / threads.z);
}

inline void get_grid_dims(dim3 &blocks, dim3 &threads, index_t size0,
                          index_t size1, index_t size2, index_t size3,
                          int max_cta_size = 1024)
{
  int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  while (nt < max_cta_size) {
    if (static_cast<index_t>(threads.x) < size3) {
      threads.x *= 2;
    }
    else if (static_cast<index_t>(threads.y) < size1 * size2) {
      threads.y *= 2;
    }
    else if (static_cast<index_t>(threads.z) < size0) {
      threads.z *= 2;
    }
    nt *= 2;
  }
  // launch as many blocks as necessary
  blocks.x = static_cast<int>((size3 + threads.x - 1) / threads.x);
  blocks.y = static_cast<int>((size1 * size2 + threads.y - 1) / threads.y);
  blocks.z = static_cast<int>((size0 + threads.z - 1) / threads.z);
}

} // end namespace matx
