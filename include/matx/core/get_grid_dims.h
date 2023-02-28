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
#include <numeric>

namespace matx {
namespace detail {

template <int RANK>
inline bool get_grid_dims(dim3 &blocks, dim3 &threads, const std::array<index_t, RANK> &sizes,
                          int max_cta_size = 1024)
{
  bool stride = false;
  [[maybe_unused]] int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;
  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  if constexpr (RANK == 1) {
    while (nt < max_cta_size) {
      if (static_cast<index_t>(threads.x) < sizes[0]) {
        threads.x *= 2;
      }
      nt *= 2;
    }
    // launch as many blocks as necessary
    blocks.x = static_cast<int>((sizes[0] + threads.x - 1) / threads.x);
    blocks.y = 1;
    blocks.z = 1;  
  }
  else if constexpr (RANK == 2) {
    while (nt < max_cta_size) {
      if (static_cast<index_t>(threads.x) < sizes[1]) {
        threads.x *= 2;
      }
      else if (static_cast<index_t>(threads.y) < sizes[0]) {
        threads.y *= 2;
      }
      nt *= 2;
    }
    // launch as many blocks as necessary
    blocks.x = static_cast<int>((sizes[1] + threads.x - 1) / threads.x);
    blocks.y = static_cast<int>((sizes[0] + threads.y - 1) / threads.y);
    blocks.z = 1; 

    if(blocks.y > 65535) {
      blocks.y = 65535;
      stride = true;
    }

  }  
  else if constexpr (RANK == 3) {
    while (nt < max_cta_size) {
      if (static_cast<index_t>(threads.x) < sizes[2]) {
        threads.x *= 2;
      }
      else if (static_cast<index_t>(threads.y) < sizes[1]) {
        threads.y *= 2;
      }
      else if (static_cast<index_t>(threads.z) < sizes[0]) {
        threads.z *= 2;
      }
      nt *= 2;
    }
 
    // cuda restricts maximum block size in z to 64
    if( threads.z > 64 ) {
      threads.z = 64;
    }

    // launch as many blocks as necessary
    blocks.x = static_cast<int>((sizes[2] + threads.x - 1) / threads.x);
    blocks.y = static_cast<int>((sizes[1] + threads.y - 1) / threads.y);
    blocks.z = static_cast<int>((sizes[0] + threads.z - 1) / threads.z);
    
    if(blocks.y > 65535) {
      blocks.y = 65535;
      stride = true;
    }
    if(blocks.z > 65535) {
      blocks.z = 65535;
      stride = true;
    }

  }  
  else if constexpr (RANK == 4) {
    while (nt < max_cta_size) {
      if (static_cast<index_t>(threads.x) < sizes[3]) {
        threads.x *= 2;
      }
      else if (static_cast<index_t>(threads.y) < sizes[1] * sizes[2]) {
        threads.y *= 2;
      }
      else if (static_cast<index_t>(threads.z) < sizes[0]) {
        threads.z *= 2;
      }
      nt *= 2;
    }
    
    // cuda restricts maximum block size in z to 64
    if( threads.z > 64 ) {
      threads.z = 64;
    }
    
    // launch as many blocks as necessary
    blocks.x = static_cast<int>((sizes[3] + threads.x - 1) / threads.x);
    blocks.y = static_cast<int>((sizes[1] * sizes[2] + threads.y - 1) / threads.y);
    blocks.z = static_cast<int>((sizes[0] + threads.z - 1) / threads.z);
    
    if(blocks.y > 65535) {
      blocks.y = 65535;
      stride = true;
    }
    if(blocks.z > 65535) {
      blocks.z = 65535;
      stride = true;
    }
  }  
  else {
    size_t dims = std::accumulate(std::begin(sizes), std::end(sizes), 1, std::multiplies<index_t>());
    threads.x = std::min(((int)dims + 31)/32 * 32, max_cta_size);

    // launch as many blocks as necessary
    blocks.x = static_cast<int>((dims + threads.x - 1) / threads.x);
    blocks.y = 1;
    blocks.z = 1;
  } 
  return stride;
}
} // end namespace detail
} // end namespace matx
