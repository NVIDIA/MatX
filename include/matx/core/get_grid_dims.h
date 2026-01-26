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

#include "matx/core/defines.h"
#include "matx/core/error.h"
#include <cuda/std/array>
#include <cuda/std/functional>
#include <cuda/std/__numeric/accumulate.h>

namespace matx {
namespace detail {

template <int RANK>
inline bool get_grid_dims(dim3 &blocks, dim3 &threads, const cuda::std::array<index_t, RANK> &sizes, index_t ept,
                          int max_cta_size = 1024)
{
  bool stride = false;
  [[maybe_unused]] int nt = 1;
  threads.x = 1;
  threads.y = 1;
  threads.z = 1;

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  if constexpr (RANK == 0) {
    blocks.x = 1;
    blocks.y = 1;
    blocks.z = 1;
  }
  else if constexpr (RANK == 1) {
    while (nt < max_cta_size) {
      if ((static_cast<index_t>(threads.x) * ept) < sizes[0]) {
        threads.x *= 2;
      }
      nt *= 2;
    }
    // launch as many blocks as necessary
    index_t rnd_threads = ept * threads.x;
    blocks.x = static_cast<int>((sizes[0] + rnd_threads - 1) / rnd_threads);  
    blocks.y = 1;
    blocks.z = 1;  
  }
  else if constexpr (RANK == 2) {
    while (nt < max_cta_size) {
      if ((static_cast<index_t>(threads.x) * ept) < sizes[1]) {
        threads.x *= 2;
      }
      else if (static_cast<index_t>(threads.y) < sizes[0]) {
        threads.y *= 2;
      }
      nt *= 2;
    }
    // launch as many blocks as necessary
    index_t rnd_threads_x = ept * threads.x;
    blocks.x = static_cast<int>((sizes[1] + rnd_threads_x - 1) / rnd_threads_x);
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
    size_t dims = cuda::std::accumulate(cuda::std::begin(sizes), cuda::std::end(sizes), 1, cuda::std::multiplies<index_t>());
    threads.x = std::min(((int)dims + 31)/32 * 32, max_cta_size);

    // launch as many blocks as necessary
    blocks.x = static_cast<int>((dims + threads.x - 1) / threads.x);
    blocks.y = 1;
    blocks.z = 1;
  } 
  return stride;
}

// For JIT code we want to use a grid-stride loop always
template <int RANK>
inline bool get_grid_dims_block(dim3 &blocks, dim3 &threads, const cuda::std::array<index_t, RANK> &sizes, index_t ept, int groups_per_block,
                          int max_cta_size = 1024, bool force_size = false)
{
  bool stride = false;
  [[maybe_unused]] int nt = 1;
  threads.x = force_size ? max_cta_size : 1;
  threads.y = groups_per_block;
  threads.z = 1;
  blocks.x = 1;  
  blocks.y = 1;
  blocks.z = 1;    

  if (RANK > 1) {
    MATX_ASSERT_STR_EXP(sizes[sizes.size() - 2] % groups_per_block, 0, matxInvalidParameter, "Second to last dimension must be divisible by groups_per_block");
  }

  // Dynamic logic to pick thread block size.
  //   Fill in order x, y, z up to 1024 threads
  if constexpr (RANK == 0) {
    if (!force_size) {
      threads.x = 1;
    }
  }
  else if constexpr (RANK == 1) {
    if (!force_size) {
      while (nt < max_cta_size) {
        if ((static_cast<index_t>(threads.x) * ept) < sizes[0]) {
          threads.x *= 2;
        }
        
        nt *= 2;
      }
    }
  }
  else if constexpr (RANK == 2) {
    if (!force_size) {
      while (nt < max_cta_size) {
        if ((static_cast<index_t>(threads.x) * ept) < sizes[1]) {
          threads.x *= 2;
        }

        nt *= 2;
      }
    }

    // If we have multiple groups per block, we need to adjust the block size
    if (threads.y > 1) {
      blocks.x = static_cast<int>(static_cast<int64_t>(sizes[0]) / static_cast<int64_t>(threads.y));
    }
    else {
      blocks.x = static_cast<int>(sizes[0]);
    }
  }
  else if constexpr (RANK == 3) {
    if (!force_size) {
      while (nt < max_cta_size) {
        if (static_cast<index_t>(threads.x) * ept < sizes[2]) {
          threads.x *= 2;
        }

        nt *= 2;
      }
    }

    // If we have multiple groups per block, we need to adjust the block size
    if (threads.y > 1) {
      blocks.x = static_cast<int>(static_cast<int64_t>(sizes[1]) / static_cast<int64_t>(threads.y));
    }
    else {
      blocks.x = static_cast<int>(sizes[1]);
    }    

    // launch as many blocks as necessary
    blocks.y = static_cast<int>(sizes[0]);
    
    if(blocks.x > 65535) {
      blocks.x = 65535;
      stride = true;
    }
    if(blocks.y > 65535) {
      blocks.y = 65535;
      stride = true;
    }

  }  
  else if constexpr (RANK == 4) {
    if (!force_size) {
      while (nt < max_cta_size) {
        if (static_cast<index_t>(threads.x) * ept < sizes[3]) {
          threads.x *= 2;
      }

        nt *= 2;
      }
    }
    
    // If we have multiple groups per block, we need to adjust the block size
    if (threads.y > 1) {
      blocks.x = static_cast<int>(static_cast<int64_t>(sizes[2]) / static_cast<int64_t>(threads.y));
    }
    else {
      blocks.x = static_cast<int>(sizes[2]);
    }  

    // launch as many blocks as necessary
    blocks.y = static_cast<int>(sizes[1]);
    blocks.z = static_cast<int>(sizes[0]);
    
    if(blocks.x > 65535) {
      blocks.x = 65535;
      stride = true;
    }
    if(blocks.y > 65535) {
      blocks.y = 65535;
      stride = true;
    }
    if(blocks.z > 65535) {
      blocks.z = 65535;
      stride = true;
    }    
  }  

  MATX_LOG_DEBUG("Blocks {}x{}x{} Threads {}x{}x{} groups_per_block={}", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z, groups_per_block);
  return stride;
}

// For 2D block operators (e.g., cuBLASDx GEMM) where all threads in a block cooperate 
// on the last 2 dimensions and blockIdx is used purely for batching
template <int RANK>
inline bool get_grid_dims_block_2d(dim3 &blocks, dim3 &threads, 
                                    const cuda::std::array<index_t, RANK> &sizes,
                                    int block_dim) {
  // Threads are set to block_dim in x, y and z are 1
  // All threads cooperate via flattened thread ID in the kernel
  threads.x = block_dim;
  threads.y = 1;
  threads.z = 1;
  
  // Grid covers batch dimensions only (dims 0 to RANK-3)
  blocks.x = 1;
  blocks.y = 1;
  blocks.z = 1;
  
  if constexpr (RANK == 2) {
    blocks.x = 1;  // Single block for entire 2D output
  }
  else if constexpr (RANK == 3) {
    blocks.x = static_cast<int>(sizes[0]);  // Batch dim
  }
  else if constexpr (RANK == 4) {
    blocks.x = static_cast<int>(sizes[1]);  // Second-to-last batch
    blocks.y = static_cast<int>(sizes[0]);  // First batch dim
  }
  else if constexpr (RANK > 4) {
    MATX_THROW(matxNotSupported, "Block2D grid dims not supported for rank > 4");
    return true;
  }
  
  if constexpr (RANK >= 2 && RANK <= 4) {
    constexpr int kMaxGridDim = 65535;
    if (blocks.x > kMaxGridDim || blocks.y > kMaxGridDim) {
      MATX_THROW(matxInvalidParameter, "Block2D grid dims exceed CUDA limit (65535)");
      return true;
    }
  }
  
  MATX_LOG_DEBUG("Block2D: Blocks {}x{}x{} Threads {}x{}x{}", blocks.x, blocks.y, blocks.z, threads.x, threads.y, threads.z);
  
  // No stride needed for now - could be extended for very large batches
  return false;
}
} // end namespace detail
} // end namespace matx
