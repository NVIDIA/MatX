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

namespace matx
{

  template <typename T>
  __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ T MAX(T a)
  {
    return a;
  }

  template <typename T, typename... Args>
  __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ T MAX(T a, Args... args)
  {
    auto v = MAX(args...);
    return (a >= v) ? a : v;
  }

  template <class T, class M = T>
  __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t get_rank()
  {
    if constexpr (is_matx_op<M>())
      return T::Rank();
    else
      return -1;

    // work around for compiler bug/warning
    if constexpr (!is_matx_op<M>())
      return -1;
    else
      return T::Rank();
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto get_size([[maybe_unused]] T &a,
                                              [[maybe_unused]] uint32_t dim)
  {
    if constexpr (is_matx_op<M>())
      return a.Size(dim);
    else
      return 0;

    // work around for compiler bug/warning
    if constexpr (!is_matx_op<M>())
      return 0;
    else
      return a.Size(dim);
  }

  template <int RANK, class T, class M = T>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto
  get_expanded_size([[maybe_unused]] T &a, [[maybe_unused]] uint32_t dim)
  {
    index_t size = 0;
    constexpr int32_t rank = get_rank<T>();

    if constexpr (rank > 0)
    {
      constexpr int32_t diff = RANK - rank;
      if constexpr (diff > 0)
      {
        // auto expansion case,  remap dimension by difference in ranks
        if (dim > diff)
        {
          size = get_size(a, dim - diff);
        }
      }
      else
      {
        size = get_size(a, dim);
      }
    }

    return size;
  }

  // get_matx_value is a work around some compiler bugs
  // it only works with matxop types.  It should only be
  // called from get_value below.
  // We also have to do this recursively to get around bug
  // We also have to invert logic and repeat to get around bug
  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(T &i, index_t idx)
  {
    if constexpr (T::Rank() == 1)
    {
      return i(idx);
    }
    else
    {
      return i();
    }

    // Bug WAR
    if constexpr (T::Rank() != 1)
    {
      return i();
    }
    else
    {
      return i(idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(T &i, index_t idy, index_t idx)
  {
    if constexpr (T::Rank() == 2)
    {
      return i(idy, idx);
    }
    else
    {
      return get_matx_value(i, idx);
    }

    // Bug WAR
    if constexpr (T::Rank() != 2)
    {
      return get_matx_value(i, idx);
    }
    else
    {
      return i(idy, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(T &i, index_t idz, index_t idy,
                                        index_t idx)
  {
    if constexpr (T::Rank() == 3)
    {
      return i(idz, idy, idx);
    }
    else
    {
      return get_matx_value(i, idy, idx);
    }

    // Bug WAR
    if constexpr (T::Rank() != 3)
    {
      return get_matx_value(i, idy, idx);
    }
    else
    {
      return i(idz, idy, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_matx_value(T &i, index_t idw, index_t idz,
                                        index_t idy, index_t idx)
  {
    if constexpr (T::Rank() == 4)
    {
      return i(idw, idz, idy, idx);
    }
    else
    {
      return get_matx_value(i, idz, idy, idx);
    }

    // Bug WAR
    if constexpr (T::Rank() != 4)
    {
      return get_matx_value(i, idz, idy, idx);
    }
    else
    {
      return i(idw, idz, idy, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i)
  {
    if constexpr (is_matx_op<M>())
    {
      return i();
    }
    else
    {
      return i;
    }

    // Bug WAR
    if constexpr (!is_matx_op<M>())
    {
      return i;
    }
    else
    {
      return i();
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i, [[maybe_unused]] index_t idx)
  {
    if constexpr (is_matx_op<M>())
    {
      return get_matx_value(i, idx);
    }
    else
    {
      return i;
    }

    // Bug WAR
    if constexpr (!is_matx_op<M>())
    {
      return i;
    }
    else
    {
      return get_matx_value(i, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i, [[maybe_unused]] index_t idy, [[maybe_unused]] index_t idx)
  {
    if constexpr (is_matx_op<M>())
    {
      return get_matx_value(i, idy, idx);
    }
    else
    {
      return i;
    }

    // Bug WAR
    if constexpr (!is_matx_op<M>())
    {
      return i;
    }
    else
    {
      return get_matx_value(i, idy, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i, [[maybe_unused]] index_t idz, [[maybe_unused]] index_t idy, [[maybe_unused]] index_t idx)
  {
    if constexpr (is_matx_op<M>())
    {
      return get_matx_value(i, idz, idy, idx);
    }
    else
    {
      return i;
    }

    // Bug WAR
    if constexpr (!is_matx_op<M>())
    {
      return i;
    }
    else
    {
      return get_matx_value(i, idz, idy, idx);
    }
  }

  template <class T, class M = T>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto get_value(T &i, [[maybe_unused]] index_t idw, [[maybe_unused]] index_t idz, [[maybe_unused]] index_t idy,
                                   index_t idx)
  {
    if constexpr (is_matx_op<M>())
    {
      return get_matx_value(i, idw, idz, idy, idx);
    }
    else
    {
      return i;
    }

    // Bug WAR
    if constexpr (!is_matx_op<M>())
    {
      return i;
    }
    else
    {
      return get_matx_value(i, idw, idz, idy, idx);
    }
  }

  // Returns an address of a pointer of type T aligned to new address
  template <typename T>
  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ T *AlignAddr(uint8_t *addr)
  {
    if (((uint64_t)addr % std::alignment_of_v<T>) != 0) {
      return reinterpret_cast<T *>(
          ((uint64_t)addr + (std::alignment_of_v<T> - 1)) /
          std::alignment_of_v<T> * std::alignment_of_v<T>);
    }

    return reinterpret_cast<T *>(addr);
  }    
}