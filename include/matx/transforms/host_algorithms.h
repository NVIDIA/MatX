////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2026, NVIDIA Corporation
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

#include <algorithm>
#include <functional>
#include <numeric>

#include "matx/executors/host.h"

#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
#include <thrust/copy.h>
#include <thrust/extrema.h>
#include <thrust/logical.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sort.h>
#include <thrust/system/omp/execution_policy.h>
#include <thrust/unique.h>
#endif

namespace matx::detail {

template <ThreadsMode MODE>
__MATX_INLINE__ bool use_parallel_host_thrust(const HostExecutor<MODE> &exec)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    const int threads = exec.GetNumThreads();
    if (threads > 1) {
      omp_set_num_threads(threads);
      return true;
    }
  }
#else
  static_cast<void>(exec);
#endif

  return false;
}

template <typename InputIt, typename T, typename BinaryOp, ThreadsMode MODE>
__MATX_INLINE__ T host_reduce(const HostExecutor<MODE> &exec, InputIt first, InputIt last, T init, BinaryOp op)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::reduce(thrust::omp::par, first, last, init, op);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::accumulate(first, last, init, op);
}

template <typename InputIt, typename T, ThreadsMode MODE>
__MATX_INLINE__ T host_reduce(const HostExecutor<MODE> &exec, InputIt first, InputIt last, T init)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::reduce(thrust::omp::par, first, last, init);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::accumulate(first, last, init);
}

template <typename RandomIt, typename Compare, ThreadsMode MODE>
__MATX_INLINE__ void host_sort(const HostExecutor<MODE> &exec, RandomIt first, RandomIt last, Compare comp)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      thrust::sort(thrust::omp::par, first, last, comp);
      return;
    }
  }
#else
  static_cast<void>(exec);
#endif

  std::sort(first, last, comp);
}

template <typename RandomIt, ThreadsMode MODE>
__MATX_INLINE__ void host_sort(const HostExecutor<MODE> &exec, RandomIt first, RandomIt last)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      thrust::sort(thrust::omp::par, first, last);
      return;
    }
  }
#else
  static_cast<void>(exec);
#endif

  std::sort(first, last);
}

template <typename InputIt, typename OutputIt, typename Compare, ThreadsMode MODE>
__MATX_INLINE__ void host_sort_copy(const HostExecutor<MODE> &exec,
                                    InputIt first,
                                    InputIt last,
                                    OutputIt out_first,
                                    OutputIt out_last,
                                    Compare comp)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  // Thrust does not provide partial_sort_copy; copy+sort is equivalent only when ranges match.
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (((last - first) == (out_last - out_first)) && use_parallel_host_thrust(exec)) {
      thrust::copy(thrust::omp::par, first, last, out_first);
      thrust::sort(thrust::omp::par, out_first, out_last, comp);
      return;
    }
  }
#else
  static_cast<void>(exec);
#endif

  std::partial_sort_copy(first, last, out_first, out_last, comp);
}

template <typename InputIt, typename OutputIt, ThreadsMode MODE>
__MATX_INLINE__ void host_sort_copy(const HostExecutor<MODE> &exec,
                                    InputIt first,
                                    InputIt last,
                                    OutputIt out_first,
                                    OutputIt out_last)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  // Thrust does not provide partial_sort_copy; copy+sort is equivalent only when ranges match.
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (((last - first) == (out_last - out_first)) && use_parallel_host_thrust(exec)) {
      thrust::copy(thrust::omp::par, first, last, out_first);
      thrust::sort(thrust::omp::par, out_first, out_last);
      return;
    }
  }
#else
  static_cast<void>(exec);
#endif

  std::partial_sort_copy(first, last, out_first, out_last);
}

template <typename InputIt, typename OutputIt, ThreadsMode MODE>
__MATX_INLINE__ void host_inclusive_scan(const HostExecutor<MODE> &exec,
                                         InputIt first,
                                         InputIt last,
                                         OutputIt out)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      thrust::inclusive_scan(thrust::omp::par, first, last, out);
      return;
    }
  }
#else
  static_cast<void>(exec);
#endif

  std::partial_sum(first, last, out);
}

template <typename ForwardIt, ThreadsMode MODE>
__MATX_INLINE__ ForwardIt host_max_element(const HostExecutor<MODE> &exec, ForwardIt first, ForwardIt last)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::max_element(thrust::omp::par, first, last);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::max_element(first, last);
}

template <typename ForwardIt, ThreadsMode MODE>
__MATX_INLINE__ ForwardIt host_min_element(const HostExecutor<MODE> &exec, ForwardIt first, ForwardIt last)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::min_element(thrust::omp::par, first, last);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::min_element(first, last);
}

template <typename InputIt, typename UnaryPredicate, ThreadsMode MODE>
__MATX_INLINE__ bool host_any_of(const HostExecutor<MODE> &exec, InputIt first, InputIt last, UnaryPredicate pred)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::any_of(thrust::omp::par, first, last, pred);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::any_of(first, last, pred);
}

template <typename InputIt, typename UnaryPredicate, ThreadsMode MODE>
__MATX_INLINE__ bool host_all_of(const HostExecutor<MODE> &exec, InputIt first, InputIt last, UnaryPredicate pred)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::all_of(thrust::omp::par, first, last, pred);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::all_of(first, last, pred);
}

template <typename ForwardIt, ThreadsMode MODE>
__MATX_INLINE__ ForwardIt host_unique(const HostExecutor<MODE> &exec, ForwardIt first, ForwardIt last)
{
  static_cast<void>(exec);
#if defined(MATX_EN_OMP) && !defined(__CUDACC_RTC__)
  if constexpr (MODE != ThreadsMode::SINGLE) {
    if (use_parallel_host_thrust(exec)) {
      return thrust::unique(thrust::omp::par, first, last);
    }
  }
#else
  static_cast<void>(exec);
#endif

  return std::unique(first, last);
}

} // namespace matx::detail
