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
////////////////////////////////////////////////////////////////////////////////

#pragma once

#include "matx/core/defines.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>

namespace matx::examples::sarbp {

inline constexpr double AUTO_SOFT_L2_TARGET_MULTIPLIER = 0.50;
inline constexpr double AUTO_HARD_L2_LIMIT_MULTIPLIER = 0.80;
inline constexpr std::size_t AUTO_MIN_CACHE_TARGET_BYTES = 16ULL * 1024ULL * 1024ULL;
inline constexpr index_t AUTO_BLOCK_GRANULARITY = 256;
inline constexpr index_t AUTO_MIN_BLOCK_SIZE = 256;

struct AutoConfig {
  index_t block_size{};
  index_t image_tiles{1};
  std::size_t soft_cache_target_bytes{};
  std::size_t hard_cache_limit_bytes{};
  double estimated_working_set_bytes{};
};

inline double estimate_working_set_bytes(index_t block_size,
                                         index_t image_tiles,
                                         std::size_t profile_bytes_per_pulse,
                                         std::size_t phase_lut_bytes)
{
  const double tiles = static_cast<double>(std::max<index_t>(image_tiles, 1));
  const double profile_bytes =
      static_cast<double>(std::max<index_t>(block_size, 0)) *
      static_cast<double>(profile_bytes_per_pulse);
  return (static_cast<double>(phase_lut_bytes) + profile_bytes) /
      (tiles * tiles);
}

inline index_t round_to_nearest_multiple(double value, index_t multiple)
{
  if (value <= 0.0 || multiple <= 0) {
    return 0;
  }

  const index_t max_value = std::numeric_limits<index_t>::max();
  const index_t max_multiple = max_value - max_value % multiple;
  if (value >= static_cast<double>(max_multiple)) {
    return max_multiple;
  }

  return static_cast<index_t>(
      std::floor(value / static_cast<double>(multiple) + 0.5)) * multiple;
}

inline index_t choose_image_tiles(index_t block_size,
                                  index_t max_image_tiles,
                                  std::size_t profile_bytes_per_pulse,
                                  std::size_t phase_lut_bytes,
                                  double hard_cache_limit_bytes)
{
  const index_t max_tiles = std::max<index_t>(max_image_tiles, 1);
  for (index_t tiles = 1; tiles <= max_tiles; ++tiles) {
    if (estimate_working_set_bytes(block_size, tiles, profile_bytes_per_pulse,
                                   phase_lut_bytes) <= hard_cache_limit_bytes) {
      return tiles;
    }
  }

  return max_tiles;
}

// A requested size/count of zero means that dimension should be selected
// automatically. Positive requested values are preserved as explicit overrides.
inline AutoConfig choose_auto_config(index_t num_pulses,
                                     index_t max_image_tiles,
                                     std::size_t profile_bytes_per_pulse,
                                     std::size_t phase_lut_bytes,
                                     std::size_t l2_cache_bytes,
                                     index_t requested_block_size = 0,
                                     index_t requested_image_tiles = 0)
{
  AutoConfig result{};
  if (num_pulses <= 0) {
    return result;
  }

  const bool auto_block = requested_block_size <= 0;
  const bool auto_tiles = requested_image_tiles <= 0;
  result.block_size = auto_block
      ? num_pulses
      : std::min(requested_block_size, num_pulses);
  result.image_tiles = auto_tiles ? 1 : requested_image_tiles;

  if (profile_bytes_per_pulse == 0 || l2_cache_bytes == 0) {
    result.estimated_working_set_bytes = estimate_working_set_bytes(
        result.block_size, result.image_tiles, profile_bytes_per_pulse,
        phase_lut_bytes);
    return result;
  }

  const double l2_bytes = static_cast<double>(l2_cache_bytes);
  const double hard_limit = l2_bytes * AUTO_HARD_L2_LIMIT_MULTIPLIER;
  const double soft_target = std::min(
      hard_limit,
      std::max(l2_bytes * AUTO_SOFT_L2_TARGET_MULTIPLIER,
               static_cast<double>(AUTO_MIN_CACHE_TARGET_BYTES)));
  result.hard_cache_limit_bytes = static_cast<std::size_t>(hard_limit);
  result.soft_cache_target_bytes = static_cast<std::size_t>(soft_target);

  const index_t min_block = std::min(num_pulses, AUTO_MIN_BLOCK_SIZE);
  if (auto_tiles) {
    const index_t block_for_tiling = auto_block ? min_block : result.block_size;
    result.image_tiles = choose_image_tiles(
        block_for_tiling, max_image_tiles, profile_bytes_per_pulse,
        phase_lut_bytes, hard_limit);
  }

  if (auto_block) {
    const double tiles = static_cast<double>(result.image_tiles);
    const double profile_budget = std::max(
        0.0, soft_target * tiles * tiles -
            static_cast<double>(phase_lut_bytes));
    const double raw_block_size =
        profile_budget / static_cast<double>(profile_bytes_per_pulse);

    result.block_size = std::max(
        min_block,
        round_to_nearest_multiple(raw_block_size, AUTO_BLOCK_GRANULARITY));
    result.block_size = std::min(result.block_size, num_pulses);

    // Rounding to the nearest 256 pulses may cross the hard cache limit. Back
    // down by full kernel amortization units, but never select fewer than 256
    // pulses (or fewer than the complete input when it has under 256 pulses).
    while (result.block_size > min_block &&
           estimate_working_set_bytes(result.block_size, result.image_tiles,
                                      profile_bytes_per_pulse, phase_lut_bytes) > hard_limit) {
      if (result.block_size == num_pulses &&
          result.block_size % AUTO_BLOCK_GRANULARITY != 0) {
        result.block_size =
            (result.block_size / AUTO_BLOCK_GRANULARITY) * AUTO_BLOCK_GRANULARITY;
      } else {
        result.block_size -= AUTO_BLOCK_GRANULARITY;
      }
      result.block_size = std::max(result.block_size, min_block);
    }
  }

  result.estimated_working_set_bytes = estimate_working_set_bytes(
      result.block_size, result.image_tiles, profile_bytes_per_pulse,
      phase_lut_bytes);
  return result;
}

}  // namespace matx::examples::sarbp
