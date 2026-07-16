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

// Shared internal helpers for the streaming polyphase objects.

#pragma once

#include "matx/core/operator_options.h"
#include "matx/core/type_utils.h"

namespace matx {
namespace detail {

// Output plan for one streaming feed()/flush() call: the number of outputs this
// call owns and the start index of that window in the local output grid.
// Computed from sizes alone (no segment data), so a feed() can validate the
// output buffer BEFORE running the segment operator's lifecycle.
struct StreamSlicePlan {
  index_t lo;
  index_t cnt;
};

// RAII balance for an explicitly-run operator lifecycle. Construction runs the
// operand's PreRun (materializing any operator that stages into a temporary)
// and destruction runs the matching PostRun on every exit path, including
// exceptions. A throw between the two (e.g. from exec.Exec, or a size check)
// therefore neither leaks the temporary nor leaves a half-run lifecycle. Both
// calls are guarded by is_matx_op, so a non-MatX operand is a no-op.
template <typename Op, typename ExecT>
class SegmentLifecycleGuard {
public:
  SegmentLifecycleGuard(const Op &op, ExecT &exec) : op_(op), exec_(exec)
  {
    if constexpr (is_matx_op<Op>()) {
      op_.PreRun(NoShape{}, exec_);
    }
  }

  ~SegmentLifecycleGuard()
  {
    if constexpr (is_matx_op<Op>()) {
      op_.PostRun(NoShape{}, exec_);
    }
  }

  SegmentLifecycleGuard(const SegmentLifecycleGuard &) = delete;
  SegmentLifecycleGuard &operator=(const SegmentLifecycleGuard &) = delete;

private:
  const Op &op_;
  ExecT &exec_;
};

}  // namespace detail
}  // namespace matx
