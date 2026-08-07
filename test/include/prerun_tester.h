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

// Test-only utility: a lifecycle probe operator used to verify that composing
// operators forward PreRun()/PostRun() to every operand that needs it. Wrap each
// operand passed to a composing operator in make_prerun_tester() and assert the
// resulting PreRunLifecycle counters after run(). This is intentionally NOT part
// of the public MatX headers.
//
// The probe is a transparent pass-through: operator() forwards directly to the
// wrapped operand (so the composed result is unchanged and correct for every
// element type and executor), while PreRun()/PostRun() record that the composing
// operator forwarded the lifecycle calls. The counters are the assertion: if a
// composing operator fails to forward PreRun to an operand, prerun_count stays 0.
// This directly measures the forwarding contract that the real bug violated
// (an unforwarded transform operand never allocates/fills its temporary).
//
// SUPPORTS_JIT is always false for PreRunTesterOp. Tests using this probe MUST
// use the *WithoutJIT typed-test suites (e.g. OperatorTestsNumericAllExecsWithoutJIT),
// NOT the regular *AllExecs suites. When MATX_EN_JIT is enabled, the *AllExecs
// suites include CUDAJITExecutor; BaseOp::run() calls ex.Exec() directly and
// CUDAJITExecutor throws for operators with SUPPORTS_JIT=false.

#include "matx.h"
#include "gtest/gtest.h"

#include <string>
#include <string_view>
#include <utility>

namespace matx {
namespace test {

// Observable PreRun/PostRun lifecycle state. PreRunTesterOp is copied into the
// expression tree (including by value into __device__ code), so it holds only a
// raw observer pointer to this state; the state itself is owned by the caller
// (e.g. a stack local in the test) and outlives the run.
struct PreRunLifecycle {
  int prerun_count = 0;
  int postrun_count = 0;
  int active = 0;                     // logical ownership: ++ in PreRun, -- in PostRun
  const void *tracked_ptr = nullptr;  // set in PreRun, cleared in PostRun
};

// Plain (non-transform) test operator that wraps an operand and records the
// PreRun/PostRun calls forwarded to it by a composing operator. operator()
// transparently forwards to the wrapped operand.
//
// By design there is no prerun_done_ idempotency guard: every PreRun/PostRun
// call is counted so a test can detect both missing AND duplicated forwarding.
// Use one PreRunLifecycle per operand per run() and assert the expected count.
template <typename Op>
class PreRunTesterOp : public matx::BaseOp<PreRunTesterOp<Op>> {
public:
  using value_type = typename Op::value_type;
  using matxop = bool;

private:
  static constexpr int RANK = Op::Rank();

  mutable typename matx::detail::base_type_t<Op> op_;
  // Raw observer pointer, NOT a shared_ptr: this operator is copied (including
  // by value into __device__ code, e.g. shift's get_impl), and copying a
  // std::shared_ptr on the device is invalid. The PreRunLifecycle is a
  // test-local variable that outlives the run(); this pointer is only
  // dereferenced host-side in PreRun/PostRun.
  PreRunLifecycle *state_;

public:
  PreRunTesterOp(const Op &op, PreRunLifecycle *state) : op_(op), state_(state) {}

  __MATX_INLINE__ std::string str() const { return "prerun_tester(" + op_.str() + ")"; }

  template <typename CapType, typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
    return matx::detail::get_value<CapType>(op_, indices...);
  }

  template <typename... Is>
  __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ decltype(auto) operator()(Is... indices) const {
    return this->operator()<matx::detail::DefaultCapabilities>(indices...);
  }

  static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank() {
    return RANK;
  }

  constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const {
    return op_.Size(dim);
  }

  template <matx::detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability(InType &in) const {
    if constexpr (Cap == matx::detail::OperatorCapability::SUPPORTS_JIT) {
      return false;  // test operator does not support JIT; force the normal path
    } else if constexpr (Cap == matx::detail::OperatorCapability::ELEMENTS_PER_THREAD) {
      const auto my_cap = cuda::std::array<matx::detail::ElementsPerThread, 2>{
          matx::detail::ElementsPerThread::ONE, matx::detail::ElementsPerThread::ONE};
      return matx::detail::combine_capabilities<Cap>(
          my_cap, matx::detail::get_operator_capability<Cap>(op_, in));
    } else {
      auto self_has_cap = matx::detail::capability_attributes<Cap>::default_value;
      return matx::detail::combine_capabilities<Cap>(
          self_has_cap, matx::detail::get_operator_capability<Cap>(op_, in));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PreRun(ShapeType &&shape, Executor &&ex) const noexcept {
    state_->prerun_count++;
    state_->active++;
    state_->tracked_ptr = state_;
    if constexpr (matx::is_matx_op<Op>()) {
      op_.PreRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }

  template <typename ShapeType, typename Executor>
  __MATX_INLINE__ void PostRun(ShapeType &&shape, Executor &&ex) const noexcept {
    state_->postrun_count++;
    state_->active--;
    state_->tracked_ptr = nullptr;
    if constexpr (matx::is_matx_op<Op>()) {
      op_.PostRun(std::forward<ShapeType>(shape), std::forward<Executor>(ex));
    }
  }
};

// Wrap `op` so its PreRun/PostRun lifecycle is recorded in `state`. Use one
// PreRunLifecycle per operand and pass the wrapped operands to the composing
// operator under test. `state` must outlive the run() (e.g. a test-local
// variable); taking it by reference makes a dangling temporary a compile error.
template <typename Op>
__MATX_INLINE__ auto make_prerun_tester(const Op &op, PreRunLifecycle &state) {
  return PreRunTesterOp<Op>(op, &state);
}

// Pure predicate (no gtest dependency) for programmatic checks.
inline bool lifecycle_clean(const PreRunLifecycle &s, int expected_calls = 1) {
  return s.prerun_count == expected_calls && s.postrun_count == s.prerun_count &&
         s.active == 0 && s.tracked_ptr == nullptr;
}

// gtest helper: one call per operand. Each invariant is its own EXPECT, and the
// SCOPED_TRACE label identifies which operand failed.
inline void ExpectLifecycleClean(const PreRunLifecycle &s,
                                 std::string_view operand, int expected_calls = 1) {
  SCOPED_TRACE(std::string("prerun_tester operand: ") + std::string(operand));
  EXPECT_EQ(s.prerun_count, expected_calls);      // PreRun forwarded exactly expected_calls times
  EXPECT_EQ(s.postrun_count, s.prerun_count);     // PostRun balanced PreRun
  EXPECT_EQ(s.active, 0);                          // lifecycle ownership released
  EXPECT_EQ(s.tracked_ptr, nullptr);               // tracked pointer cleared
}

}  // namespace test
}  // namespace matx
