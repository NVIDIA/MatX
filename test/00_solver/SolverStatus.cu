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


#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <limits>
#include <string>
#include <thread>

using namespace matx;

namespace {

// Holds a stream inside a host function until released, so a test can tell
// whether a call returned without waiting for the stream. Gives up after a
// timeout so a call that does synchronize cannot hang the test. The destructor
// releases the gate and waits for the stream, so a test that throws or returns
// early while the gate is armed cannot free it under the host function that
// still reads it.
struct StreamGate {
  std::atomic<bool> release{false};
  cudaStream_t stream = nullptr;
  bool launched = false;

  StreamGate() = default;
  StreamGate(const StreamGate &) = delete;
  StreamGate &operator=(const StreamGate &) = delete;
  ~StreamGate()
  {
    release = true;
    if (launched) {
      [[maybe_unused]] const cudaError_t err = cudaStreamSynchronize(stream);
    }
  }

  cudaError_t Launch(cudaStream_t s)
  {
    stream = s;
    const cudaError_t err = cudaLaunchHostFunc(s, &StreamGate::Wait, this);
    launched = err == cudaSuccess;
    return err;
  }

  static void CUDART_CB Wait(void *p)
  {
    auto *gate = static_cast<StreamGate *>(p);
    const auto start = std::chrono::steady_clock::now();
    while (!gate->release.load()) {
      if (std::chrono::steady_clock::now() - start > std::chrono::seconds(10)) {
        break;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
  }
};

template <typename T, typename OutT>
auto MaxAbsDiff(const OutT &a, const OutT &b, cudaExecutor &exec)
{
  using SType = typename inner_op_type_t<T>::type;
  auto d = make_tensor<SType>({});
  (d = max(abs(a - b))).run(exec);
  exec.sync();
  return d();
}

// The first `stages` stages of a host-readable status, stage-major. The caller
// has synchronized.
std::vector<int> Vals(const SolverStatus &st, index_t stages)
{
  const auto t = st.infos();
  return std::vector<int>(t.Data(), t.Data() + stages * t.Size(1));
}

// A non-blocking stream released when the test leaves its scope, including
// through a failed ASSERT. Cached plans free their workspaces on the stream they
// were created on, so they are released before the stream. Declare it before
// the tensors, gates, and executors that use it, so it is destroyed after them.
struct TestStream {
  cudaStream_t stream = nullptr;

  TestStream() { EXPECT_EQ(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), cudaSuccess); }
  TestStream(const TestStream &) = delete;
  TestStream &operator=(const TestStream &) = delete;
  ~TestStream()
  {
    if (stream != nullptr) {
      EXPECT_EQ(cudaStreamSynchronize(stream), cudaSuccess);
      ClearCaches();
      EXPECT_EQ(cudaStreamDestroy(stream), cudaSuccess);
    }
  }
  operator cudaStream_t() const { return stream; }
};

struct TestEvent {
  cudaEvent_t event = nullptr;

  TestEvent() { EXPECT_EQ(cudaEventCreateWithFlags(&event, cudaEventDisableTiming), cudaSuccess); }
  TestEvent(const TestEvent &) = delete;
  TestEvent &operator=(const TestEvent &) = delete;
  ~TestEvent()
  {
    if (event != nullptr) {
      EXPECT_EQ(cudaEventDestroy(event), cudaSuccess);
    }
  }
  operator cudaEvent_t() const { return event; }
};

cudaMemoryType MemoryType(const void *p)
{
  cudaPointerAttributes attrs{};
  EXPECT_EQ(cudaPointerGetAttributes(&attrs, p), cudaSuccess);
  return attrs.type;
}

std::string ExceptionText(const std::function<void()> &f)
{
  try {
    f();
  }
  catch (const detail::matxException &e) {
    return e.what();
  }
  return "";
}

} // namespace

template <typename T>
class SolverStatusTypedTest : public ::testing::Test {};

TYPED_TEST_SUITE(SolverStatusTypedTest, MatXFloatNonHalfTypesCUDAExec);

// A solver call with a status does not wait for the stream, ok() and check()
// read without waiting, and ok(exec), check(exec), and a user event order the
// read after the call. A gate holds the stream so the solver cannot run until
// released.
TEST(SolverStatusTest, NoHostSynchronization)
{
  MATX_ENTER_HANDLER();
  using T = float;
  // Without stream-ordered memory pools, MatX's temporary buffers fall back to
  // cudaMalloc and cudaFree, which synchronize the device, so the solver call
  // would wait behind the gate
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  if (!detail::GetDeviceAttributes(device).memory_pools_supported) {
    GTEST_SKIP() << "Device " << device << " does not support stream-ordered memory pools, so solver calls synchronize";
  }
  TestStream stream;
  TestEvent event;
  {
    cudaExecutor exec{stream};
    auto A = make_tensor<T>({4, 16, 16});
    (A = random<T>(A.Shape(), NORMAL)).run(exec);
    auto out = make_tensor<T>(A.Shape());
    auto tau = make_tensor<T>({4, 16});
    auto q = make_tensor<T>(A.Shape());
    auto r = make_tensor<T>(A.Shape());
    auto st_solver = make_solver_status(A, exec, {.stages = 1});
    auto st_econ = make_solver_status(A, exec);
    auto run = [&](int which) {
      if (which == 0) {
        (mtie(out, tau) = qr_solver(A, st_solver)).run(exec);
      }
      else {
        (mtie(q, r) = qr_econ(A, st_econ)).run(exec);
      }
    };

    // Build and cache the plans first. Plan creation is allowed to synchronize.
    run(0);
    run(1);
    exec.sync();

    {
      // A status MatX allocated reports a read before its first call writes it
      // as a failure, not success
      auto st_fresh = make_solver_status(A, exec, {.stages = 1});
      StreamGate gate;
      ASSERT_EQ(gate.Launch(stream), cudaSuccess);
      (mtie(out, tau) = qr_solver(A, st_fresh)).run(exec);
      EXPECT_FALSE(st_fresh.ok());
      const auto msg = ExceptionText([&] { st_fresh.check(); });
      EXPECT_NE(msg.find("4 failures"), std::string::npos) << msg;
      EXPECT_NE(msg.find("batch 0: no solver result written yet"), std::string::npos) << msg;
      gate.release = true;
      EXPECT_TRUE(st_fresh.ok(exec));
    }

    for (int which = 0; which < 2; which++) {
      SCOPED_TRACE(which == 0 ? "qr_solver" : "qr_econ");
      auto &st = which == 0 ? st_solver : st_econ;
      {
        // A stale failure the next call overwrites
        st.infos()(0, 1) = -7;
        StreamGate gate;
        ASSERT_EQ(gate.Launch(stream), cudaSuccess);
        const auto start = std::chrono::steady_clock::now();
        run(which);
        // ok() and check() read what the status holds now, before the call runs
        EXPECT_FALSE(st.ok());
        EXPECT_THROW(st.check(), detail::matxException);
        // A synchronizing call or read would block until the gate times out
        EXPECT_LT(std::chrono::steady_clock::now() - start, std::chrono::seconds(5));
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
        gate.release = true;
        EXPECT_TRUE(st.ok(exec));
        EXPECT_NO_THROW(st.check(exec));
      }
      {
        st.infos()(0, 1) = -7;
        // example-begin solver-status-test-4
        // An event recorded after the call orders the read without waiting for later work
        run(which);
        cudaEventRecord(event, exec.getStream());
        StreamGate later_work;   // stands in for work queued after the solver
        ASSERT_EQ(later_work.Launch(exec.getStream()), cudaSuccess);
        cudaEventSynchronize(event);
        EXPECT_NO_THROW(st.check());
        // example-end solver-status-test-4
        EXPECT_EQ(cudaStreamQuery(stream), cudaErrorNotReady);
      }
    }
  }
  {
    // Single-matrix inv uses cuSolver rather than cuBLAS, and its getrf and
    // getrs, and so its status writes, run on the executor's stream: waiting on
    // the legacy default stream while the gate holds it writes nothing
    SCOPED_TRACE("single-matrix inv");
    cudaExecutor exec{stream};
    auto A = make_tensor<T>({4, 4});
    (A = random<T>(A.Shape(), NORMAL)).run(exec);
    auto ref = make_tensor<T>(A.Shape());
    (ref = inv(A)).run(exec);
    auto out = make_tensor<T>(A.Shape());
    auto st = make_solver_status(A, exec);
    (out = inv(A, st)).run(exec);   // build and cache the plan
    exec.sync();
    std::fill_n(st.infos().Data(), st.infos().TotalSize(), detail::solver_info_not_written);

    StreamGate gate;
    ASSERT_EQ(gate.Launch(stream), cudaSuccess);
    (out = inv(A, st)).run(exec);
    ASSERT_EQ(cudaStreamSynchronize(cudaStreamLegacy), cudaSuccess);
    EXPECT_EQ(Vals(st, 2), std::vector<int>(2, detail::solver_info_not_written));
    gate.release = true;
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  MATX_EXIT_HANDLER();
}

// A CUDA status works on any stream of its device, and is read only through
// executors of its kind
TEST(SolverStatusTest, StatusUsableOnAnyStreamOfItsKind)
{
  MATX_ENTER_HANDLER();
  using T = float;
  TestStream s1;
  TestStream s2;
  {
    cudaExecutor exec1{s1};
    cudaExecutor exec2{s2};
    auto A = make_tensor<T>({4, 16, 16});
    (A = random<T>(A.Shape(), NORMAL)).run(exec1);
    exec1.sync();
    auto out = make_tensor<T>(A.Shape());
    auto tau = make_tensor<T>({4, 16});

    auto st = make_solver_status(A, exec1, {.stages = 1});
    (mtie(out, tau) = qr_solver(A, st)).run(exec2);
    EXPECT_TRUE(st.ok(exec2));
    (mtie(out, tau) = qr_solver(A, st)).run(exec1);
    EXPECT_TRUE(st.ok(exec1));

    // Reading through an executor of the other kind throws
    SingleThreadedHostExecutor host_exec{};
    auto hst = make_solver_status(A, host_exec, {.stages = 1});
    EXPECT_THROW(hst.ok(exec1), detail::matxException);
    EXPECT_THROW(st.ok(host_exec), detail::matxException);
    EXPECT_THROW(st.check(host_exec), detail::matxException);
    exec1.sync();
  }
  MATX_EXIT_HANDLER();
}

TEST(SolverStatusTest, CheckReportsFailingMatrices)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  auto A = make_tensor<T>({2, 3, 4, 4});
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto info = make_tensor<int>({2, 2, 3}, MATX_HOST_MEMORY);
  auto st = make_solver_status(A, info, exec);
  auto q = make_tensor<T>(A.Shape());
  auto r = make_tensor<T>(A.Shape());
  (mtie(q, r) = qr_econ(A, st)).run(exec);
  ASSERT_TRUE(st.ok(exec));

  // Simulate failures: orgqr on matrix [1][1] and geqrf on matrix [0][2], whose
  // linear indices over the [2, 3] batch dimensions are 4 and 2
  info(1, 1, 1) = -4;
  info(0, 0, 2) = -6;
  EXPECT_FALSE(st.ok());
  const std::string msg = ExceptionText([&] { st.check(); });
  EXPECT_NE(msg.find("matxSolverError"), std::string::npos) << msg;
  EXPECT_NE(msg.find("2 failures"), std::string::npos) << msg;
  EXPECT_NE(msg.find("stage 0, batch 2: parameter 6 had an illegal value"), std::string::npos) << msg;
  EXPECT_NE(msg.find("stage 1, batch 4: parameter 4 had an illegal value"), std::string::npos) << msg;

  // Positive values are given as is
  info(0, 0, 1) = 2;
  const std::string msg2 = ExceptionText([&] { st.check(); });
  EXPECT_NE(msg2.find("3 failures"), std::string::npos) << msg2;
  EXPECT_NE(msg2.find("stage 0, batch 1: info = 2"), std::string::npos) << msg2;

  // Running again overwrites the previous results
  (mtie(q, r) = qr_econ(A, st)).run(exec);
  EXPECT_TRUE(st.ok(exec));
  MATX_EXIT_HANDLER();
}

// Status tensors and params that no solver could use are rejected at creation
TEST(SolverStatusTest, RejectsInvalidStatusTensors)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  SingleThreadedHostExecutor host_exec{};
  auto A = make_tensor<T>({3, 4, 4});

  // Batch dimensions must match the input
  auto wrong_batch = make_tensor<int>({2, 4}, MATX_DEVICE_MEMORY);
  EXPECT_THROW(make_solver_status(A, wrong_batch, exec), detail::matxException);

  // Must be contiguous
  auto wide = make_tensor<int>({2, 6}, MATX_DEVICE_MEMORY);
  auto strided = slice(wide, {0, 0}, {2, 6}, {1, 2});
  EXPECT_THROW(make_solver_status(A, strided, exec), detail::matxException);

  // No stages
  EXPECT_THROW(make_solver_status(A, exec, {.stages = 0}), detail::matxException);

  // A zero-size batch dimension, for both overloads and both executor kinds.
  // Zero-size matrix dimensions do not affect the status's shape and are allowed.
  {
    const auto zero_batch = make_tensor<T>({0, 4, 4});
    const auto msg = ExceptionText([&] { make_solver_status(zero_batch, exec); });
    EXPECT_NE(msg.find("batch sizes must be positive"), std::string::npos) << msg;
    EXPECT_THROW(make_solver_status(zero_batch, host_exec), detail::matxException);
    EXPECT_THROW(make_solver_status(make_tensor<T>({2, 0, 4, 4}), exec), detail::matxException);
    auto zinfo = make_tensor<int>({2, 0}, MATX_DEVICE_MEMORY);
    EXPECT_THROW(make_solver_status(zero_batch, zinfo, exec), detail::matxException);

    EXPECT_NO_THROW(make_solver_status(make_tensor<T>({3, 0, 4}), exec));
    EXPECT_NO_THROW(make_solver_status(make_tensor<T>({3, 4, 0}), exec));
    EXPECT_NO_THROW(make_solver_status(make_tensor<T>({0, 0}), exec));
    EXPECT_NO_THROW(make_solver_status(make_tensor<T>({3, 0, 4}), make_tensor<int>({2, 3}, MATX_DEVICE_MEMORY), exec));
  }

  // A CUDA executor needs device or host-pinned memory
  auto pageable = make_tensor<int>({2, 3}, MATX_HOST_MALLOC_MEMORY);
  EXPECT_THROW(make_solver_status(A, pageable, exec), detail::matxException);
  auto managed = make_tensor<int>({2, 3}, MATX_MANAGED_MEMORY);
  EXPECT_THROW(make_solver_status(A, managed, exec), detail::matxException);
  // A host executor accepts both, but not device memory, which host solvers cannot write
  EXPECT_NO_THROW(make_solver_status(A, pageable, host_exec));
  EXPECT_NO_THROW(make_solver_status(A, managed, host_exec));
  auto device_info = make_tensor<int>({2, 3}, MATX_DEVICE_MEMORY);
  const auto device_msg = ExceptionText([&] { (void)make_solver_status(A, device_info, host_exec); });
  EXPECT_NE(device_msg.find("host-accessible memory"), std::string::npos) << device_msg;

  // Device memory must be on the device of the executor's stream
  int device = 0;
  int device_count = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceCount(&device_count), cudaSuccess);
  if (device_count > 1) {
    ASSERT_EQ(cudaSetDevice((device + 1) % device_count), cudaSuccess);
    const auto other_msg = ExceptionText([&] { (void)make_solver_status(A, device_info, cudaExecutor{}); });
    EXPECT_EQ(cudaSetDevice(device), cudaSuccess);
    EXPECT_NE(other_msg.find("is on device"), std::string::npos) << other_msg;
  }
  exec.sync();
  MATX_EXIT_HANDLER();
}

// A solver call whose status does not fit it throws before queuing work, leaves
// the status as it was, and leaks none of the solver's temporaries
TEST(SolverStatusTest, RejectsMismatchedSolverCalls)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  SingleThreadedHostExecutor host_exec{};
  auto A = make_tensor<T>({3, 8, 8});
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto out = make_tensor<T>(A.Shape());
  auto tau = make_tensor<T>({3, 8});
  auto q = make_tensor<T>(A.Shape());
  auto r = make_tensor<T>(A.Shape());

  auto wrong_batch = make_solver_status(make_tensor<T>({2, 8, 8}), exec);
  auto one_stage = make_solver_status(A, exec, {.stages = 1});
  auto host_status = make_solver_status(A, host_exec);
  // A status serves one solver: its first call fixes the stages it holds
  auto used_by_solver = make_solver_status(A, exec);
  (mtie(out, tau) = qr_solver(A, used_by_solver)).run(exec);
  exec.sync();

  size_t before = 0, after = 0, total = 0, max = 0;
  matxGetMemoryStats(&before, &total, &max);
  EXPECT_THROW((mtie(out, tau) = qr_solver(A, wrong_batch)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(q, r) = qr_econ(A, wrong_batch)).run(exec), detail::matxException);
  const auto stage_msg = ExceptionText([&] { (mtie(q, r) = qr_econ(A, one_stage)).run(exec); });
  EXPECT_NE(stage_msg.find("this solver needs 2"), std::string::npos) << stage_msg;
  EXPECT_THROW((mtie(out, tau) = qr_solver(A, host_status)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(q, r) = qr_econ(A, host_status)).run(exec), detail::matxException);
  const auto solver_msg = ExceptionText([&] { (mtie(q, r) = qr_econ(A, used_by_solver)).run(exec); });
  EXPECT_NE(solver_msg.find("use a separate status for each solver"), std::string::npos) << solver_msg;
  EXPECT_NE(solver_msg.find("writes 1 stage, so it cannot be used by one that writes 2"), std::string::npos) << solver_msg;
  exec.sync();
  matxGetMemoryStats(&after, &total, &max);
  EXPECT_EQ(after, before);

  // The rejected statuses still work with a fitting call
  EXPECT_TRUE(used_by_solver.ok());
  (mtie(out, tau) = qr_solver(A, one_stage)).run(exec);
  EXPECT_TRUE(one_stage.ok(exec));
  MATX_EXIT_HANDLER();
}

TEST(SolverStatusTest, DeviceMemoryStatus)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  auto A = make_tensor<T>({3, 6, 4});
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto q = make_tensor<T>({3, 6, 4});
  auto r = make_tensor<T>({3, 4, 4});

  // example-begin solver-status-test-3
  // A status tensor in device memory is read only on the device
  auto info = make_tensor<int>({2, 3}, MATX_ASYNC_DEVICE_MEMORY, exec.getStream());
  auto st = make_solver_status(A, info, exec);
  (mtie(q, r) = qr_econ(A, st)).run(exec);
  auto nfail = make_tensor<int>({});
  (nfail = sum(abs(st.infos()))).run(exec);   // ordered after the status writes on the same stream
  // example-end solver-status-test-3
  exec.sync();
  EXPECT_EQ(nfail(), 0);
  const auto msg = ExceptionText([&] { (void)st.ok(); });
  EXPECT_NE(msg.find("in device memory"), std::string::npos) << msg;
  EXPECT_THROW(st.check(), detail::matxException);
  EXPECT_THROW(st.ok(exec), detail::matxException);
  EXPECT_EQ(st.infos().Data(), info.Data());
  MATX_EXIT_HANDLER();
}

TEST(SolverStatusTest, EmptyAndCopiedStatuses)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  auto A = make_tensor<T>({3, 6, 4});
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto out = make_tensor<T>(A.Shape());
  auto tau = make_tensor<T>({3, 4});

  // A default-constructed status is empty: its members throw, and solvers treat it as no status
  SolverStatus empty;
  EXPECT_EQ(empty.state(), nullptr);
  EXPECT_THROW(empty.ok(), detail::matxException);
  EXPECT_THROW(empty.ok(exec), detail::matxException);
  EXPECT_THROW(empty.check(), detail::matxException);
  EXPECT_THROW(empty.check(exec), detail::matxException);
  EXPECT_THROW(empty.infos(), detail::matxException);
  auto ref = make_tensor<T>(A.Shape());
  auto tau_ref = make_tensor<T>({3, 4});
  (mtie(ref, tau_ref) = qr_solver(A)).run(exec);
  EXPECT_NO_THROW((mtie(out, tau) = qr_solver(A, empty)).run(exec));
  EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);

  // Copies refer to the same status: a call through one is read through the other
  auto st = make_solver_status(A, exec, {.stages = 1});
  const SolverStatus copy = st;
  (mtie(out, tau) = qr_solver(A, copy)).run(exec);
  EXPECT_EQ(st.infos().Data(), copy.infos().Data());
  exec.sync();
  st.infos()(0, 1) = -2;
  EXPECT_FALSE(copy.ok());
  MATX_EXIT_HANDLER();
}

TEST(SolverStatusTest, CreationDefaults)
{
  MATX_ENTER_HANDLER();
  cudaExecutor exec{};
  auto A = make_tensor<float>({3, 6, 4});

  // A CUDA executor's status is host-pinned memory with two stages, filled
  // with a value that reports that no result was written
  auto st_default = make_solver_status(A, exec);
  EXPECT_EQ(st_default.infos().Size(0), 2);
  EXPECT_EQ(st_default.infos().Size(1), 3);
  EXPECT_EQ(MemoryType(st_default.infos().Data()), cudaMemoryTypeHost);
  EXPECT_EQ(std::count(st_default.infos().Data(), st_default.infos().Data() + 6, detail::solver_info_not_written), 6);
  // A status no solver has written has no values to report
  EXPECT_TRUE(st_default.ok());
  EXPECT_NO_THROW(st_default.check());

  EXPECT_EQ(make_solver_status(A, exec, {.stages = 1}).infos().Size(0), 1);
  EXPECT_EQ(make_solver_status(A, exec, {.stages = 3}).infos().Size(0), 3);

  // A user tensor sets the stage count
  auto info = make_tensor<int>({1, 3}, MATX_HOST_MEMORY);
  EXPECT_EQ(make_solver_status(A, info, exec).infos().Size(0), 1);

  // A host executor's status is ordinary host memory, filled the same way
  SingleThreadedHostExecutor host_exec{};
  auto st_host = make_solver_status(A, host_exec);
  EXPECT_EQ(MemoryType(st_host.infos().Data()), cudaMemoryTypeUnregistered);
  EXPECT_EQ(std::count(st_host.infos().Data(), st_host.infos().Data() + 6, detail::solver_info_not_written), 6);
  EXPECT_TRUE(st_host.ok(host_exec));
  MATX_EXIT_HANDLER();
}

namespace {

// Fill A, of shape [batch..., n, n], with Hermitian positive-definite matrices:
// B * B^H plus n on the diagonal. A must be host-accessible after the sync.
template <typename T, typename ATensor, typename Executor>
void MakeHermitianPD(ATensor &A, Executor &exec)
{
  constexpr int R = ATensor::Rank();
  auto B = make_tensor<T>(A.Shape());
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  (A = matmul(B, conj(transpose_matrix(B)))).run(exec);
  exec.sync();
  const index_t n = A.Size(R - 1);
  const index_t batches = A.TotalSize() / (n * n);
  for (index_t b = 0; b < batches; b++) {
    for (index_t i = 0; i < n; i++) {
      A.Data()[b * n * n + i * n + i] += static_cast<typename inner_op_type_t<T>::type>(n);
    }
  }
}

// Set matrix `b` of a [batch, n, n] host-accessible tensor to zero, which every
// LU-based solver reports as singular
template <typename ATensor>
void ZeroMatrix(ATensor &A, index_t b)
{
  const index_t n = A.Size(ATensor::Rank() - 1);
  std::fill_n(A.Data() + b * n * n, n * n, typename ATensor::value_type{0});
}

} // namespace

// Each solver's results with a status match those without one, for the mtie
// and projection forms, and a status reports no failures for well-posed inputs
TYPED_TEST(SolverStatusTypedTest, SolversMatchNoStatus)
{
  MATX_ENTER_HANDLER();
  using T = cuda::std::tuple_element_t<0, TypeParam>;
  using S = typename inner_op_type_t<T>::type;
  cudaExecutor exec{};
  constexpr index_t batch = 3;
  constexpr index_t n = 6;

  auto H = make_tensor<T>({batch, n, n});
  MakeHermitianPD<T>(H, exec);
  auto A = make_tensor<T>({batch, n, n});
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto B = make_tensor<T>({batch, n, 2});
  (B = random<T>(B.Shape(), NORMAL)).run(exec);

  {
    SCOPED_TRACE("qr_solver");
    auto Aq = make_tensor<T>({2, 3, 6, 4});
    (Aq = random<T>(Aq.Shape(), NORMAL)).run(exec);
    auto out_ref = make_tensor<T>(Aq.Shape());
    auto tau_ref = make_tensor<T>({2, 3, 4});
    (mtie(out_ref, tau_ref) = qr_solver(Aq)).run(exec);

    // example-begin solver-status-test-1
    auto st = make_solver_status(Aq, exec);   // [2, 2, 3] int tensor in host-pinned memory
    auto out = make_tensor<T>(Aq.Shape());
    auto tau = make_tensor<T>({2, 3, 4});
    (mtie(out, tau) = qr_solver(Aq, st)).run(exec);   // does not synchronize
    // ... other work on the stream ...
    EXPECT_TRUE(st.ok(exec));      // synchronizes exec's stream, then reads
    EXPECT_NO_THROW(st.check());   // reads without waiting; throws matxSolverError naming failing matrices
    // example-end solver-status-test-1
    EXPECT_EQ(Vals(st, 1), std::vector<int>(6, 0));
    EXPECT_EQ(MaxAbsDiff<T>(out, out_ref, exec), 0);
    EXPECT_EQ(MaxAbsDiff<T>(tau, tau_ref, exec), 0);
    (out = qr_solver(Aq, st).Out).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(out, out_ref, exec), 0);
  }
  {
    SCOPED_TRACE("qr_econ");
    auto Aq = make_tensor<T>({4, 8, 5});
    (Aq = random<T>(Aq.Shape(), NORMAL)).run(exec);
    auto q_ref = make_tensor<T>({4, 8, 5});
    auto r_ref = make_tensor<T>({4, 5, 5});
    (mtie(q_ref, r_ref) = qr_econ(Aq)).run(exec);

    // example-begin solver-status-test-2
    // [stages, batch] status tensor in host-pinned memory, written directly by the device
    auto info = make_tensor<int>({2, 4}, MATX_HOST_MEMORY);
    auto st = make_solver_status(Aq, info, exec);
    auto q = make_tensor<T>({4, 8, 5});
    auto r = make_tensor<T>({4, 5, 5});
    (mtie(q, r) = qr_econ(Aq, st)).run(exec);
    exec.sync();
    // info(0, i) holds matrix i's geqrf info and info(1, i) its orgqr info
    EXPECT_TRUE(st.ok());
    // example-end solver-status-test-2
    EXPECT_EQ(Vals(st, 2), std::vector<int>(8, 0));
    EXPECT_EQ(MaxAbsDiff<T>(q, q_ref, exec), 0);
    EXPECT_EQ(MaxAbsDiff<T>(r, r_ref, exec), 0);
    (r = qr_econ(Aq, st).R).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(r, r_ref, exec), 0);
  }
  {
    SCOPED_TRACE("chol");
    auto ref = make_tensor<T>(H.Shape());
    (ref = chol(H)).run(exec);
    // example-begin solver-status-test-5
    auto st = make_solver_status(H, exec, {.stages = 1});
    auto out = make_tensor<T>(H.Shape());
    (out = chol(H, st)).run(exec);                     // potrf info into stage 0
    EXPECT_TRUE(st.ok(exec));
    // example-end solver-status-test-5
    EXPECT_EQ(Vals(st, 1), std::vector<int>(batch, 0));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
    // The status overload keeps the fill mode argument
    (ref = chol(H, SolverFillMode::LOWER)).run(exec);
    (out = chol(H, st, SolverFillMode::LOWER)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  {
    SCOPED_TRACE("lu");
    auto ref = make_tensor<T>(A.Shape());
    auto ref_piv = make_tensor<int64_t>({batch, n});
    (mtie(ref, ref_piv) = lu(A)).run(exec);
    auto st = make_solver_status(A, exec, {.stages = 1});
    auto out = make_tensor<T>(A.Shape());
    auto piv = make_tensor<int64_t>({batch, n});
    (mtie(out, piv) = lu(A, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
    // The projection API takes a status too
    (out = lu(A, st).LU).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  {
    SCOPED_TRACE("eig");
    auto ref_vec = make_tensor<T>(H.Shape());
    auto ref_val = make_tensor<S>({batch, n});
    (mtie(ref_vec, ref_val) = eig(H)).run(exec);
    auto st = make_solver_status(H, exec, {.stages = 1});
    auto vec = make_tensor<T>(H.Shape());
    auto val = make_tensor<S>({batch, n});
    (mtie(vec, val) = eig(H, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<S>(val, ref_val, exec), 0);
    (val = eig(H, st, EigenMode::NO_VECTOR).Values).run(exec);
    EXPECT_TRUE(st.ok(exec));
  }
  {
    SCOPED_TRACE("svd");
    auto st = make_solver_status(A, exec, {.stages = 1});
    auto u = make_tensor<T>(A.Shape());
    auto sv = make_tensor<S>({batch, n});
    auto vt = make_tensor<T>(A.Shape());
    auto ref_s = make_tensor<S>({batch, n});
    (mtie(u, ref_s, vt) = svd(A)).run(exec);
    (mtie(u, sv, vt) = svd(A, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(MaxAbsDiff<S>(sv, ref_s, exec), 0);
    // The values-only projection also writes the status
    (sv = svd(A, st, SVDMode::NONE).S).run(exec);
    EXPECT_TRUE(st.ok(exec));
  }
  {
    SCOPED_TRACE("solve");
    auto ref = make_tensor<T>(B.Shape());
    (ref = solve(A, B)).run(exec);
    // example-begin solver-status-test-6
    auto st = make_solver_status(A, exec);             // two stages: getrf, then getrs
    auto X = make_tensor<T>(B.Shape());
    (X = solve(A, B, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    // example-end solver-status-test-6
    EXPECT_EQ(Vals(st, 2), std::vector<int>(2 * batch, 0));
    EXPECT_EQ(MaxAbsDiff<T>(X, ref, exec), 0);
  }
  {
    SCOPED_TRACE("inv");
    auto ref = make_tensor<T>(A.Shape());
    (ref = inv(A)).run(exec);
    auto st = make_solver_status(A, exec);
    auto out = make_tensor<T>(A.Shape());
    (out = inv(A, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(Vals(st, 2), std::vector<int>(2 * batch, 0));
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  MATX_EXIT_HANDLER();
}

namespace {

// One status per kind of status tensor for an input shaped like A: MatX-allocated
// (host-pinned), a user host-pinned tensor, and a user device-memory tensor
template <typename AType>
std::vector<std::pair<std::string, SolverStatus>> StatusKinds(const AType &A, cudaExecutor &exec, int stages)
{
  constexpr int R = AType::Rank();
  cuda::std::array<index_t, R - 1> shape{};
  shape[0] = stages;
  for (int i = 0; i < R - 2; i++) {
    shape[i + 1] = A.Size(i);
  }
  auto pinned = make_tensor<int>(shape, MATX_HOST_MEMORY);
  auto device = make_tensor<int>(shape, MATX_DEVICE_MEMORY);
  return {{"MatX-allocated", make_solver_status(A, exec, {.stages = stages})},
          {"user pinned", make_solver_status(A, pinned, exec)},
          {"user device", make_solver_status(A, device, exec)}};
}

// The first `stages` stages of a status in any memory, copied to the host
std::vector<int> StatusValues(const SolverStatus &st, index_t stages, cudaExecutor &exec)
{
  const auto t = st.infos();
  auto h = make_tensor<int>(t.Shape(), MATX_HOST_MEMORY);
  matx::copy(h, t, exec);
  exec.sync();
  return std::vector<int>(h.Data(), h.Data() + stages * t.Size(1));
}

// Every solver that takes a status, at the rank of `shape`, with each kind of
// status tensor: results match the call without a status, and the status
// reports no failures
template <int RANK>
void CheckSolversWithStatusKinds(const cuda::std::array<index_t, RANK> &shape)
{
  using T = float;
  cudaExecutor exec{};
  cuda::std::array<index_t, RANK - 1> vshape{};
  for (int i = 0; i < RANK - 1; i++) {
    vshape[i] = shape[i];
  }
  index_t batches = 1;
  for (int i = 0; i < RANK - 2; i++) {
    batches *= shape[i];
  }
  const std::vector<int> ok1(static_cast<size_t>(batches), 0);
  const std::vector<int> ok2(static_cast<size_t>(2 * batches), 0);

  auto H = make_tensor<T>(shape);
  MakeHermitianPD<T>(H, exec);
  auto A = make_tensor<T>(shape);
  auto B = make_tensor<T>(shape);
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  auto ref = make_tensor<T>(shape);
  auto out = make_tensor<T>(shape);
  auto ref2 = make_tensor<T>(shape);
  auto out2 = make_tensor<T>(shape);
  auto ref_v = make_tensor<T>(vshape);
  auto v = make_tensor<T>(vshape);
  auto piv = make_tensor<int64_t>(vshape);

  for (auto &[kind, st] : StatusKinds(A, exec, 1)) {
    SCOPED_TRACE("qr_solver, " + kind);
    (mtie(ref, ref_v) = qr_solver(A)).run(exec);
    (mtie(out, v) = qr_solver(A, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 1, exec), ok1);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(A, exec, 2)) {
    SCOPED_TRACE("qr_econ, " + kind);
    (mtie(ref, ref2) = qr_econ(A)).run(exec);
    (mtie(out, out2) = qr_econ(A, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 2, exec), ok2);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(H, exec, 1)) {
    SCOPED_TRACE("chol, " + kind);
    (ref = chol(H)).run(exec);
    (out = chol(H, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 1, exec), ok1);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(A, exec, 1)) {
    SCOPED_TRACE("lu, " + kind);
    (mtie(ref, piv) = lu(A)).run(exec);
    (mtie(out, piv) = lu(A, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 1, exec), ok1);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(H, exec, 1)) {
    SCOPED_TRACE("eig, " + kind);
    (mtie(ref, ref_v) = eig(H)).run(exec);
    (mtie(out, v) = eig(H, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 1, exec), ok1);
    EXPECT_EQ(MaxAbsDiff<T>(v, ref_v, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(A, exec, 1)) {
    SCOPED_TRACE("svd, " + kind);
    (mtie(ref, ref_v, ref2) = svd(A)).run(exec);
    (mtie(out, v, out2) = svd(A, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 1, exec), ok1);
    EXPECT_EQ(MaxAbsDiff<T>(v, ref_v, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(A, exec, 2)) {
    SCOPED_TRACE("solve, " + kind);
    (ref = solve(A, B)).run(exec);
    (out = solve(A, B, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 2, exec), ok2);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
  for (auto &[kind, st] : StatusKinds(A, exec, 2)) {
    SCOPED_TRACE("inv, " + kind);
    (ref = inv(A)).run(exec);
    (out = inv(A, st)).run(exec);
    EXPECT_EQ(StatusValues(st, 2, exec), ok2);
    EXPECT_EQ(MaxAbsDiff<T>(out, ref, exec), 0);
  }
}

} // namespace

// Unbatched (rank 2) and two batch dimensions (rank 4), with each kind of
// status tensor. Rank 2 also takes the single-matrix paths, such as cuSolver
// getrf/getrs for inv and solve.
TEST(SolverStatusTest, SolversAtOtherRanksWithEachStatusKind)
{
  MATX_ENTER_HANDLER();
  {
    SCOPED_TRACE("rank 2");
    CheckSolversWithStatusKinds<2>({6, 6});
  }
  {
    SCOPED_TRACE("rank 4");
    CheckSolversWithStatusKinds<4>({2, 3, 6, 6});
  }
  MATX_EXIT_HANDLER();
}

// Statuses report the failures each solver's library reports, per matrix
TEST(SolverStatusTest, OtherSolversReportFailures)
{
  MATX_ENTER_HANDLER();
  using T = double;
  cudaExecutor exec{};
  constexpr index_t batch = 3;
  constexpr index_t n = 5;

  {
    SCOPED_TRACE("chol, not positive definite");
    auto H = make_tensor<T>({batch, n, n});
    MakeHermitianPD<T>(H, exec);
    H(1, 2, 2) = -100.0;
    auto st = make_solver_status(H, exec, {.stages = 1});
    auto out = make_tensor<T>(H.Shape());
    (out = chol(H, st)).run(exec);
    EXPECT_FALSE(st.ok(exec));
    const auto v = Vals(st, 1);
    EXPECT_EQ(v[0], 0);
    EXPECT_GT(v[1], 0);   // the order of the leading minor that is not positive definite
    EXPECT_EQ(v[2], 0);
    const auto msg = ExceptionText([&] { st.check(); });
    EXPECT_NE(msg.find("stage 0, batch 1: info = "), std::string::npos) << msg;
  }
  {
    SCOPED_TRACE("lu, singular");
    auto A = make_tensor<T>({batch, n, n});
    (A = random<T>(A.Shape(), NORMAL)).run(exec);
    exec.sync();
    ZeroMatrix(A, 2);
    auto st = make_solver_status(A, exec, {.stages = 1});
    auto out = make_tensor<T>(A.Shape());
    auto piv = make_tensor<int64_t>({batch, n});
    (mtie(out, piv) = lu(A, st)).run(exec);
    EXPECT_FALSE(st.ok(exec));
    const auto v = Vals(st, 1);
    EXPECT_EQ(v[0], 0);
    EXPECT_EQ(v[1], 0);
    EXPECT_GT(v[2], 0);   // U(i,i) is exactly zero
  }
  for (const index_t solve_batch : {index_t{1}, batch}) {
    // One matrix uses the cuSolver getrf/getrs loop, a batch the cuBLAS batched path
    SCOPED_TRACE(solve_batch == 1 ? "solve, cuSolver loop" : "solve, cuBLAS batched");
    auto A = make_tensor<T>({solve_batch, n, n});
    (A = random<T>(A.Shape(), NORMAL)).run(exec);
    auto B = make_tensor<T>({solve_batch, n});
    (B = random<T>(B.Shape(), NORMAL)).run(exec);
    exec.sync();
    ZeroMatrix(A, solve_batch - 1);
    auto st = make_solver_status(A, exec);
    auto X = make_tensor<T>(B.Shape());
    EXPECT_NO_THROW((X = solve(A, B, st)).run(exec));
    EXPECT_FALSE(st.ok(exec));
    const auto v = Vals(st, 2);
    for (index_t i = 0; i < solve_batch; i++) {
      if (i == solve_batch - 1) {
        EXPECT_GT(v[static_cast<size_t>(i)], 0) << "stage 0, matrix " << i;
      }
      else {
        EXPECT_EQ(v[static_cast<size_t>(i)], 0) << "stage 0, matrix " << i;
      }
      EXPECT_EQ(v[static_cast<size_t>(solve_batch + i)], 0) << "stage 1, matrix " << i;
    }
  }
  struct InvCase { index_t batch; index_t n; const char *backend; };
  for (const auto &c : {InvCase{1, 4, "cuSolver getrf/getrs"}, InvCase{batch, 4, "cuBLAS matinvBatched"},
                        InvCase{batch, 40, "cuBLAS getrf/getriBatched"}}) {
    SCOPED_TRACE(std::string("inv, ") + c.backend);
    auto A = make_tensor<T>({c.batch, c.n, c.n});
    (A = random<T>(A.Shape(), NORMAL)).run(exec);
    exec.sync();
    ZeroMatrix(A, c.batch - 1);
    // Without a status, a singular matrix throws
    auto out = make_tensor<T>(A.Shape());
    EXPECT_THROW((out = inv(A)).run(exec), detail::matxException);
    // With one, the call does not throw, and stage 0 reports the singular matrix
    auto st = make_solver_status(A, exec);
    EXPECT_NO_THROW((out = inv(A, st)).run(exec));
    EXPECT_FALSE(st.ok(exec));
    const auto v = Vals(st, 2);
    for (index_t i = 0; i < c.batch; i++) {
      if (i == c.batch - 1) {
        EXPECT_GT(v[static_cast<size_t>(i)], 0) << "stage 0, matrix " << i;
      }
      else {
        EXPECT_EQ(v[static_cast<size_t>(i)], 0) << "stage 0, matrix " << i;
        EXPECT_EQ(v[static_cast<size_t>(c.batch + i)], 0) << "stage 1, matrix " << i;
      }
    }
  }
  exec.sync();
  MATX_EXIT_HANDLER();
}

// The stage-count rule applies across solvers: one-stage solvers can share a
// status, a two-stage solver rejects a one-stage status, and a first call that
// throws does not fix the count
TEST(SolverStatusTest, StageCountRules)
{
  MATX_ENTER_HANDLER();
  using T = float;
  cudaExecutor exec{};
  auto H = make_tensor<T>({2, 4, 4});
  MakeHermitianPD<T>(H, exec);
  auto B = make_tensor<T>({2, 4});
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  auto out = make_tensor<T>(H.Shape());
  auto piv = make_tensor<int64_t>({2, 4});
  auto X = make_tensor<T>(B.Shape());

  auto one = make_solver_status(H, exec, {.stages = 1});
  (out = chol(H, one)).run(exec);
  EXPECT_NO_THROW((mtie(out, piv) = lu(H, one)).run(exec));
  EXPECT_TRUE(one.ok(exec));
  const auto msg = ExceptionText([&] { (X = solve(H, B, one)).run(exec); });
  EXPECT_NE(msg.find("this solver needs 2"), std::string::npos) << msg;
  EXPECT_THROW((out = inv(H, one)).run(exec), detail::matxException);

  // A two-stage status first used by solve() cannot then serve a one-stage solver
  auto two = make_solver_status(H, exec);
  (X = solve(H, B, two)).run(exec);
  EXPECT_THROW((out = chol(H, two)).run(exec), detail::matxException);
  EXPECT_NO_THROW((out = inv(H, two)).run(exec));
  EXPECT_TRUE(two.ok(exec));

  // A call that throws after validating its status leaves the count as it was.
  // This drives the solver-side hook directly, as no solver here fails reliably
  // after validation.
  auto fresh = make_solver_status(H, exec);
  EXPECT_THROW([&] {
    const auto run = detail::BeginStatusRun(detail::ExecutorKind::Cuda, fresh.state(), 1, H);
    MATX_THROW(matxSolverError, "solver failed after validating its status");
  }(), detail::matxException);
  EXPECT_NO_THROW((out = inv(H, fresh)).run(exec));
  EXPECT_TRUE(fresh.ok(exec));
  MATX_EXIT_HANDLER();
}

// A status lets a batched dense solve run without synchronizing, so it can be
// captured in a CUDA graph. The captured graph must not copy from host memory
// that the call freed when it returned, so replays after later direct calls
// still solve the captured matrices.
TEST(SolverStatusTest, BatchedSolveGraphReplay)
{
  MATX_ENTER_HANDLER();
  using T = float;
  constexpr index_t batch = 4;
  constexpr index_t n = 8;
  constexpr index_t nrhs = 3;
  // Without stream-ordered memory pools, the solve's temporary buffers fall back
  // to cudaMalloc and cudaFree instead of becoming graph allocation nodes, so a
  // captured graph cannot keep them for replay
  int device = 0;
  ASSERT_EQ(cudaGetDevice(&device), cudaSuccess);
  if (!detail::GetDeviceAttributes(device).memory_pools_supported) {
    GTEST_SKIP() << "Device " << device << " does not support stream-ordered memory pools, so a solve cannot be captured";
  }
  TestStream stream;
  cudaExecutor exec{stream};
  auto A = make_tensor<T>({batch, n, n});
  MakeHermitianPD<T>(A, exec);
  auto B = make_tensor<T>({batch, n, nrhs});
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  auto X = make_tensor<T>(B.Shape());
  auto ref = make_tensor<T>(B.Shape());
  auto st = make_solver_status(A, exec);
  // Create the cached plan before capturing
  (X = solve(A, B, st)).run(exec);
  exec.sync();

  struct Graph {
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t exec = nullptr;
    ~Graph()
    {
      if (exec != nullptr) {
        cudaGraphExecDestroy(exec);
      }
      if (graph != nullptr) {
        cudaGraphDestroy(graph);
      }
    }
  } g;
  ASSERT_EQ(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), cudaSuccess);
  (X = solve(A, B, st)).run(exec);
  ASSERT_EQ(cudaStreamEndCapture(stream, &g.graph), cudaSuccess);

  size_t count = 0;
  ASSERT_EQ(cudaGraphGetNodes(g.graph, nullptr, &count), cudaSuccess);
  std::vector<cudaGraphNode_t> nodes(count);
  ASSERT_EQ(cudaGraphGetNodes(g.graph, nodes.data(), &count), cudaSuccess);
  for (const auto node : nodes) {
    cudaGraphNodeType type;
    ASSERT_EQ(cudaGraphNodeGetType(node, &type), cudaSuccess);
    if (type == cudaGraphNodeTypeMemcpy) {
      cudaMemcpy3DParms params{};
      ASSERT_EQ(cudaGraphMemcpyNodeGetParams(node, &params), cudaSuccess);
      EXPECT_NE(params.kind, cudaMemcpyHostToDevice) << "The graph copies from host memory";
    }
  }
  ASSERT_EQ(cudaGraphInstantiate(&g.exec, g.graph, 0), cudaSuccess);

  // Direct solves of other matrices before the replay
  auto A2 = make_tensor<T>({batch, n, n});
  MakeHermitianPD<T>(A2, exec);
  auto X2 = make_tensor<T>(B.Shape());
  for (int i = 0; i < 3; i++) {
    (X2 = solve(A2, B, st)).run(exec);
  }

  // The replay solves the captured A for new right-hand sides
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  (ref = solve(A, B)).run(exec);
  ASSERT_EQ(cudaGraphLaunch(g.exec, stream), cudaSuccess);
  exec.sync();
  EXPECT_TRUE(st.ok());
  EXPECT_EQ(MaxAbsDiff<T>(X, ref, exec), 0);
  MATX_EXIT_HANDLER();
}

// Statuses support host executors and cudaExecutor. Creating a status with a
// JIT or distributed executor throws, and reading one with it is a mismatch.
TEST(SolverStatusTest, RejectsJitAndDistributedExecutors)
{
  MATX_ENTER_HANDLER();
  using T = float;
  using detail::ExecutorKind;
  static_assert(detail::executor_kind_v<SingleThreadedHostExecutor> == ExecutorKind::Host);
  static_assert(detail::executor_kind_v<cudaExecutor> == ExecutorKind::Cuda);
  static_assert(detail::executor_kind_v<CUDAJITExecutor> == ExecutorKind::CudaJit);
  static_assert(detail::executor_kind_v<distributedCUDAExecutor> == ExecutorKind::Distributed);

  cudaExecutor exec{};
  CUDAJITExecutor jit_exec{};
  distributedCUDAExecutor dist_exec{std::vector<int>{0}};
  auto A = make_tensor<T>({2, 8, 8});
  auto info = make_tensor<int>({1, 2}, MATX_DEVICE_MEMORY);
  auto status = make_solver_status(A, exec);

  const auto expect_rejected = [](const std::function<void()> &f, const std::string &expected) {
    try {
      f();
      ADD_FAILURE() << "Expected an exception: " << expected;
    }
    catch (const detail::matxException &e) {
      EXPECT_EQ(e.e, matxInvalidExecutor);
      EXPECT_NE(std::string(e.what()).find(expected), std::string::npos) << e.what();
    }
  };
  const auto expect_executor_rejected = [&](const auto &other, const std::string &name) {
    const std::string created = "support host executors and cudaExecutor, not a " + name;
    const std::string used = "created with a cudaExecutor, so it cannot be used with a " + name;
    expect_rejected([&] { (void)make_solver_status(A, other); }, created);
    expect_rejected([&] { (void)make_solver_status(A, info, other); }, created);
    expect_rejected([&] { (void)status.ok(other); }, used);
    expect_rejected([&] { status.check(other); }, used);
  };
  expect_executor_rejected(jit_exec, "CUDAJITExecutor");
  expect_executor_rejected(dist_exec, "distributedCUDAExecutor");
  EXPECT_TRUE(status.ok(exec));
  MATX_EXIT_HANDLER();
}

#if MATX_EN_CPU_SOLVER
// Host solvers write a host status in place, in any host-accessible memory
TEST(SolverStatusTest, HostQrSolver)
{
  MATX_ENTER_HANDLER();
  using T = double;
  SingleThreadedHostExecutor exec{};
  auto A = make_tensor<T>({3, 5, 4}, MATX_HOST_MALLOC_MEMORY);
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto out_ref = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
  auto tau_ref = make_tensor<T>({3, 4}, MATX_HOST_MALLOC_MEMORY);
  (mtie(out_ref, tau_ref) = qr_solver(A)).run(exec);
  auto out = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
  auto tau = make_tensor<T>({3, 4}, MATX_HOST_MALLOC_MEMORY);

  for (const auto space : {MATX_HOST_MALLOC_MEMORY, MATX_HOST_MEMORY, MATX_MANAGED_MEMORY}) {
    SCOPED_TRACE(static_cast<int>(space));
    auto info = make_tensor<int>({1, 3}, space);
    std::fill_n(info.Data(), 3, 12345);
    auto st = make_solver_status(A, info, exec);
    (mtie(out, tau) = qr_solver(A, st)).run(exec);
    // Host solver calls finish before returning, so no synchronization is needed
    EXPECT_TRUE(st.ok());
    EXPECT_NO_THROW(st.check(exec));
    EXPECT_EQ(std::count(info.Data(), info.Data() + 3, 0), 3);
    for (index_t i = 0; i < A.TotalSize(); i++) {
      EXPECT_EQ(out.Data()[i], out_ref.Data()[i]);
    }
  }

  // A MatX-owned status for the host executor is host memory
  auto st_owned = make_solver_status(A, exec);
  (mtie(out, tau) = qr_solver(A, st_owned)).run(exec);
  EXPECT_TRUE(st_owned.ok());

  // A status created for a CUDA executor is rejected before any work
  cudaExecutor cuda_exec{};
  auto st_dev = make_solver_status(A, cuda_exec);
  const auto msg = ExceptionText([&] { (mtie(out, tau) = qr_solver(A, st_dev)).run(exec); });
  EXPECT_NE(msg.find("created with a cudaExecutor, so it cannot be used with a host executor"), std::string::npos) << msg;
  cuda_exec.sync();
  MATX_EXIT_HANDLER();
}

// The other host solvers write a host status in place, per matrix
TEST(SolverStatusTest, HostOtherSolvers)
{
  MATX_ENTER_HANDLER();
  using T = double;
  SingleThreadedHostExecutor exec{};
  constexpr index_t batch = 3;
  constexpr index_t n = 5;
  auto H = make_tensor<T>({batch, n, n}, MATX_HOST_MALLOC_MEMORY);
  MakeHermitianPD<T>(H, exec);
  auto A = make_tensor<T>({batch, n, n}, MATX_HOST_MALLOC_MEMORY);
  (A = random<T>(A.Shape(), NORMAL)).run(exec);
  auto B = make_tensor<T>({batch, n}, MATX_HOST_MALLOC_MEMORY);
  (B = random<T>(B.Shape(), NORMAL)).run(exec);
  auto out = make_tensor<T>({batch, n, n}, MATX_HOST_MALLOC_MEMORY);

  {
    SCOPED_TRACE("chol");
    auto st = make_solver_status(H, exec, {.stages = 1});
    (out = chol(H, st)).run(exec);
    // Host solver calls finish before returning, so no synchronization is needed
    EXPECT_TRUE(st.ok());
    auto bad = make_tensor<T>(H.Shape(), MATX_HOST_MALLOC_MEMORY);
    (bad = H).run(exec);
    bad(0, 3, 3) = -100.0;
    (out = chol(bad, st)).run(exec);
    const auto v = Vals(st, 1);
    EXPECT_GT(v[0], 0);
    EXPECT_EQ(v[1], 0);
  }
  {
    SCOPED_TRACE("lu");
    auto st = make_solver_status(A, exec, {.stages = 1});
    auto piv = make_tensor<lapack_int_t>({batch, n}, MATX_HOST_MALLOC_MEMORY);
    auto sing = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
    (sing = A).run(exec);
    ZeroMatrix(sing, 1);
    (mtie(out, piv) = lu(sing, st)).run(exec);
    const auto v = Vals(st, 1);
    EXPECT_EQ(v[0], 0);
    EXPECT_GT(v[1], 0);
    EXPECT_EQ(v[2], 0);
  }
  {
    SCOPED_TRACE("eig");
    auto st = make_solver_status(H, exec, {.stages = 1});
    auto val = make_tensor<T>({batch, n}, MATX_HOST_MALLOC_MEMORY);
    (mtie(out, val) = eig(H, st)).run(exec);
    EXPECT_TRUE(st.ok(exec));
    EXPECT_EQ(Vals(st, 1), std::vector<int>(batch, 0));
  }
  {
    SCOPED_TRACE("svd");
    auto st = make_solver_status(A, exec, {.stages = 1});
    auto u = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
    auto sv = make_tensor<T>({batch, n}, MATX_HOST_MALLOC_MEMORY);
    auto vt = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
    (mtie(u, sv, vt) = svd(A, st)).run(exec);
    EXPECT_TRUE(st.ok());
    EXPECT_EQ(Vals(st, 1), std::vector<int>(batch, 0));
  }
  {
    SCOPED_TRACE("solve");
    auto st = make_solver_status(A, exec);
    auto X = make_tensor<T>(B.Shape(), MATX_HOST_MALLOC_MEMORY);
    auto sing = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
    (sing = A).run(exec);
    ZeroMatrix(sing, 2);
    (X = solve(sing, B, st)).run(exec);
    EXPECT_FALSE(st.ok());
    const auto v = Vals(st, 2);
    EXPECT_EQ(v, (std::vector<int>{0, 0, v[2], 0, 0, 0}));
    EXPECT_GT(v[2], 0);
  }
  MATX_EXIT_HANDLER();
}
#else
// Without a CPU solver backend, every host solver throws in every build type,
// with or without a status, rather than returning with its outputs unwritten
TEST(SolverStatusTest, HostSolversThrowWithoutCpuSolver)
{
  MATX_ENTER_HANDLER();
  using T = double;
  SingleThreadedHostExecutor exec{};
  auto A = make_tensor<T>({2, 4, 4}, MATX_HOST_MALLOC_MEMORY);
  auto B = make_tensor<T>({2, 4}, MATX_HOST_MALLOC_MEMORY);
  auto out = make_tensor<T>(A.Shape(), MATX_HOST_MALLOC_MEMORY);
  auto val = make_tensor<T>({2, 4}, MATX_HOST_MALLOC_MEMORY);
  auto piv = make_tensor<lapack_int_t>({2, 4}, MATX_HOST_MALLOC_MEMORY);
  auto X = make_tensor<T>(B.Shape(), MATX_HOST_MALLOC_MEMORY);
  auto d = make_tensor<T>({2}, MATX_HOST_MALLOC_MEMORY);
  auto one = make_solver_status(A, exec, {.stages = 1});
  auto two = make_solver_status(A, exec);

  EXPECT_THROW((mtie(out, val) = qr_solver(A, one)).run(exec), detail::matxException);
  EXPECT_THROW((out = chol(A, one)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, piv) = lu(A, one)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, val) = eig(A, one)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, val, out) = svd(A, one)).run(exec), detail::matxException);
  EXPECT_THROW((X = solve(A, B, two)).run(exec), detail::matxException);

  EXPECT_THROW((mtie(out, val) = qr_solver(A)).run(exec), detail::matxException);
  EXPECT_THROW((out = chol(A)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, piv) = lu(A)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, val) = eig(A)).run(exec), detail::matxException);
  EXPECT_THROW((mtie(out, val, out) = svd(A)).run(exec), detail::matxException);
  EXPECT_THROW((X = solve(A, B)).run(exec), detail::matxException);
  EXPECT_THROW((d = det(A)).run(exec), detail::matxException);
  EXPECT_THROW((out = pinv(A)).run(exec), detail::matxException);
  MATX_EXIT_HANDLER();
}
#endif

#if defined(MATX_EN_MATHDX) && defined(MATX_EN_JIT) && defined(__CUDACC__)
// A status disables JIT fusion, so a JIT expression with one throws. That holds
// both on a cold JIT cache and after the same expression without a status was
// fused and its launch cached, since a cached launch skips the support check.
TEST(SolverStatusTest, StatusRejectsJit)
{
  MATX_ENTER_HANDLER();
  using T = float;
  using detail::OperatorCapability;
  cudaExecutor exec{};
  CUDAJITExecutor jit_exec{};
  auto H = make_tensor<T>({2, 8, 8});
  MakeHermitianPD<T>(H, exec);
  auto out = make_tensor<T>(H.Shape());
  auto q = make_tensor<T>(H.Shape());
  auto r = make_tensor<T>(H.Shape());
  auto val = make_tensor<T>({2, 8});
  auto one = make_solver_status(H, exec, {.stages = 1});
  auto two = make_solver_status(H, exec);
  auto expect_rejected = [&] {
    EXPECT_THROW((out = qr_solver(H, one).Out).run(jit_exec), detail::matxException);
    EXPECT_THROW((q = qr_econ(H, two).Q).run(jit_exec), detail::matxException);
    EXPECT_THROW((r = qr_econ(H, two).R).run(jit_exec), detail::matxException);
    EXPECT_THROW((out = chol(H, one)).run(jit_exec), detail::matxException);
    EXPECT_THROW((out = lu(H, one).LU).run(jit_exec), detail::matxException);
    EXPECT_THROW((val = eig(H, one).Values).run(jit_exec), detail::matxException);
    EXPECT_THROW((out = inv(H, two)).run(jit_exec), detail::matxException);
  };
  expect_rejected();

  // Fuse each expression without a status where cuSolverDx supports it, which
  // caches its launch
  int fused = 0;
  auto fuse = [&](auto &lhs, const auto &rhs) {
    if (detail::get_operator_capability<OperatorCapability::SUPPORTS_JIT>(rhs)) {
      (lhs = rhs).run(jit_exec);
      fused++;
    }
  };
  fuse(out, qr_solver(H).Out);
  fuse(q, qr_econ(H).Q);
  fuse(r, qr_econ(H).R);
  fuse(out, chol(H));
  fuse(out, lu(H).LU);
  fuse(val, eig(H).Values);
  fuse(out, inv(H));
  exec.sync();
  if (fused == 0) {
    GTEST_SKIP() << "cuSolverDx does not support these solvers on this device";
  }
  expect_rejected();
  MATX_EXIT_HANDLER();
}
#endif
