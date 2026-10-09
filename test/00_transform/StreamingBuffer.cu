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

// Tests for the streaming objects' two-segment input
// (matx/streaming/stream_detail.h): which reader VisitStreamBuffer passes to
// the transform, that both readers return [head | tail], the head_offset
// overload, and SplitUnitStride1DOp's ALIASED_MEMORY
// capability. The streaming transform tests match either reader, so they
// cannot tell whether the two-segment path was taken.

#include "matx.h"
#include "matx/streaming/stream_detail.h"

#include "gtest/gtest.h"

#include <type_traits>
#include <vector>

using namespace matx;

namespace {

template <typename Buf>
constexpr bool is_split_v = detail::is_split_unit_stride_input_v<cuda::std::remove_cvref_t<Buf>>;

// Fills head with 0, 1, ... and tail with 100, 101, ...
template <typename HeadT, typename TailT>
void fill(HeadT &head, TailT &tail)
{
  for (index_t i = 0; i < head.Size(0); i++) head(i) = static_cast<float>(i);
  for (index_t i = 0; i < tail.Size(0); i++) tail(i) = static_cast<float>(100 + i);
}

// Checks that buf reads as [head[head_offset:] | tail].
template <typename Buf, typename HeadT, typename TailT>
void expect_buffer(const Buf &buf, const HeadT &head, index_t head_offset, const TailT &tail)
{
  const index_t hn = head.Size(0) - head_offset;
  ASSERT_EQ(buf.Size(0), hn + tail.Size(0));
  for (index_t i = 0; i < hn; i++) {
    EXPECT_EQ(buf(i), head(head_offset + i)) << "i=" << i;
  }
  for (index_t i = 0; i < tail.Size(0); i++) {
    EXPECT_EQ(buf(hn + i), tail(i)) << "i=" << hn + i;
  }
}

} // namespace

TEST(StreamingBuffer, UnitStrideCudaUsesSplitReader)
{
  auto head = make_tensor<float>({5});
  auto tail = make_tensor<float>({7});
  fill(head, tail);
  bool called = false;
  detail::VisitStreamBuffer<cudaExecutor>(head, tail, [&](const auto &buf) {
    called = true;
    EXPECT_TRUE(is_split_v<decltype(buf)>);
    expect_buffer(buf, head, 0, tail);
  });
  EXPECT_TRUE(called);

  // A unit-stride slice is still a unit-stride tensor view.
  auto base = make_tensor<float>({20});
  for (index_t i = 0; i < 20; i++) base(i) = static_cast<float>(200 + i);
  auto tail_slice = slice(base, {3}, {10});
  detail::VisitStreamBuffer<cudaExecutor>(head, tail_slice, [&](const auto &buf) {
    EXPECT_TRUE(is_split_v<decltype(buf)>);
    expect_buffer(buf, head, 0, tail_slice);
  });
}

TEST(StreamingBuffer, StridedSegmentUsesConcat)
{
  auto head = make_tensor<float>({5});
  auto base = make_tensor<float>({14});
  for (index_t i = 0; i < 14; i++) base(i) = static_cast<float>(100 + i);
  auto tail = slice(base, {0}, {14}, {2});
  for (index_t i = 0; i < 5; i++) head(i) = static_cast<float>(i);
  detail::VisitStreamBuffer<cudaExecutor>(head, tail, [&](const auto &buf) {
    EXPECT_FALSE(is_split_v<decltype(buf)>);
    expect_buffer(buf, head, 0, tail);
  });
}

TEST(StreamingBuffer, HostExecutorUsesConcat)
{
  auto head = make_tensor<float>({5});
  auto tail = make_tensor<float>({7});
  fill(head, tail);
  detail::VisitStreamBuffer<SingleThreadedHostExecutor>(head, tail, [&](const auto &buf) {
    EXPECT_FALSE(is_split_v<decltype(buf)>);
    expect_buffer(buf, head, 0, tail);
  });
}

TEST(StreamingBuffer, HeadOffset)
{
  auto head = make_tensor<float>({5});
  auto tail = make_tensor<float>({7});
  fill(head, tail);
  auto base = make_tensor<float>({14});
  for (index_t i = 0; i < 14; i++) base(i) = static_cast<float>(100 + i);
  auto strided_tail = slice(base, {0}, {14}, {2});

  for (index_t off : {index_t(0), index_t(2), index_t(5)}) {
    detail::VisitStreamBuffer<cudaExecutor>(head, off, tail, [&](const auto &buf) {
      EXPECT_TRUE(is_split_v<decltype(buf)>);
      expect_buffer(buf, head, off, tail);
    });
    detail::VisitStreamBuffer<cudaExecutor>(head, off, strided_tail, [&](const auto &buf) {
      EXPECT_FALSE(is_split_v<decltype(buf)>);
      expect_buffer(buf, head, off, strided_tail);
    });
  }
}

// The multi-element evaluation path is never selected (the op reports
// ELEMENTS_PER_THREAD == ONE) but is compiled for every EPT; it must gather the
// same elements a tensor would rather than return zeros. idx is in units of EPT.
TEST(StreamingBuffer, MultiElementEvaluationGathers)
{
  auto head = make_tensor<float>({5});
  auto tail = make_tensor<float>({7});
  fill(head, tail);
  detail::SplitUnitStride1DOp<float> op(head.Data(), 5, tail.Data(), 7);
  using Cap4 = detail::CapabilityParams<detail::ElementsPerThread::FOUR, false>;
  for (index_t v = 0; v < 3; v++) {
    const auto vec = op.template operator()<Cap4>(v);
    for (int k = 0; k < 4; k++) {
      EXPECT_EQ(vec.data[k], op(4 * v + k)) << "v=" << v << " k=" << k;
    }
  }
}

// The split reader indexes Data() as a dense array, so segments whose storage
// is not one (sparse tensors, planar-complex element types) must use concat.
TEST(StreamingBuffer, NonDenseStorageIsIneligible)
{
  using Dense = tensor_t<float, 1>;
  using Planar = tensor_t<matxFp16ComplexPlanar, 1>;
  using Sparse = experimental::sparse_tensor_t<float, index_t, index_t, experimental::SpVec>;
  EXPECT_TRUE((detail::split_stream_buffer_eligible_v<cudaExecutor, Dense, Dense>));
  EXPECT_FALSE((detail::split_stream_buffer_eligible_v<cudaExecutor, Planar, Planar>));
  EXPECT_FALSE((detail::split_stream_buffer_eligible_v<cudaExecutor, Dense, Sparse>));
  EXPECT_FALSE((detail::split_stream_buffer_eligible_v<SingleThreadedHostExecutor, Dense, Dense>));
}

TEST(StreamingBuffer, SplitReaderReportsAliasedMemory)
{
  auto head = make_tensor<float>({5});
  auto tail = make_tensor<float>({7});
  auto other = make_tensor<float>({4});
  detail::SplitUnitStride1DOp<float> op(head.Data(), 5, tail.Data(), 7);

  auto query = [&](float *start, index_t n) {
    detail::AliasedMemoryQueryInput q{false, false, static_cast<void *>(start),
                                      static_cast<void *>(start + n)};
    return op.template get_capability<detail::OperatorCapability::ALIASED_MEMORY>(q);
  };
  EXPECT_TRUE(query(head.Data() + 2, 1));
  EXPECT_TRUE(query(tail.Data(), 7));
  EXPECT_TRUE(query(tail.Data() + 6, 4));  // partial overlap at the end of the tail
  EXPECT_FALSE(query(other.Data(), 4));
  EXPECT_FALSE(query(head.Data() + 5, 1)); // one past the end of the head

  // An empty head (head_offset == head.Size(0)) aliases nothing.
  detail::SplitUnitStride1DOp<float> tail_only(head.Data() + 5, 0, tail.Data(), 7);
  detail::AliasedMemoryQueryInput q{false, false, static_cast<void *>(head.Data()),
                                    static_cast<void *>(head.Data() + 5)};
  EXPECT_FALSE(tail_only.template get_capability<detail::OperatorCapability::ALIASED_MEMORY>(q));
}
