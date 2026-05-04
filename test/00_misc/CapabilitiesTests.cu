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

#include "gtest/gtest.h"

#include <cuda/std/tuple>

using namespace matx;

namespace {

struct PlainScalarLikeOp {
  int value;
};

struct CapabilityTestOp {
  using emits_jit_str = void;

  bool supports_jit = true;
  bool async_loads = false;
  bool global_kernel = true;
  bool pass_through_threads = false;
  bool unit_stride_last = true;
  bool element_wise = true;
  int max_ept_vec_load = 32;
  int dyn_shm_size = 0;
  std::string jit_type;
  cuda::std::array<detail::ElementsPerThread, 2> ept_range{
      detail::ElementsPerThread::ONE, detail::ElementsPerThread::MAX};
  cuda::std::array<int, 2> groups_range{1, 32};
  cuda::std::array<int, 2> block_range{1, 1024};

  template <detail::OperatorCapability Cap, typename InType>
  __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const
  {
    if constexpr (Cap == detail::OperatorCapability::SUPPORTS_JIT) {
      return supports_jit;
    }
    else if constexpr (Cap == detail::OperatorCapability::ASYNC_LOADS_REQUESTED) {
      return async_loads;
    }
    else if constexpr (Cap == detail::OperatorCapability::GLOBAL_KERNEL) {
      return global_kernel;
    }
    else if constexpr (Cap == detail::OperatorCapability::PASS_THROUGH_THREADS) {
      return pass_through_threads;
    }
    else if constexpr (Cap == detail::OperatorCapability::UNIT_STRIDE_LAST) {
      return unit_stride_last;
    }
    else if constexpr (Cap == detail::OperatorCapability::ELEMENT_WISE) {
      return element_wise;
    }
    else if constexpr (Cap == detail::OperatorCapability::MAX_EPT_VEC_LOAD) {
      return max_ept_vec_load;
    }
    else if constexpr (Cap == detail::OperatorCapability::DYN_SHM_SIZE) {
      return dyn_shm_size;
    }
    else if constexpr (Cap == detail::OperatorCapability::JIT_TYPE_QUERY) {
      return jit_type;
    }
    else if constexpr (Cap == detail::OperatorCapability::ELEMENTS_PER_THREAD) {
      return ept_range;
    }
    else if constexpr (Cap == detail::OperatorCapability::GROUPS_PER_BLOCK) {
      return groups_range;
    }
    else if constexpr (Cap == detail::OperatorCapability::BLOCK_DIM) {
      return block_range;
    }
    else if constexpr (Cap == detail::OperatorCapability::SET_ELEMENTS_PER_THREAD ||
                       Cap == detail::OperatorCapability::SET_GROUPS_PER_BLOCK ||
                       Cap == detail::OperatorCapability::JIT_CLASS_QUERY ||
                       Cap == detail::OperatorCapability::GENERATE_LTOIR) {
      return true;
    }
    else if constexpr (Cap == detail::OperatorCapability::ALIASED_MEMORY) {
      return false;
    }
    else {
      return detail::capability_attributes<Cap>::default_value;
    }
  }
};

template <typename T>
void ExpectRange(const cuda::std::array<T, 2> &actual, T lower, T upper)
{
  EXPECT_EQ(actual[0], lower);
  EXPECT_EQ(actual[1], upper);
}

void ExpectQueryType(detail::OperatorCapability cap, detail::CapabilityQueryType type)
{
  EXPECT_EQ(static_cast<int>(detail::get_query_type(cap)), static_cast<int>(type));
}

} // namespace

TEST(CapabilitiesTests, DefaultCapabilitiesForNonMatXOps)
{
  PlainScalarLikeOp op{42};
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op));
  EXPECT_TRUE(jit_supported(op));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::GLOBAL_KERNEL>(op));
  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::UNIT_STRIDE_LAST>(op));
  EXPECT_EQ(detail::get_operator_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(op), 32);
  detail::EPTQueryInput ept_input{false};
  ExpectRange(detail::get_operator_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(op, ept_input),
              detail::ElementsPerThread::ONE, detail::ElementsPerThread::MAX);
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(op).empty());
}

TEST(CapabilitiesTests, QueryTypeMappingCoversEveryCapability)
{
  ExpectQueryType(detail::OperatorCapability::SUPPORTS_JIT, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::ASYNC_LOADS_REQUESTED, detail::CapabilityQueryType::OR_QUERY);
  ExpectQueryType(detail::OperatorCapability::GLOBAL_KERNEL, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::ELEMENTS_PER_THREAD, detail::CapabilityQueryType::RANGE_QUERY);
  ExpectQueryType(detail::OperatorCapability::SET_ELEMENTS_PER_THREAD, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::SET_GROUPS_PER_BLOCK, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::GROUPS_PER_BLOCK, detail::CapabilityQueryType::RANGE_QUERY);
  ExpectQueryType(detail::OperatorCapability::ALIASED_MEMORY, detail::CapabilityQueryType::OR_QUERY);
  ExpectQueryType(detail::OperatorCapability::MAX_EPT_VEC_LOAD, detail::CapabilityQueryType::MIN_QUERY);
  ExpectQueryType(detail::OperatorCapability::JIT_CLASS_QUERY, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::JIT_TYPE_QUERY, detail::CapabilityQueryType::STR_CAT_QUERY);
  ExpectQueryType(detail::OperatorCapability::DYN_SHM_SIZE, detail::CapabilityQueryType::MAX_QUERY);
  ExpectQueryType(detail::OperatorCapability::BLOCK_DIM, detail::CapabilityQueryType::RANGE_QUERY);
  ExpectQueryType(detail::OperatorCapability::GENERATE_LTOIR, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::PASS_THROUGH_THREADS, detail::CapabilityQueryType::OR_QUERY);
  ExpectQueryType(detail::OperatorCapability::UNIT_STRIDE_LAST, detail::CapabilityQueryType::AND_QUERY);
  ExpectQueryType(detail::OperatorCapability::NONE, detail::CapabilityQueryType::OR_QUERY);
}

TEST(CapabilitiesTests, CombinesBoolCapabilities)
{
  EXPECT_FALSE(detail::combine_capabilities<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(false));
  EXPECT_TRUE(detail::combine_capabilities<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(false, true, false));
  EXPECT_TRUE(detail::combine_capabilities<detail::OperatorCapability::SUPPORTS_JIT>(true));
  EXPECT_TRUE(detail::combine_capabilities<detail::OperatorCapability::SUPPORTS_JIT>(true, true, true));
  EXPECT_FALSE(detail::combine_capabilities<detail::OperatorCapability::SUPPORTS_JIT>(true, true, false));
  EXPECT_FALSE(detail::combine_capabilities<detail::OperatorCapability::GLOBAL_KERNEL>(true, true, false));
  EXPECT_TRUE(detail::combine_capabilities<detail::OperatorCapability::PASS_THROUGH_THREADS>(false, false, true));
  EXPECT_FALSE(detail::combine_capabilities<detail::OperatorCapability::UNIT_STRIDE_LAST>(true, true, false));
}

TEST(CapabilitiesTests, CombinesNumericAndStringCapabilities)
{
  EXPECT_EQ(detail::combine_capabilities<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(16), 16);
  EXPECT_EQ(detail::combine_capabilities<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(16, 8, 32), 8);
  EXPECT_EQ(detail::combine_capabilities<detail::OperatorCapability::DYN_SHM_SIZE>(64), 64);
  EXPECT_EQ(detail::combine_capabilities<detail::OperatorCapability::DYN_SHM_SIZE>(64, 128, 32), 128);
  EXPECT_EQ(detail::combine_capabilities<detail::OperatorCapability::JIT_TYPE_QUERY>(
                std::string("self:"), std::string("child0:"), std::string("child1")),
            "self:child0:child1");
}

TEST(CapabilitiesTests, CombinesRangeCapabilities)
{
  using detail::ElementsPerThread;
  using EptRange = cuda::std::array<ElementsPerThread, 2>;
  using IntRange = cuda::std::array<int, 2>;

  ExpectRange(detail::combine_capabilities<detail::OperatorCapability::ELEMENTS_PER_THREAD>(
                  EptRange{ElementsPerThread::TWO, ElementsPerThread::SIXTEEN}),
              ElementsPerThread::TWO, ElementsPerThread::SIXTEEN);

  ExpectRange(detail::combine_capabilities<detail::OperatorCapability::ELEMENTS_PER_THREAD>(
                  EptRange{ElementsPerThread::TWO, ElementsPerThread::SIXTEEN},
                  EptRange{ElementsPerThread::FOUR, ElementsPerThread::THIRTY_TWO},
                  EptRange{ElementsPerThread::ONE, ElementsPerThread::EIGHT}),
              ElementsPerThread::FOUR, ElementsPerThread::EIGHT);

  ExpectRange(detail::combine_capabilities<detail::OperatorCapability::ELEMENTS_PER_THREAD>(
                  EptRange{ElementsPerThread::ONE, ElementsPerThread::TWO},
                  EptRange{ElementsPerThread::FOUR, ElementsPerThread::EIGHT}),
              ElementsPerThread::INVALID, ElementsPerThread::INVALID);

  ExpectRange(detail::combine_capabilities<detail::OperatorCapability::GROUPS_PER_BLOCK>(
                  IntRange{2, 16}, IntRange{4, 32}, IntRange{1, 8}),
              4, 8);

  ExpectRange(detail::combine_capabilities<detail::OperatorCapability::BLOCK_DIM>(
                  IntRange{128, 1024}, IntRange{256, 512}),
              256, 512);
}

TEST(CapabilitiesTests, GetsAndCombinesMatXOperatorCapabilities)
{
  using detail::ElementsPerThread;

  CapabilityTestOp op0;
  op0.supports_jit = true;
  op0.async_loads = true;
  op0.global_kernel = true;
  op0.max_ept_vec_load = 16;
  op0.dyn_shm_size = 96;
  op0.jit_type = "op0;";
  op0.ept_range = {ElementsPerThread::TWO, ElementsPerThread::SIXTEEN};
  op0.groups_range = {1, 8};

  CapabilityTestOp op1;
  op1.supports_jit = false;
  op1.async_loads = false;
  op1.global_kernel = false;
  op1.max_ept_vec_load = 8;
  op1.dyn_shm_size = 192;
  op1.jit_type = "op1;";
  op1.ept_range = {ElementsPerThread::FOUR, ElementsPerThread::THIRTY_TWO};
  op1.groups_range = {4, 16};

  detail::EPTQueryInput ept_input{false};
  const auto ops = cuda::std::make_tuple(op0, op1);

  EXPECT_TRUE(detail::get_operator_capability<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(op0));
  EXPECT_FALSE(detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op1));
  EXPECT_FALSE(detail::get_combined_ops_capability<detail::OperatorCapability::SUPPORTS_JIT>(ops));
  EXPECT_TRUE(detail::get_combined_ops_capability<detail::OperatorCapability::ASYNC_LOADS_REQUESTED>(ops));
  EXPECT_FALSE(detail::get_combined_ops_capability<detail::OperatorCapability::GLOBAL_KERNEL>(ops));
  EXPECT_EQ(detail::get_combined_ops_capability<detail::OperatorCapability::MAX_EPT_VEC_LOAD>(ops), 8);
  EXPECT_EQ(detail::get_combined_ops_capability<detail::OperatorCapability::DYN_SHM_SIZE>(ops), 192);
  EXPECT_EQ(detail::get_combined_ops_capability<detail::OperatorCapability::JIT_TYPE_QUERY>(ops), "op0;op1;");
  ExpectRange(detail::get_combined_ops_capability<detail::OperatorCapability::ELEMENTS_PER_THREAD>(ept_input, ops),
              ElementsPerThread::FOUR, ElementsPerThread::SIXTEEN);
  ExpectRange(detail::get_combined_ops_capability<detail::OperatorCapability::GROUPS_PER_BLOCK>(ops), 4, 8);
}

TEST(CapabilitiesTests, CapabilityParamsExposeCompileTimeDefaults)
{
  using Params = detail::CapabilityParams<detail::ElementsPerThread::FOUR, true, true>;

  static_assert(detail::is_scoped_enum_v<detail::ElementsPerThread>);
  static_assert(!detail::is_scoped_enum_v<int>);
  static_assert(detail::DefaultCapabilities::ept == detail::ElementsPerThread::ONE);
  static_assert(!detail::DefaultCapabilities::jit);
  static_assert(!detail::DefaultCapabilities::unit_stride_last);
  static_assert(Params::ept == detail::ElementsPerThread::FOUR);
  static_assert(Params::jit);
  static_assert(Params::unit_stride_last);
  static_assert(Params::osize == 0);
  static_assert(Params::block_size == 0);

  EXPECT_EQ(Params::ept, detail::ElementsPerThread::FOUR);
  EXPECT_TRUE(Params::jit);
  EXPECT_TRUE(Params::unit_stride_last);
}
