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

#include "assert.h"
#include "matx.h"
#include "test_types.h"
#include "utilities.h"
#include "gtest/gtest.h"
#include "matx/core/props.h"
#include <iostream>
#include <vector>
#include <unordered_map>

using namespace matx;

struct PropTagTest1 {};
struct PropTagTest2 {};

template <typename OpA, typename List>
struct make_property_accum_test_op;

template <typename OpA, typename List>
struct make_property_tag_test_op;

template<typename OpA, typename... CurrentProps>
class PropertyAccumOutputTest : public BaseOp<PropertyAccumOutputTest<OpA, CurrentProps...>>
{
  private:
    OpA a_;
  public:
    __MATX_INLINE__ PropertyAccumOutputTest(const OpA &a) : a_(a) {}
    using value_type = typename OpA::value_type;

    // Return the size of the accumulator in bytes. The default is float (4 bytes).
    size_t sizeof_accum() const {
        using accum_t = detail::get_property_or<PropAccum, float, CurrentProps...>::type;
        return sizeof(accum_t);
    }

    size_t sizeof_output() const {
        using output_t = detail::get_property_or<PropOutput, value_type, CurrentProps...>::type;
        return sizeof(output_t);
    }

    template <typename... NewProps>
    constexpr auto props() {
        using AllProps = typename detail::merge_props_unique<detail::type_list<CurrentProps...>, NewProps...>::type;
        return make_property_accum_test_op<OpA, AllProps>::make(a_);
    }
};

template <typename OpA, typename... Props>
struct make_property_accum_test_op<OpA, detail::type_list<Props...>> {
    static auto make(OpA a) {
        return PropertyAccumOutputTest<OpA, Props...>(a);
    }
};

template<typename OpA, typename... CurrentProps>
class PropertyTagTest : public BaseOp<PropertyTagTest<OpA, CurrentProps...>>
{
  private:
    OpA a_;
  public:
    __MATX_INLINE__ PropertyTagTest(const OpA &a) : a_(a) {}

    template <typename Tag>
    constexpr bool has_prop_tag() const {
        return detail::has_property_tag<Tag, CurrentProps...>;
    }

    template <typename... NewProps>
    constexpr auto props() {
        using AllProps = typename detail::merge_props_unique<detail::type_list<CurrentProps...>, NewProps...>::type;
        return make_property_tag_test_op<OpA, AllProps>::make(a_);
    }
};

template <typename OpA, typename... Props>
struct make_property_tag_test_op<OpA, detail::type_list<Props...>> {
    static auto make(OpA a) {
        return PropertyTagTest<OpA, Props...>(a);
    }
};

TEST(PropertyTests, PropertyAccumTest) {
    auto a = make_tensor<float>({1000});
    auto op = PropertyAccumOutputTest(a);    
    EXPECT_EQ(op.sizeof_accum(), sizeof(float));
    auto op_fp64_accum = op.props<PropAccum<double>>();
    EXPECT_EQ(op_fp64_accum.sizeof_accum(), sizeof(double));
    auto op_fp32_accum = op.props<PropAccum<float>>();
    EXPECT_EQ(op_fp32_accum.sizeof_accum(), sizeof(float));
    auto op_fp16_accum = op.props<PropAccum<__half>>();
    EXPECT_EQ(op_fp16_accum.sizeof_accum(), sizeof(__half));
    auto op_fp8_accum = op.props<PropAccum<int8_t>>();
    EXPECT_EQ(op_fp8_accum.sizeof_accum(), sizeof(int8_t));
}

TEST(PropertyTests, PropertyTagTest) {
    auto a = make_tensor<float>({1000});
    auto op = PropertyTagTest(a);
    EXPECT_FALSE(op.has_prop_tag<PropTagTest1>());
    auto op_with_tag1 = op.props<PropTagTest1>();
    EXPECT_TRUE(op_with_tag1.has_prop_tag<PropTagTest1>());
    EXPECT_FALSE(op_with_tag1.has_prop_tag<PropTagTest2>());
    EXPECT_FALSE(op.has_prop_tag<float>());
    EXPECT_FALSE(op_with_tag1.has_prop_tag<int>());
    auto op_with_tag2 = op.props<PropTagTest2>();
    EXPECT_FALSE(op_with_tag2.has_prop_tag<PropTagTest1>());
    EXPECT_TRUE(op_with_tag2.has_prop_tag<PropTagTest2>());
    auto op_with_tags12_combined = op.props<PropTagTest1, PropTagTest2>();
    EXPECT_TRUE(op_with_tags12_combined.has_prop_tag<PropTagTest1>());
    EXPECT_TRUE(op_with_tags12_combined.has_prop_tag<PropTagTest2>());
    auto op_with_tags12_chained = op.props<PropTagTest1>().props<PropTagTest2>();
    EXPECT_TRUE(op_with_tags12_chained.has_prop_tag<PropTagTest1>());
    EXPECT_TRUE(op_with_tags12_chained.has_prop_tag<PropTagTest2>());    
}

TEST(PropertyTests, PropOutputTest) {
    auto a = make_tensor<float>({1000});
    auto op = PropertyAccumOutputTest(a);
    EXPECT_EQ(op.sizeof_output(), sizeof(float));
    auto op_fp64_output = op.props<PropOutput<double>>();
    EXPECT_EQ(op_fp64_output.sizeof_output(), sizeof(double));
    auto op_fp32_output = op.props<PropOutput<float>>();
    EXPECT_EQ(op_fp32_output.sizeof_output(), sizeof(float));
    auto op_fp16_output = op.props<PropOutput<__half>>();
    EXPECT_EQ(op_fp16_output.sizeof_output(), sizeof(__half));
    auto op_fp8_output = op.props<PropOutput<int8_t>>();
    EXPECT_EQ(op_fp8_output.sizeof_output(), sizeof(int8_t));
}