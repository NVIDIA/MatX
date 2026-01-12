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
//  list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright notice,
//  this list of conditions and the following disclaimer in the documentation
//  and/or other materials provided with the distribution.
//
// 3. Neither the name of the copyright holder nor the names of its
//  contributors may be used to endorse or promote products derived from
//  this software without specific prior written permission.
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

#include "matx/core/type_utils.h"
#include <type_traits>
#include <utility>

namespace matx
{

template <typename... Ts>
struct type_list {};

// Check if a type is contained in a type_list
template <typename T, typename List>
struct contains;

template <typename T>
struct contains<T, type_list<>> : std::false_type {};

template <typename T, typename Head, typename... Tail>
struct contains<T, type_list<Head, Tail...>>
    : cuda::std::conditional_t<std::is_same_v<T, Head>, std::true_type, contains<T, type_list<Tail...>>> {};

// Helper to append a type if not already in the list
template <typename List, typename T>
struct append_unique;

template <typename... Ts, typename T>
struct append_unique<type_list<Ts...>, T> {
    using type = cuda::std::conditional_t<
        contains<T, type_list<Ts...>>::value,
        type_list<Ts...>,           // already present -> keep list as is
        type_list<Ts..., T>         // otherwise append
    >;
};

// Recursively merge multiple types into a list
template <typename List, typename... NewTs>
struct merge_props_unique;

template <typename List>
struct merge_props_unique<List> { using type = List; };

template <typename List, typename Head, typename... Tail>
struct merge_props_unique<List, Head, Tail...> {
    using type = typename merge_props_unique<
        typename append_unique<List, Head>::type,
        Tail...
    >::type;
};

// Tag property
template <typename T>
struct prop_tag {};

// Template category
template <template <typename> class C>
struct prop_category {};

template <typename Spec, typename... Props>
struct has_property;

template <typename Tag, typename... Props>
inline constexpr bool has_property_tag = (cuda::std::is_same_v<Tag, Props> || ...);

template <template <typename> class C, typename T>
struct is_property_category : std::false_type {};

template <template <typename> class C, typename T>
struct is_property_category<C, C<T>> : std::true_type {};

template <template <typename> class C, typename... Props>
inline constexpr bool has_property_category = (is_property_category<C, Props>::value || ...);

// The get_property_or helpers are used to extract a type from a category-style
// property (i.e., a templated property) or return a default type if the property
// is not found. Using the PropAccum property as an example, we can extract the accumulator
// type or use the default output_t as follows:
//   using accum_type = get_property_or<PropAccum, output_t, CurrentProps...>::type;

// Base case: no properties, use the default type
template <template <typename> class Prop, typename Default, typename... Props>
struct get_property_or {
    using type = Default;
};

// Case 1: Head matches Prop<T>
template <template <typename> class Prop,
          typename Default,
          typename T,
          typename... Tail>
struct get_property_or<Prop, Default, Prop<T>, Tail...> {
    using type = T;
};

// Case 2: Head is something else
template <template <typename> class Prop,
          typename Default,
          typename Head,
          typename... Tail>
struct get_property_or<Prop, Default, Head, Tail...> {
    using type = typename get_property_or<Prop, Default, Tail...>::type;
};

// Below are common properties meant to be reused by multiple operators. If a property only
// makes sense for a single operator (such as a tag specificying a specific algorithm), then
// the property should be defined directly in the operator.

// PropAccum is a property to set the type of an accumulator. This should be used for operations
// where a single accumulation type is sufficient. For operations where this type would be
// ambiguous (e.g., those requiring multiple accumulators), more fine-grained properties
// could be added.
template <typename T>
struct PropAccum { using type = T; };

// PropOutput is a property to set the output type of an operator. By default, output types of
// operators are deduced at compile time based on the input types and other operator properties.
// The PropOutput property can be used to override the default output type. This can be useful,
// for example, when an operator has single-precision inputs, but the user wishes to override
// the accumulator type to double precision and avoid any implicit conversions to single
// precision when writing the outputs.
template <typename T>
struct PropOutput { using type = T; };

};
