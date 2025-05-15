<<<<<<< HEAD
////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include "matx/core/defines.h"
#include "matx/core/type_utils.h"
#include <type_traits>
#include <limits>
#include <algorithm>
#include <cuda/std/__algorithm/min.h>

namespace matx {

namespace detail {

  enum class ElementsPerThread {
    ONE = 1,
    TWO = 2,
    FOUR = 4,
    EIGHT = 8,
    SIXTEEN = 16,
    THIRTY_TWO = 32,

    MAX = THIRTY_TWO
  };

  // Enum for different operator capabilities
  enum class OperatorCapability {
    NONE,
    SUPPORTS_JIT,                 // Can this operation be JIT-compiled?
    ELEMENTS_PER_THREAD,          // How many elements per thread?
    // Add more capabilities as needed
  };

  // Enum to define how a capability query is propagated/aggregated
  enum class CapabilityQueryType {
    OR_QUERY,  // Result is true if ANY relevant operator in the expression has the capability.
            // The operator itself OR its children.
    AND_QUERY,  // Result is true only if ALL relevant operators in the expression have the capability.
            // The operator itself AND its children.
    MIN_QUERY,  // Result is the minimum of the capabilities of the operator and its children.
  };

  // Trait to get default values and identities based on capability
  template <OperatorCapability Cap>
  struct capability_attributes; // Forward declaration

  template <>
  struct capability_attributes<OperatorCapability::SUPPORTS_JIT> {
    using type = bool;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };

  template <>
  struct capability_attributes<OperatorCapability::ELEMENTS_PER_THREAD> {
    using type = ElementsPerThread;
    static constexpr ElementsPerThread default_value = ElementsPerThread::MAX; // Example: 1 element per thread by default
    static constexpr ElementsPerThread min_identity = static_cast<ElementsPerThread>(std::numeric_limits<int>::max());
  };

  // Helper to safely get capability from an operator.
  // OperandType is likely base_type_t<ActualOpType> or a raw scalar/functor type.
  template <OperatorCapability Cap, typename OperatorType>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  get_operator_capability(const OperatorType& op) {
    if constexpr (matx::is_matx_op<OperatorType>()) {
      return op.template get_capability<Cap>();
    } else {
      // Default capabilities for non-MatX ops
      return capability_attributes<Cap>::default_value;
    }
  }

  // Helper function to get the query type associated with a capability
  // This defines the default aggregation logic for each capability.
  inline CapabilityQueryType get_query_type(OperatorCapability cap) {
    switch (cap) {
      case OperatorCapability::SUPPORTS_JIT:
        return CapabilityQueryType::OR_QUERY; // If any sub-operator supports JIT, the expression might be JIT-able.
      case OperatorCapability::ELEMENTS_PER_THREAD:
        return CapabilityQueryType::MIN_QUERY; // The expression should use the minimum elements per thread of its children.
      default:
        // Default to OR_QUERY or handle as an error/assertion if a capability isn't mapped.
        return CapabilityQueryType::OR_QUERY; 
    }
  }

  // Helper function to combine self capability with child capabilities
  // based on the query type of the OperatorCapability.
  template <OperatorCapability Cap, typename SelfCapType, typename... ChildCapTypes>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  combine_capabilities(
      SelfCapType self_val, // This will be capability_attributes<Cap>::type
      ChildCapTypes... child_vals) // These will also be capability_attributes<Cap>::type
  {
    using CapType = typename capability_attributes<Cap>::type;
    static_assert(std::is_same_v<SelfCapType, CapType>, "Self capability type mismatch.");
    // Ensure all child_vals are also CapType (checked by caller context or can add static_asserts here too)
    // ( (static_assert(std::is_same_v<ChildCapTypes, CapType>,"Child capability type mismatch.")), ...); // C++17 fold for static_assert

    CapabilityQueryType query_type = get_query_type(Cap);
    CapType children_aggregated_val;

    // Step 1: Aggregate children capabilities
    if constexpr (sizeof...(ChildCapTypes) == 0) {
      // No children: result is the identity for the query type.
      if constexpr (std::is_same_v<CapType, bool>) {
        children_aggregated_val = (query_type == CapabilityQueryType::AND_QUERY) ?
                                  capability_attributes<Cap>::and_identity :
                                  capability_attributes<Cap>::or_identity;
      } else if constexpr (std::is_same_v<CapType, int>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::min_identity;
        } else {
          // Default identity for int if not MIN_QUERY (e.g. if it was SUM_QUERY, identity would be 0)
          // This path needs clear definition if other query types are used for int.
          children_aggregated_val = capability_attributes<Cap>::default_value; // Fallback
        }
      } else {
        // Fallback for other types, should be defined in capability_attributes
        children_aggregated_val = capability_attributes<Cap>::default_value;
      }
    } else { // One or more children
      if constexpr (std::is_same_v<CapType, bool>) {
          if (query_type == CapabilityQueryType::OR_QUERY) {
              children_aggregated_val = (child_vals || ...);
          } else { // AND_QUERY
              children_aggregated_val = (child_vals && ...);
          }
      } else if constexpr (std::is_same_v<CapType, int> || std::is_same_v<std::underlying_type_t<CapType>, int>) {
          if (query_type == CapabilityQueryType::MIN_QUERY) {
              children_aggregated_val = capability_attributes<Cap>::min_identity;
              // C++17 way to apply cuda::std::min over a parameter pack
              cuda::std::initializer_list<CapType> values = {child_vals...};
              for (CapType val : values) {
                  children_aggregated_val = static_cast<CapType>(cuda::std::min(static_cast<int>(children_aggregated_val), static_cast<int>(val)));
              }
          } else {
              // Not implemented for other query types.
              MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
          }
      } else {
          // Not implemented for other types.
          MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other types.");
      }
    }

    // Step 2: Combine self's capability with the children's combined result.
    if constexpr (std::is_same_v<CapType, bool>) {
        if (query_type == CapabilityQueryType::OR_QUERY) {
            return self_val || children_aggregated_val;
        } else { // AND_QUERY
            return self_val && children_aggregated_val;
        }
    } else if constexpr (std::is_same_v<CapType, int> || std::is_same_v<std::underlying_type_t<CapType>, int>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
            return static_cast<CapType>(cuda::std::min(static_cast<int>(self_val), static_cast<int>(children_aggregated_val)));
        } else {
            MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
            return self_val;
        }
    } else {
        MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other types.");
        return self_val;
    }
  }

} // namespace detail
=======
#pragma once

#include "matx/core/defines.h"
#include <type_traits>

namespace matx {

// Enum for different operator capabilities
enum class OperatorCapability {
    NONE,
    SUPPORTS_JIT,                 // Can this operation be JIT-compiled?
    // Add more capabilities as needed
};

// Enum to define how a capability query is propagated/aggregated
enum class CapabilityQueryType {
    OR_QUERY,  // Result is true if ANY relevant operator in the expression has the capability.
               // The operator itself OR its children.
    AND_QUERY  // Result is true only if ALL relevant operators in the expression have the capability.
               // The operator itself AND its children.
};

// Helper function to get the query type associated with a capability
// This defines the default aggregation logic for each capability.
inline CapabilityQueryType get_query_type(OperatorCapability cap) {
    switch (cap) {
        case OperatorCapability::SUPPORTS_JIT:
            return CapabilityQueryType::OR_QUERY; // If any sub-operator supports JIT, the expression might be JIT-able.
        default:
            // Default to OR_QUERY or handle as an error/assertion if a capability isn't mapped.
            return CapabilityQueryType::OR_QUERY; 
    }
}

// Trait to check for the matxop typedef
template <typename T, typename = void>
struct has_matxop_trait : std::false_type {};

template <typename T>
struct has_matxop_trait<T, std::void_t<typename T::matxop>> : std::true_type {};

template<typename T>
inline constexpr bool has_matxop_trait_v = has_matxop_trait<T>::value;

namespace detail {
  // Helper to safely get capability from an operand.
  // OperandType is likely base_type_t<ActualOpType> or a raw scalar/functor type.
  template <typename OperandType>
  __MATX_INLINE__ __MATX_HOST__ bool get_operand_capability(const OperandType& operand, OperatorCapability cap) {
    if constexpr (is_matx_op<OperandType>()) {
      return operand.has_capability(cap); // Call the member function if it exists
    } else {
      // Default capabilities for non-MatX ops (e.g., scalars if not wrapped in a BaseOp derivative, simple functors)
      switch (cap) {
        case OperatorCapability::SUPPORTS_JIT:
          // Scalars and simple types are generally JIT-friendly (can be inlined).
          // A more precise check could be std::is_arithmetic_v<OperandType> if OperandType is guaranteed to be the raw type.
          return true;
        default:
          return false; // Default to false for other capabilities for non-MatX ops.
      }
    }
  }
} // namespace detail

// Helper function to combine self capability with child capabilities
// based on the query type of the OperatorCapability.
template <typename... ChildBools>
__MATX_INLINE__ __MATX_HOST__ bool combine_capabilities(
    OperatorCapability cap,
    bool self_has_cap,
    ChildBools... child_caps) // Variadic pack of children's capabilities
{
    CapabilityQueryType query_type = get_query_type(cap);
    bool children_combined_result;

    // Determine the combined capability of children based on the query type.
    if constexpr (sizeof...(ChildBools) == 0) {
        // No children: result is the identity for the query type.
        // Identity for AND is true, identity for OR is false.
        children_combined_result = (query_type == CapabilityQueryType::AND_QUERY);
    } else if constexpr (sizeof...(ChildBools) == 1) {
        // One child: result is simply that child's capability.
        // Helper to extract the single boolean from the pack.
        auto get_single_value = [](bool val) { return val; };
        children_combined_result = get_single_value(child_caps...); // Expands to get_single_value(the_one_child_bool)
    } else { // sizeof...(ChildBools) > 1
        // Multiple children: use fold expressions, which are now distinct.
        if (query_type == CapabilityQueryType::OR_QUERY) {
            children_combined_result = (child_caps || ...); // True if any child has the capability
        } else { // AND_QUERY
            children_combined_result = (child_caps && ...); // True if all children have the capability
        }
    }

    // Combine self's capability with the children's combined result.
    if (query_type == CapabilityQueryType::OR_QUERY) {
        return self_has_cap || children_combined_result;
    } else { // AND_QUERY
        return self_has_cap && children_combined_result;
    }
}

>>>>>>> 0e008fc46 (Beginning of getting RTC to work)
} // namespace matx 