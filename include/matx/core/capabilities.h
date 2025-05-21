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

} // namespace matx 