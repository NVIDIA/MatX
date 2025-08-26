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
#include "matx/core/type_utils_both.h"
#include <type_traits>
#include <limits>
#include <algorithm>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/array>

namespace matx {

namespace detail {

  struct VoidCapabilityType {
    static constexpr bool value = true;
  };

  enum class ElementsPerThread {
    INVALID = 0,
    ONE = 1,
    TWO = 2,
    FOUR = 4,
    EIGHT = 8,
    SIXTEEN = 16,
    THIRTY_TWO = 32,

    MAX = THIRTY_TWO
  };

  // Input structure for types that require it
  struct ShmQueryInput {
    ElementsPerThread ept;
  };

  struct EPTQueryInput {
    bool jit;
  };

  struct BlockSizeQueryInput {
    ElementsPerThread ept;
  };

  struct JITQueryInput {
    ElementsPerThread ept;
  };

  using BlockDimType = int;


  // Enum for different operator capabilities
  enum class OperatorCapability {
    NONE,
    SUPPORTS_JIT,                 // Can this operation be JIT-compiled?
    ELEMENTS_PER_THREAD,          // How many elements per thread?
    JIT_CAP_QUERY,  // Result is the concatenation of the capabilities of the operator and its children.
    DYN_SHM_SIZE,   // Result is the dynamic shared memory size required for the operator.
    BLOCK_DIM,      // Result is the block dimensions required for the operator.
    // Add more capabilities as needed
  };

  // Enum to define how a capability query is propagated/aggregated
  enum class CapabilityQueryType {
    OR_QUERY,  // Result is true if ANY relevant operator in the expression has the capability.
            // The operator itself OR its children.
    AND_QUERY,  // Result is true only if ALL relevant operators in the expression have the capability.
            // The operator itself AND its children.
    MIN_QUERY,  // Result is the minimum of the capabilities of the operator and its children.
    MAX_QUERY,  // Result is the maximum of the capabilities of the operator and its children.
    STR_CAT_QUERY,  // Result is the concatenation of the capabilities of the operator and its children.
    RANGE_QUERY,  // Result is the range of the capabilities of the operator and its children.
  };
  

#if !defined(__CUDACC_RTC__)
  template <ElementsPerThread EPT, bool JIT>
  struct CapabilityParams {
    static constexpr ElementsPerThread ept = EPT;
    static constexpr bool jit = JIT;
    static constexpr int osize = 0;
    static constexpr int block_size = 0;

    // For JIT there will be other capabilties patched in with a string
  };  

  using DefaultCapabilities = CapabilityParams<ElementsPerThread::ONE, false>;  
  
  // C++17-compatible trait to detect scoped enums
  template<typename T, typename = void>
  struct is_scoped_enum : std::false_type {};

  template<typename T>
  struct is_scoped_enum<T, std::enable_if_t<cuda::std::is_enum_v<T>>> 
      : cuda::std::bool_constant<!cuda::std::is_convertible_v<T, cuda::std::underlying_type_t<T>>> {};

  template<typename T>
  constexpr bool is_scoped_enum_v = is_scoped_enum<T>::value;

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
  struct capability_attributes<OperatorCapability::BLOCK_DIM> {
    using type = int; // min/max elements per thread
    static constexpr int default_value = 1024; // Example: 1 element per thread by default
    static constexpr int min_identity = 1024;
    static constexpr int max_identity = 1024;
  };  

  template <>
  struct capability_attributes<OperatorCapability::ELEMENTS_PER_THREAD> {
    using type = cuda::std::array<ElementsPerThread, 2>; // min/max elements per thread
    static constexpr ElementsPerThread invalid = ElementsPerThread::INVALID;
    static constexpr cuda::std::array<ElementsPerThread, 2> default_value = {ElementsPerThread::ONE, ElementsPerThread::MAX}; // Example: 1 element per thread by default
    static constexpr cuda::std::array<ElementsPerThread, 2> min_identity = {ElementsPerThread::MAX, ElementsPerThread::ONE};
    static constexpr cuda::std::array<ElementsPerThread, 2> max_identity = {ElementsPerThread::ONE, ElementsPerThread::MAX};
  };

  template <>
  struct capability_attributes<OperatorCapability::JIT_CAP_QUERY> {
    using type = std::string;
    static inline const std::string default_value = "";
    static inline const std::string min_identity = "";
  };  

  template <>
  struct capability_attributes<OperatorCapability::DYN_SHM_SIZE> {
    using type = int;
    static constexpr int default_value = 0;
    static constexpr int min_identity = std::numeric_limits<int>::max();
    static constexpr int max_identity = 0;
  };    

  // Helper to safely get capability from an operator.
  // OperandType is likely base_type_t<ActualOpType> or a raw scalar/functor type.
  template <OperatorCapability Cap, typename OperatorType>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  get_operator_capability(const OperatorType& op) {
    if constexpr (matx::is_matx_op<OperatorType>()) {
      return op.template get_capability<Cap, VoidCapabilityType>(VoidCapabilityType{});
    } else {
      // Default capabilities for non-MatX ops
      return capability_attributes<Cap>::default_value;
    }
  }

  template <OperatorCapability Cap, typename OperatorType, typename InType>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  get_operator_capability(const OperatorType& op, const InType& in) {
    if constexpr (matx::is_matx_op<OperatorType>()) {
      return op.template get_capability<Cap, InType>(in);
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
        return CapabilityQueryType::AND_QUERY; // If any sub-operator supports JIT, the expression might be JIT-able.
      case OperatorCapability::ELEMENTS_PER_THREAD:
        return CapabilityQueryType::RANGE_QUERY; // The expression should use the range of elements per thread of its children.
      case OperatorCapability::JIT_CAP_QUERY:
        return CapabilityQueryType::STR_CAT_QUERY; // The expression should use the concatenation of the capabilities of its children.
      case OperatorCapability::DYN_SHM_SIZE:
        return CapabilityQueryType::MAX_QUERY; // The expression should use the maximum dynamic shared memory size of its children.
      case OperatorCapability::BLOCK_DIM:
        return CapabilityQueryType::MIN_QUERY; // The expression should use the minimum block size supported by all operators.
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
      } else if constexpr (std::is_same_v<CapType, int> || is_scoped_enum_v<CapType>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::min_identity;
        } else if (query_type == CapabilityQueryType::MAX_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::max_identity;
        } else {
          // Default identity for int if not MIN_QUERY or MAX_QUERY (e.g. if it was SUM_QUERY, identity would be 0)
          // This path needs clear definition if other query types are used for int.
          children_aggregated_val = capability_attributes<Cap>::default_value; // Fallback
        }
      } else if constexpr (std::is_same_v<CapType, BlockDimType>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::min_identity;
        } else {
          children_aggregated_val = capability_attributes<Cap>::default_value; // Fallback
        }
      } else if constexpr (std::is_same_v<CapType, std::string>) {
        children_aggregated_val = capability_attributes<Cap>::default_value;
      } else {
        // Check if it's a cuda::std::array<T, 2> for RANGE_QUERY
        if (query_type == CapabilityQueryType::RANGE_QUERY) {
          // For RANGE_QUERY with no children, use the default_value as identity
          children_aggregated_val = capability_attributes<Cap>::default_value;
        } else {
          // Fallback for other types, should be defined in capability_attributes
          children_aggregated_val = capability_attributes<Cap>::default_value; // Fallback
        }
      }
    } else { // One or more children
      if constexpr (std::is_same_v<CapType, bool>) {
          if (query_type == CapabilityQueryType::OR_QUERY) {
              children_aggregated_val = capability_attributes<Cap>::or_identity;
              ((children_aggregated_val = children_aggregated_val || child_vals), ...);     
          } else { // AND_QUERY
              children_aggregated_val = capability_attributes<Cap>::and_identity;
              ((children_aggregated_val = children_aggregated_val && child_vals), ...);
          }
      } else if constexpr (std::is_same_v<CapType, int> || is_scoped_enum_v<CapType>) {
          if (query_type == CapabilityQueryType::MIN_QUERY) {
              children_aggregated_val = capability_attributes<Cap>::min_identity;
              // C++17 way to apply cuda::std::min over a parameter pack
              cuda::std::initializer_list<CapType> values = {child_vals...};
              for (CapType val : values) {
                  children_aggregated_val = static_cast<CapType>(cuda::std::min(static_cast<int>(children_aggregated_val), static_cast<int>(val)));
              }
          } else if (query_type == CapabilityQueryType::MAX_QUERY) {
              children_aggregated_val = capability_attributes<Cap>::max_identity;
              // C++17 way to apply cuda::std::max over a parameter pack
              cuda::std::initializer_list<CapType> values = {child_vals...};
              for (CapType val : values) {
                  children_aggregated_val = static_cast<CapType>(cuda::std::max(static_cast<int>(children_aggregated_val), static_cast<int>(val)));
              }
          } else {
              // Not implemented for other query types.
              MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
          }
      } else if constexpr (std::is_same_v<CapType, BlockDimType>) {
          if (query_type == CapabilityQueryType::MIN_QUERY) {
              children_aggregated_val = capability_attributes<Cap>::min_identity;
              // For BLOCK_DIM, we only care about the third element (index 2)
              cuda::std::initializer_list<CapType> values = {child_vals...};
              for (const CapType& val : values) {
                  children_aggregated_val[2] = cuda::std::min(children_aggregated_val[2], val[2]);
              }
          } else {
              // Not implemented for other query types.
              MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
          }
      } else if constexpr (std::is_same_v<CapType, std::string>) {
        if (query_type == CapabilityQueryType::STR_CAT_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::default_value;
          ((children_aggregated_val += child_vals), ...);
        } else {
          children_aggregated_val = capability_attributes<Cap>::default_value;
        }
      } else {
          // Handle RANGE_QUERY for cuda::std::array<T, 2> types
          if (query_type == CapabilityQueryType::RANGE_QUERY) {
            // Initialize with the first child's value
            cuda::std::initializer_list<CapType> values = {child_vals...};
            auto it = values.begin();
            children_aggregated_val = *it;
            ++it;
            
            // Apply range intersection logic for remaining children
            for (; it != values.end(); ++it) {
              const auto& child_range = *it;
              // Minimum is the maximum of the two range's minimums
              // Maximum is the minimum of the two range's maximums
              // Check that the maximum (second element) is not smaller than the minimum on the other value
              if (static_cast<int>(child_range[1]) < static_cast<int>(children_aggregated_val[0]) || 
                  static_cast<int>(children_aggregated_val[1]) < static_cast<int>(child_range[0])) {
                // If the max of the new range is less than the min of the current, clamp to empty/invalid range
                children_aggregated_val[0] = capability_attributes<Cap>::invalid;
                children_aggregated_val[1] = capability_attributes<Cap>::invalid;
                break;
              }
              else {
                children_aggregated_val[0] = static_cast<typename CapType::value_type>(
                  cuda::std::max(static_cast<int>(children_aggregated_val[0]), static_cast<int>(child_range[0])));
                children_aggregated_val[1] = static_cast<typename CapType::value_type>(
                  cuda::std::min(static_cast<int>(children_aggregated_val[1]), static_cast<int>(child_range[1])));
              }
            }
          } else {
            // Not implemented for other types.
            MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other types.");
          }
      }
    }

    // Step 2: Combine self's capability with the children's combined result.
    if constexpr (std::is_same_v<CapType, bool>) {
        // Optimize when identity values make the operation redundant
        if (query_type == CapabilityQueryType::OR_QUERY) {
            // self_val || or_identity
            if (children_aggregated_val == capability_attributes<Cap>::or_identity) {
                return self_val; // self_val || false = self_val
            }

            return self_val || children_aggregated_val;
        } else { // AND_QUERY
            // self_val && and_identity
            if (children_aggregated_val == capability_attributes<Cap>::and_identity) {
                return self_val; // self_val && true = self_val
            }
            return self_val && children_aggregated_val;
        }
    } else if constexpr (std::is_same_v<CapType, int> || is_scoped_enum_v<CapType>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
            return static_cast<CapType>(cuda::std::min(static_cast<int>(self_val), static_cast<int>(children_aggregated_val)));
        } else if (query_type == CapabilityQueryType::MAX_QUERY) {
            return static_cast<CapType>(cuda::std::max(static_cast<int>(self_val), static_cast<int>(children_aggregated_val)));
        } else {
            MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
            return self_val;
        }
    } else if constexpr (std::is_same_v<CapType, BlockDimType>) {
        if (query_type == CapabilityQueryType::MIN_QUERY) {
            CapType result = self_val;
            // For BLOCK_DIM, we only care about the third element (index 2)
            result[2] = cuda::std::min(self_val[2], children_aggregated_val[2]);
            return result;
        } else {
            MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
            return self_val;
        }
    } else if constexpr (std::is_same_v<CapType, std::string>) {
        if (query_type == CapabilityQueryType::STR_CAT_QUERY) {
            return self_val + children_aggregated_val;
        } else {
            MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other query types.");
            return self_val;
        }
    } else {
        // Handle RANGE_QUERY for cuda::std::array<T, 2> types
        if (query_type == CapabilityQueryType::RANGE_QUERY) {
          CapType result = self_val;
          // Apply range intersection logic: 
          // Minimum is the maximum of the two range's minimums
          // Maximum is the minimum of the two range's maximums
          // Check that the maximum (second element) is not smaller than the minimum on the other value
          if (static_cast<int>(children_aggregated_val[1]) < static_cast<int>(self_val[0]) || 
              static_cast<int>(self_val[1]) < static_cast<int>(children_aggregated_val[0])) {
            // If the max of the new range is less than the min of the current, clamp to empty/invalid range
            result[0] = capability_attributes<Cap>::invalid;
            result[1] = capability_attributes<Cap>::invalid;  
          }
          else {
            result[0] = static_cast<typename CapType::value_type>(
              cuda::std::max(static_cast<int>(self_val[0]), static_cast<int>(children_aggregated_val[0])));
            result[1] = static_cast<typename CapType::value_type>(
              cuda::std::min(static_cast<int>(self_val[1]), static_cast<int>(children_aggregated_val[1])));
          }
          return result;
        } else {
          MATX_ASSERT_STR(false, matxInvalidParameter, "Not implemented for other types.");
          return self_val;
        }
    }
  }

#endif

} // namespace detail
} // namespace matx 