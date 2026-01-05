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
#include "matx/core/utils.h"
#include "matx/core/operator_options.h"
#include <cuda/std/type_traits>
#include <cuda/std/limits>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/array>
#include <string>
#include <set>

namespace matx {

namespace detail {

  struct LTOIRQueryInput {
    std::set<std::string> ltoir_symbols;
    ElementsPerThread ept;
  };  

  // Enum for different operator capabilities
  enum class OperatorCapability {
    NONE,
    SUPPORTS_JIT,                 // Can this operation be JIT-compiled?
    ELEMENTS_PER_THREAD,          // How many elements per thread?
    SET_ELEMENTS_PER_THREAD,      // Set the elements per thread for the operator.
    JIT_CLASS_QUERY,  // Result is the concatenation of the capabilities of the operator and its children.
    DYN_SHM_SIZE,   // Result is the dynamic shared memory size required for the operator.
    BLOCK_DIM,      // Result is the block dimensions required for the operator.
    GENERATE_LTOIR, // Generate LTOIR code for the operator.
    JIT_TYPE_QUERY, // Result is the type of JIT code to generate for the operator.
    GROUPS_PER_BLOCK, // Result is the number of groups per block required for the operator.(ie FFTs per block)
    SET_GROUPS_PER_BLOCK, // Set the number of groups per block for the operator.
    ASYNC_LOADS_REQUESTED, // Whether the operator requires asynchronous loads.
    MAX_EPT_VEC_LOAD, // The maximum EPT for a vector load.
    ELEMENT_WISE, // Whether the operator is element-wise (safe with aliasing)
    ALIASED_MEMORY, // Whether the operator's input and output pointers alias
    GLOBAL_KERNEL, // Kernel operates entirely on a global level per chunk of data. False when at least one operator works on a block level
    PASS_THROUGH_THREADS, // All threads must call operator() on nested operators; bounds checking done at tensor level
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
  
  // Concept to detect scoped enums
  template<typename T>
  concept is_scoped_enum_c = cuda::std::is_enum_v<T> && 
                             !cuda::std::is_convertible_v<T, cuda::std::underlying_type_t<T>>;

  // Legacy struct for backwards compatibility
  template<typename T, typename = void>
  struct is_scoped_enum : cuda::std::false_type {};

  template<typename T>
    requires is_scoped_enum_c<T>
  struct is_scoped_enum<T> : cuda::std::true_type {};

  template<typename T>
  constexpr bool is_scoped_enum_v = is_scoped_enum_c<T>;

  // Trait to get default values and identities based on capability
  template <OperatorCapability Cap>
  struct capability_attributes; // Forward declaration

  template <>
  struct capability_attributes<OperatorCapability::SUPPORTS_JIT> {
    using type = bool;
    using input_type = VoidCapabilityType;
    static constexpr bool default_value = false;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };

  template <>
  struct capability_attributes<OperatorCapability::GENERATE_LTOIR> {
    using type = bool;
    using input_type = LTOIRQueryInput;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };  

  template <>
  struct capability_attributes<OperatorCapability::ASYNC_LOADS_REQUESTED> {
    using type = bool;
    using input_type = VoidCapabilityType;
    static constexpr bool default_value = false;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  }; 
  
  template <>
  struct capability_attributes<OperatorCapability::GLOBAL_KERNEL> {
    using type = bool;
    using input_type = VoidCapabilityType;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };   

  template <>
  struct capability_attributes<OperatorCapability::ALIASED_MEMORY> {
    using type = bool;
    using input_type = AliasedMemoryQueryInput;
    static constexpr bool default_value = false;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };    

  template <>
  struct capability_attributes<OperatorCapability::GROUPS_PER_BLOCK> {
    using type = cuda::std::array<int, 2>; // min/max elements per thread
    using input_type = VoidCapabilityType;
    static constexpr int invalid = -1;
    static constexpr cuda::std::array<int, 2> default_value = {1, 32}; // Example: 1 element per thread by default
    static constexpr cuda::std::array<int, 2> min_identity = {32, 1};
    static constexpr cuda::std::array<int, 2> max_identity = {1, 32};
  };    

  template <>
  struct capability_attributes<OperatorCapability::BLOCK_DIM> {
    using type = cuda::std::array<int, 2>; // min/max elements per thread
    using input_type = VoidCapabilityType;
    static constexpr int invalid = -1;
    static constexpr cuda::std::array<int, 2> default_value = {1, 1024}; // Example: 1 element per thread by default
    static constexpr cuda::std::array<int, 2> min_identity = {1024, 1};
    static constexpr cuda::std::array<int, 2> max_identity = {1, 1024};
  };  

  template <>
  struct capability_attributes<OperatorCapability::SET_ELEMENTS_PER_THREAD> {
    using type = bool;
    using input_type = SetEPTQueryInput;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };  

  template <>
  struct capability_attributes<OperatorCapability::SET_GROUPS_PER_BLOCK> {
    using type = bool;
    using input_type = SetGroupsPerBlockQueryInput;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };  

  template <>
  struct capability_attributes<OperatorCapability::ELEMENTS_PER_THREAD> {
    using type = cuda::std::array<ElementsPerThread, 2>; // min/max elements per thread
    using input_type = EPTQueryInput;
    static constexpr ElementsPerThread invalid = ElementsPerThread::INVALID;
    static constexpr cuda::std::array<ElementsPerThread, 2> default_value = {ElementsPerThread::ONE, ElementsPerThread::MAX}; // Example: 1 element per thread by default
    static constexpr cuda::std::array<ElementsPerThread, 2> min_identity = {ElementsPerThread::MAX, ElementsPerThread::ONE};
    static constexpr cuda::std::array<ElementsPerThread, 2> max_identity = {ElementsPerThread::ONE, ElementsPerThread::MAX};
  };

  template <>
  struct capability_attributes<OperatorCapability::JIT_CLASS_QUERY> {
    using type = bool;
    using input_type = std::unordered_map<std::string, std::string>;
    static constexpr bool default_value = true;
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };  

  template <>
  struct capability_attributes<OperatorCapability::JIT_TYPE_QUERY> {
    using type = std::string;
    using input_type = VoidCapabilityType;
    static inline const std::string default_value = "";
    static inline const std::string min_identity = "";
  };    

  template <>
  struct capability_attributes<OperatorCapability::DYN_SHM_SIZE> {
    using type = int;
    using input_type = VoidCapabilityType;
    static constexpr int default_value = 0;
    static constexpr int min_identity = cuda::std::numeric_limits<int>::max();
    static constexpr int max_identity = 0;
  };    

  template <>
  struct capability_attributes<OperatorCapability::MAX_EPT_VEC_LOAD> {
    using type = int;
    using input_type = VoidCapabilityType;
    static constexpr int default_value = 32;
    static constexpr int min_identity = 32;
    static constexpr int max_identity = 1;
  };

  template <>
  struct capability_attributes<OperatorCapability::PASS_THROUGH_THREADS> {
    using type = bool;
    using input_type = VoidCapabilityType;
    static constexpr bool default_value = false;  // Default: operators do their own bounds checking
    static constexpr bool or_identity = false;
    static constexpr bool and_identity = true;
  };    


  template <OperatorCapability Cap, typename OperatorType, typename InType>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  get_operator_capability(const OperatorType& op, InType& in) {
    static_assert(std::is_same_v<remove_cvref_t<InType>, typename capability_attributes<Cap>::input_type>, "Input type mismatch");
    if constexpr (is_matx_jit_class<OperatorType>) {
      return op.template get_capability<Cap, InType>(in);
    } else {
      // Default capabilities for non-MatX ops
      if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
        return detail::type_to_string<OperatorType>();
      }
      else if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
        // If this is not a matx operator (like a constant or a lambda), we assume it supports JIT.
        return true;
      }
      else {
        return capability_attributes<Cap>::default_value;
      }
    }
  }  

  // Helper to safely get capability from an operator.
  // OperandType is likely base_type_t<ActualOpType> or a raw scalar/functor type.
  template <OperatorCapability Cap, typename OperatorType>
  __MATX_INLINE__ __MATX_HOST__ typename capability_attributes<Cap>::type
  get_operator_capability(const OperatorType& op) {
    VoidCapabilityType void_type{};
    return get_operator_capability<Cap>(op, void_type);
  }     


  // Helper function to get the query type associated with a capability
  // This defines the default aggregation logic for each capability.
  inline CapabilityQueryType get_query_type(OperatorCapability cap) {
    switch (cap) {
      case OperatorCapability::SUPPORTS_JIT:
        return CapabilityQueryType::AND_QUERY; // If any sub-operator supports JIT, the expression might be JIT-able.
      case OperatorCapability::ASYNC_LOADS_REQUESTED:
        return CapabilityQueryType::OR_QUERY; // If any sub-operator requires asynchronous loads, the expression might require asynchronous loads.
      case OperatorCapability::GLOBAL_KERNEL:
        return CapabilityQueryType::AND_QUERY; // If any sub-operator operates on a global level, the expression might operate on a global level.
      case OperatorCapability::ELEMENTS_PER_THREAD:
        return CapabilityQueryType::RANGE_QUERY; // The expression should use the range of elements per thread of its children.
      case OperatorCapability::SET_ELEMENTS_PER_THREAD:
        return CapabilityQueryType::AND_QUERY; // The expression should use the range of elements per thread of its children.
      case OperatorCapability::SET_GROUPS_PER_BLOCK:
        return CapabilityQueryType::AND_QUERY; // The expression should use the range of groups per block of its children.
      case OperatorCapability::GROUPS_PER_BLOCK:
        return CapabilityQueryType::RANGE_QUERY; // The expression should use the range of groups per block of its children.
      case OperatorCapability::ALIASED_MEMORY:
        return CapabilityQueryType::OR_QUERY; // The expression should use the aliased memory of its children.
      case OperatorCapability::MAX_EPT_VEC_LOAD:
        return CapabilityQueryType::MIN_QUERY; // The expression should use the minimum EPT for a vector load of its children.
      case OperatorCapability::JIT_CLASS_QUERY:
        return CapabilityQueryType::AND_QUERY; // The expression should succeed if all its children succeed.
      case OperatorCapability::JIT_TYPE_QUERY:
        return CapabilityQueryType::STR_CAT_QUERY; // The expression should use the concatenation of the capabilities of its children.
      case OperatorCapability::DYN_SHM_SIZE:
        return CapabilityQueryType::MAX_QUERY; // The expression should use the maximum dynamic shared memory size of its children.
      case OperatorCapability::BLOCK_DIM:
        return CapabilityQueryType::RANGE_QUERY; // The expression should use the minimum block size supported by all operators.
      case OperatorCapability::GENERATE_LTOIR:
        return CapabilityQueryType::AND_QUERY; // The expression should generate LTOIR code if all its children generate it.
      case OperatorCapability::PASS_THROUGH_THREADS:
        return CapabilityQueryType::OR_QUERY; // If ANY operator needs pass-through, all threads must call operator()
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
        MATX_IGNORE_WARNING_PUSH_GCC("-Wduplicated-branches")
        if (query_type == CapabilityQueryType::MIN_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::min_identity;
        } else if (query_type == CapabilityQueryType::MAX_QUERY) {
          children_aggregated_val = capability_attributes<Cap>::max_identity;
        } else {
          // Default identity for int if not MIN_QUERY or MAX_QUERY (e.g. if it was SUM_QUERY, identity would be 0)
          // This path needs clear definition if other query types are used for int.
          children_aggregated_val = capability_attributes<Cap>::default_value; // Fallback
        }
        MATX_IGNORE_WARNING_POP_GCC
      } else if constexpr (std::is_same_v<CapType, std::string>) {
        children_aggregated_val = capability_attributes<Cap>::default_value;
      } else {
        // Fallback for other types, should be defined in capability_attributes
        children_aggregated_val = capability_attributes<Cap>::default_value;
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

  template <OperatorCapability Cap, typename OpsTuple>
  __MATX_INLINE__ __MATX_HOST__ auto get_combined_ops_capability(const OpsTuple& ops_tuple) {
    return cuda::std::apply([](const auto&... ops) {
      return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(ops)...);
    }, ops_tuple);
  }

  template <OperatorCapability Cap, typename InType, typename OpsTuple>
  __MATX_INLINE__ __MATX_HOST__ auto get_combined_ops_capability(const InType &in, const OpsTuple& ops_tuple) {
    return cuda::std::apply([&in](const auto&... ops) {
      return combine_capabilities<Cap>(detail::get_operator_capability<Cap>(ops, in)...);
    }, ops_tuple);
  }     

#endif

} // namespace detail

  // Helper function to check if an operator supports JIT compilation.
  template <typename Op>
  __MATX_INLINE__ __MATX_HOST__ bool jit_supported(const Op &op) {
    return detail::get_operator_capability<detail::OperatorCapability::SUPPORTS_JIT>(op);
  }  
} // namespace matx 