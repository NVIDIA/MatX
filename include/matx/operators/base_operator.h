////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2021, NVIDIA Corporation
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

#include "matx/core/vector.h"
#include "matx/core/type_utils.h"
#include "matx/core/nvtx.h"
#include "matx/core/operator_utils.h"
#include "matx/core/capabilities.h"
#include "matx/core/error.h"
#include "matx/core/log.h"

namespace matx
{
  namespace detail {
    /**
     * @brief Helper to get memory range from a tensor/lvalue for aliasing check
     * 
     * @tparam T Type of the lvalue
     * @param lval The lvalue to get memory range from
     * @return AliasedMemoryQueryInput with start and end pointers
     */
    template <typename T>
    __MATX_INLINE__ __MATX_HOST__ AliasedMemoryQueryInput get_memory_range(const T& lval, bool is_prerun) {
      // Handle both direct tensor views and operators with lvalue capability (like reshape, slice, etc.)
      if constexpr ((is_tensor_view_v<T> || is_matx_op_lvalue<T>()) && T::Rank() > 0) {
        using value_type = typename T::value_type;
        
        // Get address of first element using operator()(0, 0, ...)
        auto get_first = [&lval]<size_t... Is>(cuda::std::index_sequence<Is...>) {
          return &(const_cast<T&>(lval)(static_cast<index_t>(Is*0)...));
        };
        auto* start = static_cast<void*>(const_cast<value_type*>(get_first(cuda::std::make_index_sequence<T::Rank()>{})));
        
        // Get address of last element using operator()(Size(0)-1, Size(1)-1, ...)
        auto get_last = [&lval]<size_t... Is>(cuda::std::index_sequence<Is...>) {
          return &(const_cast<T&>(lval)(static_cast<index_t>(lval.Size(Is)-1)...));
        };
        auto* end = static_cast<void*>(const_cast<value_type*>(get_last(cuda::std::make_index_sequence<T::Rank()>{})) + 1);
        
        return AliasedMemoryQueryInput{false, is_prerun, start, end};
      }
      else {
        // For non-tensor types or rank 0, return null pointers (no aliasing checks needed)
        return AliasedMemoryQueryInput{false, is_prerun, nullptr, nullptr};
      }
    }

    /**
     * @brief Check if RHS operator aliases with LHS memory range
     * 
     * @tparam RHS Type of the right-hand side operator
     * @tparam LHS Type of the left-hand side operator
     * @param rhs The right-hand side operator
     * @param lhs The left-hand side operator
     * @param is_prerun Whether we are in prerun mode
     * @return true if memory is aliased, false otherwise
     */
    template <typename LHS, typename RHS>
    __MATX_INLINE__ __MATX_HOST__ bool check_aliased_memory([[maybe_unused]] const LHS& lhs, [[maybe_unused]] const RHS& rhs, [[maybe_unused]] bool is_prerun) {
#ifdef MATX_EN_UNSAFE_ALIAS_DETECTION      
      auto mem_range = get_memory_range(lhs, is_prerun);
      
      // If we got null pointers, no aliasing is possible
      if (mem_range.start_ptr == nullptr || mem_range.end_ptr == nullptr) {
        return false;
      }
      
      // Query the RHS operator to see if it uses aliased memory
      return get_operator_capability<OperatorCapability::ALIASED_MEMORY>(rhs, mem_range);
#else
      return false;
#endif
    }

    /**
     * @brief Check if mtie operation has aliased memory between LHS elements and RHS
     * 
     * @tparam MtieType Type of the mtie object
     * @param mtie_obj The mtie object to check
     */
    template <typename MtieType>
    __MATX_INLINE__ __MATX_HOST__ void check_mtie_aliased_memory([[maybe_unused]] const MtieType& mtie_obj) {
#ifdef MATX_EN_UNSAFE_ALIAS_DETECTION      
      constexpr size_t tuple_size = cuda::std::tuple_size_v<decltype(mtie_obj.ts_)>;
      
      // Get the RHS (last element of the tuple)
      auto& rhs = cuda::std::get<tuple_size - 1>(mtie_obj.ts_);
      
      // Check each LHS element (all elements except the last one)
      bool has_alias = false;
      
      // Use C++20 template lambda with index_sequence
      [&]<size_t... Is>(cuda::std::index_sequence<Is...>) {
        ([&] {
          if constexpr (Is < tuple_size - 1) {
            auto& lhs_elem = cuda::std::get<Is>(mtie_obj.ts_);
            if (check_aliased_memory(lhs_elem, rhs, false)) {
              has_alias = true;
            }
          }
        }(), ...);
      }(cuda::std::make_index_sequence<tuple_size>{});
      
      if (has_alias) {
        MATX_THROW(matxInvalidParameter, "Aliased memory detected: One or more LHS tensors overlap with RHS memory");
      }
#endif      
    }
  } // namespace detail

  /**
   * @brief Provides a base class with functions common to all operators
   * 
   * @tparam T Type of operator
   */
  template <typename T>
    class BaseOp
    {
      public:
        using matxop = bool;  ///< Is a MatX custom operator
        using value_type = T; ///< Value type for type traits

        //static constexpr uint64_t unique_id_ = detail::fnv1a_64(detail::get_type_name<T>());
	      __MATX_INLINE__ std::string str() const { return "BaseOp"; }

      private:
        // Helper template to safely check if T is a matx_set_op with transform and tensor_view
        template<typename U>
        static constexpr bool is_matx_set_op_with_transform_and_tensor_view() {
          if constexpr (is_matx_set_op<U>()) {
            return is_matx_transform_op<typename U::op_type>() && is_tensor_view_v<typename U::tensor_type>;
          } else {
            return false;
          }
        }

      public:
        /**
         * @brief Launch work in an arbitrary executor
         * 
         * @tparam Ex Executor type
         * @param ex Executor
         */
        template <typename Ex>
        __MATX_INLINE__ void run (Ex &&ex) {
          MATX_NVTX_START(static_cast<T *>(this)->str(), matx::MATX_NVTX_LOG_API)
          static_assert(is_executor_t<Ex>(), "Ex must be a MatX executor type");

          auto tp = static_cast<T *>(this);

          // For JIT CUDA executors, we don't need to run PreRun/PostRun since there's no async allocation.
          if constexpr (is_jit_cuda_executor_t<Ex>()) {
            ex.Exec(*tp);
          }
          else if constexpr (is_mtie<T>() ) {
            detail::check_mtie_aliased_memory(*tp);
            tp->Exec(ex);
          }
          else if constexpr (is_matx_set_op<T>()) {
            if constexpr (is_matx_transform_op<typename T::op_type>() && is_tensor_view_v<typename T::tensor_type>) {
              // If we're doing a simple set operation from a transform we take a shorcut to avoid the extra
              // async allocation we'd normally have to do   
              if (!can_alias<decltype(tp->get_rhs())>() && detail::check_aliased_memory(tp->get_lhs(), tp->get_rhs(), false)) {
                MATX_THROW(matxInvalidParameter, "Possible aliased memory detected: LHS and RHS memory ranges overlap");
              }

              tp->TransformExec(tp->Shape(), ex);
            }
            else if constexpr (is_tensor_view_v<typename T::tensor_type> && is_tensor_view_v<typename T::op_type> && is_cuda_executor_v<Ex>) {
              // If we are doing a tensor to tensor assignment we should prefer cudaMemcpyAsync instead of a kernel
              if (detail::check_aliased_memory(tp->get_lhs(), tp->get_rhs(), true)) {
                MATX_THROW(matxInvalidParameter, "Possible aliased memory detected: LHS and RHS memory ranges overlap");
              }

              if (tp->get_lhs().IsContiguous() && tp->get_rhs().IsContiguous() && tp->get_lhs().Rank() == tp->get_rhs().Rank()) {
                MATX_ASSERT_STR(tp->get_lhs().Bytes() >= tp->get_rhs().Bytes(), matxInvalidSize, "LHS tensor is smaller than RHS tensor in assignment");
                MATX_LOG_TRACE("Copying {} bytes from {} to {} using cudaMemcpyAsync",
                  tp->get_lhs().Bytes(), reinterpret_cast<void*>(tp->get_rhs().Data()), reinterpret_cast<void*>(tp->get_lhs().Data()));
                cudaMemcpyAsync(reinterpret_cast<void*>(tp->get_lhs().Data()),
                                reinterpret_cast<void*>(tp->get_rhs().Data()),
                                tp->get_rhs().Bytes(),
                                cudaMemcpyDefault,
                                ex.getStream());
              }
              else {
                MATX_LOG_TRACE("Copying {} bytes from {} to {} using kernel",
                  tp->get_lhs().Bytes(), reinterpret_cast<void*>(tp->get_rhs().Data()), reinterpret_cast<void*>(tp->get_lhs().Data()));
                ex.Exec(*tp);
              }
            }
            else {
              if (detail::check_aliased_memory(tp->get_lhs(), tp->get_rhs(), true)) {
                MATX_THROW(matxInvalidParameter, "Possible aliased memory detected: LHS and RHS memory ranges overlap");
              }

              if constexpr (is_matx_op<T>()) {
                tp->PreRun(tp->Shape(), ex);
              }
              
              ex.Exec(*tp);

              if constexpr (is_matx_op<T>()) {
                tp->PostRun(tp->Shape(), ex);
              }              
            }
          }
          else {
            // These can be operators like custom operators that take an output as a parameter. In those cases we cannot 
            // check for aliasing since we don't control the operator.

            if constexpr (is_matx_op<T>()) {
              tp->PreRun(tp->Shape(), ex);
            }

            ex.Exec(*tp);

            if constexpr (is_matx_op<T>()) {
              tp->PostRun(tp->Shape(), ex);
            }
          }  
        }

        /**
         * @brief Launch kernel in a GPU stream
         * 
         * @param stream CUDA stream
         */
        __MATX_INLINE__ void run(cudaStream_t stream = 0)
        {
          MATX_NVTX_START(detail::get_type_str(*static_cast<T *>(this)), matx::MATX_NVTX_LOG_API)
          run(cudaExecutor{stream, false});
        }

        /**
         * @brief Launch work in a GPU stream and record an event
         * 
         * @param ev CUDA event
         * @param stream CUDA stream
         */
        __MATX_INLINE__ void run(cudaEvent_t ev, cudaStream_t stream = 0)
        {
          MATX_NVTX_START(static_cast<T *>(this)->str(), matx::MATX_NVTX_LOG_API)

          run(cudaExecutor{stream, false});
          cudaEventRecord(ev, stream);
        }

        /**
         * @brief Function to run before the executor
         * 
         * @tparam ShapeType Type of shape
         * @tparam Executor Executor type
         * @param shape Shape
         * @param ex Executor
         */
        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
        }

        /**
         * @brief Function to run before the executor
         * 
         * @tparam ShapeType Type of shape
         * @tparam Executor Executor type
         * @param shape Shape
         * @param ex Executor
         */
        template <typename ShapeType, typename Executor>
        __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, [[maybe_unused]] Executor &&ex) const noexcept
        {
        }


        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ auto Shape() const {
          if constexpr (T::Rank() == 0) {
            return cuda::std::array<index_t, 0> {};
          }
          else {
            cuda::std::array<index_t, static_cast<size_t>(T::Rank())> sizes_;

            for(int i = 0 ; i < T::Rank(); i++) {
              sizes_[i] = static_cast<const T*>(this)->Size(i);
            }
            return sizes_;
          }
        }

        __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t TotalSize() const {

          index_t size = 1;
          for(int i = 0 ; i < T::Rank(); i++) {
            size *= static_cast<const T*>(this)->Size(i);
          }
          return size;
        }


        /* This must be in derived class.  Copy paste line below to derived case if it is an lvalue
           template<typename R> __MATX_INLINE__ auto operator=(const R &rhs) { return set(*reinterpret_cast<T*>(this), rhs); }
           */
    };

  /**
   * @brief Pprovides a base class for reducing the boilerplate code needed
   * when defining an operator. This is particularly useful in user code with
   * many custom operators that don't want to repeat the Size and Rank functions.
   * 
   * @tparam T Type of operator
   * @tparam RankOp Type of operator providing rank information
   */
  template <typename T, typename RankOp>
    class BaseOpCustom : public BaseOp<T>
  {
    public:
      using matxop = bool;  ///< Type trait to indicate this is an operator
      using value_type = T; ///< Type trait of operator type
      cuda::std::array<index_t, RankOp::Rank()> size_; ///< Size of each dimension

      BaseOpCustom() = delete;

      /**
       * @brief Construct a new Base Op Custom object
       * 
       * @param size Size of each dimension
       */
      __MATX_INLINE__ BaseOpCustom(const cuda::std::array<index_t, RankOp::Rank()> &size) :
        size_(size) {}

      /**
       * @brief Return rank of operator
       * 
       * @return Operator rank
       */
      static __MATX_INLINE__ constexpr int32_t Rank()
      {
        return RankOp::Rank();
      }  

      /**
       * @brief Return size of operator on a given dimension
       * 
       * @return Operator size on dimension dim
       */
      index_t __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ Size(int dim) const
      {
        return size_[dim];
      }
  };

} // end namespace matx
