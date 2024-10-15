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


#include "matx/core/type_utils.h"
#include "matx/core/nvtx.h"
#include "matx/core/operator_utils.h"

namespace matx
{

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

	      __MATX_INLINE__ std::string str() const { return "BaseOp"; }

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

          // If we're doing a simple set operation from a transform we take a shorcut to avoid the extra
          // async allocation we'd normally have to do
          if constexpr (is_mtie<T>() ) {
            tp->Exec(ex);
          }          
          else if constexpr (is_matx_set_op<T>()) {
            if constexpr (static_cast<const T *>(this)->IsTransformSet()) {
              tp->TransformExec(tp->Shape(), ex);
            }
            else {
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
          run(cudaExecutor{stream});
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

          run(cudaExecutor{stream});
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

        template <detail::VecWidth InWidth, detail::VecWidth OutWidth, typename Idx>
        __MATX_INLINE__ auto ApplyVec(const Idx &idx) {
          auto apply_op = [this](auto... args) {
              return static_cast<T *>(this)->template operator()<InWidth, OutWidth>(args...);
          };

          return cuda::std::apply(apply_op, idx);
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
