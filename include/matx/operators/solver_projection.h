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

#pragma once

#include "matx/core/type_utils.h"
#include "matx/operators/base_operator.h"

namespace matx {
namespace detail {

template <int RANK, typename Op>
__MATX_INLINE__ cuda::std::array<index_t, RANK> SolverShapeFromInput(const Op &op)
{
  cuda::std::array<index_t, RANK> shape{};
  for (int i = 0; i < RANK; i++) {
    shape[i] = op.Size(i);
  }
  return shape;
}

template <int RANK>
__MATX_INLINE__ cuda::std::array<index_t, RANK - 1> SolverVectorShapeFromMatrixShape(
  const cuda::std::array<index_t, RANK> &shape)
{
  cuda::std::array<index_t, RANK - 1> vec_shape{};
  for (int i = 0; i < RANK - 2; i++) {
    vec_shape[i] = shape[i];
  }
  vec_shape[RANK - 2] = shape[RANK - 2] < shape[RANK - 1] ? shape[RANK - 2] : shape[RANK - 1];
  return vec_shape;
}

template <typename State, int Component, typename TensorType>
class SolverProjectionOp : public BaseOp<SolverProjectionOp<State, Component, TensorType>>
{
  private:
    State *state_;
    cuda::std::array<index_t, TensorType::Rank()> shape_;
    mutable TensorType tensor_;
    const char *name_;

  public:
    using matxop = bool;
    using value_type = typename TensorType::value_type;

    __MATX_INLINE__ SolverProjectionOp(State *state,
                                       const cuda::std::array<index_t, TensorType::Rank()> &shape,
                                       const char *name) :
      state_(state), shape_(shape), name_(name)
    {
    }

    __MATX_INLINE__ std::string str() const { return name_; }

    template <typename... Is>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const noexcept
    {
      return tensor_(indices...);
    }

    template <typename CapType, typename... Is>
    __MATX_INLINE__ __MATX_DEVICE__ __MATX_HOST__ auto operator()(Is... indices) const noexcept
    {
      return tensor_.template operator()<CapType>(indices...);
    }

    static __MATX_INLINE__ constexpr __MATX_HOST__ __MATX_DEVICE__ int32_t Rank()
    {
      return TensorType::Rank();
    }

    constexpr __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ index_t Size(int dim) const
    {
      return shape_[dim];
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PreRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
    {
      state_->Materialize(std::forward<Executor>(ex));
      tensor_ = state_->template Tensor<Component>();
    }

    template <typename ShapeType, typename Executor>
    __MATX_INLINE__ void PostRun([[maybe_unused]] ShapeType &&shape, Executor &&ex) const noexcept
    {
      state_->Release(std::forward<Executor>(ex));
    }

    template <OperatorCapability Cap, typename InType>
    __MATX_INLINE__ __MATX_HOST__ auto get_capability([[maybe_unused]] InType &in) const
    {
      if constexpr (Cap == OperatorCapability::SUPPORTS_JIT) {
        return false;
      }
      else if constexpr (Cap == OperatorCapability::JIT_TYPE_QUERY) {
        return std::string(name_);
      }
      else {
        return capability_attributes<Cap>::default_value;
      }
    }
};

} // end namespace detail
} // end namespace matx
