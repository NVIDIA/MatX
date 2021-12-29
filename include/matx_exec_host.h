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
#include <type_traits>

#include "matx_error.h"
#include "matx_get_grid_dims.h"

namespace matx 
{
/**
 * @brief Executor for running an operator on a single host thread
 * 
 */
class SingleThreadHostExecutor {
  public:
    using matx_executor = bool; ///< Type trait indicating this is an executor
    
    /**
     * @brief Execute an operator
     * 
     * @tparam Op Operator type
     * @param op Operator to execute
     */
    template <typename Op>
    void Exec(Op &op) const noexcept {
      if constexpr (op.Rank() == 0) {
        op();
      }
      else if constexpr (op.Rank() == 1) {
        index_t size0 = op.Size(0);
        for (index_t idx = 0; idx < size0; idx++) {
          op(idx);
        }
      }
      else if constexpr (op.Rank() == 2) {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);

        for (index_t idx = 0; idx < size0; idx++) {
          for (index_t idy = 0; idy < size1; idy++) {
            op(idx, idy);
          }
        }
      }
      else if constexpr (op.Rank() == 3) {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);
        index_t size2 = op.Size(2);

        for (index_t idx = 0; idx < size0; idx++) {
          for (index_t idy = 0; idy < size1; idy++) {
            for (index_t idz = 0; idz < size2; idz++) {
              op(idx, idy, idz);
            }
          }
        }
      }
      else {
        index_t size0 = op.Size(0);
        index_t size1 = op.Size(1);
        index_t size2 = op.Size(2);
        index_t size3 = op.Size(3);

        for (index_t idx = 0; idx < size0; idx++) {
          for (index_t idy = 0; idy < size1; idy++) {
            for (index_t idz = 0; idz < size2; idz++) {
              for (index_t idw = 0; idw < size3; idw++) {
                op(idx, idy, idz, idw);
              }
            }
          }
        }
      }        
    }
};

}