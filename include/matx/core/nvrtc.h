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

#include <cuda.h>

#define JITIFY_ENABLE_EMBEDDED_FILES 1
#define JITIFY_IGNORE_NOT_TRIVIALLY_COPYABLE_ARGS 1
#include "matx/core/jitify2.hpp"
#include "matx/executors/kernel.h"
#include "matx/core/type_utils.h"


namespace matx {


template <typename Op>
auto nvrtc_compile_and_run(const std::string &src, const std::string &name, Op op, index_t size0) {
  dim3 grid(1), block(1);

  // jitify2::PreprocessedProgram preprog =
  //     jitify2::Program(name, src)
  //         ->preprocess({"-std=c++17"});
  // if (!preprog) {
  //   // The call failed, we can access the error.
  //   std::cerr << preprog.error() << std::endl;
  //   // This will either throw an exception or terminate the application.
  //   *preprog;
  // } else {
  //   // The call succeeded, we can access the data object.
  //   jitify2::PreprocessedProgramData preprog_data = *preprog;
  //   // Or we can directly call a method on the data object.
  //   //jitify2::CompiledProgram compiled = preprog->compile("matx::detail::matxOpT1Kernel");
  //   //This will throw (or terminate) if any of the chained methods fails.
  //   preprog->compile("matx::detail::matxOpT1Kernel")
  //       ->link()
  //       ->load()
  //       ->get_kernel("matx::detail::matxOpT1Kernel")
  //       ->configure(1, 1)
  //       ->launch(op, size0);
  // }
      jitify2::Program(name, src)
          // Preprocess source code and load all included headers.
          ->preprocess({"-DJITIFY", "-I/repro/tmp/MatX/include", "-I/usr/local/cuda/include", "-no-preinclude-workarounds",
                      "-no-system-headers-workaround",
                      "-arch=sm_80","-std=c++17"})
          // Compile, link, and load the program, and obtain the loaded kernel.
          ->get_kernel("matx::detail::matxOpT1Kernel<Op>")
          // Configure the kernel launch.
          ->configure(grid, block)
          // Launch the kernel.
          ->launch(op, size0);
}

}
