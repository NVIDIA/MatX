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

// adapted from https://idlebox.net/2008/0901-stacktrace-demangled/ and licensed
// under WTFPL v2.0
#pragma once

#ifdef _WIN32
#else
#include <cxxabi.h>
#include <dlfcn.h>
#include <execinfo.h>
#include <stdlib.h>
#include <unistd.h>
#endif

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

namespace matx {
namespace detail{

/** Print a demangled stack backtrace of the caller function to FILE* out. */
static inline void printStackTrace(std::ostream &eout = std::cerr,
                                   unsigned int max_frames = 63)
{
#ifdef _WIN32
  // TODO add code for windows stack trace
#else
  std::stringstream out;
  // storage array for stack trace address data
  void *addrlist[max_frames + 1];
  // retrieve current stack addresses
  int addrlen =
      backtrace(addrlist, static_cast<int>(sizeof(addrlist) / sizeof(void *)));

  if (addrlen == 0) {
    out << "  <empty, possibly corrupt>\n";
    return;
  }

  // resolve addresses into strings containing "filename(function+address)",
  // this array must be free()-ed
  char **symbollist = backtrace_symbols(addrlist, addrlen);
  // allocate string which will be filled with the demangled function name
  size_t funcnamesize = 256;
  char *funcname = (char *)malloc(funcnamesize);

  // iterate over the returned symbol lines. skip the first, it is the
  // address of this function.
  for (int i = 1; i < addrlen; i++) {
    char *begin_name = 0, *begin_offset = 0, *end_offset = 0;

    // find parentheses and +address offset surrounding the mangled name:
    // ./module(function+0x15c) [0x8048a6d]
    for (char *p = symbollist[i]; *p; ++p) {
      if (*p == '(') {
        begin_name = p;
      }
      else if (*p == '+') {
        begin_offset = p;
      }
      else if (*p == ')' && begin_offset) {
        end_offset = p;
        break;
      }
    }

    if (begin_name && begin_offset && end_offset && begin_name < begin_offset) {
      *begin_name++ = '\0';
      *begin_offset++ = '\0';
      *end_offset = '\0';
      // mangled name is now in [begin_name, begin_offset) and caller
      // offset in [begin_offset, end_offset). now apply
      // __cxa_demangle():
      int status;
      char *ret =
          abi::__cxa_demangle(begin_name, funcname, &funcnamesize, &status);

      if (status == 0) {
        funcname = ret; // use possibly realloc()-ed string
        out << " " << symbollist[i] << " : " << funcname << "+" << begin_offset
            << "\n";
      }
      else {
        // demangling failed. Output function name as a C function with
        // no arguments.
        out << " " << symbollist[i] << " : " << begin_name << "()+"
            << begin_offset << "\n";
      }
    }
    else {
      // couldn't parse the line? print the whole line.
      out << " " << symbollist[i] << "\n";
    }
  }

  eout << out.str();
  // error_output(out.str().c_str(),out.str().size());
  free(funcname);
  free(symbollist);
  // printf("PID of failing process: %d\n",getpid());
  // while(1);
#endif
}

}
} // end namespace matx
