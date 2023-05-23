# ////////////////////////////////////////////////////////////////////////////////
# // BSD 3-Clause License
# //
# // Copyright (c) 2021, NVIDIA Corporation
# // All rights reserved.
# //
# // Redistribution and use in source and binary forms, with or without
# // modification, are permitted provided that the following conditions are met:
# //
# // 1. Redistributions of source code must retain the above copyright notice, this
# //    list of conditions and the following disclaimer.
# //
# // 2. Redistributions in binary form must reproduce the above copyright notice,
# //    this list of conditions and the following disclaimer in the documentation
# //    and/or other materials provided with the distribution.
# //
# // 3. Neither the name of the copyright holder nor the names of its
# //    contributors may be used to endorse or promote products derived from
# //    this software without specific prior written permission.
# //
# // THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# // AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# // IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# // DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# // FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# // DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# // SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# // CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# // OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# // OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
# /////////////////////////////////////////////////////////////////////////////////
function(check_python_libs)
    foreach (pack IN LISTS ARGN)
        message(STATUS "checking python import module ${pack}")
        set(CMD "import ${pack}")

        execute_process(COMMAND ${Python3_EXECUTABLE} -c ${CMD} RESULT_VARIABLE EXIT_CODE OUTPUT_QUIET)
        if (NOT ${EXIT_CODE} EQUAL 0)
            message(FATAL_ERROR
                    "The ${pack} Python3 package is not installed. Please install it using the following command: \"pip3 install ${pack}\".")
        else()
            set(${pack}_PYTHON_PACKAGE 1 PARENT_SCOPE)
        endif()
    endforeach ()
endfunction()

function(check_optional_python_libs)
    foreach (pack IN LISTS ARGN)
        message(STATUS "checking python import module ${pack}")
        set(CMD "import ${pack}")

        execute_process(COMMAND ${Python3_EXECUTABLE} -c ${CMD} RESULT_VARIABLE EXIT_CODE OUTPUT_QUIET)
        if (NOT ${EXIT_CODE} EQUAL 0)
            message(STATUS
                    "The optional python package ${pack} package is not installed. Some unit tests and functionality may not work")
        else()
            set(${pack}_PYTHON_PACKAGE 1 PARENT_SCOPE)
        endif()
    endforeach ()
endfunction()

