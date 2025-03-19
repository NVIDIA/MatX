////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2024, NVIDIA Corporation
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

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdio.h>
#include <matx.h>
#include <matx/core/dlpack.h>

namespace py = pybind11;

const char* get_capsule_name(py::capsule capsule)
{
  return capsule.name();
}

typedef DLManagedTensor* PTR_DLManagedTensor;
int attempt_unpack_dlpack(py::capsule dlpack_capsule, PTR_DLManagedTensor& p_dlpack)
{
  const char* capsule_name = dlpack_capsule.name();

  if (strncmp(capsule_name,"dltensor",8) != 0)
  {
    fprintf(stderr,"capsule_name %s\n",capsule_name);
    return -1;
  }

  p_dlpack = static_cast<PTR_DLManagedTensor>(dlpack_capsule.get_pointer());

  if (p_dlpack == nullptr) {
    fprintf(stderr,"p_dlpack == nullptr\n");
    return -2;
  }

  return 0;
}

int check_dlpack_status(py::capsule dlpack_capsule)
{
  PTR_DLManagedTensor unused;
  return attempt_unpack_dlpack(dlpack_capsule, unused);
}

const char* dlpack_device_type_to_string(DLDeviceType device_type)
{
  switch(device_type)
  {
    case kDLCPU: return "kDLCPU";
    case kDLCUDA: return "kDLCUDA";
    case kDLCUDAHost: return "kDLCUDAHost";
    case kDLOpenCL: return "kDLOpenCL";
    case kDLVulkan: return "kDLVulkan";
    case kDLMetal: return "kDLMetal";
    case kDLVPI: return "kDLVPI";
    case kDLROCM: return "kDLROCM";
    case kDLROCMHost: return "kDLROCMHost";
    case kDLExtDev: return "kDLExtDev";
    case kDLCUDAManaged: return "kDLCUDAManaged";
    case kDLOneAPI: return "kDLOneAPI";
    case kDLWebGPU: return "kDLWebGPU";
    case kDLHexagon: return "kDLHexagon";
    default: return "Unknown DLDeviceType";
  }
}

const char* dlpack_code_to_string(uint8_t code)
{
  switch(code)
  {
    case kDLInt: return "kDLInt";
    case kDLUInt: return "kDLUInt";
    case kDLFloat: return "kDLFloat";
    case kDLOpaqueHandle: return "kDLOpaqueHandle";
    case kDLBfloat: return "kDLBfloat";
    case kDLComplex: return "kDLComplex";
    case kDLBool: return "kDLBool";
    default: return "Unknown DLDataTypeCode";
  }
}

void print_dlpack_info(py::capsule dlpack_capsule) {
  PTR_DLManagedTensor p_tensor;
  if (attempt_unpack_dlpack(dlpack_capsule, p_tensor))
  {
    fprintf(stderr,"Error: capsule not valid dlpack");
    return;
  }

  printf("  data: %p\n",p_tensor->dl_tensor.data);
  printf("  device: device_type %s, device_id %d\n",
    dlpack_device_type_to_string(p_tensor->dl_tensor.device.device_type),
    p_tensor->dl_tensor.device.device_id
  );
  printf("  ndim: %d\n",p_tensor->dl_tensor.ndim);
  printf("  dtype: code %s, bits %u, lanes %u\n",
    dlpack_code_to_string(p_tensor->dl_tensor.dtype.code),
    p_tensor->dl_tensor.dtype.bits,
    p_tensor->dl_tensor.dtype.lanes
  );
  printf("  shape: ");
  for (int k=0; k<p_tensor->dl_tensor.ndim; k++)
  {
    printf("%ld, ",p_tensor->dl_tensor.shape[k]);
  }
  printf("\n");
  printf("  strides: ");
  for (int k=0; k<p_tensor->dl_tensor.ndim; k++)
  {
    printf("%ld, ",p_tensor->dl_tensor.strides[k]);
  }
  printf("\n");
  printf("  byte_offset: %lu\n",p_tensor->dl_tensor.byte_offset);
}

template<typename T, int RANK>
void print(py::capsule dlpack_capsule)
{
  PTR_DLManagedTensor p_tensor;
  if (attempt_unpack_dlpack(dlpack_capsule, p_tensor))
  {
    fprintf(stderr,"Error: capsule not valid dlpack");
    return;
  }

  matx::tensor_t<T, RANK> a;
  matx::make_tensor(a, *p_tensor);
  matx::print(a);
}

void call_python_example(py::capsule dlpack_capsule)
{
  PTR_DLManagedTensor p_tensor;
  if (attempt_unpack_dlpack(dlpack_capsule, p_tensor))
  {
    fprintf(stderr,"Error: capsule not valid dlpack");
    return;
  }

  matx::tensor_t<float, 2> a;
  matx::make_tensor(a, *p_tensor);

  auto pb = matx::detail::MatXPybind{};

  // Example use of python's print
  pybind11::print("  Example use of python's print function from C++: ", 1, 2.0, "three");
  pybind11::print("  The dlpack_capsule is a ", dlpack_capsule);

  auto mypythonlib = pybind11::module_::import("mypythonlib");
  mypythonlib.attr("my_func")(dlpack_capsule);
}

template<typename T, int RANK>
void add(py::capsule capsule_c, py::capsule capsule_a, py::capsule capsule_b, int64_t stream = 0)
{
  PTR_DLManagedTensor p_tensor_c;
  PTR_DLManagedTensor p_tensor_a;
  PTR_DLManagedTensor p_tensor_b;

  // TODO these should matx throw
  if (attempt_unpack_dlpack(capsule_c, p_tensor_c))
  {
    fprintf(stderr,"Error: capsule c not valid dlpack\n");
    return;
  }

  if (attempt_unpack_dlpack(capsule_a, p_tensor_a))
  {
    fprintf(stderr,"Error: capsule a not valid dlpack\n");
    return;
  }

  if (attempt_unpack_dlpack(capsule_b, p_tensor_b))
  {
    fprintf(stderr,"Error: capsule b not valid dlpack\n");
    return;
  }

  matx::tensor_t<T, RANK> c;
  matx::tensor_t<T, RANK> a;
  matx::tensor_t<T, RANK> b;
  matx::make_tensor(c, *p_tensor_c);
  matx::make_tensor(a, *p_tensor_a);
  matx::make_tensor(b, *p_tensor_b);

  matx::cudaExecutor exec{reinterpret_cast<cudaStream_t>(stream)};
  (c = a + b).run(exec);
}

PYBIND11_MODULE(matxutil, m) {
    m.def("get_capsule_name", &get_capsule_name, "Returns PyCapsule name");
    m.def("print_dlpack_info", &print_dlpack_info, "Print the DLPack tensor metadata");
    m.def("check_dlpack_status", &check_dlpack_status, "Returns 0 if DLPack is valid, negative error code otherwise");
    m.def("print_float_2D", &print<float,2>, "Prints a float32 2D tensor");
    m.def("call_python_example", &call_python_example, "Example C++ function that calls python code");
    m.def("add_float_2D",
          &add<float,2>,
          "Add two float32 2D tensors together",
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("stream") = 0);
}