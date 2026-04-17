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
#include <cstring>
#include <stdexcept>
#include <string>
#include <matx.h>
#include <matx/core/dlpack.h>

namespace py = pybind11;


/**
 * @brief Import a Python DLPack capsule into a MatX tensor with ownership transfer.
 *
 * This helper consumes the capsule exactly once by renaming it to the
 * corresponding used state (`used_dltensor` or `used_dltensor_versioned`)
 * before calling MatX's pointer-owning `make_tensor` overload. After this call,
 * the capsule must not be reused.
 *
 * @tparam TensorType Destination MatX tensor type
 * @param tensor Destination tensor to be shallow-populated
 * @param dlpack_capsule Python capsule named `dltensor` or `dltensor_versioned`
 *
 * @throws py::value_error If capsule name/pointer is invalid
 * @throws std::runtime_error If the capsule cannot be marked as consumed
 */
template <typename TensorType>
void make_tensor_from_capsule(TensorType &tensor, py::capsule dlpack_capsule)
{
  const char* capsule_name = dlpack_capsule.name();
  if (capsule_name == nullptr) {
    throw py::value_error("DLPack capsule name is null");
  }

  if (strcmp(capsule_name, "dltensor") == 0) {
    /* Consume the legacy DLPack capsule */
    auto *managed = static_cast<DLManagedTensor*>(dlpack_capsule.get_pointer());
    if (managed == nullptr) {
      throw py::value_error("Legacy DLPack capsule pointer is null");
    }
    /* Mark the capsule as consumed */
    if (PyCapsule_SetName(dlpack_capsule.ptr(), "used_dltensor") != 0) {
      PyErr_Clear();
      throw std::runtime_error("Failed to mark DLPack capsule as consumed");
    }
    /* Create the MatX tensor, consuming the capsule */
    matx::make_tensor(tensor, managed);
    return;
  }

  if (strcmp(capsule_name, "dltensor_versioned") == 0) {
    /* Consume the versioned DLPack capsule */
    auto *managed = static_cast<DLManagedTensorVersioned*>(dlpack_capsule.get_pointer());
    if (managed == nullptr) {
      throw py::value_error("Versioned DLPack capsule pointer is null");
    }
    /* Mark the capsule as consumed */
    if (PyCapsule_SetName(dlpack_capsule.ptr(), "used_dltensor_versioned") != 0) {
      PyErr_Clear();
      throw std::runtime_error("Failed to mark DLPack capsule as consumed");
    }
    /* Create the MatX tensor, consuming the capsule */
    matx::make_tensor(tensor, managed);
    return;
  }

  /* Capsule name is unsupported */
  throw py::value_error(std::string("Unsupported DLPack capsule name: ") + capsule_name);
}

template<typename T, int RANK>
void print(py::capsule dlpack_capsule)
{
  matx::tensor_t<T, RANK> a;
  make_tensor_from_capsule(a, dlpack_capsule);

  matx::print(a);
}

template<typename T, int RANK>
void python_print(py::capsule dlpack_capsule)
{
  // Create a MatX tensor from the DLPack capsule
  matx::tensor_t<T, RANK> a;
  make_tensor_from_capsule(a, dlpack_capsule);

  auto pb = matx::detail::MatXPybind{};
  // Convert the MatX tensor to a DLPack capsule
  auto out = a.ToDlPack();
  py::capsule out_capsule(out, "dltensor", [](PyObject *capsule) {
    const char *name = PyCapsule_GetName(capsule);
    if (name != nullptr && strcmp(name, "used_dltensor") == 0) {
      return;
    }

    auto *managed = static_cast<DLManagedTensor *>(PyCapsule_GetPointer(capsule, "dltensor"));
    if (managed != nullptr && managed->deleter != nullptr) {
      managed->deleter(managed);
    }
  });

  // Example use calling python code from C++
  auto mypythonlib = pybind11::module_::import("mypythonlib");
  mypythonlib.attr("python_print")(out_capsule);
}

template<typename T, int RANK>
void add(py::capsule capsule_c, py::capsule capsule_a, py::capsule capsule_b, int64_t stream = 0)
{
  matx::tensor_t<T, RANK> c;
  matx::tensor_t<T, RANK> a;
  matx::tensor_t<T, RANK> b;

  make_tensor_from_capsule(c, capsule_c);
  make_tensor_from_capsule(a, capsule_a);
  make_tensor_from_capsule(b, capsule_b);

  matx::cudaExecutor exec{reinterpret_cast<cudaStream_t>(stream)};
  (c = a + b).run(exec);
}

PYBIND11_MODULE(matxutil, m) {
    m.def("print_float_2D", &print<float,2>, "Prints a float32 2D tensor", py::arg("dlpack_capsule"));
    m.def("python_print_float_2D", &python_print<float,2>, "Example C++ function that calls python code", py::arg("dlpack_capsule"));
    m.def("add_float_2D",
          &add<float,2>,
          "Add two float32 2D tensors together",
          py::arg("c"),
          py::arg("a"),
          py::arg("b"),
          py::arg("stream") = 0);
}