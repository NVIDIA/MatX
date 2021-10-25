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

#include "matx_pybind.h"
#include "matx_tensor.h"
#include <cstdio>
#include <iterator>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
// #include <cudf/table/table.hpp>
// #include <cudf/io/csv.hpp>
// #include <cudf/types.hpp>

#include "matx_error.h"

#pragma once

namespace matx {
namespace io {

using namespace pybind11::literals;

// cuDF version. Leave this out for now since it's more complex than we need
// /**
//  * Read a CSV file into a tensor view
//  *
//  * CSV files typically have a large number of rows with columns representing
//  individual features/fields.
//  * Since these columns are typically accessed together, this function will
//  take CSV columns and
//  * lay them out as rows in the tensor view. This allows for contiguous memory
//  accesses on the data
//  * within a field. Currently this routine will allocate twice the amount of
//  memory that's needed
//  * to hold the data since the cuDF function used for reading will put the
//  data in its own format, which
//  * we have to convert ours to.
//  **/
// template <typename T, int RANK>
// void ReadCSV(tensor_t<T, RANK> &t, [[maybe_unused]] const std::string
// fname, [[maybe_unused]] bool header=true) {
//   if (!t.IsLinear()) {
//     MATX_THROW(matxInvalidParameter, "Tensor reading into a CSV must have
//     linear layout!");
//   }

//   if (RANK != 1 && RANK != 2) {
//     MATX_THROW(matxInvalidDim, "CSV reading limited to tensors of rank 2 and
//     below");
//   }

//   index_t inner_dim = (RANK == 1) ? 1 ? t.Size(RANK-2)

//   std::vector<cudf::data_type> types;
//   std::fill_n(std::back_inserter(types), inner_dim,
//   cudf::data_type{cudf::type_to_id<T>()});

//   // Create options for reading in. Currently header and column types are
//   supported cudf::io::csv_reader_options in_opts =
//     cudf::io::csv_reader_options::builder(cudf::io::source_info{fname}).header(header
//     ? 0 : -1).dtypes(types);

//   auto result = cudf::io::read_csv(in_opts);

//   // Swap the order of rows/cols since CSV is more common to have lots of
//   rows with fewer columns, and
//   // since our memory order is row-major, we want to make a contiguous copy
//   MATX_ASSERT(result.tbl->num_rows() <= t.Size(1), matxInvalidSize);
//   MATX_ASSERT(result.tbl->num_columns() <= t.Size(0), matxInvalidSize);

//   for (index_t col = 0; col < result.tbl->num_columns(); col++) {
//     auto const& col_view = result.tbl->view();
//     printf("%d\n", col_view.column(col).size());

//     // cuDF saves this in device memory
//     cudaMemcpy(&t(col, 0), col_view.column(col).begin<T>(),
//     col_view.column(col).size() * sizeof(T), cudaMemcpyDeviceToDevice);
//   }

//   t.Print();
// }

/**
 * Read a CSV file into a tensor view
 *
 * CSVs are currently read in using the Python interpreter through pybind11.
 *This has a startup performance hit, but CSV reading is intended to be a
 *slow-path function, so this is not a critical component to speed up. Currently
 *1D and 2D tensors are supported only.
 **/
template <typename T, int RANK>
void ReadCSV(tensor_t<T, RANK> &t, const std::string fname,
             const std::string delimiter, bool header = true)
{
  if (RANK != 1 && RANK != 2) {
    MATX_THROW(matxInvalidDim,
               "CSV reading limited to tensors of rank 1 and 2");
  }

  std::unique_ptr<MatXPybind> pb;
  auto np = py::module_::import("numpy");
  auto obj = np.attr("genfromtxt")(fname, "delimiter"_a = delimiter,
                                   "skip_header"_a = header ? 1 : 0);

  pb->NumpyToTensorView(t, obj);
}

/**
 * Read a CSV file into a tensor view
 *
 * CSVs are currently read in using the Python interpreter through pybind11.
 *This has a startup performance hit, but CSV reading is intended to be a
 *slow-path function, so this is not a critical component to speed up. Currently
 *1D and 2D tensors are supported only.
 **/
template <typename T, int RANK>
void WriteCSV(tensor_t<T, RANK> &t, const std::string fname,
              const std::string delimiter)
{
  if (RANK != 1 && RANK != 2) {
    MATX_THROW(matxInvalidDim,
               "CSV reading limited to tensors of rank 1 and 2");
  }

  py::list ndims;
  for (int i = 0; i < RANK; i++) {
    ndims.append(t.Size(i));
  }

  std::unique_ptr<MatXPybind> pb;
  auto np = py::module_::import("numpy");

  auto np_ten = np.attr("empty")(ndims);
  pb->TensorViewToNumpy(np_ten, t);
  auto obj = np.attr("savetxt")(fname, np_ten, "delimiter"_a = delimiter);
}

}; // namespace io
}; // namespace matx