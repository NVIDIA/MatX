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

#include <cstdio>
#include <filesystem>
#include <iterator>
#include <shared_mutex>
#include <unordered_map>
#include <utility>
// #include <cudf/table/table.hpp>
// #include <cudf/io/csv.hpp>
// #include <cudf/types.hpp>

#include "matx/core/error.h"
#include "matx/core/nvtx.h"
#include "matx/core/pybind.h"
#include "matx/core/tensor.h"

#include "tiff.h"

#if defined(MATX_ENABLE_FILEIO) || defined(DOXYGEN_ONLY)

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
// void read_csv(tensor_t<T, RANK> &t, [[maybe_unused]] const std::string
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
 * @brief Read a CSV file into a tensor view
 *
 * CSVs are currently read in using the Python interpreter through pybind11.
 * This has a startup performance hit, but CSV reading is intended to be a
 * slow-path function, so this is not a critical component to speed up. Currently
 * 1D and 2D tensors are supported only.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to read data into
 * @param fname
 *   File path of .csv file
 * @param delimiter
 *   Delimiter to use for CSV file
 * @param skip_header
 *   Skip the header row of the CSV file, default as `true`.
 **/
template <typename TensorType>
void read_csv(TensorType &t, const std::string fname,
             const std::string delimiter, bool skip_header = true)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if (TensorType::Rank() != 1 && TensorType::Rank() != 2) {
    MATX_THROW(matxInvalidDim,
               "CSV reading limited to tensors of rank 1 and 2");
  }


  if (!std::filesystem::exists(fname)) {
    const std::string errorMessage = "Failed to read [" + fname + "], Does not Exist";
    MATX_THROW(matxIOError, errorMessage.c_str());
  }

  auto pb = std::make_unique<detail::MatXPybind>();

  auto np = pybind11::module_::import("numpy");
  auto obj = np.attr("genfromtxt")("fname"_a = fname.c_str(), "delimiter"_a = delimiter,
                                   "skip_header"_a = skip_header,
                                   "dtype"_a = detail::MatXPybind::GetNumpyDtype<typename TensorType::value_type>());
  pb->NumpyToTensorView(t, obj);
}

/**
 * Write a CSV file from a tensor view
 *
 * CSVs are currently written using the Python interpreter through pybind11.
 * This has a startup performance hit, but CSV writing is intended to be a
 * slow-path function, so this is not a critical component to speed up. Currently
 * 1D and 2D tensors are supported only.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to write data from
 * @param fname
 *   File path of .csv file
 * @param delimiter
 *   Delimiter to use for CSV file
 **/
template <typename TensorType>
void write_csv(const TensorType &t, const std::string fname,
              const std::string delimiter)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if (TensorType::Rank() != 1 && TensorType::Rank() != 2) {
    MATX_THROW(matxInvalidDim,
               "CSV reading limited to tensors of rank 1 and 2");
  }


  auto pb = std::make_unique<detail::MatXPybind>();
  auto np = pybind11::module_::import("numpy");

  auto np_ten = pb->TensorViewToNumpy(t);
  auto obj = np.attr("savetxt")(fname, np_ten, "delimiter"_a = delimiter);
}

/**
 * @brief Read a MAT file into a tensor view
 *
 * MAT files use SciPy's loadmat() function to read various MATLAB file
 * types in. MAT files are supersets of HDF5 files, and are allowed to
 * have multiple fields in them.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to read data into
 * @param fname
 *   File name of .mat file
 * @param var
 *   Variable name inside of .mat to read
 *
 **/
template <typename TensorType>
void read_mat(TensorType &t, const std::string fname,
             const std::string var)
{


  if (!std::filesystem::exists(fname)) {
    const std::string errorMessage = "Failed to read [" + fname + "], Does not Exist";
    MATX_THROW(matxIOError, errorMessage.c_str());
  }


  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)


  auto pb = std::make_unique<detail::MatXPybind>();

  auto sp = pybind11::module_::import("scipy.io");
  auto obj = (pybind11::dict)sp.attr("loadmat")("file_name"_a = fname);
  auto v = obj[var.c_str()];

  pb->NumpyToTensorView(t, v);
}

/**
 * @brief Read a MAT file and return a tensor view
 *
 * MAT files use SciPy's loadmat() function to read various MATLAB file
 * types in. MAT files are supersets of HDF5 files, and are allowed to
 * have multiple fields in them.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param fname
 *   File name of .mat file
 * @param var
 *   Variable name inside of .mat to read
 * @return
 *  Tensor view of data read from file
 **/
template <typename TensorType>
auto read_mat(const std::string fname,
             const std::string var)
{

  if (!std::filesystem::exists(fname)) {
    const std::string errorMessage = "Failed to read [" + fname + "], Does not Exist";
    MATX_THROW(matxIOError, errorMessage.c_str());
  }

  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  auto pb = std::make_unique<detail::MatXPybind>();

  auto sp = pybind11::module_::import("scipy.io");
  auto obj = (pybind11::dict)sp.attr("loadmat")("file_name"_a = fname);
  auto v = obj[var.c_str()];

  return pb->NumpyToTensorView<TensorType>(v);
}

/**
 * @brief Write a MAT file from a tensor view
 *
 * Writes a single tensor value into a .mat file.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to read data into
 * @param fname
 *   File name of .mat file
 * @param var
 *   Variable name to save inside of mat file
 */
template <typename TensorType>
void write_mat(const TensorType &t, const std::string fname,
              const std::string var)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  auto pb = std::make_unique<detail::MatXPybind>();
  auto np = pybind11::module_::import("numpy");
  auto sp = pybind11::module_::import("scipy.io");

  auto np_ten = pb->TensorViewToNumpy(t);

  auto td = pybind11::dict{};
  td[var.c_str()] = np_ten;
  auto obj = sp.attr("savemat")("file_name"_a = fname, "mdict"_a = td);
}

/**
 * @brief Read a NPY file into a tensor view
 *
 * NPY files are a simple binary format for storing arrays of numbers.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to read data into
 * @param fname
 *   File name of .npy file
 */
template <typename TensorType>
void read_npy(TensorType &t, const std::string& fname)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  if (!std::filesystem::exists(fname)) {
    const std::string errorMessage = "Failed to read [" + fname + "], Does not Exist";
    MATX_THROW(matxIOError, errorMessage.c_str());
  }

  auto pb = std::make_unique<detail::MatXPybind>();

  auto np = pybind11::module_::import("numpy");
  auto obj = np.attr("load")("file"_a = fname);

  pb->NumpyToTensorView(t, obj);
}

/**
 * @brief Write a NPY file from a tensor view
 *
 * NPY files are a simple binary format for storing arrays of numbers.
 *
 * @tparam TensorType
 *   Data type of tensor
 * @param t
 *   Tensor to write data from
 * @param fname
 *   File name of .npy file
 */
template <typename TensorType>
void write_npy(const TensorType &t, const std::string& fname)
{
  MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

  auto pb = std::make_unique<detail::MatXPybind>();
  auto np = pybind11::module_::import("numpy");

  auto np_ten = pb->TensorViewToNumpy(t);

  np.attr("save")("file"_a = fname, "arr"_a = np_ten);
}

}; // namespace io
}; // namespace matx

#endif
