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
#include "matx/core/make_tensor.h"

#ifdef MATX_ENABLE_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <optional>
#include <filesystem>

namespace fs = std::filesystem;

namespace matx {

#define MATX_TEST_ASSERT_NEAR(__x, __y, __t)                                   \
  if constexpr (IsHalfType<decltype(__x)>())                                   \
    if (fabs(static_cast<float>(__x) - static_cast<float>(__y)) > __t)         \
      return false;                                                            \
    else if (fabs((__x) - (__y)) > __t)                                        \
      return false;

#define MATX_TEST_ASSERT_COMPARE(__pbp, __x, __f, __t)                         \
  {                                                                            \
    auto __y = __pbp->CompareOutput((__x), (__f), (__t));                      \
    if (__y) {                                                                 \
      __pbp->PrintTestError(*__y, __LINE__, __FILE__);                         \
      FAIL();                                                                  \
    }                                                                          \
  }

#define MATX_ASSERT_EQ(a, b)                                                   \
  if constexpr (IsHalfType<TypeParam>())                                       \
    ASSERT_EQ(static_cast<double>(a), static_cast<double>(b));                 \
  else                                                                         \
    ASSERT_EQ((a), (b));

namespace detail{

using namespace pybind11::literals;

const static std::string GENERATORS_PATH = "/test/test_vectors/generators/";

template <typename T> struct TestFailResult {
  using my_type = T;
  std::string index;
  std::string fname;
  T computed;
  T filedat;
  double thresh;
};


class MatXPybind {
public:
  MatXPybind() { Init(); }

  void Init() {
    if (gil == nullptr) {
      try {
        gil = new pybind11::scoped_interpreter{};
      }
      catch (...) {
        // Interpreter already running
      }
    }

    const auto current_dir = fs::path(__FILE__).parent_path();
    AddPath((current_dir.string() + "/../../..") + GENERATORS_PATH);
  }

  void AddPath(const std::string &path)
  {
    sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")(path);
  }

  void RunTVGenerator(const char *func)
  {
    res_dict = sx_obj.attr(func)();
  }

  template <typename T>
  void InitTVGenerator(const std::string mname, const std::string cname,
                       const std::vector<index_t> &sizes)
  {
    mod = pybind11::module_::import(mname.c_str());
    pybind11::array_t<index_t> arr{static_cast<index_t>(sizes.size())};
    for (size_t i = 0; i < sizes.size(); i++) {
      arr.mutable_at(i) = sizes[i];
    }

    sx_obj = mod.attr(cname.c_str())(GetNumpyDtype<T>(), arr);
  }

  template <typename T>
  void InitAndRunTVGenerator(const std::string &mname, const std::string &cname,
                             const char *func,
                             const std::vector<index_t> &sizes)
  {
    InitTVGenerator<T>(mname, cname, sizes);
    RunTVGenerator(func);
  }

  template <typename T>
  void InitAndRunTVGeneratorWithCfg(const std::string &mname, const std::string &cname,
                             const char *func,
                             const pybind11::dict &cfg)
  {
    mod = pybind11::module_::import(mname.c_str());

    sx_obj = mod.attr(cname.c_str())(GetNumpyDtype<T>(), cfg);
    RunTVGenerator(func);
  }

  auto GetMethod(const std::string meth) { return sx_obj.attr(meth.c_str()); }

  ~MatXPybind() {

  }

  template <typename T> static auto GetNumpyDtype()
  {
    auto np = pybind11::module_::import("numpy");
    if constexpr (std::is_same_v<T, int32_t>)
      return np.attr("dtype")("i4");
    if constexpr (std::is_same_v<T, uint32_t>)
      return np.attr("dtype")("u4");
    if constexpr (std::is_same_v<T, int64_t>)
      return np.attr("dtype")("i8");
    if constexpr (std::is_same_v<T, uint64_t>)
      return np.attr("dtype")("u8");
    if constexpr (std::is_same_v<T, float> || is_matx_half_v<T>)
      return np.attr("dtype")("float32");
    if constexpr (std::is_same_v<T, double>)
      return np.attr("dtype")("float64");
    if constexpr (std::is_same_v<T, cuda::std::complex<double>> ||
                  std::is_same_v<T, std::complex<double>>) {
      return np.attr("dtype")("complex128");
    }
    if constexpr (std::is_same_v<T, cuda::std::complex<float>> ||
                  std::is_same_v<T, std::complex<float>> ||
                  is_complex_half_v<T>) {
      return np.attr("dtype")("complex64");
    }

  }

  template <typename TensorType>
  static pybind11::object GetEmptyNumpy(const TensorType &ten)
  {
    using T = typename TensorType::value_type;
    auto np = pybind11::module_::import("numpy");
    pybind11::list dims;

    // Set up dims of numpy tensor
    for (int i = 0; i < TensorType::Rank(); i++) {
      dims.append(ten.Size(i));
    }

    auto np_ten = np.attr("empty")("shape"_a = dims,
                                   "dtype"_a = MatXPybind::GetNumpyDtype<T>());
    return np_ten;
  }

  void PrintTensor(const std::string name)
  {
    pybind11::print(res_dict[name.c_str()]);
  }

  auto GetTensor(const std::string name) { return res_dict[name.c_str()]; }

  template <typename T>
  static void PrintTestError(const TestFailResult<T> &res, int line,
                             std::string file)
  {
    if constexpr (matx::is_complex_v<T>) {
      printf("Comparison failed at %s:%d:%s: val=%f+%fj file=%f+%fj (%s)\n",
             file.c_str(), line, res.index.c_str(),
             static_cast<float>(res.computed.real()),
             static_cast<float>(res.computed.imag()),
             static_cast<float>(res.filedat.real()),
             static_cast<float>(res.filedat.imag()), res.fname.c_str());
    }
    else if constexpr (is_matx_half_v<T>) {
      std::cout << "Comparison failed at " << file << ":" << line << ":"
                << res.index.c_str()
                << ": val=" << static_cast<float>(res.computed)
                << " file=" << static_cast<float>(res.filedat) << " ("
                << res.fname << ")" << std::endl;
    }
    else {
      std::cout << "Comparison failed at " << file << ":" << line << ":"
                << res.index.c_str() << ": val=" << res.computed
                << " file=" << res.filedat << " (" << res.fname << ")"
                << std::endl;
    }
  }

  // Helper to convert indices array to string
  template <size_t RANK>
  static std::string Index2Str(const cuda::std::array<index_t, RANK>& indices)
  {
    if constexpr (RANK == 0) {
      return "0";
    }
    else {
      std::string result = std::to_string(indices[0]);
      for (size_t i = 1; i < RANK; i++) {
        result += "/" + std::to_string(indices[i]);
      }
      return result;
    }
  }

  // Legacy overloads for backward compatibility
  static std::string Index2Str(index_t x) { return std::to_string(x); }

  static std::string Index2Str(index_t x, index_t y)
  {
    return std::to_string(x) + "/" + std::to_string(y);
  }

  static std::string Index2Str(index_t x, index_t y, index_t z)
  {
    return std::to_string(x) + "/" + std::to_string(y) + "/" +
           std::to_string(z);
  }

  static std::string Index2Str(index_t x, index_t y, index_t z, index_t w)
  {
    return std::to_string(x) + "/" + std::to_string(y) + "/" +
           std::to_string(z) + "/" + std::to_string(w);
  }

  template <typename T1, typename T2>
  static inline bool CompareVals(const T1 &ut_data, T2 &file_data,
                                 const double &thresh, const std::string &name,
                                 bool debug)
  {
    // All this ugly stuff is because there is no cuda::std::cout support
    // complex
    if constexpr (is_complex_v<T1> || is_complex_v<T2>) {
      if (debug) {
        printf("FileName=%s Vector=%f%+f File=%f%+f\n", name.c_str(),
               static_cast<double>(ut_data.real()),
               static_cast<double>(ut_data.imag()),
               static_cast<double>(file_data.real()),
               static_cast<double>(file_data.imag()));
      }

      if (fabs(static_cast<double>(ut_data.real()) -
               static_cast<double>(file_data.real())) > thresh) {
        return false;
      }
      if (fabs(static_cast<double>(ut_data.imag()) -
               static_cast<double>(file_data.imag())) > thresh) {
        return false;
      }
    }
    else {
      if (debug) {
        std::cout << "FileName=" << name.c_str()
                  << " Vector=" << static_cast<double>(ut_data)
                  << " File=" << static_cast<double>(file_data) << "\n";
      }
      else if (fabs(static_cast<double>(ut_data) -
                    static_cast<double>(file_data)) > thresh) {
        return false;
      }
    }

    return true;
  }

  template <typename T, typename... Ix>
  const T &at(const std::string fname, Ix... index) const
  {
    auto resobj = res_dict[fname.c_str()];
    auto ften = pybind11::array_t<T>(resobj);
    return ften.at(std::forward<Ix>(index)...);
  }

  template <typename T> inline T ConvertComplex(const T &in) { return in; }

  template <typename T>
  inline cuda::std::complex<T> ConvertComplex(const std::complex<T> in)
  {
    return {in.real(), in.imag()};
  }

  template <typename T>
  inline std::complex<T> ConvertComplex(const cuda::std::complex<T> in)
  {
    return {in.real(), in.imag()};
  }

  template <typename T>
  inline matxFp16Complex ConvertComplex(const matxFp16Complex in)
  {
    return {in.real(), in.imag()};
  }

  template <typename T>
  inline matxBf16Complex ConvertComplex(const matxBf16Complex in)
  {
    return {in.real(), in.imag()};
  }

  /**
   * @brief Converts an absolute (flat) index to multi-dimensional indices
   * 
   * Local utility function to avoid circular header dependencies
   *
   * @param op Operator or tensor
   * @param abs Absolute/flat index
   * @return cuda::std::array of indices
   */
  template <typename Op>
  static auto AbsToIdx(const Op &op, index_t abs) {
    constexpr int RANK = Op::Rank();
    cuda::std::array<index_t, RANK> indices;

    for (int idx = 0; idx < RANK; idx++) {
      if (idx == RANK-1) {
        indices[RANK-1] = abs;
      }
      else {
        index_t prod = 1;
        for (int i = idx + 1; i < RANK; i++) {
          prod *= op.Size(i);
        }
        indices[idx] = abs / prod;
        abs = abs % prod;
      }
    }

    return indices;
  }

  template <typename TensorType>
  void NumpyToTensorView(TensorType &ten,
                         const std::string fname)
  {
    auto resobj = res_dict[fname.c_str()];
    NumpyToTensorView(ten, resobj);
  }

  template <typename TensorType>
  void NumpyToTensorView(TensorType ten,
                         const pybind11::object &np_ten)
  {
    using T = typename TensorType::value_type;
    constexpr int RANK = TensorType::Rank();

    using ntype = matx_convert_complex_type<T>;
    auto ften = pybind11::array_t<ntype>(np_ten);

    if constexpr (RANK == 0) {
      ten() = ConvertComplex(ften.at());
    }
    else {
      // Iterate through all elements
      for (index_t i = 0; i < ten.TotalSize(); i++) {
        auto indices = AbsToIdx(ten, i);
        
        // Use cuda::std::apply to expand indices and access both arrays
        cuda::std::apply([&](auto&&... idx) {
          ten(idx...) = ConvertComplex(ften.at(idx...));
        }, indices);
      }
    }
  }

  template <typename TensorType>
  auto NumpyToTensorView(const pybind11::object &np_ten)
  {
    using T = typename TensorType::value_type;
    constexpr int RANK = TensorType::Rank();
    using ntype = matx_convert_complex_type<T>;
    auto ften = pybind11::array_t<ntype, pybind11::array::c_style | pybind11::array::forcecast>(np_ten);

    auto info = ften.request();

    assert(info.ndim == RANK);

    cuda::std::array<matx::index_t, RANK> shape;
    std::copy_n(info.shape.begin(), RANK, std::begin(shape));

    auto ten =  make_tensor<T> (shape);
    for (int n = 0; n < ften.size(); ++n) {
      ten.Data()[n] = ConvertComplex(ften.data()[n]);
    }
    return ten;
  }

  template <typename TensorType>
  auto TensorViewToNumpy(const TensorType &ten) {
    using tensor_type = typename TensorType::value_type;
    using ntype = matx_convert_complex_type<tensor_type>;
    constexpr int RANK = TensorType::Rank();

    // If this is a half-precision type pybind/numpy doesn't support it, so we fall back to the
    // slow method where we convert everything
    if constexpr (is_matx_type<tensor_type>()) {
      auto ften = pybind11::array_t<ntype, pybind11::array::c_style | pybind11::array::forcecast>(ten.Shape());

      // Iterate through all elements
      for (index_t i = 0; i < ten.TotalSize(); i++) {
        auto indices = AbsToIdx(ten, i);
        
        // Use cuda::std::apply to expand indices and access both arrays
        cuda::std::apply([&](auto&&... idx) {
          ften.mutable_at(idx...) = ConvertComplex(ten(idx...));
        }, indices);
      }

      return ften;      
    }
    else {
      const auto tshape = ten.Shape();
      const auto tstrides = ten.Strides();
      std::vector<pybind11::ssize_t> shape{tshape.begin(), tshape.end()};
      std::vector<pybind11::ssize_t> strides{tstrides.begin(), tstrides.end()};
      std::for_each(strides.begin(), strides.end(), [](pybind11::ssize_t &x) {
        x *= sizeof(tensor_type);
      });      

      auto buf = pybind11::buffer_info(
          ten.Data(), 
          sizeof(tensor_type),
          pybind11::format_descriptor<ntype>::format(),
          RANK,
          shape,
          strides
      );

      return pybind11::array_t<ntype, pybind11::array::c_style | pybind11::array::forcecast>(buf);      
    }
  }


  template <typename TensorType,
            typename CT = matx_convert_cuda_complex_type<typename TensorType::value_type>>
  std::optional<TestFailResult<CT>>
  CompareOutput(const TensorType &ten,
                const std::string fname, double thresh, bool debug = false)
  {
    using raw_type = typename TensorType::value_type;    
    using ntype = matx_convert_complex_type<raw_type>;
    using ctype = matx_convert_cuda_complex_type<raw_type>;
    auto resobj = res_dict[fname.c_str()];
    auto ften = pybind11::array_t<ntype>(resobj);
    constexpr int RANK = TensorType::Rank();

    cudaDeviceSynchronize();

    if constexpr (RANK == 0) {
      auto file_val = ften.at();
      auto ten_val = ConvertComplex(ten());
      if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
        return TestFailResult<ctype>{Index2Str(0), fname, ten_val, file_val,
                                     thresh};
      }
    }
    else {
      // Iterate through all elements
      for (index_t i = 0; i < ten.TotalSize(); i++) {
        auto indices = AbsToIdx(ten, i);
        
        // Use cuda::std::apply to expand indices and compare values
        auto comparison_failed = cuda::std::apply([&](auto&&... idx) {
          auto file_val = ften.at(idx...);
          auto ten_val = ConvertComplex(ten(idx...));
          return !CompareVals(ten_val, file_val, thresh, fname, debug) ? 
                 std::make_optional(std::make_pair(ten_val, file_val)) : 
                 std::nullopt;
        }, indices);
        
        if (comparison_failed) {
          return TestFailResult<ctype>{Index2Str(indices), fname, 
                                       comparison_failed->first, 
                                       comparison_failed->second,
                                       thresh};
        }
      }
    }

    return std::nullopt;
  }

private:
  inline static pybind11::scoped_interpreter *gil = nullptr;
  pybind11::module_ mod;
  pybind11::object res_dict;
  pybind11::object sx_obj;
  pybind11::module_ sys;
};

}; //namespace detail
}; // namespace matx

#endif
