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


#if MATX_ENABLE_PYBIND11

#include <pybind11/embed.h>
#include <pybind11/numpy.h> 
#include <optional>

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
    AddPath(std::string(MATX_ROOT) + GENERATORS_PATH); 
  }

  void AddPath(const std::string &path)
  {
    sys = pybind11::module_::import("sys");
    sys.attr("path").attr("append")(path);
  }

  void RunTVGenerator(const std::string &func)
  {
    res_dict = sx_obj.attr(func.c_str())();
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
                             const std::string &func,
                             const std::vector<index_t> &sizes)
  {
    InitTVGenerator<T>(mname, cname, sizes);
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
    using T = typename TensorType::scalar_type;
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
               static_cast<float>(ut_data.real()),
               static_cast<float>(ut_data.imag()),
               static_cast<float>(file_data.real()),
               static_cast<float>(file_data.imag()));
      }

      if (fabs(static_cast<float>(ut_data.real()) -
               static_cast<float>(file_data.real())) > thresh) {
        return false;
      }
      if (fabs(static_cast<float>(ut_data.imag()) -
               static_cast<float>(file_data.imag())) > thresh) {
        return false;
      }
    }
    else {
      if (debug) {
        std::cout << "FileName=" << name.c_str()
                  << " Vector=" << static_cast<float>(ut_data)
                  << " File=" << static_cast<float>(file_data) << "\n";
      }
      else if (fabs(static_cast<float>(ut_data) -
                    static_cast<float>(file_data)) > thresh) {
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
    using T = typename TensorType::scalar_type;
    constexpr int RANK = TensorType::Rank();
    static_assert(RANK <=5, "NumpyToTensorView only supports max(RANK) = 5 at the moment.");

    using ntype = matx_convert_complex_type<T>;
    auto ften = pybind11::array_t<ntype>(np_ten);

    for (index_t s1 = 0; s1 < ten.Size(0); s1++) {
      if constexpr (RANK > 1) {
        for (index_t s2 = 0; s2 < ten.Size(1); s2++) {
          if constexpr (RANK > 2) {
            for (index_t s3 = 0; s3 < ten.Size(2); s3++) {
              if constexpr (RANK > 3) {
                for (index_t s4 = 0; s4 < ten.Size(3); s4++) {
                  if constexpr (RANK > 4) {
                    for (index_t s5 = 0; s5 < ten.Size(4); s5++) {
                      ten(s1, s2, s3, s4, s5) = ConvertComplex(ften.at(s1, s2, s3, s4, s5));
                    }
                  }
                  else {
                    ten(s1, s2, s3, s4) = ConvertComplex(ften.at(s1, s2, s3, s4));
                  }
                }
              }
              else {
                ten(s1, s2, s3) = ConvertComplex(ften.at(s1, s2, s3));
              }
            }
          }
          else {
            ten(s1, s2) = ConvertComplex(ften.at(s1, s2));
          }
        }
      }
      else {
        ten(s1) = ConvertComplex(ften.at(s1));
      }
    }
  }

  template <typename TensorType>
  auto TensorViewToNumpy(const TensorType &ten)
  {
    constexpr int RANK = TensorType::Rank();
    static_assert(RANK <=5, "TensorViewToNumpy only supports max(RANK) = 5 at the moment.");

    using ntype = matx_convert_complex_type<typename TensorType::scalar_type>;
    auto ften = pybind11::array_t<ntype>(ten.Shape());

    for (index_t s1 = 0; s1 < ten.Size(0); s1++) {
      if constexpr (RANK > 1) {
        for (index_t s2 = 0; s2 < ten.Size(1); s2++) {
          if constexpr (RANK > 2) {
            for (index_t s3 = 0; s3 < ten.Size(2); s3++) {
              if constexpr (RANK > 3) {
                for (index_t s4 = 0; s4 < ten.Size(3); s4++) {
                  if constexpr (RANK > 4) {
                    for (index_t s5 = 0; s5 < ten.Size(4); s5++) {
                      ften.mutable_at(s1, s2, s3, s4, s5) =
                          ConvertComplex(ten(s1, s2, s3, s4, s5));
                    }
                  } else {
                    ften.mutable_at(s1, s2, s3, s4) =
                        ConvertComplex(ten(s1, s2, s3, s4));
                  }
                }
              }
              else {
                ften.mutable_at(s1, s2, s3) = ConvertComplex(ten(s1, s2, s3));
              }
            }
          }
          else {
            ften.mutable_at(s1, s2) = ConvertComplex(ten(s1, s2));
          }
        }
      }
      else {
        ften.mutable_at(s1) = ConvertComplex(ten(s1));
      }
    }

    return ften;
  }

  template <typename TensorType, 
            typename CT = matx_convert_complex_type<typename TensorType::scalar_type>>
  std::optional<TestFailResult<CT>>
  CompareOutput(const TensorType &ten,
                const std::string fname, double thresh, bool debug = false)
  {
    using ntype = matx_convert_complex_type<typename TensorType::scalar_type>;
    auto resobj = res_dict[fname.c_str()];
    auto ften = pybind11::array_t<ntype>(resobj);
    constexpr int RANK = TensorType::Rank();

    cudaDeviceSynchronize();

    if constexpr (RANK == 0) {
      auto file_val = ften.at();
      auto ten_val = ConvertComplex(ten());
      if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
        return TestFailResult<ntype>{Index2Str(0), "0", ten_val, file_val,
                                     thresh};
      }
    }
    else {
      for (index_t s1 = 0; s1 < ten.Size(0); s1++) {
        if constexpr (RANK > 1) {
          for (index_t s2 = 0; s2 < ten.Size(1); s2++) {
            if constexpr (RANK > 2) {
              for (index_t s3 = 0; s3 < ten.Size(2); s3++) {
                if constexpr (RANK > 3) {
                  for (index_t s4 = 0; s4 < ten.Size(3); s4++) {
                    auto file_val = ften.at(s1, s2, s3, s4);
                    auto ten_val = ConvertComplex(ten(s1, s2, s3, s4));
                    if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
                      return TestFailResult<ntype>{Index2Str(s1, s2, s3, s4),
                                                   fname, ten_val, file_val,
                                                   thresh};
                    }
                  }
                }
                else {
                  auto file_val = ften.at(s1, s2, s3);
                  auto ten_val = ConvertComplex(ten(s1, s2, s3));
                  if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
                    return TestFailResult<ntype>{Index2Str(s1, s2, s3), fname,
                                                 ten_val, file_val, thresh};
                  }
                }
              }
            }
            else {
              auto file_val = ften.at(s1, s2);
              auto ten_val = ConvertComplex(ten(s1, s2));
              if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
                return TestFailResult<ntype>{Index2Str(s1, s2), fname, ten_val,
                                             file_val, thresh};
              }
            }
          }
        }
        else {
          auto file_val = ften.at(s1);
          auto ten_val = ConvertComplex(ten(s1));
          if (!CompareVals(ten_val, file_val, thresh, fname, debug)) {
            return TestFailResult<ntype>{Index2Str(s1), fname, ten_val,
                                         file_val, thresh};
          }
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
