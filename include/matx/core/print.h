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

#include <matx/core/type_utils.h>

namespace matx {
  namespace detail {
    /**
     * Print a value
     *
     * Type-agnostic function to print a value to stdout
     *
     * @param val
     */
    template <typename T>
    __MATX_INLINE__ __MATX_HOST__ void PrintVal(FILE* fp, const T &val)
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

      using namespace std::literals::string_literals;

      if constexpr (is_complex_v<T>) {
        const auto prec = std::to_string(PRINT_PRECISION);
        const auto fmt_s = ("% ."s + prec + "e%+." + prec + "ej ");
        fprintf(fp, fmt_s.c_str(), static_cast<float>(val.real()),
              static_cast<float>(val.imag()));
      }
      else if constexpr (is_matx_half_v<T> || is_half_v<T>) {
        const auto prec = std::to_string(PRINT_PRECISION);
        const auto fmt_s = ("% ."s + prec + "e ");
        fprintf(fp, fmt_s.c_str(), static_cast<float>(val));
      }
      else if constexpr (std::is_floating_point_v<T>) {
        const auto prec = std::to_string(PRINT_PRECISION);
        const auto fmt_s = ("% ."s + prec + "e ");
        fprintf(fp, fmt_s.c_str(), val);
      }
      else if constexpr (std::is_same_v<T, long long int>) {
        fprintf(fp, "% lld ", val);
      }
      else if constexpr (std::is_same_v<T, int64_t>) {
        fprintf(fp, "% " PRId64 " ", val);
      }
      else if constexpr (std::is_same_v<T, int32_t>) {
        fprintf(fp, "% " PRId32 " ", val);
      }
      else if constexpr (std::is_same_v<T, int16_t>) {
        fprintf(fp, "% " PRId16 " ", val);
      }
      else if constexpr (std::is_same_v<T, int8_t>) {
        fprintf(fp, "% " PRId8 " ", val);
      }
      else if constexpr (std::is_same_v<T, uint64_t>) {
        fprintf(fp, "+%" PRIu64 " ", val);
      }
      else if constexpr (std::is_same_v<T, uint32_t>) {
        fprintf(fp, "+%" PRIu32 " ", val);
      }
      else if constexpr (std::is_same_v<T, uint16_t>) {
        fprintf(fp, "+%" PRIu16 " ", val);
      }
      else if constexpr (std::is_same_v<T, uint8_t>) {
        fprintf(fp, "+%" PRIu8 " ", val);
      }
      else if constexpr (std::is_same_v<T, bool>) {
        fprintf(fp, "% d ", val);
      }
    }

    /**
     * convert Type to string
     *
     * function convert a tensor type to a string
     *
     */
    template <typename T> static std::string GetTensorTypeString()
    {
      if constexpr (std::is_same_v<T, bool>)
        return "bool";
      if constexpr (std::is_same_v<T, int32_t>)
        return "int32_t";
      if constexpr (std::is_same_v<T, uint32_t>)
        return "uint32_t";
      if constexpr (std::is_same_v<T, int64_t>)
        return "int64_t";
      if constexpr (std::is_same_v<T, uint64_t>)
        return "uint64_t";
      if constexpr (std::is_same_v<T, float> )
        return "float";
      if constexpr (std::is_same_v<T, matxFp16>)
        return "float16";
      if constexpr (std::is_same_v<T, matxBf16>)
        return "bfloat16";
      if constexpr (std::is_same_v<T, double>)
        return "double";
      if constexpr (std::is_same_v<T, cuda::std::complex<double>> || std::is_same_v<T, std::complex<double>>)
        return "complex<double>";
      if constexpr (std::is_same_v<T, cuda::std::complex<float>> || std::is_same_v<T, std::complex<float>>)
        return "complex<float>";
      if constexpr (std::is_same_v<T, matxFp16Complex>)
        return "complex<float16>";
      if constexpr (std::is_same_v<T, matxBf16Complex>)
        return "complex<bfloat16>";

      return "unknown";
    }

    template <typename Op>
    void PrintShapeImpl(const Op& op, FILE *fp) {
      if (is_tensor_view_v<Op>) {
        fprintf(fp, "%s: ",op.str().c_str());
      }

      std::string type = (is_tensor_view_v<Op>) ? "Tensor" : "Operator";
      fprintf(fp, "%s{%s} Rank: %d, Sizes:[", type.c_str(), detail::GetTensorTypeString<typename Op::value_type>().c_str(), op.Rank());
      for (index_t dimIdx = 0; dimIdx < op.Rank(); dimIdx++)
      {
        fprintf(fp, "%" MATX_INDEX_T_FMT, op.Size(static_cast<int>(dimIdx)) );
        if( dimIdx < (op.Rank() - 1) )
          fprintf(fp, ", ");
      }

      if constexpr (is_tensor_view_v<Op>)
      {
        fprintf(fp, "], Strides:[");
        if constexpr (Op::Rank() > 0)
        {
          for (index_t dimIdx = 0; dimIdx < (op.Rank() ); dimIdx++ )
          {
            fprintf(fp, "%" MATX_INDEX_T_FMT, op.Stride(static_cast<int>(dimIdx)) );
            if( dimIdx < (op.Rank() - 1) )
            {
              fprintf(fp, ",");
            }
          }
        }
      }

      fprintf(fp, "]\n");
    }


    /**
     * Print a tensor
     *
     * Type-agnostic function to print a tensor to stdout
     *
     */
    template <typename Op, typename ... Args>
    __MATX_HOST__ void InternalPrint(FILE* fp, const Op &op, Args ...dims)
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

      MATX_STATIC_ASSERT(Op::Rank() == sizeof...(Args), "Number of dimensions to print must match tensor rank");
      MATX_STATIC_ASSERT(Op::Rank() <= 4, "Printing is only supported on tensors of rank 4 or lower currently");

      if constexpr (sizeof...(Args) == 0) {
        PrintVal(fp, op.operator()());
        fprintf(fp, "\n");
      }
      else if constexpr (sizeof...(Args) == 1) {
        auto& k =detail:: pp_get<0>(dims...);
        for (index_t _k = 0; _k < ((k == 0) ? op.Size(0) : k); _k++) {
          if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
            if (_k == 0) {
              fprintf(fp, "[");
            }
            else {
              fprintf(fp, " ");
            }
          }
          if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
            fprintf(fp, "%06" MATX_INDEX_T_FMT ": ", _k);
          }
          PrintVal(fp, op.operator()(_k));
          if (_k == (op.Size(0)-1)) {
            if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
              fprintf(fp, "]");
            }
          }
          else {
            if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
              fprintf(fp, ",");
            }
          }
          fprintf(fp, "\n");
        }
      }
      else if constexpr (sizeof...(Args) == 2) {
        auto& k = detail::pp_get<0>(dims...);
        auto& l = detail::pp_get<1>(dims...);
        for (index_t _k = 0; _k < ((k == 0) ? op.Size(0) : k); _k++) {
          if (_k == 0) {
            if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
              fprintf(fp, "[");
            }
          }
          for (index_t _l = 0; _l < ((l == 0) ? op.Size(1) : l); _l++) {
            if (_l == 0) {
              if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
                fprintf(fp, "%06" MATX_INDEX_T_FMT ": ", _k);
              }
              else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                if (_k == 0) {
                  fprintf(fp, "[");
                }
                else {
                  fprintf(fp, " [");
                }
              }
              else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                if (_k != 0) {
                  fprintf(fp, " ");
                }
              }
            }

            PrintVal(fp, op.operator()(_k, _l));

            if (_l == (op.Size(1)-1)) {
              if (_k == (op.Size(0)-1)) {
                if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                  fprintf(fp, "]");
                }
                else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                  fprintf(fp, "]]");
                }
              }
              else {
                if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                  fprintf(fp, "; ...");
                }
                else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                  fprintf(fp, "],");
                }
              }
            }
            else
            {
              if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
                fprintf(fp, ", ");
              }
            }
          }
          fprintf(fp, "\n");
        }
      }
      else if constexpr (sizeof...(Args) == 3) {
        auto& j = detail::pp_get<0>(dims...);
        auto& k = detail::pp_get<1>(dims...);
        auto& l = detail::pp_get<2>(dims...);
        for (index_t _j = 0; _j < ((j == 0) ? op.Size(0) : j); _j++) {
          if (_j == 0) {
            if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
              fprintf(fp, "cat(3, ...\n");
            }
            else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
              fprintf(fp, "[");
            }
          }
          if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
            fprintf(fp, "[%06" MATX_INDEX_T_FMT ",:,:]\n", _j);
          }
          for (index_t _k = 0; _k < ((k == 0) ? op.Size(1) : k); _k++) {
            if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
              fprintf(fp, "       ");
            }
            if (_k == 0) {
              if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                fprintf(fp, "[");
              }
              else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                if (_j == 0) {
                  fprintf(fp, "[");
                }
                else {
                  fprintf(fp, " [");
                }
              }
            }
            else {
              if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                fprintf(fp, " ");
              }
            }
            for (index_t _l = 0; _l < ((l == 0) ? op.Size(2) : l); _l++) {
              if (_l == 0) {
                if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
                  fprintf(fp, "%06" MATX_INDEX_T_FMT ": ", _k);
                }
                else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                  if (_k == 0) {
                    fprintf(fp, "[");
                  }
                  else {
                    fprintf(fp, "  [");
                  }
                }
              }

              PrintVal(fp, op.operator()(_j, _k, _l));

              if (_l == (op.Size(2)-1)) {
                if (_k == (op.Size(1)-1)) {
                  if (_j == (op.Size(0)-1)) {
                    if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                      fprintf(fp, "])\n");
                    }
                    else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                      fprintf(fp, "]]]");
                    }
                  }
                  else {
                    if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                      fprintf(fp, "], ...");
                    }
                    else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                      fprintf(fp, "]],");
                    }
                  }
                }
                else {
                  if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                    fprintf(fp, "; ...");
                  }
                  else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                    fprintf(fp, "],");
                  }
                }
              }
              else {
                if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
                  fprintf(fp, ", ");
                }
              }
            }
            fprintf(fp, "\n");
          }
          if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
            fprintf(fp, "\n");
          }
        }
      }
      else if constexpr (sizeof...(Args) == 4) {
        auto& i = detail::pp_get<0>(dims...);
        auto& j = detail::pp_get<1>(dims...);
        auto& k = detail::pp_get<2>(dims...);
        auto& l = detail::pp_get<3>(dims...);
        for (index_t _i = 0; _i < ((i == 0) ? op.Size(0) : i); _i++) {
          if (_i == 0) {
            if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
              fprintf(fp, "cat(4, ...\n");
            }
            else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
              fprintf(fp, "[");
            }
          }
          for (index_t _j = 0; _j < ((j == 0) ? op.Size(1) : j); _j++) {
            if (_j == 0) {
              if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                fprintf(fp, "       cat(3, ...\n");
              }
              else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                if (_i == 0) {
                  fprintf(fp, "[");
                }
                else {
                  fprintf(fp, " [");
                }
              }
            }
            if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
              fprintf(fp, "[%06" MATX_INDEX_T_FMT ",%06" MATX_INDEX_T_FMT ",:,:]\n", _i, _j);
            }
            for (index_t _k = 0; _k < ((k == 0) ? op.Size(2) : k); _k++) {
              if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                fprintf(fp, "              ");
              }
              if (_k == 0) {
                if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                  fprintf(fp, "[");
                }
                else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                  if (_j == 0) {
                    fprintf(fp, "[");
                  }
                  else {
                    fprintf(fp, "  [");
                  }
                }
              }
              else {
                if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                  fprintf(fp, " ");
                }
              }
              for (index_t _l = 0; _l < ((l == 0) ? op.Size(3) : l); _l++) {
                if (_l == 0) {
                  if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
                    fprintf(fp, "%06" MATX_INDEX_T_FMT ": ", _k);
                  }
                  else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                    if (_k == 0) {
                      fprintf(fp, "[");
                    }
                    else {
                      fprintf(fp, "   [");
                    }
                  }
                }

                PrintVal(fp, op.operator()(_i, _j, _k, _l));

                if (_l == (op.Size(3)-1)) {
                  if (_k == (op.Size(2)-1)) {
                    if (_j == (op.Size(1)-1)) {
                      if (_i == (op.Size(0)-1)) {
                        if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                          fprintf(fp, "]))\n");
                        }
                        else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                          fprintf(fp, "]]]]");
                        }
                      }
                      else {
                        if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                          fprintf(fp, "]), ...");
                        }
                        else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                          fprintf(fp, "]]],");
                        }
                      }
                    }
                    else {
                      if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                        fprintf(fp, "], ...");
                      }
                      else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                        fprintf(fp, "]],");
                      }
                    }
                  }
                  else {
                    if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
                      fprintf(fp, "; ...");
                    }
                    else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
                      fprintf(fp, "],");
                    }
                  }
                }
                else {
                  if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
                    fprintf(fp, ", ");
                  }
                }
              }
              fprintf(fp, "\n");
            }
            if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
              fprintf(fp, "\n");
            }
          }
        }
      }
    }

    template <typename Op,
    typename... Args,
            std::enable_if_t<((std::is_integral_v<Args>)&&...) &&
                                  (Op::Rank() == 0 || sizeof...(Args) > 0),
                              bool> = true>
    void DevicePrint(FILE*fp, [[maybe_unused]] const Op &op, [[maybe_unused]] Args... dims) {
  #ifdef __CUDACC__
      if constexpr (PRINT_ON_DEVICE) {
        PrintKernel<<<1, 1>>>(op, dims...);
      }
      else {
        auto tmpv = make_tensor<typename matx::remove_cvref_t<typename Op::value_type>>(op.Shape());
        (tmpv = op).run();
        PrintData(fp, tmpv, dims...);
      }
  #endif
    }


    /**
     * @brief Print a tensor's values to output file stream
     *
     * This is the interal `Print()` takes integral values for each index, and prints that as many values
     * in each dimension as the arguments specify. For example:
     *
     * `a.Print(2, 3, 2);`
     *
     * Will print 2 values of the first, 3 values of the second, and 2 values of the third dimension
     * of a 3D tensor. The number of parameters must match the rank of the tensor. A special value of
     * 0 can be used if the entire tensor should be printed:
     *
     * `a.Print(0, 0, 0);` // Prints the whole tensor
     *
     * For more fine-grained printing, see the over `Print()` overloads.
     *
     * @tparam Args Integral argument types
     * @param fp output file stream
     * @param op input Operator
     * @param dims Number of values to print for each dimension
     */
    template <typename Op, typename... Args,
              std::enable_if_t<((std::is_integral_v<Args>)&&...) &&
                                    (Op::Rank() == 0 || sizeof...(Args) > 0),
                                bool> = true>
    void PrintData(FILE* fp, const Op &op, Args... dims) {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    #ifdef __CUDACC__
      cudaDeviceSynchronize();
      if constexpr (is_tensor_view_v<Op>) {
        // If the user is printing a tensor with a const pointer underlying the data, we need to do the lookup
        // as if it's not const. This is because the ownership decision is done at runtime instead of compile-time,
        // so even though the lookup will never be done, the compilation path happens.
        auto ptr_strip = const_cast<typename matx::remove_cvref_t<typename Op::value_type>*>(op.Data());
        auto kind = GetPointerKind(ptr_strip);

        // Try to get pointer from cuda
        if (kind == MATX_INVALID_MEMORY) {
          CUmemorytype mtype;
          void *data[] = {&mtype};
          CUpointer_attribute attrs[] = {CU_POINTER_ATTRIBUTE_MEMORY_TYPE};
          [[maybe_unused]] auto ret = cuPointerGetAttributes(1,
                                            &attrs[0],
                                            data,
                                            reinterpret_cast<CUdeviceptr>(op.Data()));
          MATX_ASSERT_STR_EXP(ret, CUDA_SUCCESS, matxCudaError, "Failed to get memory type");
          MATX_ASSERT_STR(mtype == CU_MEMORYTYPE_HOST || mtype == 0 || mtype == CU_MEMORYTYPE_DEVICE,
            matxNotSupported, "Invalid memory type for printing");

          if (mtype == CU_MEMORYTYPE_DEVICE) {
            detail::DevicePrint(fp, op, dims...);
          }
          else {
            detail::InternalPrint(fp, op, dims...);
          }
        }
        else if (kind == MATX_INVALID_MEMORY || HostPrintable(kind)) {
          detail::InternalPrint(fp, op, dims...);
        }
        else if (DevicePrintable(kind)) {
          detail::DevicePrint(fp, op, dims...);
        }
      }
      else {
        auto tmpv = make_tensor<typename Op::value_type>(op.Shape());
        (tmpv = op).run();
        cudaStreamSynchronize(0);
        InternalPrint(fp, tmpv, dims...);
      }
    #else
      InternalPrint(fp, op, dims...);
    #endif
    }
  };


  /**
   * @brief Print the shape and type of an operator
   *
   * Prints the shape and type of an operator. Data is not printed
   *
   * @param op input Operator
   */
  template <typename T>
  void print_shape(const T& op) {
    detail::PrintShapeImpl(op, stdout);
  }

  /**
   * @brief print a tensor's values to output file stream
   *
   * This is a wrapper utility function to print the type, size and stride of tensor,
   * see PrintData for details of internal tensor printing options
   *
   * @tparam Args Integral argument types
   * @param fp output file stream
   * @param op input Operator
   * @param dims Number of values to print for each dimension
   */
  #ifndef DOXYGEN_ONLY
  template <typename Op, typename... Args,
            std::enable_if_t<((std::is_integral_v<Args>)&&...) &&
                                  (Op::Rank() == 0 || sizeof...(Args) > 0),
                              bool> = true>
  #else
  template <typename Op, typename... Args>
  #endif
  void fprint(FILE* fp, const Op &op, Args... dims)
  {
    MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    detail::PrintShapeImpl(op, fp);
    detail::PrintData(fp, op, dims...);

  }


  #ifndef DOXYGEN_ONLY
  // Complete hide this version from doxygen, otherwise we get
  // "error: argument 'op' from the argument list of matx::fprint has multiple @param documentation sections"
  // due to the Rank==0 definition above
  /**
   * @brief Print a tensor's all values to output file stream
   *
   * This form of `fprint()` is an alias of `fprint(fp, op, 0)`, `fprint(fp, op, 0, 0)`,
   * `fprint(fp, op, 0, 0, 0)` and `fprint(fp, op, 0, 0, 0, 0)` for 1D, 2D, 3D and 4D tensor
   * respectively. It passes the proper number of zeros to `fprint(...)`
   * automatically according to the rank of this tensor. The user only have to
   * invoke `fprint(fp, op)` to print the whole tensor, instead of passing zeros
   * manually.
   *
   * @tparam Op Operator input type
   * @tparam Args Bounds type
   * @param fp Output file stream
   * @param op Operator input
   * @param dims Bounds for printing
   */
  template <typename Op, typename... Args,
            std::enable_if_t<(Op::Rank() > 0 && sizeof...(Args) == 0), bool> = true>
  void fprint(FILE* fp, const Op &op, [[maybe_unused]] Args... dims) {
    cuda::std::array<int, Op::Rank()> arr = {0};
    auto tp = cuda::std::tuple_cat(arr);
    cuda::std::apply([&](auto &&...args) { fprint(fp, op, args...); }, tp);
  }

  // Complete hide this version from doxygen, otherwise we get
  // "error: argument 'op' from the argument list of matx::print has multiple @param documentation sections"
  // due to the Rank==0 definition above
  /**
   * @brief Print a tensor's all values to stdout
   *
   * This form of `print()` is an alias of `print(op, 0)`, `print(op, 0, 0)`,
   * `print(op, 0, 0, 0)` and `print(op, 0, 0, 0, 0)` for 1D, 2D, 3D and 4D tensor
   * respectively. It passes the proper number of zeros to `print(...)`
   * automatically according to the rank of this tensor. The user only have to
   * invoke `print(op)` to print the whole tensor, instead of passing zeros
   * manually.
   *
   * @tparam Op Operator input type
   * @tparam Args Bounds type
   * @param fp Output file stream
   * @param op Operator input
   * @param dims Bounds for printing
   */
  template <typename Op, typename... Args,
            std::enable_if_t<(Op::Rank() > 0 && sizeof...(Args) == 0), bool> = true>
  void print(const Op &op, [[maybe_unused]] Args... dims) {
    cuda::std::array<int, Op::Rank()> arr = {0};
    auto tp = cuda::std::tuple_cat(arr);
    cuda::std::apply([&](auto &&...args) { fprint(stdout, op, args...); }, tp);
  }

  /**
   * @brief Print a tensor's all values to stdout
   *
   * This form of `print()` is a specialization for 0D tensors.
   *
   * @tparam Op Operator input type
   * @param op Operator input
   */
  template <typename Op,
          std::enable_if_t<(Op::Rank() == 0), bool> = true>
  void print(const Op &op)
  {
    fprint(stdout, op);
  }

  #endif // not DOXYGEN_ONLY

  /**
   * @brief Set the print() precision for floating point values
   *
   * @param precision Number of digits of precision for floating point output (default 4).
   */
  __MATX_INLINE__ __MATX_HOST__ void set_print_precision(unsigned int precision) {
    PRINT_PRECISION = precision;
  }

  /**
   * @brief Get the print() precision for floating point values
   *
   * @return Number of digits of precision for floating point output (default 4).
   */
  __MATX_INLINE__ __MATX_HOST__ unsigned int get_print_precision() {
    return PRINT_PRECISION;
  }

  /**
   * @brief Set the print() format type
   *
   * @param format_type print format type (default MATX_PRINT_FORMAT_DEFAULT)
   */
  __MATX_INLINE__ __MATX_HOST__ void set_print_format_type(enum PrintFormatType format_type) {
    PRINT_FORMAT_TYPE = format_type;
  }

  /**
   * @brief Get the print() format type
   *
   * @return The print format type
   */
  __MATX_INLINE__ __MATX_HOST__ enum PrintFormatType get_print_format_type() {
    return PRINT_FORMAT_TYPE;
  }

} // End namespace matx
