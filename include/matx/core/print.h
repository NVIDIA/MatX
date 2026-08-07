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

#include <array>
#include <cinttypes>
#include <utility>
#include <matx/core/type_utils.h>

namespace matx {
  namespace detail {
    /**
     * Print a value
     *
     * Type-agnostic function to print a value
     *
     * @param fp
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
      if constexpr (is_tensor_view_v<Op>) {
        fprintf(fp, "%s: ", op.str().c_str());
      }

      std::string type;
      if constexpr (is_sparse_tensor_v<Op>)
        type = "SparseTensor";
      else if constexpr (is_tensor_view_v<Op>)
        type = "Tensor";
      else
        type = "Operator";
      fprintf(fp, "%s{%s} Rank: %d, Sizes:[", type.c_str(), detail::GetTensorTypeString<typename Op::value_type>().c_str(), op.Rank());
      for (index_t dimIdx = 0; dimIdx < op.Rank(); dimIdx++)
      {
        fprintf(fp, "%" MATX_INDEX_T_FMT, op.Size(static_cast<int>(dimIdx)) );
        if( dimIdx < (op.Rank() - 1) )
          fprintf(fp, ", ");
      }

      if constexpr (is_sparse_tensor_v<Op>)
      {
        // A sparse tensor has no strides, so show the level sizes instead.
        // These are obtained by translating dims to levels using the format.
        cuda::std::array<index_t, Op::Format::LVL> lvlsz;
        Op::Format::template dim2lvl<true>(op.Shape().data(), lvlsz.data());
        fprintf(fp, "], Levels:[");
        for (int lvlIdx = 0; lvlIdx < Op::Format::LVL; lvlIdx++) {
          fprintf(fp, "%" MATX_INDEX_T_FMT, lvlsz[lvlIdx]);
          if (lvlIdx < (Op::Format::LVL - 1)) {
            fprintf(fp, ", ");
          }
        }
      }
      else if constexpr (is_tensor_view_v<Op>)
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



    template <typename Op>
    using PrintIndexArray = std::array<index_t, static_cast<size_t>(Op::Rank())>;

    inline void PrintSpaces(FILE* fp, int count)
    {
      for (int i = 0; i < count; i++) {
        fprintf(fp, " ");
      }
    }

    template <typename Op, size_t... Is>
    __MATX_HOST__ void PrintValAtImpl(FILE* fp, const Op &op, const PrintIndexArray<Op> &idx,
                                      std::index_sequence<Is...>)
    {
      PrintVal(fp, op.operator()(idx[Is]...));
    }

    template <typename Op>
    __MATX_HOST__ void PrintValAt(FILE* fp, const Op &op, const PrintIndexArray<Op> &idx)
    {
      PrintValAtImpl(fp, op, idx, std::make_index_sequence<static_cast<size_t>(Op::Rank())>{});
    }

    template <typename Op, typename... Args>
    __MATX_HOST__ PrintIndexArray<Op> GetPrintExtents(const Op &op, Args... dims)
    {
      PrintIndexArray<Op> extents{static_cast<index_t>(dims)...};
      for (int dim = 0; dim < Op::Rank(); dim++) {
        if (extents[static_cast<size_t>(dim)] == 0) {
          extents[static_cast<size_t>(dim)] = op.Size(dim);
        }
      }
      return extents;
    }

    template <typename Op>
    __MATX_HOST__ void PrintDefaultMatrix(FILE* fp, const Op &op, const PrintIndexArray<Op> &extents,
                                          PrintIndexArray<Op> &idx)
    {
      constexpr int rank = Op::Rank();
      for (index_t row = 0; row < extents[rank - 2]; row++) {
        idx[rank - 2] = row;
        fprintf(fp, "%06" MATX_INDEX_T_FMT ": ", row);
        for (index_t col = 0; col < extents[rank - 1]; col++) {
          idx[rank - 1] = col;
          PrintValAt(fp, op, idx);
        }
        fprintf(fp, "\n");
      }
    }

    template <typename Op>
    __MATX_HOST__ void PrintDefaultSliceHeader(FILE* fp, const PrintIndexArray<Op> &idx)
    {
      constexpr int rank = Op::Rank();
      fprintf(fp, "[");
      for (int dim = 0; dim < rank - 2; dim++) {
        fprintf(fp, "%06" MATX_INDEX_T_FMT ",", idx[static_cast<size_t>(dim)]);
      }
      fprintf(fp, ":,:]\n");
    }

    template <int Dim, typename Op>
    __MATX_HOST__ void PrintDefaultSlices(FILE* fp, const Op &op, const PrintIndexArray<Op> &extents,
                                          PrintIndexArray<Op> &idx)
    {
      constexpr int rank = Op::Rank();
      if constexpr (Dim == rank - 2) {
        PrintDefaultSliceHeader<Op>(fp, idx);
        PrintDefaultMatrix(fp, op, extents, idx);
        fprintf(fp, "\n");
      }
      else {
        for (index_t i = 0; i < extents[Dim]; i++) {
          idx[Dim] = i;
          PrintDefaultSlices<Dim + 1>(fp, op, extents, idx);
        }
      }
    }

    template <typename Op>
    __MATX_HOST__ void PrintMlabMatrix(FILE* fp, const Op &op, const PrintIndexArray<Op> &extents,
                                       PrintIndexArray<Op> &idx, int indent)
    {
      constexpr int rank = Op::Rank();
      for (index_t row = 0; row < extents[rank - 2]; row++) {
        idx[rank - 2] = row;
        PrintSpaces(fp, indent);
        if (row == 0) {
          fprintf(fp, "[");
        }
        else {
          fprintf(fp, " ");
        }

        for (index_t col = 0; col < extents[rank - 1]; col++) {
          idx[rank - 1] = col;
          PrintValAt(fp, op, idx);
          if (col != extents[rank - 1] - 1) {
            fprintf(fp, ", ");
          }
        }

        if (row == extents[rank - 2] - 1) {
          fprintf(fp, "]");
        }
        else {
          fprintf(fp, "; ...\n");
        }
      }
    }

    template <int Dim, typename Op>
    __MATX_HOST__ void PrintMlabCat(FILE* fp, const Op &op, const PrintIndexArray<Op> &extents,
                                    PrintIndexArray<Op> &idx)
    {
      constexpr int rank = Op::Rank();
      constexpr int indent = Dim * 7;
      PrintSpaces(fp, indent);
      fprintf(fp, "cat(%d, ...\n", rank - Dim);
      for (index_t i = 0; i < extents[Dim]; i++) {
        idx[Dim] = i;
        if constexpr (Dim == rank - 3) {
          PrintMlabMatrix(fp, op, extents, idx, indent + 7);
        }
        else {
          PrintMlabCat<Dim + 1>(fp, op, extents, idx);
        }

        if (i == extents[Dim] - 1) {
          fprintf(fp, ")");
        }
        else {
          fprintf(fp, ", ...\n");
        }
      }
    }

    template <int Dim, typename Op>
    __MATX_HOST__ void PrintPythonRecursive(FILE* fp, const Op &op, const PrintIndexArray<Op> &extents,
                                            PrintIndexArray<Op> &idx)
    {
      constexpr int rank = Op::Rank();
      fprintf(fp, "[");
      if constexpr (Dim == rank - 1) {
        for (index_t i = 0; i < extents[Dim]; i++) {
          idx[Dim] = i;
          PrintValAt(fp, op, idx);
          if (i != extents[Dim] - 1) {
            fprintf(fp, ", ");
          }
        }
      }
      else {
        for (index_t i = 0; i < extents[Dim]; i++) {
          idx[Dim] = i;
          if (i != 0) {
            fprintf(fp, ",\n");
            PrintSpaces(fp, Dim + 1);
          }
          PrintPythonRecursive<Dim + 1>(fp, op, extents, idx);
        }
      }
      fprintf(fp, "]");
    }

    /**
     * Print a tensor
     *
     * Type-agnostic function to print a tensor
     *
     */
    template <typename Op, typename ... Args>
    __MATX_HOST__ void InternalPrint(FILE* fp, const Op &op, Args ...dims)
    {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_INTERNAL)

      MATX_STATIC_ASSERT(Op::Rank() == sizeof...(Args), "Number of dimensions to print must match tensor rank");

      if constexpr (sizeof...(Args) == 0) {
        PrintVal(fp, op.operator()());
        fprintf(fp, "\n");
      }
      else if constexpr (sizeof...(Args) == 1) {
        auto& k = detail::pp_get<0>(dims...);
        const index_t k_extent = (k == 0) ? op.Size(0) : static_cast<index_t>(k);
        for (index_t _k = 0; _k < k_extent; _k++) {
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
          if (_k == (k_extent - 1)) {
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
        const index_t k_extent = (k == 0) ? op.Size(0) : static_cast<index_t>(k);
        const index_t l_extent = (l == 0) ? op.Size(1) : static_cast<index_t>(l);
        for (index_t _k = 0; _k < k_extent; _k++) {
          if (_k == 0) {
            if ((PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) || (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON)) {
              fprintf(fp, "[");
            }
          }
          for (index_t _l = 0; _l < l_extent; _l++) {
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

            if (_l == (l_extent - 1)) {
              if (_k == (k_extent - 1)) {
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
      else {
        auto extents = GetPrintExtents(op, dims...);
        PrintIndexArray<Op> idx{};
        if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_DEFAULT) {
          PrintDefaultSlices<0>(fp, op, extents, idx);
        }
        else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_MLAB) {
          PrintMlabCat<0>(fp, op, extents, idx);
          fprintf(fp, "\n\n");
        }
        else if (PRINT_FORMAT_TYPE == MATX_PRINT_FORMAT_PYTHON) {
          PrintPythonRecursive<0>(fp, op, extents, idx);
          fprintf(fp, "\n");
        }
      }
    }

    template <typename Op, typename... Args>
      requires (((std::is_integral_v<Args>)&&...) &&
                (Op::Rank() == 0 || sizeof...(Args) > 0))
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
    template <typename Op, typename... Args>
      requires (((std::is_integral_v<Args>)&&...) &&
                (Op::Rank() == 0 || sizeof...(Args) > 0))
    void PrintData(FILE* fp, const Op &op, Args... dims) {
      MATX_NVTX_START("", matx::MATX_NVTX_LOG_API)

    #ifdef __CUDACC__
      cudaDeviceSynchronize();
      if constexpr (is_sparse_tensor_v<Op>) {
        using Format = typename Op::Format;
        fprintf(fp, "format = ");
        Format::print();
        const auto kind = GetPointerKind(op.Data());
        fprintf(fp, "space  = %s\n", SpaceString(kind).c_str());
        const auto nse = op.Nse();
        fprintf(fp, "nse    = %" MATX_INDEX_T_FMT "\n", nse);
        if (HostPrintable(kind)) {
          for (int lvlIdx = 0; lvlIdx < Format::LVL; lvlIdx++) {
            if (const index_t pend = op.posSize(lvlIdx)) {
              fprintf(fp, "pos[%d] = (", lvlIdx);
              for (index_t i = 0; i < pend; i++) {
                PrintVal(fp, op.POSData(lvlIdx)[i]);
              }
              fprintf(fp, ")\n");
            }
            if (const index_t cend = op.crdSize(lvlIdx)) {
              fprintf(fp, "crd[%d] = (", lvlIdx);
              for (index_t i = 0; i < cend; i++) {
                PrintVal(fp, op.CRDData(lvlIdx)[i]);
              }
              fprintf(fp, ")\n");
            }
          }
          fprintf(fp, "values = (");
          for (index_t i = 0; i < nse; i++) {
            PrintVal(fp, op.Data()[i]);
          }
          fprintf(fp, ")\n");
        }
      }
      else if constexpr (is_tensor_view_v<Op>) {
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
        if constexpr (is_dynamic_rank_op_v<Op>) {
          // Dynamic rank: dispatch to concrete rank at runtime
          const int r = detail::get_dyn_rank(op);
          auto materialize_and_print = [&]<int R>() {
            cuda::std::array<index_t, R> shape;
            for (int i = 0; i < R; i++) shape[i] = op.Size(i);
            auto tmpv = make_tensor<typename Op::value_type>(shape);
            (tmpv = op).run(CUDAJITExecutor{});
            cudaStreamSynchronize(0);
            detail::InternalPrint(fp, tmpv, dims...);
          };
          switch (r) {
            case 0: materialize_and_print.template operator()<0>(); break;
            case 1: materialize_and_print.template operator()<1>(); break;
            case 2: materialize_and_print.template operator()<2>(); break;
            case 3: materialize_and_print.template operator()<3>(); break;
            case 4: materialize_and_print.template operator()<4>(); break;
            case 5: materialize_and_print.template operator()<5>(); break;
            case 6: materialize_and_print.template operator()<6>(); break;
            case 7: materialize_and_print.template operator()<7>(); break;
            case 8: materialize_and_print.template operator()<8>(); break;
            default: MATX_THROW(matxInvalidParameter, "Dynamic tensor rank exceeds MATX_MAX_DYNAMIC_RANK");
          }
        } else {
          auto tmpv = make_tensor<typename Op::value_type>(op.Shape());
          (tmpv = op).run();
          cudaStreamSynchronize(0);
          detail::InternalPrint(fp, tmpv, dims...);
        }
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
  template <typename Op, typename... Args>
    requires (((std::is_integral_v<Args>)&&...) &&
              (Op::Rank() == 0 || sizeof...(Args) > 0))
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
  template <typename Op, typename... Args>
    requires (Op::Rank() > 0 && sizeof...(Args) == 0)
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
  template <typename Op, typename... Args>
    requires (Op::Rank() > 0 && sizeof...(Args) == 0)
  void print(const Op &op, [[maybe_unused]] Args... dims) {
    cuda::std::array<int, Op::Rank()> arr = {0};
    auto tp = cuda::std::tuple_cat(arr);
    cuda::std::apply([&](auto &&...args) { fprint(stdout, op, args...); }, tp);
  }

  /**
   * @brief Print all of a tensor's values to stdout
   *
   * This form of `print()` is a specialization for 1D+ tensors. A size of zero in
   * dimension prints all elements in that dimension.
   *
   * @tparam Op Operator input type
   * @param op Operator input
   */
  template <typename Op, typename... Args>
    requires (Op::Rank() > 0 && sizeof...(Args) > 0)
  void print(const Op &op, [[maybe_unused]] Args... dims) {
    fprint(stdout, op, dims...);
  }

  /**
   * @brief Print a tensor's all values to stdout
   *
   * This form of `print()` is a specialization for 0D tensors.
   *
   * @tparam Op Operator input type
   * @param op Operator input
   */
  template <typename Op>
    requires (Op::Rank() == 0)
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
