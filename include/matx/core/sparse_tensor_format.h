////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
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
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
/////////////////////////////////////////////////////////////////////////////////

#pragma once

#include <string>

namespace matx {
namespace experimental {

//
// A level expression consists of an expression in terms of dimension
// variables (e.g. d0, d0 + d1, d1 - d0, d0 div 2, or d0 mod 2).
//
// TODO: more elegant to have Var(i) and Const(c) in type "syntax"
//
enum class LvlOp { Id, Add, Sub, Div, Mod };
template <LvlOp O, int I, int J = 0> class LvlExpr {
public:
  static constexpr LvlOp op = O;
  static constexpr int di = I; // dim var
  static constexpr int cj = J; // dim var or const

  static constexpr bool isId(int d) { return op == LvlOp::Id && di == d; }

  static std::string toString() {
    if constexpr (op == LvlOp::Id) {
      return "d" + std::to_string(di);
    } else if constexpr (op == LvlOp::Add) {
      return "d" + std::to_string(di) + " + d" + std::to_string(cj);
    } else if constexpr (op == LvlOp::Sub) {
      return "d" + std::to_string(di) + " - d" + std::to_string(cj);
    } else if constexpr (op == LvlOp::Div) {
      return "d" + std::to_string(di) + " div " + std::to_string(cj);
    } else if constexpr (op == LvlOp::Mod) {
      return "d" + std::to_string(di) + " mod " + std::to_string(cj);
    } else { // Should not happen
      return "?";
    }
  }
};

//
// A level type consists of a level format together with a set of
// level properties (unique and ordered by default).
//
enum class LvlFormat { Dense, Compressed, Singleton, Range };
template <LvlFormat F, bool U = true, bool O = true> class LvlType {
public:
  static constexpr LvlFormat format = F;
  static constexpr bool unique = U;
  static constexpr bool ordered = O;

  // TODO: support other useful combinations
  static_assert((unique || format == LvlFormat::Compressed) && ordered);

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static constexpr bool
  isDense() {
    return format == LvlFormat::Dense;
  }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static constexpr bool
  isCompressed() {
    return format == LvlFormat::Compressed && unique;
  }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static constexpr bool
  isCompressedNU() {
    return format == LvlFormat::Compressed && !unique;
  }
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static constexpr bool
  isSingleton() {
    return format == LvlFormat::Singleton;
  }

  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static constexpr bool
  isRange() {
    return format == LvlFormat::Range;
  }

  static std::string toString() {
    if constexpr (isDense()) {
      return "dense";
    } else if constexpr (isCompressed()) {
      return "compressed";
    } else if constexpr (isCompressedNU()) {
      return "compressed(non-unique)";
    } else if constexpr (isSingleton()) {
      return "singleton";
    } else if constexpr (isRange()) {
      return "range";
    } else { // Should not happen
      return "?";
    }
  }
};

//
// A level specification consists of a level expression and a level type.
//
template <class E, class T> class LvlSpec {
public:
  using Expr = E;
  using Type = T;

  static std::string toString() {
    return Expr::toString() + " : " + Type::toString();
  }
};

//
// A tensor format consists of an implicit ordered sequence of dimension
// specifications (d0, d1, etc.) and an explicit ordered sequence of level
// specifications (e.g. d0 : Dense, d1 : Compressed).
//
template <int D, class... S> class SparseTensorFormat {
public:
  using LvlSpecs = cuda::std::tuple<S...>;
  static constexpr int DIM = D;
  static constexpr int LVL = sizeof...(S);

  static_assert(DIM <= LVL);

  static constexpr bool isSpVec() {
    if constexpr (LVL == 1) {
      using type0 = cuda::std::tuple_element_t<0, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isCompressed();
    }
    return false;
  }

  static constexpr bool isCOO() {
    if constexpr (LVL == 2) {
      using type0 = cuda::std::tuple_element_t<0, LvlSpecs>;
      using type1 = cuda::std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isCompressedNU() &&
             type1::Expr::isId(1) && type1::Type::isSingleton();
    }
    return false;
  }

  static constexpr bool isCSR() {
    if constexpr (LVL == 2) {
      using type0 = cuda::std::tuple_element_t<0, LvlSpecs>;
      using type1 = cuda::std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isDense() &&
             type1::Expr::isId(1) && type1::Type::isCompressed();
    }
    return false;
  }

  static constexpr bool isCSC() {
    if constexpr (LVL == 2) {
      using type0 = cuda::std::tuple_element_t<0, LvlSpecs>;
      using type1 = cuda::std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(1) && type0::Type::isDense() &&
             type1::Expr::isId(0) && type1::Type::isCompressed();
    }
    return false;
  }

  template <bool SZ, class CRD, int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static void
  dim2lvl(const CRD *dims, CRD *lvls) {
    if constexpr (L < LVL) {
      using ftype = cuda::std::tuple_element_t<L, LvlSpecs>;
      static_assert(ftype::Expr::di < DIM);
      if constexpr (ftype::Expr::op == LvlOp::Id) {
        lvls[L] = dims[ftype::Expr::di];
      } else if constexpr (ftype::Expr::op == LvlOp::Add) {
        static_assert(ftype::Expr::cj < DIM);
        if constexpr (SZ) // range [0:M+N-2]
          lvls[L] = dims[ftype::Expr::di] + dims[ftype::Expr::cj] - 1;
        else
          lvls[L] = dims[ftype::Expr::di] + dims[ftype::Expr::cj];
      } else if constexpr (ftype::Expr::op == LvlOp::Sub) {
        static_assert(ftype::Expr::cj < DIM);
        if constexpr (SZ) // range [-M+1:N-1]
          lvls[L] = dims[ftype::Expr::di] + dims[ftype::Expr::cj] - 1;
        else
          lvls[L] = dims[ftype::Expr::di] - dims[ftype::Expr::cj];
      } else if constexpr (ftype::Expr::op == LvlOp::Div) {
        lvls[L] = dims[ftype::Expr::di] / ftype::Expr::cj;
      } else if constexpr (ftype::Expr::op == LvlOp::Mod) {
        if constexpr (SZ) // range [0:i % C]
          lvls[L] = ftype::Expr::cj;
        else
          lvls[L] = dims[ftype::Expr::di] % ftype::Expr::cj;
      } else {
#ifndef __CUDACC__
        MATX_THROW(matxNotSupported, "unimplemented case");
#endif
      }
      if constexpr (L + 1 < LVL) {
        dim2lvl<SZ, CRD, L + 1>(dims, lvls);
      }
    }
  }

  template <class CRD, int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static void
  lvl2dim(const CRD *lvls, CRD *dims) {
    if constexpr (L < LVL) {
      using ftype = cuda::std::tuple_element_t<L, LvlSpecs>;
      static_assert(ftype::Expr::di < DIM);
      if constexpr (ftype::Expr::op == LvlOp::Id) {
        dims[ftype::Expr::di] = lvls[L];
      } else if constexpr (ftype::Expr::op == LvlOp::Add) {
        dims[ftype::Expr::cj] = lvls[L + 1] + lvls[L];
      } else if constexpr (ftype::Expr::op == LvlOp::Sub) {
        dims[ftype::Expr::cj] = lvls[L + 1] - lvls[L]; // order!
      } else if constexpr (ftype::Expr::op == LvlOp::Div) {
        dims[ftype::Expr::di] = lvls[L] * ftype::Expr::cj;
      } else if constexpr (ftype::Expr::op == LvlOp::Mod) {
        dims[ftype::Expr::di] += lvls[L]; // update (seen second)
      } else {
#ifndef __CUDACC__
        MATX_THROW(matxNotSupported, "unimplemented case");
#endif
      }
      if constexpr (L + 1 < LVL) {
        lvl2dim<CRD, L + 1>(lvls, dims);
      }
    }
  }

  static void print() {
    std::cout << "(";
    for (int d = 0; d < DIM; d++) {
      std::cout << " d" << d;
      if (d != DIM - 1)
        std::cout << ",";
    }
    std::cout << " ) -> (";
    printLevel();
    std::cout << " )" << std::endl;
  };

private:
  template <int L = 0> static void printLevel() {
    if constexpr (L < LVL) {
      using ftype = cuda::std::tuple_element_t<L, LvlSpecs>;
      std::cout << " " << ftype::toString();
      if constexpr (L + 1 < LVL) {
        std::cout << ",";
        printLevel<L + 1>();
      }
    }
  }
};

//
// Predefined common tensor formats. Note that even though the tensor
// format was introduced to define the universal sparse tensor type, the
// "all-dense" format also naturally describes dense scalars, vectors,
// matrices, and tensors, with all dimension-major format variants.
//

// Dimension short-cuts (d0, d1, d3, d4).
using D0 = LvlExpr<LvlOp::Id, 0>;
using D1 = LvlExpr<LvlOp::Id, 1>;
using D2 = LvlExpr<LvlOp::Id, 2>;
using D3 = LvlExpr<LvlOp::Id, 3>;
using D4 = LvlExpr<LvlOp::Id, 4>;

// Operation short-cuts.
template <int I, int J> using Add = LvlExpr<LvlOp::Add, I, J>;
template <int I, int J> using Sub = LvlExpr<LvlOp::Sub, I, J>;
template <int I, int J> using Div = LvlExpr<LvlOp::Div, I, J>;
template <int I, int J> using Mod = LvlExpr<LvlOp::Mod, I, J>;

// Level type short-cuts.
using Dense = LvlType<LvlFormat::Dense>;
using Compressed = LvlType<LvlFormat::Compressed>;
using CompressedNU = LvlType<LvlFormat::Compressed, false>;
using Singleton = LvlType<LvlFormat::Singleton>;
using Range = LvlType<LvlFormat::Range>;

// Scalars.
using Scalar = SparseTensorFormat<0>;

// Vectors.
using DnVec = SparseTensorFormat<1, LvlSpec<D0, Dense>>;
using SpVec = SparseTensorFormat<1, LvlSpec<D0, Compressed>>;

// Dense Matrices.
using DnMat = SparseTensorFormat<2, LvlSpec<D0, Dense>, LvlSpec<D1, Dense>>;
using DnMatCol = SparseTensorFormat<2, LvlSpec<D1, Dense>, LvlSpec<D0, Dense>>;

// Sparse Matrices.
using COO =
    SparseTensorFormat<2, LvlSpec<D0, CompressedNU>, LvlSpec<D1, Singleton>>;
using CSR = SparseTensorFormat<2, LvlSpec<D0, Dense>, LvlSpec<D1, Compressed>>;
using CSC = SparseTensorFormat<2, LvlSpec<D1, Dense>, LvlSpec<D0, Compressed>>;
using DCSR =
    SparseTensorFormat<2, LvlSpec<D0, Compressed>, LvlSpec<D1, Compressed>>;
using DCSC =
    SparseTensorFormat<2, LvlSpec<D1, Compressed>, LvlSpec<D0, Compressed>>;
using CROW = SparseTensorFormat<2, LvlSpec<D0, Compressed>, LvlSpec<D1, Dense>>;
using CCOL = SparseTensorFormat<2, LvlSpec<D1, Compressed>, LvlSpec<D0, Dense>>;
using DIA =
    SparseTensorFormat<2, LvlSpec<Sub<1, 0>, Compressed>, LvlSpec<D1, Range>>;
using SkewDIA =
    SparseTensorFormat<2, LvlSpec<Add<1, 0>, Compressed>, LvlSpec<D1, Range>>;

// Sparse Block Matrices.
template <int M, int N>
using BSR =
    SparseTensorFormat<2, LvlSpec<Div<0, M>, Dense>,
                       LvlSpec<Div<1, N>, Compressed>,
                       LvlSpec<Mod<0, M>, Dense>, LvlSpec<Mod<1, N>, Dense>>;
template <int M, int N>
using BSRCol =
    SparseTensorFormat<2, LvlSpec<Div<0, M>, Dense>,
                       LvlSpec<Div<1, N>, Compressed>,
                       LvlSpec<Mod<1, N>, Dense>, LvlSpec<Mod<0, M>, Dense>>;

// 3-D Tensors.
using Dn3 = SparseTensorFormat<3, LvlSpec<D0, Dense>, LvlSpec<D1, Dense>,
                               LvlSpec<D2, Dense>>;
using COO3 = SparseTensorFormat<3, LvlSpec<D0, CompressedNU>,
                                LvlSpec<D1, Singleton>, LvlSpec<D2, Singleton>>;
using CSF3 =
    SparseTensorFormat<3, LvlSpec<D0, Compressed>, LvlSpec<D1, Compressed>,
                       LvlSpec<D2, Compressed>>;

// Sparse Block 3-D Tensors.
template <int M, int N, int K>
using BSR3 = SparseTensorFormat<
    3, LvlSpec<Div<0, M>, Dense>, LvlSpec<Div<1, N>, Compressed>,
    LvlSpec<Div<2, K>, Compressed>, LvlSpec<Mod<0, M>, Dense>,
    LvlSpec<Mod<1, N>, Dense>, LvlSpec<Mod<2, K>, Dense>>;

// 4-D Tensors.
using Dn4 = SparseTensorFormat<4, LvlSpec<D0, Dense>, LvlSpec<D1, Dense>,
                               LvlSpec<D2, Dense>, LvlSpec<D3, Dense>>;
using COO4 =
    SparseTensorFormat<4, LvlSpec<D0, CompressedNU>, LvlSpec<D1, Singleton>,
                       LvlSpec<D2, Singleton>, LvlSpec<D3, Singleton>>;
using CSF4 =
    SparseTensorFormat<4, LvlSpec<D0, Compressed>, LvlSpec<D1, Compressed>,
                       LvlSpec<D2, Compressed>, LvlSpec<D3, Compressed>>;

// 5-D Tensors.
using Dn5 = SparseTensorFormat<5, LvlSpec<D0, Dense>, LvlSpec<D1, Dense>,
                               LvlSpec<D2, Dense>, LvlSpec<D3, Dense>,
                               LvlSpec<D4, Dense>>;
using COO5 = SparseTensorFormat<5, LvlSpec<D0, CompressedNU>,
                                LvlSpec<D1, Singleton>, LvlSpec<D2, Singleton>,
                                LvlSpec<D3, Singleton>, LvlSpec<D4, Singleton>>;
using CSF5 =
    SparseTensorFormat<5, LvlSpec<D0, Compressed>, LvlSpec<D1, Compressed>,
                       LvlSpec<D2, Compressed>, LvlSpec<D3, Compressed>,
                       LvlSpec<D4, Compressed>>;

} // namespace experimental
} // namespace matx
