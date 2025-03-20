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
// variables (e.g. di, di div 2, or di mod 2).
//
enum class LvlOp { Id, Div, Mod };
template <LvlOp o, int d, int c> class LvlExpr {
public:
  static constexpr LvlOp op = o;
  static constexpr int di = d;
  static constexpr int cj = c;

  static constexpr bool isId(int i) { return op == LvlOp::Id && di == i; }

  static std::string toString() {
    if constexpr (op == LvlOp::Id) {
      return "d" + std::to_string(di);
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
enum class LvlFormat { Dense, Compressed, Singleton };
template <LvlFormat f, bool u = true, bool o = true> class LvlType {
public:
  static constexpr LvlFormat format = f;
  static constexpr bool unique = u;
  static constexpr bool ordered = o;

  static_assert(ordered);

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

  static std::string toString() {
    if constexpr (isDense()) {
      return "dense";
    } else if constexpr (isCompressed()) {
      return "compressed";
    } else if constexpr (isCompressedNU()) {
      return "compressed(non-unique)";
    } else if constexpr (isSingleton()) {
      return "singleton";
    } else { // Should not happen
      return "?";
    }
  }
};

//
// A level specification consists of a level expression and a level type.
//
template <typename E, typename T> class LvlSpec {
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
template <int D, typename... S> class SparseTensorFormat {
public:
  using LvlSpecs = std::tuple<S...>;
  static constexpr int DIM = D;
  static constexpr int LVL = sizeof...(S);

  static_assert(DIM <= LVL);

  static constexpr bool isSpVec() {
    if constexpr (LVL == 1) {
      using type0 = std::tuple_element_t<0, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isCompressed();
    }
    return false;
  }

  static constexpr bool isCOO() {
    if constexpr (LVL == 2) {
      using type0 = std::tuple_element_t<0, LvlSpecs>;
      using type1 = std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isCompressedNU() &&
             type1::Expr::isId(1) && type1::Type::isSingleton();
    }
    return false;
  }

  static constexpr bool isCSR() {
    if constexpr (LVL == 2) {
      using type0 = std::tuple_element_t<0, LvlSpecs>;
      using type1 = std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(0) && type0::Type::isDense() &&
             type1::Expr::isId(1) && type1::Type::isCompressed();
    }
    return false;
  }

  static constexpr bool isCSC() {
    if constexpr (LVL == 2) {
      using type0 = std::tuple_element_t<0, LvlSpecs>;
      using type1 = std::tuple_element_t<1, LvlSpecs>;
      return type0::Expr::isId(1) && type0::Type::isDense() &&
             type1::Expr::isId(0) && type1::Type::isCompressed();
    }
    return false;
  }

  template <typename CRD, int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static void
  dim2lvl(const CRD *dims, CRD *lvls, bool asSize) {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, LvlSpecs>;
      if constexpr (ftype::Expr::op == LvlOp::Id) {
        lvls[L] = dims[ftype::Expr::di];
      } else if constexpr (ftype::Expr::op == LvlOp::Div) {
        lvls[L] = dims[ftype::Expr::di] / ftype::expr::cj;
      } else if constexpr (ftype::Expr::op == LvlOp::Mod && asSize) {
        lvls[L] = ftype::Expr::cj;
      } else if constexpr (ftype::Expr::op == LvlOp::Mod && !asSize) {
        lvls[L] = dims[ftype::Expr::di] % ftype::Expr::cj;
      }
      if constexpr (L + 1 < LVL) {
        dim2lvl<CRD, L + 1>(dims, lvls, asSize);
      }
    }
  }

  template <typename CRD, int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static void
  lvl2dim(const CRD *lvls, CRD *dims) {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, LvlSpecs>;
      if constexpr (ftype::Expr::op == LvlOp::Id) {
        dims[ftype::Expr::di] = lvls[L];
      } else if constexpr (ftype::Expr::op == LvlOp::Div) {
        dims[ftype::Expr::di] = lvls[L] * ftype::expr::cj;
      } else if constexpr (ftype::Expr::op == LvlOp::Mod) {
        dims[ftype::Expr::di] += lvls[L]; // update (seen second)
      }
      if constexpr (L + 1 < LVL) {
        lvl2dim<CRD, L + 1>(lvls, dims);
      }
    }
  }

  template <int L = 0> static void printLevel() {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, LvlSpecs>;
      std::cout << " " << ftype::toString();
      if constexpr (L + 1 < LVL) {
        std::cout << ",";
        printLevel<L + 1>();
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
};

//
// Predefined common tensor formats. Note that even though the tensor
// format was introduced to define the universal sparse tensor type, the
// "all-dense" format also naturally describes dense scalars, vectors,
// matrices, and tensors, with all d-major format variants.
//

// Dimension short-cuts (d0, d1, d3, d4).
using D0 = LvlExpr<LvlOp::Id, 0, 1>;
using D1 = LvlExpr<LvlOp::Id, 1, 1>;
using D2 = LvlExpr<LvlOp::Id, 2, 1>;
using D3 = LvlExpr<LvlOp::Id, 3, 1>;
using D4 = LvlExpr<LvlOp::Id, 4, 1>;

// Level type short-cuts.
using dense = LvlType<LvlFormat::Dense>;
using compressed = LvlType<LvlFormat::Compressed>;
using compressedNU = LvlType<LvlFormat::Compressed, false>;
using singleton = LvlType<LvlFormat::Singleton>;

// Scalars.
using Scalar = SparseTensorFormat<0>;

// Vectors.
using DnVec = SparseTensorFormat<1, LvlSpec<D0, dense>>;
using SpVec = SparseTensorFormat<1, LvlSpec<D0, compressed>>;

// dense Matrices.
using DnMat = SparseTensorFormat<2, LvlSpec<D0, dense>, LvlSpec<D1, dense>>;
using DnMatCol = SparseTensorFormat<2, LvlSpec<D1, dense>, LvlSpec<D0, dense>>;

// Sparse Matrices.
using COO =
    SparseTensorFormat<2, LvlSpec<D0, compressedNU>, LvlSpec<D1, singleton>>;
using CSR = SparseTensorFormat<2, LvlSpec<D0, dense>, LvlSpec<D1, compressed>>;
using CSC = SparseTensorFormat<2, LvlSpec<D1, dense>, LvlSpec<D0, compressed>>;
using DCSR =
    SparseTensorFormat<2, LvlSpec<D0, compressed>, LvlSpec<D1, compressed>>;
using DCSC =
    SparseTensorFormat<2, LvlSpec<D1, compressed>, LvlSpec<D0, compressed>>;
using CROW = SparseTensorFormat<2, LvlSpec<D0, compressed>, LvlSpec<D1, dense>>;
using CCOL = SparseTensorFormat<2, LvlSpec<D1, compressed>, LvlSpec<D0, dense>>;

// Sparse Block Matrices.
template <int m, int n>
using BSR = SparseTensorFormat<2, LvlSpec<LvlExpr<LvlOp::Div, 0, m>, dense>,
                               LvlSpec<LvlExpr<LvlOp::Div, 1, n>, compressed>,
                               LvlSpec<LvlExpr<LvlOp::Mod, 0, m>, dense>,
                               LvlSpec<LvlExpr<LvlOp::Mod, 1, n>, dense>>;

// 3-D Tensors.
using Dn3 = SparseTensorFormat<3, LvlSpec<D0, dense>, LvlSpec<D1, dense>,
                               LvlSpec<D2, dense>>;
using COO3 = SparseTensorFormat<3, LvlSpec<D0, compressedNU>,
                                LvlSpec<D1, singleton>, LvlSpec<D2, singleton>>;
using CSF3 =
    SparseTensorFormat<3, LvlSpec<D0, compressed>, LvlSpec<D1, compressed>,
                       LvlSpec<D2, compressed>>;

// 4-D Tensors.
using Dn4 = SparseTensorFormat<4, LvlSpec<D0, dense>, LvlSpec<D1, dense>,
                               LvlSpec<D2, dense>, LvlSpec<D3, dense>>;
using COO4 =
    SparseTensorFormat<4, LvlSpec<D0, compressedNU>, LvlSpec<D1, singleton>,
                       LvlSpec<D2, singleton>, LvlSpec<D3, singleton>>;
using CSF4 =
    SparseTensorFormat<4, LvlSpec<D0, compressed>, LvlSpec<D1, compressed>,
                       LvlSpec<D2, compressed>, LvlSpec<D3, compressed>>;

// 5-D Tensors.
using Dn5 = SparseTensorFormat<5, LvlSpec<D0, dense>, LvlSpec<D1, dense>,
                               LvlSpec<D2, dense>, LvlSpec<D3, dense>,
                               LvlSpec<D4, dense>>;
using COO5 = SparseTensorFormat<5, LvlSpec<D0, compressedNU>,
                                LvlSpec<D1, singleton>, LvlSpec<D2, singleton>,
                                LvlSpec<D3, singleton>, LvlSpec<D4, singleton>>;
using CSF5 =
    SparseTensorFormat<5, LvlSpec<D0, compressed>, LvlSpec<D1, compressed>,
                       LvlSpec<D2, compressed>, LvlSpec<D3, compressed>,
                       LvlSpec<D4, compressed>>;

} // namespace experimental
} // namespace matx
