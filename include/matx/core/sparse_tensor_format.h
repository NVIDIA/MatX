////////////////////////////////////////////////////////////////////////////////
// BSD 3-Clause License
//
// Copyright (c) 2025, NVIDIA Corporation
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

#include <string>

namespace matx {
namespace experimental {

//
// MatX implements a universal sparse tensor type that uses a tensor format
// DSL (Domain Specific Language) to describe a vast space of storage formats.
// Although the tensor format can easily define many common storage formats
// (such as Dense, CSR, CSC, BSR), it can also define many less common storage
// formats. In addition, the tensor format DSL can be extended to include even
// more storage formats in the future.
//
// In the tensor format, the term **dimension** is used to refer to the axes of
// the semantic tensor (as seen by the user), and the term **level** to refer to
// the axes of the actual storage format (how it eventually resides in memory).
//
// The tensor format contains a map that provides the following:
//
// (1) An ordered sequence of dimension specifications, each of which includes:
//
//     (*) a dimension-expression, which provides a reference to each dimension
//
// (2) An ordered sequence of level specifications, each of which includes:
//
//     (*) a level expression, which defines what is stored in each level
//     (*) a required level type, which defines how the level is stored,
//     including:
//         (+) a required level format
//         (+) a collection of level properties
//
// Currently, the following level formats are supported:
//
// (1) dense: level is dense, entries along the level are stored and linearized
// (2) compressed: level is sparse, only nonzeros along the level are stored
//     with the compact positions and coordinates encoding
// (3) singleton: a variant of the compressed format, for when coordinates have
//     no siblings
//
// All level formats have the following level properties:
//
// (1) non/unique (are duplicates allowed at that level),
// (2) un/ordered (are coordinates sorted at that level).
//
// Matrix Examples (dimension == 2, level >= dimension)
//
// COO:
//     map = (i, j) -> ( i : compressed(non-unique), j : singleton )
//
// CSR:
//     map = (i, j) -> ( i : dense, j : compressed )
//
// DCSR:
//     map = (i, j) -> ( i : compressed, j : compressed )
//
// CSC:
//     map = (i, j) -> ( j : dense, i : compressed )
//
// BSR with 2x3 blocks:
//      map = ( i, j ) -> ( i floordiv 2 : dense,
//                          j floordiv 3 : compressed,
//                          i mod 2      : dense,
//                          j mod 3      : dense )
//
// Tensor Examples (dimension > 2, level >= dimension)
//
// COO3:
//     map = (i, j, k) -> ( i : compressed(non-unique),
//                          j : singleton,
//                          k : singleton )
//
// CSF3:
//     map = (i, j, k) -> ( i : compressed,
//                          j : compressed,
//                          k : compressed )
//
// The idea of a universal sparse tensor type has its roots in
// sparse compilers, first pioneered for sparse linear algebra in [Bik96]
// and formalized to sparse tensor algebra in [Kjolstad20]. The generalization
// to higher-dimensional levels was introduced in [MLIR22].
//
// [Bik96] Aart J.C. Bik. Compiler Support for Sparse Matrix Computations.
//         PhD thesis, Leiden University, May 1996.
// [Kjolstad20] Fredrik Berg Kjolstad. Sparse Tensor Algebra Compilation.
//              PhD thesis, MIT, February, 2020.
// [MLIR22] Aart J.C. Bik, Penporn Koanantakool, Tatiana Shpeisman,
//        Nicolas Vasilache, Bixia Zheng, and Fredrik Kjolstad.
//        Compiler Support for Sparse Tensor Computations in MLIR.
//        ACM Transactions on Architecture and Code Optimization, June, 2022.
//

//
// A level type consists of a level format together with a set of
// level properties (ordered and unique by default).
//
// TODO: split out format and properties, generalize into class
//
enum class LvlType { Dense, Singleton, Compressed, CompressedNonUnique };

//
// A level expression consists of an expression in terms of dimension
// variables. Currently, the following expressions are supported:
//
// (1) di
// (2) di div 2
// (3) di mod 2
//
enum class LvlOp { Id, Div, Mod };
template <LvlOp o, int d, int c> class LvlExpr {
public:
  static constexpr LvlOp op = o;
  static constexpr int di = d;
  static constexpr int cj = c;

  static std::string toString() {
    if constexpr (op == LvlOp::Id) {
      return "d" + std::to_string(di);
    } else if constexpr (op == LvlOp::Div) {
      return "d" + std::to_string(di) + " div " + std::to_string(cj);
    } else if constexpr (op == LvlOp::Mod) {
      return "d" + std::to_string(di) + " mod " + std::to_string(cj);
    } else { // Should not happen
      return "";
    }     
  }
};

//
// A level specification consists of a level expression and a level type.
//
template <typename Expr, LvlType ltype> class LvlSpec {
public:
  using expr = Expr;
  static constexpr LvlType lvltype = ltype;

  static std::string toString() {
    return Expr::toString() + " : " + LvlTypeToString();
  }

  // TODO: move to LvlType when that class evolves
  static inline std::string LvlTypeToString() {
    if constexpr (ltype == LvlType::Dense) {
      return "dense";
    } else if constexpr (ltype == LvlType::Singleton) {
      return "singleton";
    } else if constexpr (ltype == LvlType::Compressed) {
      return "compressed";
    } else if constexpr (ltype == LvlType::CompressedNonUnique) {
      return "compressed(non-unique)";
    } else { // Should not happen
      return "";
    } 
  }
};

//
// A tensor format consists of an implicit ordered sequence of dimension
// specifications (d0, d1, etc.) and an explicit ordered sequence of level
// specifications (e.g. d0 : Dense, d1 : Compressed).
//
template <int D, typename... LvlSpecs> class SparseTensorFormat {
public:
  using LVLSPECS = std::tuple<LvlSpecs...>;
  static constexpr int DIM = D;
  static constexpr int LVL = sizeof...(LvlSpecs);

  static_assert(DIM <= LVL);

  static constexpr bool isSpVec() {
    if constexpr (LVL == 1) {
      using first_type = std::tuple_element_t<0, LVLSPECS>;
      return first_type::lvltype == LvlType::Compressed &&
             first_type::expr::op == LvlOp::Id && first_type::expr::di == 0;
    }
    return false;
  }

  static constexpr bool isCOO() {
    if constexpr (LVL == 2) {
      using first_type = std::tuple_element_t<0, LVLSPECS>;
      using second_type = std::tuple_element_t<1, LVLSPECS>;
      return first_type::lvltype == LvlType::CompressedNonUnique &&
             first_type::expr::op == LvlOp::Id && first_type::expr::di == 0 &&
             second_type::lvltype == LvlType::Singleton &&
             second_type::expr::op == LvlOp::Id && second_type::expr::di == 1;
    }
    return false;
  }

  static constexpr bool isCSR() {
    if constexpr (LVL == 2) {
      using first_type = std::tuple_element_t<0, LVLSPECS>;
      using second_type = std::tuple_element_t<1, LVLSPECS>;
      return first_type::lvltype == LvlType::Dense &&
             first_type::expr::op == LvlOp::Id && first_type::expr::di == 0 &&
             second_type::lvltype == LvlType::Compressed &&
             second_type::expr::op == LvlOp::Id && second_type::expr::di == 1;
    }
    return false;
  }

  static constexpr bool isCSC() {
    if constexpr (LVL == 2) {
      using first_type = std::tuple_element_t<0, LVLSPECS>;
      using second_type = std::tuple_element_t<1, LVLSPECS>;
      return first_type::lvltype == LvlType::Dense &&
             first_type::expr::op == LvlOp::Id && first_type::expr::di == 1 &&
             second_type::lvltype == LvlType::Compressed &&
             second_type::expr::op == LvlOp::Id && second_type::expr::di == 0;
    }
    return false;
  }

  template <typename CRD, int L = 0>
  __MATX_INLINE__ __MATX_HOST__ __MATX_DEVICE__ static void
  dim2lvl(const CRD *dims, CRD *lvls, bool asSize) {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, LVLSPECS>;
      if constexpr (ftype::expr::op == LvlOp::Id) {
        lvls[L] = dims[ftype::expr::di];
      } else if constexpr (ftype::expr::op == LvlOp::Div) {
        lvls[L] = dims[ftype::expr::di] / ftype::expr::cj;
      } else if constexpr (ftype::expr::op == LvlOp::Mod) {
        lvls[L] = asSize ? ftype::expr::cj
                         : (dims[ftype::expr::di] % ftype::expr::cj);
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
      using ftype = std::tuple_element_t<L, LVLSPECS>;
      if constexpr (ftype::expr::op == LvlOp::Id) {
        dims[ftype::expr::di] = lvls[L];
      } else if constexpr (ftype::expr::op == LvlOp::Div) {
        dims[ftype::expr::di] = lvls[L] * ftype::expr::cj;
      } else if constexpr (ftype::expr::op == LvlOp::Mod) {
        dims[ftype::expr::di] += lvls[L]; // update (seen second)
      }
      if constexpr (L + 1 < LVL) {
        lvl2dim<CRD, L + 1>(lvls, dims);
      }
    }
  }

  template <int L = 0> static void printLevel() {
    if constexpr (L < LVL) {
      using ftype = std::tuple_element_t<L, LVLSPECS>;
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

// Scalars.
using Scalar = SparseTensorFormat<0>;

// Vectors.
using DnVec = SparseTensorFormat<1, LvlSpec<D0, LvlType::Dense>>;
using SpVec = SparseTensorFormat<1, LvlSpec<D1, LvlType::Compressed>>;

// Dense Matrices.
using DnMat = SparseTensorFormat<2, LvlSpec<D0, LvlType::Dense>,
                                 LvlSpec<D1, LvlType::Dense>>;
using DnMatCol = SparseTensorFormat<2, LvlSpec<D1, LvlType::Dense>,
                                    LvlSpec<D0, LvlType::Dense>>;

// Sparse Matrices.
using COO = SparseTensorFormat<2, LvlSpec<D0, LvlType::CompressedNonUnique>,
                               LvlSpec<D1, LvlType::Singleton>>;
using CSR = SparseTensorFormat<2, LvlSpec<D0, LvlType::Dense>,
                               LvlSpec<D1, LvlType::Compressed>>;
using CSC = SparseTensorFormat<2, LvlSpec<D1, LvlType::Dense>,
                               LvlSpec<D0, LvlType::Compressed>>;
using DCSR = SparseTensorFormat<2, LvlSpec<D0, LvlType::Compressed>,
                                LvlSpec<D1, LvlType::Compressed>>;
using DCSC = SparseTensorFormat<2, LvlSpec<D1, LvlType::Compressed>,
                                LvlSpec<D0, LvlType::Compressed>>;
using CROW = SparseTensorFormat<2, LvlSpec<D0, LvlType::Compressed>,
                                LvlSpec<D1, LvlType::Dense>>;
using CCOL = SparseTensorFormat<2, LvlSpec<D1, LvlType::Compressed>,
                                LvlSpec<D0, LvlType::Dense>>;

// Sparse Block Matrices.
template <int m, int n>
using BSR =
    SparseTensorFormat<2, LvlSpec<LvlExpr<LvlOp::Div, 0, m>, LvlType::Dense>,
                       LvlSpec<LvlExpr<LvlOp::Div, 1, n>, LvlType::Compressed>,
                       LvlSpec<LvlExpr<LvlOp::Mod, 0, m>, LvlType::Dense>,
                       LvlSpec<LvlExpr<LvlOp::Mod, 1, n>, LvlType::Dense>>;

// 3-D Tensors.
using Dn3 = SparseTensorFormat<3, LvlSpec<D0, LvlType::Dense>,
                               LvlSpec<D1, LvlType::Dense>,
                               LvlSpec<D2, LvlType::Dense>>;
using COO3 = SparseTensorFormat<3, LvlSpec<D0, LvlType::CompressedNonUnique>,
                                LvlSpec<D1, LvlType::Singleton>,
                                LvlSpec<D2, LvlType::Singleton>>;
using CSF3 = SparseTensorFormat<3, LvlSpec<D0, LvlType::Compressed>,
                                LvlSpec<D1, LvlType::Compressed>,
                                LvlSpec<D2, LvlType::Compressed>>;

// 4-D Tensors.
using Dn4 =
    SparseTensorFormat<4, LvlSpec<D0, LvlType::Dense>,
                       LvlSpec<D1, LvlType::Dense>, LvlSpec<D2, LvlType::Dense>,
                       LvlSpec<D3, LvlType::Dense>>;
using COO4 = SparseTensorFormat<4, LvlSpec<D0, LvlType::CompressedNonUnique>,
                                LvlSpec<D1, LvlType::Singleton>,
                                LvlSpec<D2, LvlType::Singleton>,
                                LvlSpec<D3, LvlType::Singleton>>;
using CSF4 = SparseTensorFormat<
    4, LvlSpec<D0, LvlType::Compressed>, LvlSpec<D1, LvlType::Compressed>,
    LvlSpec<D2, LvlType::Compressed>, LvlSpec<D3, LvlType::Compressed>>;

// 5-D Tensors.
using Dn5 =
    SparseTensorFormat<5, LvlSpec<D0, LvlType::Dense>,
                       LvlSpec<D1, LvlType::Dense>, LvlSpec<D2, LvlType::Dense>,
                       LvlSpec<D3, LvlType::Dense>,
                       LvlSpec<D4, LvlType::Dense>>;
using COO5 = SparseTensorFormat<
    5, LvlSpec<D0, LvlType::CompressedNonUnique>,
    LvlSpec<D1, LvlType::Singleton>, LvlSpec<D2, LvlType::Singleton>,
    LvlSpec<D3, LvlType::Singleton>, LvlSpec<D4, LvlType::Singleton>>;
using CSF5 = SparseTensorFormat<
    5, LvlSpec<D0, LvlType::Compressed>, LvlSpec<D1, LvlType::Compressed>,
    LvlSpec<D2, LvlType::Compressed>, LvlSpec<D3, LvlType::Compressed>,
    LvlSpec<D4, LvlType::Compressed>>;

} // namespace experimental
} // namespace matx
