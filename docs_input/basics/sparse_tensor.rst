.. _sparse_tensor_api:

Sparse Tensor Type
##################

MatX is in the process of adding experimental support for sparse tensors.
The implementation is based on the **Universal Sparse Tensor (UST)** type
that uses a tensor format DSL (Domain Specific Language) to describe a vast
space of storage formats. Although the UST type can easily define many common
storage formats (such as dense vectors and matrices, sparse vectors, sparse
matrices in formats like COO, CSR, CSC, DCSR, DCSC, BSR, BSC, DIA, and with
generalizations to sparse tensors), it can also define many less common storage
formats. From MatX's perspective, the advantage of using the UST type (rather
than various specific sparse storage formats) is that the framework code only
has to deal with a single new sparse type (and only dispatch to specific formats
when required by a high performance library implementation). Also, the tensor
format DSL can be easily extended to include even more sparse storage formats
in the future. From the user's perspective, the UST type provides more
flexibility in changing storage formats by merely changing annotations in the
type definitions, which allows for rapid experimentation with different ways
of storing sparse tensors in a MatX computation.

Quick Start
-----------

Despite the forward looking design of using the UST type, the current
experimental support provides a few factory methods with the common
formats COO, CSR, CSC, and DIA. The factory methods look similar to e.g.
sparse construction methods found in SciPy sparse or torch sparse.

For example, to create a COO representation of the following
4x8 matrix with 5 nonzero elements::

       | 1, 2, 0, 0, 0, 0, 0, 0 |
   A = | 0, 0, 0, 0, 0, 0, 0, 0 |
       | 0, 0, 0, 0, 0, 0, 0, 0 |
       | 0, 0, 3, 4, 0, 5, 0, 0 |

First, using a uniform memory space, set up the constituent 1-dim buffers
that contain, respectively, the value, i-index, and j-index of each nonzero
element, ordered lexicographically by row-then-column index, as follows::
  
  auto vals = make_tensor<float>({5});
  auto idxi = make_tensor<int>({5});
  auto idxj = make_tensor<int>({5});
  vals.SetVals({1, 2, 3, 4, 5});
  idxi.SetVals({0, 0, 3, 3, 3});
  idxj.SetVals({0, 1, 2, 3, 5});

Then, the COO representation of ``A``, residing in the same memory space as
its constituent buffers is constructed as follows::

  auto Acoo = experimental::make_tensor_coo(vals, idxi, idxj, {4, 8});

  print(Acoo);

The result of the print statement is shown below::

  tensor_impl_2_f32: SparseTensor{float} Rank: 2, Sizes:[4, 8], Levels:[4, 8]
  format = ( d0, d1 ) -> ( d0 : compressed(non-unique), d1 : singleton )
  space  = CUDA managed memory
  nse    = 5
  pos[0] = ( 0  5 )
  crd[0] = ( 0  0  3  3  3 )
  crd[1] = ( 0  1  2  3  5 )
  values = ( 1.0000e+00  2.0000e+00  3.0000e+00  4.0000e+00  5.0000e+00 )

Note that, like dense tensors, sparse tensors provide ()-operations
for indexing.  However, users should **never** use the ()-operator
in performance critical code, since sparse storage formats do not
provide O(1) random access to their elements (compressed levels will
use some form of search to determine if an element is present)::

  // Naive way to convert the sparse matrix back to a dense matrix.
  auto A = make_tensor<float>({4, 8});
  for (index_t i = 0; i < 4; i++) {
    for (index_t j = 0; j < 8; j++) {
      A(i, j) = Acoo(i, j);
    }
  }

Instead, conversions (and other operations) should use sparse operations
that are specifically optimized for the sparse storage format. The
correct way of performing the conversion above is as follows::

  auto A = make_tensor<float>({4, 8});
  (A = sparse2dense(Acoo)).run(exec);

The current experimental sparse support in MatX provides efficient
operations for sparse-to-dense, dense-to-sparse, sparse-to-sparse,
matvec, matmul, and solve::

   (A = sparse2dense(Acoo)).run(exec);
   (Acoo = dense2sparse(D)).run(exec);
   (Acsr = sparse2sparse(Acoo)).run(exec);
   (V = matvec(Acoo, W)).run(exec); // only Sparse-Matrix x Vector (SpMV)
   (C = matmul(Acoo, B)).run(exec); // only Sparse-Matrix x Matrix (SpMM)
   (X = solve(Acsr, Y)).run(exec);  // only on CSR or (batched) tri-DIA format

We expect the assortment of supported sparse operations and storage
formats to grow if the experimental implementation is well-received.

Matx Sparse Tensor Factory Methods
----------------------------------

The MatX implementation of the factory methods for common cases of
the UST type can be found in the `make_sparse_tensor.h`_ file.
All methods build a sparse tensor storage format from constituent
1-dim buffers similar to methods found in SciPy or torch sparse.
A sample usage was already shown above. Currently only methods
to construct COO, CSR, CSC, and DIA are provided::

  // Constructs a sparse matrix in COO format directly from the values and
  // the two coordinates vectors. The entries should be sorted by row, then
  // column. Duplicate entries should not occur. Explicit zeros may be stored.
  template <typename ValTensor, typename CrdTensor>
  auto make_tensor_coo(ValTensor &val,
                       CrdTensor &row,
                       CrdTensor &col, const index_t (&shape)[2]);

  // Constructs a sparse matrix in CSR format directly from the values, the
  // row positions, and column coordinates vectors. The entries should be
  // sorted by row, then column. Duplicate entries should not occur. Explicit
  // zeros may be stored.
  template <typename ValTensor, typename PosTensor, typename CrdTensor>
  auto make_tensor_csr(ValTensor &val,
                       PosTensor &rowp,
                       CrdTensor &col, const index_t (&shape)[2]);

  // Constructs a sparse matrix in CSC format directly from the values, the
  // column positions, and row coordinates vectors. The entries should be
  // sorted by columns, then row. Duplicate entries should not occur. Explicit
  // zeros may be stored.
  template <typename ValTensor, typename PosTensor, typename CrdTensor>
  auto make_tensor_csc(ValTensor &val,
                       PosTensor &colp,
                       CrdTensor &row, const index_t (&shape)[2]);


  // Constructs a sparse matrix in DIA format directly from the values and the
  // offset vectors. For an m x n matrix, this format uses a linearized storage
  // where each diagonal has m or n entries and is accessed by either index I or
  // index J, respectively. For index I, diagonals are padded with zeros on the
  // left for the lower triangular part and padded with zeros on the right for
  // the upper triagonal part. This is vv. when using index J. This format is
  // most efficient for matrices with only a few nonzero diagonals that are
  // close to the main diagonal.
  template <typename IDX, typename ValTensor, typename CrdTensor>
  auto make_tensor_dia(ValTensor &val,
                       CrdTensor &off,
                       const index_t (&shape)[2]);

  // Constructs a sparse tensor in uniform batched DIA format directly from
  // the values and the offset vectors. For a b x m x n tensor, this format
  // effectively stores b times m x n matrices in DIA format, using a uniform
  // nonzero structure for each (non-uniform formats are possible as well).
  // All diagonals are stored consecutively in linearized format, sorted lower
  // to upper, with all diagonals at a certain offset appearing consecutively
  // for all batches. With DIA(b,i,j) as indexing, can be indexed by i or j.
  template <typename IDX, typename ValTensor, typename CrdTensor>
  auto make_tensor_uniform_batched_dia(ValTensor &val,
                                       CrdTensor &off,
                                       const index_t (&shape)[2]);

Matx Implementation of the UST Type
-----------------------------------

The MatX implementation of the UST type can be found in the `sparse_tensor.h`_
file. Similar to a dense tensor ``tensor_t``, the ``sparse_tensor_t`` is a
memory-backed, reference-counted operator that contains metadata about the
size, rank, and other properties, such as the storage format. Unlike dense
tensors, that consist of primary storage for the elements only, a sparse tensor
format consists of **primary storage** for the nonzero values (named ``values``
when printed) and **secondary storage** (named ``pos[]`` and ``crd[]``,
respectively, for each level, when printed) to indicate the position of each
nonzero value. Note that this latter storage is not called metadata on purpose,
to not confuse it with the other metadata properties mentioned above.

The type of primary and secondary storage can be anything that is accessible
to where the tensor is being used, including device memory, managed memory,
and host memory. MatX sparse tensors are very similar to e.g. SciPy's or
cuPy sparse arrays.

The implementation of the UST follows the MatX design philosophy of using
a header-only, ``constexpr``-heavy, templated approach, which facilitates
applications to only compile what is used, and nothing more.
The ``sparse_tensor_t`` type is essentially the following class,
where the tensor format ``TF`` is part of the template::

  template <typename VAL, typename CRD, typename POS, typename TF, ...>
  class sparse_tensor_t : public detail::tensor_impl_t<...> {
    
    static constexpr int DIM = TF::DIM;
    static constexpr int LVL = TF::LVL;

  private:
    // Primary storage of sparse tensor (explicitly stored element values).
    StorageV values_;

    // Secondary storage of sparse tensor (coordinates and positions).
    StorageC coordinates_[LVL];
    StorageP positions_[LVL];
  }

Using this design, many tests (e.g. is this tensor in COO format) 
evaluate as ``constexpr`` at compile-time, keeping the binary
size restricted to only what is actually used in a MatX computation.


Matx Implementation of the Tensor Format DSL
--------------------------------------------

The MatX implementation of the tensor format DSL can be found in the
`sparse_tensor_format.h`_ file. Most users do not have to concern
themselves with the details of this DSL, but can directly use predefined
type definitions for common tensor formats, like COO and CSR.

In the tensor format DSL, the term **dimension** is used to refer to the axes of
the semantic tensor (as seen by the user), and the term **level** to refer to
the axes of the actual storage format (how it eventually resides in memory).

The tensor format contains a map that provides the following:

(1) An ordered sequence of dimension specifications, each of which includes:

    * a **dimension-expression**, which provides a reference to each dimension

(2) An ordered sequence of level specifications, each of which includes:

    * a **level expression**, which defines what is stored in each level
    * a required **level type**, which defines how the level is stored, including:

      * a required **level format**
      * a collection of **level properties**

Currently, the following level formats are supported:

(1) **dense**: level is dense, entries along the level are stored and linearized
(2) **compressed**: level is sparse, only nonzeros along the level are stored
    with positions and coordinates
(3) **singleton**: a variant of the compressed format, for when coordinates have
    no siblings
(4) **range**: a variant of the dense format, restricting the range based on a
    compression expression in the previous level

All level formats have the following level properties:

(1) **non/unique** (are duplicates allowed at that level),
(2) **un/ordered** (are coordinates sorted at that level).

Some 2-dim matrix examples are shown below (note that 
block format has 2 dimensions and 4 levels)::

  COO: (i, j) -> ( i : compressed(non-unique), j : singleton )

  CSR: (i, j) -> ( i : dense, j : compressed )

  CSC: (i, j) -> ( j : dense, i : compressed )  # j and i swapped!

  DCSR: (i, j) -> ( i : compressed, j : compressed )

  DCSC: (i, j) -> ( j : compressed, i : compressed )

  DIA-J: (i, j) -> ( j - i : compressed, j : range )

  BSR with 2x3 blocks: ( i, j ) -> ( i floordiv 2 : dense,
                                     j floordiv 3 : compressed,
                                     i mod 2      : dense,
                                     j mod 3      : dense )

Two 3-dim tensor examples are shown below::

  COO3: (i, j, k) -> ( i : compressed(non-unique),
                       j : singleton,
                       k : singleton )
  CSF3: (i, j, k) -> ( i : compressed,
                       j : compressed,
                       k : compressed )

Lastly, a 4-dim tensor examples is given here::

  COO4: (i, j, k, l) -> ( i : compressed(non-unique),
                          j : singleton,
                          k : singleton,
                          l : singleton )
 
The C++ representation of the latter is given below::

  using COO4 = SparseTensorFormat<4,
                 LvlSpec<D0, LvlType::CompressedNonUnique>,
                 LvlSpec<D1, LvlType::Singleton>,
                 LvlSpec<D2, LvlType::Singleton>,
                 LvlSpec<D3, LvlType::Singleton>>;

More examples can be found in the code.

Historical Background of the UST Type
-------------------------------------

The concept of the UST type has its roots in sparse compilers, first pioneered
for sparse linear algebra in [`B&W95`_, `B&W96`_, `Bik96`_, `Bik98`_] and
formalized to sparse tensor algebra in [`Kjolstad20`_, `Chou22`_, `Yadav22`_].
The tensor format DSL for the UST type, including the generalization to
higher-dimensional levels, was introduced in [`MLIR22`_, `MLIR`_]. Please
refer to this literature for a more extensive presentation of all topics only
briefly discussed in this online documentation.

.. _B&W95: https://dl.acm.org/doi/10.1006/jpdc.1995.1141
.. _B&W96: https://ieeexplore.ieee.org/document/485501
.. _Bik96: https://theses.liacs.nl/1315
.. _Bik98: https://dl.acm.org/doi/10.1145/290200.287636
.. _Chou22: http://tensor-compiler.org/files/chou-phd-thesis-taco-formats.pdf
.. _Kjolstad20: http://tensor-compiler.org/files/kjolstad-phd-thesis-taco-compiler.pdf
.. _MLIR22: https://dl.acm.org/doi/10.1145/3544559
.. _MLIR: https://developers.google.com/mlir-sparsifier
.. _Yadav22: http://tensor-compiler.org/files/yadav-pldi22-distal.pdf
.. _make_sparse_tensor.h: https://github.com/NVIDIA/MatX/blob/main/include/matx/core/make_sparse_tensor.h
.. _sparse_tensor.h: https://github.com/NVIDIA/MatX/blob/main/include/matx/core/sparse_tensor.h
.. _sparse_tensor_format.h: https://github.com/NVIDIA/MatX/blob/main/include/matx/core/sparse_tensor_format.h
