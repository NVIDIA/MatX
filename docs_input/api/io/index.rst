.. _io:

Input/Output
############

MatX file IO helpers read and write common array file formats through the
optional ``MATX_ENABLE_FILEIO`` support. Include ``matx.h`` and use the
``matx::io`` namespace functions shown below.

Read functions can write into an already-sized tensor, or into a
default-constructed tensor with the desired rank and value type. When the tensor
has no storage yet, MatX allocates it after discovering the shape from the file.

CSV
===

CSV files support rank-1 and rank-2 tensors. The delimiter is passed explicitly,
and ``read_csv`` skips the first row by default.

.. code-block:: cpp

   tensor_t<float, 2> samples;
   io::read_csv(samples, "samples.csv", ",");

   io::write_csv(samples, "samples_out.csv", ",");

To read a file without skipping the first row, pass ``false`` for the final
argument.

.. code-block:: cpp

   io::read_csv(samples, "samples_out.csv", ",", false);

MAT
===

MAT files can contain multiple named variables. ``read_mat`` and ``write_mat``
therefore take a variable name in addition to the file name.

.. code-block:: cpp

   tensor_t<float, 2> A;
   io::read_mat(A, "arrays.mat", "A");

   auto B = io::read_mat<tensor_t<float, 2>>("arrays.mat", "B");

   io::write_mat(A, "arrays_out.mat", "A");

MATLAB v7.3 MAT files are HDF5-based, but the MatX MAT helpers are variable
oriented and use SciPy's MAT-file routines. Treat them as MAT-file readers and
writers rather than as a general HDF5 interface.

NPY
===

NPY files store a single NumPy array per file.

.. code-block:: cpp

   tensor_t<float, 2> x;
   io::read_npy(x, "x.npy");

   io::write_npy(x, "x_out.npy");

.. toctree::
   :maxdepth: 1
   :glob:

   *
