.. _write_mat_func:

write_mat
=========

Write one or more variables to a MAT file

.. note::
   Enable file I/O with ``-DMATX_EN_FILEIO=ON`` and install NumPy and SciPy

The tensor overload writes one variable under the name passed in ``var``.
The dictionary overload writes all variables from a ``pybind11::dict``.
Both overloads replace an existing file at the given path.


.. versionadded:: 0.3.0
.. doxygenfunction:: write_mat(const TensorType &t, const std::string fname, const std::string var)
.. doxygenfunction:: write_mat(const pybind11::dict &variables, const std::string &filename)
.. doxygenfunction:: add_mat_variable(pybind11::dict &variables, const std::string &name, const TensorType &tensor)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_mat-test-1
   :end-before: example-end write_mat-test-1
   :dedent:

To save two tensors in one file, initialize Python before creating the
dictionary and hold the GIL when creating, using, or destroying Python objects.
For example, a C++ application that embeds Python can create a
``pybind11::scoped_interpreter`` before the dictionary.

.. code-block:: cpp

   pybind11::dict variables;

Add the tensors under distinct MATLAB variable names. The helper copies the
values, so the source tensors can go out of scope before saving. The tensor
memory must be host-accessible and any pending GPU writes must finish before
the copy.

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_mat-dictionary-add
   :end-before: example-end write_mat-dictionary-add
   :dedent:

Write the populated dictionary to a path such as ``"arrays_out.mat"``.

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_mat-dictionary-save
   :end-before: example-end write_mat-dictionary-save
   :dedent:
