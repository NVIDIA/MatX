.. _write_mat_func:

write_mat
=========

Write one or more variables to a MAT file

.. note::
   Enable file I/O with ``-DMATX_EN_FILEIO=ON`` and install NumPy and SciPy

The tensor overload writes one variable under the name passed in ``var``.
The dictionary overload writes the named variables from a ``pybind11::dict``.
It rejects non-string keys, empty names, and names starting with ``_`` before
opening the output file. Both overloads replace an existing file at the given
path when writing.


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

To save two tensors in one file, hold the GIL when creating, using, or
destroying Python objects. Reuse the active Python interpreter if your
application or an earlier MatX file I/O call has already started it. Otherwise,
create a ``pybind11::scoped_interpreter`` before the dictionary and keep it alive
until the Python objects have been destroyed.

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
