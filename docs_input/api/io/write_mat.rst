.. _write_mat_func:

write_mat
=========

Write an operator to a MAT file

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: write_mat(const TensorType &t, const std::string fname, const std::string var)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_mat-test-1
   :end-before: example-end write_mat-test-1
   :dedent:
