.. _read_mat_func:

read_mat
========

Read a CSV file into a tensor

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: matx::io::read_mat(TensorType &t, const std::string fname, const std::string var)
.. doxygenfunction:: matx::io::read_mat(const std::string fname, const std::string var)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin read_mat-test-1
   :end-before: example-end read_mat-test-1
   :dedent:
