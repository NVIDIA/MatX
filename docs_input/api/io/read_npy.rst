.. _read_npy_func:

read_npy
========

Read an NPY file into a tensor

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: read_npy(TensorType &t, const std::string& fname)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin read_npy-test-1
   :end-before: example-end read_npy-test-1
   :dedent:
