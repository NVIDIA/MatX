.. _write_npy_func:

write_npy
=========

Write an NPY file from a tensor

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: write_npy(const TensorType &t, const std::string& fname)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_npy-test-1
   :end-before: example-end write_npy-test-1
   :dedent:
