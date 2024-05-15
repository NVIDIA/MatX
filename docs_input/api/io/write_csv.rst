.. _write_csv_func:

write_csv
=========

Write an operator to a CSV file

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: write_csv(const TensorType &t, const std::string fname, const std::string delimiter)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin write_csv-test-1
   :end-before: example-end write_csv-test-1
   :dedent:
