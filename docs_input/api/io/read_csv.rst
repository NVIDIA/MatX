.. _read_csv_func:

read_csv
========

Read a CSV file into a tensor

.. note::
   This function requires the optional ``MATX_ENABLE_FILEIO`` compile flag


.. doxygenfunction:: read_csv(TensorType &t, const std::string fname, const std::string delimiter, bool header = true)

Examples
~~~~~~~~

.. literalinclude:: ../../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin read_csv-test-1
   :end-before: example-end read_csv-test-1
   :dedent:
