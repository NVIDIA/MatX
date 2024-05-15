.. io:

MatX IO
#######

MatX provides several ways of both reading and writing common file types from a MatX tensor. Python support is required 
to use IO functionality.

At this time MatX supports CSV and .mat files for IO. CSV provides a simple, common format used across platforms, 
while .mat gives a powerful, structured format used by both MATLAB and NumPy.

.. doxygenfunction:: ReadCSV(TensorType &t, const std::string fname, const std::string delimiter, bool header = true)

Examples
~~~~~~~~

.. literalinclude:: ../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin readcsv-test-1
   :end-before: example-end readcsv-test-1
   :dedent:


.. doxygenfunction:: WriteCSV(const TensorType &t, const std::string fname, const std::string delimiter)

Examples
~~~~~~~~

.. literalinclude:: ../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin writecsv-test-1
   :end-before: example-end writecsv-test-1
   :dedent:


.. doxygenfunction:: ReadMAT(TensorType &t, const std::string fname, const std::string var)
.. doxygenfunction:: ReadMAT(const std::string fname, const std::string var)

Examples
~~~~~~~~

.. literalinclude:: ../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin readmat-test-1
   :end-before: example-end readmat-test-1
   :dedent:


.. doxygenfunction:: WriteMAT(const TensorType &t, const std::string fname, const std::string var)

Examples
~~~~~~~~

.. literalinclude:: ../../test/00_io/FileIOTests.cu
   :language: cpp
   :start-after: example-begin writemat-test-1
   :end-before: example-end writemat-test-1
   :dedent:
