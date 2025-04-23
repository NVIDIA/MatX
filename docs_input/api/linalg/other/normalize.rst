.. _normalize_func:

normalize
===========

normalize all values in a tensor according to specified method, equivalent to MATLAB's implementation

.. doxygenfunction:: normalize(const OpA &op, const NORMALIZE_RANGE normalize_method)
.. doxygenfunction:: normalize(const OpA &op, const NORMALIZE_RANGE normalize_method, const float p)
.. doxygenfunction:: normalize(const OpA &op, const NORMALIZE_RANGE normalize_method, const float a, const float b)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin normalize-test-maxnorm
   :end-before: example-end normalize-test-maxnorm
   :dedent:

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin normalize-test-lpnorm
   :end-before: example-end normalize-test-lpnorm
   :dedent:

.. literalinclude:: ../../../../test/00_transform/Norm.cu
   :language: cpp
   :start-after: example-begin normalize-test-range
   :end-before: example-end normalize-test-range
   :dedent:

