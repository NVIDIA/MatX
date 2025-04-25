.. _normalize_func:

normalize
===========

normalize all values in a tensor according to specified method, equivalent to `MATLAB's implementation <https://www.mathworks.com/help/matlab/ref/double.normalize.html>`_

.. list-table:: Normalize method is defined as NORMALIZE_RANGE::method_option, available options are:
   :widths: 30 70
   :header-rows: 1

   * - NORMALIZE_RANGE\:\:
     - Description
   * - ZSCORE
     - Center data to have mean 0 and std 1
   * - NORM
     - Normalizes using Lp-norm, p can be any float \> 0. It defaults to p=-1 (Max norm)
   * - SCALE
     - Scale data to have standard deviation 1
   * - RANGE
     - rescale in range [a, b]; by default [0.0, 1.0]
   * - CENTER
     - Center data to have mean 0

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

