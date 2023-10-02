.. _percentile_func:

percentile
##########

Find the q-th percentile of an input sequence. ``q`` is a value between 0 and 100 representing the percentile. A value
of 0 is equivalent to mean, 100 is max, and 50 is the median when using the ``LINEAR`` method.

.. note::
    Multiple q values are not supported yet

Supported methods for interpolation are: LINEAR, HAZEN, WEIBULL, LOWER, HIGHER, MIDPOINT, NEAREST, MEDIAN_UNBIASED, and NORMAL_UNBIASED

.. doxygenfunction:: percentile(const InType &in, unsigned char q, PercentileMethod method = PercentileMethod::LINEAR)
.. doxygenfunction:: percentile(const InType &in, unsigned char q, const int (&dims)[D], PercentileMethod method = PercentileMethod::LINEAR)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin percentile-test-1
   :end-before: example-end percentile-test-1
   :dedent:
