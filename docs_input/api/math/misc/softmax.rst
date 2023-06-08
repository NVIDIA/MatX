.. _softmax_func:

softmax
=======

Reduce to softmax values in input. Softmax is defined as:

.. math::

   \sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K

.. doxygenfunction:: softmax(OutType dest, const InType &in, cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin softmax-test-1
   :end-before: example-end softmax-test-1
   :dedent:

.. doxygenfunction:: softmax(OutType dest, const InType &in, const int (&dims)[D], cudaStream_t stream = 0)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin softmax-test-2
   :end-before: example-end softmax-test-2
   :dedent: