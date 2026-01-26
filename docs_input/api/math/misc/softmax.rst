.. _softmax_func:

softmax
=======

Reduce to softmax values in input. Softmax is defined as:

.. math::

   \sigma(z_i) = \frac{e^{z_{i}}}{\sum_{j=1}^K e^{z_{j}}} \ \ \ for\ i=1,2,\dots,K

.. versionadded:: 0.6.0

.. doxygenfunction:: softmax(const InType &in, const int (&dims)[D])
.. doxygenfunction:: softmax(const InType &in)

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin softmax-test-1
   :end-before: example-end softmax-test-1
   :dedent:


Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/ReductionTests.cu
   :language: cpp
   :start-after: example-begin softmax-test-2
   :end-before: example-end softmax-test-2
   :dedent: