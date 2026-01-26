.. _apply_func:

apply
#####

Apply a custom function to one or more operators element-wise. The apply operator allows users to
define custom transformations using lambda functions or functors that can be applied to tensor
operations. The function can be annotated with ``__host__``, ``__device__``, or both to control
where it executes.

``apply()`` assumes the rank and size of the output is the same as the first input operator. For 
cases where this is not true, a custom operator should be used instead. In general, ``apply`` will 
perform better than a custom operator because of optimization that most custom operators do not take 
advantage of. Running the ``black_scholes`` example shows the performance difference.

Note you may see a naming collision with ``std::apply`` or ``cuda::std::apply``. For this function 
it's best to use the ``matx::apply`` form instead.

.. versionadded:: 0.9.4

.. doxygenfunction:: matx::apply

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/apply_test.cu
   :language: cpp
   :start-after: example-begin apply-test-1
   :end-before: example-end apply-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/apply_test.cu
   :language: cpp
   :start-after: example-begin apply-test-2
   :end-before: example-end apply-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/apply_test.cu
   :language: cpp
   :start-after: example-begin apply-test-3
   :end-before: example-end apply-test-3
   :dedent:

