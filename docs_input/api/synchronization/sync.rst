.. _sync_func:

sync
====

Wait for any code running on an executor to complete.

.. doxygenfunction:: matx::cudaExecutor::sync()
.. doxygenfunction:: matx::HostExecutor::sync()

Examples
~~~~~~~~

.. literalinclude:: ../../../examples/cgsolve.cu
   :language: cpp
   :start-after: example-begin sync-test-1
   :end-before: example-end sync-test-1
   :dedent:
