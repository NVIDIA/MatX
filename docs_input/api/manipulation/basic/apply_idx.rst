.. _apply_idx_func:

apply_idx
#########

Apply a custom function to one or more operators with full index access. The apply_idx operator 
allows users to define custom transformations using lambda functions or functors that receive both
the current indices as a ``cuda::std::array`` and the operators themselves. This provides more 
flexibility than ``apply()`` by allowing access to any element of the input operators, not just 
the current element.

The function can be provided as either an inline lambda or a functor. Inline ``__device__`` lambdas 
work in regular code (like ``main()`` functions) but NOT in Google Test fixtures due to CUDA's 
restriction on extended lambdas in private/protected methods. For test code, use functors instead.

``apply_idx()`` assumes the rank and size of the output is the same as the first input operator. 
Unlike ``apply()``, the lambda receives the indices and operators directly, allowing for stencil 
operations, neighbor access, and other spatially-aware computations.

Use Cases
~~~~~~~~~

``apply_idx`` is particularly useful for:

* Stencil operations (accessing neighboring elements)
* Convolution-like operations with custom kernels
* Operations that depend on element position/indices
* Accessing non-local elements based on the current position
* Implementing custom boundary conditions

Using cuda::std::apply
~~~~~~~~~~~~~~~~~~~~~~

A powerful pattern is to use ``cuda::std::apply`` inside your functor to convert the index array 
back into a parameter pack. Since lambdas with captures can be problematic in device code, we use 
a helper functor:

.. code-block:: cpp

   // Helper for unpacking indices
   template<typename Op>
   struct ApplyIndices2D {
     const Op& op;
     __host__ __device__ ApplyIndices2D(const Op& o) : op(o) {}
     __host__ __device__ auto operator()(index_t i, index_t j) const {
       return op(i, j);
     }
   };

   struct MyFunctor {
     template<typename Op>
     __host__ __device__ auto operator()(cuda::std::array<index_t, 2> idx, const Op& op) const {
       // Use cuda::std::apply to unpack the index array
       return cuda::std::apply(ApplyIndices2D<Op>(op), idx);  // Calls op(idx[0], idx[1])
     }
   };

You can also modify the indices before unpacking to implement operations like transposition or 
neighbor access:

.. code-block:: cpp

   // Access with swapped indices (transpose-like access)
   cuda::std::array<index_t, 2> swapped = {idx[1], idx[0]};
   return cuda::std::apply(ApplyIndices2D<Op>(op), swapped);  // Calls op(idx[1], idx[0])

Note you may see a naming collision with ``std::apply`` or ``cuda::std::apply``. For the MatX function 
it's best to use the ``matx::apply_idx`` form instead. For the standard library function used to unpack
indices, use ``cuda::std::apply``.

.. doxygenfunction:: matx::apply_idx

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_operators/apply_idx_test.cu
   :language: cpp
   :start-after: example-begin apply-idx-test-1
   :end-before: example-end apply-idx-test-1
   :dedent:

.. literalinclude:: ../../../../test/00_operators/apply_idx_test.cu
   :language: cpp
   :start-after: example-begin apply-idx-test-2
   :end-before: example-end apply-idx-test-2
   :dedent:

.. literalinclude:: ../../../../test/00_operators/apply_idx_test.cu
   :language: cpp
   :start-after: example-begin apply-idx-test-3
   :end-before: example-end apply-idx-test-3
   :dedent:

.. literalinclude:: ../../../../test/00_operators/apply_idx_test.cu
   :language: cpp
   :start-after: example-begin apply-idx-test-4
   :end-before: example-end apply-idx-test-4
   :dedent:


