Tensor Transformations
######################

Transformations take a tensor view as input and transforms it into another tensor view, typically of a different size and/or rank. Transformations 
are used for more complex operations compared to operators, where race conditions can arise from input and output. Examples of this are FFTs, GEMMs,
linear solvers, and others. Usually an existing CUDA library or custom kernel is called for a transformation. Some libraries require some setup
before the execution is done, and do this by creating a handle or plan before the kernel is executed. 

Transformations can hide the internals of the plan/handles by creating and caching them on the first use. The caching interface provides a very
simply API that requires no configuration, but at the cost of a small first-use penalty. Users wanting more control of the underlying libraries
can use the non-cached interface by manually creating the plan first, and using this in subsequent calls. Each one of the transformation types
below provide both cached and non-cached interfaces.

.. toctree::
  :maxdepth: 4

  fft.rst
  matmul.rst
  solver.rst
  inverse.rst
  filter.rst
  reduce.rst
  sort.rst
  