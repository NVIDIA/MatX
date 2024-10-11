.. _make_tensor:

Making Tensors
==============

`make_tensor` is a utility function for creating tensors. Where possible, using `make_tensor` is preferred over
directly declaring a `tensor_t` since it allows the tensor type to change in the future without breaking. See :ref:`creating`
for a detailed walkthrough on creating tensors.

`make_tensor` provides numerous overloads for different arguments and use cases:

Return by Value
~~~~~~~~~~~~~~~

.. doxygenfunction:: make_tensor( const index_t (&shape)[RANK], matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( TensorType &tensor, const index_t (&shape)[TensorType::Rank()], matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( ShapeType &&shape, matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( TensorType &tensor, ShapeType &&shape,  matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( TensorType &tensor, matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( T *data, const index_t (&shape)[RANK], bool owning = false)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::value_type *data, const index_t (&shape)[TensorType::Rank()])
.. doxygenfunction:: make_tensor( T *data, ShapeType &&shape, bool owning = false)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::value_type *data, typename TensorType::shape_container &&shape)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::value_type *ptr)
.. doxygenfunction:: make_tensor( Storage &&s, ShapeType &&shape)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::storage_type &&s, typename TensorType::shape_container &&shape)
.. doxygenfunction:: make_tensor( T* const data, D &&desc, bool owning = false)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::value_type* const data, typename TensorType::desc_type &&desc)
.. doxygenfunction:: make_tensor( D &&desc, matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( TensorType &&tensor, typename TensorType::desc_type &&desc, matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor( T *const data, const index_t (&shape)[RANK], const index_t (&strides)[RANK], bool owning = false)
.. doxygenfunction:: make_tensor( TensorType &tensor, typename TensorType::value_type *const data, const index_t (&shape)[TensorType::Rank()], const index_t (&strides)[TensorType::Rank()])

Return by Pointer
~~~~~~~~~~~~~~~~~
.. doxygenfunction:: make_tensor_p( const index_t (&shape)[RANK],  matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor_p( ShapeType &&shape, matxMemorySpace_t space = MATX_MANAGED_MEMORY, cudaStream_t stream = 0)
.. doxygenfunction:: make_tensor_p( T *const data, ShapeType &&shape, bool owning = false)
