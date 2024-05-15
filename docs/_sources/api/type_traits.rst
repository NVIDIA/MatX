Type Traits
###########

MatX type traits help advanced developers to make compile-time decisions about types. Most of these are used extensively
inside of MatX, and are not needed in most user applications.

.. doxygentypedef:: matx::promote_half_t
.. doxygenstruct:: matx::remove_cvref
.. doxygenfunction:: matx::is_matx_op
.. doxygenfunction:: matx::is_executor_t
.. doxygenvariable:: matx::is_tensor_view_v
.. doxygenvariable:: matx::is_matx_reduction_v
.. doxygenvariable:: matx::is_matx_index_reduction_v
.. doxygenvariable:: matx::is_cuda_complex_v
.. doxygenvariable:: matx::is_complex_v
.. doxygenvariable:: matx::is_complex_half_v
.. doxygenfunction:: matx::IsHalfType
.. doxygenvariable:: matx::is_half_v
.. doxygenvariable:: matx::is_matx_type_v
.. doxygenvariable:: matx::is_matx_shape_v
.. doxygenvariable:: matx::is_matx_storage_v
.. doxygenvariable:: matx::is_matx_storage_container_v
.. doxygenvariable:: matx::is_matx_descriptor_v