Type Traits
###########

MatX type traits help advanced developers to make compile-time decisions about types. Most of these are used extensively
inside of MatX, and are not needed in most user applications.

.. versionadded:: 0.1.0

MatX now uses C++20 concepts for type traits. Legacy variable templates (ending in ``_v``) and functions (ending in ``_t()``) 
are maintained for backward compatibility.

Type Manipulation
=================

.. doxygentypedef:: matx::promote_half_t
.. doxygenstruct:: matx::remove_cvref

Concepts (C++20)
================

.. doxygenconcept:: matx::is_tensor
.. doxygenconcept:: matx::is_matx_op_c
.. doxygenconcept:: matx::is_executor
.. doxygenconcept:: matx::is_matx_reduction
.. doxygenconcept:: matx::is_matx_index_reduction
.. doxygenconcept:: matx::is_cuda_complex
.. doxygenconcept:: matx::is_complex
.. doxygenconcept:: matx::is_complex_half
.. doxygenconcept:: matx::is_half
.. doxygenconcept:: matx::is_matx_half
.. doxygenconcept:: matx::is_matx_type
.. doxygenconcept:: matx::is_matx_shape
.. doxygenconcept:: matx::is_matx_storage
.. doxygenconcept:: matx::is_matx_storage_container
.. doxygenconcept:: matx::is_matx_descriptor

Legacy Compatibility
====================

Legacy functions and variables for backward compatibility:

.. doxygenfunction:: matx::is_matx_op
.. doxygenfunction:: matx::is_executor_t
.. doxygenfunction:: matx::IsHalfType

Note: Legacy variable templates (``is_tensor_v``, ``is_matx_reduction_v``, etc.) are available 
for backward compatibility but are not documented here. Use the concepts above instead for new code.