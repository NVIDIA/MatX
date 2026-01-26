.. _sar_bp_func:

sar_bp
#######

Synthetic aperture radar backprojection. The sar_bp operator is currently in the experimental
namespace as its API is subject to change.

.. versionadded:: head

.. doxygenfunction:: sar_bp(const ImageType &initial_image, const RangeProfilesType &range_profiles, const PlatPosType &platform_positions, const VoxLocType &voxel_locations, const RangeToMcpType &range_to_mcp, const SarBpParams &params)
.. doxygenenum:: matx::SarBpComputeType
.. doxygenenum:: matx::SarBpFeature
.. doxygenstruct:: matx::SarBpParams
   :members:

Examples
~~~~~~~~

.. literalinclude:: ../../../../test/00_transform/SarBp.cu
   :language: cpp
   :start-after: example-begin sar-bp-1
   :end-before: example-end sar-bp-1
   :dedent:
