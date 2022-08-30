API Reference
#############


Common
******

The `rapids_cmake` functions provide common CMake logic that numerous projects
require.

.. toctree::
   :titlesonly:

   /command/rapids_cmake_build_type
   /command/rapids_cmake_install_lib_dir
   /command/rapids_cmake_make_global
   /command/rapids_cmake_parse_version
   /command/rapids_cmake_support_conda_env
   /command/rapids_cmake_write_git_revision_file
   /command/rapids_cmake_write_version_file

CPM
***

The `rapids_cpm` functions allow projects to find or build dependencies with built-in
tracking of these dependencies for correct export support.

.. toctree::
   :titlesonly:

   /command/rapids_cpm_init
   /command/rapids_cpm_find

CPM Pre-Configured Packages
***************************

These `rapids_cpm` functions allow projects to easily find or build common
RAPIDS dependencies.

These allow projects to make sure they use the same version and flags for
dependencies as the rest of RAPIDS. The exact versions that each pre-configured
package uses :ref:`can be found here. <cpm_versions>`

.. toctree::
   :titlesonly:

   /packages/rapids_cpm_cuco
   /packages/rapids_cpm_gtest
   /packages/rapids_cpm_libcudacxx
   /packages/rapids_cpm_nvbench
   /packages/rapids_cpm_nvcomp
   /packages/rapids_cpm_rmm
   /packages/rapids_cpm_spdlog
   /packages/rapids_cpm_thrust
   /command/rapids_cpm_package_override

Cython
******

The `rapids_cython` functions allow projects to easily build cython modules using
`scikit-build <https://scikit-build.readthedocs.io/en/latest/>`_.

.. note::
  Use of the rapids-cython component of rapids-cmake requires scikit-build. The behavior of the functions provided by
  this component is undefined if they are invoked outside of a build managed by scikit-build.

.. toctree::
   :titlesonly:

   /command/rapids_cython_init
   /command/rapids_cython_create_modules


Find
****

The `rapids_find` functions allow projects to find system dependencies with built-in
tracking of these dependencies for correct export support.

.. toctree::
   :titlesonly:

   /command/rapids_find_generate_module
   /command/rapids_find_package

CUDA
****

The `rapids_cuda` functions provide common CMake CUDA logic that numerous projects
require.

.. toctree::
   :titlesonly:

    rapids_cuda_init_architectures </command/rapids_cuda_init_architectures>
    rapids_cuda_init_runtime </command/rapids_cuda_init_runtime>
    rapids_cuda_set_architectures [Advanced] </command/rapids_cuda_set_architectures>


Export Set Generation
*********************

These `rapids_export` functions allow projects to generate correct build and install tree `Project-Config.cmake` modules including required dependencies.

For the vast majority of projects :cmake:command:`rapids_export` should be sufficient. But when
not projects may use commands such as :cmake:command:`rapids_write_dependencies` and
cmake:command:`rapids_write_language` to create a custom `Project-Config.cmake`.


.. toctree::
   :maxdepth: 1

   rapids_export </command/rapids_export>
   rapids_export_write_dependencies [Advanced] </command/rapids_export_write_dependencies>
   rapids_export_write_language     [Advanced] </command/rapids_export_write_language>

Export Set Tracking
*******************

These `rapids_export` functions allow projects to track track dependencies for
correct export generation. These should only be used when :cmake:command:`rapids_find_package`,
:cmake:command:`rapids_cpm_find`, or :cmake:command:`rapids_find_generate_module` are insufficient.


.. toctree::
   :titlesonly:

   rapids_export_cpm [Advanced] </command/rapids_export_cpm>
   rapids_export_find_package_file [Advanced] </command/rapids_export_find_package_file>
   rapids_export_find_package_root [Advanced] </command/rapids_export_find_package_root>
   rapids_export_package [Advanced] </command/rapids_export_package>

