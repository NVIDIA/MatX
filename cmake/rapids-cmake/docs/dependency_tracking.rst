
Dependency Tracking
###################

One of the biggest features of rapids-cmake is that it can track dependencies ( `find_package`, `cpm` ),
allowing projects to easily generate ``<project>-config.cmake`` files with correct dependency requirements.
In a normal CMake project, public dependencies need to be recorded in two locations: the original ``CMakeLists.txt`` file and the generated ``<project>-config.cmake``. This dual source of truth increases
developer burden, and adds a common source of error.

``rapids-cmake`` is designed to remove this dual source of truth by expanding the concept of Modern CMake `Usage Requirements <https://cmake.org/cmake/help/latest/manual/cmake-buildsystem.7.html#build-specification-and-usage-requirements>`_ to include external packages.
This is done via the ``BUILD_EXPORT_SET`` ( maps to ``<BUILD_INTERFACE>`` ) and ``INSTALL_EXPORT_SET`` ( maps to ``<INSTALL_INTERFACE>`` ) keywords on commands such as :cmake:command:`rapids_find_package` and :cmake:command:`rapids_cpm_find`.
Let's go over an example of how these come together to make dependency tracking for projects easy.

.. code-block:: cmake

  rapids_find_package(Threads REQUIRED
    BUILD_EXPORT_SET example-targets
    INSTALL_EXPORT_SET example-targets
    )
  rapids_cpm_find(nlohmann_json 3.9.1
    BUILD_EXPORT_SET example-targets
    )

  add_library(example src.cpp)
  target_link_libraries(example
    PUBLIC Threads::Threads
      $<BUILD_INTERFACE:nlohmann_json::nlohmann_json>
    )
  install(TARGETS example
    DESTINATION lib
    EXPORT example-targets
    )

  set(doc_string [=[Provide targets for the example library.]=])
  rapids_export(INSTALL example
    EXPORT_SET example-targets
    NAMESPACE example::
    DOCUMENTATION doc_string
    )
  rapids_export(BUILD example
    EXPORT_SET example-targets
    NAMESPACE example::
    DOCUMENTATION doc_string
    )


Tracking find_package
*********************

.. code-block:: cmake

  rapids_find_package(Threads REQUIRED
    BUILD_EXPORT_SET example-targets
    INSTALL_EXPORT_SET example-targets
    )

The :cmake:command:`rapids_find_package(<PackageName>)` combines the :cmake:command:`find_package <cmake:command:find_package>` command and dependency tracking.
This example records that the ``Threads`` package is required by both the export set ``example-targets`` for both the install and build configuration.

This means that when :cmake:command:`rapids_export` is called the generated ``example-config.cmake`` file will include the call
`find_dependency(Threads)`, removing the need for developers to maintain that dual source of truth.

The :cmake:command:`rapids_find_package(<PackageName>)` command also supports non-required finds correctly. In those cases ``rapids-cmake`` only records
the dependency when the underlying :cmake:command:`find_package <cmake:command:find_package>` command is successful.

It is common for projects to have dependencies for which CMake doesn't have a ``Find<Package>``. In those cases projects will have a custom
``Find<Package>`` that they need to use, and install for consumers. Rapids-cmake tries to help projects simplify this process with the commands
:cmake:command:`rapids_find_generate_module` and :cmake:command:`rapids_export_package`.

The :cmake:command:`rapids_find_generate_module` allows projects to automatically generate a ``Find<Package>`` and encode via the ``BUILD_EXPORT_SET``
and ```INSTALL_EXPORT_SET``` parameters when the generated module should also be installed and added to ``CMAKE_MODULE_PATH`` so that consumers can use it.

If you already have an existing ``Find<Package>`` written, :cmake:command:`rapids_export_package` simplifies the process of installing the module and
making sure it is part of ``CMAKE_MODULE_PATH`` for consumers.

Tracking CPM
************

.. code-block:: cmake

  rapids_cpm_find(nlohmann_json 3.9.1
    BUILD_EXPORT_SET example-targets
    )

The :cmake:command:`rapids_cpm_find` combines the :cmake:command:`CPMFindPackage` command and dependency tracking, in a very similar way
to :cmake:command:`rapids_find_package`. In this example what we are saying is that nlohmann_json is only needed by the build directory ``example-config``
and not needed by the installed ``example-config``. While this pattern is rare, it occurs when projects have some dependencies that aren't needed by consumers but are
propagated through the usage requirements inside a project via $<BUILD_INTERFACE>. Why use a build directory `config` file at all? The most common
reason is that developers need to work on multiple dependent projects in a fast feedback loop. In that case this workflow avoids having to re-install a project each time
a change needs to be tested in a dependent project.

When used with ``BUILD_EXPORT_SET``, :cmake:command:`rapids_cpm_find` will generate a :cmake:command:`CPMFindPackage(<PackageName> ...)` call, and when used
with ``INSTALL_EXPORT_SET`` it will generate a :cmake:command:`find_dependency(<PackageName> ...) <cmake:command:find_dependency>` call. The theory behind this is that most packages currently don't have
great build ``config.cmake`` support so it is best to have a fallback to cpm, while it is expected that all CMake packages have install rules.
If this isn't the case for a CPM package you can instead use :cmake:command:`rapids_export_cpm`, and :cmake:command:`rapids_export_package` to specify the correct generated commands
and forgo using ``[BUILD|INSTALL]_EXPORT_SET``.


Generating example-config.cmake
*******************************

.. code-block:: cmake

  set(doc_string [=[Provide targets for the example library.]=])
  rapids_export(INSTALL example
    EXPORT_SET example-targets
    NAMESPACE example::
    DOCUMENTATION doc_string
    )
  rapids_export(BUILD example
    EXPORT_SET example-targets
    NAMESPACE example::
    DOCUMENTATION doc_string
    )

Before ``rapids-cmake``, if a project wanted to generate a config module they would follow the example in
the :ref:`cmake-packages docs <cmake:Creating Packages>` and use :cmake:command:`install(EXPORT) <cmake:command:install>`, :cmake:command:`export(EXPORT) <cmake:command:export>`, :cmake:command:`write_basic_package_version_file <cmake:command:write_basic_package_version_file>`, and a custom `config.cmake.in` file.

The goal of :cmake:command:`rapids_export` is to replace all the boilerplate with an easy to use function that also embeds the necessary
dependency calls collected by ``BUILD_EXPORT_SET`` and ``INSTALL_EXPORT_SET``.

:cmake:command:`rapids_export` uses CMake best practises to generate all the necessary components of a project config file. It handles generating
a correct version file, finding dependencies and all the other boilerplate necessary to make well-behaved CMake config files. Moreover,
the files generated by :cmake:command:`rapids_export` are completely standalone with no dependency on ``rapids-cmake``.
