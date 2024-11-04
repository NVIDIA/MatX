CPM Reference
#############

The `rapids_cpm` functions allow projects to find or build dependencies with built-in
tracking of these dependencies for correct export support.

rapids-cmake package defaults
#############################

rapids-cmake provides a collection of pre-defined packages to ensure that all consumers use
the same version of dependencies. The exact versions that each pre-configured package uses
can be found at the bottom of the page.

.. include:: /packages/packages.rst

rapids-cmake package override
#############################

rapids-cmake allows projects to override the default values for the pre-defined packages provided by rapids-cmake while also overriding any :cmake:command:`rapids_cpm_find`, `CPM <https://github.com/cpm-cmake/CPM.cmake>`_, and :cmake:module:`FetchContent() <cmake:module:FetchContent>` package calls.

When an override for a project exists no :cmake:module:`find_package()<cmake:command:find_package>` search for that project will occur. This is done to make sure that the requested modified version is used.

If a project is listed in multiple override files, the first file values will be used,
and all later calls for that packaged will be ignored.  This "first to record, wins"
approach is used to match FetchContent, and allows parent projects to override child
projects.

If the override is of an existing default project, it only needs to specify the
fields it wants to override.

For example if you wanted to change the version of `fmt` you would do the following:

.. literalinclude:: /packages/example_git_tag_override.json
  :language: json

If you want to specify a completely new project, you need to specify at a minimum all
the required fields:

.. literalinclude:: /packages/example_new_project_override.json
  :language: json


Reproducible rapids-cmake builds
################################

The rapids-cmake default `versions.json` uses branch names or git tag names
for dependencies. This is done so that projects can 'live at head' of dependencies.
This directly conflicts with the concept of reproducible release packages that each
time they are built produce bitwise-identical artifacts.

:cmake:command:`rapids_cpm_generate_pinned_versions` can be used to generate a set of explicit
pinned git SHA1 values corresponding to the set of rapids-cmake dependencies in use. This results in a fully reproducible set of dependencies when building.

To utilize this behavior in your release CI/CD something like the following needs
to be set up:

  1. Enable generation of a pinned versions file during builds via
  :cmake:command:`rapids_cpm_generate_pinned_versions` or by specifying the :cmake:variable:`RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE`.
  2. If the build is good, create the release branch and commit the generated pinned `versions.json` to the repository
  3. Rebuilds of the project using the pinned version are now possible by setting the :cmake:variable:`RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` to the path of the generated pinned versions file.


rapids-cpm command line controls
################################

rapids-cpm offers multiple command line options to control behavior.

Some builds are performed fully offline and the default package and override urls
can't be used. In those cases you can use the variable :cmake:variable:`RAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE` to provide a new versions.json that will be
used instead of ALL overrides specified in that project. This would allow you to
specify custom internal urls for all dependencies without modifying the project source code.


```
cmake -DRAPIDS_CMAKE_CPM_OVERRIDE_VERSION_FILE=<abs/path/to/custom/override_versions.json> ....
```

To request the generation of a pinned package override file without having to modify
the project use the :cmake:variable:`RAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE` variable:

```
cmake -DRAPIDS_CMAKE_CPM_PINNED_VERSIONS_FILE
```

Additional CPM command line controls
************************************

In addition any of the CPM options can be used with rapids-cpm. A
full list of CPM options can be found in the `CPM README.md <https://github.com/cpm-cmake/CPM.cmake/blob/master/README.md>`_;
we document some of the most important ones below.


If you need to explicitly state a package must be downloaded and not searched
for locally you enable the variable :cmake:variable:`CPM_DOWNLOAD_<package_name>`.

```
cmake -DCPM_DOWNLOAD_<package_name>=ON ....
```

If you need to explicitly state all packages must be downloaded and not searched
for locally you enable the variable :cmake:variable:`CPM_DOWNLOAD_ALL`.

```
cmake -DCPM_DOWNLOAD_ALL=ON ....
```


rapids-cmake package version format
###################################


rapids-cmake uses a JSON file to encode the version of a project and how to download the project.

The JSON format is a root object that contains the ``packages`` object.

The ``packages`` object contains a key/value map of all supported
packages where the key is the case-sensitive name of the project and
the value is a ``project`` object, as seen in this example:

.. literalinclude:: /packages/example_all_fields.json
  :language: json


Project Object Fields
*********************

Each ``project`` object must contain the following fields so that
rapids-cmake can properly use CPM to find or download the project
as needed.

``version``

    A required string representing the version of the project to be used
    by :cmake:command:`rapids_cpm_find` when looking for a local installed
    copy of the project.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evaluated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``$ENV{variable}`` will be evaluated to the contents of the listed environment variable

``git_url``

    A required string representing the git url to be used when cloning the
    project locally by the :cmake:command:`rapids_cpm_find` when a locally
    installed copy of the project can't be found.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evaluated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``${version}`` will be evaluated to the contents of the ``version`` field.
        - ``$ENV{variable}`` will be evaluated to the contents of the listed environment variable

``git_tag``

    A required string representing the git tag to be used when cloning the
    project locally by the :cmake:command:`rapids_cpm_find` when a locally
    installed copy of the project can't be found.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evaluated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``${version}`` will be evaluated to the contents of the ``version`` field.
        - ``$ENV{variable}`` will be evaluated to the contents of the listed environment variable

``git_shallow``

    An optional boolean value that represents if we should do a shallow git clone
    or not.

    If no such field exists the default is ``true``.

``exclude_from_all``

    An optional boolean value that represents the CMake ```EXCLUDE_FROM_ALL``` property.
    If this is set to ``true``, and the project is built from source all targets of that
    project will be excluded from the ``ALL`` build rule. This means that any target
    that isn't used by the consuming project will not be compiled. This is useful
    when a project generates multiple targets that aren't required and the cost
    of building them isn't desired.

    If no such field exists the default is ``false``.

``always_download``

    An optional boolean value that represents if CPM should just download the
    package ( ``CPM_DOWNLOAD_ALL`` ) instead of first searching for it on the machine.

    The default value for this field is ``false`` unless all of the following criteria is met.
        - The projects exists in both the default and override files
        - The ``git_url``, ``git_tag``, ``patches`` keys exist in the override
        - Existence of a patch entry in the definition

``patches``
    An optional list of dictionary sets of git patches to apply to the project
    when it is built from source.

    If this field exists in the default package, the value will be ignored when an override file
    entry exists for the package. This ensures that patches only git url or `proprietary_binary` entry in the override will be used.

    The existence of a patch entry in the package definition being used will cause the `always_download` value always to be true.

    .. literalinclude:: /packages/patches.json
        :language: json

    Each dictionary in the array of patches contains the following fields:

        ``file``
            A required string representing the git diff ( .diff ) or patch ( .patch ) to apply.
            Absolute and relative paths are supported. Relative paths are
            evaluated in relation to the ``rapids-cmake/cpm/patches`` directory.

            Supports the following placeholders:
                - ``${current_json_dir}`` will be evaluated to the absolute path to the directory holding the current json file

        ``issue``
            A required string that explains the need for the patch. Preference is for the
            string to also contain the URL to the upstream issue or PR that
            this patch addresses.

        ``fixed_in``
            A required entry that represents which version this patch
            is no longer needed in. If this patch is required for all
            versions an empty string should be supplied.

        ``required``
            An optional boolean value that represents if it is required that the patch
            apply correctly.

            The default value for this field is ``false``.

``proprietary_binary``

    An optional dictionary of cpu architecture and operating system keys to url values that represents a download for a pre-built proprietary version of the library. This creates a new entry in the search
    logic for a project:

        - Search for a local version matching the ``version`` key
            - disabled by ``always_download``
        - Download proprietary version if a valid OS + CPU Arch exists
            - disabled by ``USE_PROPRIETARY_BLOB`` being off
        - Fallback to using git url and tag

    To determine the correct key, CMake will query for a key that matches the lower case value of `<arch>-<os>` where `arch` maps to
    :cmake:variable:`CMAKE_SYSTEM_PROCESSOR <cmake:variable:CMAKE_SYSTEM_PROCESSOR>` and `os` maps to :cmake:variable:`CMAKE_SYSTEM_NAME <cmake:variable:CMAKE_SYSTEM_NAME>`.

    If no such key exists the request to use a `proprietary_binary` will be ignored.

    .. literalinclude:: /packages/proprietary_binary.json
        :language: json

    As this represents a proprietary binary only the following packages support this command:
        - nvcomp

    Due to requirements of proprietary binaries, explicit opt-in by the user on usage is required.
    Therefore for this binary to be used the caller must call the associated `rapids_cpm` command
    with the ``USE_PROPRIETARY_BLOB`` set to ``ON``.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evaluated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``${version}`` will be evaluated to the contents of the ``version`` field.
        - ``${cuda-toolkit-version}`` will be evaluated to 'major.minor' of the current CUDA Toolkit version.
        - ``${cuda-toolkit-version-major}`` will be evaluated to 'major' of the current CUDA Toolkit version.
        - ``${cuda-toolkit-version-mapping}`` will be evaluated to the contents of the json `cuda_version_mapping` entry
          of the current CUDA Toolkit major version value
        - ``$ENV{variable}`` will be evaluated to the contents of the listed environment variable

    If this field exists in the default package, the value will be ignored when an override file
    entry exists for the package. This ensures that the git url or `proprietary_binary` entry in the override will be used.

``proprietary_binary_cuda_version_mapping``

    An optional dictionary of CUDA major version keys to arbitrary values that are needed to compute download urls for a pre-built proprietary binaries
    in the ``proprietary_binary`` dictionary

    .. literalinclude:: /packages/cuda_version_mapping.json
        :language: json

    As this represents meta information needed by the proprietary binary dictionary only the following packages support this entry:
        - nvcomp


rapids-cmake package versions
#############################


.. _cpm_versions:
.. literalinclude:: /../rapids-cmake/cpm/versions.json
    :language: json
