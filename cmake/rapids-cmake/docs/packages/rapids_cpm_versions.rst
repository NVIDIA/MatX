:orphan:

.. _cpm_version_format:

rapids-cmake package version format
###################################


rapids-cmake uses a JSON file to encode the version of a project and how to download the project.

The JSON format is a root object that contains the ``packages`` object.

The ``packages`` object contains a key/value map of all supported
packages where the key is the case sensitive name of the project and
the value is a ``project`` object, as seen in this example:

.. literalinclude:: /packages/example.json
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
        - ``${rapids-cmake-version}`` will be evulated to 'major.minor' of the current rapids-cmake cal-ver value.

``git_url``

    A required string representing the git url to be used when cloning the
    project locally by the :cmake:command:`rapids_cpm_find` when a locally
    installed copy of the project can't be found.

``git_tag``

    A required string representing the git tag to be used when cloning the
    project locally by the :cmake:command:`rapids_cpm_find` when a locally
    installed copy of the project can't be found.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evulated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``${version}`` will be evaluated to the contents of the ``version`` field.

``git_shallow``

    An optional boolean value that represents if we should do a shallow git clone
    or not.

    If no such field exists the default is `true`.

``exclude_from_all``

    An optional boolean value that represents the CMake `EXCLUDE_FROM_ALL` property.
    If this is set to `true`, and the project is built from source all targets of that
    project will be excluded from the `ALL` build rule. This means that any target
    that isn't used by the consuming project will not be compiled. This is useful
    when a project generates multiple targets that aren't required and the cost
    of building them isn't desired.

    If no such field exists the default is `false`.

``always_download``

    An optional boolean value that represents if CPM should just download the
    package ( `CPM_DOWNLOAD_ALL` ) instead of first searching for it on the machine.

    The default value for this field is `false` unless all of the following criteria is met.
        - The projects exists in both the default and override files
        - The `git_url`, `git_tag`, `patches` keys exist in the override

``patches``
    An optional list of dictionary sets of git patches to apply to the project
    when it is built from source.

    .. literalinclude:: /packages/patches.json
        :language: json

    Each dictionary in the array of patches contains the following fields:

        ``file``
            A required string representing the git diff ( .diff ) or patch ( .patch ) to apply.
            Absolute and relative paths are supported. Relative paths are
            evaluated in relation to the `rapids-cmake/cpm/patches` directory.

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

``proprietary_binary``

    An optional dictionary of cpu architecture and operating system keys to url values that represents a download for a pre-built proprietary version of the library. This creates a new entry in the search
    logic for a project:

        - Search for a local version matching the `version` key
            - disabled by `always_download`
        - Download proprietary version if a valid OS + CPU Arch exists
            - disabled by `USE_PROPRIETARY_BLOB` being off
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
    with the `USE_PROPRIETARY_BLOB` set to `ON`.

    Supports the following placeholders:
        - ``${rapids-cmake-version}`` will be evaluated to 'major.minor' of the current rapids-cmake cal-ver value.
        - ``${version}`` will be evaluated to the contents of the ``version`` field.
        - ``${cuda-toolkit-version}`` will be evaluated to 'major.minor' of the current CUDA Toolkit version.
        - ``${cuda-toolkit-version-major}`` will be evaluated to 'major' of the current CUDA Toolkit version.

    If this field exists in the default package, the value will be ignored when an override file
    entry exists for the package. This ensures that the git url or `proprietary_binary` entry in the override will be used.

rapids-cmake package versions
#############################


.. _cpm_versions:
.. literalinclude:: /../rapids-cmake/cpm/versions.json
    :language: json
