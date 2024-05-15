# Contributing to rapids-cmake

If you are interested in contributing to rapids-cmake, your contributions will fall
into three categories:
1. You want to report a bug, feature request, or documentation issue
    - File an [issue](https://github.com/rapidsai/rapids-cmake/issues/new/choose)
    describing what you encountered or what you want to see changed.
    - The RAPIDS team will evaluate the issues and triage them, scheduling
    them for a release. If you believe the issue needs priority attention
    comment on the issue to notify the team.
2. You want to propose a new Feature and implement it
    - Post about your intended feature, and we shall discuss the design and
    implementation.
    - Once we agree that the plan looks good, go ahead and implement it, using
    the [code contributions](#code-contributions) guide below.
3. You want to implement a feature or bug-fix for an outstanding issue
    - Follow the [code contributions](#code-contributions) guide below.
    - If you need more context on a particular issue, please ask and we shall
    provide.

## Code contributions

While RAPIDS core provides commonly used scripts we know that they aren't universal and might need to be composed in different ways.

This means that the code we are developing should be designed for composability, and all side-effects
or CMake behavior changes should be explicitly opt-in.

So when writing new rapids-cmake features make sure to think about how users might want to opt-in, and
provide the necessary function decomposition. For example lets look at an example of wanting to have an
easy wrapper around creating libraries and setting properties.

```
[=[ BAD ]=]

function(rapids_add_library target )
  add_library(${target} ${ARGN})
  set_target_properties(cudf
      PROPERTIES
                 CUDA_STANDARD                        17
                 CUDA_STANDARD_REQUIRED               ON
  )
endfunction()

rapids_add_library(example SHARED ...)


[=[ GOOD ]=]

function(rapids_cmake_setup_target target )
  set_target_properties(${target}
      PROPERTIES
                 CUDA_STANDARD                        17
                 CUDA_STANDARD_REQUIRED               ON
  )
endfunction()

function(rapids_add_library target)
  add_library(example ${ARGN})
  rapids_cmake_setup_target( example )
endfunction()

rapids_add_library(example SHARED ...)

```

Here we can see that breaking out `rapids_cmake_setup_target` is important as it allows users
that don't/can't use `rapids_add_library` to still opt-in to other features.


Please ensure that when you are creating new features you follow the following guidelines:
   - Each function should follow the `rapids_<component>_<file_name>` naming pattern
   - Each function should go into a separate `.cmake` file in the appropriate directory
   - Each user facing `.cmake` file should have include guards (`include_guard(GLOBAL)`)
   - Each user facing `.cmake` file should be documented following the rst structure
   - Each user facing function should be added to the `cmake-format.json` document
    - Run `cmake-genparsers -f json` on the `.cmake` file as a starting point
   - Each function first line should be `list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.<component>.<function>")`
   - A file should not modify any state simply by being included. State modification should
     only occur inside functions unless absolutely necessary due to restrictions of the CMake
     language.
        - Any files that do need to break this rule can't be part of `rapids-<component>.cmake`.

### Your first issue

1. Read the project's [README.md](https://github.com/rapidsai/rapids-cmake/blob/main/README.md)
    to learn how to setup the development environment
2. Find an issue to work on. The best way is to look for the [good first issue](https://github.com/rapidsai/rapids-cmake/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue%22)
    or [help wanted](https://github.com/rapidsai/rapids-cmake/issues?q=is%3Aissue+is%3Aopen+label%3A%22help+wanted%22) labels
3. Comment on the issue saying you are going to work on it
4. Code! Make sure to update unit tests!
5. When done, [create your pull request](https://github.com/rapidsai/rapids-cmake/compare)
6. Verify that CI passes all [status checks](https://help.github.com/articles/about-status-checks/). Fix if needed
7. Wait for other developers to review your code and update code as needed
8. Once reviewed and approved, a RAPIDS developer will merge your pull request

Remember, if you are unsure about anything, don't hesitate to comment on issues
and ask for clarifications!

### Seasoned developers

Once you have gotten your feet wet and are more comfortable with the code, you
can look at the prioritized issues of our next release in our [project boards](https://github.com/rapidsai/rapids-cmake/projects).

> **Pro Tip:** Always look at the release board with the highest number for
issues to work on. This is where RAPIDS developers also focus their efforts.

Look at the unassigned issues, and find an issue you are comfortable with
contributing to. Start with _Step 3_ from above, commenting on the issue to let
others know you are working on it. If you have any questions related to the
implementation of the issue, ask them in the issue instead of the PR.

## Attribution
Portions adopted from https://github.com/pytorch/pytorch/blob/master/CONTRIBUTING.md
