<!--

Thank you for contributing to rapids-cmake :)

Here are some guidelines to help the review process go smoothly.

1. Please write a description in this text box of the changes that are being
   made.

2. Please ensure that you have followed the following guidelines for the changes made/features
   added.

   - Each function should follow the `rapids_<component>_<file_name>` naming pattern
   - Each function should go into a separate `.cmake` file in the appropriate directory
   - Each user facing `.cmake` file should have include guards (`include_guard(GLOBAL)`)
   - Each user facing `.cmake` file should be documented using the rst structure
   - Each user facing function should be added to the `cmake-format.json` document
    - Run `cmake-genparsers -f json` on the `.cmake` file as a starting point
   - Each function first line should be `list(APPEND CMAKE_MESSAGE_CONTEXT "rapids.<component>.<function>")`
   - Internal variables for things like `cmake_parse_arguments` should use `_RAPIDS`
     instead of `RAPIDS`, so we don't incorrectly use an existing variable like `RAPIDS_VERSION`

   - A file should not modify any state simply by being included. State modification should
     only occur inside functions unless absolutely necessary due to restrictions of the CMake
     language.
      - Any files that do need to break this rule can't be part of `rapids-<component>.cmake`.

3. If you are closing an issue please use one of the automatic closing words as
   noted here: https://help.github.com/articles/closing-issues-using-keywords/

4. If your pull request is not ready for review but you want to make use of the
   continuous integration testing facilities please mark your pull request as Draft.
   https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/changing-the-stage-of-a-pull-request#converting-a-pull-request-to-a-draft

5. If your pull request is ready to be reviewed without requiring additional
   work on top of it, then remove it from "Draft" and make it "Ready for Review".
   https://docs.github.com/en/github/collaborating-with-issues-and-pull-requests/changing-the-stage-of-a-pull-request#marking-a-pull-request-as-ready-for-review

   If assistance is required to complete the functionality, for example when the
   CMake code of a feature is complete but downstream testing is still required,
   then add the label `help wanted` so that others can triage and assist.
   The additional changes then can be implemented on top of the same PR.
   If the assistance is done by members of the rapidsAI team, then no
   additional actions are required by the creator of the original PR for this,
   otherwise the original author of the PR needs to give permission to the
   person(s) assisting to commit to their personal fork of the project. If that
   doesn't happen then a new PR based on the code of the original PR can be
   opened by the person assisting, which then will be the PR that will be
   merged.

6. Once all work has been done and review has taken place please do not add
   features or make changes out of the scope of those requested by the reviewer
   (doing this just add delays as already reviewed code ends up having to be
   re-reviewed/it is hard to tell what is new etc!). Further, please do not
   rebase your branch on main/force push/rewrite history, doing any of these
   causes the context of any comments made by reviewers to be lost. If
   conflicts occur against main they should be resolved by merging main
   into the branch used for making the pull request.

Many thanks in advance for your cooperation!

-->
