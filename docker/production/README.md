# MatX Container Generation and Usage scripts

## Steps for running a Matx container

1. Run the run_matx.sh script, optionally specifying a version tag

`./run_matx.sh # defaults to latest tag in setup.sh`

or

`MATX_VERSION_TAG="12.6.314_ubuntu22.04" ./run_matx.sh`

Note: architecture (`-amd64` or `-arm64`) is automatically added to the tag by the scripts


## Steps for building a new 'build' container

1. Make your changes to the container recipe

2. Build the container

`MATX_VERSION_TAG="someNewTag" create_build_container.sh`

The MATX_VERSION_TAG must be different than the current value in setup.sh, to avoid accidentally overwriting the working container.
Example: MATX_VERSION_TAG="12.6.314_ubuntu22.04"

Note: architecture (`-amd64` or `-arm64`) is automatically added to the tag by the scripts

3. Test the container

4. Retag the container as latest and push it

Exercise left to the reader, to prevent accidentally pushing the latest tag.
We should have a CICD step that does this...

5. Modify setup.sh to update the MATX_VERSION_TAG and commit your updates to setup.sh and recipe.py

