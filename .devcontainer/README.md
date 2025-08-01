# MatX Container Generation and Usage scripts

## Steps for running a Matx container

1. Run the run_matx.sh script, optionally specifying a different repo, image base name, or version tag

`./run_matx.sh # defaults to latest tag in setup.sh`

or

`MATX_VERSION_TAG="12.9.1_ubuntu24.04" ./run_matx.sh`

Note: architecture (`-amd64` or `-arm64`) is automatically added to the tag by the scripts


## Steps for building a new container

1. Make your changes to the container recipe

2. Build the container

`MATX_IMAGE_NAME="someTestName" MATX_VERSION_TAG="someNewTag" create_base_container.sh`

The MATX_REPO, MATX_IMAGE_NAME, and/or MATX_VERSION_TAG must be different than the current values in setup.sh, to avoid accidentally overwriting the working container.

Note: architecture (`-amd64` or `-arm64`) is automatically added to the tag by the scripts

3. Test the container

4. Push the container.  Also retag the container as latest and push that too

Exercise left to the reader, to prevent accidentally pushing the latest tag.

5. Modify setup.sh to update the MATX_VERSION_TAG and commit your updates to setup.sh and recipe.py

