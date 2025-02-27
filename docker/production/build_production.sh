#!/bin/bash
# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR


if [ ! -v MATX_VERSION_TAG ]; then
   MATX_VERSION=$(git describe --tags "$(git rev-list --tags --max-count=1)")
   MATX_VERSION_TAG="${MATX_VERSION}_ubuntu22.04"
fi

if [ ! -v MATX_REPO ]; then
   MATX_REPO=" ghcr.io/nvidia/matx/"
fi

if [ ! -v MATX_IMAGE_NAME ]; then
   MATX_IMAGE_NAME="production"
fi

if [ -z "$MATX_PLATFORM" ]; then
    case "$(arch)" in
        "x86_64")
            MATX_PLATFORM="linux/amd64"
            ;;
        "aarch64")
            MATX_PLATFORM="linux/arm64"
            ;;
        *)
            echo "Unsupported arch type"
            exit 1
            ;;
    esac
fi

# add the architecture name to the tag
TARGETARCH=$(basename $MATX_PLATFORM)
case "$TARGETARCH" in
    "amd64")
        CPU_TARGET=x86_64
        ;;
    "arm64")
        CPU_TARGET=aarch64
        ;;
    *)
        echo "Unsupported target architecture"
        exit 1
        ;;
esac
# create the Dockerfile
hpccm  --recipe matx-production.py --cpu-target $CPU_TARGET --format docker > production.Dockerfile

# build the container
DOCKER_BUILDKIT=1 docker  build -f production.Dockerfile --platform $MATX_PLATFORM -t $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG-$TARGETARCH .


docker tag $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG-$TARGETARCH ghcr.io/nvidia/matx/production:latest

# push the container to the repository
# docker push $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG-$TARGETARCH
