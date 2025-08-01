# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR
source ./setup.sh

current_image="${MATX_REPO}${MATX_IMAGE_NAME}:${MATX_VERSION_TAG}"

if [[ "$current_image" == "$(unset MATX_REPO && unset MATX_IMAGE_NAME && unset MATX_VERSION_TAG && source ./setup.sh && echo ${MATX_REPO}${MATX_IMAGE_NAME}:$MATX_VERSION_TAG)" ]]
then
    echo "Error: Do not run this script without updating the MATX_REPO, MATX_IMAGE_NAME, and/or MATX_VERSION_TAG variables from command line"
    exit 1
fi

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

hpccm  --recipe recipe.py --cpu-target $CPU_TARGET --format docker > matx.build.Dockerfile
DOCKER_BUILDKIT=1 docker  build -f matx.build.Dockerfile --platform $MATX_PLATFORM -t $current_image-$TARGETARCH .

echo Finished building container "$current_image-$TARGETARCH"

