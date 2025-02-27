#!/bin/bash

USER_ID=$(id -u)
GROUP_ID=$(id -g)

# Switch to SCRIPT_DIR directory
SCRIPT=$(readlink -f $0)
SCRIPT_DIR=$(dirname $SCRIPT)
echo $SCRIPT starting...
cd $SCRIPT_DIR
source ./setup.sh

TARGETARCH=$(basename $MATX_PLATFORM)
MATX_INSTANCE_NAME="c_matx"

if [ -z "$1" ]; then
   echo Start container instance at bash prompt
   CMDS="/bin/bash"
else
   CMDS="$@"
   echo Run command then exit container
   echo Command: $CMDS
fi

if [[ $(lspci | grep -i NV) ]]; then
   GPU_FLAG="--gpus all"

else
   GPU_FLAG=""
   echo This system has no GPU, running without --gpus all parameter
   echo Creating soft link for libcuda.so.1 for any host-without-GPU code dependency
   CMDS="sudo ln -s /usr/local/cuda/compat/libcuda.so.1 /usr/lib/\$(arch)-linux-gnu/libcuda.so.1 && $CMDS"
fi

echo Command: $CMDS

docker pull --platform=$MATX_PLATFORM $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG
if [[ "$?" != "0" ]]; then
    echo "WARNING - The docker pull for $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG with platform $MATX_PLATFORM FAILED"
    echo "You may have an image locally that could be used. This may be stale."
    read -p "Do you want to continue? y/n " ret
    if [[ "$ret" == "y" ]]; then
        echo "Continuing..."
    else
        echo "Exiting."
        exit 1
    fi
fi

docker run --platform=$MATX_PLATFORM \
    --privileged \
    --cap-add=SYS_ADMIN \
    --cap-add=SYS_PTRACE \
    --security-opt seccomp=unconfined \
    -it --rm \
    $MATX_EXTRA_FLAGS \
    $GPU_FLAG \
    --name ${MATX_INSTANCE_NAME}_${USER} \
    --hostname ${MATX_INSTANCE_NAME}_${USER} \
    --add-host ${MATX_INSTANCE_NAME}_${USER}:127.0.0.1 \
    --network host --shm-size=4096m \
    -u $USER_ID:$GROUP_ID \
    -w `pwd` \
    -v $(echo ~):$(echo ~) \
    -v /nfs:/nfs \
    -v /dev/hugepages:/dev/hugepages \
    -v /usr/src:/usr/src \
    -v /lib/modules:/lib/modules \
    -v /vol0:/vol0 \
    --userns=host \
    --ipc=host \
    $MATX_REPO$MATX_IMAGE_NAME:$MATX_VERSION_TAG fixuid /bin/bash -c "$CMDS"
