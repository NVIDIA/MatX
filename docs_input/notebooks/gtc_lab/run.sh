#!/bin/bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
IMAGE_NAME=gitlab-master.nvidia.com:5005/tylera/playground/gtc-lab:latest
CMDS="useradd -u $(id -u) -g $(id -g) -m -s /bin/bash tylera && su tylera"

docker run -it --rm \
    -p 8888:8888 \
    --gpus all \
    --ipc=host \
    -v /scratch/tylera/:/scratch/tylera \
    -w $(pwd) \
    $IMAGE_NAME 