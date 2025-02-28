#!/bin/bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
IMAGE_NAME=gitlab-master.nvidia.com:5005/tylera/playground/gtc-lab:latest

LAB_FOLDER="$( cd "$(dirname "$0")" >/dev/null 2>&1 ; pwd -P )"
MATX_ROOT_DIR="${LAB_FOLDER%/*/*/*}/"
SCRATCH_DIR="/$(echo "$LAB_FOLDER" | cut -d'/' -f2-3)/"

docker run -it --rm \
    -p 8888:8888 \
    --gpus all \
    --ipc=host \
    -v $MATX_ROOT_DIR/:/opt/xeus/cling/tools/Jupyter/kernel/MatX/ \
    -v $SCRATCH_DIR:$SCRATCH_DIR \
    -w $LAB_FOLDER \
    $IMAGE_NAME