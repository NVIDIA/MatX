#!/bin/bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
IMAGE_NAME=gitlab-master.nvidia.com:5005/tylera/playground/gtc-lab:latest


LAB_FOLDER=$(pwd)
MATX_ROOT_DIR="${LAB_FOLDER%/*/*/*}/"
echo $MATX_ROOT_DIR

docker run -it --rm \
    -p 8888:8888 \
    --gpus all \
    --ipc=host \
    -v $MATX_ROOT_DIR/:/opt/xeus/cling/tools/Jupyter/kernel/MatX/ \
    -v $LAB_FOLDER:$LAB_FOLDER \
    -w $LAB_FOLDER \
    $IMAGE_NAME 