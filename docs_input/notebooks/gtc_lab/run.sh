#!/bin/bash
USER_ID=$(id -u)
GROUP_ID=$(id -g)
CMDS="/bin/bash"
    # -u $USER_ID:$GROU P_ID \

docker run -it --rm \
    -p 8888:8888 \
    --gpus all \
    --ipc=host \
    -v /home/scratch.tylera_sw/:/scratch \
    gitlab-master.nvidia.com:5005/devtech-compute/sigx-group/container/cling:latest \
    /bin/bash



# docker run -it --rm \
#     -p 8888:8888 \
#     --gpus all \
#     -v /home/scratch.tylera_sw/:/scratch \
#     -v /home/scratch.tylera_sw/projects/matx_holo_lab_2025:/notebooks \
#     --env="DISPLAY" \
#     --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
#     --cap-add CAP_SYS_PTRACE \
#     --ipc=host \
#     --ulimit memlock=-1 \
#     --ulimit stack=67108864 \
#     gitlab-master.nvidia.com:5005/devtech-compute/sigx-group/container/cling:latest \
#     /bin/bash