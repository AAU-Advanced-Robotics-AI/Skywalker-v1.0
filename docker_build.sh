#!/bin/bash

VOLUME_SRC="$(pwd)"
VOLUME_DST="/ros2_ws"
ROS_DISTRO="humble"
CONTAINER="p10_container"
IMAGE="p10_ros2_$ROS_DISTRO"
IMAGE_VERSION="latest"
IMAGE_FULL="$IMAGE:$IMAGE_VERSION"

# Check if container exist and delete
if [ "$(docker ps -a | grep -cw $CONTAINER)" -gt 0 ]; then
    echo "[---- Container with name: $CONTAINER exist, deleting it to make sure it gets run with new updated image. ]"
    docker rm -f $CONTAINER
else
  echo "[---- Container with name: $CONTAINER  doesn't exist. ]"
fi
# # Build image if Dockerfile is modified
docker build --rm -t "$IMAGE_FULL" .
