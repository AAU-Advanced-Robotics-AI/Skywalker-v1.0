#!/bin/bash

VOLUME_SRC="$(pwd)"
VOLUME_DST="/ros2_ws"
ROS_DISTRO="humble"
CONTAINER="p10_container"
IMAGE="p10_ros2_$ROS_DISTRO"
IMAGE_VERSION="latest"
IMAGE_FULL="$IMAGE:$IMAGE_VERSION"

# Allow container to access X server
#xhost +local:root 

# Start new container shell if running
if [ "$(docker container list | grep -w "$CONTAINER")" ]; then
    docker exec -it "$CONTAINER" bash
    exit 0
fi

# Start container if it exists but not running
if [ "$(docker container list -a | grep -w "$CONTAINER")" ]; then
    docker start "$CONTAINER" && docker attach "$CONTAINER"
    exit 0
fi

# # Build image if Dockerfile is modified
# docker build --rm -t "$IMAGE_FULL" .

# Create container if it doesn't exist
docker run \
     --name $CONTAINER \
     --user ${ROS_DISTRO} \
     --network=host \
     --ipc=host \
     -v $VOLUME_SRC:$VOLUME_DST \
     --workdir=$VOLUME_DST \
     -v /tmp/.X11-unix:/tmp/.X11-unix:rw \
     --env=DISPLAY=:1 \
     -v /dev:/dev \
     --device-cgroup-rule="c *:* rmw" \
     -v /etc/timezone:/etc/timezone:ro \
     -v /etc/localtime:/etc/localtime:ro \
     -it $IMAGE_FULL
