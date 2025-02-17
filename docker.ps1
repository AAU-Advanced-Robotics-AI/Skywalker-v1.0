# Set variables
$VOLUME_SRC = (Get-Location).Path
$VOLUME_DST = "/ros2_ws"
$ROS_DISTRO = "humble"
$CONTAINER = "p10_container"
$IMAGE = "p10_ros2_$ROS_DISTRO"
$IMAGE_VERSION = "latest"
$IMAGE_FULL = "$IMAGE`:$IMAGE_VERSION"

# Check if Docker Desktop is running, start it if not
$dockerProcess = Get-Process -Name "Docker Desktop" -ErrorAction SilentlyContinue
if (-not $dockerProcess) {
    Write-Output "[---- Docker Desktop is not running. Starting it now... ]"
    Start-Process -FilePath "C:\Program Files\Docker\Docker\Docker Desktop.exe" -NoNewWindow
    Start-Sleep -Seconds 10  # Wait for Docker to start
}

# Check if container is running, attach if it is
if ((docker ps --format '{{.Names}}' | Where-Object { $_ -eq $CONTAINER })) {
    Write-Output "[---- Container $CONTAINER is already running. Attaching to it. ]"
    docker exec -it $CONTAINER powershell
    exit
}

# Start container if it exists but is not running
if ((docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $CONTAINER })) {
    Write-Output "[---- Container $CONTAINER exists but is not running. Starting it now. ]"
    docker start $CONTAINER
    docker attach $CONTAINER
    exit
}

# Build image if Dockerfile is modified
# docker build --rm -t $IMAGE_FULL .

# Create container if it doesn't exist
docker run `
    --name $CONTAINER `
    --user $ROS_DISTRO `
    --network=host `
    --ipc=host `
    -v $VOLUME_SRC`:$VOLUME_DST `
    --workdir=$VOLUME_DST `
    -v /tmp/.X11-unix:/tmp/.X11-unix:rw `
    --env=DISPLAY=:1 `
    -v /dev:/dev `
    --device-cgroup-rule="c *:* rmw" `
    -v /etc/timezone:/etc/timezone:ro `
    -v /etc/localtime:/etc/localtime:ro `
    -it $IMAGE_FULL
