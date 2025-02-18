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

# Check if container exists and delete it if present
if ((docker ps -a --format '{{.Names}}' | Where-Object { $_ -eq $CONTAINER })) {
    Write-Output "[---- Container with name: $CONTAINER exists, deleting it to make sure it gets run with new updated image. ]"
    docker rm -f $CONTAINER
}
else {
    Write-Output "[---- Container with name: $CONTAINER doesn't exist. ]"
    # docker create --name $CONTAINER #-v "$VOLUME_SRC`:$VOLUME_DST" $IMAGE_FULL
}

# Build image if Dockerfile is modified
docker build --rm -t $IMAGE_FULL .
