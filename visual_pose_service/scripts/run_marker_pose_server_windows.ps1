# Set variables
$ImageTarFile = "visual_pose_server_image.tar"
$ImageName = "visual_pose_server_image"
$ContainerName = "marker_pose_server_container"
$ImagePath = "aliyunregistry-gz-mobi-registry.cn-guangzhou.cr.aliyuncs.com/dm/mobili-visual-pose-server-image:presubmit-20250714-bafe3"

$DataDir = "C:\Users\YourUsername\Data"
$MarkerConfigName = "marker_config.json"  # Marker config file name in DataDir (optional)
$MarkerConfigPath = Join-Path $DataDir $MarkerConfigName
$MarkerImageDir = Join-Path $DataDir "marker_image"  # Marker image directory in DataDir

# Server port
$Port = if ($env:PORT) { $env:PORT } else { "40011" }

# Pool size and max workers
$PoolSize = if ($env:POOL_SIZE) { $env:POOL_SIZE } else { "4" }
$MaxWorkers = if ($env:MAX_WORKERS) { $env:MAX_WORKERS } else { "10" }

# Log level
$LogLevel = if ($env:LOG_LEVEL) { $env:LOG_LEVEL } else { "INFO" }

# Log directory (empty means no logging)
$LogDir = if ($env:LOG_DIR) { $env:LOG_DIR } else { "" }

# Check if Docker is installed
$scriptDir = $PSScriptRoot
& "$scriptDir\install_docker_windows.ps1"

# Check if image exists
$ImageExists = docker image inspect $ImageName -ErrorAction SilentlyContinue

if ($ImageExists) {
    Write-Host "$ImageName exists in Docker!"
} else {
    Write-Host "$ImageName does not exist in Docker!"
    Write-Host "Attempting to pull $ImagePath..."
    docker pull $ImagePath
    if ($LASTEXITCODE -eq 0) {
        Write-Host "Successfully pulled the image!"
        docker tag $ImagePath $ImageName
        Write-Host "Successfully tagged the image as $ImageName"
    } else {
        Write-Error "Failed to pull the image!"
        exit 1
    }
}

# Check if marker config exists (optional, will use default configuration if not found)
if ($MarkerConfigName -and !(Test-Path $MarkerConfigPath)) {
    Write-Warning "Marker config not found at $MarkerConfigPath. Will use default configuration."
    $MarkerConfigName = ""
}

# Check if marker_image directory exists
if (!(Test-Path $MarkerImageDir -PathType Container)) {
    Write-Warning "Marker image directory not found at $MarkerImageDir"
    Write-Host "Please create the directory and place marker images there."
}

# Stop existing container if it exists
$existingContainer = docker ps -aq -f "name=^${ContainerName}$"
if ($existingContainer) {
    docker rm -f $ContainerName | Out-Null
}

Write-Host "[*] Running marker_pose_server container from image: $ImageName"

# Start container
docker run `
    --name $ContainerName `
    -p "${Port}:${Port}" `
    -v "${DataDir}:/data" `
    -v "D:\Data\deploy\entrypoint_marker_pose.sh:/app/deployment/entrypoint_marker_pose.sh" `
    -e "MARKER_CONFIG_NAME=${MarkerConfigName}" `
    -e "PORT=${Port}" `
    -e "POOL_SIZE=${PoolSize}" `
    -e "MAX_WORKERS=${MaxWorkers}" `
    -e "LOG_LEVEL=${LogLevel}" `
    -e "LOG_DIR=${LogDir}" `
    $ImageName `
    bash deployment/entrypoint_marker_pose.sh

