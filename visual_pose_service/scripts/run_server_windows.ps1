# Set variables
$ImageTarFile = "visual_pose_server_image.tar"
$ImageName = "visual_pose_server_image"
$ContainerName = "visual_pose_server_container"
$ImagePath = "aliyunregistry-gz-mobi-registry.cn-guangzhou.cr.aliyuncs.com/dm/mobili-visual-pose-server-image:presubmit-20250714-bafe3"

$DataDir = "C:\Users\YourUsername\Data"
$DbName = "NanshaOffice\database_3d.db"
$DbPath = Join-Path $DataDir $DbName

$ModelDir = "C:\Users\YourUsername\Models"

# Log directory (empty means no logging)
$LogDir = ""

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
    } else {
        Write-Error "Failed to pull the image!"
        exit 1
    }
}

# Check if superpoint and superglue models exist (default location: model_repository directory)
if (-Not (Test-Path -Path $ModelDir -PathType Container) -Or -Not (Get-ChildItem -Path $ModelDir -Recurse)) {
    Write-Error "Model directory $ModelDir does not exist or is empty!"
    exit 1
}

# Start a new PowerShell window to launch Triton server
Write-Host "[*] Starting Triton Inference Server..."
Start-Process powershell -ArgumentList "-NoExit", "-Command", "docker run --gpus 'device=0' --rm -p 8001:8001 --name 'tritonserver' -v '${ModelDir}:/models' aliyunregistry.deepmirror.com.cn/dm/inference-server-app:0.3.1 tritonserver --model-repository=/models"

# Check if database exists
if (!(Test-Path $DbPath)) {
    Write-Error "Database not found at $DbPath"
    Write-Host "Please ensure the data folder contains the .db file before running."
    exit 1
}

# Stop existing container if it exists
$existingContainer = docker ps -aq -f "name=^${ContainerName}$"
if ($existingContainer) {
    docker rm -f $ContainerName | Out-Null
}

Write-Host "[*] Running container from image: $ImageName"

# Start container
docker run `
    --name $ContainerName `
    -p 40010:40010 `
    -v "${DataDir}:/data" `
    -v "D:\Data\deploy\entrypoint.sh:/app/deployment/entrypoint.sh" `
    -e "DB_NAME=${DbName}" `
    -e "LOG_DIR=${LogDir}" `
    $ImagePath