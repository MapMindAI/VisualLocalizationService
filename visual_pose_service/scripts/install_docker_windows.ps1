# Docker Desktop Windows Installation Script
# Requires administrator privileges to run

param(
    [string]$InstallDrive = "",
    [switch]$SkipWSL = $false,
    [switch]$Force = $false
)

# Check administrator privileges
if (-NOT ([Security.Principal.WindowsPrincipal] [Security.Principal.WindowsIdentity]::GetCurrent()).IsInRole([Security.Principal.WindowsBuiltinRole] "Administrator")) {
    Write-Host "❌ This script requires administrator privileges to run" -ForegroundColor Red
    Write-Host "Please right-click and select 'Run as Administrator' for PowerShell, then re-run this script" -ForegroundColor Yellow
    exit 1
}

Write-Host "🐳 Docker Desktop Windows Installation Script" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

# Check if Docker is already installed
Write-Host "🔍 Checking if Docker is already installed..." -ForegroundColor Yellow
if (Get-Command "docker" -ErrorAction SilentlyContinue) {
    Write-Host "✅ Docker is already installed, version information:" -ForegroundColor Green
    docker version

    if (-not $Force) {
        $choice = Read-Host "Docker already exists, force reinstall? (y/N)"
        if ($choice -ne 'y' -and $choice -ne 'Y') {
            exit 0
        }
    }
    Write-Host "🔄 Reinstalling Docker Desktop..." -ForegroundColor Yellow
}

# Check system requirements
Write-Host "🔍 Checking system requirements..." -ForegroundColor Yellow

# Check Windows version
$osVersion = [System.Environment]::OSVersion.Version
if ($osVersion.Major -lt 10) {
    Write-Host "❌ Docker Desktop requires Windows 10 or higher" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Windows version check passed: $($osVersion)" -ForegroundColor Green

# Check Hyper-V support (optional for Windows 10 Home)
$hyperVFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Hyper-V-All -ErrorAction SilentlyContinue
if ($hyperVFeature -and $hyperVFeature.State -eq "Enabled") {
    Write-Host "✅ Hyper-V is enabled" -ForegroundColor Green
} else {
    Write-Host "⚠️ Hyper-V is not enabled, will use WSL 2 backend" -ForegroundColor Yellow
}

# Check and enable WSL 2
if (-not $SkipWSL) {
    Write-Host "🔍 Checking WSL 2..." -ForegroundColor Yellow

    $wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
    if ($wslFeature.State -ne "Enabled") {
        Write-Host "📦 Enabling WSL feature..." -ForegroundColor Yellow
        Enable-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux -NoRestart
    }

    $vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform
    if ($vmFeature.State -ne "Enabled") {
        Write-Host "📦 Enabling Virtual Machine Platform feature..." -ForegroundColor Yellow
        Enable-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform -NoRestart
    }

    # Check WSL 2 kernel update
    try {
        $wslVersion = wsl --version 2>$null
        if ($LASTEXITCODE -ne 0) {
            Write-Host "📦 Downloading and installing WSL 2 kernel update..." -ForegroundColor Yellow
            $wslUpdateUrl = "https://wslstorestorage.blob.core.windows.net/wslblob/wsl_update_x64.msi"
            $wslUpdatePath = "$env:TEMP\wsl_update_x64.msi"
            Invoke-WebRequest -Uri $wslUpdateUrl -OutFile $wslUpdatePath
            Start-Process -FilePath "msiexec.exe" -ArgumentList "/i", $wslUpdatePath, "/quiet" -Wait
            Remove-Item $wslUpdatePath -Force

            # Set WSL 2 as default version
            wsl --set-default-version 2
        }
        Write-Host "✅ WSL 2 configuration completed" -ForegroundColor Green
    } catch {
        Write-Host "⚠️ WSL 2 configuration may require a restart to complete" -ForegroundColor Yellow
    }
}

# Determine installation drive
if (-not $InstallDrive) {
    if (Test-Path "D:\") {
        $InstallDrive = "D:"
        Write-Host "📁 D drive detected, will attempt to install Docker to D drive..." -ForegroundColor Green
    } else {
        $InstallDrive = "C:"
        Write-Host "⚠️ D drive not detected, will install to default C drive." -ForegroundColor Yellow
    }
} else {
    if (-not (Test-Path "$InstallDrive\")) {
        Write-Host "❌ Specified drive $InstallDrive does not exist" -ForegroundColor Red
        exit 1
    }
}

# Check disk space (Docker Desktop requires at least 4GB)
$drive = Get-WmiObject -Class Win32_LogicalDisk | Where-Object { $_.DeviceID -eq $InstallDrive }
$freeSpaceGB = [math]::Round($drive.FreeSpace / 1GB, 2)
if ($freeSpaceGB -lt 4) {
    Write-Host "❌ Insufficient disk space, requires at least 4GB, currently available: $freeSpaceGB GB" -ForegroundColor Red
    exit 1
}
Write-Host "✅ Disk space check passed: $freeSpaceGB GB available" -ForegroundColor Green

# Download Docker Desktop installer
Write-Host "📥 Downloading Docker Desktop installer..." -ForegroundColor Yellow
$dockerInstallerUrl = "https://desktop.docker.com/win/main/amd64/Docker Desktop Installer.exe"
$installerPath = "$env:TEMP\DockerDesktopInstaller.exe"

try {
    # Show download progress
    $webClient = New-Object System.Net.WebClient
    $webClient.DownloadFile($dockerInstallerUrl, $installerPath)
    Write-Host "✅ Download completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Download failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
}

# Construct installation parameters
$customInstallDir = "$InstallDrive\Docker"
Write-Host "📦 Installing Docker Desktop to $customInstallDir ..." -ForegroundColor Yellow

# Create installation directory
New-Item -ItemType Directory -Force -Path $customInstallDir | Out-Null

# Construct installation parameters
$installArgs = @(
    "install",
    "--quiet",
    "--accept-license"
)

# Note: Docker Desktop installation path parameter may vary by version
# If the specified installation directory parameter doesn't work, Docker will install to default location
if ($InstallDrive -ne "C:") {
    $installArgs += "--installation-dir=`"$customInstallDir`""
}

# Execute installation
try {
    Write-Host "⏳ Installing, please wait..." -ForegroundColor Yellow
    Start-Process -FilePath $installerPath -ArgumentList $installArgs -Wait -NoNewWindow
    Write-Host "✅ Docker Desktop installation completed" -ForegroundColor Green
} catch {
    Write-Host "❌ Installation failed: $($_.Exception.Message)" -ForegroundColor Red
    exit 1
} finally {
    # Clean up installer
    if (Test-Path $installerPath) {
        Remove-Item $installerPath -Force
    }
}

# Wait for service to start
Write-Host "⏳ Waiting for Docker service to start..." -ForegroundColor Yellow
$maxWait = 60  # Maximum wait time: 60 seconds
$waited = 0
do {
    Start-Sleep -Seconds 2
    $waited += 2
    $dockerService = Get-Service -Name "com.docker.service" -ErrorAction SilentlyContinue
} while ((-not $dockerService -or $dockerService.Status -ne "Running") -and $waited -lt $maxWait)

if ($dockerService -and $dockerService.Status -eq "Running") {
    Write-Host "✅ Docker service has started" -ForegroundColor Green
} else {
    Write-Host "⚠️ Docker service startup may require more time" -ForegroundColor Yellow
}

# Start Docker Desktop GUI
Write-Host "🚀 Starting Docker Desktop..." -ForegroundColor Yellow
try {
    Start-Process "Docker Desktop" -ErrorAction SilentlyContinue
} catch {
    # Try to start from default installation path
    $defaultPath = "$env:ProgramFiles\Docker\Docker\Docker Desktop.exe"
    if (Test-Path $defaultPath) {
        Start-Process $defaultPath
    } else {
        Write-Host "⚠️ Unable to automatically start Docker Desktop, please start it manually from the Start menu" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "🎉 Installation completed!" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "📋 Next steps:"
Write-Host "1️⃣ Wait for Docker Desktop to fully start (first startup may take a few minutes)"
Write-Host "2️⃣ If prompted, restart your computer"
Write-Host "3️⃣ Open Command Prompt or PowerShell, run 'docker --version' to verify installation"
Write-Host ""
Write-Host "🔧 Common commands:"
Write-Host "• Check version: docker --version"
Write-Host "• Test installation: docker run hello-world"
Write-Host "• View help: docker --help"
Write-Host ""
Write-Host "💡 Tips:"
Write-Host "• Docker Desktop will display an icon in the system tray"
Write-Host "• First run may require WSL 2 or Hyper-V configuration"
Write-Host "• If you encounter issues, check Docker Desktop settings"

# Check if restart is required
$restartRequired = $false
$wslFeature = Get-WindowsOptionalFeature -Online -FeatureName Microsoft-Windows-Subsystem-Linux
$vmFeature = Get-WindowsOptionalFeature -Online -FeatureName VirtualMachinePlatform

if ($wslFeature.RestartRequired -or $vmFeature.RestartRequired) {
    $restartRequired = $true
}

if ($restartRequired) {
    Write-Host ""
    Write-Host "⚠️ System restart is required to complete WSL 2 configuration" -ForegroundColor Yellow
    $restart = Read-Host "Restart now? (y/N)"
    if ($restart -eq 'y' -or $restart -eq 'Y') {
        Restart-Computer -Force
    } else {
        Write-Host "Please manually restart your computer later to complete the configuration" -ForegroundColor Yellow
    }
}