# setup.ps1 - Automated Setup for HumanIntent (Windows / PowerShell)

$ErrorActionPreference = "Stop"

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  HumanIntent - Initial Setup" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$RootDir = $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$VenvDir = Join-Path $BackendDir ".venv"

# 1. Check Python
Write-Host "[1/4] Checking Python installation..." -ForegroundColor Yellow
if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Python is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Python 3.10+ and re-run this script." -ForegroundColor Red
    exit 1
}
$pythonVersion = python --version
Write-Host "  Found: $pythonVersion" -ForegroundColor Green

# 2. Setup Backend Python Virtual Environment
Write-Host "[2/4] Setting up backend virtual environment..." -ForegroundColor Yellow
if (-not (Test-Path $VenvDir)) {
    Write-Host "  Creating venv at: $VenvDir"
    python -m venv $VenvDir
} else {
    Write-Host "  Virtual environment already exists."
}

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

Write-Host "  Installing backend dependencies from requirements.txt..."
& $VenvPython -m pip install --upgrade pip --quiet
& $VenvPython -m pip install -r (Join-Path $BackendDir "requirements.txt") --quiet
Write-Host "  Backend dependencies installed successfully." -ForegroundColor Green

# 3. Setup Frontend Node Dependencies
Write-Host "[3/4] Setting up frontend dependencies..." -ForegroundColor Yellow
if (-not (Get-Command npm -ErrorAction SilentlyContinue)) {
    Write-Host "ERROR: Node.js / npm is not installed or not in PATH." -ForegroundColor Red
    Write-Host "Please install Node.js 18+ and re-run this script." -ForegroundColor Red
    exit 1
}
$nodeVersion = node --version
Write-Host "  Found Node: $nodeVersion"

Push-Location $FrontendDir
try {
    Write-Host "  Installing frontend npm packages..."
    npm install --quiet
    Write-Host "  Frontend dependencies installed successfully." -ForegroundColor Green
} finally {
    Pop-Location
}

# 4. Check Ollama Status & Instructions
Write-Host "[4/4] Checking Ollama setup..." -ForegroundColor Yellow
if (Get-Command ollama -ErrorAction SilentlyContinue) {
    Write-Host "  Ollama CLI detected." -ForegroundColor Green
} else {
    Write-Host "  WARNING: Ollama CLI not detected in PATH." -ForegroundColor Yellow
    Write-Host "  Please install Ollama from https://ollama.com" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Setup Completed Successfully!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host ""
Write-Host "IMPORTANT PREREQUISITES BEFORE RUNNING:" -ForegroundColor Cyan
Write-Host "1. Ensure Ollama is running locally:      ollama serve"
Write-Host "2. Ensure the model is installed:        ollama run llama3.2"
Write-Host "   (or fallback to llama3.1:8b if desired)"
Write-Host ""
Write-Host "TO START BOTH FRONTEND AND BACKEND:" -ForegroundColor Cyan
Write-Host "   Run: .\dev.ps1"
Write-Host ""
