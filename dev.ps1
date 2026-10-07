# dev.ps1 - Launch HumanIntent Backend & Frontend for Local Development

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "  Starting HumanIntent Local Dev" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

$RootDir = $PSScriptRoot
$BackendDir = Join-Path $RootDir "backend"
$FrontendDir = Join-Path $RootDir "frontend"
$VenvPython = Join-Path $BackendDir ".venv\Scripts\python.exe"

if (-not (Test-Path $VenvPython)) {
    $VenvPython = "python"
}

Write-Host "Launching Backend (FastAPI on http://localhost:8000)..." -ForegroundColor Yellow
$BackendCommand = "cd '$BackendDir'; & '$VenvPython' -m uvicorn app.main:app --reload --port 8000"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $BackendCommand

Start-Sleep -Seconds 2

Write-Host "Launching Frontend (Next.js on http://localhost:3000)..." -ForegroundColor Yellow
$FrontendCommand = "cd '$FrontendDir'; npm run dev"
Start-Process powershell -ArgumentList "-NoExit", "-Command", $FrontendCommand

Write-Host ""
Write-Host "==========================================" -ForegroundColor Green
Write-Host "  Both services launched in separate windows!" -ForegroundColor Green
Write-Host "==========================================" -ForegroundColor Green
Write-Host "Frontend: http://localhost:3000" -ForegroundColor Cyan
Write-Host "Backend:  http://localhost:8000 (Docs: http://localhost:8000/docs)" -ForegroundColor Cyan
Write-Host ""
Write-Host "Manual startup commands (alternative):" -ForegroundColor DarkGray
Write-Host "  Terminal 1 (Backend):  cd backend; python -m uvicorn app.main:app --reload --port 8000" -ForegroundColor DarkGray
Write-Host "  Terminal 2 (Frontend): cd frontend; npm run dev" -ForegroundColor DarkGray
Write-Host ""
