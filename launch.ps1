# AirControl Launcher (PowerShell)

Set-Location $PSScriptRoot

# Find Python
$pythonPath = $null
$candidates = @("python", "python3")
foreach ($c in $candidates) {
    try {
        $ver = & $c --version 2>$null
        if ($LASTEXITCODE -eq 0) { $pythonPath = $c; break }
    } catch {}
}

if (-not $pythonPath) {
    Write-Host "ERROR: Python not found" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}
Write-Host "Python: $pythonPath" -ForegroundColor Green

# Check MediaPipe model
$mpModel = "models\hand_landmarker.task"
if (-not (Test-Path $mpModel)) {
    Write-Host "WARNING: MediaPipe model not found at $mpModel" -ForegroundColor Red
    Read-Host "Press Enter to exit"
    exit 1
}

# Launch
Write-Host "Launching AirControl..." -ForegroundColor Cyan
& $pythonPath -m app.main_ui

if ($LASTEXITCODE -ne 0) {
    Write-Host "Exit code: $LASTEXITCODE" -ForegroundColor Red
    Read-Host "Press Enter to exit"
}
