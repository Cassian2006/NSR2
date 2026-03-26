param(
  [int]$BackendPort = 8001,
  [int]$FrontendPort = 5175
)

$ErrorActionPreference = "Stop"

$repoRoot = Split-Path -Parent $PSScriptRoot
$runDir = Join-Path $repoRoot ".codex-run"
$backendDir = Join-Path $repoRoot "backend"
$frontendDir = Join-Path $repoRoot "frontend"
$pythonExe = "C:\Users\cai yuan qi\AppData\Local\Programs\Python\Python311\python.exe"
$npmCmd = "C:\Program Files\nodejs\npm.cmd"

New-Item -ItemType Directory -Force -Path $runDir | Out-Null

function Stop-PortListeners {
  param([int[]]$Ports)
  foreach ($port in $Ports) {
    $listeners = Get-NetTCPConnection -State Listen -LocalPort $port -ErrorAction SilentlyContinue
    foreach ($listener in $listeners) {
      try {
        Stop-Process -Id $listener.OwningProcess -Force -ErrorAction Stop
        Write-Host "Stopped process $($listener.OwningProcess) on port $port"
      } catch {
        Write-Warning ("Failed to stop process {0} on port {1}: {2}" -f $listener.OwningProcess, $port, $_.Exception.Message)
      }
    }
  }
}

function Wait-HttpReady {
  param(
    [string]$Url,
    [int]$TimeoutSec = 30
  )
  $deadline = (Get-Date).AddSeconds($TimeoutSec)
  while ((Get-Date) -lt $deadline) {
    try {
      $resp = Invoke-WebRequest -UseBasicParsing -Uri $Url -TimeoutSec 3
      if ($resp.StatusCode -ge 200 -and $resp.StatusCode -lt 500) {
        return $true
      }
    } catch {
      Start-Sleep -Milliseconds 800
    }
  }
  return $false
}

function Start-Backend {
  $stdout = Join-Path $runDir "backend-$BackendPort.out.log"
  $stderr = Join-Path $runDir "backend-$BackendPort.err.log"
  Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue
  $cmd = @"
Set-Location '$backendDir'
\$env:PYTHONPATH='.'
& '$pythonExe' -m uvicorn app.main:app --host 127.0.0.1 --port $BackendPort
"@
  Start-Process powershell.exe `
    -WindowStyle Minimized `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$cmd `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr | Out-Null
}

function Start-Frontend {
  $stdout = Join-Path $runDir "frontend-$FrontendPort.out.log"
  $stderr = Join-Path $runDir "frontend-$FrontendPort.err.log"
  Remove-Item $stdout,$stderr -Force -ErrorAction SilentlyContinue
  $cmd = @"
Set-Location '$frontendDir'
& '$npmCmd' run dev -- --host 127.0.0.1 --port $FrontendPort
"@
  Start-Process powershell.exe `
    -WindowStyle Minimized `
    -ArgumentList "-NoProfile","-ExecutionPolicy","Bypass","-Command",$cmd `
    -RedirectStandardOutput $stdout `
    -RedirectStandardError $stderr | Out-Null
}

Write-Host "Cleaning listeners on ports $BackendPort and $FrontendPort ..."
Stop-PortListeners -Ports @($BackendPort, $FrontendPort)

Write-Host "Starting backend ..."
Start-Backend
if (-not (Wait-HttpReady -Url "http://127.0.0.1:$BackendPort/healthz" -TimeoutSec 40)) {
  throw "Backend failed to start on port $BackendPort. Check $runDir"
}

Write-Host "Starting frontend ..."
Start-Frontend
if (-not (Wait-HttpReady -Url "http://127.0.0.1:$FrontendPort" -TimeoutSec 40)) {
  throw "Frontend failed to start on port $FrontendPort. Check $runDir"
}

$backendPid = (Get-NetTCPConnection -State Listen -LocalPort $BackendPort | Select-Object -First 1 -ExpandProperty OwningProcess)
$frontendPid = (Get-NetTCPConnection -State Listen -LocalPort $FrontendPort | Select-Object -First 1 -ExpandProperty OwningProcess)

Write-Host ""
Write-Host "NSR2 is ready."
Write-Host "Frontend: http://127.0.0.1:$FrontendPort"
Write-Host "Backend : http://127.0.0.1:$BackendPort"
Write-Host "Docs    : http://127.0.0.1:$BackendPort/docs"
Write-Host "Frontend PID: $frontendPid"
Write-Host "Backend PID : $backendPid"
Write-Host "Logs: $runDir"

Start-Process "http://127.0.0.1:$FrontendPort"
