# Script de desarrollo local en Windows (SOLO para programar y probar en tu PC).
# Recuerda: para la demostración y entrega final, el sistema debe correr en la nube (Render/VPS + RunPod).

$ErrorActionPreference = "Stop"

Write-Host "=== 1. Verificando Python y Node.js ===" -ForegroundColor Cyan
$py = (Get-Command python -ErrorAction SilentlyContinue | Select-Object -First 1 -ExpandProperty Source)
if (-not $py -or $py -like "*WindowsApps*") {
    $candidate = "$env:LOCALAPPDATA\Python\bin\python.exe"
    if (Test-Path $candidate) { $py = $candidate }
}
if (-not $py) {
    Write-Error "No se encontró Python instalado. Instala Python 3.11+ desde python.org."
    exit 1
}
Write-Host "Usando Python: $py"

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Se creó .env desde .env.example. Completa GOOGLE_CLIENT_ID e INFERENCE_URL." -ForegroundColor Yellow
}

Write-Host "=== 2. Verificando dependencias del Backend ===" -ForegroundColor Cyan
& $py -m pip install -q fastapi uvicorn sqlalchemy httpx google-auth PyJWT python-multipart pydantic-settings websockets pytest python-dotenv

Write-Host "=== 3. Compilando Frontend React ===" -ForegroundColor Cyan
if (-not (Test-Path "web\frontend\node_modules")) {
    npm.cmd --prefix web\frontend install
}
npm.cmd --prefix web\frontend run build

Write-Host "=== 4. Iniciando Servidor Web Unificado (puerto 8000) ===" -ForegroundColor Green
Write-Host "Abre en tu navegador: http://localhost:8000  (Swagger API: http://localhost:8000/docs)" -ForegroundColor Green

$env:DATABASE_URL = "sqlite:///./peaje_dev.db"
$env:FRONTEND_DIST_DIR = "$PWD\web\frontend\dist"
Set-Location web\backend
& $py -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
