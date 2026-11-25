# Script para iniciar la App Móvil en tu celular (Expo Go) o compilar el APK en la nube (EAS Build)
param(
    [ValidateSet("expo", "apk")]
    [string]$Modo = "expo"
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Test-Path "node_modules")) {
    Write-Host "Instalando dependencias de la App Móvil..." -ForegroundColor Cyan
    npm.cmd install
}

if ($Modo -eq "apk") {
    Write-Host "Compilando APK Android en la nube con EAS Build..." -ForegroundColor Green
    npx.cmd eas login
    npx.cmd eas build -p android --profile preview
} else {
    Write-Host "Iniciando Expo: escanea el código QR con la app 'Expo Go' en tu celular..." -ForegroundColor Green
    npx.cmd expo start --tunnel
}
