#!/usr/bin/env bash
set -euo pipefail

echo "=== Ejecutando diagnóstico de entorno en RunPod ==="
python /app/diagnostico_entorno.py

echo "=== Iniciando servidor único de inferencia (M1+M2+M3+M4) en puerto ${PORT:-8000} ==="
exec uvicorn main:app --host 0.0.0.0 --port "${PORT:-8000}" --workers 1
