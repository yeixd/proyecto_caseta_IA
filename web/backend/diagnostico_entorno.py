#!/usr/bin/env python3
"""
Script de diagnóstico de entorno del Backend Web (Sección 10 - Bonificación +5).
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone


def safe_ver(mod: str):
    try:
        m = __import__(mod)
        return getattr(m, "__version__", "installed")
    except Exception as exc:
        return f"NOT AVAILABLE ({exc})"


def main():
    report = {
        "servicio": "web-backend",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "packages": {
            "fastapi": safe_ver("fastapi"),
            "uvicorn": safe_ver("uvicorn"),
            "sqlalchemy": safe_ver("sqlalchemy"),
            "httpx": safe_ver("httpx"),
            "google.auth": safe_ver("google.auth"),
            "jwt": safe_ver("jwt"),
        },
        "inference_url_configured": bool(os.environ.get("INFERENCE_URL")),
        "ollama_url_configured": bool(os.environ.get("OLLAMA_URL")),
    }
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
