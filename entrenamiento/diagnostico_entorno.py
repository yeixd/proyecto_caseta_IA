#!/usr/bin/env python3
"""
Script de diagnóstico de entorno de entrenamiento (Google Colab / Conda / Venv).
Guardar la salida en la carpeta de evidencias en la nube (Sección 10).
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone


def safe_ver(name: str):
    try:
        m = __import__(name)
        return getattr(m, "__version__", "installed")
    except Exception as exc:
        return f"NOT AVAILABLE ({exc})"


def main():
    info = {
        "servicio": "entrenamiento",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "os": platform.platform(),
        "python": sys.version.split()[0],
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "packages": {
            "torch": safe_ver("torch"),
            "torchvision": safe_ver("torchvision"),
            "ultralytics": safe_ver("ultralytics"),
            "roboflow": safe_ver("roboflow"),
            "easyocr": safe_ver("easyocr"),
            "cv2": safe_ver("cv2"),
            "numpy": safe_ver("numpy"),
        },
        "cuda": {},
    }
    try:
        import torch

        info["cuda"] = {
            "available": torch.cuda.is_available(),
            "version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
    except Exception as exc:
        info["cuda"] = {"error": str(exc)}

    print(json.dumps(info, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
