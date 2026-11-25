#!/usr/bin/env python3
"""
Script de diagnóstico de entorno (Sección 10 - Bonificación +5).
Imprime versiones de Python, PyTorch, CUDA, GPU, Ultralytics, OpenCV y EasyOCR.
Ejecutar en RunPod y/o Google Colab y guardar la salida en la carpeta de evidencias.
"""

import json
import os
import platform
import sys
from datetime import datetime, timezone


def safe_version(module_name: str):
    try:
        mod = __import__(module_name)
        return getattr(mod, "__version__", "installed (no __version__)")
    except Exception as exc:
        return f"NOT AVAILABLE ({exc})"


def main():
    report = {
        "servicio": "inferencia",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "os": platform.platform(),
        "python_version": sys.version.split()[0],
        "python_executable": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "packages": {
            "torch": safe_version("torch"),
            "torchvision": safe_version("torchvision"),
            "ultralytics": safe_version("ultralytics"),
            "cv2": safe_version("cv2"),
            "easyocr": safe_version("easyocr"),
            "fastapi": safe_version("fastapi"),
            "numpy": safe_version("numpy"),
        },
        "cuda": {
            "available": False,
            "cuda_version": None,
            "device_count": 0,
            "devices": [],
        },
        "models_configured": {
            "M1_placas": os.environ.get("MODEL_M1_PLATES_PATH", "models/m1_placas.pt"),
            "M2_vehiculos_seg": os.environ.get("MODEL_M2_VEHICLE_SEG_PATH", "models/m2_vehiculos_seg.pt"),
            "M3_ruedas_pose": os.environ.get("MODEL_M3_WHEEL_POSE_PATH", "models/m3_ruedas_pose.pt"),
            "M4_pasajeros_seg": os.environ.get("MODEL_M4_PASSENGERS_SEG_PATH", "models/m4_pasajeros_seg.pt"),
        },
    }

    try:
        import torch

        report["cuda"]["available"] = bool(torch.cuda.is_available())
        report["cuda"]["cuda_version"] = torch.version.cuda
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            report["cuda"]["device_count"] = count
            report["cuda"]["devices"] = [
                {
                    "index": i,
                    "name": torch.cuda.get_device_name(i),
                    "memory_gb": round(
                        torch.cuda.get_device_properties(i).total_memory / (1024**3), 2
                    ),
                }
                for i in range(count)
            ]
    except Exception as exc:
        report["cuda"]["error"] = str(exc)

    print("=" * 72)
    print("DIAGNOSTICO DE ENTORNO - SERVICIO DE INFERENCIA (PEAJE IA)")
    print("=" * 72)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    print("=" * 72)


if __name__ == "__main__":
    main()
