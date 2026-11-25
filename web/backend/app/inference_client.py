"""
Cliente HTTP asíncrono hacia el endpoint único de inferencia (INFERENCE_URL).
Cambiar entre RunPod y Google Colab consiste únicamente en actualizar INFERENCE_URL en .env.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import httpx
from fastapi import HTTPException

from app.config import get_settings


class InferenceClient:
    def _headers(self) -> Dict[str, str]:
        settings = get_settings()
        headers: Dict[str, str] = {}
        if settings.inference_api_key:
            headers["X-API-Key"] = settings.inference_api_key
        return headers

    async def check_health(self) -> Dict[str, Any]:
        settings = get_settings()
        if not settings.inference_url:
            return {
                "online": False,
                "url": "",
                "error": "INFERENCE_URL no está configurada en el entorno.",
            }
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=6.0) as client:
                resp = await client.get(
                    f"{settings.inference_url}/health",
                    headers=self._headers(),
                )
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resp.status_code == 200:
                return {
                    "online": True,
                    "url": settings.inference_url,
                    "latency_ms": latency_ms,
                    "details": resp.json(),
                }
            return {
                "online": False,
                "url": settings.inference_url,
                "latency_ms": latency_ms,
                "status_code": resp.status_code,
                "error": resp.text[:300],
            }
        except Exception as exc:
            return {
                "online": False,
                "url": settings.inference_url,
                "error": str(exc),
            }

    async def infer_full(self, image_bytes: bytes, filename: str = "frame.jpg") -> Dict[str, Any]:
        settings = get_settings()
        if not settings.inference_url:
            raise HTTPException(
                status_code=503,
                detail="El endpoint de inferencia (INFERENCE_URL) no está configurado.",
            )
        try:
            async with httpx.AsyncClient(timeout=settings.inference_timeout_seconds) as client:
                resp = await client.post(
                    f"{settings.inference_url}/infer",
                    headers=self._headers(),
                    files={"file": (filename, image_bytes, "image/jpeg")},
                )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Error del servicio de inferencia ({resp.status_code}): {resp.text[:300]}",
                )
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo conectar al endpoint de inferencia ({settings.inference_url}): {exc}",
            )

    async def infer_plate(self, image_bytes: bytes, filename: str = "placa.jpg") -> Dict[str, Any]:
        settings = get_settings()
        if not settings.inference_url:
            raise HTTPException(
                status_code=503,
                detail="El endpoint de inferencia (INFERENCE_URL) no está configurado.",
            )
        try:
            async with httpx.AsyncClient(timeout=settings.inference_timeout_seconds) as client:
                resp = await client.post(
                    f"{settings.inference_url}/infer/placa",
                    headers=self._headers(),
                    files={"file": (filename, image_bytes, "image/jpeg")},
                )
            if resp.status_code != 200:
                raise HTTPException(
                    status_code=502,
                    detail=f"Error en OCR/M1 del servicio de inferencia ({resp.status_code}): {resp.text[:300]}",
                )
            return resp.json()
        except HTTPException:
            raise
        except Exception as exc:
            raise HTTPException(
                status_code=502,
                detail=f"No se pudo conectar al servicio de inferencia para leer placa: {exc}",
            )


inference_client = InferenceClient()
