"""
Cliente HTTP asíncrono hacia el endpoint único de inferencia (INFERENCE_URL).
Cambiar entre RunPod y Google Colab consiste únicamente en actualizar INFERENCE_URL
(desde .env / Render o en vivo desde la página Estado como administrador).
"""

from __future__ import annotations

import time
from typing import Any, Dict, Optional

import httpx
from fastapi import HTTPException

from app.config import get_settings


class InferenceClient:
    def __init__(self) -> None:
        self._override_url: Optional[str] = None
        self._override_api_key: Optional[str] = None

    def set_runtime_config(
        self,
        url: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        if url is not None:
            self._override_url = url.strip().rstrip("/")
        if api_key is not None:
            self._override_api_key = api_key.strip()

    @property
    def active_url(self) -> str:
        if self._override_url is not None:
            return self._override_url
        return get_settings().inference_url.strip().rstrip("/")

    @property
    def active_api_key(self) -> str:
        if self._override_api_key is not None:
            return self._override_api_key
        return get_settings().inference_api_key.strip()

    def _headers(self) -> Dict[str, str]:
        headers: Dict[str, str] = {}
        if self.active_api_key:
            headers["X-API-Key"] = self.active_api_key
        return headers

    async def check_health(self) -> Dict[str, Any]:
        url = self.active_url
        if not url:
            return {
                "online": False,
                "url": "",
                "error": (
                    "INFERENCE_URL no está configurada. Pega la URL HTTPS de tu servidor en "
                    "RunPod o Google Colab en esta misma página (Configurar Endpoint)."
                ),
            }
        t0 = time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=8.0) as client:
                resp = await client.get(f"{url}/health", headers=self._headers())
            latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)
            if resp.status_code == 200:
                return {
                    "online": True,
                    "url": url,
                    "latency_ms": latency_ms,
                    "details": resp.json(),
                }
            return {
                "online": False,
                "url": url,
                "latency_ms": latency_ms,
                "status_code": resp.status_code,
                "error": resp.text[:300],
            }
        except Exception as exc:
            return {"online": False, "url": url, "error": str(exc)}

    async def infer_full(self, image_bytes: bytes, filename: str = "frame.jpg") -> Dict[str, Any]:
        settings = get_settings()
        url = self.active_url
        if not url:
            raise HTTPException(
                status_code=503,
                detail=(
                    "El endpoint de inferencia (INFERENCE_URL) no está configurado. "
                    "Ve a la pestaña 'Estado' y pega la URL HTTPS de RunPod o Google Colab."
                ),
            )
        try:
            async with httpx.AsyncClient(timeout=settings.inference_timeout_seconds) as client:
                resp = await client.post(
                    f"{url}/infer",
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
                detail=f"No se pudo conectar al endpoint de inferencia ({url}): {exc}",
            )

    async def infer_plate(self, image_bytes: bytes, filename: str = "placa.jpg") -> Dict[str, Any]:
        settings = get_settings()
        url = self.active_url
        if not url:
            raise HTTPException(
                status_code=503,
                detail=(
                    "El endpoint de inferencia (INFERENCE_URL) no está configurado. "
                    "Ve a la pestaña 'Estado' y pega la URL HTTPS de RunPod o Google Colab."
                ),
            )
        try:
            async with httpx.AsyncClient(timeout=settings.inference_timeout_seconds) as client:
                resp = await client.post(
                    f"{url}/infer/placa",
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
