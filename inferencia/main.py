"""
Servidor HTTP único de inferencia (RunPod Pod o Google Colab).
Expone M1 (placas + OCR), M2 (segmentación vehículo), M3 (keypoints ruedas) y M4 (pasajeros).
"""

from __future__ import annotations

import base64
import logging
import os
from typing import Any, Dict, Optional

import cv2
import httpx
import numpy as np
from dotenv import load_dotenv
from fastapi import Body, Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from pipeline import TollInferencePipeline

load_dotenv()
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("peaje.inferencia.api")

INFERENCE_API_KEY = os.environ.get("INFERENCE_API_KEY", "").strip()
LOCAL_OLLAMA_BASE = os.environ.get("LOCAL_OLLAMA_BASE", "http://127.0.0.1:11434").rstrip("/")

app = FastAPI(
    title="Peaje IA - Servicio Único de Inferencia",
    description="M1 (Placas + OCR) + M2 (YOLO-seg Vehículos) + M3 (YOLO-pose Ruedas) + M4 (Segmentación Pasajeros)",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline: Optional[TollInferencePipeline] = None


@app.on_event("startup")
def startup_event():
    global pipeline
    logger.info("Inicializando TollInferencePipeline...")
    pipeline = TollInferencePipeline()
    logger.info("Pipeline listo: %s", pipeline.health())


def verify_api_key(x_api_key: Optional[str] = Header(default=None)):
    """Protege el endpoint si INFERENCE_API_KEY está configurado."""
    if INFERENCE_API_KEY and x_api_key != INFERENCE_API_KEY:
        raise HTTPException(status_code=401, detail="X-API-Key de inferencia inválida")


class Base64ImageRequest(BaseModel):
    image_b64: str
    include_annotated: bool = True


def decode_image_bytes(raw: bytes) -> np.ndarray:
    arr = np.frombuffer(raw, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(status_code=400, detail="No se pudo decodificar la imagen enviada")
    return img


def decode_base64_image(b64_str: str) -> np.ndarray:
    if "," in b64_str:
        b64_str = b64_str.split(",", 1)[1]
    try:
        raw = base64.b64decode(b64_str)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Base64 inválido: {exc}")
    return decode_image_bytes(raw)


@app.get("/health")
def health():
    if pipeline is None:
        return {"status": "initializing"}
    return pipeline.health()


@app.post("/infer", dependencies=[Depends(verify_api_key)])
async def infer_upload(
    file: UploadFile = File(...),
    include_annotated: bool = True,
):
    """Inferencia completa vía multipart/form-data (usado por el backend web)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline aún inicializando")
    raw = await file.read()
    img = decode_image_bytes(raw)
    return pipeline.infer_full(img, include_annotated=include_annotated)


@app.post("/infer/json", dependencies=[Depends(verify_api_key)])
def infer_json(payload: Base64ImageRequest):
    """Inferencia completa vía JSON base64 (compatible con clientes REST y pruebas)."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline aún inicializando")
    img = decode_base64_image(payload.image_b64)
    return pipeline.infer_full(img, include_annotated=payload.include_annotated)


@app.post("/infer/placa", dependencies=[Depends(verify_api_key)])
async def infer_plate_upload(file: UploadFile = File(...)):
    """Inferencia rápida solo de M1 + OCR para consultas por imagen del agente."""
    if pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline aún inicializando")
    raw = await file.read()
    img = decode_image_bytes(raw)
    return pipeline.infer_plate_only(img)


# ----------------------------------------------------------------------
# Proxy integrado hacia Ollama local (permite usar UN SOLO túnel/URL para
# Inferencia + Agente LLM tanto en Colab como en RunPod)
# ----------------------------------------------------------------------
@app.get("/ollama/api/tags")
async def proxy_ollama_tags():
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(f"{LOCAL_OLLAMA_BASE}/api/tags")
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Ollama local no responde: {exc}")


@app.post("/ollama/api/chat")
async def proxy_ollama_chat(payload: Dict[str, Any] = Body(...)):
    try:
        async with httpx.AsyncClient(timeout=90.0) as client:
            resp = await client.post(f"{LOCAL_OLLAMA_BASE}/api/chat", json=payload)
        if resp.status_code != 200:
            raise HTTPException(status_code=resp.status_code, detail=resp.text[:300])
        return resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Error llamando a Ollama local: {exc}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", "8000"))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=False)
