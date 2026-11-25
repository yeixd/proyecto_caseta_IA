"""
Agente Conversacional del Sistema de Control de Peaje (Sección 4.8).

Características obligatorias:
  - Solo disponible para usuarios autenticados con Google.
  - Recibe placa por texto o por imagen (M1 + OCR).
  - Responde exclusivamente con datos reales obtenidos de la BD/API (sin inventar datos).
  - Clasifica exactamente en uno de los 4 estados:
      1. Registrada y con paso (fecha, caseta, categoría, monto)
      2. Registrada sin paso
      3. Identificada no registrada (posible evasión)
      4. No existe
  - Normaliza la placa y sugiere coincidencias cercanas por confusión OCR (O↔0, I↔1, B↔8, S↔5):
    "¿Quisiste decir...?"
  - Si la imagen no tiene placa legible, lo indica y pide otra imagen.
  - Usa Ollama en RunPod con function calling / contexto verificado, con fallback determinista
    garantizado si Ollama no está disponible.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.inference_client import inference_client
from app.plate_utils import extract_plate_candidate_from_text, lookup_plate_status

logger = logging.getLogger("peaje.agent")

SYSTEM_PROMPT = """Eres el Agente Conversacional oficial del Sistema Inteligente de Control de Peaje.
REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con los datos exactos entregados en el bloque JSON [DATOS_VERIFICADOS_BD].
2. NUNCA inventes placas, propietarios, fechas, casetas, montos ni saldos.
3. Menciona explícitamente el estado oficial exacto:
   - "Registrada y con paso"
   - "Registrada sin paso"
   - "Identificada no registrada"
   - "No existe"
4. Si el estado es "Registrada y con paso", incluye fecha, caseta, categoría y monto del último paso.
5. Si el campo "sugerencias" contiene placas, pregunta claramente: "¿Quisiste decir <placas>?"
6. Responde en español, de forma concisa, clara y profesional.
"""


def format_deterministic_reply(result: Dict[str, Any], from_image: bool = False) -> str:
    """Genera una respuesta exacta y libre de alucinaciones usando solo datos de la BD."""
    placa = result["placa_normalizada"]
    estado = result["estado"]
    vehiculo = result.get("vehiculo")
    ultimo = result.get("ultimo_paso")
    sugerencias = result.get("sugerencias") or []
    origen = "Imagen analizada (M1 + OCR)" if from_image else "Consulta por texto"

    if estado == "Registrada y con paso":
        lines = [
            f"Estado: **Registrada y con paso** ({origen}).",
            f"- **Placa:** `{placa}` | **Propietario:** {vehiculo['propietario']} | **Saldo actual:** Bs. {vehiculo['saldo']:.2f}",
            f"- **Total de pasos registrados:** {result['total_pasos']}",
        ]
        if ultimo:
            lines.append(
                f"- **Último paso:** {ultimo['fecha']} | **Caseta:** {ultimo['caseta']} | "
                f"**Categoría:** {ultimo['categoria']} ({ultimo['ejes']} ejes, {ultimo['pasajeros']} pasajeros) | "
                f"**Monto:** Bs. {ultimo['monto']:.2f} | **Resultado:** {ultimo['estado_paso']}"
            )
        return "\n".join(lines)

    if estado == "Registrada sin paso":
        estado_veh = "Activo" if vehiculo["activo"] else "Inactivo"
        return (
            f"Estado: **Registrada sin paso** ({origen}).\n"
            f"- La placa `{placa}` está registrada a nombre de **{vehiculo['propietario']}** "
            f"(Categoría: {vehiculo['categoria']}, {vehiculo['ejes']} ejes, Saldo: Bs. {vehiculo['saldo']:.2f}, Estado: {estado_veh}), "
            "pero nunca ha registrado un paso por el peaje."
        )

    if estado == "Identificada no registrada":
        msg = (
            f"Estado: **Identificada no registrada** (posible evasión) ({origen}).\n"
            f"- La placa `{placa}` fue detectada por cámara en **{result['total_pasos']}** paso(s), "
            "pero NO figura en el registro de vehículos autorizados."
        )
        if ultimo:
            msg += (
                f"\n- **Última detección:** {ultimo['fecha']} en **{ultimo['caseta']}** "
                f"(Categoría: {ultimo['categoria']}, Monto estimado: Bs. {ultimo['monto']:.2f})."
            )
        if sugerencias:
            sug_str = ", ".join(f"`{s}`" for s in sugerencias)
            msg += f"\n- Posible error de lectura OCR: **¿Quisiste decir {sug_str}?**"
        return msg

    # Estado: No existe
    msg = (
        f"Estado: **No existe** ({origen}).\n"
        f"- No existe ningún registro de la placa `{placa}` en vehículos registrados ni en el historial de pasos."
    )
    if not result.get("formato_valido", True):
        msg += "\n- Nota: el texto ingresado no coincide con el formato estándar de placa (5 a 8 caracteres alfanuméricos)."
    if sugerencias:
        sug_str = ", ".join(f"`{s}`" for s in sugerencias)
        msg += f"\n- Coincidencias cercanas (confusión OCR O↔0, I↔1, B↔8, S↔5): **¿Quisiste decir {sug_str}?**"
    return msg


async def _try_ollama_response(
    user_message: str,
    verified_data: Dict[str, Any],
    from_image: bool,
) -> Optional[str]:
    """
    Consulta Ollama en RunPod para redactar la respuesta natural a partir del JSON
    verificado de la BD. Si Ollama no está disponible, retorna None para usar el fallback determinista.
    """
    settings = get_settings()
    if not settings.ollama_url:
        return None

    payload = {
        "model": settings.ollama_model,
        "stream": False,
        "options": {"temperature": 0.1},
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Mensaje del usuario: {user_message or '(consulta mediante imagen de placa)'}\n"
                    f"Origen: {'imagen' if from_image else 'texto'}\n"
                    f"[DATOS_VERIFICADOS_BD]\n{json.dumps(verified_data, ensure_ascii=False)}"
                ),
            },
        ],
    }

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            resp = await client.post(f"{settings.ollama_url}/api/chat", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            content = body.get("message", {}).get("content", "").strip()
            if content:
                return content
    except Exception as exc:
        logger.warning("Ollama no disponible en %s (%s); usando motor determinista.", settings.ollama_url, exc)
    return None


async def check_ollama_health() -> Dict[str, Any]:
    settings = get_settings()
    if not settings.ollama_url:
        return {
            "online": False,
            "mode": "determinista_sin_alucinaciones",
            "detail": "OLLAMA_URL no configurado; usando motor determinista sobre BD.",
        }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.ollama_url}/api/tags")
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {
                "online": True,
                "url": settings.ollama_url,
                "configured_model": settings.ollama_model,
                "available_models": models,
            }
        return {"online": False, "url": settings.ollama_url, "status_code": resp.status_code}
    except Exception as exc:
        return {"online": False, "url": settings.ollama_url, "error": str(exc)}


async def run_conversational_agent(
    db: Session,
    mensaje: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "placa.jpg",
) -> Dict[str, Any]:
    """Ejecuta el flujo completo del agente por imagen o por texto."""
    from_image = image_bytes is not None and len(image_bytes) > 0
    ocr_info: Optional[Dict[str, Any]] = None
    plate_raw = ""

    if from_image:
        infer_res = await inference_client.infer_plate(image_bytes, filename=filename)
        ocr_info = infer_res.get("placa", {})
        plate_raw = str(ocr_info.get("texto", "")).strip()
        legible = bool(ocr_info.get("legible", len(plate_raw) >= 4))

        if not plate_raw or not legible:
            return {
                "respuesta": (
                    "No se pudo detectar una placa legible en la imagen enviada. "
                    "Por favor, toma o sube otra imagen más clara y enfocada de la placa."
                ),
                "estado": "Placa ilegible",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "ocr": ocr_info,
                "motor": "validacion_ocr",
            }
    else:
        if not mensaje or not mensaje.strip():
            return {
                "respuesta": (
                    "Por favor escribe una placa (por ejemplo: `2481-KLM`) o adjunta una fotografía de la placa para consultarla."
                ),
                "estado": "Sin consulta",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "ocr": None,
                "motor": "determinista",
            }

        extracted = extract_plate_candidate_from_text(mensaje)
        if not extracted:
            return {
                "respuesta": (
                    "No identifiqué una placa válida en tu mensaje. "
                    "Indica el número de placa (por ejemplo: `2481KLM` o `4510-BGD`) o envía una foto de la placa."
                ),
                "estado": "Sin placa detectada",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "ocr": None,
                "motor": "determinista",
            }
        plate_raw = extracted

    # Consultar datos reales en BD (Tool execution)
    verified = lookup_plate_status(db, plate_raw)

    # Intentar redactar con Ollama en RunPod; si no responde, usar respuesta determinista exacta
    ollama_reply = await _try_ollama_response(mensaje or "", verified, from_image=from_image)
    if ollama_reply:
        final_reply = ollama_reply
        engine_used = f"ollama:{get_settings().ollama_model}"
    else:
        final_reply = format_deterministic_reply(verified, from_image=from_image)
        engine_used = "determinista_bd"

    return {
        "respuesta": final_reply,
        "estado": verified["estado"],
        "placa_normalizada": verified["placa_normalizada"],
        "sugerencias": verified["sugerencias"],
        "datos": verified,
        "ocr": ocr_info,
        "motor": engine_used,
    }
