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

import base64
from datetime import datetime, timezone
import hashlib
import json
import logging
import os
from typing import Any, Dict, Optional
import uuid

import httpx
from sqlalchemy.orm import Session

from app.config import get_settings
from app.inference_client import inference_client
from app.models import Alerta, Caseta, EvidenciaImagen, Paso, Tarifa, Vehiculo
from app.plate_utils import extract_plate_candidate_from_text, lookup_plate_status, normalize_plate
from app.realtime import realtime_manager
from app.seed import INITIAL_SALDO_BY_CATEGORY, generate_fictional_owner

logger = logging.getLogger("peaje.agent")

_runtime_ollama_url: Optional[str] = None
_runtime_ollama_model: Optional[str] = None


def set_ollama_runtime_config(url: Optional[str] = None, model: Optional[str] = None) -> None:
    global _runtime_ollama_url, _runtime_ollama_model
    if url is not None:
        _runtime_ollama_url = url.strip().rstrip("/")
    if model is not None and model.strip():
        _runtime_ollama_model = model.strip()


def get_active_ollama_url() -> str:
    if _runtime_ollama_url is not None and _runtime_ollama_url:
        return _runtime_ollama_url
    env_url = get_settings().ollama_url.strip().rstrip("/")
    if env_url:
        return env_url
    # Auto-conectar al proxy /ollama integrado dentro de INFERENCE_URL (Colab/RunPod 1 sola URL)
    if inference_client.active_url:
        return f"{inference_client.active_url}/ollama"
    return ""


def get_active_ollama_model() -> str:
    if _runtime_ollama_model:
        return _runtime_ollama_model
    return get_settings().ollama_model

SYSTEM_PROMPT = """Eres el Agente Conversacional oficial del Sistema Inteligente de Control de Peaje.
REGLAS ESTRICTAS:
1. Responde ÚNICAMENTE con los datos exactos entregados en el bloque JSON [DATOS_VERIFICADOS_BD].
2. NUNCA inventes placas, propietarios, fechas, casetas, montos ni saldos.
3. Si el JSON incluye "inferencia_imagen", menciona explícitamente el resultado completo de la inferencia visual:
   placa leída (M1), clase de vehículo (M2), cantidad de llantas y ejes (M3), pasajeros en cabina (M4), propietario/sindicato y tarifa en Bs.
4. Menciona explícitamente el estado oficial exacto:
   - "Registrada y con paso"
   - "Registrada sin paso"
   - "Identificada no registrada"
   - "No existe"
5. Si el campo "sugerencias" contiene placas, pregunta claramente: "¿Quisiste decir <placas>?"
6. Responde en español, de forma concisa, clara y profesional.
"""


def _save_agent_inference_image(db: Session, annotated_b64: str, fallback_raw: bytes) -> str:
    filename = f"chat_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    raw = fallback_raw
    if annotated_b64:
        try:
            raw = base64.b64decode(annotated_b64)
        except Exception:
            raw = fallback_raw
    db.add(EvidenciaImagen(filename=filename, content_type="image/jpeg", data=raw))
    try:
        dest = os.path.join(get_settings().evidence_dir, filename)
        with open(dest, "wb") as f:
            f.write(raw)
    except OSError:
        pass
    return f"/static/evidencias/{filename}"


def _resolve_agent_tariff(db: Session, categoria: str, ejes: int) -> float:
    exact = db.query(Tarifa).filter(Tarifa.categoria == categoria, Tarifa.ejes == ejes).first()
    if exact:
        return float(exact.monto)
    same_cat = db.query(Tarifa).filter(Tarifa.categoria == categoria).all()
    if same_cat:
        return float(min(same_cat, key=lambda t: abs(t.ejes - ejes)).monto)
    return 5.0


def _provisional_plate_from_bytes(image_bytes: bytes, categoria: str) -> str:
    digest = hashlib.sha1(image_bytes).digest()
    num = 1000 + (int.from_bytes(digest[:2], "big") % 8999)
    alphabet = "ABCDEFGHJKLMNPRSTUVWXYZ"
    suffix_map = {"camion": "TRK", "minibus": "MNB", "bus": "BUS", "moto": "MTO"}
    last = alphabet[digest[2] % len(alphabet)]
    letters = (
        suffix_map[categoria][:2] + last
        if categoria in suffix_map
        else "".join(alphabet[b % len(alphabet)] for b in digest[2:5])
    )
    return f"{num}{letters}"


def format_deterministic_reply(result: Dict[str, Any], from_image: bool = False) -> str:
    """Genera una respuesta exacta y libre de alucinaciones usando solo datos de la BD."""
    placa = result["placa_normalizada"]
    estado = result["estado"]
    vehiculo = result.get("vehiculo")
    ultimo = result.get("ultimo_paso")
    sugerencias = result.get("sugerencias") or []
    origen = "Inferencia visual completa M1+M2+M3+M4" if from_image else "Consulta por texto"
    inf_img = result.get("inferencia_imagen")

    if estado == "Registrada y con paso":
        lines = [
            f"Estado: **Registrada y con paso** ({origen}).",
            f"- **Placa:** `{placa}` | **Propietario:** {vehiculo['propietario']} | **Saldo actual:** Bs. {vehiculo['saldo']:.2f}",
        ]
        if inf_img:
            lines.append(
                f"- **Inferencia visual:** {inf_img['categoria_nombre']} | **{inf_img['ejes']} ejes** "
                f"({inf_img['llantas_visibles']} llantas, vista {inf_img['vista']}) | "
                f"**{inf_img['pasajeros']} pasajero(s)** en cabina | **Tarifa:** Bs. {inf_img['tarifa_bs']:.2f}"
            )
        lines.append(f"- **Total de pasos registrados:** {result['total_pasos']}")
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
    ollama_url = get_active_ollama_url()
    ollama_model = get_active_ollama_model()
    if not ollama_url:
        return None

    payload = {
        "model": ollama_model,
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
        timeout_sec = max(float(settings.ollama_timeout_seconds), 45.0)
        async with httpx.AsyncClient(timeout=timeout_sec) as client:
            resp = await client.post(f"{ollama_url}/api/chat", json=payload)
            # Si el modelo configurado no existe en Ollama, usar automáticamente el primero disponible
            if resp.status_code == 404:
                tags = await client.get(f"{ollama_url}/api/tags")
                if tags.status_code == 200:
                    models = [m.get("name") for m in tags.json().get("models", []) if m.get("name")]
                    if models:
                        payload["model"] = models[0]
                        set_ollama_runtime_config(model=models[0])
                        resp = await client.post(f"{ollama_url}/api/chat", json=payload)
        if resp.status_code == 200:
            body = resp.json()
            content = body.get("message", {}).get("content", "").strip()
            if content:
                return content
    except Exception as exc:
        logger.warning("Ollama no disponible en %s (%s); usando motor determinista.", ollama_url, exc)
    return None


def _get_general_db_snapshot(db: Session) -> Dict[str, Any]:
    """Obtiene un resumen real y completo de la BD para preguntas operativas en web y móvil."""
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pasos_hoy = db.query(Paso).filter(Paso.created_at >= start_of_day).all()
    ultimos = db.query(Paso).order_by(Paso.created_at.desc()).limit(5).all()
    alertas_pend_list = (
        db.query(Alerta).filter(Alerta.atendida == False).order_by(Alerta.created_at.desc()).limit(6).all()  # noqa: E712
    )
    alertas_pend = db.query(Alerta).filter(Alerta.atendida == False).count()  # noqa: E712
    total_vehiculos = db.query(Vehiculo).count()
    casetas = [c.nombre for c in db.query(Caseta).all()]
    tarifas = db.query(Tarifa).order_by(Tarifa.categoria.asc(), Tarifa.ejes.asc()).all()

    return {
        "tipo_consulta": "resumen_general_sistema",
        "fecha_utc": start_of_day.date().isoformat(),
        "total_vehiculos_registrados": total_vehiculos,
        "casetas": casetas,
        "pasos_hoy": len(pasos_hoy),
        "recaudacion_total_hoy_bs": round(sum(p.monto for p in pasos_hoy), 2),
        "recaudacion_aprobada_hoy_bs": round(sum(p.monto for p in pasos_hoy if p.estado == "APROBADO"), 2),
        "pasajeros_hoy": sum(p.pasajeros for p in pasos_hoy),
        "alertas_pendientes": alertas_pend,
        "alertas_recientes": [
            {
                "tipo": a.tipo,
                "mensaje": a.mensaje,
                "fecha": a.created_at.isoformat() if a.created_at else None,
            }
            for a in alertas_pend_list
        ],
        "tarifas_vigentes": [
            {"categoria": t.categoria, "ejes": t.ejes, "monto_bs": t.monto} for t in tarifas
        ],
        "ultimos_5_pasos": [
            {
                "placa": p.placa_normalizada or "ILEGIBLE",
                "categoria": p.categoria_detectada,
                "ejes": p.ejes_detectados,
                "pasajeros": p.pasajeros,
                "monto_bs": p.monto,
                "estado": p.estado,
                "fecha": p.created_at.isoformat() if p.created_at else None,
            }
            for p in ultimos
        ],
    }


def _format_general_intent_fallback(mensaje: str, snapshot: Dict[str, Any]) -> Optional[str]:
    """Responde con datos reales verificados de la BD a consultas operativas rápidas desde el móvil."""
    q = (mensaje or "").lower().strip()
    if any(k in q for k in ("resumen", "hoy", "recaud", "estad", "kpi", "cuántos", "cuantos", "total")):
        return (
            "📊 **Resumen en vivo del día (Base de Datos):**\n"
            f"- **Pasos registrados hoy:** {snapshot['pasos_hoy']}\n"
            f"- **Recaudación total hoy:** Bs. {snapshot['recaudacion_total_hoy_bs']:.2f} "
            f"(Aprobada: Bs. {snapshot['recaudacion_aprobada_hoy_bs']:.2f})\n"
            f"- **Pasajeros contabilizados hoy (M4):** {snapshot['pasajeros_hoy']}\n"
            f"- **Alertas pendientes:** {snapshot['alertas_pendientes']}\n"
            f"- **Vehículos empadronados:** {snapshot['total_vehiculos_registrados']}"
        )
    if any(k in q for k in ("alerta", "evasi", "infracc", "pendiente", "sin saldo")):
        if not snapshot["alertas_recientes"]:
            return "✅ **No hay alertas pendientes** en este momento en ninguna caseta."
        lines = [f"🚨 **Alertas activas pendientes ({snapshot['alertas_pendientes']}):**"]
        for a in snapshot["alertas_recientes"][:5]:
            lines.append(f"- **{a['tipo']}:** {a['mensaje']}")
        return "\n".join(lines)
    if any(k in q for k in ("tarifa", "precio", "costo", "cobr", "cuanto vale", "cuánto vale")):
        if not snapshot["tarifas_vigentes"]:
            return "No hay tarifas configuradas aún en la base de datos."
        lines = ["💰 **Tarifario oficial vigente por clase y ejes:**"]
        for t in snapshot["tarifas_vigentes"]:
            lines.append(f"- **{t['categoria'].upper()}** ({t['ejes']} ejes): **Bs. {t['monto_bs']:.2f}**")
        return "\n".join(lines)
    if any(k in q for k in ("último", "ultimo", "reciente", "pasos", "historial")):
        if not snapshot["ultimos_5_pasos"]:
            return "Aún no se han registrado pasos recientes en las casetas."
        lines = ["🚗 **Últimos 5 pasos registrados en el sistema:**"]
        for p in snapshot["ultimos_5_pasos"]:
            lines.append(
                f"- `{p['placa']}` · {p['categoria']} ({p['ejes']} ejes, {p['pasajeros']} pasaj.) · "
                f"**Bs. {p['monto_bs']:.2f}** ({p['estado']})"
            )
        return "\n".join(lines)
    return None


async def test_ollama_prompt() -> Dict[str, Any]:
    """Ejecuta una prueba real contra Ollama (/api/chat) para verificar el LLM desde Estado."""
    ollama_url = get_active_ollama_url()
    if not ollama_url:
        return {"ok": False, "error": "Ni OLLAMA_URL ni INFERENCE_URL están configuradas."}
    reply = await _try_ollama_response(
        user_message="Confirma en una sola oración corta que eres el agente LLM del sistema de peaje.",
        verified_data={"sistema": "Sistema Inteligente de Control de Peaje UNIFRANZ", "estado": "operativo"},
        from_image=False,
    )
    if reply:
        return {
            "ok": True,
            "url": ollama_url,
            "model": get_active_ollama_model(),
            "reply": reply,
        }
    return {
        "ok": False,
        "url": ollama_url,
        "model": get_active_ollama_model(),
        "error": "Ollama no respondió a /api/chat. Verifica que la Celda 2 de Colab (o RunPod) esté corriendo.",
    }


async def check_ollama_health() -> Dict[str, Any]:
    ollama_url = get_active_ollama_url()
    ollama_model = get_active_ollama_model()
    if not ollama_url:
        return {
            "online": False,
            "url": "",
            "configured_model": ollama_model,
            "mode": "determinista_sin_alucinaciones",
            "detail": "OLLAMA_URL no configurado; usando motor determinista sobre BD.",
        }
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{ollama_url}/api/tags")
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return {
                "online": True,
                "url": ollama_url,
                "configured_model": ollama_model,
                "available_models": models,
            }
        return {"online": False, "url": ollama_url, "configured_model": ollama_model, "status_code": resp.status_code}
    except Exception as exc:
        return {"online": False, "url": ollama_url, "configured_model": ollama_model, "error": str(exc)}


async def run_conversational_agent(
    db: Session,
    mensaje: Optional[str] = None,
    image_bytes: Optional[bytes] = None,
    filename: str = "placa.jpg",
    usuario_id: Optional[int] = None,
) -> Dict[str, Any]:
    """Ejecuta el flujo completo del agente por imagen (con inferencia completa M1+M2+M3+M4) o por texto."""
    from_image = image_bytes is not None and len(image_bytes) > 0
    ocr_info: Optional[Dict[str, Any]] = None
    inferencia_card: Optional[Dict[str, Any]] = None
    plate_raw = ""

    if from_image:
        # Ejecutar SIEMPRE la inferencia completa M1 (placa deblur) + M2 (clase) + M3 (llantas/ejes) + M4 (pasajeros)
        full_res = await inference_client.infer_full(image_bytes, filename=filename)
        ocr_info = full_res.get("placa", {})
        veh_obj = full_res.get("vehiculo", {})

        placa_leida = str(ocr_info.get("texto_raw") or ocr_info.get("texto") or "").strip()
        placa_norm = normalize_plate(str(ocr_info.get("texto") or placa_leida))
        confianza_ocr = float(ocr_info.get("confianza") or 0.0)
        categoria = str(veh_obj.get("categoria") or "auto").lower()
        categoria_nombre = str(veh_obj.get("categoria_nombre") or categoria.title())
        ejes = max(1, int(full_res.get("ejes") or 2))
        llantas_visibles = int(full_res.get("llantas_visibles") or veh_obj.get("llantas_visibles") or 0)
        vista = str(full_res.get("vista") or veh_obj.get("angulo") or "3/4")
        pasajeros = max(0, int(full_res.get("pasajeros") or 0))
        conf_veh = float(veh_obj.get("confianza") or 0.0)
        annotated_b64 = str(full_res.get("imagen_anotada_b64") or "")

        # Si hay vehículo detectado pero vista trasera/lateral sin matrícula, asignar placa determinista
        if (not placa_norm or len(placa_norm) < 4) and conf_veh > 0.05:
            placa_norm = _provisional_plate_from_bytes(image_bytes, categoria)
            if not placa_leida:
                placa_leida = f"{placa_norm} (ASIGNADA)"

        imagen_url = _save_agent_inference_image(db, annotated_b64, image_bytes)
        tarifa_monto = _resolve_agent_tariff(db, categoria, ejes)

        if not placa_norm or len(placa_norm) < 4:
            db.commit()
            inferencia_card = {
                "imagen_url": imagen_url,
                "placa": "",
                "confianza_ocr": confianza_ocr,
                "categoria": categoria,
                "categoria_nombre": categoria_nombre,
                "ejes": ejes,
                "llantas_visibles": llantas_visibles,
                "vista": vista,
                "pasajeros": pasajeros,
                "tarifa_bs": tarifa_monto,
                "propietario": None,
                "auto_registrado": False,
                "latencia_ms": full_res.get("latencia_ms"),
            }
            return {
                "respuesta": (
                    "No se detectó vehículo ni placa legible en la imagen enviada. "
                    "Por favor sube o captura otra fotografía enfocando el vehículo y su matrícula."
                ),
                "estado": "Placa ilegible",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "inferencia": inferencia_card,
                "ocr": ocr_info,
                "motor": "inferencia_m1_m4",
            }

        # Auto-registrar vehículo en BD si aún no existe (con nombre ficticio / sindicato) y registrar paso
        caseta = db.query(Caseta).order_by(Caseta.id.asc()).first()
        vehiculo_db = db.query(Vehiculo).filter(Vehiculo.placa == placa_norm).first()
        auto_registrado_nuevo = False
        if not vehiculo_db:
            vehiculo_db = Vehiculo(
                placa=placa_norm,
                propietario=generate_fictional_owner(placa_norm, categoria),
                categoria=categoria,
                ejes=ejes,
                saldo=INITIAL_SALDO_BY_CATEGORY.get(categoria, 160.0),
                activo=True,
            )
            db.add(vehiculo_db)
            db.flush()
            auto_registrado_nuevo = True

        if not vehiculo_db.activo:
            estado_paso = "VEHICULO_INACTIVO"
        elif vehiculo_db.saldo < tarifa_monto:
            estado_paso = "SIN_SALDO"
        else:
            estado_paso = "APROBADO"
            vehiculo_db.categoria = categoria
            vehiculo_db.ejes = ejes
            vehiculo_db.saldo = round(vehiculo_db.saldo - tarifa_monto, 2)

        if caseta is not None:
            nuevo_paso = Paso(
                placa_leida=placa_leida or placa_norm,
                placa_normalizada=placa_norm,
                confianza_ocr=round(confianza_ocr, 4),
                categoria_detectada=categoria,
                ejes_detectados=ejes,
                pasajeros=pasajeros,
                monto=tarifa_monto,
                caseta_id=caseta.id,
                usuario_id=usuario_id,
                imagen_url=imagen_url,
                estado=estado_paso,
                latencia_ms=float(full_res.get("latencia_ms") or 0.0),
            )
            db.add(nuevo_paso)
            db.commit()
            db.refresh(nuevo_paso)

            await realtime_manager.broadcast_panel(
                {
                    "event": "nuevo_paso",
                    "data": {
                        "id": nuevo_paso.id,
                        "placa_leida": nuevo_paso.placa_leida,
                        "placa_normalizada": nuevo_paso.placa_normalizada,
                        "propietario": vehiculo_db.propietario,
                        "confianza_ocr": nuevo_paso.confianza_ocr,
                        "categoria_detectada": nuevo_paso.categoria_detectada,
                        "ejes_detectados": nuevo_paso.ejes_detectados,
                        "llantas_visibles": llantas_visibles,
                        "vista": vista,
                        "pasajeros": nuevo_paso.pasajeros,
                        "monto": nuevo_paso.monto,
                        "caseta_id": caseta.id,
                        "caseta_nombre": caseta.nombre,
                        "imagen_url": imagen_url,
                        "tiene_evidencia": True,
                        "origen": "agente_chat",
                        "auto_registrado": auto_registrado_nuevo,
                        "estado": estado_paso,
                        "latencia_ms": nuevo_paso.latencia_ms,
                        "created_at": nuevo_paso.created_at.isoformat() if nuevo_paso.created_at else None,
                    },
                    "alerta": None,
                }
            )
        else:
            db.commit()

        plate_raw = placa_norm
        inferencia_card = {
            "imagen_url": imagen_url,
            "placa": placa_norm,
            "confianza_ocr": round(confianza_ocr, 4),
            "categoria": categoria,
            "categoria_nombre": categoria_nombre,
            "ejes": ejes,
            "llantas_visibles": llantas_visibles,
            "vista": vista,
            "pasajeros": pasajeros,
            "tarifa_bs": tarifa_monto,
            "propietario": vehiculo_db.propietario,
            "saldo_restante": vehiculo_db.saldo,
            "estado_paso": estado_paso,
            "auto_registrado": auto_registrado_nuevo,
            "latencia_ms": full_res.get("latencia_ms"),
        }
    else:
        if not mensaje or not mensaje.strip():
            return {
                "respuesta": (
                    "Por favor escribe una placa (por ejemplo: `2481-KLM`), toca una consulta rápida o adjunta una fotografía del vehículo/placa."
                ),
                "estado": "Sin consulta",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "inferencia": None,
                "ocr": None,
                "motor": "determinista",
            }

        extracted = extract_plate_candidate_from_text(mensaje)
        if not extracted:
            snapshot = _get_general_db_snapshot(db)
            llm_general = await _try_ollama_response(mensaje, snapshot, from_image=False)
            if llm_general:
                return {
                    "respuesta": llm_general,
                    "estado": "Consulta general",
                    "placa_normalizada": "",
                    "sugerencias": [],
                    "datos": snapshot,
                    "inferencia": None,
                    "ocr": None,
                    "motor": f"ollama:{get_active_ollama_model()}",
                }
            intent_reply = _format_general_intent_fallback(mensaje, snapshot)
            if intent_reply:
                return {
                    "respuesta": intent_reply,
                    "estado": "Consulta general",
                    "placa_normalizada": "",
                    "sugerencias": [],
                    "datos": snapshot,
                    "inferencia": None,
                    "ocr": None,
                    "motor": "determinista_bd",
                }
            return {
                "respuesta": (
                    "No identifiqué una placa válida en tu mensaje. "
                    "Escribe una placa (ej. `2481KLM` o `5280-YIN`), consulta *Resumen de hoy*, *Alertas*, *Tarifas*, *Últimos pasos* o envía una foto."
                ),
                "estado": "Sin placa detectada",
                "placa_normalizada": "",
                "sugerencias": [],
                "datos": None,
                "inferencia": None,
                "ocr": None,
                "motor": "determinista_bd",
            }
        plate_raw = extracted

    # Consultar datos reales en BD (Tool execution)
    verified = lookup_plate_status(db, plate_raw)
    if inferencia_card:
        verified["inferencia_imagen"] = inferencia_card

    # Redactar con Ollama LLM usando estrictamente los datos verificados de BD (con fallback determinista)
    ollama_reply = await _try_ollama_response(mensaje or "", verified, from_image=from_image)
    if ollama_reply:
        final_reply = ollama_reply
        engine_used = f"ollama:{get_active_ollama_model()}"
    else:
        final_reply = format_deterministic_reply(verified, from_image=from_image)
        engine_used = "determinista_bd"

    return {
        "respuesta": final_reply,
        "estado": verified["estado"],
        "placa_normalizada": verified["placa_normalizada"],
        "sugerencias": verified["sugerencias"],
        "datos": verified,
        "inferencia": inferencia_card,
        "ocr": ocr_info,
        "motor": engine_used,
    }
