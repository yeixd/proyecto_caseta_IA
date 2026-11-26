"""
Aplicación principal FastAPI - Sistema Inteligente de Control de Peaje.

Rutas mínimas obligatorias (Sección 4.5):
  - POST /api/v1/auth/google
  - POST /api/v1/pasos/frame
  - POST /api/v1/placas/consultar
  - POST /api/v1/placas/consultar-imagen
  - GET  /api/v1/placas/{placa}/historial
  - GET  /api/v1/casetas/{id}/resumen
  - POST /api/v1/agente/chat

Además: CRUD protegido por rol (vehiculos, tarifas, casetas, usuarios), historial de pasos,
gestión de alertas, estado de inferencia/Ollama/hardware y WebSockets en tiempo real.
"""

from __future__ import annotations

import base64
import logging
import os
import time
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import (
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from sqlalchemy import func
from sqlalchemy.orm import Session

from app.agent import (
    check_ollama_health,
    run_conversational_agent,
    set_ollama_runtime_config,
    test_ollama_prompt,
)
from app.auth import (
    authenticate_google_user,
    create_access_token,
    decode_access_token,
    get_current_user,
    require_admin,
)
from app.config import get_settings
from app.database import Base, SessionLocal, engine, get_db
from app.inference_client import inference_client
from app.models import Alerta, Caseta, Configuracion, Paso, Tarifa, Usuario, Vehiculo
from app.plate_utils import lookup_plate_status, normalize_plate
from app.realtime import realtime_manager
from app.schemas import (
    AgentChatJsonRequest,
    AlertaOut,
    AuthResponse,
    CasetaCreate,
    CasetaOut,
    GoogleAuthRequest,
    HardwareTestRequest,
    PasoOut,
    PlacaConsultarRequest,
    RuntimeEndpointsUpdate,
    TarifaCreate,
    TarifaOut,
    UserCreate,
    UserOut,
    UserUpdate,
    VehiculoCreate,
    VehiculoOut,
    VehiculoUpdate,
)
from app.seed import seed_database

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("peaje.api")

settings = get_settings()
os.makedirs(settings.evidence_dir, exist_ok=True)

app = FastAPI(
    title="Sistema Inteligente de Control de Peaje — API REST v1",
    description=(
        "API oficial del proyecto IAM-811 (UNIFRANZ). Autenticación exclusiva con Google OAuth 2.0 "
        "y lista blanca en BD. Incluye tiempo real por WebSockets, agente conversacional con Ollama "
        "y señalización de hardware ESP32."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount(
    "/static/evidencias",
    StaticFiles(directory=settings.evidence_dir),
    name="evidencias",
)


@app.on_event("startup")
def on_startup():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
        # Cargar configuración dinámica persistida (INFERENCE_URL / OLLAMA_URL)
        cfg = {row.clave: row.valor for row in db.query(Configuracion).all()}
        if "inference_url" in cfg or "inference_api_key" in cfg:
            inference_client.set_runtime_config(
                url=cfg.get("inference_url"),
                api_key=cfg.get("inference_api_key"),
            )
        if "ollama_url" in cfg or "ollama_model" in cfg:
            set_ollama_runtime_config(
                url=cfg.get("ollama_url"),
                model=cfg.get("ollama_model"),
            )
        logger.info("Base de datos inicializada y seed aplicado.")
    finally:
        db.close()


# ----------------------------------------------------------------------
# Helpers internos
# ----------------------------------------------------------------------
def _save_evidence_image(annotated_b64: str, fallback_raw: bytes) -> str:
    filename = f"paso_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}.jpg"
    dest = os.path.join(settings.evidence_dir, filename)
    raw = fallback_raw
    if annotated_b64:
        try:
            raw = base64.b64decode(annotated_b64)
        except Exception:
            raw = fallback_raw
    with open(dest, "wb") as f:
        f.write(raw)
    return f"{settings.evidence_public_base_url}/{filename}"


def _resolve_tariff(db: Session, categoria: str, ejes: int) -> float:
    exact = (
        db.query(Tarifa)
        .filter(Tarifa.categoria == categoria, Tarifa.ejes == ejes)
        .first()
    )
    if exact:
        return float(exact.monto)
    same_cat = (
        db.query(Tarifa)
        .filter(Tarifa.categoria == categoria)
        .order_by(Tarifa.ejes.desc())
        .all()
    )
    if same_cat:
        # Elegir la tarifa más cercana al número de ejes detectados
        closest = min(same_cat, key=lambda t: abs(t.ejes - ejes))
        return float(closest.monto)
    return 5.0


def _serialize_paso(paso: Paso, caseta_nombre: Optional[str] = None) -> Dict[str, Any]:
    return {
        "id": paso.id,
        "placa_leida": paso.placa_leida,
        "placa_normalizada": paso.placa_normalizada,
        "confianza_ocr": paso.confianza_ocr,
        "categoria_detectada": paso.categoria_detectada,
        "ejes_detectados": paso.ejes_detectados,
        "pasajeros": paso.pasajeros,
        "monto": paso.monto,
        "caseta_id": paso.caseta_id,
        "caseta_nombre": caseta_nombre or (paso.caseta.nombre if paso.caseta else None),
        "usuario_id": paso.usuario_id,
        "imagen_url": paso.imagen_url,
        "estado": paso.estado,
        "latencia_ms": paso.latencia_ms,
        "created_at": paso.created_at.isoformat() if paso.created_at else None,
    }


def _serialize_alerta(alerta: Alerta) -> Dict[str, Any]:
    paso = alerta.paso
    caseta_nombre = paso.caseta.nombre if (paso and paso.caseta) else None
    return {
        "id": alerta.id,
        "paso_id": alerta.paso_id,
        "tipo": alerta.tipo,
        "mensaje": alerta.mensaje,
        "atendida": alerta.atendida,
        "created_at": alerta.created_at.isoformat() if alerta.created_at else None,
        "placa": paso.placa_normalizada if paso else None,
        "caseta_nombre": caseta_nombre,
        "imagen_url": paso.imagen_url if paso else None,
    }


# ----------------------------------------------------------------------
# 0. HEALTHCHECK PÚBLICO (para Render / Docker / monitoreo)
# ----------------------------------------------------------------------
@app.get("/health", tags=["Estado del Sistema"])
def public_health():
    return {"status": "ok"}


# ----------------------------------------------------------------------
# 1. AUTENTICACIÓN GOOGLE OAUTH 2.0 (única ruta pública de /api/v1)
# ----------------------------------------------------------------------
@app.get("/api/v1/auth/config", tags=["Autenticación"])
def auth_public_config():
    """Devuelve el Client ID público de Google y dominio permitido para inicializar el frontend."""
    cid = settings.google_client_id.strip()
    is_placeholder = not cid or cid.startswith("TU_GOOGLE") or "CONFIGURAR" in cid
    return {
        "google_client_id": "" if is_placeholder else cid,
        "allowed_email_domain": settings.allowed_email_domain,
    }


@app.post("/api/v1/auth/google", response_model=AuthResponse, tags=["Autenticación"])
def auth_google(payload: GoogleAuthRequest, db: Session = Depends(get_db)):
    """Valida el token de Google (ID token o Access token), verifica la lista blanca en BD y emite JWT."""
    user = authenticate_google_user(
        db,
        id_token_str=payload.id_token,
        access_token_str=payload.access_token,
    )
    token = create_access_token(user)
    return AuthResponse(access_token=token, usuario=UserOut.model_validate(user))


@app.get("/api/v1/auth/me", response_model=UserOut, tags=["Autenticación"])
def auth_me(current_user: Usuario = Depends(get_current_user)):
    return current_user


# ----------------------------------------------------------------------
# 2. REGISTRO DE PASO DESDE APP MÓVIL (FRAME -> INFERENCIA -> BD -> TIEMPO REAL)
# ----------------------------------------------------------------------
@app.post("/api/v1/pasos/frame", tags=["Pasos y Caseta"])
async def registrar_paso_frame(
    file: UploadFile = File(..., description="Frame JPEG/PNG capturado en Modo Caseta"),
    caseta_id: int = Form(default=1, description="ID de la caseta que captura el frame"),
    current_user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Recibe un frame desde la app móvil, notifica al hardware (INFERENCIA), llama al endpoint único
    INFERENCE_URL (M1+OCR, M2, M3, M4 pasajeros), calcula tarifa, actualiza saldo, registra paso,
    genera alerta si corresponde y emite el resultado en tiempo real al panel web y al ESP32.
    """
    t0 = time.perf_counter()
    caseta = db.query(Caseta).filter(Caseta.id == caseta_id).first()
    if not caseta:
        # Si el ID enviado (ej. default 1) no coincide con el autoincremento en PostgreSQL, usar la primera caseta activa
        caseta = db.query(Caseta).order_by(Caseta.id.asc()).first()
    if not caseta:
        raise HTTPException(status_code=404, detail="No existen casetas registradas en la base de datos.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="La imagen del frame está vacía.")

    # Notificar inicio de inferencia al dispositivo de hardware
    await realtime_manager.broadcast_hardware(
        evento="INFERENCIA",
        detalle=f"Procesando frame en {caseta.nombre}",
        nivel="info",
    )

    infer_data = await inference_client.infer_full(
        image_bytes,
        filename=file.filename or "frame.jpg",
    )

    placa_obj = infer_data.get("placa", {})
    vehiculo_obj = infer_data.get("vehiculo", {})

    placa_leida = str(placa_obj.get("texto_raw") or placa_obj.get("texto") or "")
    placa_norm = normalize_plate(str(placa_obj.get("texto") or placa_leida))
    confianza_ocr = float(placa_obj.get("confianza") or 0.0)
    categoria = str(vehiculo_obj.get("categoria") or "auto").lower()
    ejes = max(1, int(infer_data.get("ejes") or 1))
    pasajeros = max(0, int(infer_data.get("pasajeros") or 0))
    annotated_b64 = str(infer_data.get("imagen_anotada_b64") or "")

    imagen_url = _save_evidence_image(annotated_b64, image_bytes)
    tarifa_monto = _resolve_tariff(db, categoria, ejes)

    alerta_creada: Optional[Alerta] = None
    monto_cobrado = tarifa_monto
    hw_signal = "APROBADO"
    hw_level = "success"

    if not placa_norm or len(placa_norm) < 4:
        estado_paso = "PLACA_ILEGIBLE"
        monto_cobrado = 0.0
        hw_signal = "ALERTA_NO_REGISTRADO"
        hw_level = "warning"
        vehiculo_db = None
    else:
        vehiculo_db = db.query(Vehiculo).filter(Vehiculo.placa == placa_norm).first()
        if not vehiculo_db:
            estado_paso = "NO_REGISTRADO"
            hw_signal = "ALERTA_NO_REGISTRADO"
            hw_level = "danger"
        elif not vehiculo_db.activo:
            estado_paso = "VEHICULO_INACTIVO"
            monto_cobrado = 0.0
            hw_signal = "ALERTA_INACTIVO"
            hw_level = "danger"
        elif vehiculo_db.saldo < tarifa_monto:
            estado_paso = "SIN_SALDO"
            hw_signal = "SIN_SALDO"
            hw_level = "warning"
        else:
            estado_paso = "APROBADO"
            vehiculo_db.saldo = round(vehiculo_db.saldo - tarifa_monto, 2)
            hw_signal = "APROBADO"
            hw_level = "success"

    total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

    paso = Paso(
        placa_leida=placa_leida or "ILEGIBLE",
        placa_normalizada=placa_norm,
        confianza_ocr=round(confianza_ocr, 4),
        categoria_detectada=categoria,
        ejes_detectados=ejes,
        pasajeros=pasajeros,
        monto=monto_cobrado,
        caseta_id=caseta.id,
        usuario_id=current_user.id,
        imagen_url=imagen_url,
        estado=estado_paso,
        latencia_ms=total_latency_ms,
    )
    db.add(paso)
    db.flush()

    if estado_paso == "NO_REGISTRADO":
        alerta_creada = Alerta(
            paso_id=paso.id,
            tipo="PLACA_NO_REGISTRADA",
            mensaje=f"Placa {placa_norm} detectada en {caseta.nombre} NO está registrada (posible evasión).",
            atendida=False,
        )
        db.add(alerta_creada)
    elif estado_paso == "SIN_SALDO":
        alerta_creada = Alerta(
            paso_id=paso.id,
            tipo="SIN_SALDO",
            mensaje=(
                f"Placa {placa_norm} sin saldo suficiente en {caseta.nombre} "
                f"(Saldo: Bs. {vehiculo_db.saldo:.2f}, Tarifa: Bs. {tarifa_monto:.2f})."
            ),
            atendida=False,
        )
        db.add(alerta_creada)
    elif estado_paso == "VEHICULO_INACTIVO":
        alerta_creada = Alerta(
            paso_id=paso.id,
            tipo="VEHICULO_INACTIVO",
            mensaje=f"Vehículo con placa {placa_norm} está inactivo/dado de baja.",
            atendida=False,
        )
        db.add(alerta_creada)

    db.commit()
    db.refresh(paso)
    if alerta_creada:
        db.refresh(alerta_creada)

    paso_payload = _serialize_paso(paso, caseta_nombre=caseta.nombre)
    alerta_payload = _serialize_alerta(alerta_creada) if alerta_creada else None

    # Emitir paso y alerta en tiempo real al panel web
    await realtime_manager.broadcast_panel(
        {
            "event": "nuevo_paso",
            "data": paso_payload,
            "alerta": alerta_payload,
        }
    )

    # Emitir señal al hardware ESP32
    await realtime_manager.broadcast_hardware(
        evento=hw_signal,
        placa=placa_norm or "ILEGIBLE",
        detalle=f"{estado_paso} | {categoria} ({ejes} ejes) | Bs. {monto_cobrado:.2f}",
        nivel=hw_level,
    )

    return {
        "paso": paso_payload,
        "alerta": alerta_payload,
        "saldo_restante": vehiculo_db.saldo if vehiculo_db else None,
        "inferencia": {
            "placa": placa_obj,
            "vehiculo": vehiculo_obj,
            "ejes": ejes,
            "keypoints": infer_data.get("keypoints", []),
            "pasajeros": pasajeros,
            "latencia_inferencia_ms": infer_data.get("latencia_ms"),
            "latencia_total_ms": total_latency_ms,
        },
    }


# ----------------------------------------------------------------------
# 3. CONSULTAS DE PLACAS E HISTORIAL (Sección 4.5)
# ----------------------------------------------------------------------
@app.post("/api/v1/placas/consultar", tags=["Placas"])
def consultar_placa_texto(
    payload: PlacaConsultarRequest,
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Consulta una placa por texto, la normaliza y devuelve su estado oficial y sugerencias OCR."""
    return lookup_plate_status(db, payload.placa)


@app.post("/api/v1/placas/consultar-imagen", tags=["Placas"])
async def consultar_placa_imagen(
    file: UploadFile = File(..., description="Fotografía de la placa"),
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Consulta una placa a partir de una imagen usando M1 + OCR en el endpoint de inferencia."""
    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Imagen vacía.")

    await realtime_manager.broadcast_hardware(
        evento="IDENTIFICANDO",
        detalle="Consulta de placa por imagen",
        nivel="info",
    )

    infer_res = await inference_client.infer_plate(image_bytes, filename=file.filename or "placa.jpg")
    placa_obj = infer_res.get("placa", {})
    texto = str(placa_obj.get("texto") or "")

    if not texto or not placa_obj.get("legible", len(texto) >= 4):
        return {
            "legible": False,
            "mensaje": "No se detectó una placa legible en la imagen. Por favor intenta con otra foto.",
            "ocr": placa_obj,
            "estado": "Placa ilegible",
        }

    status_data = lookup_plate_status(db, texto)
    status_data["legible"] = True
    status_data["ocr"] = placa_obj
    return status_data


@app.get("/api/v1/placas/{placa}/historial", tags=["Placas"])
def historial_placa(
    placa: str,
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Historial completo de pasos de una placa normalizada."""
    norm = normalize_plate(placa)
    pasos = (
        db.query(Paso)
        .filter(Paso.placa_normalizada == norm)
        .order_by(Paso.created_at.desc())
        .all()
    )
    return {
        "placa_normalizada": norm,
        "total": len(pasos),
        "pasos": [_serialize_paso(p) for p in pasos],
    }


# ----------------------------------------------------------------------
# 4. RESUMEN DE CASETA DEL DÍA (Sección 4.5)
# ----------------------------------------------------------------------
@app.get("/api/v1/casetas/{caseta_id}/resumen", tags=["Casetas"])
def resumen_caseta(
    caseta_id: int,
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Conteo de pasos, recaudación, pasajeros y alertas del día para una caseta."""
    caseta = db.query(Caseta).filter(Caseta.id == caseta_id).first()
    if not caseta:
        raise HTTPException(status_code=404, detail="Caseta no encontrada.")

    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pasos_hoy = (
        db.query(Paso)
        .filter(Paso.caseta_id == caseta_id, Paso.created_at >= start_of_day)
        .all()
    )

    total_pasos = len(pasos_hoy)
    recaudacion = round(sum(p.monto for p in pasos_hoy if p.estado == "APROBADO"), 2)
    pasajeros_total = sum(p.pasajeros for p in pasos_hoy)
    por_categoria: Dict[str, int] = {}
    for p in pasos_hoy:
        por_categoria[p.categoria_detectada] = por_categoria.get(p.categoria_detectada, 0) + 1

    alertas_hoy = (
        db.query(Alerta)
        .join(Paso, Alerta.paso_id == Paso.id)
        .filter(Paso.caseta_id == caseta_id, Alerta.created_at >= start_of_day)
        .count()
    )

    return {
        "caseta": {"id": caseta.id, "nombre": caseta.nombre, "ubicacion": caseta.ubicacion},
        "fecha_utc": start_of_day.date().isoformat(),
        "total_pasos": total_pasos,
        "recaudacion_bs": recaudacion,
        "pasajeros_total": pasajeros_total,
        "alertas_hoy": alertas_hoy,
        "desglose_categorias": por_categoria,
    }


@app.get("/api/v1/resumen/global", tags=["Panel"])
def resumen_global(
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Métricas agregadas del día para el panel en tiempo real."""
    start_of_day = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    pasos_hoy = db.query(Paso).filter(Paso.created_at >= start_of_day).all()
    alertas_pendientes = db.query(Alerta).filter(Alerta.atendida == False).count()  # noqa: E712
    total_vehiculos = db.query(Vehiculo).count()

    return {
        "pasos_hoy": len(pasos_hoy),
        "recaudacion_hoy": round(sum(p.monto for p in pasos_hoy if p.estado == "APROBADO"), 2),
        "pasajeros_hoy": sum(p.pasajeros for p in pasos_hoy),
        "alertas_pendientes": alertas_pendientes,
        "total_vehiculos": total_vehiculos,
        "hardware_conectado": len(realtime_manager.hardware_clients),
    }


# ----------------------------------------------------------------------
# 5. AGENTE CONVERSACIONAL (Sección 4.5 y 4.8)
# ----------------------------------------------------------------------
@app.post("/api/v1/agente/chat", tags=["Agente Conversacional"])
async def agente_chat_multipart(
    mensaje: Optional[str] = Form(default=None),
    file: Optional[UploadFile] = File(default=None),
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """
    Envía un mensaje de texto y/o una imagen de placa al agente conversacional.
    Soporta multipart/form-data (ideal para subir imagen desde web o app móvil).
    """
    img_bytes = await file.read() if file is not None else None
    fname = file.filename if file is not None else "placa.jpg"
    return await run_conversational_agent(
        db=db,
        mensaje=mensaje,
        image_bytes=img_bytes,
        filename=fname or "placa.jpg",
    )


@app.post("/api/v1/agente/chat/json", tags=["Agente Conversacional"])
async def agente_chat_json(
    payload: AgentChatJsonRequest,
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Alternativa JSON (texto o imagen base64) para el agente conversacional."""
    img_bytes: Optional[bytes] = None
    if payload.image_b64:
        raw_b64 = payload.image_b64.split(",", 1)[1] if "," in payload.image_b64 else payload.image_b64
        try:
            img_bytes = base64.b64decode(raw_b64)
        except Exception:
            raise HTTPException(status_code=400, detail="image_b64 inválido")
    return await run_conversational_agent(db=db, mensaje=payload.mensaje, image_bytes=img_bytes)


# ----------------------------------------------------------------------
# 6. CRUD VEHÍCULOS, TARIFAS, CASETAS, USUARIOS (Admin) + PASOS Y ALERTAS
# ----------------------------------------------------------------------
@app.get("/api/v1/vehiculos", response_model=List[VehiculoOut], tags=["Vehículos"])
def listar_vehiculos(
    q: Optional[str] = Query(default=None, description="Filtro por placa o propietario"),
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Vehiculo)
    if q:
        like = f"%{q.strip()}%"
        query = query.filter((Vehiculo.placa.ilike(like)) | (Vehiculo.propietario.ilike(like)))
    return query.order_by(Vehiculo.id.desc()).all()


@app.post("/api/v1/vehiculos", response_model=VehiculoOut, status_code=201, tags=["Vehículos"])
def crear_vehiculo(
    payload: VehiculoCreate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    placa_norm = normalize_plate(payload.placa)
    if not placa_norm:
        raise HTTPException(status_code=400, detail="Placa inválida.")
    if db.query(Vehiculo).filter(Vehiculo.placa == placa_norm).first():
        raise HTTPException(status_code=409, detail=f"La placa {placa_norm} ya está registrada.")
    v = Vehiculo(
        placa=placa_norm,
        propietario=payload.propietario.strip(),
        categoria=payload.categoria,
        ejes=payload.ejes,
        saldo=payload.saldo,
        activo=payload.activo,
    )
    db.add(v)
    db.commit()
    db.refresh(v)
    return v


@app.put("/api/v1/vehiculos/{vehiculo_id}", response_model=VehiculoOut, tags=["Vehículos"])
def actualizar_vehiculo(
    vehiculo_id: int,
    payload: VehiculoUpdate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    v = db.query(Vehiculo).filter(Vehiculo.id == vehiculo_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehículo no encontrado.")
    for field, val in payload.model_dump(exclude_unset=True).items():
        setattr(v, field, val)
    db.commit()
    db.refresh(v)
    return v


@app.delete("/api/v1/vehiculos/{vehiculo_id}", status_code=204, tags=["Vehículos"])
def eliminar_vehiculo(
    vehiculo_id: int,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    v = db.query(Vehiculo).filter(Vehiculo.id == vehiculo_id).first()
    if not v:
        raise HTTPException(status_code=404, detail="Vehículo no encontrado.")
    db.delete(v)
    db.commit()


@app.get("/api/v1/casetas", response_model=List[CasetaOut], tags=["Casetas"])
def listar_casetas(_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Caseta).order_by(Caseta.id.asc()).all()


@app.post("/api/v1/casetas", response_model=CasetaOut, status_code=201, tags=["Casetas"])
def crear_caseta(
    payload: CasetaCreate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    if db.query(Caseta).filter(Caseta.nombre == payload.nombre.strip()).first():
        raise HTTPException(status_code=409, detail="Ya existe una caseta con ese nombre.")
    c = Caseta(nombre=payload.nombre.strip(), ubicacion=payload.ubicacion.strip())
    db.add(c)
    db.commit()
    db.refresh(c)
    return c


@app.delete("/api/v1/casetas/{caseta_id}", status_code=204, tags=["Casetas"])
def eliminar_caseta(
    caseta_id: int,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    c = db.query(Caseta).filter(Caseta.id == caseta_id).first()
    if not c:
        raise HTTPException(status_code=404, detail="Caseta no encontrada.")
    db.delete(c)
    db.commit()


@app.get("/api/v1/tarifas", response_model=List[TarifaOut], tags=["Tarifas"])
def listar_tarifas(_user: Usuario = Depends(get_current_user), db: Session = Depends(get_db)):
    return db.query(Tarifa).order_by(Tarifa.categoria.asc(), Tarifa.ejes.asc()).all()


@app.post("/api/v1/tarifas", response_model=TarifaOut, status_code=201, tags=["Tarifas"])
def crear_o_actualizar_tarifa(
    payload: TarifaCreate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    existing = (
        db.query(Tarifa)
        .filter(Tarifa.categoria == payload.categoria, Tarifa.ejes == payload.ejes)
        .first()
    )
    if existing:
        existing.monto = payload.monto
        db.commit()
        db.refresh(existing)
        return existing
    t = Tarifa(categoria=payload.categoria, ejes=payload.ejes, monto=payload.monto)
    db.add(t)
    db.commit()
    db.refresh(t)
    return t


@app.get("/api/v1/usuarios", response_model=List[UserOut], tags=["Usuarios (Lista Blanca)"])
def listar_usuarios(_admin: Usuario = Depends(require_admin), db: Session = Depends(get_db)):
    return db.query(Usuario).order_by(Usuario.id.asc()).all()


@app.post("/api/v1/usuarios", response_model=UserOut, status_code=201, tags=["Usuarios (Lista Blanca)"])
def agregar_usuario_whitelist(
    payload: UserCreate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    email = payload.email_google.strip().lower()
    if settings.allowed_email_domain and not email.endswith(f"@{settings.allowed_email_domain}"):
        raise HTTPException(
            status_code=400,
            detail=f"Solo se permiten correos institucionales terminados en @{settings.allowed_email_domain}.",
        )
    if db.query(Usuario).filter(Usuario.email_google == email).first():
        raise HTTPException(status_code=409, detail="Ese correo ya está en la lista blanca.")
    u = Usuario(email_google=email, nombre=payload.nombre.strip(), rol=payload.rol, activo=payload.activo)
    db.add(u)
    db.commit()
    db.refresh(u)
    return u


@app.put("/api/v1/usuarios/{usuario_id}", response_model=UserOut, tags=["Usuarios (Lista Blanca)"])
def actualizar_usuario_whitelist(
    usuario_id: int,
    payload: UserUpdate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    u = db.query(Usuario).filter(Usuario.id == usuario_id).first()
    if not u:
        raise HTTPException(status_code=404, detail="Usuario no encontrado.")
    for field, val in payload.model_dump(exclude_unset=True).items():
        setattr(u, field, val)
    db.commit()
    db.refresh(u)
    return u


@app.get("/api/v1/pasos", tags=["Pasos y Caseta"])
def listar_pasos(
    limit: int = Query(default=50, ge=1, le=200),
    placa: Optional[str] = Query(default=None),
    estado: Optional[str] = Query(default=None),
    caseta_id: Optional[int] = Query(default=None),
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Paso)
    if placa:
        query = query.filter(Paso.placa_normalizada.ilike(f"%{normalize_plate(placa)}%"))
    if estado:
        query = query.filter(Paso.estado == estado)
    if caseta_id:
        query = query.filter(Paso.caseta_id == caseta_id)
    pasos = query.order_by(Paso.created_at.desc()).limit(limit).all()
    return [_serialize_paso(p) for p in pasos]


@app.get("/api/v1/alertas", tags=["Alertas"])
def listar_alertas(
    solo_pendientes: bool = Query(default=False),
    limit: int = Query(default=50, ge=1, le=200),
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(Alerta)
    if solo_pendientes:
        query = query.filter(Alerta.atendida == False)  # noqa: E712
    alertas = query.order_by(Alerta.created_at.desc()).limit(limit).all()
    return [_serialize_alerta(a) for a in alertas]


@app.patch("/api/v1/alertas/{alerta_id}/atender", tags=["Alertas"])
async def atender_alerta(
    alerta_id: int,
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    alerta = db.query(Alerta).filter(Alerta.id == alerta_id).first()
    if not alerta:
        raise HTTPException(status_code=404, detail="Alerta no encontrada.")
    alerta.atendida = True
    db.commit()
    db.refresh(alerta)
    payload = _serialize_alerta(alerta)
    await realtime_manager.broadcast_panel({"event": "alerta_atendida", "data": payload})
    return payload


# ----------------------------------------------------------------------
# 7. ESTADO DEL SISTEMA (INFERENCIA, OLLAMA, BD, HARDWARE)
# ----------------------------------------------------------------------
@app.get("/api/v1/estado", tags=["Estado del Sistema"])
async def estado_sistema(
    _user: Usuario = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Verifica conectividad en vivo hacia INFERENCE_URL, Ollama, Base de Datos y Hardware ESP32."""
    inference_status = await inference_client.check_health()
    ollama_status = await check_ollama_health()
    db_ok = True
    try:
        db.execute(func.count(Usuario.id)).scalar()
    except Exception:
        db_ok = False

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "database": {"online": db_ok},
        "inference": inference_status,
        "ollama": ollama_status,
        "realtime": {
            "paneles_conectados": len(realtime_manager.panel_clients),
            "hardware_conectados": len(realtime_manager.hardware_clients),
            "ultimo_evento_hardware": realtime_manager.last_hardware_event,
        },
    }


@app.put("/api/v1/estado/config", tags=["Estado del Sistema"])
async def actualizar_endpoints_runtime(
    payload: RuntimeEndpointsUpdate,
    _admin: Usuario = Depends(require_admin),
    db: Session = Depends(get_db),
):
    """
    Permite a un administrador cambiar INFERENCE_URL (RunPod <-> Colab) y OLLAMA_URL en vivo
    sin necesidad de redesplegar el servidor web.
    """
    updates = {}
    if payload.inference_url is not None:
        updates["inference_url"] = payload.inference_url.strip().rstrip("/")
    if payload.inference_api_key is not None:
        updates["inference_api_key"] = payload.inference_api_key.strip()
    if payload.ollama_url is not None:
        updates["ollama_url"] = payload.ollama_url.strip().rstrip("/")
    if payload.ollama_model is not None and payload.ollama_model.strip():
        updates["ollama_model"] = payload.ollama_model.strip()

    for clave, valor in updates.items():
        row = db.query(Configuracion).filter(Configuracion.clave == clave).first()
        if row:
            row.valor = valor
        else:
            db.add(Configuracion(clave=clave, valor=valor))
    db.commit()

    inference_client.set_runtime_config(
        url=updates.get("inference_url"),
        api_key=updates.get("inference_api_key"),
    )
    set_ollama_runtime_config(
        url=updates.get("ollama_url"),
        model=updates.get("ollama_model"),
    )

    return {
        "ok": True,
        "inference": await inference_client.check_health(),
        "ollama": await check_ollama_health(),
    }


@app.post("/api/v1/estado/test-ollama", tags=["Estado del Sistema"])
async def probar_ollama_llm(_user: Usuario = Depends(get_current_user)):
    """Envía un prompt real a Ollama LLM para verificar su respuesta en vivo."""
    return await test_ollama_prompt()


@app.post("/api/v1/hardware/test", tags=["Hardware ESP32"])
async def probar_hardware(
    payload: HardwareTestRequest,
    _user: Usuario = Depends(get_current_user),
):
    """Envía una señal de prueba al dispositivo físico ESP32 conectado."""
    ev = await realtime_manager.broadcast_hardware(
        evento=payload.tipo,
        placa=payload.placa,
        detalle=payload.detalle,
        nivel="warning" if "ALERTA" in payload.tipo or payload.tipo == "SIN_SALDO" else "info",
    )
    return {
        "ok": True,
        "dispositivos_notificados": len(realtime_manager.hardware_clients),
        "evento": ev,
    }


@app.get("/api/v1/hardware/ultimo-evento", tags=["Hardware ESP32"])
def hardware_ultimo_evento(x_hardware_key: Optional[str] = Header(default=None)):
    """Endpoint de respaldo HTTP polling para ESP32 en caso de redes con bloqueo WebSocket."""
    if x_hardware_key != settings.hardware_api_key:
        raise HTTPException(status_code=401, detail="X-Hardware-Key inválida.")
    return {"evento": realtime_manager.last_hardware_event}


# ----------------------------------------------------------------------
# 8. WEBSOCKETS TIEMPO REAL (/ws/panel y /ws/hardware)
# ----------------------------------------------------------------------
@app.websocket("/ws/panel")
async def ws_panel(websocket: WebSocket, token: Optional[str] = Query(default=None)):
    if not token:
        await websocket.close(code=4401)
        return
    try:
        decode_access_token(token)
    except HTTPException:
        await websocket.close(code=4401)
        return

    await realtime_manager.connect_panel(websocket)
    try:
        while True:
            # Mantener viva la conexión (el cliente puede enviar "ping")
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_json({"event": "pong"})
    except WebSocketDisconnect:
        realtime_manager.disconnect_panel(websocket)


@app.websocket("/ws/hardware")
async def ws_hardware(websocket: WebSocket, key: Optional[str] = Query(default=None)):
    if key != settings.hardware_api_key:
        await websocket.close(code=4401)
        return

    await realtime_manager.connect_hardware(websocket)
    try:
        while True:
            msg = await websocket.receive_text()
            if msg == "ping":
                await websocket.send_json({"event": "pong"})
    except WebSocketDisconnect:
        await realtime_manager.disconnect_hardware(websocket)


# ----------------------------------------------------------------------
# 9. SERVIR FRONTEND REACT COMPILADO (para despliegue unificado en Render/Railway)
# ----------------------------------------------------------------------
_frontend_dist = os.getenv("FRONTEND_DIST_DIR", "/app/frontend_dist")
if os.path.isdir(_frontend_dist) and os.path.isfile(os.path.join(_frontend_dist, "index.html")):
    _assets_dir = os.path.join(_frontend_dist, "assets")
    if os.path.isdir(_assets_dir):
        app.mount("/assets", StaticFiles(directory=_assets_dir), name="frontend_assets")

    @app.get("/{full_path:path}", include_in_schema=False)
    async def serve_spa(full_path: str):
        if full_path.startswith(("api/", "ws/", "static/", "docs", "redoc", "openapi.json", "health")):
            raise HTTPException(status_code=404, detail="Ruta no encontrada")
        candidate = os.path.join(_frontend_dist, full_path)
        if full_path and os.path.isfile(candidate):
            return FileResponse(candidate)
        return FileResponse(os.path.join(_frontend_dist, "index.html"))

