"""Esquemas Pydantic para validación y documentación automática OpenAPI / Swagger."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class GoogleAuthRequest(BaseModel):
    id_token: Optional[str] = Field(
        default=None,
        description="ID Token emitido por Google OAuth 2.0 / Google Sign-In",
    )
    access_token: Optional[str] = Field(
        default=None,
        description="Access Token OAuth 2.0 de Google (usado al elegir cuenta con prompt=select_account)",
    )


class UserOut(BaseModel):
    id: int
    email_google: str
    nombre: str
    rol: str
    activo: bool

    model_config = {"from_attributes": True}


class UserCreate(BaseModel):
    email_google: str
    nombre: str
    rol: str = Field(default="operador", pattern="^(administrador|operador)$")
    activo: bool = True


class UserUpdate(BaseModel):
    nombre: Optional[str] = None
    rol: Optional[str] = Field(default=None, pattern="^(administrador|operador)$")
    activo: Optional[bool] = None


class AuthResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    usuario: UserOut


class VehiculoCreate(BaseModel):
    placa: str = Field(..., examples=["2481-KLM"])
    propietario: str = Field(..., examples=["Carlos Mendoza"])
    categoria: str = Field(..., pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: int = Field(default=2, ge=1, le=12)
    saldo: float = Field(default=0.0, ge=0.0)
    activo: bool = True


class VehiculoUpdate(BaseModel):
    propietario: Optional[str] = None
    categoria: Optional[str] = Field(default=None, pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: Optional[int] = Field(default=None, ge=1, le=12)
    saldo: Optional[float] = Field(default=None, ge=0.0)
    activo: Optional[bool] = None


class VehiculoOut(BaseModel):
    id: int
    placa: str
    propietario: str
    categoria: str
    ejes: int
    saldo: float
    activo: bool

    model_config = {"from_attributes": True}


class CasetaCreate(BaseModel):
    nombre: str
    ubicacion: str


class CasetaOut(BaseModel):
    id: int
    nombre: str
    ubicacion: str

    model_config = {"from_attributes": True}


class TarifaCreate(BaseModel):
    categoria: str = Field(..., pattern="^(moto|auto|minibus|camion|bus)$")
    ejes: int = Field(..., ge=1, le=12)
    monto: float = Field(..., ge=0.0)


class RuntimeEndpointsUpdate(BaseModel):
    inference_url: Optional[str] = Field(default=None, description="URL pública de RunPod o Colab")
    inference_api_key: Optional[str] = Field(default=None)
    ollama_url: Optional[str] = Field(default=None, description="URL pública de Ollama en RunPod")
    ollama_model: Optional[str] = Field(default=None)


class TarifaOut(BaseModel):
    id: int
    categoria: str
    ejes: int
    monto: float

    model_config = {"from_attributes": True}


class PasoOut(BaseModel):
    id: int
    placa_leida: str
    placa_normalizada: str
    confianza_ocr: float
    categoria_detectada: str
    ejes_detectados: int
    pasajeros: int
    monto: float
    caseta_id: int
    caseta_nombre: Optional[str] = None
    usuario_id: Optional[int] = None
    imagen_url: Optional[str] = None
    estado: str
    latencia_ms: float
    created_at: datetime


class AlertaOut(BaseModel):
    id: int
    paso_id: int
    tipo: str
    mensaje: str
    atendida: bool
    created_at: datetime
    placa: Optional[str] = None
    caseta_nombre: Optional[str] = None
    imagen_url: Optional[str] = None


class PlacaConsultarRequest(BaseModel):
    placa: str = Field(..., examples=["2481-KLM"])


class AgentChatJsonRequest(BaseModel):
    mensaje: Optional[str] = Field(default=None, examples=["Consulta el estado de la placa 2481-KLM"])
    image_b64: Optional[str] = Field(default=None, description="Imagen de placa en Base64 opcional")


class HardwareTestRequest(BaseModel):
    tipo: str = Field(
        default="TEST",
        pattern="^(IDENTIFICANDO|INFERENCIA|APROBADO|SIN_SALDO|ALERTA_NO_REGISTRADO|TEST)$",
    )
    placa: str = "2481KLM"
    detalle: str = "Prueba manual desde panel web"
