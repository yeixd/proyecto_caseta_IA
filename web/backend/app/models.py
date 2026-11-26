"""
Modelos de base de datos (Sección 4.6 + campo pasajeros de Sección 10).
Tablas:
  - usuarios(id, email_google, nombre, rol, activo)
  - vehiculos(id, placa, propietario, categoria, ejes, saldo, activo)
  - casetas(id, nombre, ubicacion)
  - tarifas(id, categoria, ejes, monto)
  - pasos(id, placa_leida, placa_normalizada, confianza_ocr, categoria_detectada,
          ejes_detectados, pasajeros, monto, caseta_id, usuario_id, imagen_url,
          estado, latencia_ms, created_at)
  - alertas(id, paso_id, tipo, mensaje, atendida, created_at)
"""

from datetime import datetime, timezone

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from app.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Usuario(Base):
    __tablename__ = "usuarios"

    id = Column(Integer, primary_key=True, index=True)
    email_google = Column(String(255), unique=True, index=True, nullable=False)
    nombre = Column(String(200), nullable=False)
    rol = Column(String(30), nullable=False, default="operador")  # administrador | operador
    activo = Column(Boolean, nullable=False, default=True)

    pasos = relationship("Paso", back_populates="usuario")


class Vehiculo(Base):
    __tablename__ = "vehiculos"

    id = Column(Integer, primary_key=True, index=True)
    placa = Column(String(20), unique=True, index=True, nullable=False)
    propietario = Column(String(200), nullable=False)
    categoria = Column(String(30), nullable=False)  # moto, auto, minibus, camion, bus
    ejes = Column(Integer, nullable=False, default=2)
    saldo = Column(Float, nullable=False, default=0.0)
    activo = Column(Boolean, nullable=False, default=True)


class Caseta(Base):
    __tablename__ = "casetas"

    id = Column(Integer, primary_key=True, index=True)
    nombre = Column(String(150), nullable=False, unique=True)
    ubicacion = Column(String(255), nullable=False)

    pasos = relationship("Paso", back_populates="caseta")


class Tarifa(Base):
    __tablename__ = "tarifas"
    __table_args__ = (UniqueConstraint("categoria", "ejes", name="uq_tarifa_categoria_ejes"),)

    id = Column(Integer, primary_key=True, index=True)
    categoria = Column(String(30), nullable=False, index=True)
    ejes = Column(Integer, nullable=False)
    monto = Column(Float, nullable=False)


class Paso(Base):
    __tablename__ = "pasos"

    id = Column(Integer, primary_key=True, index=True)
    placa_leida = Column(String(40), nullable=False, default="")
    placa_normalizada = Column(String(20), index=True, nullable=False, default="")
    confianza_ocr = Column(Float, nullable=False, default=0.0)
    categoria_detectada = Column(String(30), nullable=False, default="auto")
    ejes_detectados = Column(Integer, nullable=False, default=2)
    # Bonificación +5 (Sección 10): conteo de pasajeros por segmentación
    pasajeros = Column(Integer, nullable=False, default=0)
    monto = Column(Float, nullable=False, default=0.0)
    caseta_id = Column(Integer, ForeignKey("casetas.id"), nullable=False, index=True)
    usuario_id = Column(Integer, ForeignKey("usuarios.id"), nullable=True, index=True)
    imagen_url = Column(String(500), nullable=True)
    # APROBADO | SIN_SALDO | NO_REGISTRADO | VEHICULO_INACTIVO | PLACA_ILEGIBLE
    estado = Column(String(40), nullable=False, index=True)
    latencia_ms = Column(Float, nullable=False, default=0.0)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    caseta = relationship("Caseta", back_populates="pasos")
    usuario = relationship("Usuario", back_populates="pasos")
    alertas = relationship("Alerta", back_populates="paso", cascade="all, delete-orphan")


class Alerta(Base):
    __tablename__ = "alertas"

    id = Column(Integer, primary_key=True, index=True)
    paso_id = Column(Integer, ForeignKey("pasos.id"), nullable=False, index=True)
    # PLACA_NO_REGISTRADA | SIN_SALDO | VEHICULO_INACTIVO
    tipo = Column(String(50), nullable=False, index=True)
    mensaje = Column(String(500), nullable=False, default="")
    atendida = Column(Boolean, nullable=False, default=False, index=True)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow, index=True)

    paso = relationship("Paso", back_populates="alertas")


class Configuracion(Base):
    """Configuración dinámica en BD (permite actualizar INFERENCE_URL / OLLAMA_URL sin redesplegar)."""

    __tablename__ = "configuracion"

    clave = Column(String(80), primary_key=True, index=True)
    valor = Column(String(500), nullable=False, default="")


class EvidenciaImagen(Base):
    """Almacenamiento persistente de imágenes de evidencia en PostgreSQL (sobrevive reinicios de Render)."""

    __tablename__ = "evidencias_imagenes"

    filename = Column(String(200), primary_key=True, index=True)
    content_type = Column(String(80), nullable=False, default="image/jpeg")
    data = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False, default=utcnow)

