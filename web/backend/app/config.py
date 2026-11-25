"""Configuración centralizada desde variables de entorno (.env)."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import List

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# Cargar .env desde la raíz del repositorio (y/o el directorio actual)
_ROOT_ENV = Path(__file__).resolve().parents[3] / ".env"
if _ROOT_ENV.is_file():
    load_dotenv(_ROOT_ENV, override=True)
else:
    load_dotenv(override=True)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(extra="ignore")
    app_env: str = os.getenv("APP_ENV", "production")
    app_url: str = os.getenv("APP_URL", "https://peaje.tudominio.com")
    frontend_url: str = os.getenv("FRONTEND_URL", "https://peaje.tudominio.com")
    cors_origins: str = os.getenv("CORS_ORIGINS", "*")

    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./peaje.db")

    # Google OAuth 2.0 (Web, Android, iOS)
    google_client_id: str = os.getenv("GOOGLE_CLIENT_ID", "")
    google_android_client_id: str = os.getenv("GOOGLE_ANDROID_CLIENT_ID", "")
    google_ios_client_id: str = os.getenv("GOOGLE_IOS_CLIENT_ID", "")

    # JWT interno emitido tras validar Google OAuth
    jwt_secret_key: str = os.getenv("JWT_SECRET_KEY", "cambiar-en-produccion-con-secreto-largo")
    jwt_algorithm: str = os.getenv("JWT_ALGORITHM", "HS256")
    jwt_expire_minutes: int = int(os.getenv("JWT_EXPIRE_MINUTES", "720"))

    # Dominio institucional permitido y correos adicionales en lista blanca
    # (el docente doc.alvaro.pinto.ma@unifranz.edu.bo siempre se incluye en seed.py)
    allowed_email_domain: str = os.getenv("ALLOWED_EMAIL_DOMAIN", "unifranz.edu.bo").strip().lower().lstrip("@")
    extra_admin_emails: str = os.getenv("EXTRA_ADMIN_EMAILS", "")
    extra_operator_emails: str = os.getenv("EXTRA_OPERATOR_EMAILS", "")

    # Endpoint único de inferencia (RunPod o Colab)
    inference_url: str = os.getenv("INFERENCE_URL", "").rstrip("/")
    inference_api_key: str = os.getenv("INFERENCE_API_KEY", "")
    inference_timeout_seconds: float = float(os.getenv("INFERENCE_TIMEOUT_SECONDS", "20"))

    # Agente conversacional (Ollama en RunPod)
    ollama_url: str = os.getenv("OLLAMA_URL", "").rstrip("/")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    ollama_timeout_seconds: float = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "25"))

    # Imágenes de evidencia
    evidence_dir: str = os.getenv("EVIDENCE_DIR", "uploads/evidencias")
    evidence_public_base_url: str = os.getenv("EVIDENCE_PUBLIC_BASE_URL", "/static/evidencias").rstrip("/")

    # Dispositivo Hardware ESP32
    hardware_api_key: str = os.getenv("HARDWARE_API_KEY", "esp32-dev-key-cambiar")

    @property
    def allowed_google_client_ids(self) -> List[str]:
        ids = [
            self.google_client_id.strip(),
            self.google_android_client_id.strip(),
            self.google_ios_client_id.strip(),
        ]
        return [cid for cid in ids if cid]

    @property
    def cors_origin_list(self) -> List[str]:
        raw = self.cors_origins.strip()
        if raw == "*" or not raw:
            return ["*"]
        return [item.strip() for item in raw.split(",") if item.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
