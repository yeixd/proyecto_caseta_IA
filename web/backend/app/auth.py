"""
Autenticación exclusiva con Google OAuth 2.0 + Lista blanca en base de datos + JWT interno.
No existen contraseñas locales.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Any, Dict, Optional

import jwt
import requests as http_requests
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from google.auth.transport import requests as google_requests
from google.oauth2 import id_token as google_id_token
from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import get_db
from app.models import Usuario

bearer_scheme = HTTPBearer(auto_error=False)


def verify_google_id_token(token_str: str) -> Dict[str, Any]:
    """Valida el ID token emitido por Google Identity (Web, Android o iOS)."""
    settings = get_settings()
    allowed_audiences = settings.allowed_google_client_ids

    if not token_str or len(token_str) < 20:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de Google inválido o vacío.",
        )

    try:
        request_adapter = google_requests.Request()
        idinfo = google_id_token.verify_oauth2_token(
            token_str,
            request_adapter,
            audience=None,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"No se pudo verificar el token de Google: {exc}",
        )

    issuer = idinfo.get("iss")
    if issuer not in ("accounts.google.com", "https://accounts.google.com"):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Emisor de token Google no válido.",
        )

    aud = idinfo.get("aud")
    if allowed_audiences and aud not in allowed_audiences:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="El Client ID del token de Google no corresponde a esta aplicación.",
        )

    if not idinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La cuenta de Google no tiene el correo verificado.",
        )

    return idinfo


def verify_google_access_token(access_token: str) -> Dict[str, Any]:
    """Valida un Access Token de Google OAuth 2.0 contra los endpoints oficiales de Google."""
    settings = get_settings()
    allowed_audiences = settings.allowed_google_client_ids

    if not access_token or len(access_token) < 20:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Access token de Google inválido o vacío.",
        )

    try:
        tokeninfo_resp = http_requests.get(
            "https://www.googleapis.com/oauth2/v3/tokeninfo",
            params={"access_token": access_token},
            timeout=8,
        )
        if tokeninfo_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No se pudo verificar el token de Google (expirado o inválido).",
            )
        tinfo = tokeninfo_resp.json()
        aud = tinfo.get("aud") or tinfo.get("azp")
        if allowed_audiences and aud not in allowed_audiences:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="El Client ID del token de Google no corresponde a esta aplicación.",
            )

        userinfo_resp = http_requests.get(
            "https://www.googleapis.com/oauth2/v3/userinfo",
            headers={"Authorization": f"Bearer {access_token}"},
            timeout=8,
        )
        if userinfo_resp.status_code != 200:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="No se pudo obtener el perfil de Google.",
            )
        uinfo = userinfo_resp.json()
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Error validando acceso con Google: {exc}",
        )

    if not uinfo.get("email") or not uinfo.get("email_verified", False):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La cuenta de Google no tiene un correo verificado.",
        )

    return uinfo


def create_access_token(user: Usuario) -> str:
    """Emite un JWT firmado del sistema tras validar Google OAuth y lista blanca."""
    settings = get_settings()
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(user.id),
        "email": user.email_google,
        "nombre": user.nombre,
        "rol": user.rol,
        "iat": int(now.timestamp()),
        "exp": int((now + timedelta(minutes=settings.jwt_expire_minutes)).timestamp()),
    }
    return jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)


def decode_access_token(token: str) -> Dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
        )
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="La sesión ha expirado. Inicia sesión nuevamente con Google.",
        )
    except jwt.InvalidTokenError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token de sesión inválido.",
        )


def authenticate_google_user(
    db: Session,
    id_token_str: Optional[str] = None,
    access_token_str: Optional[str] = None,
) -> Usuario:
    """Verifica el token de Google (ID token o Access token) y exige lista blanca activa."""
    if id_token_str:
        idinfo = verify_google_id_token(id_token_str)
    elif access_token_str:
        idinfo = verify_google_access_token(access_token_str)
    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Debes enviar 'id_token' o 'access_token' de Google.",
        )
    email = str(idinfo.get("email", "")).strip().lower()
    google_name = str(idinfo.get("name", "")).strip()

    settings = get_settings()
    if settings.allowed_email_domain and not email.endswith(f"@{settings.allowed_email_domain}"):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Acceso denegado: solo se permiten cuentas institucionales @{settings.allowed_email_domain}. "
                f"La cuenta '{email}' no pertenece al dominio autorizado. Pulsa 'Elegir cuenta de Google' para usar tu correo @{settings.allowed_email_domain}."
            ),
        )

    user = db.query(Usuario).filter(Usuario.email_google == email).first()
    if not user or not user.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Acceso denegado: la cuenta '{email}' no está autorizada en la lista blanca "
                "del sistema. Solicita acceso a un administrador."
            ),
        )

    # Actualizar nombre si cambió o era placeholder
    if google_name and (not user.nombre or user.nombre.lower() == email.split("@")[0].lower()):
        user.nombre = google_name
        db.commit()
        db.refresh(user)

    return user


def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme),
    db: Session = Depends(get_db),
) -> Usuario:
    if credentials is None or not credentials.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Autenticación requerida. Inicia sesión con Google.",
        )

    payload = decode_access_token(credentials.credentials)
    user_id = int(payload.get("sub", "0"))
    user = db.query(Usuario).filter(Usuario.id == user_id).first()
    if not user or not user.activo:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Acceso denegado: usuario deshabilitado o removido de la lista blanca.",
        )
    return user


def require_admin(user: Usuario = Depends(get_current_user)) -> Usuario:
    if user.rol != "administrador":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Permiso denegado: esta acción requiere rol de administrador.",
        )
    return user
