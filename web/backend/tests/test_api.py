"""
Pruebas automatizadas del backend FastAPI:
  - Seed obligatorio (doc.alvaro.pinto.ma@unifranz.edu.bo como administrador y >= 20 vehículos)
  - Protección de rutas por JWT (todas excepto /api/v1/auth/google)
  - Normalización de placas y sugerencias por confusión OCR (O↔0, I↔1, B↔8, S↔5)
  - Los 4 estados oficiales del agente conversacional (texto e imagen)
  - Flujo de frame -> registro de paso -> alerta automática ante placa no registrada / sin saldo
"""

import os
import sys
from unittest.mock import AsyncMock, patch

os.environ["DATABASE_URL"] = "sqlite:///./test_peaje.db"
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-pytest-only-32bytes"
os.environ["OLLAMA_URL"] = ""  # Forzar motor determinista durante tests

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient  # noqa: E402

from app.auth import create_access_token  # noqa: E402
from app.database import Base, SessionLocal, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.models import Usuario, Vehiculo  # noqa: E402
from app.plate_utils import find_ocr_suggestions, normalize_plate  # noqa: E402
from app.seed import DOCENTE_EMAIL, seed_database  # noqa: E402


def setup_module():
    Base.metadata.drop_all(bind=engine)
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        seed_database(db)
    finally:
        db.close()


def teardown_module():
    if os.path.exists("./test_peaje.db"):
        try:
            os.remove("./test_peaje.db")
        except OSError:
            pass


def _admin_headers():
    db = SessionLocal()
    try:
        docente = db.query(Usuario).filter(Usuario.email_google == DOCENTE_EMAIL).first()
        token = create_access_token(docente)
        return {"Authorization": f"Bearer {token}"}
    finally:
        db.close()


client = TestClient(app)


def test_seed_docente_and_20_vehicles():
    db = SessionLocal()
    try:
        docente = db.query(Usuario).filter(Usuario.email_google == DOCENTE_EMAIL).first()
        assert docente is not None
        assert docente.rol == "administrador"
        assert docente.activo is True
        assert db.query(Vehiculo).count() >= 20
    finally:
        db.close()


def test_routes_require_auth():
    assert client.get("/api/v1/vehiculos").status_code == 401
    assert client.post("/api/v1/placas/consultar", json={"placa": "2481KLM"}).status_code == 401
    assert client.post("/api/v1/agente/chat", data={"mensaje": "2481KLM"}).status_code == 401


def test_plate_normalization_and_ocr_suggestions():
    assert normalize_plate("bolivia 2481-klm ") == "2481KLM"
    db = SessionLocal()
    try:
        # 248IKLM tiene confusión I<->1 respecto a 2481KLM
        sugs = find_ocr_suggestions(db, "248IKLM")
        assert "2481KLM" in sugs
    finally:
        db.close()


def test_four_agent_states_and_frame_flow():
    headers = _admin_headers()

    # 1. Registrada sin paso (2481KLM aún no pasó)
    r1 = client.post("/api/v1/agente/chat", headers=headers, data={"mensaje": "Estado de 2481-KLM"})
    assert r1.status_code == 200
    assert r1.json()["estado"] == "Registrada sin paso"

    # Registrar paso APROBADO con mock de inferencia (3 pasajeros)
    mock_approved = {
        "placa": {"texto": "2481KLM", "texto_raw": "2481-KLM", "confianza": 0.94, "bbox": [10, 10, 80, 40]},
        "vehiculo": {"categoria": "auto", "confianza": 0.92, "area_mascara": 0.28},
        "ejes": 2,
        "keypoints": [[100, 200, 0.95], [260, 200, 0.93]],
        "pasajeros": 3,
        "latencia_ms": 145.0,
        "imagen_anotada_b64": "",
    }
    with patch("app.main.inference_client.infer_full", new=AsyncMock(return_value=mock_approved)):
        rf = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1"},
            files={"file": ("frame.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )
    assert rf.status_code == 200
    body_rf = rf.json()
    assert body_rf["paso"]["estado"] == "APROBADO"
    assert body_rf["paso"]["pasajeros"] == 3
    assert body_rf["paso"]["monto"] == 5.0
    assert body_rf["alerta"] is None

    # Verificar que la imagen de evidencia se sirve desde la base de datos (persistente en Render)
    img_resp = client.get(body_rf["paso"]["imagen_url"])
    assert img_resp.status_code == 200
    assert img_resp.headers["content-type"].startswith("image/jpeg")

    # 2. Registrada y con paso
    r2 = client.post("/api/v1/agente/chat", headers=headers, data={"mensaje": "2481KLM"})
    assert r2.status_code == 200
    assert r2.json()["estado"] == "Registrada y con paso"

    # Registrar paso de placa NO REGISTRADA -> debe generar alerta
    mock_unreg = {
        "placa": {"texto": "9999ZZZ", "texto_raw": "9999-ZZZ", "confianza": 0.89, "bbox": [10, 10, 80, 40]},
        "vehiculo": {"categoria": "camion", "confianza": 0.90, "area_mascara": 0.41},
        "ejes": 3,
        "keypoints": [[100, 200, 0.91], [180, 200, 0.90], [400, 200, 0.94]],
        "pasajeros": 1,
        "latencia_ms": 160.0,
        "imagen_anotada_b64": "",
    }
    with patch("app.main.inference_client.infer_full", new=AsyncMock(return_value=mock_unreg)):
        rf2 = client.post(
            "/api/v1/pasos/frame",
            headers=headers,
            data={"caseta_id": "1"},
            files={"file": ("frame2.jpg", b"\xff\xd8\xff\xd9", "image/jpeg")},
        )
    assert rf2.status_code == 200
    assert rf2.json()["paso"]["estado"] == "NO_REGISTRADO"
    assert rf2.json()["alerta"]["tipo"] == "PLACA_NO_REGISTRADA"

    # 3. Identificada no registrada
    r3 = client.post("/api/v1/agente/chat", headers=headers, data={"mensaje": "9999-ZZZ"})
    assert r3.status_code == 200
    assert r3.json()["estado"] == "Identificada no registrada"

    # 4. No existe + sugerencia OCR
    r4 = client.post("/api/v1/agente/chat", headers=headers, data={"mensaje": "248IKLM"})
    assert r4.status_code == 200
    assert r4.json()["estado"] == "No existe"
    assert "2481KLM" in r4.json()["sugerencias"]
    assert "Quisiste decir" in r4.json()["respuesta"]


def test_only_unifranz_emails_allowed():
    headers = _admin_headers()

    # Rechazar correo externo en lista blanca
    bad = client.post(
        "/api/v1/usuarios",
        headers=headers,
        json={"email_google": "externo@gmail.com", "nombre": "Externo", "rol": "operador"},
    )
    assert bad.status_code == 400
    assert "@unifranz.edu.bo" in bad.json()["detail"]

    # Aceptar correo institucional @unifranz.edu.bo
    ok = client.post(
        "/api/v1/usuarios",
        headers=headers,
        json={"email_google": "est.prueba@unifranz.edu.bo", "nombre": "Estudiante", "rol": "operador"},
    )
    assert ok.status_code == 201
    assert ok.json()["email_google"] == "est.prueba@unifranz.edu.bo"

    # Rechazar intento de login con cuenta no @unifranz.edu.bo
    with patch(
        "app.auth.verify_google_id_token",
        return_value={"email": "personal@gmail.com", "name": "Personal", "email_verified": True},
    ):
        denied = client.post("/api/v1/auth/google", json={"id_token": "fake-token-for-test-1234567890"})
    assert denied.status_code == 403
    assert "@unifranz.edu.bo" in denied.json()["detail"]

