"""
Utilidades de normalización de placas, validación de formato, sugerencias por errores
probables de OCR (O<->0, I<->1, B<->8, S<->5) y motor de los 4 estados obligatorios
del sistema (Sección 4.8).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from sqlalchemy.orm import Session

from app.models import Caseta, Paso, Vehiculo

# Pares de confusión típicos de OCR requeridos explícitamente en Sección 4.8
OCR_CONFUSION_PAIRS = {
    ("O", "0"),
    ("0", "O"),
    ("D", "0"),
    ("0", "D"),
    ("Q", "0"),
    ("0", "Q"),
    ("I", "1"),
    ("1", "I"),
    ("L", "1"),
    ("1", "L"),
    ("B", "8"),
    ("8", "B"),
    ("S", "5"),
    ("5", "S"),
    ("Z", "2"),
    ("2", "Z"),
    ("G", "6"),
    ("6", "G"),
    ("A", "4"),
    ("4", "A"),
    ("Y", "V"),
    ("V", "Y"),
    ("N", "M"),
    ("M", "N"),
}

IGNORED_TOKENS = ("ARGENTINA", "MERCOSUR", "BOLIVIA", "CHILE", "PLACA", "PERU", "BOL")

# Formatos válidos: placas bolivianas (3-4 dígitos + 3 letras, ej. 2481KLM) o alfanuméricas 5-8 chars
PLATE_REGEX = re.compile(r"^[A-Z0-9]{5,8}$")


def normalize_plate(raw: str) -> str:
    """Normaliza una placa: mayúsculas, sin encabezados, sin guiones ni espacios."""
    if not raw:
        return ""
    text = raw.upper().strip()
    for token in IGNORED_TOKENS:
        text = re.sub(rf"\b{token}\b", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def is_valid_plate_format(normalized: str) -> bool:
    """Valida formato básico de placa: 5-8 caracteres alfanuméricos con al menos 1 dígito y 1 letra."""
    if not normalized or not PLATE_REGEX.match(normalized):
        return False
    has_digit = any(ch.isdigit() for ch in normalized)
    has_alpha = any(ch.isalpha() for ch in normalized)
    return has_digit and has_alpha


def extract_plate_candidate_from_text(message: str) -> Optional[str]:
    """Extrae una posible placa desde una frase natural (ej: 'consulta la placa 2481-KLM')."""
    if not message:
        return None
    upper = message.upper().strip()

    # 1. Prioridad máxima: formato boliviano (3-4 dígitos + separador opcional + 3 letras)
    m_bo = re.search(r"\b\d{3,4}[-\s]?[A-Z]{3}\b", upper)
    if m_bo:
        return normalize_plate(m_bo.group(0))

    # 2. Formato explícito con guion que combine letras y dígitos (ej: 248I-KLM, ABC-123)
    for m in re.finditer(r"\b[A-Z0-9]{2,4}-[A-Z0-9]{2,4}\b", upper):
        norm = normalize_plate(m.group(0))
        if is_valid_plate_format(norm):
            return norm

    # 3. Token alfanumérico continuo de 5-8 caracteres con letras y dígitos (ej: 248IKLM, 9999ZZZ)
    for m in re.finditer(r"\b[A-Z0-9]{5,8}\b", upper):
        norm = normalize_plate(m.group(0))
        if is_valid_plate_format(norm):
            return norm

    # 4. Formato tipo letras + espacio + dígitos (ej: ABC 1234)
    for m in re.finditer(r"\b[A-Z]{2,3}\s+\d{3,4}\b", upper):
        norm = normalize_plate(m.group(0))
        if is_valid_plate_format(norm):
            return norm

    # 5. Si el mensaje completo normalizado es una placa válida
    whole = normalize_plate(upper)
    if is_valid_plate_format(whole):
        return whole
    return None


def ocr_weighted_distance(a: str, b: str) -> float:
    """
    Distancia de Levenshtein ponderada:
      - Confusiones OCR conocidas (O<->0, I<->1, B<->8, S<->5) cuestan 0.35
      - Sustitución/inserción/borrado normal cuesta 1.0
    """
    if a == b:
        return 0.0
    la, lb = len(a), len(b)
    if la == 0:
        return float(lb)
    if lb == 0:
        return float(la)

    dp = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = float(i)
    for j in range(lb + 1):
        dp[0][j] = float(j)

    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            ca, cb = a[i - 1], b[j - 1]
            if ca == cb:
                cost = 0.0
            elif (ca, cb) in OCR_CONFUSION_PAIRS:
                cost = 0.35
            else:
                cost = 1.0
            dp[i][j] = min(
                dp[i - 1][j] + 1.0,
                dp[i][j - 1] + 1.0,
                dp[i - 1][j - 1] + cost,
            )
    return dp[la][lb]


def find_ocr_suggestions(db: Session, normalized_plate: str, limit: int = 3) -> List[str]:
    """Busca placas cercanas en vehiculos y pasos considerando confusiones OCR (O↔0, I↔1, B↔8, S↔5)."""
    if not normalized_plate:
        return []

    known_plates = set()
    for (p,) in db.query(Vehiculo.placa).all():
        if p:
            known_plates.add(p)
    for (p,) in db.query(Paso.placa_normalizada).distinct().all():
        if p:
            known_plates.add(p)

    scored: List[Tuple[float, str]] = []
    for candidate in known_plates:
        if candidate == normalized_plate:
            continue
        dist = ocr_weighted_distance(normalized_plate, candidate)
        if dist <= 1.5:
            scored.append((dist, candidate))

    scored.sort(key=lambda item: (item[0], item[1]))
    return [plate for _dist, plate in scored[:limit]]


def lookup_plate_status(db: Session, raw_plate: str) -> Dict[str, Any]:
    """
    Determina con datos reales de BD uno de los 4 estados exactos (Sección 4.8):
      1. 'Registrada y con paso'
      2. 'Registrada sin paso'
      3. 'Identificada no registrada'
      4. 'No existe'
    """
    normalized = normalize_plate(raw_plate)
    formato_valido = is_valid_plate_format(normalized)

    if not normalized:
        return {
            "placa_consultada": raw_plate,
            "placa_normalizada": "",
            "formato_valido": False,
            "estado": "No existe",
            "vehiculo": None,
            "total_pasos": 0,
            "ultimo_paso": None,
            "sugerencias": [],
        }

    vehiculo = db.query(Vehiculo).filter(Vehiculo.placa == normalized).first()
    pasos = (
        db.query(Paso)
        .filter(Paso.placa_normalizada == normalized)
        .order_by(Paso.created_at.desc())
        .all()
    )

    ultimo_paso_dict = None
    if pasos:
        lp = pasos[0]
        caseta = db.query(Caseta).filter(Caseta.id == lp.caseta_id).first()
        ultimo_paso_dict = {
            "id": lp.id,
            "fecha": lp.created_at.isoformat() if lp.created_at else None,
            "caseta": caseta.nombre if caseta else f"Caseta #{lp.caseta_id}",
            "categoria": lp.categoria_detectada,
            "ejes": lp.ejes_detectados,
            "pasajeros": lp.pasajeros,
            "monto": lp.monto,
            "estado_paso": lp.estado,
            "imagen_url": lp.imagen_url,
        }

    vehiculo_dict = None
    if vehiculo:
        vehiculo_dict = {
            "id": vehiculo.id,
            "placa": vehiculo.placa,
            "propietario": vehiculo.propietario,
            "categoria": vehiculo.categoria,
            "ejes": vehiculo.ejes,
            "saldo": vehiculo.saldo,
            "activo": vehiculo.activo,
        }

    if vehiculo and pasos:
        estado = "Registrada y con paso"
        sugerencias: List[str] = []
    elif vehiculo and not pasos:
        estado = "Registrada sin paso"
        sugerencias = []
    elif not vehiculo and pasos:
        estado = "Identificada no registrada"
        sugerencias = find_ocr_suggestions(db, normalized)
    else:
        estado = "No existe"
        sugerencias = find_ocr_suggestions(db, normalized)

    return {
        "placa_consultada": raw_plate,
        "placa_normalizada": normalized,
        "formato_valido": formato_valido,
        "estado": estado,
        "vehiculo": vehiculo_dict,
        "total_pasos": len(pasos),
        "ultimo_paso": ultimo_paso_dict,
        "sugerencias": sugerencias,
    }
