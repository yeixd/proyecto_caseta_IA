"""
Pipeline unificado de inferencia v3.1 (Alta Precisión en Llantas, Ejes, Clase y Placas).

Basado en la metodología de IAM-811 (keypoints_llantas_yolo_detectron2_v3.ipynb):
  - M1: Detección ajustada de placa blanca/clara (HSV + BlackHat + YOLO) + deskew + EasyOCR.
  - M2: Clasificación abierta con YOLOE-11l-seg: Auto particular, Minibús, Motocicleta, Camión y Bus.
  - M3: Doble pasada de detección de llantas (escena global imgsz=1024 + zoom en tren rodante del
        vehículo imgsz=640 con umbral adaptativo 0.04) -> 5 keypoints por llanta (centro, contacto,
        superior, izquierda, derecha) + emparejamiento de perspectiva 3/4 y lateral para conteo
        exacto de ejes.
  - M4: Segmentación de pasajeros estrictamente dentro de cabina (excluye transeúntes).
"""

from __future__ import annotations

import base64
import logging
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

try:
    from ultralytics import YOLOE
    HAS_YOLOE = True
except Exception:
    YOLOE = None
    HAS_YOLOE = False

logger = logging.getLogger("peaje.inferencia")

PIPELINE_VERSION = "3.2-deblur-superocr"

CLASE_NOMBRES = {
    "auto": "Vehículo particular (Auto)",
    "minibus": "Minibús",
    "moto": "Motocicleta",
    "camion": "Camión",
    "bus": "Bus",
}

ETIQUETA_IMG = {
    "auto": "Auto particular",
    "minibus": "Minibus",
    "moto": "Moto",
    "camion": "Camion",
    "bus": "Bus",
}

COLORES_BGR = {
    "auto": (219, 152, 52),
    "bus": (60, 76, 231),
    "minibus": (113, 204, 46),
    "camion": (18, 156, 243),
    "moto": (220, 110, 190),
}

# Conjunto limpio de prompts (igual al notebook v3 + motorcycle) para no diluir puntajes CLIP de llantas
PROMPTS_VEHICULO = {
    "car": "auto",
    "sedan": "auto",
    "bus": "bus",
    "minibus": "minibus",
    "van": "minibus",
    "truck": "camion",
    "motorcycle": "moto",
}

PROMPTS_LLANTA = ["wheel", "tire", "car wheel"]
PROMPTS_LLANTA_DEDICADO = ["wheel", "tire", "car wheel", "truck tire", "vehicle wheel"]
PROMPTS_LLANTA_SET = set(PROMPTS_LLANTA_DEDICADO)
PROMPTS_PERSONA = ["person"]
PROMPTS_PLACA = ["license plate", "number plate"]
PROMPTS_PLACA_SET = set(PROMPTS_PLACA)

KPT_NOMBRES = ["centro", "contacto", "superior", "izquierda", "derecha"]
KPT_COLORES_BGR = [
    (255, 255, 255),  # 0: centro
    (0, 0, 255),      # 1: contacto
    (255, 200, 0),    # 2: superior
    (0, 255, 0),      # 3: izquierda
    (255, 0, 255),    # 4: derecha
]
ESQUELETO = [(0, 1), (0, 2), (0, 3), (0, 4)]
UMBRAL_KP = 0.03

UMBRAL_CONTENCION = 0.70
UMBRAL_CERCANA = 0.58
SEPARACION_EJE = 0.60
IMGSZ_YOLOE = 1024
CONF_VEHICULO = 0.22
CONF_LLANTA_GLOBAL = 0.05
CONF_LLANTA_ZOOM = 0.04

DEFAULT_AXLES_BY_CATEGORY = {
    "moto": 2,
    "auto": 2,
    "minibus": 2,
    "bus": 2,
    "camion": 2,
}

MAX_PASSENGERS_BY_CATEGORY = {
    "moto": 2,
    "auto": 5,
    "minibus": 15,
    "camion": 3,
    "bus": 45,
}

CATEGORY_ALIASES = {
    "moto": "moto",
    "motocicleta": "moto",
    "motorcycle": "moto",
    "motorbike": "moto",
    "auto": "auto",
    "automovil": "auto",
    "car": "auto",
    "sedan": "auto",
    "suv": "auto",
    "minibus": "minibus",
    "minibús": "minibus",
    "van": "minibus",
    "camion": "camion",
    "camión": "camion",
    "truck": "camion",
    "bus": "bus",
}

IGNORED_OCR_TOKENS = ("ARGENTINA", "MERCOSUR", "BOLIVIA", "CHILE", "PLACA", "PERU", "BOL")
BO_PLATE_RE = re.compile(r"^\d{3,4}[A-Z]{3}$")
STD_PLATE_RE = re.compile(r"^[A-Z0-9]{5,8}$")

LETTER_TO_DIGIT_POS = {
    "O": "0",
    "Q": "0",
    "D": "0",
    "I": "1",
    "L": "1",
    "Z": "2",
    "S": "5",
    "G": "6",
    "B": "8",
}
DIGIT_TO_LETTER_POS = {
    "0": "O",
    "1": "I",
    "2": "Z",
    "4": "A",
    "5": "S",
    "6": "G",
    "8": "B",
}


def correct_bolivian_plate_positions(cleaned: str) -> str:
    """
    Corrige confusiones dígito<->letra en placas borrosas bolivianas ([3-4 dígitos] + [3 letras]),
    p. ej. '2534R15' -> '2534RIS' o 'S2B0YIN' -> '5280YIN', sin alterar formatos extranjeros.
    """
    if not cleaned or BO_PLATE_RE.match(cleaned):
        return cleaned
    if len(cleaned) not in (6, 7):
        return cleaned
    num_len = 3 if len(cleaned) == 6 else 4
    prefix, suffix = cleaned[:num_len], cleaned[num_len:]
    digits_in_prefix = sum(c.isdigit() for c in prefix)
    alphas_in_suffix = sum(c.isalpha() for c in suffix)
    if digits_in_prefix < (num_len - 1) and alphas_in_suffix < 2:
        return cleaned

    new_prefix = []
    subs = 0
    for c in prefix:
        if c.isdigit():
            new_prefix.append(c)
        elif c in LETTER_TO_DIGIT_POS:
            new_prefix.append(LETTER_TO_DIGIT_POS[c])
            subs += 1
        else:
            return cleaned

    new_suffix = []
    for c in suffix:
        if c.isalpha():
            new_suffix.append(c)
        elif c in DIGIT_TO_LETTER_POS:
            new_suffix.append(DIGIT_TO_LETTER_POS[c])
            subs += 1
        else:
            return cleaned

    candidate = "".join(new_prefix) + "".join(new_suffix)
    if subs <= 2 and BO_PLATE_RE.match(candidate):
        return candidate
    return cleaned


def normalize_plate(raw_text: str) -> str:
    if not raw_text:
        return ""
    text = raw_text.upper().strip()
    for token in IGNORED_OCR_TOKENS:
        text = re.sub(rf"\b{token}\b", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def plate_quality_score(normalized: str, conf: float) -> float:
    if not normalized or len(normalized) < 4:
        return 0.0
    has_digit = any(c.isdigit() for c in normalized)
    has_alpha = any(c.isalpha() for c in normalized)
    if not (has_digit and has_alpha):
        return conf * 0.25
    if BO_PLATE_RE.match(normalized):
        return conf + 0.65
    if STD_PLATE_RE.match(normalized):
        return conf + 0.35
    return conf * 0.5


def rotate_image_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    if abs(angle_deg) < 0.5:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a, sin_a = abs(mat[0, 0]), abs(mat[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))
    mat[0, 2] += (new_w / 2.0) - cx
    mat[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        img, mat, (new_w, new_h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE
    )


def iou_box(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def contenida(a: List[float], b: List[float]) -> float:
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    return inter / area


def nms_simple(items: List[Dict[str, Any]], umbral: float = 0.55) -> List[Dict[str, Any]]:
    ordered = sorted(items, key=lambda d: -d["conf"])
    res: List[Dict[str, Any]] = []
    for d in ordered:
        if all(iou_box(d["box"], r["box"]) < umbral for r in res):
            res.append(d)
    return res


def kps_de_caja(box: List[float], conf: float = 1.0) -> np.ndarray:
    """Genera los 5 keypoints geométricos desde la caja cuando la máscara de una llanta angosta colapsa."""
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    k = np.array(
        [
            [cx, cy],   # 0: centro
            [cx, y2],   # 1: contacto con el suelo
            [cx, y1],   # 2: superior
            [x1, cy],   # 3: izquierda
            [x2, cy],   # 4: derecha
        ],
        dtype=float,
    )
    return np.concatenate([k, np.full((5, 1), conf, dtype=float)], axis=1)


def kps_de_mascara(mask: np.ndarray, fallback_box: Optional[List[float]] = None, conf: float = 1.0) -> Optional[np.ndarray]:
    """5 keypoints geométricos a partir del contorno de la llanta (notebook v3) con respaldo elíptico."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return kps_de_caja(fallback_box, conf) if fallback_box else None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return kps_de_caja(fallback_box, conf) if fallback_box else None
    pts = c.reshape(-1, 2).astype(float)
    if len(pts) >= 5:
        try:
            (cx, cy), _, _ = cv2.fitEllipse(c)
            if not (0 <= cx <= mask.shape[1] and 0 <= cy <= mask.shape[0]):
                cx, cy = pts.mean(0)
        except cv2.error:
            cx, cy = pts.mean(0)
    else:
        cx, cy = pts.mean(0)
    contacto = pts[pts[:, 1].argmax()]
    superior = pts[pts[:, 1].argmin()]
    izquierda = pts[pts[:, 0].argmin()]
    derecha = pts[pts[:, 0].argmax()]
    k = np.array([[cx, cy], contacto, superior, izquierda, derecha], dtype=float)
    return np.concatenate([k, np.full((5, 1), conf, dtype=float)], axis=1)


def altura_visible(llanta: Dict[str, Any]) -> float:
    k = llanta["kps"]
    if len(k) >= 3:
        return max(1.0, float(k[1, 1] - k[2, 1]))
    return max(1.0, float(llanta["box"][3] - llanta["box"][1]))


def ancho_visible(llanta: Dict[str, Any]) -> float:
    k = llanta["kps"]
    if len(k) >= 5:
        return max(1.0, float(k[4, 0] - k[3, 0]))
    return max(1.0, float(llanta["box"][2] - llanta["box"][0]))


def limpiar_llantas(llantas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Descarta detecciones contenidas dentro de otra (ej. aro dentro de la llanta)."""
    orden = sorted(
        llantas,
        key=lambda l: -(l["box"][2] - l["box"][0]) * (l["box"][3] - l["box"][1]),
    )
    res: List[Dict[str, Any]] = []
    for l in orden:
        if all(contenida(l["box"], r["box"]) < UMBRAL_CONTENCION for r in res):
            res.append(l)
    return res


def es_llanta_geometricamente_valida(box: List[float], vehiculo_box: Optional[List[float]]) -> bool:
    """Filtra falsos positivos (parachoques enteros, espejos superiores, repuestos altos o tornillos)."""
    lx1, ly1, lx2, ly2 = box
    lw, lh = lx2 - lx1, ly2 - ly1
    if lw < 7 or lh < 10:
        return False
    ratio = lw / max(lh, 1.0)
    # Llanta de canto (frontal/3/4) tiene ratio ~0.12..0.75; lateral circular ~0.75..1.55
    if not (0.12 <= ratio <= 1.60):
        return False
    if vehiculo_box is not None:
        vx1, vy1, vx2, vy2 = vehiculo_box
        vw, vh = max(vx2 - vx1, 1.0), max(vy2 - vy1, 1.0)
        # Permite llantas grandes de camión trasero (~0.70*vh) y motos (~0.80*vh), rechazando parachoques enteros
        if lw > vw * 0.58 or lh > vh * 0.85:
            return False
        if lh < vh * 0.045:
            return False
        # El centro de una llanta de rodaje cae debajo del tercio superior del vehículo (ignora llanta de auxilio superior)
        cy = (ly1 + ly2) / 2.0
        if cy < vy1 + vh * 0.26 or cy > vy2 + vh * 0.10:
            return False
    return True


def analizar_ejes_vehiculo(
    vehiculo_box: List[float],
    categoria: str,
    llantas_asociadas: List[Dict[str, Any]],
) -> Tuple[int, str, str, List[Dict[str, Any]]]:
    """
    Calcula ejes reales combinando la geometría del notebook v3 con corrección proyectiva 3/4 y frontal/posterior:
      - En vista frontal/posterior (llantas angostas de canto al mismo nivel Y de piso): no duplica izquierda/derecha
        como ejes distintos y detecta doble rodado en camiones/buses.
      - En vista 3/4 (ej. auto donde se ven las 2 llantas de un costado + 1 llanta delantera opuesta):
        toma el costado con mayor cantidad de llantas visibles y mide separación diagonal (X, Y).
      - En vista lateral: agrupa centros X con SEPARACION_EJE * diámetro (2, 3, 4, 5, 6 ejes).
    """
    default_axles = DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)
    if not llantas_asociadas:
        return default_axles, "frontal/sin_llantas", "Estimado por categoría", []

    vx1, vy1, vx2, vy2 = vehiculo_box
    vw, vh = max(vx2 - vx1, 1.0), max(vy2 - vy1, 1.0)
    proporcion = vw / vh
    mid_x = (vx1 + vx2) / 2.0

    hmax = max(altura_visible(l) for l in llantas_asociadas)
    cercanas = [l for l in llantas_asociadas if altura_visible(l) >= UMBRAL_CERCANA * hmax]
    if not cercanas:
        cercanas = llantas_asociadas

    redondez_med = float(np.median([ancho_visible(l) / altura_visible(l) for l in cercanas]))
    ys_contact = [float(l["kps"][1, 1]) for l in cercanas]
    y_span = (max(ys_contact) - min(ys_contact)) if len(ys_contact) >= 2 else 0.0

    # Caso 1: Vista puramente frontal o posterior -> llantas vistas de canto (angostas) y apoyadas al mismo nivel Y
    if redondez_med < 0.68 and y_span <= 0.18 * hmax and proporcion < 1.80:
        if categoria in ("camion", "bus") and len(cercanas) >= 4:
            return 2, "frontal/posterior", "Doble llanta detectada (2 por lado)", cercanas
        return default_axles, "frontal/posterior", f"{len(llantas_asociadas)} llantas frontales/traseras", cercanas

    # Caso 2: Vista 3/4 (llantas elípticas o escalonadas en profundidad Y, con posible rueda delantera opuesta)
    if redondez_med < 0.76 or proporcion < 1.60:
        left_side = [l for l in cercanas if float(l["kps"][0, 0]) < mid_x]
        right_side = [l for l in cercanas if float(l["kps"][0, 0]) >= mid_x]

        # Elegir el costado principal cuando hay ruedas en ambos lados (evita contar la rueda opuesta como eje extra)
        if left_side and right_side:
            side_wheels = left_side if len(left_side) >= len(right_side) else right_side
        else:
            side_wheels = left_side or right_side

        if len(side_wheels) >= 2:
            diam = float(np.median([altura_visible(l) for l in side_wheels]))
            ancho_med = float(np.median([ancho_visible(l) for l in side_wheels]))
            umbral_paso = max(ancho_med * 1.30, diam * 0.38)
            pts_s = sorted(side_wheels, key=lambda l: (float(l["kps"][0, 0]), float(l["kps"][1, 1])))
            ejes_34 = 1 + sum(
                1
                for p, q in zip(pts_s, pts_s[1:])
                if math.hypot(q["kps"][0, 0] - p["kps"][0, 0], q["kps"][1, 1] - p["kps"][1, 1]) > umbral_paso
            )
            vista_lbl = "lateral" if redondez_med >= 0.76 else "3/4"
            return max(2, ejes_34), vista_lbl, f"{len(llantas_asociadas)} llantas detectadas", cercanas
        return default_axles, "3/4", f"{len(llantas_asociadas)} llanta(s) visible(s)", cercanas

    # Caso 3: Vista lateral franca (ruedas circulares alineadas horizontalmente)
    xs = sorted(float(l["kps"][0, 0]) for l in cercanas)
    diam = float(np.median([altura_visible(l) for l in cercanas]))
    ejes = 1 + sum(1 for p, q in zip(xs, xs[1:]) if (q - p) > SEPARACION_EJE * diam)
    nota = ""
    if ejes < 2:
        ejes, nota = 2, "Mínimo asumido (solo se detectó 1 eje)"
    return ejes, "lateral", nota, cercanas


class TollInferencePipeline:
    """Pipeline v3.1 M1+OCR, M2 (YOLOE clases), M3 (llantas doble pasada 5-KPs) y M4 (pasajeros)."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu = os.environ.get("OCR_GPU", "true").lower() == "true" and self.device == "cuda"

        self.m1_path = os.environ.get("MODEL_M1_PLATES_PATH", "models/m1_placas.pt")
        self.m2_path = os.environ.get("MODEL_M2_VEHICLE_SEG_PATH", "models/m2_vehiculos_seg.pt")
        self.m3_path = os.environ.get("MODEL_M3_WHEEL_POSE_PATH", "models/m3_ruedas_pose.pt")
        self.m4_path = os.environ.get("MODEL_M4_PASSENGERS_SEG_PATH", "models/m4_pasajeros_seg.pt")
        self.yoloe_weight = os.environ.get("MODEL_YOLOE_WEIGHT", "yoloe-11l-seg.pt")

        self.model_status: Dict[str, Dict[str, Any]] = {}

        self.m1_plates: Optional[YOLO] = self._load_custom_yolo("M1_placas", self.m1_path)
        self.m2_custom_seg: Optional[YOLO] = self._load_custom_yolo("M2_vehiculos_seg", self.m2_path)
        self.m3_wheel_pose: Optional[YOLO] = self._load_custom_yolo("M3_ruedas_pose", self.m3_path)
        self.m4_custom_seg: Optional[YOLO] = self._load_custom_yolo("M4_pasajeros_seg", self.m4_path)

        # Dos instancias o embeddings cacheados de YOLOE: escena completa + detector dedicado de llantas
        self.yoloe_scene = None
        self.yoloe_wheels = None
        self._init_yoloe_models()

        self.fallback_coco_seg: Optional[YOLO] = None
        if self.yoloe_scene is None:
            try:
                self.fallback_coco_seg = YOLO("yolo11n-seg.pt")
            except Exception as exc:
                logger.error("No se pudo cargar fallback yolo11n-seg.pt: %s", exc)

        self.ocr_reader = self._load_ocr()

    def _load_custom_yolo(self, key: str, path: str) -> Optional[YOLO]:
        if os.path.isfile(path):
            model = YOLO(path)
            self.model_status[key] = {"loaded": True, "custom": True, "path": path, "device": self.device}
            return model
        self.model_status[key] = {"loaded": False, "custom": False, "path": path, "device": self.device}
        return None

    def _init_yoloe_models(self) -> None:
        if not HAS_YOLOE:
            self.model_status["YOLOE_unificado"] = {"loaded": False, "error": "YOLOE no disponible"}
            return
        try:
            scene_names = (
                list(PROMPTS_VEHICULO.keys())
                + PROMPTS_LLANTA
                + PROMPTS_PERSONA
                + PROMPTS_PLACA
            )
            m_scene = YOLOE(self.yoloe_weight)
            m_scene.set_classes(scene_names, m_scene.get_text_pe(scene_names))
            self.yoloe_scene = m_scene

            # Segunda instancia dedicada EXCLUSIVAMENTE a prompts de llantas:
            # sin competencia softmax con clases de vehículos -> sensibilidad máxima en llantas angostas/3/4
            m_wheels = YOLOE(self.yoloe_weight)
            m_wheels.set_classes(PROMPTS_LLANTA_DEDICADO, m_wheels.get_text_pe(PROMPTS_LLANTA_DEDICADO))
            self.yoloe_wheels = m_wheels

            self.model_status["YOLOE_unificado"] = {
                "loaded": True,
                "custom": True,
                "path": self.yoloe_weight,
                "device": self.device,
                "prompts_scene": scene_names,
                "prompts_wheels_dedicated": PROMPTS_LLANTA_DEDICADO,
            }
            for key in ("M2_vehiculos_seg", "M3_ruedas_pose", "M4_pasajeros_seg"):
                if not self.model_status[key]["loaded"]:
                    self.model_status[key] = {
                        "loaded": True,
                        "custom": False,
                        "engine": f"YOLOE ({self.yoloe_weight})",
                        "device": self.device,
                    }
        except Exception as exc:
            logger.error("Error inicializando YOLOE (%s): %s", self.yoloe_weight, exc)
            self.model_status["YOLOE_unificado"] = {"loaded": False, "error": str(exc)}

    def _load_ocr(self):
        try:
            import easyocr

            reader = easyocr.Reader(["es", "en"], gpu=self.ocr_gpu, verbose=False)
            self.model_status["OCR"] = {"loaded": True, "engine": "easyocr", "gpu": self.ocr_gpu}
            return reader
        except Exception as exc:
            self.model_status["OCR"] = {"loaded": False, "engine": "easyocr", "error": str(exc)}
            return None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "pipeline_version": PIPELINE_VERSION,
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "multi_angle_enabled": True,
            "yoloe_active": self.yoloe_scene is not None,
            "models": self.model_status,
        }

    # ------------------------------------------------------------------
    # M1: Detección YOLOE + CV y Super-OCR Deblur para placas borrosas
    # ------------------------------------------------------------------
    def _estimate_sharpness(self, gray: np.ndarray) -> float:
        if gray is None or gray.size == 0:
            return 0.0
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _deskew_plate_poly(self, image_bgr: np.ndarray, poly: np.ndarray) -> Optional[np.ndarray]:
        if poly is None or len(poly) < 4:
            return None
        rect = cv2.minAreaRect(poly.astype(np.float32))
        (_cx, _cy), (rw, rh), angle = rect
        if rw < 8 or rh < 8:
            return None
        box_pts = cv2.boxPoints(rect).astype(np.float32)
        s = box_pts.sum(axis=1)
        diff = np.diff(box_pts, axis=1).reshape(-1)
        tl = box_pts[np.argmin(s)]
        br = box_pts[np.argmax(s)]
        tr = box_pts[np.argmin(diff)]
        bl = box_pts[np.argmax(diff)]
        w_top = float(np.linalg.norm(tr - tl))
        w_bot = float(np.linalg.norm(br - bl))
        h_left = float(np.linalg.norm(bl - tl))
        h_right = float(np.linalg.norm(br - tr))
        max_w = int(max(w_top, w_bot))
        max_h = int(max(h_left, h_right))
        if max_w < 16 or max_h < 10:
            return None
        if max_h > max_w * 1.15:
            return None
        dst = np.array(
            [[0, 0], [max_w - 1, 0], [max_w - 1, max_h - 1], [0, max_h - 1]], dtype=np.float32
        )
        M = cv2.getPerspectiveTransform(np.array([tl, tr, br, bl], dtype=np.float32), dst)
        return cv2.warpPerspective(
            image_bgr, M, (max_w, max_h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE
        )

    def _build_deblur_variants(self, crop: np.ndarray) -> Tuple[List[np.ndarray], bool]:
        """
        Construye variantes restauradas (Lanczos4 4x + borde protector + Unsharp Mask + Deconvolución
        Laplaciana + Umbral Adaptativo) para leer placas borrosas, lejanas o con movimiento.
        """
        if crop is None or crop.size == 0:
            return [], False
        h, w = crop.shape[:2]
        target_w = 440.0
        if w < target_w or h < 110:
            scale = min(5.5, max(target_w / max(w, 1), 115.0 / max(h, 1)))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)

        # Borde replicado para que EasyOCR nunca recorte el primer/último carácter pegado al marco
        crop_padded = cv2.copyMakeBorder(crop, 14, 14, 18, 18, cv2.BORDER_REPLICATE)
        gray = (
            cv2.cvtColor(crop_padded, cv2.COLOR_BGR2GRAY)
            if len(crop_padded.shape) == 3
            else crop_padded
        )
        sharpness = self._estimate_sharpness(gray)
        is_blurry = sharpness < 165.0

        # Estiramiento suave de percentiles (2%..98%) para recuperar contraste en fotos lavadas/borrosas
        p2, p98 = np.percentile(gray, (2, 98))
        if p98 - p2 > 12:
            stretched = np.clip((gray.astype(np.float32) - p2) * (255.0 / (p98 - p2)), 0, 255).astype(
                np.uint8
            )
        else:
            stretched = gray

        # Variante 1: CLAHE + Unsharp Mask (óptima general y desenfoque leve/medio)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(stretched)
        blur_g = cv2.GaussianBlur(clahe, (0, 0), sigmaX=1.7)
        v1_unsharp = cv2.addWeighted(clahe, 1.85, blur_g, -0.85, 0)
        v1_unsharp = cv2.bilateralFilter(v1_unsharp, 5, 35, 35)

        # Variante 2: Deconvolución espacial fuerte para desenfoque de movimiento / foco blando
        k_deblur = np.array(
            [
                [-1, -1, -1, -1, -1],
                [-1,  2,  2,  2, -1],
                [-1,  2,  8,  2, -1],
                [-1,  2,  2,  2, -1],
                [-1, -1, -1, -1, -1],
            ],
            dtype=np.float32,
        ) / 8.0
        v2_strong = cv2.filter2D(clahe, -1, k_deblur)

        # Variante 3: Lanczos limpio (evita halos si la imagen ya tenía ruido JPEG)
        v3_clean = stretched

        # Variante 4: Binarización adaptativa suave para sombras duras o placas sucias
        v4_adapt = cv2.adaptiveThreshold(
            cv2.GaussianBlur(stretched, (3, 3), 0),
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            35,
            8,
        )

        if is_blurry:
            return [v1_unsharp, v2_strong, v3_clean, v4_adapt], True
        return [v1_unsharp, v3_clean, v2_strong], False

    def _extract_plate_from_ocr_text(self, raw_text: str) -> str:
        cleaned_full = normalize_plate(raw_text)
        if not cleaned_full:
            return ""
        corrected_full = correct_bolivian_plate_positions(cleaned_full)
        if BO_PLATE_RE.match(corrected_full):
            return corrected_full
        # Buscar sub-secuencia de 7 o 6 caracteres corregible a placa boliviana (ej. encabezado pegado)
        for length in (7, 6):
            if len(cleaned_full) >= length:
                for idx in range(0, len(cleaned_full) - length + 1):
                    sub = cleaned_full[idx : idx + length]
                    sub_corr = correct_bolivian_plate_positions(sub)
                    if BO_PLATE_RE.match(sub_corr):
                        return sub_corr
        m_bo = re.search(r"\d{3,4}[A-Z]{3}", cleaned_full)
        if m_bo:
            return m_bo.group(0)
        m_alt = re.search(r"[A-Z]{2,3}\d{3,4}[A-Z]?", cleaned_full)
        if m_alt:
            return m_alt.group(0)
        return corrected_full if len(corrected_full) <= 8 else corrected_full[:8]

    def _run_ocr_on_matrix(self, mat: np.ndarray) -> Tuple[str, str, float]:
        if self.ocr_reader is None or mat is None or mat.size == 0:
            return "", "", 0.0
        results = self.ocr_reader.readtext(
            mat,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
            detail=1,
            paragraph=False,
            text_threshold=0.35,
            low_text=0.22,
            contrast_ths=0.08,
            adjust_contrast=0.75,
            mag_ratio=1.5,
        )
        if not results:
            return "", "", 0.0

        best = ("", "", 0.0, 0.0)
        # Ordenar por línea vertical aproximada y luego izquierda->derecha
        results_sorted = sorted(results, key=lambda item: (round(item[0][0][1] / 28.0), item[0][0][0]))
        parts: List[str] = []
        confs: List[float] = []
        for _box, text, conf in results_sorted:
            t_clean = text.strip()
            if not t_clean:
                continue
            extracted = self._extract_plate_from_ocr_text(t_clean)
            if extracted:
                sc = plate_quality_score(extracted, float(conf))
                if sc > best[3]:
                    best = (extracted, t_clean, float(conf), sc)
            parts.append(t_clean)
            confs.append(float(conf))

        if parts:
            raw_joined = " ".join(parts)
            extracted_joined = self._extract_plate_from_ocr_text(raw_joined)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            sc_joined = plate_quality_score(extracted_joined, avg_conf)
            if sc_joined >= best[3]:
                best = (extracted_joined, raw_joined, avg_conf, sc_joined)

        return best[0], best[1], round(best[2], 4)

    def _run_multi_angle_ocr(self, crop: np.ndarray) -> Tuple[str, str, float]:
        """Ejecuta restauración multi-variante (deblur) + votación por consenso y ángulos."""
        if crop is None or crop.size == 0:
            return "", "", 0.0

        votes: Dict[str, Dict[str, Any]] = {}

        def _register(norm: str, raw: str, conf: float) -> float:
            if not norm:
                return 0.0
            sc = plate_quality_score(norm, conf)
            entry = votes.get(norm)
            if entry is None:
                votes[norm] = {"raw": raw, "max_conf": conf, "best_sc": sc, "hits": 1}
            else:
                entry["hits"] += 1
                if conf > entry["max_conf"]:
                    entry["max_conf"] = conf
                    entry["raw"] = raw
                # Bonificación por consenso entre múltiples filtros deblur
                entry["best_sc"] = max(entry["best_sc"], sc) + 0.12
            return votes[norm]["best_sc"]

        variants_0, is_blurry = self._build_deblur_variants(crop)
        for idx, mat in enumerate(variants_0):
            norm, raw, conf = self._run_ocr_on_matrix(mat)
            sc = _register(norm, raw, conf)
            # Salida rápida si ya es una placa boliviana nítida de alta confianza
            if idx == 0 and not is_blurry and BO_PLATE_RE.match(norm) and conf >= 0.85:
                return norm, raw, conf
            if BO_PLATE_RE.match(norm) and conf >= 0.88 and votes[norm]["hits"] >= 2:
                return norm, votes[norm]["raw"], round(votes[norm]["max_conf"], 4)

        # Si sigue dudosa o inclinada, probar rotación fina sobre la mejor variante restaurada
        base_for_rot = variants_0[0] if variants_0 else crop
        for angle in (-11.0, 11.0, -21.0, 21.0):
            rotated = rotate_image_bound(base_for_rot, angle)
            norm, raw, conf = self._run_ocr_on_matrix(rotated)
            _register(norm, raw, conf)
            if BO_PLATE_RE.match(norm) and conf >= 0.84:
                break

        if not votes:
            return "", "", 0.0

        winner_norm, winner_info = max(votes.items(), key=lambda kv: kv[1]["best_sc"])
        return winner_norm, winner_info["raw"], round(float(winner_info["max_conf"]), 4)

    def _find_plate_candidates_cv(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
        candidates: List[Tuple[np.ndarray, List[int]]] = []
        seen_boxes: List[List[int]] = []
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        # Máscara HSV de placa blanca/clara (recorta exactamente el rectángulo blanco '5280 YIN')
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(hsv, np.array([0, 0, 140]), np.array([180, 75, 255]))
        white_mask = cv2.morphologyEx(
            white_mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (9, 5))
        )

        # BlackHat morfológico
        rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
        grad_x = np.absolute(cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        max_v = float(np.max(grad_x)) if grad_x.size else 0.0
        grad_x = ((255.0 * (grad_x / max_v)).astype("uint8") if max_v > 0 else np.zeros_like(gray))
        grad_x = cv2.morphologyEx(cv2.GaussianBlur(grad_x, (5, 5), 0), cv2.MORPH_CLOSE, rect_k)
        _, thresh_bh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        all_contours = []
        for mask in (white_mask, thresh_bh):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(sorted(cnts, key=cv2.contourArea, reverse=True)[:15])
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

        for cnt in all_contours:
            area = cv2.contourArea(cnt)
            if area < (h * w) * 0.0012 or area > (h * w) * 0.30:
                continue
            (_rcx, _rcy), (rw, rh), angle = cv2.minAreaRect(cnt)
            if rw < 1 or rh < 1:
                continue
            ratio = max(rw, rh) / min(rw, rh)
            if not (1.4 <= ratio <= 5.5):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad_x, pad_y = int(bw * 0.10), int(bh * 0.14)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            if any(abs(x1 - sx1) < 16 and abs(y1 - sy1) < 16 for sx1, sy1, _, _ in seen_boxes):
                continue
            seen_boxes.append([x1, y1, x2, y2])
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                deskew_angle = angle if rw >= rh else angle + 90.0
                if -35.0 <= deskew_angle <= 35.0 and abs(deskew_angle) > 2.5:
                    crop = rotate_image_bound(crop, deskew_angle)
                candidates.append((crop, [x1, y1, x2, y2]))
            if len(candidates) >= 6:
                break
        return candidates

    def infer_plate_only(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]] = None,
        yoloe_plates: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        bbox: Optional[List[int]] = None
        det_conf = 0.0
        best_norm, best_raw, best_ocr_conf, best_score = "", "", 0.0, 0.0

        # 0. Si se invoca directo (/infer/placa en Agente), localizar placa y vehículo con YOLOE (resiste desenfoque fuerte)
        if yoloe_plates is None and self.yoloe_scene is not None:
            yoloe_plates = []
            rp = self.yoloe_scene.predict(
                image_bgr, conf=0.06, imgsz=IMGSZ_YOLOE, device=self.device, verbose=False
            )[0]
            if len(rp.boxes) > 0:
                clases = rp.boxes.cls.cpu().numpy().astype(int)
                confs = rp.boxes.conf.cpu().numpy()
                cajas = rp.boxes.xyxy.cpu().numpy()
                polis = rp.masks.xy if rp.masks is not None else [np.zeros((0, 2))] * len(clases)
                best_veh_area = 0.0
                for c, s, b, poly in zip(clases, confs, cajas, polis):
                    nombre = rp.names[c]
                    bl = [float(v) for v in b.tolist()]
                    if nombre in PROMPTS_PLACA_SET and float(s) >= 0.06:
                        yoloe_plates.append(
                            {
                                "conf": float(s),
                                "box": bl,
                                "poly": poly.astype(np.int32) if len(poly) >= 4 else None,
                            }
                        )
                    elif vehicle_bbox is None and nombre in PROMPTS_VEHICULO and float(s) >= 0.20:
                        area_v = (bl[2] - bl[0]) * (bl[3] - bl[1]) * float(s)
                        if area_v > best_veh_area:
                            best_veh_area = area_v
                            vehicle_bbox = [int(v) for v in bl]

        # 1. Evaluar candidatos detectados por YOLOE (con rectificación de perspectiva 4 puntos + recorte directo)
        if yoloe_plates:
            plates_sorted = sorted(yoloe_plates, key=lambda d: -d["conf"])
            if vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                vw, vh = max(1, vx2 - vx1), max(1, vy2 - vy1)
                in_veh = [
                    p
                    for p in plates_sorted
                    if (vx1 - 0.08 * vw) <= (p["box"][0] + p["box"][2]) / 2.0 <= (vx2 + 0.08 * vw)
                    and (vy1 + 0.25 * vh) <= (p["box"][1] + p["box"][3]) / 2.0 <= (vy2 + 0.08 * vh)
                ]
                if in_veh:
                    plates_sorted = in_veh + [p for p in plates_sorted if p not in in_veh]

            for p_cand in plates_sorted[:4]:
                px1, py1, px2, py2 = [int(round(v)) for v in p_cand["box"]]
                pad_x = max(4, int((px2 - px1) * 0.12))
                pad_y = max(4, int((py2 - py1) * 0.16))
                cx1, cy1 = max(0, px1 - pad_x), max(0, py1 - pad_y)
                cx2, cy2 = min(w, px2 + pad_x), min(h, py2 + pad_y)

                crops_to_try = []
                warped = self._deskew_plate_poly(image_bgr, p_cand.get("poly"))
                if warped is not None and warped.size > 0:
                    crops_to_try.append(warped)
                if cx2 > cx1 and cy2 > cy1:
                    crops_to_try.append(image_bgr[cy1:cy2, cx1:cx2])

                for sub_crop in crops_to_try:
                    norm, raw, conf = self._run_multi_angle_ocr(sub_crop)
                    sc = plate_quality_score(norm, conf)
                    if sc > best_score:
                        best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                        det_conf = max(det_conf, float(p_cand["conf"]))
                        bbox = [cx1, cy1, cx2, cy2]
                    if BO_PLATE_RE.match(best_norm) and best_ocr_conf >= 0.84:
                        break
                if BO_PLATE_RE.match(best_norm) and best_ocr_conf >= 0.84:
                    break

        # 2. Modelo custom M1 si está cargado
        if best_score < 1.35 and self.m1_plates is not None:
            preds = self.m1_plates.predict(source=image_bgr, conf=0.18, device=self.device, verbose=False)
            if preds and len(preds[0].boxes) > 0:
                best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                box = preds[0].boxes[best_idx]
                m1_c = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                pad_x, pad_y = int((x2 - x1) * 0.10), int((y2 - y1) * 0.14)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                norm, raw, conf = self._run_multi_angle_ocr(image_bgr[y1:y2, x1:x2])
                sc = plate_quality_score(norm, conf)
                if sc > best_score:
                    best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                    det_conf = max(det_conf, m1_c)
                    bbox = [x1, y1, x2, y2]

        # 3. Candidatos morfológicos HSV + BlackHat + Zoom en parachoques para imágenes muy borrosas
        if best_score < 1.25:
            roi = image_bgr
            ox, oy = 0, 0
            if vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                ry1 = vy1 + int((vy2 - vy1) * 0.36)
                roi = image_bgr[max(0, ry1) : min(h, vy2), max(0, vx1) : min(w, vx2)]
                ox, oy = max(0, vx1), max(0, ry1)
            if roi.size > 0:
                for cand_crop, cbox in self._find_plate_candidates_cv(roi):
                    norm, raw, conf = self._run_multi_angle_ocr(cand_crop)
                    sc = plate_quality_score(norm, conf)
                    if sc > best_score:
                        best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                        bbox = [cbox[0] + ox, cbox[1] + oy, cbox[2] + ox, cbox[3] + oy]
                    if BO_PLATE_RE.match(best_norm) and best_ocr_conf >= 0.82:
                        break

                # Si la placa es tan borrosa que no generó contorno cerrado, barrer franja central-baja del parachoques
                if best_score < 0.95:
                    rh_roi, rw_roi = roi.shape[:2]
                    bumper_band = roi[int(rh_roi * 0.22) : int(rh_roi * 0.96), int(rw_roi * 0.12) : int(rw_roi * 0.88)]
                    if bumper_band.size > 0:
                        norm, raw, conf = self._run_multi_angle_ocr(bumper_band)
                        sc = plate_quality_score(norm, conf)
                        if sc > best_score:
                            best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                            if bbox is None:
                                bbox = [
                                    ox + int(rw_roi * 0.12),
                                    oy + int(rh_roi * 0.22),
                                    ox + int(rw_roi * 0.88),
                                    oy + int(rh_roi * 0.96),
                                ]

        if best_score < 0.70:
            norm, raw, conf = self._run_multi_angle_ocr(image_bgr)
            sc = plate_quality_score(norm, conf)
            if sc > best_score:
                best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc

        final_conf = round(best_ocr_conf if det_conf == 0 else (0.25 * det_conf + 0.75 * best_ocr_conf), 4)
        return {
            "placa": {
                "texto": best_norm,
                "texto_raw": best_raw,
                "confianza": final_conf,
                "confianza_deteccion": round(det_conf, 4),
                "confianza_ocr": round(best_ocr_conf, 4),
                "bbox": bbox,
                "legible": len(best_norm) >= 4,
            },
            "latencia_ms": round((time.perf_counter() - t0) * 1000.0, 2),
        }

    # ------------------------------------------------------------------
    # Detección de vehículos + Doble pasada dedicada de llantas (YOLOE)
    # ------------------------------------------------------------------
    def _extract_wheels_from_result(
        self,
        r,
        offset_x: int,
        offset_y: int,
        min_conf: float,
        vehiculo_box: Optional[List[float]],
    ) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        if len(r.boxes) == 0:
            return out
        H_r, W_r = r.orig_shape
        clases = r.boxes.cls.cpu().numpy().astype(int)
        confs = r.boxes.conf.cpu().numpy()
        cajas = r.boxes.xyxy.cpu().numpy()
        polis = r.masks.xy if r.masks is not None else [np.zeros((0, 2))] * len(clases)

        for c, s, b, poly in zip(clases, confs, cajas, polis):
            nombre = r.names[c]
            if nombre not in PROMPTS_LLANTA_SET or float(s) < min_conf:
                continue
            gx1, gy1, gx2, gy2 = (
                float(b[0] + offset_x),
                float(b[1] + offset_y),
                float(b[2] + offset_x),
                float(b[3] + offset_y),
            )
            gbox = [gx1, gy1, gx2, gy2]
            if not es_llanta_geometricamente_valida(gbox, vehiculo_box):
                continue

            if len(poly) >= 3:
                m = np.zeros((H_r, W_r), np.uint8)
                cv2.fillPoly(m, [poly.astype(np.int32)], 1)
                kps = kps_de_mascara(
                    m,
                    fallback_box=[float(b[0]), float(b[1]), float(b[2]), float(b[3])],
                    conf=float(s),
                )
                if kps is not None:
                    kps[:, 0] += offset_x
                    kps[:, 1] += offset_y
            else:
                kps = kps_de_caja(gbox, conf=float(s))

            if kps is not None:
                out.append({"conf": float(s), "box": gbox, "kps": kps})
        return out

    def _enhance_cabin_for_passengers(self, cabin_bgr: np.ndarray) -> np.ndarray:
        """
        Realza siluetas borrosas u oscuras detrás del parabrisas/ventanas (LAB CLAHE + levantamiento
        de sombras gamma + Unsharp Mask) sin alterar colores base para YOLOE.
        """
        if cabin_bgr is None or cabin_bgr.size == 0:
            return cabin_bgr
        h, w = cabin_bgr.shape[:2]
        if w < 420:
            scale = min(3.0, 480.0 / max(w, 1))
            cabin_bgr = cv2.resize(cabin_bgr, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        lab = cv2.cvtColor(cabin_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        # Levantar sombras en vidrios polarizados / oscuros (gamma = 0.78)
        lut = np.array([min(255, int(((i / 255.0) ** 0.78) * 255.0)) for i in range(256)], dtype=np.uint8)
        l_lifted = cv2.LUT(l_ch, lut)
        l_clahe = cv2.createCLAHE(clipLimit=3.4, tileGridSize=(8, 8)).apply(l_lifted)
        merged = cv2.cvtColor(cv2.merge((l_clahe, a_ch, b_ch)), cv2.COLOR_LAB2BGR)
        blur = cv2.GaussianBlur(merged, (0, 0), sigmaX=1.4)
        return cv2.addWeighted(merged, 1.55, blur, -0.55, 0)

    def _analyze_scene_yoloe(
        self, image_bgr: np.ndarray
    ) -> Tuple[Optional[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        H, W = image_bgr.shape[:2]
        img_area = float(H * W)
        vehiculos: List[Dict[str, Any]] = []
        llantas_raw: List[Dict[str, Any]] = []
        personas: List[Dict[str, Any]] = []
        placas_yoloe: List[Dict[str, Any]] = []

        if self.yoloe_scene is not None:
            r = self.yoloe_scene.predict(
                image_bgr,
                conf=CONF_LLANTA_GLOBAL,
                imgsz=IMGSZ_YOLOE,
                device=self.device,
                verbose=False,
            )[0]
            if len(r.boxes) > 0:
                clases = r.boxes.cls.cpu().numpy().astype(int)
                confs = r.boxes.conf.cpu().numpy()
                cajas = r.boxes.xyxy.cpu().numpy()
                polis = r.masks.xy if r.masks is not None else [np.zeros((0, 2))] * len(clases)
                for c, s, b, poly in zip(clases, confs, cajas, polis):
                    nombre = r.names[c]
                    box_list = [float(v) for v in b.tolist()]
                    poly_int = poly.astype(np.int32) if len(poly) >= 3 else None
                    if nombre in PROMPTS_VEHICULO and float(s) >= CONF_VEHICULO:
                        vehiculos.append(
                            {
                                "clase": PROMPTS_VEHICULO[nombre],
                                "prompt": nombre,
                                "conf": float(s),
                                "box": box_list,
                                "poly": poly_int,
                            }
                        )
                    elif nombre in PROMPTS_PERSONA and float(s) >= 0.16:
                        personas.append(
                            {"conf": float(s), "box": box_list, "poly": poly_int, "from_cabin_zoom": False}
                        )
                    elif nombre in PROMPTS_PLACA_SET and float(s) >= 0.05:
                        placas_yoloe.append(
                            {
                                "conf": float(s),
                                "box": box_list,
                                "poly": poly.astype(np.int32) if len(poly) >= 4 else None,
                            }
                        )

            vehiculos = nms_simple(vehiculos, 0.65)
            personas = nms_simple(personas, 0.55)

            principal: Optional[Dict[str, Any]] = None
            if vehiculos:
                principal = max(
                    vehiculos,
                    key=lambda v: v["conf"]
                    * (((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1])) / img_area),
                )

            v_box_ref = principal["box"] if principal else None
            llantas_raw.extend(
                self._extract_wheels_from_result(r, 0, 0, CONF_LLANTA_GLOBAL, v_box_ref)
            )

            # PASADA DEDICADA EN CABINA MEJORADA (CLAHE + Deblur) PARA PASAJEROS BORROSOS / TRAS PARABRISAS
            if principal is not None:
                vx1, vy1, vx2, vy2 = principal["box"]
                vw, vh = vx2 - vx1, vy2 - vy1
                cab_x1 = max(0, int(vx1 + vw * 0.04))
                cab_y1 = max(0, int(vy1 + vh * 0.02))
                cab_x2 = min(W, int(vx2 - vw * 0.04))
                cab_y2 = min(H, int(vy1 + vh * 0.68) if principal["clase"] != "moto" else int(vy2))
                cabin_raw = image_bgr[cab_y1:cab_y2, cab_x1:cab_x2]
                if cabin_raw.size > 0 and cabin_raw.shape[0] >= 36 and cabin_raw.shape[1] >= 36:
                    cabin_enh = self._enhance_cabin_for_passengers(cabin_raw)
                    sx = cabin_raw.shape[1] / float(cabin_enh.shape[1])
                    sy = cabin_raw.shape[0] / float(cabin_enh.shape[0])
                    rcab = self.yoloe_scene.predict(
                        cabin_enh, conf=0.10, imgsz=640, device=self.device, verbose=False
                    )[0]
                    if len(rcab.boxes) > 0:
                        c_cls = rcab.boxes.cls.cpu().numpy().astype(int)
                        c_cnf = rcab.boxes.conf.cpu().numpy()
                        c_box = rcab.boxes.xyxy.cpu().numpy()
                        c_pol = rcab.masks.xy if rcab.masks is not None else [np.zeros((0, 2))] * len(c_cls)
                        for cc, cs, cb, cpoly in zip(c_cls, c_cnf, c_box, c_pol):
                            cname = rcab.names[cc]
                            if cname in PROMPTS_PERSONA and float(cs) >= 0.10:
                                gbx = [
                                    float(cb[0] * sx + cab_x1),
                                    float(cb[1] * sy + cab_y1),
                                    float(cb[2] * sx + cab_x1),
                                    float(cb[3] * sy + cab_y1),
                                ]
                                gpoly = None
                                if len(cpoly) >= 3:
                                    pts = cpoly.astype(np.float32).copy()
                                    pts[:, 0] = pts[:, 0] * sx + cab_x1
                                    pts[:, 1] = pts[:, 1] * sy + cab_y1
                                    gpoly = pts.astype(np.int32)
                                personas.append(
                                    {"conf": float(cs), "box": gbx, "poly": gpoly, "from_cabin_zoom": True}
                                )
                personas = nms_simple(personas, 0.48)

            # PASADAS DEDICADAS DE LLANTAS (sin competencia con clases de vehículos):
            # 1) Escena completa con contexto global del vehículo + 2) Zoom en tren rodante a alta resolución
            if self.yoloe_wheels is not None:
                rw_full = self.yoloe_wheels.predict(
                    image_bgr,
                    conf=CONF_LLANTA_GLOBAL,
                    imgsz=IMGSZ_YOLOE,
                    device=self.device,
                    verbose=False,
                )[0]
                llantas_raw.extend(
                    self._extract_wheels_from_result(rw_full, 0, 0, CONF_LLANTA_GLOBAL, v_box_ref)
                )

                if principal is not None:
                    vx1, vy1, vx2, vy2 = principal["box"]
                    vw, vh = vx2 - vx1, vy2 - vy1
                    cx1 = max(0, int(vx1 - vw * 0.06))
                    cy1 = max(0, int(vy1 + vh * 0.20))
                    cx2 = min(W, int(vx2 + vw * 0.06))
                    cy2 = min(H, int(vy2 + vh * 0.06))
                    wheel_crop = image_bgr[cy1:cy2, cx1:cx2]
                    if wheel_crop.size > 0 and wheel_crop.shape[0] >= 40 and wheel_crop.shape[1] >= 40:
                        rw_crop = self.yoloe_wheels.predict(
                            wheel_crop,
                            conf=CONF_LLANTA_ZOOM,
                            imgsz=768,
                            device=self.device,
                            verbose=False,
                        )[0]
                        llantas_raw.extend(
                            self._extract_wheels_from_result(
                                rw_crop, cx1, cy1, CONF_LLANTA_ZOOM, v_box_ref
                            )
                        )
        else:
            principal = None
            if self.fallback_coco_seg is not None:
                r = self.fallback_coco_seg.predict(
                    image_bgr, conf=0.22, device=self.device, verbose=False
                )[0]
                if len(r.boxes) > 0:
                    clases = r.boxes.cls.cpu().numpy().astype(int)
                    confs = r.boxes.conf.cpu().numpy()
                    cajas = r.boxes.xyxy.cpu().numpy()
                    polis = r.masks.xy if r.masks is not None else [np.zeros((0, 2))] * len(clases)
                    for c, s, b, poly in zip(clases, confs, cajas, polis):
                        raw_name = str(r.names.get(int(c), "")).lower()
                        poly_int = poly.astype(np.int32) if len(poly) >= 3 else None
                        box_list = [float(v) for v in b.tolist()]
                        mapped = CATEGORY_ALIASES.get(raw_name)
                        if mapped:
                            vehiculos.append(
                                {"clase": mapped, "prompt": raw_name, "conf": float(s), "box": box_list, "poly": poly_int}
                            )
                        elif raw_name == "person":
                            personas.append({"conf": float(s), "box": box_list, "poly": poly_int})
                if vehiculos:
                    principal = max(
                        vehiculos,
                        key=lambda v: v["conf"]
                        * (((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1])) / img_area),
                    )

        # Si existe M3 YOLO-pose custom entrenado, usar sus keypoints cuando detecta llantas
        if self.m3_wheel_pose is not None:
            rp = self.m3_wheel_pose.predict(image_bgr, conf=0.20, device=self.device, verbose=False)[0]
            if rp.keypoints is not None and len(rp.boxes) > 0:
                custom_wheels = []
                xy = rp.keypoints.xy.cpu().numpy()
                kc = rp.keypoints.conf.cpu().numpy() if rp.keypoints.conf is not None else np.ones(xy.shape[:2])
                for b, s, p, c in zip(rp.boxes.xyxy.cpu().numpy(), rp.boxes.conf.cpu().numpy(), xy, kc):
                    custom_wheels.append(
                        {"conf": float(s), "box": b.tolist(), "kps": np.concatenate([p, c[:, None]], axis=1)}
                    )
                if custom_wheels:
                    llantas_raw = custom_wheels

        llantas_limpias = limpiar_llantas(nms_simple(llantas_raw, 0.48))

        # Desambiguar llantas de vehículos adyacentes en tráfico denso (ej. auto vecino pegado al principal)
        if principal is not None and len(vehiculos) > 1 and llantas_limpias:
            otros = [
                v for v in vehiculos if v is not principal and iou_box(v["box"], principal["box"]) < 0.50
            ]
            if otros:
                main_mask_dil = None
                if principal.get("poly") is not None and len(principal["poly"]) >= 3:
                    m0 = np.zeros((H, W), dtype=np.uint8)
                    cv2.fillPoly(m0, [principal["poly"]], 1)
                    k_dil = max(15, int(round(max(H, W) * 0.025)))
                    main_mask_dil = cv2.dilate(
                        m0, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_dil, k_dil))
                    )
                filtradas_trafico: List[Dict[str, Any]] = []
                for l in llantas_limpias:
                    lx1, ly1, lx2, ly2 = [int(round(v)) for v in l["box"]]
                    cx1, cy1 = max(0, lx1), max(0, ly1)
                    cx2, cy2 = min(W, lx2), min(H, ly2)
                    touches_main_mask = True
                    if main_mask_dil is not None and cx2 > cx1 and cy2 > cy1:
                        touches_main_mask = float(np.mean(main_mask_dil[cy1:cy2, cx1:cx2])) >= 0.15
                    c_main = contenida(l["box"], principal["box"])
                    c_other = max(contenida(l["box"], ov["box"]) for ov in otros)
                    if not touches_main_mask and c_other >= 0.65:
                        continue
                    if c_other > c_main + 0.25 and not touches_main_mask:
                        continue
                    filtradas_trafico.append(l)
                llantas_limpias = filtradas_trafico

        return principal, llantas_limpias, personas, placas_yoloe

    # ------------------------------------------------------------------
    # M4: Filtrado estricto de pasajeros dentro de cabina (excluye transeúntes)
    # ------------------------------------------------------------------
    def _filter_cabin_passengers(
        self,
        image_bgr: np.ndarray,
        vehiculo: Optional[Dict[str, Any]],
        personas_globales: List[Dict[str, Any]],
    ) -> Tuple[int, List[np.ndarray]]:
        if vehiculo is None:
            return 0, []

        img_h, img_w = image_bgr.shape[:2]
        vx1, vy1, vx2, vy2 = [int(v) for v in vehiculo["box"]]
        categoria = vehiculo["clase"]
        vehicle_poly = vehiculo.get("poly")
        bw, bh = max(vx2 - vx1, 1), max(vy2 - vy1, 1)

        vehicle_mask_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        if vehicle_poly is not None and len(vehicle_poly) >= 3:
            # Usar hull convexo suave para sellar huecos de reflejo en parabrisas sin incluir la acera
            hull = cv2.convexHull(vehicle_poly)
            cv2.fillPoly(vehicle_mask_bin, [hull], 1)
        else:
            cv2.rectangle(vehicle_mask_bin, (vx1, vy1), (vx2, vy2), 1, -1)

        if categoria == "moto":
            win_x1, win_x2 = vx1 + int(bw * 0.04), vx2 - int(bw * 0.04)
            win_y1, win_y2 = max(0, vy1 - int(bh * 0.28)), vy2
        else:
            win_x1, win_x2 = vx1 + int(bw * 0.06), vx2 - int(bw * 0.06)
            win_y1, win_y2 = vy1 + int(bh * 0.03), vy1 + int(bh * 0.66)

        def _mask_cov(gx1: int, gy1: int, gx2: int, gy2: int) -> float:
            cx1, cy1 = max(0, gx1), max(0, gy1)
            cx2, cy2 = min(img_w, gx2), min(img_h, gy2)
            if cx2 <= cx1 or cy2 <= cy1:
                return 0.0
            return float(np.mean(vehicle_mask_bin[cy1:cy2, cx1:cx2]))

        def _is_pedestrian(gx1: int, gy1: int, gx2: int, gy2: int) -> bool:
            if categoria == "moto":
                return _mask_cov(gx1, gy1, gx2, gy2) < 0.28
            pw, ph = max(gx2 - gx1, 1), max(gy2 - gy1, 1)
            # Un transeúnte en la calle extiende las piernas hacia el suelo (debajo del 72% de la altura)
            if gy2 > vy1 + bh * 0.72:
                return True
            if ph > bh * 0.64:
                return True
            if gy1 < vy1 - bh * 0.02:
                return True
            if gx1 < vx1 + bw * 0.04 or gx2 > vx2 - bw * 0.04:
                return True
            if _mask_cov(gx1, gy1, gx2, gy2) < 0.72:
                return True
            if pw > bw * 0.62:
                return True
            return False

        passenger_masks: List[np.ndarray] = []
        centers_seen: List[Tuple[float, float]] = []

        for p in personas_globales:
            gx1, gy1, gx2, gy2 = [int(v) for v in p["box"]]
            min_p_conf = 0.11 if p.get("from_cabin_zoom") else 0.20
            if _is_pedestrian(gx1, gy1, gx2, gy2) or p["conf"] < min_p_conf:
                continue
            cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
            if categoria != "moto" and not (win_x1 <= cx <= win_x2 and win_y1 <= cy <= win_y2):
                continue
            min_sep = max(18.0, min(bw, bh) * 0.07)
            if any(math.hypot(cx - ox, cy - oy) < min_sep for ox, oy in centers_seen):
                continue
            centers_seen.append((cx, cy))
            if p.get("poly") is not None:
                passenger_masks.append(p["poly"])
            else:
                passenger_masks.append(
                    np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.int32)
                )

        max_allowed = MAX_PASSENGERS_BY_CATEGORY.get(categoria, 5)
        return min(len(passenger_masks), max_allowed), passenger_masks[:max_allowed]

    # ------------------------------------------------------------------
    # Anotación visual (cajas + esqueleto 5-KPs de llantas + clase + placa + pasajeros)
    # ------------------------------------------------------------------
    def _annotate_evidence(
        self,
        image_bgr: np.ndarray,
        placa_info: Dict[str, Any],
        vehiculo: Optional[Dict[str, Any]],
        llantas_dibujar: List[Dict[str, Any]],
        ejes: int,
        vista: str,
        passenger_masks: List[np.ndarray],
    ) -> str:
        out = image_bgr.copy()
        overlay = image_bgr.copy()
        g = max(2, int(round(max(out.shape[:2]) / 450)))

        if vehiculo is not None and vehiculo.get("poly") is not None:
            col = COLORES_BGR.get(vehiculo["clase"], (219, 152, 52))
            cv2.fillPoly(overlay, [vehiculo["poly"]], col)

        for p_poly in passenger_masks:
            cv2.fillPoly(overlay, [p_poly], (0, 210, 190))
            cv2.polylines(out, [p_poly], True, (0, 245, 220), 2)

        cv2.addWeighted(overlay, 0.25, out, 0.75, 0, out)

        if vehiculo is not None:
            clase = vehiculo["clase"]
            col = COLORES_BGR.get(clase, (219, 152, 52))
            x1, y1, x2, y2 = map(int, vehiculo["box"])
            cv2.rectangle(out, (x1, y1), (x2, y2), col, g)
            texto = (
                f"{ETIQUETA_IMG.get(clase, clase)} | {len(llantas_dibujar)} llantas | "
                f"ejes: {ejes} ({vista}) | {len(passenger_masks)} pasaj."
            )
            esc, gt = g * 0.32, max(1, g // 2)
            (tw, th), _ = cv2.getTextSize(texto, cv2.FONT_HERSHEY_SIMPLEX, esc, gt)
            cv2.rectangle(out, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), col, -1)
            cv2.putText(
                out,
                texto,
                (x1 + 4, max(th + 2, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                esc,
                (255, 255, 255),
                gt,
                cv2.LINE_AA,
            )

        for l in llantas_dibujar:
            k = l["kps"]
            lx1, ly1, lx2, ly2 = map(int, l["box"])
            cv2.rectangle(out, (lx1, ly1), (lx2, ly2), (0, 255, 255), max(1, g // 2))
            if len(k) >= 5:
                for a, b in ESQUELETO:
                    if k[a, 2] > UMBRAL_KP and k[b, 2] > UMBRAL_KP:
                        cv2.line(
                            out,
                            tuple(map(int, k[a, :2])),
                            tuple(map(int, k[b, :2])),
                            (0, 255, 255),
                            max(1, g // 2),
                            cv2.LINE_AA,
                        )
            for idx in range(min(len(k), len(KPT_COLORES_BGR))):
                if k[idx, 2] > UMBRAL_KP:
                    pt = tuple(map(int, k[idx, :2]))
                    cv2.circle(out, pt, g + 2, KPT_COLORES_BGR[idx], -1, cv2.LINE_AA)
                    cv2.circle(out, pt, g + 2, (0, 0, 0), 1, cv2.LINE_AA)

        p_bbox = placa_info.get("bbox")
        if p_bbox:
            px1, py1, px2, py2 = p_bbox
            cv2.rectangle(out, (px1, py1), (px2, py2), (40, 230, 100), g)
            p_txt = placa_info.get("texto") or "PLACA"
            cv2.putText(
                out,
                f"{p_txt} ({placa_info.get('confianza', 0):.2f})",
                (px1, max(22, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 230, 100),
                2,
                cv2.LINE_AA,
            )

        max_w = 960
        if out.shape[1] > max_w:
            ratio = max_w / float(out.shape[1])
            out = cv2.resize(out, (max_w, int(out.shape[0] * ratio)), interpolation=cv2.INTER_AREA)

        ok, buffer = cv2.imencode(".jpg", out, [int(cv2.IMWRITE_JPEG_QUALITY), 84])
        return base64.b64encode(buffer.tobytes()).decode("utf-8") if ok else ""

    # ------------------------------------------------------------------
    # Inferencia completa de punta a punta
    # ------------------------------------------------------------------
    def infer_full(self, image_bgr: np.ndarray, include_annotated: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()
        H, W = image_bgr.shape[:2]
        img_area = float(H * W)

        principal, llantas, personas, placas_yoloe = self._analyze_scene_yoloe(image_bgr)

        if principal is not None:
            categoria = principal["clase"]
            conf_veh = round(float(principal["conf"]), 4)
            v_box_int = [int(v) for v in principal["box"]]
            if principal.get("poly") is not None:
                area_ratio = round(float(cv2.contourArea(principal["poly"]) / img_area), 4)
            else:
                area_ratio = round(
                    float(((v_box_int[2] - v_box_int[0]) * (v_box_int[3] - v_box_int[1])) / img_area), 4
                )

            vx1, vy1, vx2, vy2 = principal["box"]
            vw, vh = vx2 - vx1, vy2 - vy1
            llantas_veh = []
            for l in llantas:
                cx, cy = float(l["kps"][0, 0]), float(l["kps"][0, 1])
                if (vx1 - 0.06 * vw) <= cx <= (vx2 + 0.06 * vw) and (vy1 + 0.26 * vh) <= cy <= (vy2 + 0.10 * vh):
                    llantas_veh.append(l)

            ejes, vista, nota_ejes, llantas_cercanas = analizar_ejes_vehiculo(
                principal["box"], categoria, llantas_veh
            )
        else:
            categoria = "auto"
            conf_veh = 0.0
            area_ratio = 0.0
            v_box_int = None
            llantas_veh = llantas
            llantas_cercanas = llantas
            ejes = 2
            vista = "sin_vehiculo"
            nota_ejes = "No se detectó vehículo completo"

        m1_out = self.infer_plate_only(
            image_bgr, vehicle_bbox=v_box_int, yoloe_plates=placas_yoloe
        )
        placa_info = m1_out["placa"]

        pasajeros, passenger_masks = self._filter_cabin_passengers(image_bgr, principal, personas)

        kp_centros = [
            [round(float(l["kps"][0, 0]), 1), round(float(l["kps"][0, 1]), 1), round(float(l["conf"]), 4)]
            for l in llantas_cercanas
        ]

        annotated_b64 = ""
        if include_annotated:
            annotated_b64 = self._annotate_evidence(
                image_bgr=image_bgr,
                placa_info=placa_info,
                vehiculo=principal,
                llantas_dibujar=llantas_veh,
                ejes=ejes,
                vista=vista,
                passenger_masks=passenger_masks,
            )

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "placa": placa_info,
            "vehiculo": {
                "categoria": categoria,
                "categoria_nombre": CLASE_NOMBRES.get(categoria, categoria.title()),
                "confianza": conf_veh,
                "area_mascara": area_ratio,
                "angulo": vista,
                "llantas_visibles": len(llantas_veh),
                "nota_ejes": nota_ejes,
            },
            "ejes": ejes,
            "llantas_visibles": len(llantas_veh),
            "vista": vista,
            "keypoints": kp_centros,
            "pasajeros": pasajeros,
            "latencia_ms": total_latency_ms,
            "imagen_anotada_b64": annotated_b64,
        }
