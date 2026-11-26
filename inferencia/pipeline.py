"""
Pipeline unificado de inferencia v3.0 para el Sistema Inteligente de Control de Peaje.

Basado directamente en la metodología de IAM-811 (keypoints_llantas_yolo_detectron2_v3.ipynb):
  - M1: Detección de placas multi-ángulo (YOLO / BlackHat+Sobel) + deskew + EasyOCR
  - M2: Clasificación y segmentación abierta de vehículos con YOLOE-11l-seg (o peso custom):
        distingue con precisión Auto particular, Minibús, Motocicleta, Camión y Bus.
  - M3: Detección de llantas (YOLOE prompts 'wheel', 'tire', 'car wheel' o YOLO-pose entrenado)
        -> 5 keypoints geométricos por llanta (centro, contacto, superior, izquierda, derecha)
        -> descarte de aros duplicados (UMBRAL_CONTENCION=0.7) y llantas lejanas (UMBRAL_CERCANA=0.65)
        -> conteo real de ejes por agrupación en X (SEPARACION_EJE=0.6 * diámetro) y detección de doble llanta.
  - M4 (Bonificación +5): Segmentación y conteo de pasajeros estrictamente dentro de cabina
        (sin transeúntes y sin recortes de rostros).
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

PIPELINE_VERSION = "3.0-yoloe-keypoints"

# Clases oficiales y etiquetas legibles para mostrar el tipo exacto de vehículo
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

# Colores BGR por clase de vehículo
COLORES_BGR = {
    "auto": (219, 152, 52),
    "bus": (60, 76, 231),
    "minibus": (113, 204, 46),
    "camion": (18, 156, 243),
    "moto": (220, 110, 190),
}

# Prompts de texto YOLOE -> categoría oficial del peaje
PROMPTS_VEHICULO = {
    "car": "auto",
    "sedan": "auto",
    "suv": "auto",
    "pickup truck": "auto",
    "minibus": "minibus",
    "van": "minibus",
    "passenger van": "minibus",
    "bus": "bus",
    "coach bus": "bus",
    "truck": "camion",
    "cargo truck": "camion",
    "semi truck": "camion",
    "motorcycle": "moto",
    "motorbike": "moto",
}

PROMPTS_LLANTA = ["wheel", "tire", "car wheel"]
PROMPTS_PERSONA = ["person"]

# Definición de los 5 keypoints geométricos por llanta (notebook v3)
KPT_NOMBRES = ["centro", "contacto", "superior", "izquierda", "derecha"]
KPT_COLORES_BGR = [
    (255, 255, 255),  # 0: centro
    (0, 0, 255),      # 1: contacto
    (255, 200, 0),    # 2: superior
    (0, 255, 0),      # 3: izquierda
    (255, 0, 255),    # 4: derecha
]
ESQUELETO = [(0, 1), (0, 2), (0, 3), (0, 4)]
UMBRAL_KP = 0.05

# Constantes geométricas de conteo de ejes (notebook v3)
UMBRAL_CONTENCION = 0.70
UMBRAL_CERCANA = 0.65
SEPARACION_EJE = 0.60
IMGSZ_YOLOE = 1024
CONF_VEHICULO = 0.22
CONF_LLANTA = 0.14

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


def nms_simple(items: List[Dict[str, Any]], umbral: float = 0.6) -> List[Dict[str, Any]]:
    ordered = sorted(items, key=lambda d: -d["conf"])
    res: List[Dict[str, Any]] = []
    for d in ordered:
        if all(iou_box(d["box"], r["box"]) < umbral for r in res):
            res.append(d)
    return res


def kps_de_mascara(mask: np.ndarray) -> Optional[np.ndarray]:
    """5 keypoints geométricos a partir del contorno de la llanta (notebook v3)."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 12:
        return None
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
    return np.concatenate([k, np.ones((5, 1), dtype=float)], axis=1)


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


def analizar_ejes_vehiculo(
    vehiculo_box: List[float],
    categoria: str,
    llantas_asociadas: List[Dict[str, Any]],
) -> Tuple[int, str, str, List[Dict[str, Any]]]:
    """
    Algoritmo del notebook keypoints_llantas_yolo_detectron2_v3.ipynb adaptado a peaje:
      - Filtra llantas del lado cercano (altura >= 0.65 * hmax).
      - En vista lateral / 3/4 agrupa centros X con SEPARACION_EJE * diámetro para contar ejes reales.
      - En vista frontal/posterior detecta doble llanta trasera en camiones/buses o asigna ejes base por clase.
    """
    default_axles = DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)
    if not llantas_asociadas:
        return default_axles, "frontal/sin_llantas", "Estimado por categoría (llantas no visibles)", []

    hmax = max(altura_visible(l) for l in llantas_asociadas)
    cercanas = [l for l in llantas_asociadas if altura_visible(l) >= UMBRAL_CERCANA * hmax]
    if not cercanas:
        cercanas = llantas_asociadas

    x1, y1, x2, y2 = vehiculo_box
    proporcion = (x2 - x1) / max(1.0, y2 - y1)

    # Detectar vista frontal / posterior (llantas vistas de canto a la misma altura Y)
    if proporcion < 1.65 and len(cercanas) >= 2:
        ys = [l["kps"][1, 1] if len(l["kps"]) >= 2 else l["box"][3] for l in cercanas]
        misma_altura = (max(ys) - min(ys)) < 0.18 * hmax
        angostas = all((ancho_visible(l) / altura_visible(l)) < 0.72 for l in cercanas)
        if angostas and misma_altura:
            # Si es camión/bus con 4+ llantas traseras visibles (2 dobles a cada lado) -> 2 ejes pesados
            if categoria in ("camion", "bus") and len(cercanas) >= 4:
                return 2, "frontal/posterior", "Doble llanta detectada (2 por lado)", cercanas
            return default_axles, "frontal/posterior", "Vista frontal/posterior (ejes base por clase)", cercanas

    xs = sorted(float(l["kps"][0, 0]) for l in cercanas)
    diam = float(np.median([altura_visible(l) for l in cercanas]))
    ejes = 1 + sum(1 for p, q in zip(xs, xs[1:]) if (q - p) > SEPARACION_EJE * diam)
    redondez = float(np.median([ancho_visible(l) / altura_visible(l) for l in cercanas]))
    vista = "lateral" if redondez > 0.80 else "3/4"
    nota = ""
    if ejes < 2:
        ejes, nota = 2, "Mínimo asumido (solo se detectó 1 eje visible)"
    return ejes, vista, nota, cercanas


class TollInferencePipeline:
    """Pipeline unificado M1+OCR, M2 (YOLOE/YOLO-seg), M3 (llantas 5-keypoints -> ejes) y M4 (pasajeros)."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu = os.environ.get("OCR_GPU", "true").lower() == "true" and self.device == "cuda"

        self.m1_path = os.environ.get("MODEL_M1_PLATES_PATH", "models/m1_placas.pt")
        self.m2_path = os.environ.get("MODEL_M2_VEHICLE_SEG_PATH", "models/m2_vehiculos_seg.pt")
        self.m3_path = os.environ.get("MODEL_M3_WHEEL_POSE_PATH", "models/m3_ruedas_pose.pt")
        self.m4_path = os.environ.get("MODEL_M4_PASSENGERS_SEG_PATH", "models/m4_pasajeros_seg.pt")
        self.yoloe_weight = os.environ.get("MODEL_YOLOE_WEIGHT", "yoloe-11l-seg.pt")

        self.model_status: Dict[str, Dict[str, Any]] = {}

        # Modelos custom entrenados (si existen)
        self.m1_plates: Optional[YOLO] = self._load_custom_yolo("M1_placas", self.m1_path)
        self.m2_custom_seg: Optional[YOLO] = self._load_custom_yolo("M2_vehiculos_seg", self.m2_path)
        self.m3_wheel_pose: Optional[YOLO] = self._load_custom_yolo("M3_ruedas_pose", self.m3_path)
        self.m4_custom_seg: Optional[YOLO] = self._load_custom_yolo("M4_pasajeros_seg", self.m4_path)

        # Modelo unificado YOLOE open-vocabulary (vehículos exactos + llantas + personas)
        self.yoloe_model = self._load_yoloe()
        self.fallback_coco_seg: Optional[YOLO] = None
        if self.yoloe_model is None:
            try:
                self.fallback_coco_seg = YOLO("yolo11n-seg.pt")
            except Exception as exc:
                logger.error("No se pudo cargar fallback yolo11n-seg.pt: %s", exc)

        self.ocr_reader = self._load_ocr()

    def _load_custom_yolo(self, key: str, path: str) -> Optional[YOLO]:
        if os.path.isfile(path):
            logger.info("Cargando modelo custom %s desde %s", key, path)
            model = YOLO(path)
            self.model_status[key] = {"loaded": True, "custom": True, "path": path, "device": self.device}
            return model
        self.model_status[key] = {"loaded": False, "custom": False, "path": path, "device": self.device}
        return None

    def _load_yoloe(self):
        if not HAS_YOLOE:
            self.model_status["YOLOE_unificado"] = {"loaded": False, "error": "YOLOE no disponible en esta versión de ultralytics"}
            return None
        try:
            nombres = list(PROMPTS_VEHICULO.keys()) + PROMPTS_LLANTA + PROMPTS_PERSONA
            logger.info("Inicializando YOLOE (%s) con prompts de vehículos, llantas y personas...", self.yoloe_weight)
            m = YOLOE(self.yoloe_weight)
            m.set_classes(nombres, m.get_text_pe(nombres))
            self.model_status["YOLOE_unificado"] = {
                "loaded": True,
                "custom": True,
                "path": self.yoloe_weight,
                "device": self.device,
                "prompts": nombres,
            }
            # Marcar M2, M3 y M4 como respaldados por YOLOE cuando aún no hay .pt propio
            for key in ("M2_vehiculos_seg", "M3_ruedas_pose", "M4_pasajeros_seg"):
                if not self.model_status[key]["loaded"]:
                    self.model_status[key] = {
                        "loaded": True,
                        "custom": False,
                        "engine": f"YOLOE ({self.yoloe_weight})",
                        "device": self.device,
                    }
            return m
        except Exception as exc:
            logger.error("Error inicializando YOLOE (%s): %s", self.yoloe_weight, exc)
            self.model_status["YOLOE_unificado"] = {"loaded": False, "error": str(exc)}
            return None

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
            "yoloe_active": self.yoloe_model is not None,
            "models": self.model_status,
        }

    # ------------------------------------------------------------------
    # M1: Detección de placas + OCR multi-ángulo
    # ------------------------------------------------------------------
    def _preprocess_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if w < 300:
            scale = max(2.2, 320.0 / max(w, 1))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        denoise = cv2.bilateralFilter(enhanced, 5, 40, 40)
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        return cv2.filter2D(denoise, -1, kernel)

    def _extract_plate_from_ocr_text(self, raw_text: str) -> str:
        cleaned_full = normalize_plate(raw_text)
        if BO_PLATE_RE.match(cleaned_full):
            return cleaned_full
        m_bo = re.search(r"\d{3,4}[A-Z]{3}", cleaned_full)
        if m_bo:
            return m_bo.group(0)
        m_alt = re.search(r"[A-Z]{2,3}\d{3,4}[A-Z]?", cleaned_full)
        if m_alt:
            return m_alt.group(0)
        return cleaned_full if len(cleaned_full) <= 8 else cleaned_full[:8]

    def _run_ocr_single(self, img: np.ndarray) -> Tuple[str, str, float]:
        if self.ocr_reader is None or img is None or img.size == 0:
            return "", "", 0.0
        processed = self._preprocess_plate_crop(img)
        results = self.ocr_reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
            detail=1,
            paragraph=False,
            text_threshold=0.45,
            low_text=0.28,
            contrast_ths=0.1,
            adjust_contrast=0.7,
        )
        if not results:
            return "", "", 0.0

        best = ("", "", 0.0, 0.0)
        results_sorted = sorted(results, key=lambda item: item[0][0][0])
        parts: List[str] = []
        confs: List[float] = []
        for _box, text, conf in results_sorted:
            extracted = self._extract_plate_from_ocr_text(text)
            if not extracted:
                continue
            sc = plate_quality_score(extracted, float(conf))
            if sc > best[3]:
                best = (extracted, text.strip(), float(conf), sc)
            parts.append(text.strip())
            confs.append(float(conf))

        if parts:
            raw_joined = " ".join(parts)
            extracted_joined = self._extract_plate_from_ocr_text(raw_joined)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            sc_joined = plate_quality_score(extracted_joined, avg_conf)
            if sc_joined >= best[3]:
                best = (extracted_joined, raw_joined, avg_conf, sc_joined)

        return best[0], best[1], round(best[2], 4)

    def _find_plate_candidates_cv(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
        candidates: List[Tuple[np.ndarray, List[int]]] = []
        seen_boxes: List[List[int]] = []
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)

        rect_k = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_k)
        grad_x = np.absolute(cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1))
        max_v = float(np.max(grad_x)) if grad_x.size else 0.0
        grad_x = ((255.0 * (grad_x / max_v)).astype("uint8") if max_v > 0 else np.zeros_like(gray))
        grad_x = cv2.morphologyEx(cv2.GaussianBlur(grad_x, (5, 5), 0), cv2.MORPH_CLOSE, rect_k)
        _, thresh_bh = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

        edges = cv2.Canny(cv2.bilateralFilter(gray, 9, 50, 50), 35, 150)
        closed_canny = cv2.morphologyEx(
            edges, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 3))
        )

        all_contours = []
        for mask in (thresh_bh, closed_canny):
            cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            all_contours.extend(sorted(cnts, key=cv2.contourArea, reverse=True)[:18])
        all_contours = sorted(all_contours, key=cv2.contourArea, reverse=True)

        for cnt in all_contours:
            area = cv2.contourArea(cnt)
            if area < (h * w) * 0.0015 or area > (h * w) * 0.35:
                continue
            (_rcx, _rcy), (rw, rh), angle = cv2.minAreaRect(cnt)
            if rw < 1 or rh < 1:
                continue
            ratio = max(rw, rh) / min(rw, rh)
            if not (1.35 <= ratio <= 5.8):
                continue
            x, y, bw, bh = cv2.boundingRect(cnt)
            pad_x, pad_y = int(bw * 0.16), int(bh * 0.22)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            if any(abs(x1 - sx1) < 18 and abs(y1 - sy1) < 18 for sx1, sy1, _, _ in seen_boxes):
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

    def _run_multi_angle_ocr(self, crop: np.ndarray) -> Tuple[str, str, float]:
        best = ("", "", 0.0, 0.0)
        for angle in (0.0, -12.0, 12.0, -22.0, 22.0):
            rotated = rotate_image_bound(crop, angle)
            norm, raw, conf = self._run_ocr_single(rotated)
            sc = plate_quality_score(norm, conf)
            if sc > best[3]:
                best = (norm, raw, conf, sc)
            if BO_PLATE_RE.match(norm) and conf >= 0.72:
                break
        return best[0], best[1], best[2]

    def infer_plate_only(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        bbox: Optional[List[int]] = None
        det_conf = 0.0
        best_norm, best_raw, best_ocr_conf, best_score = "", "", 0.0, 0.0

        if self.m1_plates is not None:
            preds = self.m1_plates.predict(source=image_bgr, conf=0.20, device=self.device, verbose=False)
            if preds and len(preds[0].boxes) > 0:
                best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                box = preds[0].boxes[best_idx]
                det_conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                pad_x, pad_y = int((x2 - x1) * 0.08), int((y2 - y1) * 0.12)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                bbox = [x1, y1, x2, y2]
                norm, raw, conf = self._run_multi_angle_ocr(image_bgr[y1:y2, x1:x2])
                sc = plate_quality_score(norm, conf)
                if sc > best_score:
                    best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc

        if best_score < 0.90:
            roi = image_bgr
            ox, oy = 0, 0
            if vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                ry1 = vy1 + int((vy2 - vy1) * 0.35)
                roi = image_bgr[ry1:vy2, vx1:vx2]
                ox, oy = vx1, ry1
            if roi.size > 0:
                for cand_crop, cbox in self._find_plate_candidates_cv(roi):
                    norm, raw, conf = self._run_multi_angle_ocr(cand_crop)
                    sc = plate_quality_score(norm, conf)
                    if sc > best_score:
                        best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                        bbox = [cbox[0] + ox, cbox[1] + oy, cbox[2] + ox, cbox[3] + oy]
                    if BO_PLATE_RE.match(best_norm) and best_ocr_conf >= 0.72:
                        break
                if best_score < 0.85:
                    norm, raw, conf = self._run_multi_angle_ocr(roi)
                    sc = plate_quality_score(norm, conf)
                    if sc > best_score:
                        best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                        if bbox is None and vehicle_bbox is not None:
                            bbox = [ox, oy, ox + roi.shape[1], oy + roi.shape[0]]

        if best_score < 0.65:
            norm, raw, conf = self._run_multi_angle_ocr(image_bgr)
            sc = plate_quality_score(norm, conf)
            if sc > best_score:
                best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc

        final_conf = round(best_ocr_conf if det_conf == 0 else (0.3 * det_conf + 0.7 * best_ocr_conf), 4)
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
    # Análisis unificado YOLOE (M2 Vehículo + M3 Llantas 5-KPs + Personas)
    # ------------------------------------------------------------------
    def _analyze_scene_yoloe(
        self, image_bgr: np.ndarray
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
        H, W = image_bgr.shape[:2]
        vehiculos: List[Dict[str, Any]] = []
        llantas: List[Dict[str, Any]] = []
        personas: List[Dict[str, Any]] = []

        if self.yoloe_model is not None:
            r = self.yoloe_model.predict(
                image_bgr,
                conf=CONF_LLANTA,
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
                    elif nombre in PROMPTS_LLANTA and float(s) >= CONF_LLANTA and poly_int is not None:
                        m = np.zeros((H, W), np.uint8)
                        cv2.fillPoly(m, [poly_int], 1)
                        kps = kps_de_mascara(m)
                        if kps is not None:
                            kps[:, 2] = float(s)
                            llantas.append({"conf": float(s), "box": box_list, "kps": kps})
                    elif nombre in PROMPTS_PERSONA and float(s) >= 0.25:
                        personas.append({"conf": float(s), "box": box_list, "poly": poly_int})

        elif self.fallback_coco_seg is not None:
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

        # Si existe M3 YOLO-pose custom entrenado, priorizar sus llantas sobre zero-shot
        if self.m3_wheel_pose is not None:
            rp = self.m3_wheel_pose.predict(image_bgr, conf=0.25, device=self.device, verbose=False)[0]
            if rp.keypoints is not None and len(rp.boxes) > 0:
                llantas = []
                xy = rp.keypoints.xy.cpu().numpy()
                kc = rp.keypoints.conf.cpu().numpy() if rp.keypoints.conf is not None else np.ones(xy.shape[:2])
                for b, s, p, c in zip(rp.boxes.xyxy.cpu().numpy(), rp.boxes.conf.cpu().numpy(), xy, kc):
                    kps = np.concatenate([p, c[:, None]], axis=1)
                    llantas.append({"conf": float(s), "box": b.tolist(), "kps": kps})

        return nms_simple(vehiculos, 0.65), limpiar_llantas(nms_simple(llantas, 0.50)), nms_simple(personas, 0.55)

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
            cv2.fillPoly(vehicle_mask_bin, [vehicle_poly], 1)
        else:
            cv2.rectangle(vehicle_mask_bin, (vx1, vy1), (vx2, vy2), 1, -1)

        if categoria == "moto":
            win_x1, win_x2 = vx1 + int(bw * 0.05), vx2 - int(bw * 0.05)
            win_y1, win_y2 = max(0, vy1 - int(bh * 0.25)), vy2
        else:
            win_x1, win_x2 = vx1 + int(bw * 0.10), vx2 - int(bw * 0.10)
            win_y1, win_y2 = vy1 + int(bh * 0.06), vy1 + int(bh * 0.62)

        def _mask_cov(gx1: int, gy1: int, gx2: int, gy2: int) -> float:
            cx1, cy1 = max(0, gx1), max(0, gy1)
            cx2, cy2 = min(img_w, gx2), min(img_h, gy2)
            if cx2 <= cx1 or cy2 <= cy1:
                return 0.0
            return float(np.mean(vehicle_mask_bin[cy1:cy2, cx1:cx2]))

        def _is_pedestrian(gx1: int, gy1: int, gx2: int, gy2: int) -> bool:
            if categoria == "moto":
                return _mask_cov(gx1, gy1, gx2, gy2) < 0.30
            pw, ph = max(gx2 - gx1, 1), max(gy2 - gy1, 1)
            if gy2 > vy1 + bh * 0.68:
                return True
            if ph > bh * 0.50:
                return True
            if gy1 < vy1 + bh * 0.02:
                return True
            if gx1 < vx1 + bw * 0.08 or gx2 > vx2 - bw * 0.08:
                return True
            if _mask_cov(gx1, gy1, gx2, gy2) < 0.82:
                return True
            if pw > bw * 0.48:
                return True
            return False

        passenger_masks: List[np.ndarray] = []
        centers_seen: List[Tuple[float, float]] = []

        for p in personas_globales:
            gx1, gy1, gx2, gy2 = [int(v) for v in p["box"]]
            if _is_pedestrian(gx1, gy1, gx2, gy2) or p["conf"] < 0.32:
                continue
            cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
            if categoria != "moto" and not (win_x1 <= cx <= win_x2 and win_y1 <= cy <= win_y2):
                continue
            if any(math.hypot(cx - ox, cy - oy) < 28.0 for ox, oy in centers_seen):
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
    # Dibujo de evidencia estilo notebook v3 (vehículo + llantas 5-KPs + placa + pasajeros)
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

        cv2.addWeighted(overlay, 0.28, out, 0.72, 0, out)

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

        # Llantas + 5 keypoints + esqueleto en cruz (exacto al notebook v3)
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

        vehiculos, llantas, personas = self._analyze_scene_yoloe(image_bgr)

        # Seleccionar el vehículo principal (mayor score área * confianza en la caseta)
        principal: Optional[Dict[str, Any]] = None
        if vehiculos:
            principal = max(
                vehiculos,
                key=lambda v: v["conf"] * (((v["box"][2] - v["box"][0]) * (v["box"][3] - v["box"][1])) / img_area),
            )

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

            # Asociar llantas al vehículo principal (centro en parte baja del vehículo, igual que notebook v3)
            vx1, vy1, vx2, vy2 = principal["box"]
            vw, vh = vx2 - vx1, vy2 - vy1
            llantas_veh = []
            for l in llantas:
                cx, cy = float(l["kps"][0, 0]), float(l["kps"][0, 1])
                if (vx1 - 0.05 * vw) <= cx <= (vx2 + 0.05 * vw) and (vy1 + 0.35 * vh) <= cy <= (vy2 + 0.10 * vh):
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

        # M1 + OCR sobre la placa del vehículo principal
        m1_out = self.infer_plate_only(image_bgr, vehicle_bbox=v_box_int)
        placa_info = m1_out["placa"]

        # M4: Pasajeros estrictamente en cabina
        pasajeros, passenger_masks = self._filter_cabin_passengers(image_bgr, principal, personas)

        # Lista de centros de llantas para JSON
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
