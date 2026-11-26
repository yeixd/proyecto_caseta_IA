"""
Pipeline unificado de inferencia para el Sistema Inteligente de Control de Peaje.

Integra 4 modelos en un único endpoint con soporte MULTI-ÁNGULO (frontal, trasero,
lateral y 3/4 diagonal):
  - M1: Detección de placas (YOLO) + corrección de perspectiva/inclinación (deskew) + OCR (EasyOCR)
  - M2: Segmentación de instancias de vehículos (YOLO-seg) -> categoría y tarifa
  - M3: Keypoints sobre ruedas (YOLO-pose + fallback geométrico multi-ángulo) -> 1, 2, 3, 4, 5, 6+ ejes
  - M4 (Bonificación +5): Segmentación y conteo de pasajeros visibles dentro del vehículo
        (con zoom en cabina/parabrisas y filtro estricto dentro del vehículo;
         SOLO máscara y conteo numérico; NO reconocimiento facial ni recortes de rostro)
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

logger = logging.getLogger("peaje.inferencia")

VALID_CATEGORIES = {"moto", "auto", "minibus", "camion", "bus"}

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
    "vagoneta": "auto",
    "minibus": "minibus",
    "minibús": "minibus",
    "van": "minibus",
    "trufi": "minibus",
    "micro": "minibus",
    "camion": "camion",
    "camión": "camion",
    "truck": "camion",
    "trailer": "camion",
    "volqueta": "camion",
    "bus": "bus",
    "flota": "bus",
    "autobus": "bus",
    "autobús": "bus",
}

DEFAULT_AXLES_BY_CATEGORY = {
    "moto": 2,
    "auto": 2,
    "minibus": 2,
    "bus": 2,
    "camion": 3,
}

# Palabras de encabezado comunes en placas (ordenadas de mayor a menor longitud)
IGNORED_OCR_TOKENS = ("ARGENTINA", "MERCOSUR", "BOLIVIA", "CHILE", "PLACA", "PERU", "BOL")

# Patrones bolivianos y estándar para puntuar candidatos OCR multi-ángulo
BO_PLATE_RE = re.compile(r"^\d{3,4}[A-Z]{3}$")
STD_PLATE_RE = re.compile(r"^[A-Z0-9]{5,8}$")


def normalize_plate(raw_text: str) -> str:
    """Normaliza una placa: mayúsculas, elimina encabezados y caracteres no alfanuméricos."""
    if not raw_text:
        return ""
    text = raw_text.upper().strip()
    for token in IGNORED_OCR_TOKENS:
        text = re.sub(rf"\b{token}\b", "", text)
    return re.sub(r"[^A-Z0-9]", "", text)


def plate_quality_score(normalized: str, conf: float) -> float:
    """Puntúa un candidato OCR priorizando placas con formato válido (letras + dígitos)."""
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


def map_vehicle_category(raw_label: str) -> Optional[str]:
    if not raw_label:
        return None
    return CATEGORY_ALIASES.get(raw_label.strip().lower())


def rotate_image_bound(img: np.ndarray, angle_deg: float) -> np.ndarray:
    """Rota una imagen sin recortar esquinas (para placas inclinadas por ángulo de cámara)."""
    if abs(angle_deg) < 0.5:
        return img
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    mat = cv2.getRotationMatrix2D((cx, cy), angle_deg, 1.0)
    cos_a = abs(mat[0, 0])
    sin_a = abs(mat[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))
    mat[0, 2] += (new_w / 2.0) - cx
    mat[1, 2] += (new_h / 2.0) - cy
    return cv2.warpAffine(
        img,
        mat,
        (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )


class TollInferencePipeline:
    """Carga y ejecuta M1+OCR, M2 (seg), M3 (pose ruedas -> 1..6+ ejes) y M4 (pasajeros seg)."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu = os.environ.get("OCR_GPU", "true").lower() == "true" and self.device == "cuda"

        self.m1_path = os.environ.get("MODEL_M1_PLATES_PATH", "models/m1_placas.pt")
        self.m2_path = os.environ.get("MODEL_M2_VEHICLE_SEG_PATH", "models/m2_vehiculos_seg.pt")
        self.m3_path = os.environ.get("MODEL_M3_WHEEL_POSE_PATH", "models/m3_ruedas_pose.pt")
        self.m4_path = os.environ.get("MODEL_M4_PASSENGERS_SEG_PATH", "models/m4_pasajeros_seg.pt")

        self.model_status: Dict[str, Dict[str, Any]] = {}

        self.m1_plates: Optional[YOLO] = self._load_yolo("M1_placas", self.m1_path, fallback=None)
        self.m2_vehicle_seg: Optional[YOLO] = self._load_yolo(
            "M2_vehiculos_seg", self.m2_path, fallback="yolo11n-seg.pt"
        )
        self.m3_wheel_pose: Optional[YOLO] = self._load_yolo(
            "M3_ruedas_pose", self.m3_path, fallback=None
        )
        self.m4_passengers_seg: Optional[YOLO] = self._load_yolo(
            "M4_pasajeros_seg", self.m4_path, fallback="yolo11n-seg.pt"
        )
        self.ocr_reader = self._load_ocr()

    def _load_yolo(self, key: str, path: str, fallback: Optional[str]) -> Optional[YOLO]:
        if os.path.isfile(path):
            logger.info("Cargando modelo custom %s desde %s", key, path)
            model = YOLO(path)
            self.model_status[key] = {
                "loaded": True,
                "custom": True,
                "path": path,
                "device": self.device,
            }
            return model

        if fallback:
            logger.warning("Peso custom %s no encontrado en %s; cargando base %s.", key, path, fallback)
            try:
                model = YOLO(fallback)
                self.model_status[key] = {
                    "loaded": True,
                    "custom": False,
                    "path": fallback,
                    "warning": f"Usando peso base {fallback}; subir peso entrenado a {path}",
                    "device": self.device,
                }
                return model
            except Exception as exc:
                logger.error("Error cargando fallback %s para %s: %s", fallback, key, exc)

        self.model_status[key] = {
            "loaded": False,
            "custom": False,
            "path": path,
            "warning": f"Usando detector multi-ángulo OpenCV/heurístico hasta subir {path}",
            "device": self.device,
        }
        return None

    def _load_ocr(self):
        try:
            import easyocr

            logger.info("Inicializando EasyOCR (gpu=%s)", self.ocr_gpu)
            reader = easyocr.Reader(["es", "en"], gpu=self.ocr_gpu, verbose=False)
            self.model_status["OCR"] = {"loaded": True, "engine": "easyocr", "gpu": self.ocr_gpu}
            return reader
        except Exception as exc:
            logger.error("No se pudo inicializar EasyOCR: %s", exc)
            self.model_status["OCR"] = {"loaded": False, "engine": "easyocr", "error": str(exc)}
            return None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "multi_angle_enabled": True,
            "models": self.model_status,
        }

    # ------------------------------------------------------------------
    # M1: Detección de placas multi-ángulo + deskew + EasyOCR
    # ------------------------------------------------------------------
    def _preprocess_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if w < 220:
            scale = max(2.0, 260.0 / max(w, 1))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        # Ligero realce de bordes para placas tomadas en ángulo oblicuo
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
        sharp = cv2.filter2D(enhanced, -1, kernel)
        return sharp

    def _run_ocr_single(self, img: np.ndarray) -> Tuple[str, str, float]:
        if self.ocr_reader is None or img is None or img.size == 0:
            return "", "", 0.0
        processed = self._preprocess_plate_crop(img)
        results = self.ocr_reader.readtext(
            processed,
            allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
            detail=1,
            paragraph=False,
        )
        if not results:
            return "", "", 0.0

        # Evaluar tanto texto combinado como cada bloque individual (evita ruido de fondo en fotos amplias)
        best = ("", "", 0.0, 0.0)  # norm, raw, conf, score
        results_sorted = sorted(results, key=lambda item: item[0][0][0])

        parts: List[str] = []
        confs: List[float] = []
        for _box, text, conf in results_sorted:
            norm = normalize_plate(text)
            if not norm:
                continue
            sc = plate_quality_score(norm, float(conf))
            if sc > best[3]:
                best = (norm, text.strip(), float(conf), sc)
            parts.append(text.strip())
            confs.append(float(conf))

        if parts:
            raw_joined = " ".join(parts)
            norm_joined = normalize_plate(raw_joined)
            avg_conf = float(np.mean(confs)) if confs else 0.0
            sc_joined = plate_quality_score(norm_joined, avg_conf)
            if sc_joined >= best[3]:
                best = (norm_joined, raw_joined, avg_conf, sc_joined)

        return best[0], best[1], round(best[2], 4)

    def _find_plate_candidates_cv(self, image_bgr: np.ndarray) -> List[Tuple[np.ndarray, List[int]]]:
        """Localiza regiones rectangulares claras (placas inclinadas o en perspectiva 3/4) con OpenCV."""
        candidates: List[Tuple[np.ndarray, List[int]]] = []
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        blur = cv2.bilateralFilter(gray, 9, 50, 50)
        edges = cv2.Canny(blur, 40, 160)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:25]

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < (h * w) * 0.002 or area > (h * w) * 0.25:
                continue
            rect = cv2.minAreaRect(cnt)
            (rcx, rcy), (rw, rh), angle = rect
            if rw < 1 or rh < 1:
                continue
            long_s, short_s = max(rw, rh), min(rw, rh)
            ratio = long_s / short_s
            if not (1.5 <= ratio <= 5.5):
                continue

            x, y, bw, bh = cv2.boundingRect(cnt)
            pad_x = int(bw * 0.12)
            pad_y = int(bh * 0.18)
            x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
            x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)
            crop = image_bgr[y1:y2, x1:x2]
            if crop.size > 0:
                # Corregir ángulo si la placa está inclinada
                deskew_angle = angle if rw >= rh else angle + 90.0
                if -35.0 <= deskew_angle <= 35.0 and abs(deskew_angle) > 3.0:
                    crop = rotate_image_bound(crop, deskew_angle)
                candidates.append((crop, [x1, y1, x2, y2]))
            if len(candidates) >= 4:
                break
        return candidates

    def _run_multi_angle_ocr(self, crop: np.ndarray) -> Tuple[str, str, float]:
        """Prueba 0° y pequeñas correcciones de ángulo (-18°, -10°, +10°, +18°) para placas inclinadas."""
        best = ("", "", 0.0, 0.0)
        for angle in (0.0, -12.0, 12.0, -22.0, 22.0):
            rotated = rotate_image_bound(crop, angle)
            norm, raw, conf = self._run_ocr_single(rotated)
            sc = plate_quality_score(norm, conf)
            if sc > best[3]:
                best = (norm, raw, conf, sc)
            # Si ya tenemos una placa con formato boliviano y alta confianza, salir rápido
            if BO_PLATE_RE.match(norm) and conf >= 0.72:
                break
        return best[0], best[1], best[2]

    def infer_plate_only(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Ejecuta M1 (detección de placa multi-ángulo) + deskew + EasyOCR."""
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        bbox: Optional[List[int]] = None
        det_conf = 0.0
        best_norm, best_raw, best_ocr_conf, best_score = "", "", 0.0, 0.0

        # 1. Detector YOLO M1 entrenado (si existe)
        if self.m1_plates is not None:
            preds = self.m1_plates.predict(source=image_bgr, conf=0.20, device=self.device, verbose=False)
            if preds and len(preds[0].boxes) > 0:
                best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                box = preds[0].boxes[best_idx]
                det_conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                pad_x = int((x2 - x1) * 0.08)
                pad_y = int((y2 - y1) * 0.12)
                x1, y1 = max(0, x1 - pad_x), max(0, y1 - pad_y)
                x2, y2 = min(w, x2 + pad_x), min(h, y2 + pad_y)
                bbox = [x1, y1, x2, y2]
                norm, raw, conf = self._run_multi_angle_ocr(image_bgr[y1:y2, x1:x2])
                sc = plate_quality_score(norm, conf)
                if sc > best_score:
                    best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc

        # 2. Si aún no tenemos placa confiable, buscar en zona inferior del vehículo (parachoques) + candidatos OpenCV
        if best_score < 0.9:
            roi = image_bgr
            ox, oy = 0, 0
            if vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                # Zona inferior 60% del vehículo donde siempre va la placa (frontal, trasero o 3/4)
                ry1 = vy1 + int((vy2 - vy1) * 0.38)
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

                # Intento directo sobre la zona inferior del vehículo (placas inclinadas grandes)
                if best_score < 0.85:
                    norm, raw, conf = self._run_multi_angle_ocr(roi)
                    sc = plate_quality_score(norm, conf)
                    if sc > best_score:
                        best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc
                        if bbox is None and vehicle_bbox is not None:
                            bbox = [ox, oy, ox + roi.shape[1], oy + roi.shape[0]]

        # 3. Último recurso: imagen completa (fotos cercanas de la placa desde cualquier ángulo)
        if best_score < 0.65:
            norm, raw, conf = self._run_multi_angle_ocr(image_bgr)
            sc = plate_quality_score(norm, conf)
            if sc > best_score:
                best_norm, best_raw, best_ocr_conf, best_score = norm, raw, conf, sc

        final_conf = round(best_ocr_conf if det_conf == 0 else (0.3 * det_conf + 0.7 * best_ocr_conf), 4)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

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
            "latencia_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # M2: Segmentación de instancias multi-ángulo (categoría del vehículo)
    # ------------------------------------------------------------------
    def _estimate_view_angle(self, vehicle_bbox: Optional[List[int]]) -> str:
        """Estima la orientación relativa de la cámara respecto al vehículo."""
        if not vehicle_bbox:
            return "desconocido"
        vx1, vy1, vx2, vy2 = vehicle_bbox
        bw = max(vx2 - vx1, 1)
        bh = max(vy2 - vy1, 1)
        aspect = bw / float(bh)
        if aspect >= 1.65:
            return "lateral"
        if aspect >= 1.15:
            return "tres_cuartos"
        return "frontal_trasero"

    def _infer_vehicle_seg(
        self, image_bgr: np.ndarray
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[List[int]]]:
        h, w = image_bgr.shape[:2]
        img_area = float(h * w)
        default_res = {
            "categoria": "auto",
            "confianza": 0.0,
            "area_mascara": 0.0,
            "bbox": None,
            "angulo": "desconocido",
        }
        if self.m2_vehicle_seg is None:
            return default_res, None, None

        preds = self.m2_vehicle_seg.predict(
            source=image_bgr,
            conf=0.20,
            device=self.device,
            verbose=False,
        )
        if not preds or len(preds[0].boxes) == 0:
            return default_res, None, None

        res = preds[0]
        names = res.names
        best_candidate = None
        best_score = -1.0
        is_custom = bool(self.model_status.get("M2_vehiculos_seg", {}).get("custom"))

        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            raw_name = names.get(cls_id, str(cls_id))
            mapped = map_vehicle_category(raw_name)
            if mapped is None:
                continue
            conf = float(res.boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in res.boxes.xyxy[i].tolist()]
            box_area = max((x2 - x1) * (y2 - y1), 1)
            score = conf * (box_area / img_area)
            if score > best_score:
                best_score = score
                mask_poly = None
                area_ratio = round(box_area / img_area, 4)
                if res.masks is not None and i < len(res.masks.xy):
                    poly = res.masks.xy[i]
                    if len(poly) >= 3:
                        mask_poly = poly.astype(np.int32)
                        mask_area = cv2.contourArea(mask_poly)
                        area_ratio = round(float(mask_area / img_area), 4)

                # Si usamos COCO base (sin clase minibus), distinguir minibús alto de auto sedán
                final_cat = mapped
                if not is_custom and mapped in ("car", "auto", "bus"):
                    bh = max(y2 - y1, 1)
                    bw = max(x2 - x1, 1)
                    if mapped == "bus" and (box_area / img_area) < 0.45 and (bw / bh) < 1.5:
                        final_cat = "minibus"

                angle = self._estimate_view_angle([x1, y1, x2, y2])
                best_candidate = (
                    {
                        "categoria": final_cat,
                        "confianza": round(conf, 4),
                        "area_mascara": area_ratio,
                        "bbox": [x1, y1, x2, y2],
                        "angulo": angle,
                    },
                    mask_poly,
                    [x1, y1, x2, y2],
                )

        if best_candidate is None:
            return default_res, None, None
        return best_candidate

    # ------------------------------------------------------------------
    # M3: Keypoints sobre ruedas -> cálculo de 1, 2, 3, 4, 5, 6+ ejes
    # Soporta vista lateral, 3/4 y frontal/trasera
    # ------------------------------------------------------------------
    def _detect_wheels_cv_fallback(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]],
    ) -> List[List[float]]:
        """Fallback geométrico OpenCV para detectar ruedas visibles cuando M3 aún no tiene pesos custom."""
        if vehicle_bbox is None:
            return []
        vx1, vy1, vx2, vy2 = vehicle_bbox
        bw, bh = vx2 - vx1, vy2 - vy1
        if bw < 60 or bh < 60:
            return []

        # Ruedas se encuentran siempre en el 42% inferior del vehículo
        y_start = vy1 + int(bh * 0.58)
        roi = image_bgr[y_start:vy2, vx1:vx2]
        if roi.size == 0:
            return []

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 1.2)
        min_r = max(8, int(min(bw, bh) * 0.055))
        max_r = max(min_r + 6, int(min(bw, bh) * 0.24))

        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=max(min_r * 2, int(bw * 0.12)),
            param1=90,
            param2=28,
            minRadius=min_r,
            maxRadius=max_r,
        )
        found: List[List[float]] = []
        if circles is not None:
            pts = sorted(circles[0], key=lambda c: float(c[0]))
            for cx, cy, _r in pts[:6]:
                found.append([round(float(vx1 + cx), 1), round(float(y_start + cy), 1), 0.65])
        return found

    def _count_axles_from_keypoints(
        self,
        kp_list: List[List[float]],
        categoria: str,
        angulo: str,
        vehicle_bbox: Optional[List[int]],
    ) -> int:
        """
        Convierte keypoints de ruedas en número de ejes (1, 2, 3, 4, 5, 6+):
          - Vista lateral / 3/4: cada rueda en el mismo costado con separación en X corresponde a 1 eje.
          - Vista frontal/trasera pura: dos ruedas simétricas a igual altura Y pertenecen al mismo eje delantero/trasero.
          - Soporta 1 eje (remolque / monociclo / 1 rueda visible) hasta 6+ ejes.
        """
        if not kp_list:
            return DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)

        if len(kp_list) == 1:
            # Si solo se distingue 1 rueda/eje claramente
            return 1 if angulo == "lateral" else DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)

        # Agrupar keypoints demasiado cercanos (duplicados sobre la misma rueda)
        bw = (vehicle_bbox[2] - vehicle_bbox[0]) if vehicle_bbox else 400
        bh = (vehicle_bbox[3] - vehicle_bbox[1]) if vehicle_bbox else 300
        min_x_sep = max(14.0, bw * 0.06)

        sorted_kps = sorted(kp_list, key=lambda p: p[0])
        unique_wheels: List[List[float]] = []
        for kp in sorted_kps:
            if not unique_wheels or abs(kp[0] - unique_wheels[-1][0]) >= min_x_sep:
                unique_wheels.append(kp)

        if angulo == "frontal_trasero" and len(unique_wheels) == 2:
            # En ángulo puramente frontal/trasero las 2 ruedas a la misma altura Y son el mismo eje
            y_diff = abs(unique_wheels[0][1] - unique_wheels[1][1])
            if y_diff < bh * 0.10:
                return DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)

        return max(1, min(len(unique_wheels), 10))

    def _infer_wheel_keypoints(
        self,
        image_bgr: np.ndarray,
        categoria: str,
        angulo: str,
        vehicle_bbox: Optional[List[int]],
    ) -> Tuple[int, List[List[float]]]:
        kp_list: List[List[float]] = []

        if self.m3_wheel_pose is not None:
            preds = self.m3_wheel_pose.predict(
                source=image_bgr,
                conf=0.22,
                device=self.device,
                verbose=False,
            )
            if preds and preds[0].keypoints is not None and len(preds[0].keypoints.data) > 0:
                best_idx = 0
                if preds[0].boxes is not None and len(preds[0].boxes) > 0:
                    best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                kpts = preds[0].keypoints.data[best_idx].cpu().numpy()
                for kp in kpts:
                    x, y = float(kp[0]), float(kp[1])
                    conf = float(kp[2]) if len(kp) > 2 else 1.0
                    if conf >= 0.30 and (x > 1.0 or y > 1.0):
                        kp_list.append([round(x, 1), round(y, 1), round(conf, 4)])
        else:
            kp_list = self._detect_wheels_cv_fallback(image_bgr, vehicle_bbox)

        ejes = self._count_axles_from_keypoints(kp_list, categoria, angulo, vehicle_bbox)
        return ejes, kp_list

    # ------------------------------------------------------------------
    # M4 (Bonificación +5): Segmentación y conteo de pasajeros visibles
    # EXCLUSIVAMENTE DENTRO DE LA CABINA DEL VEHÍCULO (rechaza transeúntes
    # adelante, atrás o a los costados mediante máscara binaria M2 + lista negra
    # de peatones de cuerpo completo).
    # Privacidad: SOLO máscara y conteo, SIN reconocimiento facial ni recortes.
    # ------------------------------------------------------------------
    def _infer_passengers_seg(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]],
        vehicle_poly: Optional[np.ndarray],
        categoria: str = "auto",
    ) -> Tuple[int, List[np.ndarray]]:
        # Sin vehículo detectado NO se cuentan personas (evita contar transeúntes en la calle)
        if self.m4_passengers_seg is None or vehicle_bbox is None:
            return 0, []

        img_h, img_w = image_bgr.shape[:2]
        vx1, vy1, vx2, vy2 = vehicle_bbox
        bw = max(vx2 - vx1, 1)
        bh = max(vy2 - vy1, 1)

        # Máscara binaria del vehículo (M2) para verificar píxel a píxel si la persona está dentro
        vehicle_mask_bin = np.zeros((img_h, img_w), dtype=np.uint8)
        if vehicle_poly is not None and len(vehicle_poly) >= 3:
            cv2.fillPoly(vehicle_mask_bin, [vehicle_poly], 1)
        else:
            cv2.rectangle(vehicle_mask_bin, (vx1, vy1), (vx2, vy2), 1, -1)

        # Banda interior de ventanas/parabrisas para vehículos cerrados
        if categoria == "moto":
            win_x1 = vx1 + int(bw * 0.05)
            win_x2 = vx2 - int(bw * 0.05)
            win_y1 = max(0, vy1 - int(bh * 0.25))
            win_y2 = vy2
        else:
            win_x1 = vx1 + int(bw * 0.10)
            win_x2 = vx2 - int(bw * 0.10)
            win_y1 = vy1 + int(bh * 0.06)
            win_y2 = vy1 + int(bh * 0.62)

        def _mask_coverage(gx1: int, gy1: int, gx2: int, gy2: int) -> float:
            cx1, cy1 = max(0, gx1), max(0, gy1)
            cx2, cy2 = min(img_w, gx2), min(img_h, gy2)
            if cx2 <= cx1 or cy2 <= cy1:
                return 0.0
            roi = vehicle_mask_bin[cy1:cy2, cx1:cx2]
            return float(np.mean(roi))

        def _iou(box_a: Tuple[int, int, int, int], box_b: Tuple[int, int, int, int]) -> float:
            ix1, iy1 = max(box_a[0], box_b[0]), max(box_a[1], box_b[1])
            ix2, iy2 = min(box_a[2], box_b[2]), min(box_a[3], box_b[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            if inter <= 0:
                return 0.0
            area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
            area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
            return inter / float(min(area_a, area_b))

        def _is_exterior_pedestrian(gx1: int, gy1: int, gx2: int, gy2: int) -> bool:
            """Detecta si una persona en la imagen completa es un transeúnte fuera del vehículo."""
            if categoria == "moto":
                return _mask_coverage(gx1, gy1, gx2, gy2) < 0.30
            pw = max(gx2 - gx1, 1)
            ph = max(gy2 - gy1, 1)
            # Pies/base llegan a la parte baja de puertas/ruedas/suelo -> transeúnte de pie
            if gy2 > vy1 + bh * 0.68:
                return True
            # Altura demasiado grande respecto al vehículo -> persona parada adelante/atrás
            if ph > bh * 0.50:
                return True
            # Cabeza por encima del techo del vehículo -> persona detrás del vehículo
            if gy1 < vy1 + bh * 0.02:
                return True
            # Toca los bordes izquierdo o derecho del vehículo -> peatón al costado
            if gx1 < vx1 + bw * 0.08 or gx2 > vx2 - bw * 0.08:
                return True
            # Poca superposición con la máscara binaria del vehículo (ej. peatón ocluyendo por delante)
            if _mask_coverage(gx1, gy1, gx2, gy2) < 0.82:
                return True
            # Ancho excesivo para una sola persona sentada
            if pw > bw * 0.48:
                return True
            return False

        def _is_valid_cabin_passenger(
            gx1: int,
            gy1: int,
            gx2: int,
            gy2: int,
            blacklisted: List[Tuple[int, int, int, int]],
        ) -> bool:
            if _is_exterior_pedestrian(gx1, gy1, gx2, gy2):
                return False

            # Rechazar si coincide con un transeúnte de cuerpo completo detectado en la imagen global
            cand_box = (gx1, gy1, gx2, gy2)
            for ext_box in blacklisted:
                if _iou(cand_box, ext_box) >= 0.30:
                    return False

            if categoria != "moto":
                cx = (gx1 + gx2) / 2.0
                cy = (gy1 + gy2) / 2.0
                # Centro estrictamente dentro de la banda interior de ventanas/parabrisas
                if not (win_x1 <= cx <= win_x2 and win_y1 <= cy <= win_y2):
                    return False
                if vehicle_poly is not None and len(vehicle_poly) >= 3:
                    if cv2.pointPolygonTest(vehicle_poly, (float(cx), float(cy)), False) < 0.0:
                        return False
            return True

        passenger_masks: List[np.ndarray] = []
        centers_seen: List[Tuple[float, float]] = []
        exterior_blacklist: List[Tuple[int, int, int, int]] = []

        # PASADA 1 (Imagen completa PRIMERO): detectar transeúntes de cuerpo completo para bloquearlos
        preds_full = self.m4_passengers_seg.predict(
            source=image_bgr, conf=0.25, device=self.device, verbose=False
        )
        if preds_full and len(preds_full[0].boxes) > 0:
            res_f = preds_full[0]
            names_f = res_f.names
            for i in range(len(res_f.boxes)):
                cls_id = int(res_f.boxes.cls[i].item())
                label = str(names_f.get(cls_id, "")).lower()
                if label not in {"person", "persona", "pasajero", "passenger", "driver", "conductor"}:
                    continue
                conf_p = float(res_f.boxes.conf[i].item())
                gx1, gy1, gx2, gy2 = [int(v) for v in res_f.boxes.xyxy[i].tolist()]

                if _is_exterior_pedestrian(gx1, gy1, gx2, gy2):
                    exterior_blacklist.append((gx1, gy1, gx2, gy2))
                    continue

                if conf_p < 0.34:
                    continue

                cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
                if not _is_valid_cabin_passenger(gx1, gy1, gx2, gy2, exterior_blacklist):
                    continue
                if any(math.hypot(cx - ox, cy - oy) < 28.0 for ox, oy in centers_seen):
                    continue
                centers_seen.append((cx, cy))
                if res_f.masks is not None and i < len(res_f.masks.xy) and len(res_f.masks.xy[i]) >= 3:
                    passenger_masks.append(res_f.masks.xy[i].astype(np.int32))
                else:
                    passenger_masks.append(
                        np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.int32)
                    )

        # PASADA 2 (Zoom estricto en parabrisas/ventanas): solo agrega pasajeros NO bloqueados
        crop_y1, crop_y2 = max(0, win_y1), min(img_h, win_y2)
        crop_x1, crop_x2 = max(0, win_x1), min(img_w, win_x2)
        cabin_crop = image_bgr[crop_y1:crop_y2, crop_x1:crop_x2]
        if cabin_crop.size > 0 and cabin_crop.shape[1] >= 50 and cabin_crop.shape[0] >= 40:
            preds_cabin = self.m4_passengers_seg.predict(
                source=cabin_crop, conf=0.35, device=self.device, verbose=False
            )
            if preds_cabin and len(preds_cabin[0].boxes) > 0:
                res_c = preds_cabin[0]
                names_c = res_c.names
                for i in range(len(res_c.boxes)):
                    cls_id = int(res_c.boxes.cls[i].item())
                    label = str(names_c.get(cls_id, "")).lower()
                    if label not in {"person", "persona", "pasajero", "passenger", "driver", "conductor"}:
                        continue
                    px1, py1, px2, py2 = [int(v) for v in res_c.boxes.xyxy[i].tolist()]
                    gx1, gy1 = px1 + crop_x1, py1 + crop_y1
                    gx2, gy2 = px2 + crop_x1, py2 + crop_y1

                    if not _is_valid_cabin_passenger(gx1, gy1, gx2, gy2, exterior_blacklist):
                        continue

                    cx, cy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
                    if any(math.hypot(cx - ox, cy - oy) < 28.0 for ox, oy in centers_seen):
                        continue
                    centers_seen.append((cx, cy))

                    if res_c.masks is not None and i < len(res_c.masks.xy) and len(res_c.masks.xy[i]) >= 3:
                        poly = res_c.masks.xy[i].astype(np.int32)
                        poly[:, 0] += crop_x1
                        poly[:, 1] += crop_y1
                        passenger_masks.append(poly)
                    else:
                        passenger_masks.append(
                            np.array([[gx1, gy1], [gx2, gy1], [gx2, gy2], [gx1, gy2]], dtype=np.int32)
                        )

        return len(passenger_masks), passenger_masks

    # ------------------------------------------------------------------
    # Anotación visual de evidencia (solo máscaras/cajas/keypoints, SIN rostros)
    # ------------------------------------------------------------------
    def _annotate_evidence(
        self,
        image_bgr: np.ndarray,
        placa_info: Dict[str, Any],
        vehiculo_info: Dict[str, Any],
        vehicle_poly: Optional[np.ndarray],
        keypoints: List[List[float]],
        ejes: int,
        passenger_masks: List[np.ndarray],
    ) -> str:
        canvas = image_bgr.copy()
        overlay = image_bgr.copy()

        if vehicle_poly is not None:
            cv2.fillPoly(overlay, [vehicle_poly], (235, 130, 50))
            cv2.polylines(canvas, [vehicle_poly], True, (255, 160, 60), 2)

        # Máscaras opacas sólidas sobre pasajeros (privacidad estricta: sin rostros)
        for p_poly in passenger_masks:
            cv2.fillPoly(overlay, [p_poly], (0, 210, 190))
            cv2.polylines(canvas, [p_poly], True, (0, 245, 220), 2)

        cv2.addWeighted(overlay, 0.45, canvas, 0.55, 0, canvas)

        v_bbox = vehiculo_info.get("bbox")
        if v_bbox:
            x1, y1, x2, y2 = v_bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 60), 2)
            label = (
                f"{vehiculo_info['categoria']} {vehiculo_info['confianza']:.2f} | "
                f"{ejes} eje(s) | {len(passenger_masks)} pasaj. | {vehiculo_info.get('angulo', '')}"
            )
            cv2.putText(
                canvas,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 210, 80),
                2,
                cv2.LINE_AA,
            )

        p_bbox = placa_info.get("bbox")
        if p_bbox:
            px1, py1, px2, py2 = p_bbox
            cv2.rectangle(canvas, (px1, py1), (px2, py2), (40, 230, 100), 2)
            p_txt = placa_info.get("texto") or "PLACA"
            cv2.putText(
                canvas,
                f"{p_txt} ({placa_info.get('confianza', 0):.2f})",
                (px1, max(22, py1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (40, 230, 100),
                2,
                cv2.LINE_AA,
            )

        for idx, (kx, ky, _kc) in enumerate(keypoints, start=1):
            center = (int(kx), int(ky))
            cv2.circle(canvas, center, 8, (40, 60, 255), -1)
            cv2.circle(canvas, center, 10, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                f"E{idx}",
                (center[0] - 12, center[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 80, 255),
                2,
                cv2.LINE_AA,
            )

        max_w = 960
        if canvas.shape[1] > max_w:
            ratio = max_w / float(canvas.shape[1])
            canvas = cv2.resize(
                canvas, (max_w, int(canvas.shape[0] * ratio)), interpolation=cv2.INTER_AREA
            )

        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        return base64.b64encode(buffer.tobytes()).decode("utf-8") if ok else ""

    # ------------------------------------------------------------------
    # Inferencia completa (M2 -> M1+OCR multi-ángulo -> M3 ejes -> M4 pasajeros)
    # ------------------------------------------------------------------
    def infer_full(self, image_bgr: np.ndarray, include_annotated: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # M2 primero: conocer bbox/máscara del vehículo ayuda a M1 (parachoques), M3 (ruedas) y M4 (cabina)
        vehiculo_info, vehicle_poly, vehicle_bbox = self._infer_vehicle_seg(image_bgr)

        # M1 + OCR multi-ángulo
        m1_out = self.infer_plate_only(image_bgr, vehicle_bbox=vehicle_bbox)
        placa_info = m1_out["placa"]

        # M3: Keypoints ruedas -> 1, 2, 3, 4, 5, 6+ ejes
        ejes, keypoints = self._infer_wheel_keypoints(
            image_bgr,
            categoria=vehiculo_info["categoria"],
            angulo=vehiculo_info.get("angulo", "desconocido"),
            vehicle_bbox=vehicle_bbox,
        )

        # M4 (Bonus +5): Conteo de pasajeros estrictamente dentro del vehículo
        pasajeros, passenger_masks = self._infer_passengers_seg(
            image_bgr,
            vehicle_bbox=vehicle_bbox,
            vehicle_poly=vehicle_poly,
            categoria=vehiculo_info["categoria"],
        )

        annotated_b64 = ""
        if include_annotated:
            annotated_b64 = self._annotate_evidence(
                image_bgr=image_bgr,
                placa_info=placa_info,
                vehiculo_info=vehiculo_info,
                vehicle_poly=vehicle_poly,
                keypoints=keypoints,
                ejes=ejes,
                passenger_masks=passenger_masks,
            )

        total_latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "placa": placa_info,
            "vehiculo": {
                "categoria": vehiculo_info["categoria"],
                "confianza": vehiculo_info["confianza"],
                "area_mascara": vehiculo_info["area_mascara"],
                "angulo": vehiculo_info.get("angulo", "desconocido"),
            },
            "ejes": ejes,
            "keypoints": keypoints,
            "pasajeros": pasajeros,
            "latencia_ms": total_latency_ms,
            "imagen_anotada_b64": annotated_b64,
        }
