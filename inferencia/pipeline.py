"""
Pipeline unificado de inferencia para el Sistema Inteligente de Control de Peaje.

Integra 4 modelos en un único endpoint:
  - M1: Detección de placas (YOLO) + OCR (EasyOCR) + normalización
  - M2: Segmentación de instancias de vehículos (YOLO-seg) -> categoría y tarifa
  - M3: Keypoints sobre ruedas (YOLO-pose) -> estimación del número de ejes
  - M4 (Bonificación +5): Segmentación de pasajeros visibles dentro del vehículo
        (SOLO máscara y conteo numérico; NO reconocimiento facial ni recortes de rostro)
"""

from __future__ import annotations

import base64
import logging
import os
import re
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
from ultralytics import YOLO

logger = logging.getLogger("peaje.inferencia")

# Categorías válidas del sistema de peaje (Sección 4.7)
VALID_CATEGORIES = {"moto", "auto", "minibus", "camion", "bus"}

# Mapeo flexible desde nombres COCO o etiquetas en español/inglés hacia categorías del peaje
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

# Palabras de encabezado comunes en placas (Bolivia y región) ordenadas de mayor a menor longitud
IGNORED_OCR_TOKENS = ("ARGENTINA", "MERCOSUR", "BOLIVIA", "CHILE", "PLACA", "PERU", "BOL")


def normalize_plate(raw_text: str) -> str:
    """Normaliza una placa: mayúsculas, elimina encabezados y caracteres no alfanuméricos."""
    if not raw_text:
        return ""
    text = raw_text.upper().strip()
    for token in IGNORED_OCR_TOKENS:
        text = re.sub(rf"\b{token}\b", "", text)
    cleaned = re.sub(r"[^A-Z0-9]", "", text)
    return cleaned


def map_vehicle_category(raw_label: str) -> Optional[str]:
    """Convierte la etiqueta de clase del modelo a una de las 5 categorías oficiales."""
    if not raw_label:
        return None
    key = raw_label.strip().lower()
    return CATEGORY_ALIASES.get(key)


class TollInferencePipeline:
    """Carga y ejecuta M1+OCR, M2 (seg), M3 (pose) y M4 (pasajeros seg)."""

    def __init__(self) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.ocr_gpu = os.environ.get("OCR_GPU", "true").lower() == "true" and self.device == "cuda"

        self.m1_path = os.environ.get("MODEL_M1_PLATES_PATH", "models/m1_placas.pt")
        self.m2_path = os.environ.get("MODEL_M2_VEHICLE_SEG_PATH", "models/m2_vehiculos_seg.pt")
        self.m3_path = os.environ.get("MODEL_M3_WHEEL_POSE_PATH", "models/m3_ruedas_pose.pt")
        self.m4_path = os.environ.get("MODEL_M4_PASSENGERS_SEG_PATH", "models/m4_pasajeros_seg.pt")

        self.model_status: Dict[str, Dict[str, Any]] = {}

        self.m1_plates: Optional[YOLO] = self._load_yolo(
            key="M1_placas",
            path=self.m1_path,
            fallback=None,  # Si no existe peso custom, se usa fallback heurístico + OCR directo
        )
        self.m2_vehicle_seg: Optional[YOLO] = self._load_yolo(
            key="M2_vehiculos_seg",
            path=self.m2_path,
            fallback="yolo11n-seg.pt",
        )
        self.m3_wheel_pose: Optional[YOLO] = self._load_yolo(
            key="M3_ruedas_pose",
            path=self.m3_path,
            fallback=None,
        )
        self.m4_passengers_seg: Optional[YOLO] = self._load_yolo(
            key="M4_pasajeros_seg",
            path=self.m4_path,
            fallback="yolo11n-seg.pt",
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
            logger.warning(
                "No se encontró %s en %s. Cargando fallback base %s (reemplazar con modelo entrenado en Colab).",
                key,
                path,
                fallback,
            )
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

        logger.warning("Modelo %s no encontrado en %s (sin fallback).", key, path)
        self.model_status[key] = {
            "loaded": False,
            "custom": False,
            "path": path,
            "warning": f"Archivo de pesos no encontrado en {path}",
            "device": self.device,
        }
        return None

    def _load_ocr(self):
        try:
            import easyocr

            logger.info("Inicializando EasyOCR (gpu=%s)", self.ocr_gpu)
            reader = easyocr.Reader(["es", "en"], gpu=self.ocr_gpu, verbose=False)
            self.model_status["OCR"] = {
                "loaded": True,
                "engine": "easyocr",
                "gpu": self.ocr_gpu,
            }
            return reader
        except Exception as exc:
            logger.error("No se pudo inicializar EasyOCR: %s", exc)
            self.model_status["OCR"] = {
                "loaded": False,
                "engine": "easyocr",
                "error": str(exc),
            }
            return None

    def health(self) -> Dict[str, Any]:
        return {
            "status": "ok",
            "device": self.device,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "models": self.model_status,
        }

    # ------------------------------------------------------------------
    # M1: Detección de placas + OCR
    # ------------------------------------------------------------------
    def _preprocess_plate_crop(self, crop: np.ndarray) -> np.ndarray:
        """Mejora el contraste del recorte de placa para aumentar la precisión OCR."""
        if crop is None or crop.size == 0:
            return crop
        h, w = crop.shape[:2]
        if w < 180:
            scale = max(2.0, 220.0 / max(w, 1))
            crop = cv2.resize(crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)
        return enhanced

    def _run_ocr(self, img: np.ndarray) -> Tuple[str, str, float]:
        """Ejecuta EasyOCR y retorna (texto_normalizado, texto_raw, confianza)."""
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

        # Ordenar de izquierda a derecha y filtrar tokens que sean solo encabezado ("BOLIVIA")
        results_sorted = sorted(results, key=lambda item: item[0][0][0])
        parts: List[str] = []
        confs: List[float] = []
        for _bbox, text, conf in results_sorted:
            norm = normalize_plate(text)
            if not norm:
                continue
            parts.append(text.strip())
            confs.append(float(conf))

        if not parts:
            return "", "", 0.0

        raw_joined = " ".join(parts)
        normalized = normalize_plate(raw_joined)
        avg_conf = round(float(np.mean(confs)), 4) if confs else 0.0
        return normalized, raw_joined, avg_conf

    def infer_plate_only(self, image_bgr: np.ndarray) -> Dict[str, Any]:
        """Ejecuta M1 (detección de placa) + OCR. Usado por /infer y /infer/placa."""
        t0 = time.perf_counter()
        h, w = image_bgr.shape[:2]
        bbox: Optional[List[int]] = None
        det_conf = 0.0
        crop = image_bgr

        if self.m1_plates is not None:
            preds = self.m1_plates.predict(
                source=image_bgr,
                conf=0.25,
                device=self.device,
                verbose=False,
            )
            if preds and len(preds[0].boxes) > 0:
                best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                box = preds[0].boxes[best_idx]
                det_conf = float(box.conf.item())
                x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
                # Pequeño margen alrededor de la placa
                pad_x = int((x2 - x1) * 0.05)
                pad_y = int((y2 - y1) * 0.08)
                x1 = max(0, x1 - pad_x)
                y1 = max(0, y1 - pad_y)
                x2 = min(w, x2 + pad_x)
                y2 = min(h, y2 + pad_y)
                bbox = [x1, y1, x2, y2]
                crop = image_bgr[y1:y2, x1:x2]

        texto, texto_raw, ocr_conf = self._run_ocr(crop)

        # Si el detector recortó pero el OCR falló, intentar sobre imagen completa (ej. foto cercana de placa)
        if not texto and bbox is not None:
            texto, texto_raw, ocr_conf = self._run_ocr(image_bgr)

        final_conf = round(ocr_conf if det_conf == 0 else (0.3 * det_conf + 0.7 * ocr_conf), 4)
        latency_ms = round((time.perf_counter() - t0) * 1000.0, 2)

        return {
            "placa": {
                "texto": texto,
                "texto_raw": texto_raw,
                "confianza": final_conf,
                "confianza_deteccion": round(det_conf, 4),
                "confianza_ocr": round(ocr_conf, 4),
                "bbox": bbox,
                "legible": len(texto) >= 4,
            },
            "latencia_ms": latency_ms,
        }

    # ------------------------------------------------------------------
    # M2: Segmentación de instancias (categoría del vehículo)
    # ------------------------------------------------------------------
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
        }
        if self.m2_vehicle_seg is None:
            return default_res, None, None

        preds = self.m2_vehicle_seg.predict(
            source=image_bgr,
            conf=0.25,
            device=self.device,
            verbose=False,
        )
        if not preds or len(preds[0].boxes) == 0:
            return default_res, None, None

        res = preds[0]
        names = res.names
        best_candidate = None
        best_score = -1.0

        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            raw_name = names.get(cls_id, str(cls_id))
            mapped = map_vehicle_category(raw_name)
            if mapped is None:
                continue
            conf = float(res.boxes.conf[i].item())
            x1, y1, x2, y2 = [int(v) for v in res.boxes.xyxy[i].tolist()]
            box_area = max((x2 - x1) * (y2 - y1), 1)
            # Priorizar el vehículo más grande y con alta confianza (vehículo principal en caseta)
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
                best_candidate = (
                    {
                        "categoria": mapped,
                        "confianza": round(conf, 4),
                        "area_mascara": area_ratio,
                        "bbox": [x1, y1, x2, y2],
                    },
                    mask_poly,
                    [x1, y1, x2, y2],
                )

        if best_candidate is None:
            return default_res, None, None
        return best_candidate

    # ------------------------------------------------------------------
    # M3: Keypoints sobre ruedas -> conteo de ejes
    # ------------------------------------------------------------------
    def _infer_wheel_keypoints(
        self, image_bgr: np.ndarray, categoria: str
    ) -> Tuple[int, List[List[float]]]:
        kp_list: List[List[float]] = []

        if self.m3_wheel_pose is not None:
            preds = self.m3_wheel_pose.predict(
                source=image_bgr,
                conf=0.25,
                device=self.device,
                verbose=False,
            )
            if preds and preds[0].keypoints is not None and len(preds[0].keypoints.data) > 0:
                # Seleccionar la detección con mayor confianza de caja
                best_idx = 0
                if preds[0].boxes is not None and len(preds[0].boxes) > 0:
                    best_idx = int(torch.argmax(preds[0].boxes.conf).item())
                kpts = preds[0].keypoints.data[best_idx].cpu().numpy()
                for kp in kpts:
                    x, y = float(kp[0]), float(kp[1])
                    conf = float(kp[2]) if len(kp) > 2 else 1.0
                    if conf >= 0.35 and (x > 1.0 or y > 1.0):
                        kp_list.append([round(x, 1), round(y, 1), round(conf, 4)])

        if len(kp_list) >= 2:
            # 1 keypoint = centro de rueda visible en un costado => # ejes
            ejes = len(kp_list)
        else:
            ejes = DEFAULT_AXLES_BY_CATEGORY.get(categoria, 2)

        return ejes, kp_list

    # ------------------------------------------------------------------
    # M4 (Bonificación +5): Segmentación y conteo de pasajeros visibles
    # Privacidad: NO se realiza reconocimiento facial ni recortes de rostros.
    # Solo se genera la máscara semántica/instancia y el conteo numérico.
    # ------------------------------------------------------------------
    def _infer_passengers_seg(
        self,
        image_bgr: np.ndarray,
        vehicle_bbox: Optional[List[int]],
    ) -> Tuple[int, List[np.ndarray]]:
        if self.m4_passengers_seg is None:
            return 0, []

        preds = self.m4_passengers_seg.predict(
            source=image_bgr,
            conf=0.30,
            device=self.device,
            verbose=False,
        )
        if not preds or len(preds[0].boxes) == 0:
            return 0, []

        res = preds[0]
        names = res.names
        passenger_masks: List[np.ndarray] = []

        for i in range(len(res.boxes)):
            cls_id = int(res.boxes.cls[i].item())
            label = str(names.get(cls_id, "")).lower()
            if label not in {"person", "persona", "pasajero", "passenger", "driver", "conductor"}:
                continue

            px1, py1, px2, py2 = [int(v) for v in res.boxes.xyxy[i].tolist()]
            cx, cy = (px1 + px2) / 2.0, (py1 + py2) / 2.0

            # Si tenemos bbox del vehículo, verificar que el pasajero esté dentro (con tolerancia)
            if vehicle_bbox is not None:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                margin_x = (vx2 - vx1) * 0.12
                margin_y = (vy2 - vy1) * 0.15
                if not (
                    (vx1 - margin_x) <= cx <= (vx2 + margin_x)
                    and (vy1 - margin_y) <= cy <= (vy2 + margin_y)
                ):
                    continue

            if res.masks is not None and i < len(res.masks.xy) and len(res.masks.xy[i]) >= 3:
                passenger_masks.append(res.masks.xy[i].astype(np.int32))
            else:
                rect_poly = np.array(
                    [[px1, py1], [px2, py1], [px2, py2], [px1, py2]], dtype=np.int32
                )
                passenger_masks.append(rect_poly)

        return len(passenger_masks), passenger_masks

    # ------------------------------------------------------------------
    # Anotación visual de evidencia (sin recortes de rostros)
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

        # 1. Máscara de vehículo (M2)
        if vehicle_poly is not None:
            cv2.fillPoly(overlay, [vehicle_poly], (235, 130, 50))
            cv2.polylines(canvas, [vehicle_poly], True, (255, 160, 60), 2)

        # 2. Máscaras opacas de pasajeros (M4 - protege privacidad sin mostrar rostro ni recorte)
        for p_poly in passenger_masks:
            cv2.fillPoly(overlay, [p_poly], (0, 200, 180))
            cv2.polylines(canvas, [p_poly], True, (0, 240, 210), 2)

        cv2.addWeighted(overlay, 0.35, canvas, 0.65, 0, canvas)

        # 3. Caja de vehículo
        v_bbox = vehiculo_info.get("bbox")
        if v_bbox:
            x1, y1, x2, y2 = v_bbox
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 160, 60), 2)
            label = f"{vehiculo_info['categoria']} {vehiculo_info['confianza']:.2f} | {ejes} ejes | {len(passenger_masks)} pasaj."
            cv2.putText(
                canvas,
                label,
                (x1, max(22, y1 - 8)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 200, 80),
                2,
                cv2.LINE_AA,
            )

        # 4. Caja de placa (M1)
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

        # 5. Keypoints de ruedas (M3)
        for idx, (kx, ky, _kc) in enumerate(keypoints, start=1):
            center = (int(kx), int(ky))
            cv2.circle(canvas, center, 8, (40, 60, 255), -1)
            cv2.circle(canvas, center, 10, (255, 255, 255), 2)
            cv2.putText(
                canvas,
                f"K{idx}",
                (center[0] - 12, center[1] - 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (60, 80, 255),
                2,
                cv2.LINE_AA,
            )

        # Redimensionar si es muy grande para ahorrar ancho de banda
        max_w = 960
        if canvas.shape[1] > max_w:
            ratio = max_w / float(canvas.shape[1])
            canvas = cv2.resize(
                canvas,
                (max_w, int(canvas.shape[0] * ratio)),
                interpolation=cv2.INTER_AREA,
            )

        ok, buffer = cv2.imencode(".jpg", canvas, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ok:
            return ""
        return base64.b64encode(buffer.tobytes()).decode("utf-8")

    # ------------------------------------------------------------------
    # Inferencia completa (M1 + OCR + M2 + M3 + M4)
    # ------------------------------------------------------------------
    def infer_full(self, image_bgr: np.ndarray, include_annotated: bool = True) -> Dict[str, Any]:
        t0 = time.perf_counter()

        # M1 + OCR
        m1_out = self.infer_plate_only(image_bgr)
        placa_info = m1_out["placa"]

        # M2: Segmentación de vehículos
        vehiculo_info, vehicle_poly, vehicle_bbox = self._infer_vehicle_seg(image_bgr)

        # M3: Keypoints de ruedas -> número de ejes
        ejes, keypoints = self._infer_wheel_keypoints(image_bgr, vehiculo_info["categoria"])

        # M4 (Bonus +5): Conteo de pasajeros por segmentación
        pasajeros, passenger_masks = self._infer_passengers_seg(image_bgr, vehicle_bbox)

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
            },
            "ejes": ejes,
            "keypoints": keypoints,
            "pasajeros": pasajeros,
            "latencia_ms": total_latency_ms,
            "imagen_anotada_b64": annotated_b64,
        }
