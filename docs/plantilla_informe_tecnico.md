# Informe Técnico — Sistema Inteligente de Control de Peaje

**Asignatura:** IAM-811 Machine Learning y Aprendizaje Profundo — UNIFRANZ  
**Docente:** Ing. Álvaro Pinto  
**Fecha:** 25/11/2026  
**Integrantes:** *(Completar nombres y roles)*  
**Subdominio en producción:** `https://peaje.tudominio.com`  
**Repositorio Git:** `https://github.com/yeixd/proyecto_caseta_IA`

---

## 1. Resumen Ejecutivo y Arquitectura del Sistema
Describir brevemente la arquitectura 100% en la nube:
- **App Móvil (React Native / Expo):** Inicio de sesión con Google OAuth 2.0, Modo Caseta (2-5 fps) y Modo Consulta.
- **Aplicación Web (FastAPI + React + PostgreSQL):** Publicada en subdominio HTTPS, lista blanca Google (`doc.alvaro.pinto.ma@unifranz.edu.bo` como administrador), API REST documentada en Swagger (`/docs`), actualización en tiempo real mediante **WebSockets** (`/ws/panel`), agente conversacional respaldado por **Ollama en RunPod** con motor determinista libre de alucinaciones.
- **Inferencia Unificada (`INFERENCE_URL`):** Desplegada en **RunPod** (con respaldo compatible en Google Colab), ejecutando M1+OCR, M2, M3 y M4 en un único endpoint HTTP.
- **Hardware de Alertas:** ESP32 conectado vía WebSocket seguro (`/ws/hardware`) con 4 LEDs de estado y buzzer.

*(Insertar diagrama de arquitectura desde `docs/arquitectura.md`)*

---

## 2. Datasets y Etiquetado en Roboflow

| Modelo | Enlace Roboflow | Total Imágenes | Imágenes Propias | Clases / Keypoints | Aumentaciones Aplicadas | Split (Train/Val/Test) |
|---|---|---:|---:|---|---|---|
| M1 (Placas - proyecto anterior) | *(link)* | *(N)* | *(N)* | `placa` | *(ej. brillo, rotación ±10°)* | 70/20/10 |
| M2 (Segmentación Vehículos) | *(link)* | *(N)* | *(N)* | `moto`, `auto`, `minibus`, `camion`, `bus` | *(flip horizontal, brillo, blur)* | 70/20/10 |
| M3 (Keypoints Ruedas) | *(link)* | *(N)* | *(N)* | Keypoints centro de rueda lateral | *(brillo, escala, contraste)* | 70/20/10 |
| M4 (Segmentación Pasajeros) | *(link)* | *(N)* | *(N)* | `pasajero` (solo máscara y conteo) | *(brillo, recorte parcial)* | 70/20/10 |

---

## 3. Entrenamiento y Métricas Obtenidas (Google Colab)

### 3.1 M1: Detección de Placas + OCR (EasyOCR)
*(Pegar resultados de `entrenamiento/01_validacion_M1_placas_OCR.ipynb`)*
- Exact Match (%): **XX.X %**
- Character Error Rate (CER): **0.0XXX**
- Latencia promedio M1 + OCR: **XX ms**

### 3.2 M2: Segmentación de Instancias de Vehículos (YOLO11-seg)
*(Pegar resultados de `entrenamiento/02_entrenamiento_M2_segmentacion_vehiculos.ipynb` y captura de matriz de confusión / curvas PR)*
- Box mAP@50: **0.XXXX** | Box mAP@50-95: **0.XXXX**
- Mask mAP@50: **0.XXXX** | Mask mAP@50-95: **0.XXXX**
- Precisión: **0.XXXX** | Recall: **0.XXXX**

### 3.3 M3: Keypoints sobre Ruedas y Conteo de Ejes (YOLO11-pose)
*(Pegar resultados de `entrenamiento/03_entrenamiento_M3_keypoints_ruedas.ipynb`)*
- Pose mAP@50: **0.XXXX** | Pose mAP@50-95: **0.XXXX**
- Error Absoluto Medio (MAE) en conteo de ejes: **0.XX ejes**
- Exactitud exacta en conteo de ejes (%): **XX.X %**

### 3.4 M4 (Bonificación +5): Segmentación y Conteo de Pasajeros
*(Pegar resultados de `entrenamiento/04_entrenamiento_M4_segmentacion_pasajeros.ipynb`)*
- Mask mAP@50: **0.XXXX** | Mask mAP@50-95: **0.XXXX**
- Privacidad verificada: las evidencias almacenan únicamente máscaras sobre los ocupantes y el conteo numérico; no se aplica reconocimiento facial ni se guardan recortes de rostros.

---

## 4. Latencia de Punta a Punta (End-to-End)

| Etapa | Tiempo promedio (ms) | Infraestructura |
|---|---:|---|
| Captura + subida JPEG desde App Móvil | *(ej. 120 ms)* | Red móvil 4G/Wi-Fi |
| Inferencia unificada M1+M2+M3+M4 en RunPod | *(ej. 280 ms)* | RunPod GPU (RTX/L4) |
| Lógica de tarifas, BD PostgreSQL y guardado de evidencia | *(ej. 45 ms)* | VPS Cloud |
| Emisión WebSocket a Panel Web y ESP32 | *(ej. 25 ms)* | WSS en tiempo real |
| **Latencia Total Punta a Punta** | ***(ej. 470 ms)*** | **100% Nube** |

---

## 5. Mecanismo de Tiempo Real y Hardware
- **Panel Web:** WebSockets nativos de FastAPI (`/ws/panel`) con autenticación por token JWT y reconexión automática. Cada paso y alerta se inserta en el DOM de React inmediatamente sin recargar la página.
- **Dispositivo Hardware (ESP32):** Conectado a `/ws/hardware` mediante WSS. Señaliza con LED azul (inferencia), verde (aprobado), amarillo (sin saldo) y rojo + buzzer (placa no registrada).

---

## 6. Gestión de Entornos Reproducibles (Bonificación +5)
Se incluyeron archivos con versiones fijadas (`requirements.txt` y `environment.yml`) para cada servicio (`entrenamiento/`, `inferencia/`, `web/backend/`) y se ejecutó `diagnostico_entorno.py` en los entornos de nube. *(Adjuntar capturas/logs en la carpeta en la nube)*.

---

## 7. Problemas Encontrados y Soluciones Aplicadas
1. **Confusiones de OCR en placas (`O↔0`, `I↔1`, `B↔8`, `S↔5`) y lectura de la palabra "BOLIVIA":** Se implementó preprocesamiento CLAHE, filtrado de encabezados (`BOLIVIA`), normalización estricta y búsqueda por distancia ponderada OCR en el agente conversacional.
2. **Saturación de peticiones en Modo Caseta:** Se agregó control de concurrencia (`inFlightRef`) en la app móvil para enviar entre 2 y 5 fps sin acumular frames pendientes.
3. **Dependencia de inferencia entre RunPod y Colab:** Desacoplada totalmente mediante la variable `INFERENCE_URL` y un contrato JSON único.

---

## 8. Conclusiones
*(Redactar conclusiones del equipo sobre precisión lograda, viabilidad en producción y aprendizajes del hito).*
