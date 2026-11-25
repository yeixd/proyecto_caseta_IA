# Guía Paso a Paso de Despliegue 100% en la Nube (Sin Recursos Locales)

Esta guía asegura el cumplimiento estricto de la **Sección 5 (Restricción: sin recursos locales)** y las bonificaciones de la **Sección 10**.

---

## Paso 1: Google Cloud Console (OAuth 2.0)
1. Ir a [Google Cloud Console](https://console.cloud.google.com/) -> crear proyecto `Peaje-IA-UNIFRANZ`.
2. Configurar **Pantalla de consentimiento OAuth** (Externa, agregar correos de prueba o publicar en producción para que `doc.alvaro.pinto.ma@unifranz.edu.bo` pueda iniciar sesión sin restricciones).
3. Crear credencial **ID de cliente OAuth -> Aplicación web**:
   - Orígenes autorizados de JavaScript: `https://peaje.tudominio.com`
   - Copiar el Client ID en `GOOGLE_CLIENT_ID` y `VITE_GOOGLE_CLIENT_ID`.
4. Crear credencial **ID de cliente OAuth -> Android** (paquete `bo.edu.unifranz.peajeia` y huella SHA-1 obtenida con `eas credentials`). Copiar en `GOOGLE_ANDROID_CLIENT_ID` y `EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID`.

---

## Paso 2: Datasets en Roboflow y Entrenamiento en Google Colab
1. Crear proyectos en [Roboflow](https://roboflow.com/):
   - `vehiculos-peaje-seg` (Instance Segmentation, clases: `moto`, `auto`, `minibus`, `camion`, `bus`). Partir de Roboflow Universe y ampliar con imágenes propias.
   - `ruedas-ejes-pose` (Keypoint Detection, keypoints en el centro de cada rueda visible desde un costado). Ampliar con fotos propias.
   - `pasajeros-vehiculo-seg` (Instance Segmentation, clase `pasajero` dentro del vehículo).
2. Subir tu peso `m1_placas.pt` (de proyectos anteriores) a Google Drive: `/MyDrive/proyecto_peaje/models/m1_placas.pt`.
3. Abrir en Google Colab (GPU T4) los notebooks de `/entrenamiento`:
   - `01_validacion_M1_placas_OCR.ipynb`
   - `02_entrenamiento_M2_segmentacion_vehiculos.ipynb`
   - `03_entrenamiento_M3_keypoints_ruedas.ipynb`
   - `04_entrenamiento_M4_segmentacion_pasajeros.ipynb`
4. Guardar las salidas de métricas y la salida de `python entrenamiento/diagnostico_entorno.py` para el informe técnico y la carpeta en la nube.

---

## Paso 3: Despliegue de Inferencia + Ollama en RunPod (Bonificación +5)
1. Crear un **RunPod GPU Pod** (ej. RTX 3090 / RTX 4000 Ada / L4) con volumen persistente en `/workspace`.
2. Exponer los puertos HTTP: **`8000`** (inferencia FastAPI) y **`11434`** (Ollama).
3. Subir los 4 archivos `.pt` a `/workspace/models/`:
   - `m1_placas.pt`, `m2_vehiculos_seg.pt`, `m3_ruedas_pose.pt`, `m4_pasajeros_seg.pt`.
4. Clonar el repositorio en el Pod y levantar la inferencia:
   ```bash
   git clone https://github.com/yeixd/proyecto_caseta_IA.git /workspace/proyecto_caseta_IA
   cd /workspace/proyecto_caseta_IA/inferencia
   pip install -r requirements.txt
   python diagnostico_entorno.py > /workspace/diagnostico_runpod.txt
   export INFERENCE_API_KEY="TU_TOKEN_SEGURO"
   uvicorn main:app --host 0.0.0.0 --port 8000
   ```
5. En otra terminal del mismo Pod, levantar **Ollama**:
   ```bash
   curl -fsSL https://ollama.com/install.sh | sh
   OLLAMA_HOST=0.0.0.0:11434 ollama serve &
   ollama pull llama3.1:8b
   ```
6. Copiar las URLs HTTPS del proxy de RunPod:
   - `INFERENCE_URL=https://<POD_ID>-8000.proxy.runpod.net`
   - `OLLAMA_URL=https://<POD_ID>-11434.proxy.runpod.net`

*(Respaldo en Google Colab: si se necesita cambiar temporalmente a Colab, ejecutar `entrenamiento/05_servidor_inferencia_colab.ipynb` y actualizar solo `INFERENCE_URL` en el servidor web).*

---

## Paso 4: Despliegue Web + Base de Datos con Subdominio HTTPS

### Opción A (Más rápida y GRATIS — sin comprar dominio): Render.com Blueprint
1. Sube tu repositorio a GitHub.
2. Entra a [render.com](https://render.com) -> **New +** -> **Blueprint** -> conecta tu repositorio GitHub.
3. Render detectará automáticamente el archivo `render.yaml` y creará:
   - Base de datos PostgreSQL (`peaje-ia-db`)
   - Servicio web unificado con `Dockerfile.cloud` (Frontend React + Backend FastAPI + WebSockets + Swagger)
   - **Subdominio HTTPS gratuito real**, por ejemplo: `https://peaje-ia-web.onrender.com`
4. Completa las variables solicitadas en Render (`GOOGLE_CLIENT_ID`, `VITE_GOOGLE_CLIENT_ID`, `EXTRA_ADMIN_EMAILS`, `INFERENCE_URL`, `INFERENCE_API_KEY`, `OLLAMA_URL`) y espera a que termine el deploy.
5. Agrega tu URL real de Render (`https://peaje-ia-web.onrender.com`) en Google Cloud Console -> Orígenes autorizados de JavaScript.

### Opción B: VPS propio con Docker Compose y subdominio personalizado
1. Apuntar un registro DNS `A` de tu subdominio real a la IP pública de tu VPS (DigitalOcean, Hetzner, AWS EC2, Oracle Cloud, etc.).
2. En el VPS, instalar Docker y Docker Compose, clonar el repo y crear `.env` desde `.env.example`:
   ```bash
   cp .env.example .env
   nano .env  # Completar GOOGLE_CLIENT_ID, INFERENCE_URL, OLLAMA_URL, contraseñas y EXTRA_ADMIN_EMAILS
   docker compose up -d --build
   ```
3. Publicar por **HTTPS** con Caddy (certificado Let's Encrypt automático):
   ```caddyfile
   tu-subdominio-real.com {
       reverse_proxy 127.0.0.1:8080
   }
   ```
4. Verificar en tu URL real:
   - `https://<tu-subdominio-real>` (Login con Google)
   - `https://<tu-subdominio-real>/docs` (Swagger / OpenAPI)

---

## Paso 5: Compilación del APK de la App Móvil en la Nube (EAS Build)
1. Entrar a `/movil`, copiar `.env.example` a `.env` con `EXPO_PUBLIC_API_BASE_URL=https://peaje.tudominio.com` y los Client IDs de Google.
2. Instalar EAS CLI y compilar el APK en la nube:
   ```bash
   cd movil
   npm install
   npx eas login
   npx eas build -p android --profile preview
   ```
3. Descargar el archivo `.apk` generado, instalarlo en el teléfono Android y subir una copia del `.apk` a la carpeta en la nube de entregables.

---

## Paso 6: Hardware ESP32
1. Abrir `hardware/esp32_alerta_peaje/esp32_alerta_peaje.ino` en Arduino IDE.
2. Configurar `WIFI_SSID`, `WIFI_PASSWORD`, `SERVER_HOST` (`peaje.tudominio.com`) y `HARDWARE_KEY` (el mismo `HARDWARE_API_KEY` del `.env` del VPS).
3. Flashear la placa ESP32 y confirmar conexión en `https://peaje.tudominio.com` -> pestaña **Estado**.
