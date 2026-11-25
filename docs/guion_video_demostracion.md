# Guion del Video Demostración (8 a 12 minutos) — Sección 8 y Sección 10

> **Importante:** Todos los integrantes del grupo deben participar con cámara/voz. Verificar que ninguna URL sea `localhost` ni IP local.

## Checklist Minuto a Minuto

1. **Min 0:00 – 1:00 | Presentación del equipo y arquitectura (máx. 1 min)**
   - Presentación rápida de cada integrante y su rol.
   - Mostrar el diagrama de arquitectura y explicar el flujo 100% en la nube (`App Móvil -> Subdominio HTTPS -> RunPod INFERENCE_URL + Ollama -> Panel Web en Tiempo Real + ESP32`).

2. **Min 1:00 – 2:30 | Autenticación con Google y Lista Blanca (Web y App)**
   - En la web (`https://peaje.tudominio.com`), intentar iniciar sesión con una cuenta de Google **NO autorizada** -> mostrar el mensaje de **"Acceso denegado"**.
   - Iniciar sesión con una cuenta autorizada (Administrador) -> mostrar lista blanca con `doc.alvaro.pinto.ma@unifranz.edu.bo` habilitado.
   - Iniciar sesión con Google en la **App Móvil**.

3. **Min 2:30 – 4:00 | Consola de RunPod en vivo y Entornos (Bonificación +5 y +5)**
   - Mostrar la consola activa de **RunPod** con el Pod en ejecución y los logs recibiendo una petición real.
   - Mostrar la ejecución de `python diagnostico_entorno.py` (Python, PyTorch, CUDA, Ultralytics) y la página **Estado** en la web marcando `INFERENCE_URL: ONLINE`.

4. **Min 4:00 – 6:30 | Pantalla Dividida: App Móvil (Modo Caseta) + Panel Web en Tiempo Real + ESP32 + Pasajeros**
   - Colocar en **pantalla dividida** (o grabación simultánea): celular en **Modo Caseta** capturando video real de vehículos + **Panel Web en vivo** + cámara enfocando el **ESP32**.
   - **Caso 1 (Aprobado con varios pasajeros - Bonificación +5):** Vehículo registrado con saldo y varios pasajeros visibles -> mostrar en app (placa, categoría, ejes, pasajeros, tarifa, `APROBADO`), en el panel web apareciendo **al instante sin recargar** (con conteo de pasajeros y evidencia con máscara sin rostro), y el **ESP32 encendiendo LED azul -> verde + pitido**.
   - **Caso 2 (Alerta en tiempo real):** Capturar una **placa NO registrada** -> mostrar alerta apareciendo al instante en el Panel Web, en la sección Alertas y el **ESP32 activando LED rojo + sirena buzzer**.

5. **Min 6:30 – 9:00 | Agente Conversacional (Los 4 Estados + Texto e Imagen + Sugerencias OCR)**
   - Demostrar desde Web y/o App Móvil (Modo Consulta):
     1. **Registrada y con paso** (por **imagen** de placa -> muestra fecha, caseta, categoría y monto).
     2. **Registrada sin paso** (por **texto**, ej. una placa precargada que no pasó aún).
     3. **Identificada no registrada** (la placa no registrada capturada en el paso anterior -> posible evasión).
     4. **No existe** + prueba de error OCR (ej. escribir `248IKLM` o `2481KL0` -> el agente responde **"¿Quisiste decir `2481KLM`?"**).
     5. Imagen borrosa/sin placa -> el agente indica placa ilegible y pide otra foto.

6. **Min 9:00 – 10:30 | Conclusiones y Métricas Principales**
   - Mostrar métricas reales de M1+OCR, M2 (mAP seg), M3 (mAP pose y MAE ejes), M4 (mAP pasajeros) y latencia punta a punta.
   - Conclusiones finales de cierre por el equipo.
