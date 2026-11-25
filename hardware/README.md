# Dispositivo Hardware de Alertas — ESP32 (Bonificación +5)

El dispositivo físico basado en **ESP32** se conecta por Wi-Fi directamente al servidor en la nube (`wss://peaje.tudominio.com/ws/hardware?key=<HARDWARE_API_KEY>`) y señaliza visual y sonoramente cada etapa del proceso de peaje.

## Materiales
- 1x Placa ESP32 DevKit v1
- 4x LEDs (Azul, Verde, Amarillo, Rojo) + 4 resistencias de 220 Ω
- 1x Buzzer activo (3.3V / 5V)
- 1x Protoboard y cables jumper

## Diagrama de Pines (Pinout)

| Componente | GPIO ESP32 | Función en el sistema |
|---|---|---|
| LED Azul (+ R 220Ω) | `GPIO 18` | Señala inicio de identificación / inferencia |
| LED Verde (+ R 220Ω) | `GPIO 19` | Paso **APROBADO** (vehículo registrado con saldo) |
| LED Amarillo (+ R 220Ω) | `GPIO 21` | Alerta **SIN_SALDO** (saldo insuficiente) |
| LED Rojo (+ R 220Ω) | `GPIO 22` | Alerta crítica **PLACA_NO_REGISTRADA** / inactivo |
| Buzzer Activo (+) | `GPIO 23` | Patrones sonoros diferenciados (1 pitido, 2 pitidos, sirena) |
| GND común | `GND` | Tierra compartida para LEDs y buzzer |

## Prueba desde el Panel Web
1. Compilar y cargar `esp32_alerta_peaje/esp32_alerta_peaje.ino` con Arduino IDE (instalar librerías `WebSockets` de Markus Sattler y `ArduinoJson`).
2. Abrir la página **Estado** del panel web (`https://peaje.tudominio.com`).
3. Verificar que el contador **ESP32 Conectados** marque `1`.
4. Probar cada señal con los botones `IDENTIFICANDO`, `APROBADO`, `SIN_SALDO`, `ALERTA_NO_REGISTRADO` y `TEST`, o capturar un vehículo real en Modo Caseta.
