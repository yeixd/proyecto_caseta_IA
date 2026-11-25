/*
 * Sistema Inteligente de Control de Peaje - Dispositivo de Alerta en Hardware (ESP32)
 * Sección 10 (Bonificación +5): Conexión de alerta a un dispositivo de hardware.
 *
 * Señaliza en tiempo real vía WebSocket seguro (WSS) desde el backend en la nube:
 *   - IDENTIFICANDO / INFERENCIA : LED Azul parpadeando (proceso en curso)
 *   - APROBADO                   : LED Verde (3s) + 1 pitido corto
 *   - SIN_SALDO                  : LED Amarillo (4s) + 2 pitidos medios
 *   - ALERTA_NO_REGISTRADO /
 *     ALERTA_INACTIVO            : LED Rojo parpadeando (5s) + alarma buzzer
 *   - TEST                       : Secuencia de prueba de todos los LEDs + buzzer
 *
 * Librerías requeridas en Arduino IDE:
 *   - WiFi (integrada en ESP32 core)
 *   - WebSockets by Markus Sattler (WebSocketsClient.h)
 *   - ArduinoJson by Benoit Blanchon (v7)
 */

#include <WiFi.h>
#include <WebSocketsClient.h>
#include <ArduinoJson.h>

// ===================== CONFIGURACIÓN (NO SUBIR CREDENCIALES REALES) =====================
const char* WIFI_SSID     = "TU_RED_WIFI";
const char* WIFI_PASSWORD = "TU_PASSWORD_WIFI";

// Subdominio HTTPS del backend en la nube (SIN https://)
const char* SERVER_HOST   = "peaje.tudominio.com";
const uint16_t SERVER_PORT = 443;
const char* HARDWARE_KEY  = "GENERAR_TOKEN_ESP32_SEGURO";

// Pines GPIO del ESP32
const int PIN_LED_AZUL     = 18; // Proceso de identificación / inferencia
const int PIN_LED_VERDE    = 19; // Paso aprobado
const int PIN_LED_AMARILLO = 21; // Sin saldo suficiente
const int PIN_LED_ROJO     = 22; // Alerta: placa no registrada o vehículo inactivo
const int PIN_BUZZER       = 23; // Buzzer activo 3.3V/5V

WebSocketsClient webSocket;
unsigned long lastPingMs = 0;

void allLedsOff() {
  digitalWrite(PIN_LED_AZUL, LOW);
  digitalWrite(PIN_LED_VERDE, LOW);
  digitalWrite(PIN_LED_AMARILLO, LOW);
  digitalWrite(PIN_LED_ROJO, LOW);
  digitalWrite(PIN_BUZZER, LOW);
}

void beep(int times, int onMs, int offMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(PIN_BUZZER, HIGH);
    delay(onMs);
    digitalWrite(PIN_BUZZER, LOW);
    if (i < times - 1) delay(offMs);
  }
}

void handleSignal(const String& tipo, const String& placa, const String& detalle) {
  Serial.println("[HARDWARE] Evento: " + tipo + " | Placa: " + placa + " | " + detalle);
  allLedsOff();

  if (tipo == "IDENTIFICANDO" || tipo == "INFERENCIA") {
    for (int i = 0; i < 4; i++) {
      digitalWrite(PIN_LED_AZUL, HIGH);
      delay(120);
      digitalWrite(PIN_LED_AZUL, LOW);
      delay(100);
    }
    digitalWrite(PIN_LED_AZUL, HIGH);
  } else if (tipo == "APROBADO") {
    digitalWrite(PIN_LED_VERDE, HIGH);
    beep(1, 140, 80);
    delay(2500);
    allLedsOff();
  } else if (tipo == "SIN_SALDO") {
    digitalWrite(PIN_LED_AMARILLO, HIGH);
    beep(2, 220, 150);
    delay(3200);
    allLedsOff();
  } else if (tipo == "ALERTA_NO_REGISTRADO" || tipo == "ALERTA_INACTIVO") {
    for (int i = 0; i < 6; i++) {
      digitalWrite(PIN_LED_ROJO, HIGH);
      digitalWrite(PIN_BUZZER, HIGH);
      delay(220);
      digitalWrite(PIN_LED_ROJO, LOW);
      digitalWrite(PIN_BUZZER, LOW);
      delay(160);
    }
    allLedsOff();
  } else if (tipo == "TEST") {
    int pins[4] = {PIN_LED_AZUL, PIN_LED_VERDE, PIN_LED_AMARILLO, PIN_LED_ROJO};
    for (int i = 0; i < 4; i++) {
      digitalWrite(pins[i], HIGH);
      beep(1, 70, 50);
      delay(250);
      digitalWrite(pins[i], LOW);
    }
  }
}

void webSocketEvent(WStype_t type, uint8_t* payload, size_t length) {
  switch (type) {
    case WStype_DISCONNECTED:
      Serial.println("[WS] Desconectado del servidor en la nube. Reintentando...");
      break;
    case WStype_CONNECTED:
      Serial.println("[WS] Conectado al servidor de peaje en la nube!");
      handleSignal("TEST", "ESP32", "Conectado");
      break;
    case WStype_TEXT: {
      JsonDocument doc;
      DeserializationError err = deserializeJson(doc, payload, length);
      if (err) {
        Serial.println("[WS] Error parseando JSON");
        return;
      }
      const char* ev = doc["event"] | "";
      if (strcmp(ev, "hardware_signal") == 0) {
        String tipo    = doc["tipo"] | "";
        String placa   = doc["placa"] | "";
        String detalle = doc["detalle"] | "";
        handleSignal(tipo, placa, detalle);
      }
      break;
    }
    default:
      break;
  }
}

void setup() {
  Serial.begin(115200);
  pinMode(PIN_LED_AZUL, OUTPUT);
  pinMode(PIN_LED_VERDE, OUTPUT);
  pinMode(PIN_LED_AMARILLO, OUTPUT);
  pinMode(PIN_LED_ROJO, OUTPUT);
  pinMode(PIN_BUZZER, OUTPUT);
  allLedsOff();

  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Conectando a Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    digitalWrite(PIN_LED_AZUL, !digitalRead(PIN_LED_AZUL));
    delay(300);
    Serial.print(".");
  }
  digitalWrite(PIN_LED_AZUL, LOW);
  Serial.println("\nWi-Fi conectado. IP: " + WiFi.localIP().toString());

  String wsPath = String("/ws/hardware?key=") + HARDWARE_KEY;
  webSocket.beginSSL(SERVER_HOST, SERVER_PORT, wsPath.c_str());
  webSocket.onEvent(webSocketEvent);
  webSocket.setReconnectInterval(4000);
}

void loop() {
  webSocket.loop();
  if (millis() - lastPingMs > 25000) {
    lastPingMs = millis();
    webSocket.sendTXT("ping");
  }
}
