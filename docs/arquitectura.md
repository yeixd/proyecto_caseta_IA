# Diagramas de Arquitectura — Sistema Inteligente de Control de Peaje

## 1. Flujo de Secuencia: Captura en Modo Caseta -> Tiempo Real -> Hardware

```mermaid
sequenceDiagram
    participant App as App Móvil (Modo Caseta)
    participant API as Backend FastAPI (Subdominio HTTPS)
    participant Inf as Inferencia RunPod/Colab (INFERENCE_URL)
    participant DB as PostgreSQL
    participant Web as Panel Web (WebSocket /ws/panel)
    participant ESP as Hardware ESP32 (WebSocket /ws/hardware)

    App->>API: POST /api/v1/pasos/frame (JWT + Frame JPEG + caseta_id)
    API-->>ESP: Señal INFERENCIA (LED Azul)
    API->>Inf: POST /infer (M1+OCR, M2-seg, M3-pose, M4-pasajeros)
    Inf-->>API: JSON (placa, categoría, ejes, pasajeros, latencia, imagen anotada)
    API->>DB: Consulta vehículo, tarifa, descuenta saldo, inserta paso y alerta
    API-->>Web: WS broadcast nuevo_paso + alerta (sin recargar página)
    API-->>ESP: Señal APROBADO / SIN_SALDO / ALERTA_NO_REGISTRADO (LED + Buzzer)
    API-->>App: JSON con placa, categoría, ejes, pasajeros, tarifa y estado
```

## 2. Diagrama Entidad-Relación (Base de Datos)

```mermaid
erDiagram
    USUARIOS ||--o{ PASOS : registra
    CASETAS ||--o{ PASOS : captura
    PASOS ||--o{ ALERTAS : genera

    USUARIOS {
        int id PK
        string email_google UK
        string nombre
        string rol
        boolean activo
    }
    VEHICULOS {
        int id PK
        string placa UK
        string propietario
        string categoria
        int ejes
        float saldo
        boolean activo
    }
    CASETAS {
        int id PK
        string nombre UK
        string ubicacion
    }
    TARIFAS {
        int id PK
        string categoria
        int ejes
        float monto
    }
    PASOS {
        int id PK
        string placa_leida
        string placa_normalizada
        float confianza_ocr
        string categoria_detectada
        int ejes_detectados
        int pasajeros
        float monto
        int caseta_id FK
        int usuario_id FK
        string imagen_url
        string estado
        float latencia_ms
        datetime created_at
    }
    ALERTAS {
        int id PK
        int paso_id FK
        string tipo
        string mensaje
        boolean atendida
        datetime created_at
    }
```

## 3. Flujo del Agente Conversacional (Texto o Imagen)

```mermaid
flowchart LR
    U["Usuario Autenticado\n(Web o App Móvil)"] --> Choice{"¿Texto o Imagen?"}
    Choice -- Imagen --> M1["INFERENCE_URL /infer/placa\n(M1 + EasyOCR)"]
    M1 --> Leg{"¿Placa legible?"}
    Leg -- No --> Ask["Indica placa ilegible\ny solicita otra foto"]
    Leg -- Sí --> Norm["Normalización + Validación\n(MAYÚSCULAS, sin guiones/espacios)"]
    Choice -- Texto --> Norm
    Norm --> BD["Consulta BD Real\n(vehiculos + pasos)"]
    BD --> State{"Clasificación en 4 Estados"}
    State --> S1["Registrada y con paso"]
    State --> S2["Registrada sin paso"]
    State --> S3["Identificada no registrada\n+ Sugerencias OCR"]
    State --> S4["No existe\n+ Sugerencias OCR (O↔0, I↔1, B↔8, S↔5)"]
    S1 & S2 & S3 & S4 --> LLM["Ollama en RunPod\n(solo datos verificados BD)\no Fallback Determinista"]
```
