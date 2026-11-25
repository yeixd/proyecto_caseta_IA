import React, { useEffect, useRef, useState } from "react";
import { Camera, Pause, Play, RefreshCw, Video, VideoOff } from "lucide-react";
import { api } from "../api";

const FPS_MODES = [
  { label: "2 FPS", ms: 500 },
  { label: "3 FPS", ms: 333 },
  { label: "5 FPS", ms: 200 }
];

export default function CamaraCasetaPage({ casetas, onRefresh }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const inFlightRef = useRef(false);

  const [cameraActive, setCameraActive] = useState(false);
  const [facingMode, setFacingMode] = useState("environment");
  const [casetaId, setCasetaId] = useState(casetas[0]?.id || 1);
  const [fpsIdx, setFpsIdx] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [sending, setSending] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [error, setError] = useState("");

  async function startCamera(mode = facingMode) {
    setError("");
    stopCamera();
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: mode }, width: { ideal: 1280 }, height: { ideal: 720 } },
        audio: false
      });
      streamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }
      setCameraActive(true);
    } catch (err) {
      setError(
        "No se pudo acceder a la cámara del dispositivo. En celulares debes abrir la web por HTTPS (o usar la App Móvil APK) y conceder permisos de cámara."
      );
    }
  }

  function stopCamera() {
    setStreaming(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setCameraActive(false);
  }

  useEffect(() => {
    return () => stopCamera();
  }, []);

  async function captureAndAnalyze() {
    if (!cameraActive || !videoRef.current || !canvasRef.current || inFlightRef.current) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;

    inFlightRef.current = true;
    setSending(true);
    setError("");
    try {
      const canvas = canvasRef.current;
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

      const blob = await new Promise((resolve) => canvas.toBlob(resolve, "image/jpeg", 0.75));
      if (!blob) return;

      const fd = new FormData();
      fd.append("file", blob, `camara_${Date.now()}.jpg`);
      fd.append("caseta_id", String(casetaId || casetas[0]?.id || 1));

      const { data } = await api.post("/api/v1/pasos/frame", fd);
      setLastResult(data);
      setFramesSent((n) => n + 1);
      onRefresh?.();
    } catch (err) {
      setError(err?.response?.data?.detail || "Error enviando frame a la inferencia en la nube.");
    } finally {
      inFlightRef.current = false;
      setSending(false);
    }
  }

  useEffect(() => {
    if (!streaming || !cameraActive) return;
    const timer = setInterval(() => {
      captureAndAnalyze();
    }, FPS_MODES[fpsIdx].ms);
    return () => clearInterval(timer);
  }, [streaming, cameraActive, fpsIdx, casetaId]);

  const paso = lastResult?.paso;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white flex items-center gap-2">
            <Camera className="w-6 h-6 text-sky-400" />
            Modo Caseta — Cámara en Vivo (Celular / Web)
          </h2>
          <p className="text-sm text-slate-400">
            Captura frames desde la cámara de tu celular (2 a 5 fps) y envía a la nube para
            analizar placa, categoría, ejes y pasajeros en tiempo real
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-2">
          <select
            className="input-dark text-xs"
            value={casetaId}
            onChange={(e) => setCasetaId(Number(e.target.value))}
          >
            {casetas.map((c) => (
              <option key={c.id} value={c.id}>
                {c.nombre}
              </option>
            ))}
          </select>

          <select
            className="input-dark text-xs"
            value={fpsIdx}
            onChange={(e) => setFpsIdx(Number(e.target.value))}
          >
            {FPS_MODES.map((m, idx) => (
              <option key={m.label} value={idx}>
                {m.label}
              </option>
            ))}
          </select>

          {!cameraActive ? (
            <button onClick={() => startCamera(facingMode)} className="btn-primary text-xs flex items-center gap-1.5">
              <Video className="w-4 h-4" /> Activar Cámara
            </button>
          ) : (
            <>
              <button
                onClick={() => {
                  const next = facingMode === "environment" ? "user" : "environment";
                  setFacingMode(next);
                  startCamera(next);
                }}
                className="btn-secondary text-xs flex items-center gap-1.5"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                {facingMode === "environment" ? "Trasera" : "Frontal"}
              </button>
              <button onClick={stopCamera} className="btn-secondary text-xs flex items-center gap-1.5 text-red-300">
                <VideoOff className="w-4 h-4" /> Apagar
              </button>
            </>
          )}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/15 border border-red-500/40 rounded-xl px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card p-4 space-y-4">
          <div className="relative bg-black rounded-xl overflow-hidden aspect-video flex items-center justify-center">
            <video ref={videoRef} playsInline muted className="w-full h-full object-cover" />
            <canvas ref={canvasRef} className="hidden" />

            {!cameraActive && (
              <div className="absolute inset-0 flex flex-col items-center justify-center text-slate-400 gap-3 p-6 text-center">
                <Camera className="w-12 h-12 text-slate-600" />
                <p className="text-sm">
                  Pulsa <strong>Activar Cámara</strong> para apuntar al vehículo (placa y ruedas).
                </p>
              </div>
            )}

            {cameraActive && (
              <div className="pointer-events-none absolute inset-6 border-2 border-dashed border-sky-400/60 rounded-2xl flex items-end justify-center pb-2">
                <span className="bg-slate-950/75 text-sky-200 text-xs px-3 py-1 rounded-full">
                  Encuadra vehículo, ruedas y placa · Frames enviados: {framesSent}
                </span>
              </div>
            )}
          </div>

          <div className="flex flex-wrap gap-3">
            <button
              disabled={!cameraActive}
              onClick={() => setStreaming((s) => !s)}
              className={`flex-1 py-2.5 rounded-xl font-semibold text-sm flex items-center justify-center gap-2 transition disabled:opacity-40 ${
                streaming
                  ? "bg-red-500 hover:bg-red-400 text-white"
                  : "bg-sky-500 hover:bg-sky-400 text-slate-950"
              }`}
            >
              {streaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
              {streaming
                ? `Detener Análisis Continuo (${FPS_MODES[fpsIdx].label})`
                : `Iniciar Análisis Continuo (${FPS_MODES[fpsIdx].label})`}
            </button>

            <button
              disabled={!cameraActive || sending}
              onClick={captureAndAnalyze}
              className="btn-secondary text-sm px-5"
            >
              {sending ? "Analizando..." : "Capturar 1 Frame"}
            </button>
          </div>
        </div>

        {/* Resultado en vivo */}
        <div className="card p-5 space-y-4">
          <h3 className="font-semibold text-white">Último Análisis de Cámara</h3>
          {!paso ? (
            <p className="text-sm text-slate-500 py-8 text-center">
              Activa la cámara y captura un frame para ver aquí la placa, categoría, ejes,
              pasajeros y tarifa calculada.
            </p>
          ) : (
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <span className="font-mono text-2xl font-extrabold text-white">
                  {paso.placa_normalizada || "ILEGIBLE"}
                </span>
                <span
                  className={`text-xs font-bold px-2.5 py-1 rounded-full ${
                    paso.estado === "APROBADO"
                      ? "bg-emerald-500/20 text-emerald-300"
                      : "bg-red-500/20 text-red-300"
                  }`}
                >
                  {paso.estado}
                </span>
              </div>

              <div className="grid grid-cols-2 gap-3 text-sm">
                <InfoBox label="Categoría (M2)" value={paso.categoria_detectada} />
                <InfoBox label="Ejes (M3)" value={`${paso.ejes_detectados} ejes`} />
                <InfoBox label="Pasajeros (M4)" value={String(paso.pasajeros)} />
                <InfoBox label="Tarifa" value={`Bs. ${Number(paso.monto).toFixed(2)}`} />
              </div>

              <div className="text-xs text-slate-400 space-y-1 pt-2 border-t border-slate-800">
                <p>Confianza OCR (M1): {(paso.confianza_ocr * 100).toFixed(1)}%</p>
                <p>Latencia punta a punta: {Math.round(paso.latencia_ms)} ms</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function InfoBox({ label, value }) {
  return (
    <div className="bg-slate-950 border border-slate-800 rounded-xl p-3">
      <p className="text-[11px] text-slate-400">{label}</p>
      <p className="text-base font-bold text-white capitalize mt-0.5">{value}</p>
    </div>
  );
}
