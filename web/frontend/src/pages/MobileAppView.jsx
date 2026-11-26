import React, { useEffect, useRef, useState } from "react";
import {
  Bot,
  Camera,
  CheckCircle2,
  LogOut,
  Monitor,
  Pause,
  Play,
  RefreshCw,
  Send,
  Smartphone,
  Video,
  VideoOff,
  X
} from "lucide-react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";

const FPS_OPTIONS = [
  { label: "2 FPS", ms: 500 },
  { label: "3 FPS", ms: 333 },
  { label: "5 FPS", ms: 200 }
];

const STATE_BADGE = {
  "Registrada y con paso": "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  "Registrada sin paso": "bg-sky-500/20 text-sky-300 border-sky-500/40",
  "Identificada no registrada": "bg-red-500/20 text-red-300 border-red-500/40",
  "No existe": "bg-slate-700 text-slate-300 border-slate-600",
  "Placa ilegible": "bg-amber-500/20 text-amber-300 border-amber-500/40"
};

export default function MobileAppView({ casetas, wsConnected, onExitMobile, onRefresh }) {
  const { user, logout } = useAuth();
  const [tab, setTab] = useState("caseta");

  return (
    <div className="min-h-screen bg-slate-950 text-slate-100 flex flex-col max-w-md mx-auto border-x border-slate-900">
      {/* Header móvil */}
      <header className="sticky top-0 z-20 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-9 h-9 rounded-xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400 shrink-0">
            <Smartphone className="w-5 h-5" />
          </div>
          <div className="min-w-0">
            <h1 className="text-sm font-bold text-white leading-tight truncate">
              Peaje IA · App Móvil
            </h1>
            <div className="flex items-center gap-1.5 text-[11px] text-slate-400">
              <span
                className={`w-2 h-2 rounded-full ${
                  wsConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
                }`}
              />
              <span className="truncate">{user?.email_google}</span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={onExitMobile}
            title="Ir al Panel Web completo"
            className="p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-white"
          >
            <Monitor className="w-4 h-4" />
          </button>
          <button
            onClick={logout}
            title="Cerrar sesión"
            className="p-2 rounded-xl bg-slate-800 text-slate-300 hover:text-red-300"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Contenido principal */}
      <div className="flex-1 overflow-y-auto pb-20">
        {tab === "caseta" ? (
          <MobileCasetaTab casetas={casetas} onRefresh={onRefresh} />
        ) : (
          <MobileConsultaTab />
        )}
      </div>

      {/* Barra inferior estilo App Nativa */}
      <nav className="fixed bottom-0 left-0 right-0 max-w-md mx-auto bg-slate-900/95 backdrop-blur border-t border-slate-800 px-3 py-2 grid grid-cols-2 gap-2 z-20">
        <button
          onClick={() => setTab("caseta")}
          className={`flex flex-col items-center justify-center py-2 rounded-xl text-xs font-semibold transition ${
            tab === "caseta"
              ? "bg-sky-500 text-slate-950"
              : "text-slate-400 hover:bg-slate-800 hover:text-white"
          }`}
        >
          <Camera className="w-5 h-5 mb-0.5" />
          Modo Caseta
        </button>
        <button
          onClick={() => setTab("consulta")}
          className={`flex flex-col items-center justify-center py-2 rounded-xl text-xs font-semibold transition ${
            tab === "consulta"
              ? "bg-sky-500 text-slate-950"
              : "text-slate-400 hover:bg-slate-800 hover:text-white"
          }`}
        >
          <Bot className="w-5 h-5 mb-0.5" />
          Modo Consulta
        </button>
      </nav>
    </div>
  );
}

function MobileCasetaTab({ casetas, onRefresh }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const inFlightRef = useRef(false);
  const nativeCamRef = useRef(null);

  const [cameraOn, setCameraOn] = useState(false);
  const [facing, setFacing] = useState("environment");
  const [casetaId, setCasetaId] = useState(casetas[0]?.id || 1);

  useEffect(() => {
    if (casetas.length > 0 && !casetas.some((c) => c.id === casetaId)) {
      setCasetaId(casetas[0].id);
    }
  }, [casetas, casetaId]);
  const [fpsIdx, setFpsIdx] = useState(0);
  const [streaming, setStreaming] = useState(false);
  const [sending, setSending] = useState(false);
  const [framesSent, setFramesSent] = useState(0);
  const [lastResult, setLastResult] = useState(null);
  const [error, setError] = useState("");

  async function startCamera(mode = facing) {
    setError("");
    stopCamera();
    if (!navigator.mediaDevices?.getUserMedia) {
      setError(
        "El navegador requiere HTTPS para video continuo. Usa el botón 'Foto con Cámara Nativa' abajo o abre la app por HTTPS."
      );
      return;
    }
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
      setCameraOn(true);
    } catch {
      setError(
        "No se pudo activar el video en vivo (verifica permiso de cámara y conexión HTTPS). Puedes usar 'Foto con Cámara Nativa'."
      );
    }
  }

  function stopCamera() {
    setStreaming(false);
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop());
      streamRef.current = null;
    }
    if (videoRef.current) videoRef.current.srcObject = null;
    setCameraOn(false);
  }

  useEffect(() => () => stopCamera(), []);

  async function uploadFrameBlob(blob, filename = "frame.jpg") {
    inFlightRef.current = true;
    setSending(true);
    setError("");
    try {
      const fd = new FormData();
      fd.append("file", blob, filename);
      fd.append("caseta_id", String(casetaId || casetas[0]?.id || 1));
      const { data } = await api.post("/api/v1/pasos/frame", fd);
      setLastResult(data);
      setFramesSent((n) => n + 1);
      onRefresh?.();
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Error conectando con el backend o endpoint de inferencia (INFERENCE_URL)."
      );
    } finally {
      inFlightRef.current = false;
      setSending(false);
    }
  }

  async function captureFromVideo() {
    if (!cameraOn || !videoRef.current || !canvasRef.current || inFlightRef.current) return;
    const video = videoRef.current;
    if (!video.videoWidth || !video.videoHeight) return;
    const canvas = canvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.width, canvas.height);
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.72));
    if (blob) await uploadFrameBlob(blob, `mobile_${Date.now()}.jpg`);
  }

  async function handleNativePhoto(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadFrameBlob(file, file.name || "camara_movil.jpg");
    if (nativeCamRef.current) nativeCamRef.current.value = "";
  }

  useEffect(() => {
    if (!streaming || !cameraOn) return;
    const timer = setInterval(() => {
      captureFromVideo();
    }, FPS_OPTIONS[fpsIdx].ms);
    return () => clearInterval(timer);
  }, [streaming, cameraOn, fpsIdx, casetaId]);

  const paso = lastResult?.paso;

  return (
    <div className="p-4 space-y-4">
      {/* Visor cámara */}
      <div className="relative bg-black rounded-2xl overflow-hidden aspect-[4/3] border border-slate-800 flex items-center justify-center">
        <video ref={videoRef} playsInline muted className="w-full h-full object-cover" />
        <canvas ref={canvasRef} className="hidden" />

        {!cameraOn && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 p-5 text-center">
            <Camera className="w-10 h-10 text-sky-400" />
            <p className="text-xs text-slate-300">
              Activa la cámara trasera de tu celular para analizar placas, categoría, ejes y
              pasajeros en vivo.
            </p>
            <button onClick={() => startCamera(facing)} className="btn-primary text-xs">
              Activar Video en Vivo
            </button>
          </div>
        )}

        {cameraOn && (
          <>
            <div className="absolute top-2.5 left-2.5 right-2.5 flex items-center justify-between">
              <span className="bg-slate-950/80 text-white text-[11px] font-bold px-2.5 py-1 rounded-full flex items-center gap-1.5">
                <span
                  className={`w-2 h-2 rounded-full ${
                    streaming ? "bg-red-500 animate-pulse" : "bg-emerald-400"
                  }`}
                />
                {streaming ? `EN VIVO ${FPS_OPTIONS[fpsIdx].label}` : "LISTO"} · {framesSent}
              </span>
              <div className="flex gap-1.5">
                <button
                  onClick={() => {
                    const next = facing === "environment" ? "user" : "environment";
                    setFacing(next);
                    startCamera(next);
                  }}
                  className="bg-slate-950/80 text-slate-200 p-1.5 rounded-lg"
                >
                  <RefreshCw className="w-4 h-4" />
                </button>
                <button
                  onClick={stopCamera}
                  className="bg-slate-950/80 text-red-300 p-1.5 rounded-lg"
                >
                  <VideoOff className="w-4 h-4" />
                </button>
              </div>
            </div>

            <div className="pointer-events-none absolute inset-x-6 inset-y-10 border-2 border-dashed border-sky-400/60 rounded-xl flex items-end justify-center pb-1.5">
              <span className="text-[10px] bg-slate-950/75 text-sky-200 px-2 py-0.5 rounded">
                Encuadra vehículo, ruedas y placa
              </span>
            </div>
          </>
        )}
      </div>

      {/* Selector caseta y FPS */}
      <div className="grid grid-cols-2 gap-2">
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
        <div className="flex gap-1">
          {FPS_OPTIONS.map((opt, idx) => (
            <button
              key={opt.label}
              onClick={() => setFpsIdx(idx)}
              className={`flex-1 text-[11px] font-bold rounded-xl border transition ${
                fpsIdx === idx
                  ? "bg-sky-500 text-slate-950 border-sky-400"
                  : "bg-slate-900 text-slate-400 border-slate-800"
              }`}
            >
              {opt.label}
            </button>
          ))}
        </div>
      </div>

      {/* Botones de captura */}
      <div className="grid grid-cols-2 gap-2">
        <button
          disabled={!cameraOn}
          onClick={() => setStreaming((s) => !s)}
          className={`py-3 rounded-xl font-bold text-xs flex items-center justify-center gap-1.5 transition disabled:opacity-40 ${
            streaming ? "bg-red-500 text-white" : "bg-sky-500 text-slate-950"
          }`}
        >
          {streaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {streaming ? "Detener Video" : "Iniciar Caseta"}
        </button>

        <button
          disabled={!cameraOn || sending}
          onClick={captureFromVideo}
          className="btn-secondary text-xs py-3 flex items-center justify-center gap-1.5"
        >
          <Video className="w-4 h-4" />
          {sending ? "Analizando..." : "1 Frame"}
        </button>
      </div>

      {/* Botón cámara nativa del celular (funciona siempre) */}
      <label className="w-full btn-secondary py-3 text-xs flex items-center justify-center gap-2 cursor-pointer border-sky-500/30 text-sky-300">
        <Camera className="w-4 h-4" />
        Tomar Foto con Cámara Nativa del Celular
        <input
          ref={nativeCamRef}
          type="file"
          accept="image/*"
          capture="environment"
          className="hidden"
          disabled={sending}
          onChange={handleNativePhoto}
        />
      </label>

      {error && (
        <div className="bg-red-500/15 border border-red-500/40 rounded-xl p-3 text-xs text-red-200">
          {error}
        </div>
      )}

      {/* Resultado en vivo */}
      {paso && (
        <div className="card p-4 space-y-3 border-sky-500/30">
          <div className="flex items-center justify-between">
            <span className="font-mono text-2xl font-black text-white tracking-wider">
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

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Categoría (M2)</span>
              <strong className="text-white text-sm capitalize">{paso.categoria_detectada}</strong>
            </div>
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Ejes (M3)</span>
              <strong className="text-white text-sm">{paso.ejes_detectados} ejes</strong>
            </div>
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Pasajeros (M4)</span>
              <strong className="text-cyan-300 text-sm">{paso.pasajeros}</strong>
            </div>
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Tarifa</span>
              <strong className="text-emerald-300 text-sm">
                Bs. {Number(paso.monto).toFixed(2)}
              </strong>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span className="flex items-center gap-1 text-emerald-400">
              <CheckCircle2 className="w-3.5 h-3.5" /> Enviado al Panel Web en vivo
            </span>
            <span>{Math.round(paso.latencia_ms)} ms</span>
          </div>
        </div>
      )}
    </div>
  );
}

function MobileConsultaTab() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Modo Consulta activo. Escribe una placa (ej. 2481-KLM) o toma una foto de la placa con tu celular.",
      estado: null,
      sugerencias: []
    }
  ]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [sending, setSending] = useState(false);
  const fileRef = useRef(null);

  async function send(customText) {
    const msgText = typeof customText === "string" ? customText : input.trim();
    const curFile = typeof customText === "string" ? null : file;
    if (!msgText && !curFile) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: msgText || "Foto de placa", image: preview || null }
    ]);
    setInput("");
    setFile(null);
    setPreview("");
    if (fileRef.current) fileRef.current.value = "";
    setSending(true);

    try {
      const fd = new FormData();
      if (msgText) fd.append("mensaje", msgText);
      if (curFile) fd.append("file", curFile);
      const { data } = await api.post("/api/v1/agente/chat", fd);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.respuesta,
          estado: data.estado,
          placa: data.placa_normalizada,
          sugerencias: data.sugerencias || []
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err?.response?.data?.detail || "Error consultando al agente.",
          estado: null,
          sugerencias: []
        }
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="p-4 flex flex-col gap-3 min-h-[calc(100vh-9rem)]">
      <div className="flex-1 space-y-3">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`rounded-2xl p-3 text-xs space-y-1.5 max-w-[90%] ${
              m.role === "user"
                ? "ml-auto bg-sky-500 text-slate-950 font-semibold"
                : "bg-slate-900 border border-slate-800 text-slate-100"
            }`}
          >
            {m.image && <img src={m.image} alt="Placa" className="max-h-36 rounded-xl" />}
            {m.estado && (
              <span
                className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  STATE_BADGE[m.estado] || "bg-slate-800 text-slate-200 border-slate-700"
                }`}
              >
                {m.estado} {m.placa ? `· ${m.placa}` : ""}
              </span>
            )}
            <p className="whitespace-pre-line leading-relaxed">{m.text}</p>
            {m.sugerencias?.length > 0 && (
              <div className="flex flex-wrap gap-1.5 pt-1">
                {m.sugerencias.map((s) => (
                  <button
                    key={s}
                    onClick={() => send(s)}
                    className="font-mono bg-amber-500/20 text-amber-200 border border-amber-500/40 px-2 py-0.5 rounded-lg"
                  >
                    ¿{s}?
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {sending && <p className="text-xs text-sky-400 animate-pulse">Consultando...</p>}
      </div>

      <div className="card p-2.5 space-y-2">
        {preview && (
          <div className="relative inline-block">
            <img src={preview} alt="Preview" className="h-16 rounded-lg" />
            <button
              onClick={() => {
                setFile(null);
                setPreview("");
              }}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5"
            >
              <X className="w-3 h-3" />
            </button>
          </div>
        )}
        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex gap-2"
        >
          <label className="btn-secondary px-3 py-2 cursor-pointer">
            <Camera className="w-4 h-4" />
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={(e) => {
                const f = e.target.files?.[0];
                if (!f) return;
                setFile(f);
                setPreview(URL.createObjectURL(f));
              }}
            />
          </label>
          <input
            className="input-dark flex-1 text-xs"
            placeholder="Placa (ej. 2481-KLM)..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button type="submit" disabled={sending} className="btn-primary px-3">
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
