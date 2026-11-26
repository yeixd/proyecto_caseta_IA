import React, { useEffect, useRef, useState } from "react";
import {
  Bot,
  Camera,
  CheckCircle2,
  ExternalLink,
  Image as ImageIcon,
  LogOut,
  Mic,
  MicOff,
  Monitor,
  Pause,
  Play,
  RefreshCw,
  Send,
  Smartphone,
  Sparkles,
  Timer,
  Trash2,
  Video,
  VideoOff,
  X
} from "lucide-react";
import {
  api,
  compressImageForUpload,
  formatVehicleClass,
  resolveImageUrl,
  VEHICLE_CLASS_COLORS
} from "../api";
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

export default function MobileAppView({
  casetas,
  pasos = [],
  wsConnected,
  onExitMobile,
  onRefresh
}) {
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
              <span className="truncate">
                {wsConnected ? "Sincronizado con Web" : "Conectando..."} · {user?.email_google}
              </span>
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 shrink-0">
          <button
            onClick={onExitMobile}
            title="Ir al Panel Web completo"
            className="px-2.5 py-2 rounded-xl bg-sky-500/15 border border-sky-500/30 text-sky-300 hover:text-white text-[11px] font-semibold flex items-center gap-1"
          >
            <Monitor className="w-3.5 h-3.5" />
            Web
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
          <MobileCasetaTab
            casetas={casetas}
            pasos={pasos}
            onRefresh={onRefresh}
            onExitMobile={onExitMobile}
          />
        ) : (
          <MobileConsultaTab casetas={casetas} onRefresh={onRefresh} />
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

function MobileCasetaTab({ casetas, pasos = [], onRefresh, onExitMobile }) {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const streamRef = useRef(null);
  const inFlightRef = useRef(false);
  const nativeCamRef = useRef(null);
  const galleryRef = useRef(null);

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
  const [bursting, setBursting] = useState(false);
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
        video: {
          facingMode: { ideal: mode },
          width: { ideal: 1920 },
          height: { ideal: 1080 },
          advanced: [{ focusMode: "continuous" }]
        },
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

  async function uploadFrameBlob(blobOrFile, filename = "frame_movil.jpg") {
    inFlightRef.current = true;
    setSending(true);
    setError("");
    try {
      const optimized = await compressImageForUpload(blobOrFile, 1600, 0.90);
      const fd = new FormData();
      fd.append("file", optimized, filename.replace(/\.[^.]+$/, ".jpg"));
      fd.append("caseta_id", String(casetaId || casetas[0]?.id || 1));
      fd.append("origen", "movil");
      const { data } = await api.post("/api/v1/pasos/frame", fd);
      setLastResult(data);
      setFramesSent((n) => n + 1);
      onRefresh?.();
    } catch (err) {
      setError(
        err?.response?.data?.detail ||
          "Error conectando con el backend o endpoint de inferencia (verifica INFERENCE_URL en Estado)."
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
    const blob = await new Promise((res) => canvas.toBlob(res, "image/jpeg", 0.88));
    if (blob) await uploadFrameBlob(blob, `mobile_${Date.now()}.jpg`);
  }

  function startClipBurst() {
    if (!cameraOn || streaming || bursting) return;
    setBursting(true);
    setStreaming(true);
    setTimeout(() => {
      setStreaming(false);
      setBursting(false);
    }, 5000);
  }

  async function handleNativePhoto(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    await uploadFrameBlob(file, file.name || `camara_movil_${Date.now()}.jpg`);
    if (nativeCamRef.current) nativeCamRef.current.value = "";
    if (galleryRef.current) galleryRef.current.value = "";
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

      {/* Botones de captura de video */}
      <div className="grid grid-cols-3 gap-2">
        <button
          disabled={!cameraOn}
          onClick={() => {
            setBursting(false);
            setStreaming((s) => !s);
          }}
          className={`py-3 rounded-xl font-bold text-xs flex items-center justify-center gap-1.5 transition disabled:opacity-40 ${
            streaming ? "bg-red-500 text-white" : "bg-sky-500 text-slate-950"
          }`}
        >
          {streaming ? <Pause className="w-4 h-4" /> : <Play className="w-4 h-4" />}
          {streaming ? "Detener" : "En Vivo"}
        </button>

        <button
          disabled={!cameraOn || streaming || sending}
          onClick={startClipBurst}
          className="btn-secondary text-xs py-3 flex items-center justify-center gap-1"
        >
          <Timer className="w-4 h-4 text-sky-400" />
          Clip 5s
        </button>

        <button
          disabled={!cameraOn || sending}
          onClick={captureFromVideo}
          className="btn-secondary text-xs py-3 flex items-center justify-center gap-1"
        >
          <Video className="w-4 h-4" />
          {sending ? "Enviando..." : "1 Frame"}
        </button>
      </div>

      {/* Captura con Cámara Nativa o Galería del Celular (envía directo al Panel Web) */}
      <div className="grid grid-cols-2 gap-2">
        <label className="btn-primary py-3 text-xs flex items-center justify-center gap-1.5 cursor-pointer text-center">
          <Camera className="w-4 h-4 shrink-0" />
          <span>{sending ? "Procesando..." : "Cámara Celular"}</span>
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

        <label className="btn-secondary py-3 text-xs flex items-center justify-center gap-1.5 cursor-pointer border-sky-500/30 text-sky-300 text-center">
          <ImageIcon className="w-4 h-4 shrink-0" />
          <span>{sending ? "Procesando..." : "Subir de Galería"}</span>
          <input
            ref={galleryRef}
            type="file"
            accept="image/*"
            className="hidden"
            disabled={sending}
            onChange={handleNativePhoto}
          />
        </label>
      </div>

      {sending && (
        <div className="bg-sky-500/15 border border-sky-500/40 rounded-xl p-3 text-xs text-sky-200 animate-pulse text-center">
          Analizando vehículo, llantas, ejes, pasajeros y placa... enviando registro al Panel Web.
        </div>
      )}

      {error && (
        <div className="bg-red-500/15 border border-red-500/40 rounded-xl p-3 text-xs text-red-200">
          {error}
        </div>
      )}

      {/* Resultado en vivo + Evidencia anotada enviada al Panel Web */}
      {paso && (
        <div className="card p-4 space-y-3 border-emerald-500/40">
          <div className="bg-emerald-500/15 border border-emerald-500/30 rounded-xl px-3 py-2 flex items-center justify-between gap-2">
            <span className="text-[11px] font-semibold text-emerald-300 flex items-center gap-1.5">
              <CheckCircle2 className="w-4 h-4 shrink-0" />
              Registro #{paso.id} enviado a la Página Web
            </span>
            <button
              onClick={onExitMobile}
              className="text-[11px] font-bold text-sky-300 hover:text-white flex items-center gap-1 shrink-0"
            >
              Ver en Web <ExternalLink className="w-3 h-3" />
            </button>
          </div>

          {paso.imagen_url && (
            <img
              src={resolveImageUrl(paso.imagen_url)}
              alt={paso.placa_normalizada || "Evidencia anotada"}
              className="w-full rounded-xl border border-slate-700 bg-slate-950 object-contain max-h-56"
            />
          )}

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
            <div className="col-span-2 bg-slate-950 rounded-xl p-2.5 flex items-center justify-between">
              <span className="text-slate-400">Clase de vehículo (M2)</span>
              <span
                className={`px-2.5 py-0.5 rounded-full font-bold border ${
                  VEHICLE_CLASS_COLORS[paso.categoria_detectada] || VEHICLE_CLASS_COLORS.auto
                }`}
              >
                {formatVehicleClass(paso.categoria_detectada)}
              </span>
            </div>
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Ejes / Llantas (M3)</span>
              <strong className="text-white text-sm">
                {paso.ejes_detectados} ejes
                {(paso.llantas_visibles ?? lastResult?.inferencia?.vehiculo?.llantas_visibles) !==
                undefined
                  ? ` (${
                      paso.llantas_visibles ?? lastResult.inferencia.vehiculo.llantas_visibles
                    } llantas)`
                  : ""}
              </strong>
            </div>
            <div className="bg-slate-950 rounded-xl p-2.5">
              <span className="text-slate-400 block">Pasajeros (M4)</span>
              <strong className="text-cyan-300 text-sm">{paso.pasajeros}</strong>
            </div>
            <div className="col-span-2 bg-slate-950 rounded-xl p-2.5 flex items-center justify-between">
              <span className="text-slate-400">Tarifa cobrada</span>
              <strong className="text-emerald-300 text-base">
                Bs. {Number(paso.monto).toFixed(2)}
              </strong>
            </div>
          </div>

          <div className="flex items-center justify-between text-[11px] text-slate-400">
            <span>{paso.caseta_nombre || `Caseta #${paso.caseta_id}`}</span>
            <span>{Math.round(paso.latencia_ms)} ms</span>
          </div>
        </div>
      )}

      {/* Historial sincronizado con la Página Web */}
      {pasos.length > 0 && (
        <div className="card p-3.5 space-y-2.5">
          <div className="flex items-center justify-between">
            <h3 className="text-xs font-bold text-slate-200">
              Últimos registros en la Página Web ({pasos.length})
            </h3>
            <button onClick={onRefresh} className="text-[11px] text-sky-400 flex items-center gap-1">
              <RefreshCw className="w-3 h-3" /> Sincronizar
            </button>
          </div>
          <div className="space-y-2 max-h-56 overflow-y-auto pr-0.5">
            {pasos.slice(0, 6).map((p) => (
              <div
                key={p.id}
                className="bg-slate-950/90 border border-slate-800 rounded-xl p-2 flex items-center gap-2.5 text-xs"
              >
                {p.imagen_url && p.tiene_evidencia !== false ? (
                  <img
                    src={resolveImageUrl(p.imagen_url)}
                    alt={p.placa_normalizada}
                    className="w-14 h-10 rounded-lg object-cover border border-slate-700 shrink-0"
                  />
                ) : (
                  <div className="w-14 h-10 rounded-lg bg-slate-900 border border-slate-800 shrink-0" />
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center justify-between gap-1">
                    <span className="font-mono font-bold text-white truncate">
                      {p.placa_normalizada || "ILEGIBLE"}
                    </span>
                    <span className="text-[10px] text-emerald-300 font-semibold shrink-0">
                      Bs. {Number(p.monto).toFixed(2)}
                    </span>
                  </div>
                  <div className="text-[11px] text-slate-400 truncate">
                    {formatVehicleClass(p.categoria_detectada)} · {p.ejes_detectados} ejes ·{" "}
                    {p.pasajeros} pasaj.
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

const MOBILE_QUICK_PROMPTS = [
  { label: "📊 Resumen hoy", prompt: "Dame el resumen del día y recaudación" },
  { label: "🚨 Alertas", prompt: "Muéstrame las alertas activas pendientes" },
  { label: "💰 Tarifas", prompt: "Cuáles son las tarifas vigentes por vehículo y ejes" },
  { label: "🚗 Últimos pasos", prompt: "Muéstrame los últimos 5 pasos registrados" },
  { label: "🔎 2481-KLM", prompt: "2481-KLM" }
];

const INITIAL_AGENT_MSG = {
  role: "assistant",
  text: "Hola 👋 Soy tu Agente IA Móvil de Control de Peaje.\n• Escribe o dicta por voz una placa (ej. 2481-KLM o 5280-YIN).\n• Toma foto o elige de galería (incluye restauración para placas borrosas).\n• O toca un acceso rápido abajo.",
  estado: null,
  sugerencias: [],
  datos: null,
  ocr: null,
  motor: null
};

function MobileConsultaTab({ casetas = [], onRefresh }) {
  const [messages, setMessages] = useState([INITIAL_AGENT_MSG]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [sending, setSending] = useState(false);
  const [listening, setListening] = useState(false);
  const camInputRef = useRef(null);
  const galInputRef = useRef(null);
  const endRef = useRef(null);
  const recognitionRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, sending]);

  function toggleVoiceDictation() {
    const SpeechRec = window.SpeechRecognition || window.webkitSpeechRecognition;
    if (!SpeechRec) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: "Tu navegador móvil no soporta dictado por voz directo. Puedes usar el micrófono del teclado o escribir la placa.",
          estado: null,
          sugerencias: []
        }
      ]);
      return;
    }
    if (listening && recognitionRef.current) {
      recognitionRef.current.stop();
      setListening(false);
      return;
    }
    const rec = new SpeechRec();
    rec.lang = "es-BO";
    rec.interimResults = false;
    rec.maxAlternatives = 1;
    rec.onstart = () => setListening(true);
    rec.onend = () => setListening(false);
    rec.onerror = () => setListening(false);
    rec.onresult = (ev) => {
      const transcript = ev.results?.[0]?.[0]?.transcript || "";
      if (transcript.trim()) {
        setInput(transcript.trim());
        send(transcript.trim());
      }
    };
    recognitionRef.current = rec;
    rec.start();
  }

  function handleSelectImage(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function clearSelectedImage() {
    setFile(null);
    setPreview("");
    if (camInputRef.current) camInputRef.current.value = "";
    if (galInputRef.current) galInputRef.current.value = "";
  }

  async function sendAsPasoToWeb() {
    if (!file || sending) return;
    const curFile = file;
    const curPreview = preview;
    clearSelectedImage();
    setSending(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: "Registrar vehículo en Panel Web", image: curPreview || null }
    ]);
    try {
      const optimized = await compressImageForUpload(curFile, 1600, 0.90);
      const fd = new FormData();
      fd.append("file", optimized, `consulta_paso_${Date.now()}.jpg`);
      fd.append("caseta_id", String(casetas[0]?.id || 1));
      fd.append("origen", "movil");
      const { data } = await api.post("/api/v1/pasos/frame", fd);
      const p = data.paso;
      onRefresh?.();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `✅ Registro #${p.id} enviado en vivo al Panel Web:\n• Placa: ${
            p.placa_normalizada || "ILEGIBLE"
          }\n• Clase: ${formatVehicleClass(p.categoria_detectada)}\n• Ejes: ${
            p.ejes_detectados
          }${
            p.llantas_visibles !== undefined ? ` (${p.llantas_visibles} llantas)` : ""
          }\n• Pasajeros: ${p.pasajeros}\n• Tarifa: Bs. ${Number(p.monto).toFixed(2)} (${
            p.estado
          })`,
          estado: p.estado === "APROBADO" ? "Registrada y con paso" : "Identificada no registrada",
          placa: p.placa_normalizada,
          sugerencias: [],
          evidenceUrl: p.imagen_url ? resolveImageUrl(p.imagen_url) : null,
          motor: "caseta_en_vivo"
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err?.response?.data?.detail || "Error registrando paso en el Panel Web.",
          estado: null,
          sugerencias: []
        }
      ]);
    } finally {
      setSending(false);
    }
  }

  async function send(customText) {
    const msgText = typeof customText === "string" ? customText : input.trim();
    const curFile = typeof customText === "string" ? null : file;
    if (!msgText && !curFile) return;

    setMessages((prev) => [
      ...prev,
      {
        role: "user",
        text: msgText || "Analizar fotografía de placa / vehículo",
        image: curFile ? preview : null
      }
    ]);
    setInput("");
    if (curFile) clearSelectedImage();
    setSending(true);

    try {
      const fd = new FormData();
      if (msgText) fd.append("mensaje", msgText);
      if (curFile) {
        const optimized = await compressImageForUpload(curFile, 1600, 0.92);
        fd.append("file", optimized, "placa_movil.jpg");
      }
      const { data } = await api.post("/api/v1/agente/chat", fd);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.respuesta,
          estado: data.estado,
          placa: data.placa_normalizada,
          sugerencias: data.sugerencias || [],
          datos: data.datos || null,
          ocr: data.ocr || null,
          motor: data.motor || null
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: err?.response?.data?.detail || "Error consultando al agente en la nube.",
          estado: null,
          sugerencias: []
        }
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="p-3.5 flex flex-col gap-3 min-h-[calc(100vh-8.5rem)]">
      {/* Cabecera compacta del Agente Móvil + Accesos rápidos */}
      <div className="bg-slate-900/90 border border-slate-800 rounded-2xl p-2.5 space-y-2">
        <div className="flex items-center justify-between">
          <span className="text-xs font-bold text-white flex items-center gap-1.5">
            <Sparkles className="w-4 h-4 text-sky-400" />
            Agente IA de Placas y Peaje
          </span>
          <button
            onClick={() => setMessages([INITIAL_AGENT_MSG])}
            className="text-[11px] text-slate-400 hover:text-white flex items-center gap-1 px-2 py-0.5 rounded-lg bg-slate-800/80"
          >
            <Trash2 className="w-3 h-3" /> Limpiar
          </button>
        </div>

        <div className="flex gap-1.5 overflow-x-auto pb-1 no-scrollbar">
          {MOBILE_QUICK_PROMPTS.map((q) => (
            <button
              key={q.label}
              disabled={sending}
              onClick={() => send(q.prompt)}
              className="shrink-0 text-[11px] font-semibold px-2.5 py-1.5 rounded-xl bg-slate-950 border border-slate-800 text-sky-300 active:bg-sky-500/20"
            >
              {q.label}
            </button>
          ))}
        </div>
      </div>

      {/* Historial de conversación con tarjetas estructuradas */}
      <div className="flex-1 space-y-3">
        {messages.map((m, idx) => {
          const veh = m.datos?.vehiculo;
          const ult = m.datos?.ultimo_paso;
          const analisisVeh = m.datos?.analisis_vehiculo;
          return (
            <div
              key={idx}
              className={`rounded-2xl p-3 text-xs space-y-2 max-w-[94%] ${
                m.role === "user"
                  ? "ml-auto bg-sky-500 text-slate-950 font-semibold"
                  : "bg-slate-900 border border-slate-800 text-slate-100"
              }`}
            >
              {m.image && (
                <img
                  src={m.image}
                  alt="Consulta móvil"
                  className="max-h-40 rounded-xl border border-slate-700/60"
                />
              )}

              {m.evidenceUrl && (
                <img
                  src={m.evidenceUrl}
                  alt="Evidencia anotada"
                  className="w-full max-h-48 object-contain rounded-xl border border-emerald-500/40 bg-slate-950"
                />
              )}

              {(m.estado || m.motor || m.ocr?.confianza) && (
                <div className="flex flex-wrap items-center gap-1.5">
                  {m.estado && (
                    <span
                      className={`inline-block text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                        STATE_BADGE[m.estado] || "bg-slate-800 text-slate-200 border-slate-700"
                      }`}
                    >
                      {m.estado} {m.placa ? `· ${m.placa}` : ""}
                    </span>
                  )}
                  {m.ocr?.confianza > 0 && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-emerald-500/15 text-emerald-300 border border-emerald-500/30">
                      OCR {(m.ocr.confianza * 100).toFixed(0)}%
                    </span>
                  )}
                  {m.motor && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-violet-500/15 text-violet-300 border border-violet-500/30 ml-auto">
                      {m.motor.startsWith("ollama:") ? "✨ LLM Ollama" : "🛡️ BD Verificada"}
                    </span>
                  )}
                </div>
              )}

              <p className="whitespace-pre-line leading-relaxed">{m.text}</p>

              {/* Tarjeta visual de Vehículo Empadronado */}
              {veh && (
                <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2.5 grid grid-cols-2 gap-1.5 text-[11px]">
                  <div className="col-span-2 flex items-center justify-between border-b border-slate-800 pb-1">
                    <span className="text-slate-400">Propietario</span>
                    <strong className="text-white truncate max-w-[65%]">{veh.propietario}</strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Clase / Ejes</span>
                    <strong className="text-sky-300">
                      {formatVehicleClass(veh.categoria)} ({veh.ejes} ejes)
                    </strong>
                  </div>
                  <div>
                    <span className="text-slate-400 block">Saldo disponible</span>
                    <strong className={veh.saldo > 0 ? "text-emerald-300" : "text-amber-300"}>
                      Bs. {Number(veh.saldo).toFixed(2)}
                    </strong>
                  </div>
                </div>
              )}

              {/* Tarjeta visual del Último Paso + miniatura de evidencia */}
              {ult && (
                <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2.5 space-y-1.5 text-[11px]">
                  <div className="flex items-center justify-between">
                    <span className="text-slate-400 font-semibold">Último paso en peaje</span>
                    <strong className="text-emerald-300">Bs. {Number(ult.monto).toFixed(2)}</strong>
                  </div>
                  <div className="text-slate-300">
                    {ult.caseta} · {formatVehicleClass(ult.categoria)} · {ult.ejes} ejes ·{" "}
                    {ult.pasajeros} pasaj.
                  </div>
                  {ult.imagen_url && (
                    <img
                      src={resolveImageUrl(ult.imagen_url)}
                      alt="Última evidencia"
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                      className="w-full max-h-36 object-cover rounded-lg border border-slate-800 mt-1"
                    />
                  )}
                </div>
              )}

              {/* Detalle de análisis visual si fue foto de vehículo completo */}
              {analisisVeh && analisisVeh.categoria_nombre && !veh && (
                <div className="bg-slate-950/90 border border-slate-800 rounded-xl p-2 text-[11px] text-slate-300">
                  Detectado en foto: <strong>{analisisVeh.categoria_nombre}</strong> ·{" "}
                  {analisisVeh.ejes} ejes ({analisisVeh.llantas_visibles ?? 0} llantas) ·{" "}
                  {analisisVeh.pasajeros} pasaj.
                </div>
              )}

              {m.sugerencias?.length > 0 && (
                <div className="space-y-1 pt-1">
                  <span className="text-[10px] text-amber-300 block">
                    Coincidencias cercanas por OCR borroso (toca para consultar):
                  </span>
                  <div className="flex flex-wrap gap-1.5">
                    {m.sugerencias.map((s) => (
                      <button
                        key={s}
                        onClick={() => send(s)}
                        className="font-mono font-bold bg-amber-500/20 text-amber-200 border border-amber-500/40 px-2.5 py-1 rounded-lg"
                      >
                        ¿{s}?
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
          );
        })}
        {sending && (
          <div className="bg-slate-900/80 border border-slate-800 rounded-2xl px-3 py-2 text-xs text-sky-400 animate-pulse inline-flex items-center gap-2">
            <Bot className="w-4 h-4" /> Analizando con Super-OCR Deblur + Agente IA...
          </div>
        )}
        <div ref={endRef} />
      </div>

      {/* Barra inferior de entrada multimodal (Cámara + Galería + Voz + Texto) */}
      <div className="card p-2.5 space-y-2 sticky bottom-16 bg-slate-900/95 backdrop-blur border-slate-700">
        {preview && (
          <div className="flex items-center justify-between gap-2 bg-slate-950 p-2 rounded-xl border border-slate-800">
            <div className="relative inline-block">
              <img src={preview} alt="Preview" className="h-14 rounded-lg object-cover" />
              <button
                onClick={clearSelectedImage}
                className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5"
              >
                <X className="w-3 h-3" />
              </button>
            </div>
            <div className="flex flex-col gap-1 flex-1 items-end">
              <span className="text-[10px] text-slate-400">
                Envía con ➤ para consultar o registra el paso:
              </span>
              <button
                type="button"
                disabled={sending}
                onClick={sendAsPasoToWeb}
                className="btn-primary text-[11px] py-1.5 px-2.5"
              >
                📥 Registrar Paso en Web
              </button>
            </div>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send();
          }}
          className="flex items-center gap-1.5"
        >
          <label
            className="btn-secondary p-2.5 cursor-pointer shrink-0"
            title="Tomar foto con cámara"
          >
            <Camera className="w-4 h-4 text-sky-400" />
            <input
              ref={camInputRef}
              type="file"
              accept="image/*"
              capture="environment"
              className="hidden"
              onChange={handleSelectImage}
            />
          </label>

          <label
            className="btn-secondary p-2.5 cursor-pointer shrink-0"
            title="Elegir foto de galería"
          >
            <ImageIcon className="w-4 h-4 text-emerald-400" />
            <input
              ref={galInputRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleSelectImage}
            />
          </label>

          <button
            type="button"
            onClick={toggleVoiceDictation}
            title="Dictar placa o consulta por voz"
            className={`p-2.5 rounded-xl border transition shrink-0 ${
              listening
                ? "bg-red-500 text-white border-red-400 animate-pulse"
                : "bg-slate-800 text-slate-300 border-slate-700"
            }`}
          >
            {listening ? <MicOff className="w-4 h-4" /> : <Mic className="w-4 h-4" />}
          </button>

          <input
            className="input-dark flex-1 text-xs min-w-0 py-2.5"
            placeholder={listening ? "Escuchando voz..." : "Placa o pregunta..."}
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />

          <button type="submit" disabled={sending} className="btn-primary p-2.5 shrink-0">
            <Send className="w-4 h-4" />
          </button>
        </form>
      </div>
    </div>
  );
}
