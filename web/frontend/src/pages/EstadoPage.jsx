import React, { useEffect, useState } from "react";
import { Cpu, Database, Radio, RefreshCw, Save, Server, Zap } from "lucide-react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";

export default function EstadoPage() {
  const { isAdmin } = useAuth();
  const [status, setStatus] = useState(null);
  const [loading, setLoading] = useState(false);
  const [savingCfg, setSavingCfg] = useState(false);
  const [cfgMsg, setCfgMsg] = useState("");
  const [hwMsg, setHwMsg] = useState("");
  const [ollamaTestMsg, setOllamaTestMsg] = useState("");
  const [testingOllama, setTestingOllama] = useState(false);

  const [inferenceUrl, setInferenceUrl] = useState("");
  const [inferenceApiKey, setInferenceApiKey] = useState("");
  const [ollamaUrl, setOllamaUrl] = useState("");
  const [ollamaModel, setOllamaModel] = useState("llama3.1:8b");

  async function loadStatus() {
    setLoading(true);
    try {
      const { data } = await api.get("/api/v1/estado");
      setStatus(data);
      if (data.inference?.url) setInferenceUrl(data.inference.url);
      if (data.ollama?.url) setOllamaUrl(data.ollama.url);
      if (data.ollama?.configured_model) setOllamaModel(data.ollama.configured_model);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadStatus();
  }, []);

  async function saveEndpoints(e) {
    e.preventDefault();
    setSavingCfg(true);
    setCfgMsg("");
    try {
      const { data } = await api.put("/api/v1/estado/config", {
        inference_url: inferenceUrl.trim(),
        inference_api_key: inferenceApiKey.trim(),
        ollama_url: ollamaUrl.trim(),
        ollama_model: ollamaModel.trim() || "llama3.1:8b"
      });
      setStatus((prev) =>
        prev ? { ...prev, inference: data.inference, ollama: data.ollama } : prev
      );
      setCfgMsg(
        data.inference?.online
          ? `Conectado correctamente a ${data.inference.url} (${data.inference.latency_ms} ms)`
          : `URL guardada, pero el endpoint aún no responde: ${data.inference?.error || "sin respuesta"}`
      );
    } catch (err) {
      setCfgMsg(err?.response?.data?.detail || "Error guardando configuración.");
    } finally {
      setSavingCfg(false);
    }
  }

  async function testHardware(tipo) {
    setHwMsg("");
    try {
      const { data } = await api.post("/api/v1/hardware/test", {
        tipo,
        placa: "2481KLM",
        detalle: `Señal ${tipo} desde página de Estado`
      });
      setHwMsg(
        `Señal ${tipo} enviada. Dispositivos ESP32 notificados: ${data.dispositivos_notificados}`
      );
      loadStatus();
    } catch (err) {
      setHwMsg(err?.response?.data?.detail || "Error probando hardware.");
    }
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-white">Estado del Sistema e Inferencia</h2>
          <p className="text-sm text-slate-400">
            Monitoreo en vivo de INFERENCE_URL (RunPod/Colab), Ollama, Base de Datos y Hardware ESP32
          </p>
        </div>
        <button onClick={loadStatus} className="btn-secondary text-xs flex items-center gap-1.5">
          <RefreshCw className={`w-4 h-4 ${loading ? "animate-spin" : ""}`} />
          Verificar ahora
        </button>
      </div>

      {/* Configuración en vivo de INFERENCE_URL y OLLAMA_URL (solo admin) */}
      {isAdmin && (
        <form onSubmit={saveEndpoints} className="card p-5 space-y-4 border-sky-500/30">
          <div>
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Server className="w-4 h-4 text-sky-400" />
              Conectar Endpoint de Inferencia (RunPod o Google Colab) en Vivo
            </h3>
            <p className="text-xs text-slate-400 mt-1">
              Pega aquí la URL HTTPS de RunPod (puerto 8000) o el túnel de Google Colab (
              <code className="font-mono">https://....trycloudflare.com</code>). Se aplica al
              instante sin redesplegar Render.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            <div>
              <label className="text-xs text-slate-400">INFERENCE_URL (RunPod o Colab)</label>
              <input
                className="input-dark w-full mt-1 font-mono text-xs"
                placeholder="https://TU-POD-8000.proxy.runpod.net o https://xxxx.trycloudflare.com"
                value={inferenceUrl}
                onChange={(e) => setInferenceUrl(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">INFERENCE_API_KEY (opcional)</label>
              <input
                className="input-dark w-full mt-1 font-mono text-xs"
                placeholder="Dejar vacío si no configuraste clave en Colab/RunPod"
                value={inferenceApiKey}
                onChange={(e) => setInferenceApiKey(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">OLLAMA_URL (opcional — RunPod 11434)</label>
              <input
                className="input-dark w-full mt-1 font-mono text-xs"
                placeholder="https://TU-POD-11434.proxy.runpod.net"
                value={ollamaUrl}
                onChange={(e) => setOllamaUrl(e.target.value)}
              />
            </div>
            <div>
              <label className="text-xs text-slate-400">Modelo Ollama</label>
              <input
                className="input-dark w-full mt-1 font-mono text-xs"
                placeholder="llama3.1:8b"
                value={ollamaModel}
                onChange={(e) => setOllamaModel(e.target.value)}
              />
            </div>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            <button type="submit" disabled={savingCfg} className="btn-primary text-xs flex items-center gap-1.5">
              <Save className="w-4 h-4" />
              {savingCfg ? "Guardando y probando..." : "Guardar y Verificar Conexión"}
            </button>
            {cfgMsg && <span className="text-xs text-sky-300">{cfgMsg}</span>}
          </div>
        </form>
      )}

      {status && (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-5">
          {/* Endpoint de inferencia */}
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Server className="w-5 h-5 text-sky-400" />
                Endpoint de Inferencia (INFERENCE_URL)
              </h3>
              <Badge online={status.inference?.online} />
            </div>
            <p className="text-xs font-mono text-slate-400 break-all">
              {status.inference?.url || "No configurado"}
            </p>
            {status.inference?.latency_ms && (
              <p className="text-xs text-emerald-300">
                Latencia /health: {status.inference.latency_ms} ms
                {status.inference?.details?.multi_angle_enabled ? " · Multi-ángulo activo" : ""}
              </p>
            )}
            {status.inference?.error && (
              <p className="text-xs text-red-300">{status.inference.error}</p>
            )}
            {status.inference?.details?.models && (
              <div className="pt-2 space-y-1.5">
                <p className="text-xs text-slate-400 font-semibold">
                  GPU: {status.inference.details.gpu_name || status.inference.details.device}
                </p>
                {Object.entries(status.inference.details.models).map(([name, info]) => (
                  <div
                    key={name}
                    className="flex items-center justify-between bg-slate-950 rounded-lg px-3 py-1.5 text-xs"
                  >
                    <span className="font-mono text-slate-200">{name}</span>
                    <span className={info.loaded ? "text-emerald-400" : "text-amber-400"}>
                      {info.loaded
                        ? info.custom === false
                          ? "Base activo"
                          : "Custom cargado"
                        : "Fallback OpenCV activo"}
                    </span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Ollama */}
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Zap className="w-5 h-5 text-violet-400" />
                Agente LLM (Ollama en RunPod)
              </h3>
              <Badge online={status.ollama?.online} fallbackLabel="Fallback BD activo" />
            </div>
            <p className="text-xs font-mono text-slate-400 break-all">
              {status.ollama?.url || status.ollama?.detail}
            </p>
            {status.ollama?.configured_model && (
              <p className="text-xs text-slate-300">
                Modelo configurado: <strong>{status.ollama.configured_model}</strong>
              </p>
            )}
            {status.ollama?.available_models?.length > 0 && (
              <p className="text-xs text-emerald-300">
                Modelos disponibles: {status.ollama.available_models.join(", ")}
              </p>
            )}
            <div className="pt-1">
              <button
                type="button"
                disabled={testingOllama}
                onClick={async () => {
                  setTestingOllama(true);
                  setOllamaTestMsg("");
                  try {
                    const { data } = await api.post("/api/v1/estado/test-ollama");
                    setOllamaTestMsg(
                      data.ok
                        ? `LLM OK (${data.model}): ${data.reply}`
                        : `Error LLM: ${data.error}`
                    );
                  } catch (err) {
                    setOllamaTestMsg(err?.response?.data?.detail || "Error probando Ollama.");
                  } finally {
                    setTestingOllama(false);
                  }
                }}
                className="btn-secondary text-xs"
              >
                {testingOllama ? "Probando Ollama LLM..." : "Probar LLM Ollama"}
              </button>
              {ollamaTestMsg && (
                <p className="text-xs text-violet-300 mt-2 whitespace-pre-line">{ollamaTestMsg}</p>
              )}
            </div>
          </div>

          {/* Base de datos */}
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Database className="w-5 h-5 text-emerald-400" />
                Base de Datos y Tiempo Real
              </h3>
              <Badge online={status.database?.online} />
            </div>
            <p className="text-xs text-slate-300">
              Paneles web conectados (WebSocket):{" "}
              <strong>{status.realtime?.paneles_conectados ?? 0}</strong>
            </p>
            <p className="text-xs text-slate-400">
              Última verificación: {new Date(status.timestamp_utc).toLocaleString("es-BO")}
            </p>
          </div>

          {/* Hardware ESP32 */}
          <div className="card p-5 space-y-3">
            <div className="flex items-center justify-between">
              <h3 className="font-semibold text-white flex items-center gap-2">
                <Cpu className="w-5 h-5 text-amber-400" />
                Dispositivo Hardware de Alertas (ESP32)
              </h3>
              <Badge
                online={(status.realtime?.hardware_conectados ?? 0) > 0}
                fallbackLabel="Sin ESP32 conectado"
              />
            </div>
            <p className="text-xs text-slate-300">
              Dispositivos conectados: <strong>{status.realtime?.hardware_conectados ?? 0}</strong>
            </p>
            {status.realtime?.ultimo_evento_hardware && (
              <p className="text-xs text-slate-400 font-mono">
                Último evento: {status.realtime.ultimo_evento_hardware.tipo} (
                {status.realtime.ultimo_evento_hardware.placa || "-"})
              </p>
            )}
            <div className="flex flex-wrap gap-2 pt-2">
              {["IDENTIFICANDO", "APROBADO", "SIN_SALDO", "ALERTA_NO_REGISTRADO", "TEST"].map(
                (sig) => (
                  <button
                    key={sig}
                    onClick={() => testHardware(sig)}
                    className="btn-secondary text-[11px] px-2.5 py-1 flex items-center gap-1"
                  >
                    <Radio className="w-3 h-3" /> {sig}
                  </button>
                )
              )}
            </div>
            {hwMsg && <p className="text-xs text-sky-300 pt-1">{hwMsg}</p>}
          </div>
        </div>
      )}
    </div>
  );
}

function Badge({ online, fallbackLabel }) {
  return (
    <span
      className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
        online
          ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
          : "bg-amber-500/15 text-amber-300 border-amber-500/40"
      }`}
    >
      {online ? "ONLINE" : fallbackLabel || "OFFLINE"}
    </span>
  );
}
