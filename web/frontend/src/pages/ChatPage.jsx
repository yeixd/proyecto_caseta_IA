import React, { useRef, useState } from "react";
import { Bot, Camera, Send, Sparkles, User, X } from "lucide-react";
import { api } from "../api";

const STATE_BADGE = {
  "Registrada y con paso": "bg-emerald-500/20 text-emerald-300 border-emerald-500/40",
  "Registrada sin paso": "bg-sky-500/20 text-sky-300 border-sky-500/40",
  "Identificada no registrada": "bg-red-500/20 text-red-300 border-red-500/40",
  "No existe": "bg-slate-700 text-slate-300 border-slate-600",
  "Placa ilegible": "bg-amber-500/20 text-amber-300 border-amber-500/40"
};

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hola, soy el Agente Conversacional de Control de Peaje. Escribe una placa (ej. 2481-KLM) o sube una fotografía de la placa para verificar su estado real en la base de datos.",
      estado: null,
      sugerencias: []
    }
  ]);
  const [input, setInput] = useState("");
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState("");
  const [sending, setSending] = useState(false);
  const fileRef = useRef(null);

  function handleFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setPreview(URL.createObjectURL(f));
  }

  function clearFile() {
    setFile(null);
    setPreview("");
    if (fileRef.current) fileRef.current.value = "";
  }

  async function sendQuery(customText) {
    const msgText = typeof customText === "string" ? customText : input.trim();
    if (!msgText && !file) return;

    const userBubble = {
      role: "user",
      text: msgText || "Consulta por imagen de placa",
      image: preview || null
    };
    setMessages((prev) => [...prev, userBubble]);
    setInput("");
    const currentFile = file;
    clearFile();
    setSending(true);

    try {
      const fd = new FormData();
      if (msgText) fd.append("mensaje", msgText);
      if (currentFile) fd.append("file", currentFile);

      const { data } = await api.post("/api/v1/agente/chat", fd);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.respuesta,
          estado: data.estado,
          placa: data.placa_normalizada,
          sugerencias: data.sugerencias || [],
          motor: data.motor
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text:
            err?.response?.data?.detail ||
            "No se pudo completar la consulta al agente. Verifica la conexión con el backend o endpoint de inferencia.",
          estado: null,
          sugerencias: []
        }
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <div className="h-[calc(100vh-7rem)] flex flex-col gap-4">
      <div>
        <h2 className="text-2xl font-bold text-white flex items-center gap-2">
          <Sparkles className="w-6 h-6 text-sky-400" />
          Agente Conversacional de Placas
        </h2>
        <p className="text-sm text-slate-400">
          Consultas por texto o imagen (M1 + OCR) con los 4 estados oficiales, normalización y
          sugerencias OCR (O↔0, I↔1, B↔8, S↔5)
        </p>
      </div>

      <div className="card flex-1 p-4 overflow-y-auto space-y-4">
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${m.role === "user" ? "justify-end" : "justify-start"}`}
          >
            {m.role === "assistant" && (
              <div className="w-8 h-8 rounded-full bg-sky-500/20 text-sky-400 flex items-center justify-center shrink-0">
                <Bot className="w-4 h-4" />
              </div>
            )}

            <div
              className={`max-w-2xl rounded-2xl px-4 py-3 text-sm space-y-2 ${
                m.role === "user"
                  ? "bg-sky-500 text-slate-950 font-medium"
                  : "bg-slate-950 border border-slate-800 text-slate-100"
              }`}
            >
              {m.image && (
                <img
                  src={m.image}
                  alt="Placa consultada"
                  className="max-h-44 rounded-xl border border-slate-700"
                />
              )}

              {m.estado && (
                <div className="flex flex-wrap items-center gap-2">
                  <span
                    className={`text-xs font-semibold px-2.5 py-0.5 rounded-full border ${
                      STATE_BADGE[m.estado] || "bg-slate-800 text-slate-200 border-slate-700"
                    }`}
                  >
                    {m.estado}
                  </span>
                  {m.placa && (
                    <span className="font-mono text-xs text-slate-400">Placa: {m.placa}</span>
                  )}
                  {m.motor && (
                    <span
                      className={`text-[10px] px-2 py-0.5 rounded-full ml-auto font-mono ${
                        m.motor.startsWith("ollama:")
                          ? "bg-violet-500/20 text-violet-300 border border-violet-500/40"
                          : "bg-slate-800 text-slate-400"
                      }`}
                    >
                      {m.motor.startsWith("ollama:") ? `LLM · ${m.motor}` : m.motor}
                    </span>
                  )}
                </div>
              )}

              <p className="whitespace-pre-line leading-relaxed">{m.text}</p>

              {m.sugerencias?.length > 0 && (
                <div className="pt-1 flex flex-wrap items-center gap-2">
                  <span className="text-xs text-amber-300">¿Quisiste decir:</span>
                  {m.sugerencias.map((sug) => (
                    <button
                      key={sug}
                      onClick={() => sendQuery(sug)}
                      className="text-xs font-mono bg-amber-500/20 hover:bg-amber-500/30 text-amber-200 border border-amber-500/40 px-2.5 py-1 rounded-lg transition"
                    >
                      {sug}
                    </button>
                  ))}
                </div>
              )}
            </div>

            {m.role === "user" && (
              <div className="w-8 h-8 rounded-full bg-slate-800 text-slate-300 flex items-center justify-center shrink-0">
                <User className="w-4 h-4" />
              </div>
            )}
          </div>
        ))}
        {sending && (
          <div className="text-xs text-sky-400 animate-pulse pl-11">
            Consultando base de datos e inferencia...
          </div>
        )}
      </div>

      {/* Input */}
      <div className="card p-3 space-y-2">
        {preview && (
          <div className="relative inline-block">
            <img src={preview} alt="Preview" className="h-20 rounded-lg border border-slate-700" />
            <button
              onClick={clearFile}
              className="absolute -top-2 -right-2 bg-red-500 text-white rounded-full p-0.5"
            >
              <X className="w-3.5 h-3.5" />
            </button>
          </div>
        )}

        <form
          onSubmit={(e) => {
            e.preventDefault();
            sendQuery();
          }}
          className="flex gap-2 items-center"
        >
          <label className="btn-secondary px-3 py-2 cursor-pointer" title="Subir foto de placa">
            <Camera className="w-4 h-4" />
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              onChange={handleFileChange}
            />
          </label>
          <input
            className="input-dark flex-1"
            placeholder="Escribe una placa (ej. 2481-KLM, 248IKLM) o adjunta una foto..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            disabled={sending}
          />
          <button type="submit" disabled={sending} className="btn-primary flex items-center gap-1.5">
            <Send className="w-4 h-4" />
            Enviar
          </button>
        </form>
      </div>
    </div>
  );
}
