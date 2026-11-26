import React, { useEffect, useRef, useState } from "react";
import {
  Activity,
  AlertTriangle,
  Banknote,
  Car,
  CheckCircle2,
  Clock,
  Cpu,
  Upload,
  Users
} from "lucide-react";
import { api, resolveImageUrl } from "../api";

const STATUS_STYLE = {
  APROBADO: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  SIN_SALDO: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  NO_REGISTRADO: "bg-red-500/15 text-red-300 border-red-500/40",
  VEHICULO_INACTIVO: "bg-rose-500/15 text-rose-300 border-rose-500/40",
  PLACA_ILEGIBLE: "bg-slate-700/60 text-slate-300 border-slate-600"
};

export default function DashboardPage({
  summary,
  pasos,
  alertas,
  casetas,
  wsConnected,
  onSelectImage,
  onRefresh
}) {
  const [selectedCaseta, setSelectedCaseta] = useState(casetas[0]?.id || 1);
  const [uploading, setUploading] = useState(false);

  useEffect(() => {
    if (casetas.length > 0 && !casetas.some((c) => c.id === selectedCaseta)) {
      setSelectedCaseta(casetas[0].id);
    }
  }, [casetas, selectedCaseta]);
  const [uploadMsg, setUploadMsg] = useState("");
  const fileRef = useRef(null);

  async function handleTestFrame(e) {
    const file = e.target.files?.[0];
    if (!file) return;
    setUploading(true);
    setUploadMsg("");
    try {
      const fd = new FormData();
      fd.append("file", file);
      fd.append("caseta_id", String(selectedCaseta || casetas[0]?.id || 1));
      const { data } = await api.post("/api/v1/pasos/frame", fd);
      setUploadMsg(
        `Frame procesado: placa ${data.paso.placa_normalizada || "ILEGIBLE"} | ${data.paso.categoria_detectada} | ${data.paso.ejes_detectados} ejes | ${data.paso.pasajeros} pasajeros | ${data.paso.estado}`
      );
      onRefresh();
    } catch (err) {
      setUploadMsg(err?.response?.data?.detail || "Error enviando frame a inferencia.");
    } finally {
      setUploading(false);
      if (fileRef.current) fileRef.current.value = "";
    }
  }

  const pendingAlerts = alertas.filter((a) => !a.atendida);

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Panel en Tiempo Real</h2>
          <p className="text-sm text-slate-400">
            Actualización instantánea vía WebSocket sin recargar la página
          </p>
        </div>
        <div className="flex items-center gap-3">
          <span
            className={`inline-flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium border ${
              wsConnected
                ? "bg-emerald-500/15 text-emerald-300 border-emerald-500/40"
                : "bg-amber-500/15 text-amber-300 border-amber-500/40"
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                wsConnected ? "bg-emerald-400 animate-pulse" : "bg-amber-400"
              }`}
            />
            {wsConnected ? "WebSocket LIVE" : "Reconectando WS..."}
          </span>

          <select
            className="input-dark text-xs"
            value={selectedCaseta}
            onChange={(e) => setSelectedCaseta(Number(e.target.value))}
          >
            {casetas.map((c) => (
              <option key={c.id} value={c.id}>
                {c.nombre}
              </option>
            ))}
          </select>

          <label className="btn-primary text-xs inline-flex items-center gap-2 cursor-pointer">
            <Upload className="w-4 h-4" />
            {uploading ? "Infiriendo..." : "Enviar Frame Prueba"}
            <input
              ref={fileRef}
              type="file"
              accept="image/*"
              className="hidden"
              disabled={uploading}
              onChange={handleTestFrame}
            />
          </label>
        </div>
      </div>

      {uploadMsg && (
        <div className="bg-sky-500/15 border border-sky-500/40 rounded-xl px-4 py-3 text-sm text-sky-200">
          {uploadMsg}
        </div>
      )}

      {/* KPIs */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <KpiCard
          icon={<Car className="w-5 h-5 text-sky-400" />}
          label="Pasos Hoy"
          value={summary?.pasos_hoy ?? 0}
        />
        <KpiCard
          icon={<Banknote className="w-5 h-5 text-emerald-400" />}
          label="Recaudación Hoy"
          value={`Bs. ${(summary?.recaudacion_hoy ?? 0).toFixed(2)}`}
        />
        <KpiCard
          icon={<Users className="w-5 h-5 text-cyan-400" />}
          label="Pasajeros Hoy (M4)"
          value={summary?.pasajeros_hoy ?? 0}
        />
        <KpiCard
          icon={<AlertTriangle className="w-5 h-5 text-red-400" />}
          label="Alertas Pendientes"
          value={summary?.alertas_pendientes ?? pendingAlerts.length}
          highlight={(summary?.alertas_pendientes ?? 0) > 0}
        />
        <KpiCard
          icon={<Cpu className="w-5 h-5 text-violet-400" />}
          label="ESP32 Conectados"
          value={summary?.hardware_conectado ?? 0}
        />
      </div>

      {/* Feed tiempo real + Alertas en vivo */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 card p-5 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="font-semibold text-white flex items-center gap-2">
              <Activity className="w-4 h-4 text-sky-400" />
              Últimos Vehículos Registrados
            </h3>
            <span className="text-xs text-slate-400">Mostrando {pasos.length} registros</span>
          </div>

          {pasos.length === 0 ? (
            <div className="py-12 text-center text-slate-500 text-sm">
              Aún no hay pasos registrados. Captura video desde la App Móvil en Modo Caseta o envía
              un frame de prueba.
            </div>
          ) : (
            <div className="space-y-3 max-h-[540px] overflow-y-auto pr-1">
              {pasos.slice(0, 20).map((p) => (
                <div
                  key={p.id}
                  className="bg-slate-950/80 border border-slate-800 rounded-xl p-3 flex flex-col sm:flex-row gap-4 items-start sm:items-center hover:border-slate-700 transition"
                >
                  {p.imagen_url && p.tiene_evidencia !== false ? (
                    <img
                      src={resolveImageUrl(p.imagen_url)}
                      alt={p.placa_normalizada}
                      onClick={() => onSelectImage(resolveImageUrl(p.imagen_url))}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                      className="w-28 h-20 object-cover rounded-lg border border-slate-700 cursor-pointer hover:opacity-90 shrink-0 bg-slate-900"
                    />
                  ) : (
                    <div className="w-28 h-20 rounded-lg bg-slate-900 border border-slate-800 flex items-center justify-center text-[11px] text-slate-500 text-center px-2 shrink-0">
                      Sin imagen previa
                    </div>
                  )}

                  <div className="flex-1 min-w-0">
                    <div className="flex flex-wrap items-center gap-2">
                      <span className="font-mono text-lg font-bold text-white tracking-wider">
                        {p.placa_normalizada || "ILEGIBLE"}
                      </span>
                      <span
                        className={`text-[11px] font-semibold px-2.5 py-0.5 rounded-full border ${
                          STATUS_STYLE[p.estado] || STATUS_STYLE.PLACA_ILEGIBLE
                        }`}
                      >
                        {p.estado}
                      </span>
                      <span className="text-xs text-slate-400">
                        OCR: {(p.confianza_ocr * 100).toFixed(0)}%
                      </span>
                    </div>

                    <div className="mt-1.5 flex flex-wrap gap-x-4 gap-y-1 text-xs text-slate-300">
                      <span>
                        Categoría: <strong className="capitalize">{p.categoria_detectada}</strong>
                      </span>
                      <span>
                        Ejes: <strong>{p.ejes_detectados}</strong>
                      </span>
                      <span>
                        Pasajeros: <strong className="text-cyan-300">{p.pasajeros}</strong>
                      </span>
                      <span>
                        Tarifa:{" "}
                        <strong className="text-emerald-300">Bs. {Number(p.monto).toFixed(2)}</strong>
                      </span>
                    </div>

                    <div className="mt-1 flex flex-wrap items-center gap-3 text-[11px] text-slate-500">
                      <span>{p.caseta_nombre || `Caseta #${p.caseta_id}`}</span>
                      <span className="flex items-center gap-1">
                        <Clock className="w-3 h-3" />
                        {new Date(p.created_at).toLocaleString("es-BO")}
                      </span>
                      {p.latencia_ms > 0 && <span>{Math.round(p.latencia_ms)} ms</span>}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Alertas en tiempo real */}
        <div className="card p-5 space-y-4">
          <h3 className="font-semibold text-white flex items-center gap-2">
            <AlertTriangle className="w-4 h-4 text-red-400" />
            Alertas Activas ({pendingAlerts.length})
          </h3>

          {pendingAlerts.length === 0 ? (
            <div className="py-10 text-center text-slate-500 text-sm flex flex-col items-center gap-2">
              <CheckCircle2 className="w-8 h-8 text-emerald-500/60" />
              Sin alertas pendientes
            </div>
          ) : (
            <div className="space-y-3 max-h-[540px] overflow-y-auto pr-1">
              {pendingAlerts.slice(0, 15).map((a) => (
                <div
                  key={a.id}
                  className="bg-red-500/10 border border-red-500/30 rounded-xl p-3 space-y-1.5"
                >
                  <div className="flex items-center justify-between gap-2">
                    <span className="text-xs font-bold text-red-300">{a.tipo}</span>
                    <span className="font-mono text-xs text-white bg-red-500/30 px-2 py-0.5 rounded">
                      {a.placa || "N/A"}
                    </span>
                  </div>
                  <p className="text-xs text-red-100/90">{a.mensaje}</p>
                  <p className="text-[10px] text-red-300/70">
                    {new Date(a.created_at).toLocaleString("es-BO")}
                  </p>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function KpiCard({ icon, label, value, highlight }) {
  return (
    <div
      className={`card p-4 flex items-center gap-4 ${
        highlight ? "border-red-500/50 bg-red-950/20" : ""
      }`}
    >
      <div className="w-11 h-11 rounded-xl bg-slate-800/80 flex items-center justify-center shrink-0">
        {icon}
      </div>
      <div>
        <p className="text-xs text-slate-400">{label}</p>
        <p className="text-xl font-bold text-white mt-0.5">{value}</p>
      </div>
    </div>
  );
}
