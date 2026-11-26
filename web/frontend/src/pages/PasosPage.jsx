import React, { useEffect, useState } from "react";
import { Image as ImageIcon, Search, RefreshCw, Trash2 } from "lucide-react";
import { api, formatVehicleClass, resolveImageUrl } from "../api";
import { useAuth } from "../context/AuthContext";

export default function PasosPage({ livePasos = [], onSelectImage }) {
  const { isAdmin } = useAuth();
  const [pasos, setPasos] = useState([]);
  const [placaFilter, setPlacaFilter] = useState("");
  const [estadoFilter, setEstadoFilter] = useState("");
  const [loading, setLoading] = useState(false);

  // Incorporar automáticamente capturas entrantes del móvil vía WebSocket sin recargar
  useEffect(() => {
    if (!livePasos.length || placaFilter.trim() || estadoFilter) return;
    setPasos((prev) => {
      const map = new Map();
      [...livePasos, ...prev].forEach((item) => {
        if (item?.id) map.set(item.id, { ...map.get(item.id), ...item });
      });
      return Array.from(map.values()).sort((a, b) => b.id - a.id).slice(0, 100);
    });
  }, [livePasos, placaFilter, estadoFilter]);

  async function cleanMissingImages() {
    if (!confirm("¿Eliminar los registros antiguos que no tienen imagen guardada?")) return;
    await api.delete("/api/v1/pasos/limpiar-sin-evidencia");
    fetchPasos();
  }

  async function fetchPasos() {
    setLoading(true);
    try {
      const params = { limit: 100 };
      if (placaFilter.trim()) params.placa = placaFilter.trim();
      if (estadoFilter) params.estado = estadoFilter;
      const { data } = await api.get("/api/v1/pasos", { params });
      setPasos(data);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    fetchPasos();
  }, [estadoFilter]);

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Historial de Pasos con Evidencia</h2>
          <p className="text-sm text-slate-400">
            Registro completo de vehículos detectados, categoría, ejes, pasajeros y fotografía anotada
          </p>
        </div>

        <div className="flex flex-wrap gap-2">
          <div className="relative">
            <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
            <input
              className="input-dark pl-9"
              placeholder="Filtrar placa..."
              value={placaFilter}
              onChange={(e) => setPlacaFilter(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && fetchPasos()}
            />
          </div>
          <select
            className="input-dark"
            value={estadoFilter}
            onChange={(e) => setEstadoFilter(e.target.value)}
          >
            <option value="">Todos los estados</option>
            <option value="APROBADO">APROBADO</option>
            <option value="SIN_SALDO">SIN_SALDO</option>
            <option value="NO_REGISTRADO">NO_REGISTRADO</option>
            <option value="PLACA_ILEGIBLE">PLACA_ILEGIBLE</option>
          </select>
          <button onClick={fetchPasos} className="btn-secondary text-xs flex items-center gap-1.5">
            <RefreshCw className={`w-3.5 h-3.5 ${loading ? "animate-spin" : ""}`} />
            Actualizar
          </button>
          {isAdmin && (
            <button
              onClick={cleanMissingImages}
              className="btn-secondary text-xs flex items-center gap-1.5 text-amber-300 border-amber-500/30"
              title="Eliminar pruebas antiguas cuya imagen se borró antes de guardarlas en PostgreSQL"
            >
              <Trash2 className="w-3.5 h-3.5" />
              Limpiar sin imagen
            </button>
          )}
        </div>
      </div>

      <div className="card p-4 overflow-x-auto">
        <table className="w-full text-sm text-left">
          <thead className="text-xs text-slate-400 border-b border-slate-800">
            <tr>
              <th className="py-2.5 px-3">Evidencia</th>
              <th className="py-2.5 px-3">Placa</th>
              <th className="py-2.5 px-3">Categoría / Ejes</th>
              <th className="py-2.5 px-3">Pasajeros</th>
              <th className="py-2.5 px-3">Monto</th>
              <th className="py-2.5 px-3">Estado</th>
              <th className="py-2.5 px-3">Caseta</th>
              <th className="py-2.5 px-3">Fecha y Hora</th>
            </tr>
          </thead>
          <tbody className="divide-y divide-slate-800/70">
            {pasos.map((p) => (
              <tr key={p.id} className="hover:bg-slate-800/30">
                <td className="py-2.5 px-3">
                  {p.imagen_url && p.tiene_evidencia !== false ? (
                    <img
                      src={resolveImageUrl(p.imagen_url)}
                      alt={p.placa_normalizada}
                      onClick={() => onSelectImage(resolveImageUrl(p.imagen_url))}
                      onError={(e) => {
                        e.currentTarget.style.display = "none";
                      }}
                      className="w-20 h-14 object-cover rounded-lg border border-slate-700 cursor-pointer hover:scale-105 transition bg-slate-900"
                    />
                  ) : (
                    <span className="text-xs text-slate-500 flex items-center gap-1">
                      <ImageIcon className="w-4 h-4" /> Sin foto
                    </span>
                  )}
                </td>
                <td className="py-2.5 px-3">
                  <div className="font-mono font-bold text-white">{p.placa_normalizada || "ILEGIBLE"}</div>
                  <div className="text-[11px] text-slate-500">
                    OCR {(p.confianza_ocr * 100).toFixed(0)}% · {Math.round(p.latencia_ms)}ms
                  </div>
                </td>
                <td className="py-2.5 px-3 text-slate-200">
                  <div className="font-medium">{formatVehicleClass(p.categoria_detectada)}</div>
                  <div className="text-xs text-slate-400">{p.ejes_detectados} ejes</div>
                </td>
                <td className="py-2.5 px-3 text-cyan-300 font-semibold">{p.pasajeros}</td>
                <td className="py-2.5 px-3 text-emerald-300 font-semibold">
                  Bs. {Number(p.monto).toFixed(2)}
                </td>
                <td className="py-2.5 px-3">
                  <span className="text-xs px-2.5 py-0.5 rounded-full bg-slate-800 text-slate-200 border border-slate-700">
                    {p.estado}
                  </span>
                </td>
                <td className="py-2.5 px-3 text-slate-300">{p.caseta_nombre}</td>
                <td className="py-2.5 px-3 text-xs text-slate-400">
                  {new Date(p.created_at).toLocaleString("es-BO")}
                </td>
              </tr>
            ))}
            {pasos.length === 0 && (
              <tr>
                <td colSpan="8" className="py-10 text-center text-slate-500">
                  No se encontraron pasos con esos filtros.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
