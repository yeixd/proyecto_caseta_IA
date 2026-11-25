import React, { useState } from "react";
import { AlertTriangle, CheckCircle2 } from "lucide-react";
import { api, resolveImageUrl } from "../api";

export default function AlertasPage({ alertas, onRefresh, onSelectImage }) {
  const [onlyPending, setOnlyPending] = useState(false);

  async function markAttended(id) {
    await api.patch(`/api/v1/alertas/${id}/atender`);
    onRefresh();
  }

  const list = onlyPending ? alertas.filter((a) => !a.atendida) : alertas;

  return (
    <div className="space-y-5">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Centro de Alertas</h2>
          <p className="text-sm text-slate-400">
            Alertas automáticas ante placas no registradas, vehículos sin saldo o inactivos
          </p>
        </div>

        <label className="flex items-center gap-2 text-sm text-slate-300 cursor-pointer">
          <input
            type="checkbox"
            checked={onlyPending}
            onChange={(e) => setOnlyPending(e.target.checked)}
            className="rounded border-slate-700"
          />
          Solo pendientes
        </label>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {list.map((a) => (
          <div
            key={a.id}
            className={`card p-4 flex gap-4 items-start ${
              a.atendida ? "opacity-70" : "border-red-500/40 bg-red-950/15"
            }`}
          >
            {a.imagen_url && (
              <img
                src={resolveImageUrl(a.imagen_url)}
                alt={a.placa}
                onClick={() => onSelectImage(resolveImageUrl(a.imagen_url))}
                className="w-24 h-20 object-cover rounded-lg border border-slate-700 cursor-pointer shrink-0"
              />
            )}
            <div className="flex-1 space-y-1.5">
              <div className="flex items-center justify-between gap-2">
                <span className="inline-flex items-center gap-1.5 text-xs font-bold text-red-300">
                  <AlertTriangle className="w-3.5 h-3.5" />
                  {a.tipo}
                </span>
                <span className="font-mono text-xs bg-slate-800 px-2 py-0.5 rounded text-white">
                  {a.placa || "N/A"}
                </span>
              </div>
              <p className="text-sm text-slate-200">{a.mensaje}</p>
              <div className="flex items-center justify-between pt-2">
                <span className="text-[11px] text-slate-400">
                  {new Date(a.created_at).toLocaleString("es-BO")}
                </span>
                {a.atendida ? (
                  <span className="text-xs text-emerald-400 inline-flex items-center gap-1">
                    <CheckCircle2 className="w-3.5 h-3.5" /> Atendida
                  </span>
                ) : (
                  <button
                    onClick={() => markAttended(a.id)}
                    className="btn-secondary text-xs py-1 px-3"
                  >
                    Marcar atendida
                  </button>
                )}
              </div>
            </div>
          </div>
        ))}
        {list.length === 0 && (
          <div className="col-span-full card p-12 text-center text-slate-500">
            No hay alertas registradas.
          </div>
        )}
      </div>
    </div>
  );
}
