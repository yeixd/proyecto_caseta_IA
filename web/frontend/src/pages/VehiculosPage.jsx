import React, { useEffect, useState } from "react";
import { Plus, Trash2, Edit3, Shield, Search } from "lucide-react";
import { api } from "../api";
import { useAuth } from "../context/AuthContext";

export default function VehiculosPage({ onRefreshGlobal }) {
  const { isAdmin } = useAuth();
  const [subTab, setSubTab] = useState("vehiculos");
  const [vehiculos, setVehiculos] = useState([]);
  const [tarifas, setTarifas] = useState([]);
  const [casetas, setCasetas] = useState([]);
  const [usuarios, setUsuarios] = useState([]);
  const [search, setSearch] = useState("");
  const [error, setError] = useState("");

  // Form vehículo
  const [editingId, setEditingId] = useState(null);
  const [formV, setFormV] = useState({
    placa: "",
    propietario: "",
    categoria: "auto",
    ejes: 2,
    saldo: 50,
    activo: true
  });

  // Form tarifa
  const [formT, setFormT] = useState({ categoria: "camion", ejes: 3, monto: 28 });

  // Form caseta
  const [formC, setFormC] = useState({ nombre: "", ubicacion: "" });

  // Form usuario lista blanca
  const [formU, setFormU] = useState({ email_google: "", nombre: "", rol: "operador", activo: true });

  async function loadAll() {
    setError("");
    try {
      const [vRes, tRes, cRes] = await Promise.all([
        api.get("/api/v1/vehiculos", { params: search ? { q: search } : {} }),
        api.get("/api/v1/tarifas"),
        api.get("/api/v1/casetas")
      ]);
      setVehiculos(vRes.data);
      setTarifas(tRes.data);
      setCasetas(cRes.data);
      if (isAdmin) {
        const uRes = await api.get("/api/v1/usuarios");
        setUsuarios(uRes.data);
      }
    } catch (err) {
      setError(err?.response?.data?.detail || "Error cargando datos.");
    }
  }

  useEffect(() => {
    loadAll();
  }, [search]);

  async function submitVehiculo(e) {
    e.preventDefault();
    setError("");
    try {
      if (editingId) {
        await api.put(`/api/v1/vehiculos/${editingId}`, {
          propietario: formV.propietario,
          categoria: formV.categoria,
          ejes: Number(formV.ejes),
          saldo: Number(formV.saldo),
          activo: formV.activo
        });
      } else {
        await api.post("/api/v1/vehiculos", {
          ...formV,
          ejes: Number(formV.ejes),
          saldo: Number(formV.saldo)
        });
      }
      setEditingId(null);
      setFormV({ placa: "", propietario: "", categoria: "auto", ejes: 2, saldo: 50, activo: true });
      loadAll();
      onRefreshGlobal?.();
    } catch (err) {
      setError(err?.response?.data?.detail || "No se pudo guardar el vehículo.");
    }
  }

  async function deleteVehiculo(id) {
    if (!confirm("¿Eliminar este vehículo?")) return;
    await api.delete(`/api/v1/vehiculos/${id}`);
    loadAll();
    onRefreshGlobal?.();
  }

  async function submitTarifa(e) {
    e.preventDefault();
    try {
      await api.post("/api/v1/tarifas", {
        ...formT,
        ejes: Number(formT.ejes),
        monto: Number(formT.monto)
      });
      loadAll();
    } catch (err) {
      setError(err?.response?.data?.detail || "Error guardando tarifa.");
    }
  }

  async function submitCaseta(e) {
    e.preventDefault();
    try {
      await api.post("/api/v1/casetas", formC);
      setFormC({ nombre: "", ubicacion: "" });
      loadAll();
      onRefreshGlobal?.();
    } catch (err) {
      setError(err?.response?.data?.detail || "Error creando caseta.");
    }
  }

  async function submitUsuario(e) {
    e.preventDefault();
    try {
      await api.post("/api/v1/usuarios", formU);
      setFormU({ email_google: "", nombre: "", rol: "operador", activo: true });
      loadAll();
    } catch (err) {
      setError(err?.response?.data?.detail || "Error agregando usuario.");
    }
  }

  async function toggleUsuarioActivo(u) {
    await api.put(`/api/v1/usuarios/${u.id}`, { activo: !u.activo });
    loadAll();
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h2 className="text-2xl font-bold text-white">Gestión de Vehículos y Administración</h2>
          <p className="text-sm text-slate-400">
            {isAdmin
              ? "Rol Administrador: gestión completa de vehículos, tarifas, casetas y lista blanca Google."
              : "Rol Operador: modo solo lectura (solo administradores pueden modificar registros)."}
          </p>
        </div>

        <div className="flex gap-2">
          {[
            ["vehiculos", "Vehículos"],
            ["tarifas", "Tarifas"],
            ["casetas", "Casetas"],
            ...(isAdmin ? [["usuarios", "Lista Blanca"]] : [])
          ].map(([key, label]) => (
            <button
              key={key}
              onClick={() => setSubTab(key)}
              className={`px-3 py-1.5 rounded-xl text-xs font-semibold transition ${
                subTab === key
                  ? "bg-sky-500 text-slate-950"
                  : "bg-slate-800 text-slate-300 hover:bg-slate-700"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {error && (
        <div className="bg-red-500/15 border border-red-500/40 rounded-xl px-4 py-3 text-sm text-red-200">
          {error}
        </div>
      )}

      {subTab === "vehiculos" && (
        <div className="space-y-5">
          {isAdmin && (
            <form onSubmit={submitVehiculo} className="card p-5 grid grid-cols-1 md:grid-cols-6 gap-3 items-end">
              <div>
                <label className="text-xs text-slate-400">Placa</label>
                <input
                  className="input-dark w-full mt-1 uppercase font-mono"
                  placeholder="2481-KLM"
                  disabled={Boolean(editingId)}
                  required
                  value={formV.placa}
                  onChange={(e) => setFormV({ ...formV, placa: e.target.value })}
                />
              </div>
              <div className="md:col-span-2">
                <label className="text-xs text-slate-400">Propietario</label>
                <input
                  className="input-dark w-full mt-1"
                  placeholder="Nombre completo"
                  required
                  value={formV.propietario}
                  onChange={(e) => setFormV({ ...formV, propietario: e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400">Categoría</label>
                <select
                  className="input-dark w-full mt-1"
                  value={formV.categoria}
                  onChange={(e) => setFormV({ ...formV, categoria: e.target.value })}
                >
                  <option value="moto">Moto</option>
                  <option value="auto">Auto</option>
                  <option value="minibus">Minibús</option>
                  <option value="camion">Camión</option>
                  <option value="bus">Bus</option>
                </select>
              </div>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <label className="text-xs text-slate-400">Ejes</label>
                  <input
                    type="number"
                    min="2"
                    max="10"
                    className="input-dark w-full mt-1"
                    value={formV.ejes}
                    onChange={(e) => setFormV({ ...formV, ejes: e.target.value })}
                  />
                </div>
                <div>
                  <label className="text-xs text-slate-400">Saldo (Bs)</label>
                  <input
                    type="number"
                    step="0.5"
                    min="0"
                    className="input-dark w-full mt-1"
                    value={formV.saldo}
                    onChange={(e) => setFormV({ ...formV, saldo: e.target.value })}
                  />
                </div>
              </div>
              <div className="flex gap-2">
                <button type="submit" className="btn-primary w-full text-xs flex items-center justify-center gap-1">
                  <Plus className="w-4 h-4" />
                  {editingId ? "Actualizar" : "Agregar"}
                </button>
                {editingId && (
                  <button
                    type="button"
                    className="btn-secondary text-xs"
                    onClick={() => {
                      setEditingId(null);
                      setFormV({ placa: "", propietario: "", categoria: "auto", ejes: 2, saldo: 50, activo: true });
                    }}
                  >
                    Cancelar
                  </button>
                )}
              </div>
            </form>
          )}

          <div className="card p-5 space-y-4">
            <div className="flex items-center justify-between gap-3">
              <div className="relative flex-1 max-w-sm">
                <Search className="w-4 h-4 text-slate-500 absolute left-3 top-2.5" />
                <input
                  className="input-dark w-full pl-9"
                  placeholder="Buscar por placa o propietario..."
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                />
              </div>
              <span className="text-xs text-slate-400">Total: {vehiculos.length} vehículos</span>
            </div>

            <div className="overflow-x-auto">
              <table className="w-full text-sm text-left">
                <thead className="text-xs text-slate-400 border-b border-slate-800">
                  <tr>
                    <th className="py-2.5 px-3">Placa</th>
                    <th className="py-2.5 px-3">Propietario</th>
                    <th className="py-2.5 px-3">Categoría</th>
                    <th className="py-2.5 px-3">Ejes</th>
                    <th className="py-2.5 px-3">Saldo</th>
                    <th className="py-2.5 px-3">Estado</th>
                    {isAdmin && <th className="py-2.5 px-3 text-right">Acciones</th>}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/70">
                  {vehiculos.map((v) => (
                    <tr key={v.id} className="hover:bg-slate-800/30">
                      <td className="py-2.5 px-3 font-mono font-bold text-white">{v.placa}</td>
                      <td className="py-2.5 px-3 text-slate-300">{v.propietario}</td>
                      <td className="py-2.5 px-3 capitalize text-slate-300">{v.categoria}</td>
                      <td className="py-2.5 px-3 text-slate-300">{v.ejes}</td>
                      <td
                        className={`py-2.5 px-3 font-semibold ${
                          v.saldo <= 5 ? "text-amber-400" : "text-emerald-400"
                        }`}
                      >
                        Bs. {Number(v.saldo).toFixed(2)}
                      </td>
                      <td className="py-2.5 px-3">
                        <span
                          className={`text-xs px-2 py-0.5 rounded-full ${
                            v.activo
                              ? "bg-emerald-500/15 text-emerald-300"
                              : "bg-slate-700 text-slate-400"
                          }`}
                        >
                          {v.activo ? "Activo" : "Inactivo"}
                        </span>
                      </td>
                      {isAdmin && (
                        <td className="py-2.5 px-3 text-right space-x-2">
                          <button
                            onClick={() => {
                              setEditingId(v.id);
                              setFormV(v);
                            }}
                            className="text-sky-400 hover:text-sky-300"
                            title="Editar"
                          >
                            <Edit3 className="w-4 h-4 inline" />
                          </button>
                          <button
                            onClick={() => deleteVehiculo(v.id)}
                            className="text-red-400 hover:text-red-300"
                            title="Eliminar"
                          >
                            <Trash2 className="w-4 h-4 inline" />
                          </button>
                        </td>
                      )}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}

      {subTab === "tarifas" && (
        <div className="card p-5 space-y-4">
          {isAdmin && (
            <form onSubmit={submitTarifa} className="flex flex-wrap gap-3 items-end">
              <div>
                <label className="text-xs text-slate-400">Categoría</label>
                <select
                  className="input-dark block mt-1"
                  value={formT.categoria}
                  onChange={(e) => setFormT({ ...formT, categoria: e.target.value })}
                >
                  <option value="moto">Moto</option>
                  <option value="auto">Auto</option>
                  <option value="minibus">Minibús</option>
                  <option value="camion">Camión</option>
                  <option value="bus">Bus</option>
                </select>
              </div>
              <div>
                <label className="text-xs text-slate-400">Ejes</label>
                <input
                  type="number"
                  min="2"
                  max="10"
                  className="input-dark block mt-1 w-24"
                  value={formT.ejes}
                  onChange={(e) => setFormT({ ...formT, ejes: e.target.value })}
                />
              </div>
              <div>
                <label className="text-xs text-slate-400">Tarifa (Bs.)</label>
                <input
                  type="number"
                  step="0.5"
                  min="0"
                  className="input-dark block mt-1 w-28"
                  value={formT.monto}
                  onChange={(e) => setFormT({ ...formT, monto: e.target.value })}
                />
              </div>
              <button className="btn-primary text-xs">Guardar Tarifa</button>
            </form>
          )}
          <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
            {tarifas.map((t) => (
              <div key={t.id} className="bg-slate-950 border border-slate-800 rounded-xl p-3">
                <p className="text-xs text-slate-400 capitalize">{t.categoria}</p>
                <p className="text-sm text-slate-200">{t.ejes} ejes</p>
                <p className="text-lg font-bold text-emerald-400 mt-1">Bs. {Number(t.monto).toFixed(2)}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {subTab === "casetas" && (
        <div className="card p-5 space-y-4">
          {isAdmin && (
            <form onSubmit={submitCaseta} className="flex flex-wrap gap-3 items-end">
              <input
                className="input-dark flex-1 min-w-[200px]"
                placeholder="Nombre de la caseta"
                required
                value={formC.nombre}
                onChange={(e) => setFormC({ ...formC, nombre: e.target.value })}
              />
              <input
                className="input-dark flex-1 min-w-[240px]"
                placeholder="Ubicación"
                required
                value={formC.ubicacion}
                onChange={(e) => setFormC({ ...formC, ubicacion: e.target.value })}
              />
              <button className="btn-primary text-xs">Agregar Caseta</button>
            </form>
          )}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
            {casetas.map((c) => (
              <div key={c.id} className="bg-slate-950 border border-slate-800 rounded-xl p-4">
                <p className="text-xs text-sky-400 font-mono">Caseta #{c.id}</p>
                <p className="font-semibold text-white mt-1">{c.nombre}</p>
                <p className="text-xs text-slate-400 mt-0.5">{c.ubicacion}</p>
              </div>
            ))}
          </div>
        </div>
      )}

      {subTab === "usuarios" && isAdmin && (
        <div className="card p-5 space-y-4">
          <div className="flex items-center gap-2 text-sm text-sky-400 font-medium">
            <Shield className="w-4 h-4" />
            Lista Blanca de Cuentas Institucionales (@unifranz.edu.bo)
          </div>
          <form onSubmit={submitUsuario} className="flex flex-wrap gap-3 items-end">
            <input
              type="email"
              required
              pattern=".+@unifranz\.edu\.bo$"
              title="Debe ser un correo institucional terminado en @unifranz.edu.bo"
              className="input-dark flex-1 min-w-[240px]"
              placeholder="usuario@unifranz.edu.bo"
              value={formU.email_google}
              onChange={(e) => setFormU({ ...formU, email_google: e.target.value })}
            />
            <input
              required
              className="input-dark flex-1 min-w-[180px]"
              placeholder="Nombre"
              value={formU.nombre}
              onChange={(e) => setFormU({ ...formU, nombre: e.target.value })}
            />
            <select
              className="input-dark"
              value={formU.rol}
              onChange={(e) => setFormU({ ...formU, rol: e.target.value })}
            >
              <option value="operador">Operador</option>
              <option value="administrador">Administrador</option>
            </select>
            <button className="btn-primary text-xs">Autorizar Cuenta</button>
          </form>

          <div className="divide-y divide-slate-800">
            {usuarios.map((u) => (
              <div key={u.id} className="py-3 flex flex-wrap items-center justify-between gap-3">
                <div>
                  <p className="text-sm font-medium text-white">{u.nombre}</p>
                  <p className="text-xs text-slate-400 font-mono">{u.email_google}</p>
                </div>
                <div className="flex items-center gap-3">
                  <span className="text-xs uppercase px-2.5 py-0.5 rounded-full bg-sky-500/15 text-sky-300">
                    {u.rol}
                  </span>
                  <button
                    onClick={() => toggleUsuarioActivo(u)}
                    className={`text-xs px-3 py-1 rounded-lg border ${
                      u.activo
                        ? "border-emerald-500/40 text-emerald-300 bg-emerald-500/10"
                        : "border-red-500/40 text-red-300 bg-red-500/10"
                    }`}
                  >
                    {u.activo ? "Activo" : "Bloqueado"}
                  </button>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
