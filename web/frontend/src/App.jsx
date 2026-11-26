import React, { useCallback, useEffect, useState } from "react";
import { useGoogleLogin } from "@react-oauth/google";
import {
  Activity,
  AlertTriangle,
  Bot,
  Camera,
  Car,
  ClipboardList,
  LogOut,
  RefreshCw,
  Server,
  ShieldCheck,
  Smartphone,
  X
} from "lucide-react";
import { api, buildPanelWsUrl } from "./api";
import { useAuth } from "./context/AuthContext";
import LoginPage from "./pages/LoginPage";
import DashboardPage from "./pages/DashboardPage";
import CamaraCasetaPage from "./pages/CamaraCasetaPage";
import MobileAppView from "./pages/MobileAppView";
import VehiculosPage from "./pages/VehiculosPage";
import PasosPage from "./pages/PasosPage";
import AlertasPage from "./pages/AlertasPage";
import ChatPage from "./pages/ChatPage";
import EstadoPage from "./pages/EstadoPage";

const NAV_ITEMS = [
  { id: "panel", label: "Panel en Vivo", icon: Activity },
  { id: "camara", label: "Cámara Caseta", icon: Camera },
  { id: "vehiculos", label: "Vehículos", icon: Car },
  { id: "pasos", label: "Pasos", icon: ClipboardList },
  { id: "alertas", label: "Alertas", icon: AlertTriangle },
  { id: "chat", label: "Agente Chat", icon: Bot },
  { id: "estado", label: "Estado", icon: Server }
];

export default function App() {
  const { token, user, logout, loginWithGoogleAccessToken } = useAuth();
  const switchGoogleAccount = useGoogleLogin({
    scope: "openid email profile",
    prompt: "select_account",
    onSuccess: (tokenResponse) => {
      if (tokenResponse?.access_token) {
        loginWithGoogleAccessToken(tokenResponse.access_token);
      }
    }
  });
  const [activeTab, setActiveTab] = useState("panel");
  const [mobileMode, setMobileMode] = useState(
    () => window.location.pathname.startsWith("/movil")
  );
  const [summary, setSummary] = useState(null);
  const [pasos, setPasos] = useState([]);
  const [alertas, setAlertas] = useState([]);
  const [casetas, setCasetas] = useState([]);
  const [wsConnected, setWsConnected] = useState(false);
  const [modalImage, setModalImage] = useState("");

  const fetchInitialData = useCallback(async () => {
    if (!token) return;
    try {
      const [sumRes, pasosRes, alertasRes, casetasRes] = await Promise.all([
        api.get("/api/v1/resumen/global"),
        api.get("/api/v1/pasos", { params: { limit: 30 } }),
        api.get("/api/v1/alertas", { params: { limit: 50 } }),
        api.get("/api/v1/casetas")
      ]);
      setSummary(sumRes.data);
      setPasos(pasosRes.data);
      setAlertas(alertasRes.data);
      setCasetas(casetasRes.data);
    } catch (err) {
      console.error("Error cargando datos iniciales:", err);
    }
  }, [token]);

  useEffect(() => {
    fetchInitialData();
  }, [fetchInitialData]);

  // Conexión WebSocket persistente para actualizaciones en tiempo real
  useEffect(() => {
    if (!token) return;
    let ws;
    let reconnectTimer;
    let pingTimer;
    let closedByUser = false;

    function connect() {
      ws = new WebSocket(buildPanelWsUrl(token));
      ws.onopen = () => {
        setWsConnected(true);
        fetchInitialData();
        pingTimer = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) ws.send("ping");
        }, 25000);
      };
      ws.onmessage = (event) => {
        try {
          const msg = JSON.parse(event.data);
          if (msg.event === "nuevo_paso" && msg.data) {
            setPasos((prev) => [
              msg.data,
              ...prev.filter((p) => p.id !== msg.data.id)
            ].slice(0, 50));
            if (msg.alerta) {
              setAlertas((prev) => [
                msg.alerta,
                ...prev.filter((a) => a.id !== msg.alerta.id)
              ].slice(0, 50));
            }
            setSummary((prev) => {
              if (!prev) return prev;
              return {
                ...prev,
                pasos_hoy: (prev.pasos_hoy || 0) + 1,
                recaudacion_hoy:
                  (prev.recaudacion_hoy || 0) + Number(msg.data.monto || 0),
                pasajeros_hoy: (prev.pasajeros_hoy || 0) + Number(msg.data.pasajeros || 0),
                alertas_pendientes: (prev.alertas_pendientes || 0) + (msg.alerta ? 1 : 0),
                total_vehiculos:
                  (prev.total_vehiculos || 0) + (msg.data.auto_registrado ? 1 : 0)
              };
            });
          } else if (msg.event === "alerta_atendida" && msg.data) {
            setAlertas((prev) => prev.map((a) => (a.id === msg.data.id ? msg.data : a)));
            fetchInitialData();
          } else if (msg.event === "hardware_estado") {
            setSummary((prev) =>
              prev ? { ...prev, hardware_conectado: msg.connected_devices } : prev
            );
          }
        } catch {
          // Ignorar mensajes no JSON
        }
      };
      ws.onclose = () => {
        setWsConnected(false);
        clearInterval(pingTimer);
        if (!closedByUser) {
          reconnectTimer = setTimeout(connect, 3000);
        }
      };
    }

    connect();
    // Respaldo de sincronización cada 6s por si la red móvil o proxy corta temporalmente el WebSocket
    const syncTimer = setInterval(() => {
      if (!mobileMode) fetchInitialData();
    }, 6000);

    return () => {
      closedByUser = true;
      clearTimeout(reconnectTimer);
      clearInterval(pingTimer);
      clearInterval(syncTimer);
      if (ws) ws.close();
    };
  }, [token, mobileMode, fetchInitialData]);

  if (!token || !user) {
    return <LoginPage />;
  }

  if (mobileMode) {
    return (
      <MobileAppView
        casetas={casetas}
        pasos={pasos}
        wsConnected={wsConnected}
        onRefresh={fetchInitialData}
        onExitMobile={() => {
          window.history.replaceState({}, "", "/");
          setMobileMode(false);
          fetchInitialData();
        }}
      />
    );
  }

  const unreadAlerts = alertas.filter((a) => !a.atendida).length;

  return (
    <div className="min-h-screen flex flex-col md:flex-row bg-slate-950 text-slate-100">
      {/* Sidebar */}
      <aside className="w-full md:w-64 bg-slate-900 border-b md:border-b-0 md:border-r border-slate-800 p-4 flex flex-col justify-between shrink-0">
        <div className="space-y-6">
          <div className="flex items-center gap-3 px-2">
            <div className="w-10 h-10 rounded-xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400">
              <Car className="w-5 h-5" />
            </div>
            <div>
              <h1 className="font-bold text-white leading-tight">Peaje IA</h1>
              <p className="text-[11px] text-slate-400">Control Inteligente</p>
            </div>
          </div>

          <nav className="flex md:flex-col gap-1 overflow-x-auto">
            {NAV_ITEMS.map(({ id, label, icon: Icon }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={`flex items-center justify-between px-3 py-2.5 rounded-xl text-sm font-medium transition whitespace-nowrap ${
                  activeTab === id
                    ? "bg-sky-500/15 text-sky-300 border border-sky-500/30"
                    : "text-slate-400 hover:bg-slate-800/70 hover:text-slate-200"
                }`}
              >
                <span className="flex items-center gap-2.5">
                  <Icon className="w-4 h-4" />
                  {label}
                </span>
                {id === "alertas" && unreadAlerts > 0 && (
                  <span className="bg-red-500 text-white text-[10px] font-bold px-2 py-0.5 rounded-full">
                    {unreadAlerts}
                  </span>
                )}
              </button>
            ))}
          </nav>
        </div>

        <div className="mt-6 pt-4 border-t border-slate-800 space-y-3">
          <div className="px-2">
            <p className="text-xs font-semibold text-white truncate">{user.nombre}</p>
            <p className="text-[11px] text-slate-400 truncate">{user.email_google}</p>
            <span className="mt-1 inline-flex items-center gap-1 text-[10px] uppercase font-bold px-2 py-0.5 rounded-full bg-sky-500/15 text-sky-300">
              <ShieldCheck className="w-3 h-3" /> {user.rol}
            </span>
          </div>
          <button
            onClick={() => {
              window.history.replaceState({}, "", "/movil");
              setMobileMode(true);
            }}
            className="w-full btn-primary text-xs flex items-center justify-center gap-2"
          >
            <Smartphone className="w-3.5 h-3.5" />
            Abrir App Móvil (/movil)
          </button>
          <button
            onClick={() => switchGoogleAccount()}
            className="w-full btn-secondary text-xs flex items-center justify-center gap-2 text-sky-300 border-sky-500/30 hover:bg-sky-500/10"
          >
            <RefreshCw className="w-3.5 h-3.5" />
            Cambiar cuenta Google
          </button>
          <button
            onClick={logout}
            className="w-full btn-secondary text-xs flex items-center justify-center gap-2"
          >
            <LogOut className="w-3.5 h-3.5" />
            Cerrar sesión
          </button>
        </div>
      </aside>

      {/* Main Content */}
      <main className="flex-1 p-5 md:p-7 overflow-y-auto">
        {activeTab === "panel" && (
          <DashboardPage
            summary={summary}
            pasos={pasos}
            alertas={alertas}
            casetas={casetas}
            wsConnected={wsConnected}
            onSelectImage={setModalImage}
            onRefresh={fetchInitialData}
          />
        )}
        {activeTab === "camara" && (
          <CamaraCasetaPage casetas={casetas} onRefresh={fetchInitialData} />
        )}
        {activeTab === "vehiculos" && (
          <VehiculosPage liveTrigger={pasos[0]?.id} onRefreshGlobal={fetchInitialData} />
        )}
        {activeTab === "pasos" && <PasosPage livePasos={pasos} onSelectImage={setModalImage} />}
        {activeTab === "alertas" && (
          <AlertasPage alertas={alertas} onRefresh={fetchInitialData} onSelectImage={setModalImage} />
        )}
        {activeTab === "chat" && <ChatPage />}
        {activeTab === "estado" && <EstadoPage />}
      </main>

      {/* Modal de evidencia */}
      {modalImage && (
        <div
          className="fixed inset-0 bg-black/80 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          onClick={() => setModalImage("")}
        >
          <div className="relative max-w-4xl w-full" onClick={(e) => e.stopPropagation()}>
            <button
              onClick={() => setModalImage("")}
              className="absolute -top-10 right-0 text-white hover:text-sky-400 flex items-center gap-1 text-sm"
            >
              <X className="w-5 h-5" /> Cerrar
            </button>
            <img
              src={modalImage}
              alt="Evidencia del paso"
              className="w-full max-h-[82vh] object-contain rounded-2xl border border-slate-700 bg-slate-950"
            />
          </div>
        </div>
      )}
    </div>
  );
}
