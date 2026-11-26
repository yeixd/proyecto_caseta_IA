import axios from "axios";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 65000
});

api.interceptors.request.use((config) => {
  const token = localStorage.getItem("peaje_token");
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error?.response?.status === 401 && localStorage.getItem("peaje_token")) {
      localStorage.removeItem("peaje_token");
      localStorage.removeItem("peaje_user");
      window.location.reload();
    }
    return Promise.reject(error);
  }
);

export function buildPanelWsUrl(token) {
  const envWs = (import.meta.env.VITE_WS_BASE_URL || "").replace(/\/$/, "");
  if (envWs) {
    return `${envWs}/ws/panel?token=${encodeURIComponent(token)}`;
  }
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/panel?token=${encodeURIComponent(token)}`;
}

export function resolveImageUrl(url) {
  if (!url) return "";
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  return `${API_BASE_URL}${url}`;
}

export const VEHICLE_CLASS_LABELS = {
  auto: "Vehículo particular (Auto)",
  minibus: "Minibús",
  moto: "Motocicleta",
  camion: "Camión",
  bus: "Bus / Flota"
};

export const VEHICLE_CLASS_COLORS = {
  auto: "bg-sky-500/15 text-sky-300 border-sky-500/40",
  minibus: "bg-emerald-500/15 text-emerald-300 border-emerald-500/40",
  moto: "bg-fuchsia-500/15 text-fuchsia-300 border-fuchsia-500/40",
  camion: "bg-amber-500/15 text-amber-300 border-amber-500/40",
  bus: "bg-rose-500/15 text-rose-300 border-rose-500/40"
};

export function formatVehicleClass(cat) {
  if (!cat) return "Vehículo particular (Auto)";
  const key = String(cat).toLowerCase().trim();
  return VEHICLE_CLASS_LABELS[key] || cat;
}

/**
 * Redimensiona y comprime fotos pesadas de cámaras de celular (8-15 MB) a ~180 KB (máx 1280px)
 * para envío instantáneo desde el móvil hacia Render + Google Colab sin timeouts.
 */
export async function compressImageForUpload(fileOrBlob, maxDim = 1280, quality = 0.82) {
  if (!fileOrBlob || typeof document === "undefined") return fileOrBlob;
  return new Promise((resolve) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(fileOrBlob);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      let { width, height } = img;
      if (!width || !height) {
        resolve(fileOrBlob);
        return;
      }
      if (width > maxDim || height > maxDim) {
        const scale = maxDim / Math.max(width, height);
        width = Math.round(width * scale);
        height = Math.round(height * scale);
      }
      const canvas = document.createElement("canvas");
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext("2d");
      ctx.drawImage(img, 0, 0, width, height);
      canvas.toBlob(
        (blob) => resolve(blob || fileOrBlob),
        "image/jpeg",
        quality
      );
    };
    img.onerror = () => {
      URL.revokeObjectURL(objectUrl);
      resolve(fileOrBlob);
    };
    img.src = objectUrl;
  });
}
