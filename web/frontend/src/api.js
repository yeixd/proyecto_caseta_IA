import axios from "axios";

export const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || "").replace(/\/$/, "");

export const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000
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
