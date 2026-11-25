import React, { createContext, useContext, useEffect, useState } from "react";
import { googleLogout } from "@react-oauth/google";
import { api } from "../api";

const AuthContext = createContext(null);

function resetGoogleSession() {
  try {
    googleLogout();
    window.google?.accounts?.id?.disableAutoSelect?.();
    window.google?.accounts?.id?.cancel?.();
  } catch {
    // Ignorar si GSI aún no cargó
  }
}

export function AuthProvider({ children, googleConfigured = false }) {
  const [token, setToken] = useState(() => localStorage.getItem("peaje_token") || "");
  const [user, setUser] = useState(() => {
    try {
      const saved = localStorage.getItem("peaje_user");
      return saved ? JSON.parse(saved) : null;
    } catch {
      return null;
    }
  });
  const [authError, setAuthError] = useState("");
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!token) return;
    api
      .get("/api/v1/auth/me")
      .then((res) => {
        setUser(res.data);
        localStorage.setItem("peaje_user", JSON.stringify(res.data));
      })
      .catch(() => {
        logout();
      });
  }, [token]);

  async function _completeAuth(payload) {
    setLoading(true);
    setAuthError("");
    try {
      const { data } = await api.post("/api/v1/auth/google", payload);
      setToken(data.access_token);
      setUser(data.usuario);
      localStorage.setItem("peaje_token", data.access_token);
      localStorage.setItem("peaje_user", JSON.stringify(data.usuario));
    } catch (err) {
      resetGoogleSession();
      const detail =
        err?.response?.data?.detail ||
        "Error al autenticar con Google. Verifica tu conexión o solicita autorización.";
      setAuthError(detail);
      setToken("");
      setUser(null);
    } finally {
      setLoading(false);
    }
  }

  async function loginWithGoogleToken(googleIdToken) {
    return _completeAuth({ id_token: googleIdToken });
  }

  async function loginWithGoogleAccessToken(googleAccessToken) {
    return _completeAuth({ access_token: googleAccessToken });
  }

  function logout() {
    resetGoogleSession();
    setToken("");
    setUser(null);
    setAuthError("");
    localStorage.removeItem("peaje_token");
    localStorage.removeItem("peaje_user");
  }

  return (
    <AuthContext.Provider
      value={{
        token,
        user,
        isAdmin: user?.rol === "administrador",
        googleConfigured,
        authError,
        loading,
        loginWithGoogleToken,
        loginWithGoogleAccessToken,
        logout
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  return useContext(AuthContext);
}
