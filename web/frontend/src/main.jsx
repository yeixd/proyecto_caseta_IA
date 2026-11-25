import React, { useEffect, useState } from "react";
import ReactDOM from "react-dom/client";
import { GoogleOAuthProvider } from "@react-oauth/google";
import { api } from "./api";
import { AuthProvider } from "./context/AuthContext";
import App from "./App";
import "./index.css";

function isValidClientId(val) {
  return Boolean(val && !val.startsWith("TU_GOOGLE") && !val.includes("CONFIGURAR"));
}

function Bootstrap() {
  const envCid = (import.meta.env.VITE_GOOGLE_CLIENT_ID || "").trim();
  const [clientId, setClientId] = useState(isValidClientId(envCid) ? envCid : "");
  const [ready, setReady] = useState(isValidClientId(envCid));

  useEffect(() => {
    api
      .get("/api/v1/auth/config")
      .then((res) => {
        const backendCid = (res.data?.google_client_id || "").trim();
        if (isValidClientId(backendCid)) {
          setClientId(backendCid);
        }
      })
      .catch(() => {})
      .finally(() => setReady(true));
  }, []);

  if (!ready) {
    return (
      <div className="min-h-screen bg-slate-950 text-slate-300 flex items-center justify-center text-sm">
        Cargando configuración de Google OAuth...
      </div>
    );
  }

  return (
    <GoogleOAuthProvider clientId={clientId || "MISSING_GOOGLE_CLIENT_ID"}>
      <AuthProvider googleConfigured={Boolean(clientId)}>
        <App />
      </AuthProvider>
    </GoogleOAuthProvider>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(
  <React.StrictMode>
    <Bootstrap />
  </React.StrictMode>
);
