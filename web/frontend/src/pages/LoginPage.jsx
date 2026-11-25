import React from "react";
import { GoogleLogin, useGoogleLogin } from "@react-oauth/google";
import { ShieldAlert, ShieldCheck, Car, Cpu, RefreshCw } from "lucide-react";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const {
    loginWithGoogleToken,
    loginWithGoogleAccessToken,
    googleConfigured,
    authError,
    loading
  } = useAuth();
  const hasGoogleClient = Boolean(googleConfigured);

  // Fuerza siempre el selector de cuentas de Google (prompt=select_account)
  // para poder elegir un correo secundario sin quedar anclado al correo principal del navegador
  const selectAnotherAccount = useGoogleLogin({
    scope: "openid email profile",
    prompt: "select_account",
    onSuccess: (tokenResponse) => {
      if (tokenResponse?.access_token) {
        loginWithGoogleAccessToken(tokenResponse.access_token);
      }
    },
    onError: () => {
      alert("No se pudo abrir el selector de cuentas de Google.");
    }
  });

  return (
    <div className="min-h-screen bg-slate-950 flex items-center justify-center p-4">
      <div className="max-w-md w-full card p-8 space-y-6">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-sky-500/15 border border-sky-500/30 flex items-center justify-center text-sky-400">
            <Car className="w-6 h-6" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-white">Sistema Inteligente de Peaje</h1>
            <p className="text-xs text-slate-400">IAM-811 · Machine Learning y Deep Learning</p>
          </div>
        </div>

        <div className="bg-slate-950/80 border border-slate-800 rounded-xl p-4 text-xs text-slate-300 space-y-2">
          <div className="flex items-center gap-2 text-sky-400 font-medium">
            <ShieldCheck className="w-4 h-4" />
            Acceso exclusivo con Google OAuth 2.0
          </div>
          <p className="text-slate-400">
            Solo cuentas institucionales <strong className="text-sky-300">@unifranz.edu.bo</strong>{" "}
            autorizadas en la lista blanca pueden ingresar. Pulsa{" "}
            <strong className="text-slate-200">Elegir cuenta de Google</strong> para seleccionar tu
            correo institucional.
          </p>
        </div>

        {authError && (
          <div className="bg-red-500/15 border border-red-500/40 rounded-xl p-4 flex gap-3 items-start">
            <ShieldAlert className="w-5 h-5 text-red-400 shrink-0 mt-0.5" />
            <div className="space-y-2">
              <p className="text-sm font-semibold text-red-300">Acceso denegado</p>
              <p className="text-xs text-red-200/90">{authError}</p>
            </div>
          </div>
        )}

        {!hasGoogleClient ? (
          <div className="bg-amber-500/15 border border-amber-500/40 rounded-xl p-4 text-xs text-amber-100 space-y-2">
            <p className="font-semibold text-amber-300">
              Falta configurar GOOGLE_CLIENT_ID (evita Error 401: invalid_client)
            </p>
            <ol className="list-decimal pl-4 space-y-1 text-amber-200/90">
              <li>
                Entra a <strong>console.cloud.google.com</strong> → APIs y servicios → Credenciales
                → Crear ID de cliente OAuth (Aplicación web).
              </li>
              <li>
                En <strong>Orígenes autorizados de JavaScript</strong> agrega{" "}
                <code className="font-mono bg-slate-900 px-1 rounded">http://localhost:8000</code>{" "}
                y tu subdominio HTTPS de producción.
              </li>
              <li>
                Pega tu Client ID en <code className="font-mono">.env</code> como{" "}
                <code className="font-mono">GOOGLE_CLIENT_ID=...</code> y reinicia el servidor.
              </li>
            </ol>
          </div>
        ) : (
          <div className="flex flex-col items-center justify-center gap-3 py-1">
            {loading ? (
              <div className="text-sm text-sky-400 animate-pulse py-3">
                Verificando credenciales con Google y lista blanca...
              </div>
            ) : (
              <>
                <button
                  type="button"
                  onClick={() => selectAnotherAccount()}
                  className="w-full btn-primary py-3 text-sm flex items-center justify-center gap-2"
                >
                  <RefreshCw className="w-4 h-4" />
                  Elegir cuenta de Google (cambiar correo)
                </button>

                <div className="w-full flex items-center gap-3 text-[11px] text-slate-500 my-1">
                  <div className="h-px flex-1 bg-slate-800" />
                  <span>o inicio rápido</span>
                  <div className="h-px flex-1 bg-slate-800" />
                </div>

                <GoogleLogin
                  onSuccess={(credentialResponse) => {
                    if (credentialResponse?.credential) {
                      loginWithGoogleToken(credentialResponse.credential);
                    }
                  }}
                  onError={() => {
                    alert("No se pudo completar el inicio de sesión con Google.");
                  }}
                  auto_select={false}
                  useOneTap={false}
                  theme="filled_black"
                  size="large"
                  shape="pill"
                  text="signin_with"
                  locale="es"
                />
              </>
            )}
          </div>
        )}

        <div className="pt-4 border-t border-slate-800 flex items-center justify-between text-[11px] text-slate-500">
          <span className="flex items-center gap-1">
            <Cpu className="w-3.5 h-3.5" /> M1+OCR · M2 Seg · M3 Pose · M4 Pasajeros
          </span>
          <span>UNIFRANZ</span>
        </div>
      </div>
    </div>
  );
}
