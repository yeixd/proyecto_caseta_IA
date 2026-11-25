import * as SecureStore from "expo-secure-store";

const DEFAULT_API_BASE_URL = (
  process.env.EXPO_PUBLIC_API_BASE_URL || "https://peaje-ia-web.onrender.com"
).replace(/\/$/, "");

const TOKEN_KEY = "peaje_jwt_token";
const USER_KEY = "peaje_user_json";
const API_URL_KEY = "peaje_api_base_url";

export async function getApiBaseUrl() {
  const custom = await SecureStore.getItemAsync(API_URL_KEY);
  return (custom || DEFAULT_API_BASE_URL).replace(/\/$/, "");
}

export async function setApiBaseUrl(url) {
  const clean = (url || "").trim().replace(/\/$/, "");
  if (clean) {
    await SecureStore.setItemAsync(API_URL_KEY, clean);
  } else {
    await SecureStore.deleteItemAsync(API_URL_KEY);
  }
}

export async function saveSession(token, user) {
  await SecureStore.setItemAsync(TOKEN_KEY, token);
  await SecureStore.setItemAsync(USER_KEY, JSON.stringify(user));
}

export async function loadSession() {
  const token = await SecureStore.getItemAsync(TOKEN_KEY);
  const userStr = await SecureStore.getItemAsync(USER_KEY);
  if (!token || !userStr) return null;
  try {
    return { token, user: JSON.parse(userStr) };
  } catch {
    return null;
  }
}

export async function clearSession() {
  await SecureStore.deleteItemAsync(TOKEN_KEY);
  await SecureStore.deleteItemAsync(USER_KEY);
}

export async function loginWithGoogleTokens({ idToken = null, accessToken = null }) {
  const baseUrl = await getApiBaseUrl();
  const payload = idToken ? { id_token: idToken } : { access_token: accessToken };
  const res = await fetch(`${baseUrl}/api/v1/auth/google`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || "Acceso denegado o error de autenticación con Google.");
  }
  await saveSession(data.access_token, data.usuario);
  return data;
}

export async function fetchCasetas(token) {
  const baseUrl = await getApiBaseUrl();
  const res = await fetch(`${baseUrl}/api/v1/casetas`, {
    headers: { Authorization: `Bearer ${token}` }
  });
  if (!res.ok) throw new Error("No se pudieron cargar las casetas.");
  return res.json();
}

export async function sendCasetaFrame(token, imageUri, casetaId) {
  const baseUrl = await getApiBaseUrl();
  const formData = new FormData();
  formData.append("caseta_id", String(casetaId));
  formData.append("file", {
    uri: imageUri,
    name: `frame_${Date.now()}.jpg`,
    type: "image/jpeg"
  });

  const res = await fetch(`${baseUrl}/api/v1/pasos/frame`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: formData
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Error HTTP ${res.status} enviando frame.`);
  }
  return data;
}

export async function sendAgentQuery(token, mensaje, imageUri = null) {
  const baseUrl = await getApiBaseUrl();
  const formData = new FormData();
  if (mensaje) {
    formData.append("mensaje", mensaje);
  }
  if (imageUri) {
    formData.append("file", {
      uri: imageUri,
      name: `consulta_${Date.now()}.jpg`,
      type: "image/jpeg"
    });
  }

  const res = await fetch(`${baseUrl}/api/v1/agente/chat`, {
    method: "POST",
    headers: {
      Authorization: `Bearer ${token}`
    },
    body: formData
  });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) {
    throw new Error(data.detail || `Error HTTP ${res.status} consultando al agente.`);
  }
  return data;
}
