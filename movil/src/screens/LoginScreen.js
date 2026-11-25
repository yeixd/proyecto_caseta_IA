import React, { useEffect, useState } from "react";
import {
  ActivityIndicator,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from "react-native";
import * as WebBrowser from "expo-web-browser";
import * as Google from "expo-auth-session/providers/google";
import { getApiBaseUrl, loginWithGoogleTokens, setApiBaseUrl } from "../api";

WebBrowser.maybeCompleteAuthSession();

export default function LoginScreen({ onAuthenticated }) {
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [serverUrl, setServerUrl] = useState("");

  useEffect(() => {
    getApiBaseUrl().then(setServerUrl);
  }, []);

  const [request, response, promptAsync] = Google.useAuthRequest({
    clientId: process.env.EXPO_PUBLIC_GOOGLE_WEB_CLIENT_ID,
    androidClientId: process.env.EXPO_PUBLIC_GOOGLE_ANDROID_CLIENT_ID,
    iosClientId: process.env.EXPO_PUBLIC_GOOGLE_IOS_CLIENT_ID,
    scopes: ["openid", "email", "profile"],
    prompt: "select_account"
  });

  useEffect(() => {
    async function handleGoogleResponse() {
      if (response?.type === "success") {
        const idToken = response.params?.id_token || response.authentication?.idToken || null;
        const accessToken = response.authentication?.accessToken || null;
        if (!idToken && !accessToken) {
          setErrorMsg("Google no devolvió un token válido.");
          return;
        }
        setLoading(true);
        setErrorMsg("");
        try {
          await setApiBaseUrl(serverUrl);
          const data = await loginWithGoogleTokens({ idToken, accessToken });
          onAuthenticated(data.access_token, data.usuario);
        } catch (err) {
          setErrorMsg(err.message);
        } finally {
          setLoading(false);
        }
      } else if (response?.type === "error") {
        setErrorMsg("No se pudo completar el inicio de sesión con Google.");
      }
    }
    handleGoogleResponse();
  }, [response]);

  return (
    <View style={styles.container}>
      <View style={styles.card}>
        <Text style={styles.badge}>IAM-811 · UNIFRANZ</Text>
        <Text style={styles.title}>Peaje IA — App Móvil</Text>
        <Text style={styles.subtitle}>
          Tu celular captura video en Modo Caseta para analizar placas, categoría, ejes y pasajeros
          en la nube en tiempo real.
        </Text>

        <View style={styles.infoBox}>
          <Text style={styles.infoText}>
            Acceso exclusivo para cuentas institucionales @unifranz.edu.bo autorizadas en la lista
            blanca. Siempre podrás elegir tu cuenta institucional al iniciar sesión.
          </Text>
        </View>

        <Text style={styles.label}>URL del servidor web (subdominio HTTPS):</Text>
        <TextInput
          style={styles.urlInput}
          value={serverUrl}
          onChangeText={setServerUrl}
          onEndEditing={() => setApiBaseUrl(serverUrl)}
          autoCapitalize="none"
          autoCorrect={false}
          placeholder="https://peaje-ia-web.onrender.com"
          placeholderTextColor="#64748b"
        />

        {errorMsg ? (
          <View style={styles.errorBox}>
            <Text style={styles.errorTitle}>Acceso denegado</Text>
            <Text style={styles.errorText}>{errorMsg}</Text>
          </View>
        ) : null}

        <TouchableOpacity
          style={[styles.googleBtn, (!request || loading) && styles.btnDisabled]}
          disabled={!request || loading}
          onPress={async () => {
            setErrorMsg("");
            await setApiBaseUrl(serverUrl);
            promptAsync();
          }}
        >
          {loading ? (
            <ActivityIndicator color="#0f172a" />
          ) : (
            <Text style={styles.googleBtnText}>Elegir cuenta Google (@unifranz.edu.bo)</Text>
          )}
        </TouchableOpacity>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: "#020617",
    justifyContent: "center",
    padding: 20
  },
  card: {
    backgroundColor: "#0f172a",
    borderRadius: 20,
    padding: 24,
    borderWidth: 1,
    borderColor: "#1e293b"
  },
  badge: {
    color: "#38bdf8",
    fontSize: 12,
    fontWeight: "700",
    marginBottom: 6
  },
  title: {
    color: "#f8fafc",
    fontSize: 24,
    fontWeight: "800"
  },
  subtitle: {
    color: "#94a3b8",
    fontSize: 13,
    marginTop: 6,
    marginBottom: 14
  },
  infoBox: {
    backgroundColor: "#020617",
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: "#1e293b",
    marginBottom: 14
  },
  infoText: {
    color: "#cbd5e1",
    fontSize: 12,
    lineHeight: 18
  },
  label: {
    color: "#94a3b8",
    fontSize: 11,
    marginBottom: 6
  },
  urlInput: {
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 9,
    color: "#e2e8f0",
    fontSize: 12,
    marginBottom: 14
  },
  errorBox: {
    backgroundColor: "#450a0a",
    borderRadius: 12,
    padding: 12,
    borderWidth: 1,
    borderColor: "#dc2626",
    marginBottom: 14
  },
  errorTitle: {
    color: "#fca5a5",
    fontWeight: "700",
    fontSize: 13
  },
  errorText: {
    color: "#fecaca",
    fontSize: 12,
    marginTop: 4
  },
  googleBtn: {
    backgroundColor: "#38bdf8",
    paddingVertical: 14,
    borderRadius: 14,
    alignItems: "center"
  },
  btnDisabled: {
    opacity: 0.6
  },
  googleBtnText: {
    color: "#020617",
    fontWeight: "800",
    fontSize: 14
  }
});
