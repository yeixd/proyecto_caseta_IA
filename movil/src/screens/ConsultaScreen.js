import React, { useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TextInput,
  TouchableOpacity,
  View
} from "react-native";
import * as ImagePicker from "expo-image-picker";
import { formatVehicleClass, getApiBaseUrl, sendAgentQuery, sendCasetaFrame } from "../api";

const QUICK_PROMPTS = [
  { label: "📊 Resumen hoy", prompt: "Dame el resumen del día y recaudación" },
  { label: "🚨 Alertas", prompt: "Muéstrame las alertas activas pendientes" },
  { label: "💰 Tarifas", prompt: "Cuáles son las tarifas vigentes por vehículo y ejes" },
  { label: "🚗 Últimos pasos", prompt: "Muéstrame los últimos 5 pasos registrados" },
  { label: "🔎 2481-KLM", prompt: "2481-KLM" }
];

export default function ConsultaScreen({ token }) {
  const scrollRef = useRef(null);
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Hola 👋 Soy tu Agente IA Móvil de Peaje.\n• Escribe una placa (ej. 2481-KLM), envía una foto (con Super-OCR para placas borrosas) o toca un acceso rápido.",
      estado: null,
      sugerencias: [],
      datos: null,
      motor: null
    }
  ]);
  const [input, setInput] = useState("");
  const [selectedImage, setSelectedImage] = useState(null);
  const [sending, setSending] = useState(false);

  async function pickPhoto(useCamera) {
    const perm = useCamera
      ? await ImagePicker.requestCameraPermissionsAsync()
      : await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return;

    const result = useCamera
      ? await ImagePicker.launchCameraAsync({ quality: 0.88 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.88 });

    if (!result.canceled && result.assets?.[0]?.uri) {
      setSelectedImage(result.assets[0].uri);
    }
  }

  async function registerImageAsPaso() {
    if (!selectedImage || sending) return;
    const uri = selectedImage;
    setSelectedImage(null);
    setSending(true);
    setMessages((prev) => [
      ...prev,
      { role: "user", text: "Registrar paso en Panel Web", image: uri }
    ]);
    try {
      const data = await sendCasetaFrame(token, uri, 1);
      const p = data.paso;
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: `✅ Registro #${p.id} enviado a la Página Web:\n• Placa: ${
            p.placa_normalizada || "ILEGIBLE"
          }\n• Clase: ${formatVehicleClass(p.categoria_detectada)}\n• Ejes: ${
            p.ejes_detectados
          }\n• Pasajeros: ${p.pasajeros}\n• Tarifa: Bs. ${Number(p.monto).toFixed(2)} (${p.estado})`,
          estado: p.estado,
          placa: p.placa_normalizada,
          sugerencias: [],
          motor: "caseta_en_vivo"
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: err.message, estado: null, sugerencias: [] }
      ]);
    } finally {
      setSending(false);
    }
  }

  async function handleSend(overrideText) {
    const msgText = typeof overrideText === "string" ? overrideText : input.trim();
    const imgUri = typeof overrideText === "string" ? null : selectedImage;
    if (!msgText && !imgUri) return;

    setMessages((prev) => [
      ...prev,
      { role: "user", text: msgText || "Foto de placa enviada", image: imgUri }
    ]);
    setInput("");
    setSelectedImage(null);
    setSending(true);

    try {
      const data = await sendAgentQuery(token, msgText, imgUri);
      const baseUrl = await getApiBaseUrl();
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.respuesta,
          estado: data.estado,
          placa: data.placa_normalizada,
          sugerencias: data.sugerencias || [],
          datos: data.datos || null,
          inferencia: data.inferencia ? { ...data.inferencia, baseUrl } : null,
          ocr: data.ocr || null,
          motor: data.motor || null
        }
      ]);
    } catch (err) {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: err.message, estado: null, sugerencias: [] }
      ]);
    } finally {
      setSending(false);
    }
  }

  return (
    <View style={styles.container}>
      <View style={{ paddingHorizontal: 12, paddingTop: 10, paddingBottom: 4 }}>
        <ScrollView horizontal showsHorizontalScrollIndicator={false}>
          {QUICK_PROMPTS.map((q) => (
            <TouchableOpacity
              key={q.label}
              disabled={sending}
              onPress={() => handleSend(q.prompt)}
              style={{
                backgroundColor: "#0f172a",
                borderColor: "#1e293b",
                borderWidth: 1,
                paddingHorizontal: 11,
                paddingVertical: 6,
                borderRadius: 12,
                marginRight: 8
              }}
            >
              <Text style={{ color: "#38bdf8", fontSize: 12, fontWeight: "700" }}>{q.label}</Text>
            </TouchableOpacity>
          ))}
        </ScrollView>
      </View>

      <ScrollView
        ref={scrollRef}
        onContentSizeChange={() => scrollRef.current?.scrollToEnd({ animated: true })}
        style={styles.chat}
        contentContainerStyle={{ padding: 14, gap: 10 }}
      >
        {messages.map((m, idx) => (
          <View
            key={idx}
            style={[styles.bubble, m.role === "user" ? styles.userBubble : styles.botBubble]}
          >
            {m.image && <Image source={{ uri: m.image }} style={styles.previewImg} />}
            {m.inferencia?.imagen_url && m.inferencia?.baseUrl ? (
              <Image
                source={{
                  uri: m.inferencia.imagen_url.startsWith("http")
                    ? m.inferencia.imagen_url
                    : `${m.inferencia.baseUrl}${m.inferencia.imagen_url}`
                }}
                style={{
                  width: 240,
                  height: 155,
                  borderRadius: 10,
                  marginBottom: 8,
                  backgroundColor: "#020617"
                }}
                resizeMode="contain"
              />
            ) : null}
            {m.inferencia ? (
              <View
                style={{
                  marginBottom: 8,
                  padding: 8,
                  borderRadius: 10,
                  backgroundColor: "#020617",
                  borderWidth: 1,
                  borderColor: "rgba(52,211,153,0.4)"
                }}
              >
                <Text style={{ color: "#34d399", fontSize: 11, fontWeight: "700" }}>
                  {formatVehicleClass(m.inferencia.categoria)} · {m.inferencia.ejes} ejes (
                  {m.inferencia.llantas_visibles ?? 0} llantas) · {m.inferencia.pasajeros} pasaj. · Bs.{" "}
                  {Number(m.inferencia.tarifa_bs || 0).toFixed(2)}
                </Text>
              </View>
            ) : null}
            {m.estado ? (
              <Text style={styles.estadoBadge}>
                {m.estado} {m.placa ? `· ${m.placa}` : ""}{" "}
                {m.motor?.startsWith("ollama:") ? "· ✨ LLM" : ""}
              </Text>
            ) : null}
            <Text style={m.role === "user" ? styles.userText : styles.botText}>{m.text}</Text>
            {m.datos?.vehiculo && (
              <View
                style={{
                  marginTop: 8,
                  padding: 8,
                  borderRadius: 10,
                  backgroundColor: "#020617",
                  borderWidth: 1,
                  borderColor: "#1e293b"
                }}
              >
                <Text style={{ color: "#94a3b8", fontSize: 11 }}>
                  Propietario: <Text style={{ color: "#fff", fontWeight: "700" }}>{m.datos.vehiculo.propietario}</Text>
                </Text>
                <Text style={{ color: "#38bdf8", fontSize: 11, marginTop: 2 }}>
                  {formatVehicleClass(m.datos.vehiculo.categoria)} ({m.datos.vehiculo.ejes} ejes) · Saldo: Bs.{" "}
                  {Number(m.datos.vehiculo.saldo).toFixed(2)}
                </Text>
              </View>
            )}
            {m.sugerencias?.length > 0 && (
              <View style={styles.sugRow}>
                <Text style={styles.sugLabel}>¿Quisiste decir:</Text>
                {m.sugerencias.map((s) => (
                  <TouchableOpacity key={s} style={styles.sugChip} onPress={() => handleSend(s)}>
                    <Text style={styles.sugChipText}>{s}</Text>
                  </TouchableOpacity>
                ))}
              </View>
            )}
          </View>
        ))}
        {sending && <ActivityIndicator color="#38bdf8" style={{ marginTop: 6 }} />}
      </ScrollView>

      <View style={styles.inputArea}>
        {selectedImage && (
          <View style={[styles.selectedWrap, { justifyContent: "space-between" }]}>
            <View style={{ flexDirection: "row", alignItems: "center", gap: 10 }}>
              <Image source={{ uri: selectedImage }} style={styles.thumb} />
              <TouchableOpacity onPress={() => setSelectedImage(null)}>
                <Text style={{ color: "#f87171", fontSize: 12 }}>Quitar</Text>
              </TouchableOpacity>
            </View>
            <TouchableOpacity
              onPress={registerImageAsPaso}
              disabled={sending}
              style={{
                backgroundColor: "#0284c7",
                paddingHorizontal: 10,
                paddingVertical: 6,
                borderRadius: 8
              }}
            >
              <Text style={{ color: "#fff", fontSize: 11, fontWeight: "700" }}>
                Registrar Paso en Web
              </Text>
            </TouchableOpacity>
          </View>
        )}

        <View style={styles.photoBtns}>
          <TouchableOpacity style={styles.photoBtn} onPress={() => pickPhoto(true)}>
            <Text style={styles.photoBtnText}>Tomar Foto</Text>
          </TouchableOpacity>
          <TouchableOpacity style={styles.photoBtn} onPress={() => pickPhoto(false)}>
            <Text style={styles.photoBtnText}>Galería</Text>
          </TouchableOpacity>
        </View>

        <View style={styles.inputRow}>
          <TextInput
            style={styles.textInput}
            placeholder="Escribe una placa (ej. 2481-KLM)..."
            placeholderTextColor="#64748b"
            value={input}
            onChangeText={setInput}
            autoCapitalize="characters"
          />
          <TouchableOpacity
            style={styles.sendBtn}
            disabled={sending}
            onPress={() => handleSend()}
          >
            <Text style={styles.sendBtnText}>Enviar</Text>
          </TouchableOpacity>
        </View>
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#020617" },
  chat: { flex: 1 },
  bubble: { maxWidth: "88%", borderRadius: 16, padding: 12 },
  userBubble: { alignSelf: "flex-end", backgroundColor: "#38bdf8" },
  botBubble: {
    alignSelf: "flex-start",
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#1e293b"
  },
  userText: { color: "#020617", fontWeight: "700", fontSize: 14 },
  botText: { color: "#e2e8f0", fontSize: 13, lineHeight: 19 },
  estadoBadge: {
    color: "#38bdf8",
    fontSize: 11,
    fontWeight: "800",
    marginBottom: 4
  },
  previewImg: { width: 190, height: 120, borderRadius: 10, marginBottom: 8 },
  sugRow: { flexDirection: "row", flexWrap: "wrap", alignItems: "center", gap: 6, marginTop: 8 },
  sugLabel: { color: "#fbbf24", fontSize: 12 },
  sugChip: {
    backgroundColor: "#451a03",
    borderColor: "#f59e0b",
    borderWidth: 1,
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 8
  },
  sugChipText: { color: "#fde68a", fontWeight: "700", fontSize: 12 },
  inputArea: {
    padding: 12,
    backgroundColor: "#0f172a",
    borderTopWidth: 1,
    borderColor: "#1e293b",
    gap: 8
  },
  selectedWrap: { flexDirection: "row", alignItems: "center", gap: 10 },
  thumb: { width: 56, height: 42, borderRadius: 8 },
  photoBtns: { flexDirection: "row", gap: 8 },
  photoBtn: {
    backgroundColor: "#1e293b",
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 10
  },
  photoBtnText: { color: "#cbd5e1", fontSize: 12, fontWeight: "600" },
  inputRow: { flexDirection: "row", gap: 8 },
  textInput: {
    flex: 1,
    backgroundColor: "#020617",
    borderWidth: 1,
    borderColor: "#334155",
    borderRadius: 12,
    paddingHorizontal: 12,
    paddingVertical: 10,
    color: "#ffffff",
    fontSize: 14
  },
  sendBtn: {
    backgroundColor: "#38bdf8",
    paddingHorizontal: 16,
    borderRadius: 12,
    justifyContent: "center"
  },
  sendBtnText: { color: "#020617", fontWeight: "800" }
});
