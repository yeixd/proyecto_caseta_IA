import React, { useState } from "react";
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
import { sendAgentQuery } from "../api";

export default function ConsultaScreen({ token }) {
  const [messages, setMessages] = useState([
    {
      role: "assistant",
      text: "Modo Consulta activo. Escribe una placa (ej. 2481-KLM) o toma una foto de la placa para consultar su estado oficial.",
      estado: null,
      sugerencias: []
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
      ? await ImagePicker.launchCameraAsync({ quality: 0.7 })
      : await ImagePicker.launchImageLibraryAsync({ quality: 0.7 });

    if (!result.canceled && result.assets?.[0]?.uri) {
      setSelectedImage(result.assets[0].uri);
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
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          text: data.respuesta,
          estado: data.estado,
          placa: data.placa_normalizada,
          sugerencias: data.sugerencias || []
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
      <ScrollView style={styles.chat} contentContainerStyle={{ padding: 14, gap: 10 }}>
        {messages.map((m, idx) => (
          <View
            key={idx}
            style={[styles.bubble, m.role === "user" ? styles.userBubble : styles.botBubble]}
          >
            {m.image && <Image source={{ uri: m.image }} style={styles.previewImg} />}
            {m.estado ? (
              <Text style={styles.estadoBadge}>
                {m.estado} {m.placa ? `· ${m.placa}` : ""}
              </Text>
            ) : null}
            <Text style={m.role === "user" ? styles.userText : styles.botText}>{m.text}</Text>
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
          <View style={styles.selectedWrap}>
            <Image source={{ uri: selectedImage }} style={styles.thumb} />
            <TouchableOpacity onPress={() => setSelectedImage(null)}>
              <Text style={{ color: "#f87171", fontSize: 12 }}>Quitar foto</Text>
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
