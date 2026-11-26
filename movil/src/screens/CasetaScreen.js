import React, { useEffect, useRef, useState } from "react";
import {
  ActivityIndicator,
  Image,
  ScrollView,
  StyleSheet,
  Text,
  TouchableOpacity,
  View
} from "react-native";
import { CameraView, useCameraPermissions } from "expo-camera";
import * as ImagePicker from "expo-image-picker";
import { fetchCasetas, formatVehicleClass, sendCasetaFrame } from "../api";

const FPS_OPTIONS = [
  { label: "2 FPS", ms: 500 },
  { label: "3 FPS", ms: 333 },
  { label: "5 FPS", ms: 200 }
];

export default function CasetaScreen({ token }) {
  const [permission, requestPermission] = useCameraPermissions();
  const cameraRef = useRef(null);
  const inFlightRef = useRef(false);

  const [casetas, setCasetas] = useState([{ id: 1, nombre: "Caseta Autopista Norte" }]);
  const [casetaId, setCasetaId] = useState(1);
  const [fpsIdx, setFpsIdx] = useState(0);
  const [facing, setFacing] = useState("back");
  const [torch, setTorch] = useState(false);
  const [streaming, setStreaming] = useState(false);
  const [bursting, setBursting] = useState(false);
  const [sending, setSending] = useState(false);
  const [lastResult, setLastResult] = useState(null);
  const [history, setHistory] = useState([]);
  const [errorMsg, setErrorMsg] = useState("");
  const [framesSent, setFramesSent] = useState(0);

  useEffect(() => {
    fetchCasetas(token)
      .then((list) => {
        if (Array.isArray(list) && list.length > 0) {
          setCasetas(list);
          setCasetaId(list[0].id);
        }
      })
      .catch(() => {});
  }, [token]);

  async function captureAndSend() {
    if (!cameraRef.current || inFlightRef.current) return;
    inFlightRef.current = true;
    setSending(true);
    setErrorMsg("");
    try {
      const photo = await cameraRef.current.takePictureAsync({
        quality: 0.85,
        skipProcessing: false
      });
      if (!photo?.uri) return;
      const data = await sendCasetaFrame(token, photo.uri, casetaId);
      setLastResult(data);
      if (data?.paso) {
        setHistory((prev) => [data.paso, ...prev].slice(0, 6));
      }
      setFramesSent((n) => n + 1);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      inFlightRef.current = false;
      setSending(false);
    }
  }

  async function pickGalleryAndSend() {
    if (inFlightRef.current) return;
    const perm = await ImagePicker.requestMediaLibraryPermissionsAsync();
    if (!perm.granted) return;
    const result = await ImagePicker.launchImageLibraryAsync({ quality: 0.88 });
    if (result.canceled || !result.assets?.[0]?.uri) return;

    inFlightRef.current = true;
    setSending(true);
    setErrorMsg("");
    try {
      const data = await sendCasetaFrame(token, result.assets[0].uri, casetaId);
      setLastResult(data);
      if (data?.paso) {
        setHistory((prev) => [data.paso, ...prev].slice(0, 6));
      }
      setFramesSent((n) => n + 1);
    } catch (err) {
      setErrorMsg(err.message);
    } finally {
      inFlightRef.current = false;
      setSending(false);
    }
  }

  // Envío automático configurable (2 a 5 fps)
  useEffect(() => {
    if (!streaming) return;
    const intervalMs = FPS_OPTIONS[fpsIdx].ms;
    const timer = setInterval(() => {
      captureAndSend();
    }, intervalMs);
    return () => clearInterval(timer);
  }, [streaming, fpsIdx, casetaId, token]);

  // Ráfaga corta de 5 segundos (simula análisis de clip corto 5s)
  function startShortBurst() {
    if (streaming || bursting) return;
    setBursting(true);
    setStreaming(true);
    setTimeout(() => {
      setStreaming(false);
      setBursting(false);
    }, 5000);
  }

  if (!permission) {
    return (
      <View style={styles.center}>
        <ActivityIndicator color="#38bdf8" />
      </View>
    );
  }

  if (!permission.granted) {
    return (
      <View style={styles.center}>
        <Text style={styles.permText}>
          Se necesita permiso de cámara para analizar placas, categoría, ejes y pasajeros en Modo
          Caseta.
        </Text>
        <TouchableOpacity style={styles.primaryBtn} onPress={requestPermission}>
          <Text style={styles.primaryBtnText}>Otorgar permiso de cámara</Text>
        </TouchableOpacity>
      </View>
    );
  }

  const paso = lastResult?.paso;

  return (
    <View style={styles.container}>
      <View style={styles.cameraWrap}>
        <CameraView
          ref={cameraRef}
          style={styles.camera}
          facing={facing}
          enableTorch={torch}
        />

        {/* Guía visual de encuadre del vehículo y placa */}
        <View style={styles.guideOverlay} pointerEvents="none">
          <View style={styles.guideFrame}>
            <Text style={styles.guideText}>Encuadra vehículo, ruedas y placa</Text>
          </View>
        </View>

        <View style={styles.overlayTop}>
          <View style={styles.liveChip}>
            <View style={[styles.dot, streaming && styles.dotLive]} />
            <Text style={styles.liveText}>
              {bursting
                ? "CLIP 5s EN VIVO"
                : streaming
                ? `EN VIVO (${FPS_OPTIONS[fpsIdx].label})`
                : "LISTO"}{" "}
              · Frames: {framesSent}
            </Text>
          </View>

          <View style={styles.camActions}>
            <TouchableOpacity
              style={styles.camChip}
              onPress={() => setTorch((t) => !t)}
            >
              <Text style={styles.camChipText}>{torch ? "Flash ON" : "Flash OFF"}</Text>
            </TouchableOpacity>
            <TouchableOpacity
              style={styles.camChip}
              onPress={() => setFacing((f) => (f === "back" ? "front" : "back"))}
            >
              <Text style={styles.camChipText}>{facing === "back" ? "Trasera" : "Frontal"}</Text>
            </TouchableOpacity>
          </View>
        </View>

        {sending && (
          <View style={styles.inferBadge}>
            <ActivityIndicator size="small" color="#38bdf8" />
            <Text style={styles.inferBadgeText}>Analizando M1+M2+M3+M4...</Text>
          </View>
        )}
      </View>

      <ScrollView style={styles.controls} contentContainerStyle={{ paddingBottom: 28 }}>
        <Text style={styles.sectionLabel}>Caseta activa:</Text>
        <ScrollView horizontal showsHorizontalScrollIndicator={false} style={styles.rowScroll}>
          {casetas.map((c) => (
            <TouchableOpacity
              key={c.id}
              onPress={() => setCasetaId(c.id)}
              style={[styles.pill, casetaId === c.id && styles.pillActive]}
            >
              <Text style={[styles.pillText, casetaId === c.id && styles.pillTextActive]}>
                {c.nombre}
              </Text>
            </TouchableOpacity>
          ))}
        </ScrollView>

        <Text style={styles.sectionLabel}>Frecuencia de envío desde cámara (2 a 5 fps):</Text>
        <View style={styles.fpsRow}>
          {FPS_OPTIONS.map((opt, idx) => (
            <TouchableOpacity
              key={opt.label}
              onPress={() => setFpsIdx(idx)}
              style={[styles.pill, fpsIdx === idx && styles.pillActive]}
            >
              <Text style={[styles.pillText, fpsIdx === idx && styles.pillTextActive]}>
                {opt.label}
              </Text>
            </TouchableOpacity>
          ))}
        </View>

        <View style={styles.btnRow}>
          <TouchableOpacity
            style={[styles.primaryBtn, streaming && styles.stopBtn, { flex: 1 }]}
            onPress={() => {
              setBursting(false);
              setStreaming((s) => !s);
            }}
          >
            <Text style={styles.primaryBtnText}>
              {streaming ? "Detener Captura" : "Iniciar Modo Caseta"}
            </Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryBtn}
            disabled={streaming || sending}
            onPress={startShortBurst}
          >
            <Text style={styles.secondaryBtnText}>Clip 5s</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryBtn}
            disabled={sending}
            onPress={captureAndSend}
          >
            <Text style={styles.secondaryBtnText}>1 Foto</Text>
          </TouchableOpacity>

          <TouchableOpacity
            style={styles.secondaryBtn}
            disabled={sending}
            onPress={pickGalleryAndSend}
          >
            <Text style={styles.secondaryBtnText}>Galería</Text>
          </TouchableOpacity>
        </View>

        {errorMsg ? <Text style={styles.errorText}>{errorMsg}</Text> : null}

        {paso && (
          <View style={styles.resultCard}>
            <Text style={{ color: "#34d399", fontSize: 11, fontWeight: "700", marginBottom: 8 }}>
              ✓ Registro #{paso.id} enviado en tiempo real a la Página Web
            </Text>

            {paso.imagen_url && lastResult?.baseUrl ? (
              <Image
                source={{
                  uri: paso.imagen_url.startsWith("http")
                    ? paso.imagen_url
                    : `${lastResult.baseUrl}${paso.imagen_url}`
                }}
                style={{
                  width: "100%",
                  height: 175,
                  borderRadius: 10,
                  marginBottom: 10,
                  backgroundColor: "#020617"
                }}
                resizeMode="contain"
              />
            ) : null}

            <View style={styles.resultHeader}>
              <Text style={styles.plateText}>{paso.placa_normalizada || "ILEGIBLE"}</Text>
              <Text
                style={[
                  styles.statusBadge,
                  paso.estado === "APROBADO" ? styles.statusOk : styles.statusAlert
                ]}
              >
                {paso.estado}
              </Text>
            </View>

            <View style={styles.metricsGrid}>
              <Metric label="Clase (M2)" value={formatVehicleClass(paso.categoria_detectada)} />
              <Metric
                label="Ejes / Llantas (M3)"
                value={`${paso.ejes_detectados} ejes${
                  paso.llantas_visibles !== undefined ? ` (${paso.llantas_visibles} llantas)` : ""
                }`}
              />
              <Metric label="Pasajeros (M4)" value={String(paso.pasajeros)} />
              <Metric label="Tarifa" value={`Bs. ${Number(paso.monto).toFixed(2)}`} />
            </View>

            <Text style={styles.metaText}>
              OCR M1: {(paso.confianza_ocr * 100).toFixed(0)}% · Latencia punta a punta:{" "}
              {Math.round(paso.latencia_ms)} ms
            </Text>
          </View>
        )}

        {history.length > 1 && (
          <View style={styles.historyBox}>
            <Text style={styles.sectionLabel}>Últimas lecturas de esta sesión:</Text>
            {history.slice(1).map((item) => (
              <View key={item.id} style={styles.historyRow}>
                <Text style={styles.historyPlate}>{item.placa_normalizada || "ILEGIBLE"}</Text>
                <Text style={styles.historyMeta}>
                  {item.categoria_detectada} · {item.ejes_detectados} ejes · Bs.{" "}
                  {Number(item.monto).toFixed(2)}
                </Text>
                <Text
                  style={[
                    styles.historyState,
                    item.estado === "APROBADO" ? { color: "#34d399" } : { color: "#f87171" }
                  ]}
                >
                  {item.estado}
                </Text>
              </View>
            ))}
          </View>
        )}
      </ScrollView>
    </View>
  );
}

function Metric({ label, value }) {
  return (
    <View style={styles.metricItem}>
      <Text style={styles.metricLabel}>{label}</Text>
      <Text style={styles.metricValue}>{value}</Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, backgroundColor: "#020617" },
  center: {
    flex: 1,
    justifyContent: "center",
    alignItems: "center",
    padding: 24,
    backgroundColor: "#020617"
  },
  permText: { color: "#e2e8f0", textAlign: "center", marginBottom: 16 },
  cameraWrap: { height: 295, position: "relative", backgroundColor: "#000" },
  camera: { flex: 1 },
  guideOverlay: {
    position: "absolute",
    inset: 0,
    justifyContent: "center",
    alignItems: "center",
    paddingTop: 22
  },
  guideFrame: {
    width: "84%",
    height: "66%",
    borderWidth: 2,
    borderColor: "rgba(56,189,248,0.65)",
    borderStyle: "dashed",
    borderRadius: 14,
    justifyContent: "flex-end",
    alignItems: "center",
    paddingBottom: 6
  },
  guideText: {
    color: "#e0f2fe",
    fontSize: 11,
    backgroundColor: "rgba(2,6,23,0.65)",
    paddingHorizontal: 8,
    paddingVertical: 2,
    borderRadius: 6
  },
  overlayTop: {
    position: "absolute",
    top: 10,
    left: 10,
    right: 10,
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  liveChip: {
    flexDirection: "row",
    alignItems: "center",
    backgroundColor: "rgba(2,6,23,0.8)",
    paddingHorizontal: 10,
    paddingVertical: 5,
    borderRadius: 20,
    gap: 6
  },
  camActions: { flexDirection: "row", gap: 6 },
  camChip: {
    backgroundColor: "rgba(15,23,42,0.85)",
    paddingHorizontal: 8,
    paddingVertical: 5,
    borderRadius: 10,
    borderWidth: 1,
    borderColor: "#334155"
  },
  camChipText: { color: "#e2e8f0", fontSize: 10, fontWeight: "700" },
  inferBadge: {
    position: "absolute",
    bottom: 10,
    alignSelf: "center",
    flexDirection: "row",
    alignItems: "center",
    gap: 8,
    backgroundColor: "rgba(2,6,23,0.85)",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 16
  },
  inferBadgeText: { color: "#38bdf8", fontSize: 11, fontWeight: "700" },
  dot: { width: 8, height: 8, borderRadius: 4, backgroundColor: "#94a3b8" },
  dotLive: { backgroundColor: "#ef4444" },
  liveText: { color: "#f8fafc", fontSize: 11, fontWeight: "700" },
  controls: { flex: 1, padding: 14 },
  sectionLabel: { color: "#94a3b8", fontSize: 12, marginBottom: 6, marginTop: 6 },
  rowScroll: { marginBottom: 8 },
  fpsRow: { flexDirection: "row", gap: 8, marginBottom: 12 },
  pill: {
    paddingHorizontal: 12,
    paddingVertical: 7,
    borderRadius: 12,
    backgroundColor: "#0f172a",
    borderWidth: 1,
    borderColor: "#1e293b",
    marginRight: 8
  },
  pillActive: { backgroundColor: "#0284c7", borderColor: "#38bdf8" },
  pillText: { color: "#cbd5e1", fontSize: 12, fontWeight: "600" },
  pillTextActive: { color: "#ffffff" },
  btnRow: { flexDirection: "row", gap: 8, marginTop: 4 },
  primaryBtn: {
    backgroundColor: "#38bdf8",
    paddingVertical: 13,
    paddingHorizontal: 14,
    borderRadius: 14,
    alignItems: "center"
  },
  stopBtn: { backgroundColor: "#ef4444" },
  primaryBtnText: { color: "#020617", fontWeight: "800", fontSize: 13 },
  secondaryBtn: {
    backgroundColor: "#1e293b",
    paddingVertical: 13,
    paddingHorizontal: 12,
    borderRadius: 14,
    justifyContent: "center"
  },
  secondaryBtnText: { color: "#f8fafc", fontWeight: "700", fontSize: 12 },
  errorText: { color: "#f87171", fontSize: 12, marginTop: 10 },
  resultCard: {
    marginTop: 14,
    backgroundColor: "#0f172a",
    borderRadius: 16,
    padding: 15,
    borderWidth: 1,
    borderColor: "#1e293b"
  },
  resultHeader: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    marginBottom: 12
  },
  plateText: { color: "#ffffff", fontSize: 22, fontWeight: "900", letterSpacing: 1.5 },
  statusBadge: {
    paddingHorizontal: 10,
    paddingVertical: 4,
    borderRadius: 10,
    fontSize: 11,
    fontWeight: "800",
    overflow: "hidden"
  },
  statusOk: { backgroundColor: "#064e3b", color: "#6ee7b7" },
  statusAlert: { backgroundColor: "#7f1d1d", color: "#fecaca" },
  metricsGrid: { flexDirection: "row", flexWrap: "wrap", gap: 8 },
  metricItem: {
    width: "47%",
    backgroundColor: "#020617",
    borderRadius: 10,
    padding: 10
  },
  metricLabel: { color: "#64748b", fontSize: 11 },
  metricValue: {
    color: "#f8fafc",
    fontSize: 15,
    fontWeight: "700",
    marginTop: 2,
    textTransform: "capitalize"
  },
  metaText: { color: "#64748b", fontSize: 11, marginTop: 10 },
  historyBox: { marginTop: 14 },
  historyRow: {
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center",
    backgroundColor: "#0f172a",
    borderRadius: 10,
    paddingHorizontal: 12,
    paddingVertical: 8,
    marginBottom: 6,
    borderWidth: 1,
    borderColor: "#1e293b"
  },
  historyPlate: { color: "#ffffff", fontWeight: "800", fontSize: 13 },
  historyMeta: { color: "#94a3b8", fontSize: 11, textTransform: "capitalize" },
  historyState: { fontSize: 10, fontWeight: "800" }
});
