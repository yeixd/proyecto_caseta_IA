import React, { useEffect, useState } from "react";
import { SafeAreaView, StyleSheet, Text, TouchableOpacity, View } from "react-native";
import { StatusBar } from "expo-status-bar";
import { clearSession, loadSession } from "./src/api";
import LoginScreen from "./src/screens/LoginScreen";
import CasetaScreen from "./src/screens/CasetaScreen";
import ConsultaScreen from "./src/screens/ConsultaScreen";

export default function App() {
  const [token, setToken] = useState("");
  const [user, setUser] = useState(null);
  const [tab, setTab] = useState("caseta");

  useEffect(() => {
    loadSession().then((session) => {
      if (session) {
        setToken(session.token);
        setUser(session.user);
      }
    });
  }, []);

  async function handleLogout() {
    await clearSession();
    setToken("");
    setUser(null);
  }

  if (!token || !user) {
    return (
      <SafeAreaView style={styles.root}>
        <StatusBar style="light" />
        <LoginScreen
          onAuthenticated={(tok, usr) => {
            setToken(tok);
            setUser(usr);
          }}
        />
      </SafeAreaView>
    );
  }

  return (
    <SafeAreaView style={styles.root}>
      <StatusBar style="light" />
      <View style={styles.header}>
        <View>
          <Text style={styles.headerTitle}>Peaje IA</Text>
          <Text style={styles.headerSub}>
            {user.nombre} · {user.rol}
          </Text>
        </View>
        <TouchableOpacity onPress={handleLogout} style={styles.logoutBtn}>
          <Text style={styles.logoutText}>Salir</Text>
        </TouchableOpacity>
      </View>

      <View style={styles.tabs}>
        <TouchableOpacity
          style={[styles.tabBtn, tab === "caseta" && styles.tabActive]}
          onPress={() => setTab("caseta")}
        >
          <Text style={[styles.tabText, tab === "caseta" && styles.tabTextActive]}>
            Modo Caseta
          </Text>
        </TouchableOpacity>
        <TouchableOpacity
          style={[styles.tabBtn, tab === "consulta" && styles.tabActive]}
          onPress={() => setTab("consulta")}
        >
          <Text style={[styles.tabText, tab === "consulta" && styles.tabTextActive]}>
            Modo Consulta
          </Text>
        </TouchableOpacity>
      </View>

      <View style={{ flex: 1 }}>
        {tab === "caseta" ? <CasetaScreen token={token} /> : <ConsultaScreen token={token} />}
      </View>
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  root: { flex: 1, backgroundColor: "#020617" },
  header: {
    paddingHorizontal: 16,
    paddingVertical: 12,
    backgroundColor: "#0f172a",
    borderBottomWidth: 1,
    borderColor: "#1e293b",
    flexDirection: "row",
    justifyContent: "space-between",
    alignItems: "center"
  },
  headerTitle: { color: "#ffffff", fontSize: 18, fontWeight: "800" },
  headerSub: { color: "#94a3b8", fontSize: 11, textTransform: "capitalize" },
  logoutBtn: {
    backgroundColor: "#1e293b",
    paddingHorizontal: 12,
    paddingVertical: 6,
    borderRadius: 10
  },
  logoutText: { color: "#f8fafc", fontSize: 12, fontWeight: "600" },
  tabs: {
    flexDirection: "row",
    backgroundColor: "#0f172a",
    padding: 6,
    gap: 6
  },
  tabBtn: {
    flex: 1,
    paddingVertical: 10,
    borderRadius: 10,
    alignItems: "center"
  },
  tabActive: { backgroundColor: "#38bdf8" },
  tabText: { color: "#94a3b8", fontWeight: "700", fontSize: 13 },
  tabTextActive: { color: "#020617" }
});
