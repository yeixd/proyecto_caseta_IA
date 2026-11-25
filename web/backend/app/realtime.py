"""
Gestor de conexiones en tiempo real vía WebSockets:
  - /ws/panel    -> Panel web en tiempo real (pasos, alertas, contadores y estado hardware)
  - /ws/hardware -> Dispositivos ESP32 de alertas físicas (LEDs + Buzzer)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from fastapi import WebSocket

logger = logging.getLogger("peaje.realtime")


class RealtimeManager:
    def __init__(self) -> None:
        self.panel_clients: List[WebSocket] = []
        self.hardware_clients: List[WebSocket] = []
        self.last_hardware_event: Optional[Dict[str, Any]] = None

    async def connect_panel(self, ws: WebSocket) -> None:
        await ws.accept()
        self.panel_clients.append(ws)
        logger.info("Panel WS conectado. Total paneles: %d", len(self.panel_clients))

    def disconnect_panel(self, ws: WebSocket) -> None:
        if ws in self.panel_clients:
            self.panel_clients.remove(ws)
        logger.info("Panel WS desconectado. Total paneles: %d", len(self.panel_clients))

    async def connect_hardware(self, ws: WebSocket) -> None:
        await ws.accept()
        self.hardware_clients.append(ws)
        logger.info("Hardware ESP32 WS conectado. Total dispositivos: %d", len(self.hardware_clients))
        await self.broadcast_panel(
            {
                "event": "hardware_estado",
                "connected_devices": len(self.hardware_clients),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def disconnect_hardware(self, ws: WebSocket) -> None:
        if ws in self.hardware_clients:
            self.hardware_clients.remove(ws)
        logger.info("Hardware ESP32 WS desconectado. Total dispositivos: %d", len(self.hardware_clients))
        await self.broadcast_panel(
            {
                "event": "hardware_estado",
                "connected_devices": len(self.hardware_clients),
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        )

    async def broadcast_panel(self, payload: Dict[str, Any]) -> None:
        dead: List[WebSocket] = []
        for ws in list(self.panel_clients):
            try:
                await ws.send_json(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect_panel(ws)

    async def broadcast_hardware(
        self,
        evento: str,
        placa: str = "",
        detalle: str = "",
        nivel: str = "info",
    ) -> Dict[str, Any]:
        """
        Envía señal al dispositivo físico ESP32 para indicar:
          - IDENTIFICANDO / INFERENCIA (LED azul parpadeando)
          - APROBADO (LED verde + pitido corto)
          - SIN_SALDO (LED amarillo + 2 pitidos)
          - ALERTA_NO_REGISTRADO / ALERTA_INACTIVO (LED rojo + sirena buzzer)
          - TEST (secuencia de verificación)
        """
        msg = {
            "event": "hardware_signal",
            "tipo": evento,
            "placa": placa,
            "detalle": detalle,
            "nivel": nivel,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }
        self.last_hardware_event = msg

        dead: List[WebSocket] = []
        for ws in list(self.hardware_clients):
            try:
                await ws.send_json(msg)
            except Exception:
                dead.append(ws)
        for ws in dead:
            await self.disconnect_hardware(ws)

        # Notificar también al panel web para trazabilidad en vivo
        await self.broadcast_panel({"event": "hardware_evento", "data": msg})
        return msg


realtime_manager = RealtimeManager()
