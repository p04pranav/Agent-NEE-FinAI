"""
Agent-NEE FinAI — Web Server + Dashboard State
FastAPI + WebSocket broadcast + thread-safe dashboard container.
"""

import copy
import asyncio
import json
import logging
import threading
import time
from pathlib import Path
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

import config

logger = logging.getLogger("agent_nee")


# ─── Dashboard State (merged from dashboard.py) ──────────────────────

@dataclass
class DashboardState:
    """Thread-safe singleton for dashboard data."""
    market_data: dict[str, Any] = field(default_factory=dict)
    predictions: dict[str, Any] = field(default_factory=dict)
    agent_activity: dict[str, Any] = field(default_factory=dict)
    latency: dict[str, Any] = field(default_factory=dict)
    status: dict[str, Any] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    _start_time: float = field(default_factory=time.time)

    def update(self, **kwargs):
        with self._lock:
            for key, value in kwargs.items():
                if hasattr(self, key) and not key.startswith("_"):
                    setattr(self, key, value)

    def get_snapshot(self) -> dict:
        with self._lock:
            uptime_seconds = time.time() - self._start_time
            hours = int(uptime_seconds // 3600)
            minutes = int((uptime_seconds % 3600) // 60)
            return {
                "market": copy.deepcopy(self.market_data),
                "predictions": copy.deepcopy(self.predictions),
                "agent_activity": copy.deepcopy(self.agent_activity),
                "latency": copy.deepcopy(self.latency),
                "status": {
                    **copy.deepcopy(self.status),
                    "uptime": f"{hours}h {minutes}m",
                },
            }


state = DashboardState()


# ─── Connection Manager ──────────────────────────────────────────────

MAX_CONNECTIONS = 10

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        if len(self.active_connections) >= MAX_CONNECTIONS:
            await websocket.close(code=1013, reason="Too many connections")
            return
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, data: dict):
        if not self.active_connections:
            return
        message = json.dumps(data)
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_text(message)
            except Exception:
                disconnected.append(connection)
        for conn in disconnected:
            self.disconnect(conn)


manager = ConnectionManager()


# ─── Broadcast Loop ──────────────────────────────────────────────────

async def broadcast_loop():
    while True:
        try:
            snapshot = state.get_snapshot()
            await manager.broadcast(snapshot)
        except Exception as e:
            logger.warning(f"Broadcast error: {e}")
        await asyncio.sleep(config.WEB_REFRESH_INTERVAL)


# ─── Lifespan ────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app):
    asyncio.create_task(broadcast_loop())
    logger.info("Broadcast loop started")
    yield


# ─── FastAPI App ─────────────────────────────────────────────────────

app = FastAPI(title="Agent-NEE FinAI Dashboard", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        f"http://localhost:{config.WEB_PORT}",
        f"http://127.0.0.1:{config.WEB_PORT}",
    ],
    allow_methods=["GET"],
    allow_headers=["*"],
)


@app.middleware("http")
async def add_security_headers(request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Content-Security-Policy"] = (
        "default-src 'self'; "
        "script-src 'self' https://cdn.jsdelivr.net https://unpkg.com; "
        "style-src 'self' https://fonts.googleapis.com 'unsafe-inline'; "
        "font-src 'self' https://fonts.gstatic.com; "
        "connect-src 'self' ws: wss:; "
        "img-src 'self' data:;"
    )
    return response


# ─── WebSocket ───────────────────────────────────────────────────────

ALLOWED_ORIGINS = {
    f"http://localhost:{config.WEB_PORT}",
    f"http://127.0.0.1:{config.WEB_PORT}",
}

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    origin = websocket.headers.get("origin", "")
    if origin and origin not in ALLOWED_ORIGINS:
        await websocket.close(code=4403)
        return
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_text()
            if len(data) > 1024:
                await websocket.close(code=1009)
                break
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        manager.disconnect(websocket)


# ─── REST Endpoints ──────────────────────────────────────────────────

@app.get("/api/status")
async def get_status():
    return JSONResponse(content=state.get_snapshot().get("status", {}))

@app.get("/api/config")
async def get_config():
    return JSONResponse(content={
        "model": config.MODEL_NAME,
        "tickers": config.TICKER_SYMBOLS,
        "interval_minutes": config.INTERVAL_MINUTES,
    })

@app.get("/api/history")
async def get_history():
    snap = state.get_snapshot()
    return JSONResponse(content={"predictions": snap.get("predictions", {}), "market": snap.get("market", {})})


# ─── Server Startup ──────────────────────────────────────────────────

def mount_static_files():
    web_dir = Path(__file__).parent / "web"
    if web_dir.exists():
        app.mount("/web", StaticFiles(directory=str(web_dir), html=True), name="web")

def start_web_server():
    import uvicorn
    mount_static_files()
    return uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=config.WEB_PORT, log_level="warning"))
