"""RECAP direct-Hook WebSocket server for Rebot-Arm integration.

This server runs inside the lerobot-rlinf process and accepts WebSocket
connections from the Rebot-Arm web UI.  It provides:

  * Real-time joint state broadcast (latest-value, bounded queue)
  * Camera frame broadcast (JPEG, latest-value)
  * Session status broadcast (state machine snapshot)
  * Control commands from UI: pause/resume/freeze/intervention/confirm

The server uses the `websockets` library (already a dependency of the openpi
venv via the policy client).  It runs on its own thread and never blocks the
control loop — all sends use bounded queues with latest-value eviction.

Protocol version: direct-realtime/v1 (see PROTOCOL.md)
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from typing import Any

import numpy as np

from .state_machine import FreezeTarget, StateMachine

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "direct-realtime/v1"


class HookServer:
    """WebSocket server for direct Hook link to Rebot-Arm.

    The server runs an asyncio event loop on a background thread.  The control
    loop pushes data via thread-safe methods (latest-value, never blocking).
    """

    def __init__(
        self,
        state_machine: StateMachine,
        host: str = "0.0.0.0",
        port: int = 8765,
    ) -> None:
        self.sm = state_machine
        self.host = host
        self.port = port
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._server: Any = None
        self._clients: set[Any] = set()
        self._latest_state: dict[str, Any] = {}
        self._latest_joints: dict[str, float] = {}
        self._latest_front_jpeg: bytes | None = None
        self._latest_wrist_jpeg: bytes | None = None
        self._state_dirty = True
        self._running = False

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def url(self) -> str:
        return f"ws://{self.host}:{self.port}"

    def start(self) -> None:
        if self._running:
            return
        self._thread = threading.Thread(
            target=self._run_loop, name="recap-hook-server", daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._stop_async)

    def _stop_async(self) -> None:
        if self._server is not None:
            self._server.close()
        for task in asyncio.all_tasks(self._loop):
            task.cancel()

    def _run_loop(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception:
            logger.exception("Hook server error")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        import websockets

        self._running = True
        self._server = await websockets.serve(
            self._handler, self.host, self.port, max_size=10 * 1024 * 1024
        )
        logger.info("Hook server listening on ws://%s:%d", self.host, self.port)
        # Broadcast loop
        while self._running:
            await asyncio.sleep(0.033)  # ~30 Hz
            await self._broadcast()
        self._server.close()
        await self._server.wait_closed()

    async def _handler(self, ws: Any) -> None:
        self._clients.add(ws)
        try:
            # Send protocol hello + snapshot on connect
            hello = {
                "type": "hello",
                "protocol": PROTOCOL_VERSION,
                "timestamp": time.time(),
            }
            await ws.send(json.dumps(hello))
            # Send current snapshot
            snap = self.sm.snapshot()
            snap["type"] = "snapshot"
            await ws.send(json.dumps(snap, default=str))
            # Listen for commands
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                    await self._handle_command(ws, msg)
                except json.JSONDecodeError:
                    await ws.send(json.dumps({"type": "error", "message": "invalid json"}))
        except Exception:
            pass
        finally:
            self._clients.discard(ws)

    async def _handle_command(self, ws: Any, msg: dict[str, Any]) -> None:
        cmd = msg.get("cmd") or msg.get("type")
        ack_id = msg.get("ack_id", "")
        result: dict[str, Any] = {"type": "ack", "ack_id": ack_id, "cmd": cmd}

        if cmd == "pause_inference":
            evs = self.sm.pause_inference()
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "resume_inference":
            evs = self.sm.resume_inference()
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "freeze":
            target = FreezeTarget(msg.get("target", "front"))
            frame = self._latest_front_jpeg if target == FreezeTarget.FRONT else self._latest_wrist_jpeg
            ok, evs = self.sm.freeze_camera(target, frame, f"hook_{int(time.time())}")
            result["ok"] = ok
            result["state"] = self.sm.snapshot()
            if not ok:
                result["error"] = "no_first_frame"
        elif cmd == "unfreeze":
            target = FreezeTarget(msg.get("target", "front"))
            evs = self.sm.unfreeze_camera(target)
            result["ok"] = True
            result["state"] = self.sm.snapshot()
        elif cmd == "start_intervention":
            shadow = bool(msg.get("shadow_policy", False))
            evs = self.sm.start_intervention(shadow_policy=shadow)
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "end_intervention":
            evs = self.sm.end_intervention()
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "confirm_resume":
            evs = self.sm.confirm_resume()
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "human_action":
            # Human action from UI — record it
            self.sm.record_human_action()
            result["ok"] = True
        elif cmd == "next_episode":
            evs = self.sm.next_episode()
            result["ok"] = len(evs) > 0
            result["state"] = self.sm.snapshot()
        elif cmd == "get_snapshot":
            result["ok"] = True
            result["state"] = self.sm.snapshot()
        else:
            result["ok"] = False
            result["error"] = f"unknown_command:{cmd}"

        await ws.send(json.dumps(result, default=str))

    async def _broadcast(self) -> None:
        if not self._clients:
            return
        snap = self.sm.snapshot()
        snap["type"] = "state"
        state_msg = json.dumps(snap, default=str)
        joints_msg = None
        if self._latest_joints:
            joints_msg = json.dumps({
                "type": "joints",
                "joints": self._latest_joints,
                "timestamp": time.time(),
            })
        front_msg = self._latest_front_jpeg
        wrist_msg = self._latest_wrist_jpeg

        dead: list[Any] = []
        for ws in list(self._clients):
            try:
                asyncio.ensure_future(ws.send(state_msg))
                if joints_msg:
                    asyncio.ensure_future(ws.send(joints_msg))
                if front_msg:
                    asyncio.ensure_future(ws.send(front_msg))
                if wrist_msg:
                    asyncio.ensure_future(ws.send(wrist_msg))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self._clients.discard(ws)

    # -- thread-safe push methods (called from control loop) ---------------

    def push_joints(self, joints: dict[str, float]) -> None:
        """Push latest joint state (non-blocking, latest-value)."""
        self._latest_joints = dict(joints)

    def push_front_frame(self, jpeg_bytes: bytes) -> None:
        """Push latest front camera frame as JPEG bytes."""
        self._latest_front_jpeg = jpeg_bytes

    def push_wrist_frame(self, jpeg_bytes: bytes) -> None:
        """Push latest wrist camera frame as JPEG bytes."""
        self._latest_wrist_jpeg = jpeg_bytes

    def mark_state_dirty(self) -> None:
        self._state_dirty = True
