"""Fake robot, cameras, and policy for end-to-end testing without hardware.

These components implement the same interfaces as the real SO-101 backend
but use synthetic data.  They allow the full RECAP pipeline (control loop,
state machine, data recorder, hook server, LeRobot v3 export) to be tested
end-to-end on any machine.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np


class FakeRobot:
    """Fake SO-101 robot that simulates joint motion.

    Implements the same interface used by real_backend.py:
      - connect(calibrate)
      - get_observation() -> dict
      - send_action(dict)
      - disconnect()
      - is_connected
    """

    def __init__(self, n_joints: int = 6, step_hz: float = 30.0) -> None:
        self.n_joints = n_joints
        self.step_hz = step_hz
        self._connected = False
        self._state = np.zeros(n_joints, dtype=np.float64)
        self._target = np.zeros(n_joints, dtype=np.float64)
        self._tick = 0
        self._action_count = 0

    @property
    def is_connected(self) -> bool:
        return self._connected

    def connect(self, calibrate: bool = True) -> None:
        self._connected = True
        if calibrate:
            self._state[:] = 0.0

    def get_observation(self) -> dict[str, Any]:
        # Simulate slow convergence to target
        alpha = 0.3
        self._state = alpha * self._target + (1.0 - alpha) * self._state
        self._tick += 1
        return {
            "agent": {"qpos": self._state.copy()},
            "task": {"instruction": "Grab orange and place into plate"},
        }

    def send_action(self, action: dict[str, float]) -> None:
        # action is {"shoulder_pan.pos": val, ...}
        vals = list(action.values())
        self._target[: len(vals)] = vals
        self._action_count += 1

    def disconnect(self) -> None:
        self._connected = False

    @property
    def action_count(self) -> int:
        return self._action_count


class FakeCamera:
    """Fake camera that generates synthetic frames."""

    def __init__(self, key: str, width: int = 640, height: int = 480) -> None:
        self.key = key
        self.width = width
        self.height = height
        self._tick = 0

    def read(self) -> np.ndarray:
        """Return a synthetic BGR frame with a moving pattern."""
        self._tick += 1
        frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        # Draw a moving circle to simulate motion
        cx = int(self.width * (0.3 + 0.2 * np.sin(self._tick * 0.1)))
        cy = int(self.height * (0.5 + 0.1 * np.cos(self._tick * 0.15)))
        color = (0, 255, 0) if self.key == "front" else (255, 128, 0)
        import cv2
        cv2.circle(frame, (cx, cy), 30, color, -1)
        # Add tick number as text
        cv2.putText(frame, f"{self.key}:{self._tick}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return frame


class FakePolicy:
    """Fake policy server that returns deterministic action chunks.

    Implements the same interface as TimedWebsocketPolicy:
      - get_server_metadata() -> dict
      - infer(obs) -> dict
      - close()
    """

    def __init__(self, action_horizon: int = 10, n_joints: int = 6) -> None:
        self.action_horizon = action_horizon
        self.n_joints = n_joints
        self._request_count = 0
        self._connected = True

    def get_server_metadata(self) -> dict[str, Any]:
        return {
            "model": "fake_policy",
            "action_horizon": self.action_horizon,
            "image_keys": ["front", "wrist"],
        }

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._request_count += 1
        # Generate a simple sinusoidal action chunk
        t = self._request_count
        chunk = np.zeros((self.action_horizon, self.n_joints), dtype=np.float32)
        for i in range(self.action_horizon):
            for j in range(self.n_joints):
                chunk[i, j] = 10.0 * np.sin(0.1 * (t + i) + j)
        return {
            "action": chunk,
            "request_id": t,
            "metadata": {"latency_ms": 5.0},
        }

    def close(self) -> None:
        self._connected = False

    @property
    def request_count(self) -> int:
        return self._request_count


def build_fake_observation(
    state: np.ndarray,
    front_frame: np.ndarray | None = None,
    wrist_frame: np.ndarray | None = None,
    prompt: str = "Grab orange and place into plate",
) -> dict[str, Any]:
    """Build a fake observation dict matching the real robot interface."""
    if front_frame is None:
        front_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    if wrist_frame is None:
        wrist_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    return {
        "agent": {"qpos": np.asarray(state, dtype=np.float64)},
        "sensors": {
            "images": {
                "front": front_frame,
                "wrist": wrist_frame,
            },
        },
        "task": {"instruction": prompt},
    }
