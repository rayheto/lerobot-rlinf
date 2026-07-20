"""Real-world SO-101 backend for openpi websocket policies.

This module runs in the openpi venv because it needs both `openpi_client` and
LeRobot's hardware drivers. It reads two USB cameras plus the SO-101 follower
state, calls the policy server with the live observation, and streams the
returned action chunk to the real arm.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import queue
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)


@dataclass
class RateLimiter:
    hz: float

    def __post_init__(self) -> None:
        self.period_s = 1.0 / self.hz
        self.next_t = time.perf_counter()

    def sleep(self) -> dict[str, float]:
        self.next_t += self.period_s
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
            return {"sleep_s": delay, "overrun_s": 0.0}
        overrun = -delay
        self.next_t = time.perf_counter()
        return {"sleep_s": 0.0, "overrun_s": overrun}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class TelemetryRecorder:
    """Line-buffered JSONL telemetry written by a background thread."""

    def __init__(self, telemetry_dir: Path | None):
        self.telemetry_dir = telemetry_dir
        self._queue: queue.Queue[dict[str, Any] | None] | None = None
        self._thread: threading.Thread | None = None
        self._path: Path | None = None
        if telemetry_dir is not None:
            telemetry_dir.mkdir(parents=True, exist_ok=True)
            self._path = telemetry_dir / "telemetry.jsonl"
            self._queue = queue.Queue(maxsize=4096)
            self._thread = threading.Thread(target=self._worker, name="telemetry-writer", daemon=True)
            self._thread.start()

    @property
    def path(self) -> Path | None:
        return self._path

    def record(self, event: str, **payload: Any) -> None:
        if self._queue is None:
            return
        item = {
            "event": event,
            "wall_time": time.time(),
            "mono_time": time.perf_counter(),
            **payload,
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            pass

    def _worker(self) -> None:
        assert self._queue is not None
        assert self._path is not None
        with self._path.open("a", buffering=1) as f:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                f.write(json.dumps(item, default=_json_default) + "\n")

    def close(self) -> None:
        if self._queue is None:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=5)


class PolicyInputRecorder:
    """Records policy inputs without blocking the control loop on JPEG writes."""

    def __init__(self, debug_dir: Path | None, max_pending: int = 64):
        self.debug_dir = debug_dir
        self.frames: list[Path] = []
        self._queue: queue.Queue[dict[str, Any] | None] | None = None
        self._thread: threading.Thread | None = None
        if self.debug_dir is not None:
            self.debug_dir.mkdir(parents=True, exist_ok=True)
            self._queue = queue.Queue(maxsize=max_pending)
            self._thread = threading.Thread(target=self._worker, name="policy-input-writer", daemon=True)
            self._thread.start()

    def record(self, chunk_idx: int, policy_obs: dict[str, Any]) -> None:
        if self._queue is None:
            return
        item = {
            "chunk_idx": chunk_idx,
            "front": np.asarray(policy_obs["images/front"]).copy(),
            "wrist": np.asarray(policy_obs["images/wrist"]).copy(),
            "state": np.asarray(policy_obs["state"], dtype=np.float64).copy(),
            "prompt": str(policy_obs["prompt"]),
        }
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(item)
            except queue.Full:
                pass

    def _worker(self) -> None:
        assert self.debug_dir is not None
        assert self._queue is not None

        import cv2

        jsonl_path = self.debug_dir / "policy_inputs.jsonl"
        with jsonl_path.open("a", buffering=1) as f:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                chunk_idx = int(item["chunk_idx"])
                front = np.asarray(item["front"])
                wrist = np.asarray(item["wrist"])
                state = np.asarray(item["state"], dtype=np.float64)
                prompt = str(item["prompt"])

                front_bgr = cv2.cvtColor(front, cv2.COLOR_RGB2BGR)
                wrist_bgr = cv2.cvtColor(wrist, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    front_bgr,
                    f"front chunk {chunk_idx}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    wrist_bgr,
                    f"wrist chunk {chunk_idx}",
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0),
                    2,
                )

                front_path = self.debug_dir / f"chunk_{chunk_idx:04d}_front.jpg"
                wrist_path = self.debug_dir / f"chunk_{chunk_idx:04d}_wrist.jpg"
                sheet_path = self.debug_dir / f"chunk_{chunk_idx:04d}_sheet.jpg"
                cv2.imwrite(str(front_path), front_bgr)
                cv2.imwrite(str(wrist_path), wrist_bgr)
                cv2.imwrite(str(sheet_path), cv2.hconcat([front_bgr, wrist_bgr]))
                self.frames.append(sheet_path)

                meta = {
                    "chunk_idx": chunk_idx,
                    "prompt": prompt,
                    "state": state.tolist(),
                    "front": str(front_path),
                    "wrist": str(wrist_path),
                    "sheet": str(sheet_path),
                }
                f.write(json.dumps(meta) + "\n")

    def close(self) -> None:
        if self._queue is None:
            return
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)
        if self.debug_dir is None or not self.frames:
            return

        import cv2

        first = cv2.imread(str(self.frames[0]))
        if first is None:
            return
        h, w = first.shape[:2]
        video_path = self.debug_dir / "policy_inputs.mp4"
        writer = cv2.VideoWriter(
            str(video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            3.0,
            (w, h),
        )
        try:
            for path in self.frames:
                img = cv2.imread(str(path))
                if img is None:
                    continue
                writer.write(img)
        finally:
            writer.release()
        print(f"[real_backend] debug policy inputs: {self.debug_dir}", flush=True)


class TimedWebsocketPolicy:
    """Small openpi websocket client with per-request timeout and reconnect."""

    def __init__(
        self,
        host: str,
        port: int,
        request_timeout_s: float,
        connect_timeout_s: float = 10.0,
        api_key: str | None = None,
    ) -> None:
        self._uri = f"ws://{host}:{port}"
        self._request_timeout_s = request_timeout_s
        self._connect_timeout_s = connect_timeout_s
        self._api_key = api_key
        self._ws = None
        self._server_metadata: dict[str, Any] | None = None
        from openpi_client import msgpack_numpy

        self._packer = msgpack_numpy.Packer()

    def get_server_metadata(self) -> dict[str, Any]:
        self._ensure_connected()
        return dict(self._server_metadata or {})

    def infer(self, obs: dict[str, Any]) -> dict[str, Any]:
        self._ensure_connected()
        try:
            self._ws.send(self._packer.pack(obs))
            response = self._recv()
        except Exception:
            self.close()
            raise
        if isinstance(response, str):
            self.close()
            raise RuntimeError(f"Error in inference server:\n{response}")
        from openpi_client import msgpack_numpy

        return msgpack_numpy.unpackb(response)

    def close(self) -> None:
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                ws.close()
            except Exception:
                pass

    def _ensure_connected(self) -> None:
        if self._ws is not None:
            return
        import websockets.sync.client
        from openpi_client import msgpack_numpy

        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=self._connect_timeout_s,
            close_timeout=1.0,
        )
        metadata = self._recv()
        if isinstance(metadata, str):
            raise RuntimeError(f"Error while connecting to inference server:\n{metadata}")
        self._server_metadata = msgpack_numpy.unpackb(metadata)

    def _recv(self) -> bytes | str:
        try:
            return self._ws.recv(timeout=self._request_timeout_s)
        except TypeError:
            sock = getattr(self._ws, "socket", None)
            if sock is None:
                return self._ws.recv()
            old_timeout = sock.gettimeout()
            sock.settimeout(self._request_timeout_s)
            try:
                return self._ws.recv()
            finally:
                sock.settimeout(old_timeout)


class ActionPostProcessor:
    def __init__(self, mode: str, step_hz: float, ema_tau_s: float = 0.12):
        self.mode = mode
        self.ema_tau_s = float(ema_tau_s)
        self.alpha = 1.0
        self.last: np.ndarray | None = None
        if mode == "ema":
            dt = 1.0 / float(step_hz)
            self.alpha = 1.0 - math.exp(-dt / max(self.ema_tau_s, 1e-6))
        elif mode != "none":
            raise SystemExit(f"unsupported action_smoothing={mode!r}; expected none or ema")

    def apply(self, action: np.ndarray) -> np.ndarray:
        action = np.asarray(action, dtype=np.float64).reshape(6)
        if self.mode == "none" or self.last is None:
            out = action.copy()
        else:
            out = self.alpha * action + (1.0 - self.alpha) * self.last
        self.last = out.copy()
        return out


@dataclass
class InferenceRequest:
    request_id: int
    chunk_idx: int
    control_step: int
    policy_obs: dict[str, Any]
    observation_mono: float
    observation_wall: float


@dataclass
class InferenceResult:
    request: InferenceRequest
    ok: bool
    chunk: np.ndarray | None
    response: dict[str, Any] | None
    error: str | None
    request_mono: float
    response_mono: float


def _stale_prefix_steps(action_age_s: float, step_period_s: float, chunk_len: int) -> int:
    if action_age_s <= 0 or step_period_s <= 0 or chunk_len <= 0:
        return 0
    return min(chunk_len, int(action_age_s / step_period_s))


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except ModuleNotFoundError as exc:
            raise SystemExit(
                "YAML config requires PyYAML in the runtime venv; use JSON or install pyyaml."
            ) from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise SystemExit(f"real backend config must be a mapping: {path}")
    return data


def _detect_opencv_cameras() -> list[dict[str, Any]]:
    from lerobot.cameras.opencv.camera_opencv import OpenCVCamera

    cameras = OpenCVCamera.find_cameras()
    cameras.sort(key=lambda c: str(c.get("id", "")))
    return cameras


def _print_detected_cameras() -> None:
    cameras = _detect_opencv_cameras()
    print(json.dumps(cameras, indent=2, default=str), flush=True)


def _camera_id(value: Any) -> int | Path:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return Path(value) if value.startswith("/") else int(value) if value.isdigit() else Path(value)
    return value


def _resolve_camera_specs(cfg: dict[str, Any]) -> dict[str, dict[str, Any]]:
    cameras_cfg = dict(cfg.get("cameras") or {})
    detected = _detect_opencv_cameras()
    print(f"[real_backend] detected OpenCV cameras: {[c.get('id') for c in detected]}", flush=True)

    for key in ("front", "wrist"):
        cameras_cfg.setdefault(key, {})

    missing = [
        key
        for key in ("front", "wrist")
        if cameras_cfg[key].get("index_or_path") is None
    ]
    if missing:
        if len(detected) != 2:
            raise SystemExit(
                "camera index_or_path is missing for "
                f"{missing}, and auto assignment requires exactly two detected cameras; "
                f"detected {[c.get('id') for c in detected]}"
            )
        auto = {"front": detected[0]["id"], "wrist": detected[1]["id"]}
        for key in missing:
            cameras_cfg[key]["index_or_path"] = auto[key]

    return cameras_cfg


def _build_robot(cfg: dict[str, Any]):
    from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
    try:
        from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
        from lerobot.robots.so101_follower.so101_follower import SO101Follower
    except ModuleNotFoundError:
        from lerobot.robots.so_follower.config_so_follower import SO101FollowerConfig
        from lerobot.robots.so_follower.so_follower import SOFollower as SO101Follower

    robot_cfg = dict(cfg.get("robot") or {})
    port = robot_cfg.get("port")
    if not port:
        raise SystemExit("real backend config requires robot.port, e.g. /dev/ttyACM0")
    max_relative_target = robot_cfg.get("max_relative_target", 10)
    if isinstance(max_relative_target, int):
        max_relative_target = float(max_relative_target)

    cameras_cfg = _resolve_camera_specs(cfg)
    cameras = {}
    for key in ("front", "wrist"):
        spec = dict(cameras_cfg[key])
        cameras[key] = OpenCVCameraConfig(
            index_or_path=_camera_id(spec["index_or_path"]),
            fps=int(spec.get("fps", cfg.get("fps", 30))),
            width=int(spec.get("width", cfg.get("width", 640))),
            height=int(spec.get("height", cfg.get("height", 480))),
        )

    config = SO101FollowerConfig(
        port=str(port),
        id=robot_cfg.get("id"),
        calibration_dir=Path(robot_cfg["calibration_dir"]).expanduser()
        if robot_cfg.get("calibration_dir")
        else None,
        disable_torque_on_disconnect=bool(robot_cfg.get("disable_torque_on_disconnect", True)),
        max_relative_target=max_relative_target,
        cameras=cameras,
        use_degrees=bool(robot_cfg.get("use_degrees", True)),
    )
    return SO101Follower(config)


def _state_from_obs(obs: dict[str, Any]) -> np.ndarray:
    return np.asarray([obs[f"{name}.pos"] for name in MOTOR_NAMES], dtype=np.float64)


def _action_dict(action: np.ndarray) -> dict[str, float]:
    action = np.asarray(action, dtype=np.float64).reshape(-1)
    if action.shape != (6,):
        raise RuntimeError(f"expected action shape (6,), got {action.shape}")
    return {f"{name}.pos": float(value) for name, value in zip(MOTOR_NAMES, action)}


def _build_policy_obs(obs: dict[str, Any], prompt: str) -> dict[str, Any]:
    from openpi_client import image_tools

    front = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.asarray(obs["front"]), 224, 224)
    )
    wrist = image_tools.convert_to_uint8(
        image_tools.resize_with_pad(np.asarray(obs["wrist"]), 224, 224)
    )
    return {
        "images/front": front,
        "images/wrist": wrist,
        "state": _state_from_obs(obs),
        "prompt": prompt,
    }


def _extract_action_chunk(result: dict[str, Any]) -> np.ndarray:
    chunk = np.asarray(result["actions"], dtype=np.float64)
    if chunk.ndim != 2 or chunk.shape[-1] != 6:
        raise RuntimeError(f"unexpected action chunk shape: {chunk.shape}")
    return chunk


def _policy_server_timing(result: dict[str, Any] | None) -> dict[str, Any]:
    if not result:
        return {}
    timing = result.get("server_timing") or {}
    return dict(timing) if isinstance(timing, dict) else {}


def _make_policy(args: argparse.Namespace, remote_cfg: dict[str, Any]) -> TimedWebsocketPolicy:
    timeout_ms = args.request_timeout_ms
    if timeout_ms is None:
        timeout_ms = remote_cfg.get("request_timeout_ms", 500)
    connect_timeout_ms = remote_cfg.get("connect_timeout_ms", 10000)
    api_key = None
    api_key_env = remote_cfg.get("api_key_env")
    if api_key_env:
        import os

        api_key = os.environ.get(str(api_key_env))
    return TimedWebsocketPolicy(
        host=args.policy_host,
        port=args.policy_port,
        request_timeout_s=float(timeout_ms) / 1000.0,
        connect_timeout_s=float(connect_timeout_ms) / 1000.0,
        api_key=api_key,
    )


def _inference_worker(
    policy: TimedWebsocketPolicy,
    request_queue: queue.Queue[InferenceRequest | None],
    result_queue: queue.Queue[InferenceResult],
    stop_event: threading.Event,
) -> None:
    while not stop_event.is_set():
        try:
            request = request_queue.get(timeout=0.1)
        except queue.Empty:
            continue
        if request is None:
            return
        t0 = time.perf_counter()
        try:
            response = policy.infer(request.policy_obs)
            chunk = _extract_action_chunk(response)
            result = InferenceResult(
                request=request,
                ok=True,
                chunk=chunk,
                response=response,
                error=None,
                request_mono=t0,
                response_mono=time.perf_counter(),
            )
        except Exception as exc:
            result = InferenceResult(
                request=request,
                ok=False,
                chunk=None,
                response=None,
                error=repr(exc),
                request_mono=t0,
                response_mono=time.perf_counter(),
            )
        try:
            result_queue.put_nowait(result)
        except queue.Full:
            pass


def _submit_latest_request(
    request_queue: queue.Queue[InferenceRequest | None],
    recorder: PolicyInputRecorder,
    telemetry: TelemetryRecorder,
    request: InferenceRequest,
) -> bool:
    recorder.record(request.chunk_idx, request.policy_obs)
    try:
        request_queue.put_nowait(request)
    except queue.Full:
        return False
    telemetry.record(
        "infer_request",
        request_id=request.request_id,
        chunk_idx=request.chunk_idx,
        control_step=request.control_step,
        observation_wall=request.observation_wall,
    )
    return True


def _run_sync_loop(
    robot: Any,
    policy: TimedWebsocketPolicy,
    prompt: str,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    recorder: PolicyInputRecorder,
    telemetry: TelemetryRecorder,
    stop_event: threading.Event,
) -> int:
    step_hz = float(runtime["step_hz"])
    limiter = RateLimiter(step_hz)
    action_horizon = int(runtime["action_horizon"])
    max_steps = int(runtime["max_steps"])
    dry_run = bool(runtime["dry_run"])
    post = ActionPostProcessor(
        mode=str(runtime["action_smoothing"]),
        step_hz=step_hz,
        ema_tau_s=float(runtime["ema_tau_s"]),
    )
    sent_steps = 0
    chunk_idx = 0
    last_loop_mono = time.perf_counter()

    while not stop_event.is_set() and (max_steps <= 0 or sent_steps < max_steps):
        obs_wall = time.time()
        obs_mono = time.perf_counter()
        obs = robot.get_observation()
        actual_state = _state_from_obs(obs)
        policy_obs = _build_policy_obs(obs, prompt)
        recorder.record(chunk_idx, policy_obs)
        t0 = time.perf_counter()
        result = policy.infer(policy_obs)
        response_mono = time.perf_counter()
        chunk = _extract_action_chunk(result)
        server_timing = _policy_server_timing(result)
        telemetry.record(
            "infer_response",
            request_id=chunk_idx,
            chunk_idx=chunk_idx,
            ok=True,
            client_rtt_ms=(response_mono - t0) * 1000.0,
            server_timing=server_timing,
            action_age_ms=(response_mono - obs_mono) * 1000.0,
            chunk_len=int(chunk.shape[0]),
        )

        for slot_idx, raw_action in enumerate(chunk[:action_horizon]):
            if stop_event.is_set() or (max_steps > 0 and sent_steps >= max_steps):
                break
            executed = post.apply(raw_action)
            sent = _action_dict(executed)
            action_mono = time.perf_counter()
            loop_dt_ms = (action_mono - last_loop_mono) * 1000.0
            if dry_run:
                print(f"[real_backend] dry-run action {sent_steps}: {sent}", flush=True)
            else:
                robot.send_action(sent)
            sleep_info = limiter.sleep()
            telemetry.record(
                "action_step",
                mode="sync",
                control_step=sent_steps,
                chunk_idx=chunk_idx,
                slot_idx=slot_idx,
                observation_wall=obs_wall,
                raw_action=raw_action,
                executed_action=executed,
                actual_state=actual_state,
                action_age_ms=(action_mono - obs_mono) * 1000.0,
                loop_dt_ms=loop_dt_ms,
                deadline_overrun_ms=sleep_info["overrun_s"] * 1000.0,
                queue_underrun=False,
            )
            last_loop_mono = action_mono
            sent_steps += 1
        print(
            f"[real_backend] chunk infer={(response_mono - t0) * 1000.0:.1f}ms sent_steps={sent_steps}",
            flush=True,
        )
        chunk_idx += 1
    return sent_steps


def _run_async_loop(
    robot: Any,
    policy: TimedWebsocketPolicy,
    prompt: str,
    runtime: dict[str, Any],
    args: argparse.Namespace,
    recorder: PolicyInputRecorder,
    telemetry: TelemetryRecorder,
    stop_event: threading.Event,
) -> int:
    step_hz = float(runtime["step_hz"])
    period_s = 1.0 / step_hz
    limiter = RateLimiter(step_hz)
    max_steps = int(runtime["max_steps"])
    dry_run = bool(runtime["dry_run"])
    replan_steps = int(runtime["replan_steps"])
    max_action_age_s = float(runtime["max_action_age_ms"]) / 1000.0
    max_consecutive_failures = int(runtime["max_consecutive_failures"])
    post = ActionPostProcessor(
        mode=str(runtime["action_smoothing"]),
        step_hz=step_hz,
        ema_tau_s=float(runtime["ema_tau_s"]),
    )

    request_queue: queue.Queue[InferenceRequest | None] = queue.Queue(maxsize=1)
    result_queue: queue.Queue[InferenceResult] = queue.Queue(maxsize=4)
    worker = threading.Thread(
        target=_inference_worker,
        args=(policy, request_queue, result_queue, stop_event),
        name="policy-inference",
        daemon=True,
    )
    worker.start()

    pending_actions: list[dict[str, Any]] = []
    request_id = 0
    chunk_idx = 0
    in_flight = False
    sent_steps = 0
    consecutive_failures = 0
    last_command: np.ndarray | None = None
    last_loop_mono = time.perf_counter()

    try:
        while not stop_event.is_set() and (max_steps <= 0 or sent_steps < max_steps):
            loop_start_mono = time.perf_counter()
            loop_wall = time.time()
            obs = robot.get_observation()
            actual_state = _state_from_obs(obs)

            latest_result: InferenceResult | None = None
            while True:
                try:
                    latest_result = result_queue.get_nowait()
                except queue.Empty:
                    break

            if latest_result is not None:
                in_flight = False
                if latest_result.ok and latest_result.chunk is not None:
                    consecutive_failures = 0
                    age_at_receive_s = latest_result.response_mono - latest_result.request.observation_mono
                    stale_prefix = _stale_prefix_steps(
                        age_at_receive_s,
                        period_s,
                        int(latest_result.chunk.shape[0]),
                    )
                    if age_at_receive_s > max_action_age_s or stale_prefix >= latest_result.chunk.shape[0]:
                        pending_actions = []
                        telemetry.record(
                            "chunk_discarded",
                            request_id=latest_result.request.request_id,
                            chunk_idx=latest_result.request.chunk_idx,
                            reason="stale",
                            action_age_ms=age_at_receive_s * 1000.0,
                            stale_prefix=stale_prefix,
                            chunk_len=int(latest_result.chunk.shape[0]),
                        )
                    else:
                        pending_actions = [
                            {
                                "raw_action": action,
                                "chunk_idx": latest_result.request.chunk_idx,
                                "slot_idx": slot_idx,
                                "observation_mono": latest_result.request.observation_mono,
                                "observation_wall": latest_result.request.observation_wall,
                                "response_mono": latest_result.response_mono,
                            }
                            for slot_idx, action in enumerate(latest_result.chunk)
                            if slot_idx >= stale_prefix
                        ]
                        telemetry.record(
                            "infer_response",
                            request_id=latest_result.request.request_id,
                            chunk_idx=latest_result.request.chunk_idx,
                            ok=True,
                            client_rtt_ms=(latest_result.response_mono - latest_result.request_mono) * 1000.0,
                            server_timing=_policy_server_timing(latest_result.response),
                            action_age_ms=age_at_receive_s * 1000.0,
                            stale_prefix=stale_prefix,
                            chunk_len=int(latest_result.chunk.shape[0]),
                            queued_len=len(pending_actions),
                        )
                else:
                    consecutive_failures += 1
                    telemetry.record(
                        "infer_response",
                        request_id=latest_result.request.request_id,
                        chunk_idx=latest_result.request.chunk_idx,
                        ok=False,
                        error=latest_result.error,
                        client_rtt_ms=(latest_result.response_mono - latest_result.request_mono) * 1000.0,
                        consecutive_failures=consecutive_failures,
                    )
                    if consecutive_failures >= max_consecutive_failures:
                        print(
                            "[real_backend] stopping after "
                            f"{consecutive_failures} consecutive inference failures",
                            flush=True,
                        )
                        break

            should_replan = not in_flight and (not pending_actions or len(pending_actions) <= replan_steps)
            if should_replan:
                policy_obs = _build_policy_obs(obs, prompt)
                req = InferenceRequest(
                    request_id=request_id,
                    chunk_idx=chunk_idx,
                    control_step=sent_steps,
                    policy_obs=policy_obs,
                    observation_mono=loop_start_mono,
                    observation_wall=loop_wall,
                )
                if _submit_latest_request(request_queue, recorder, telemetry, req):
                    in_flight = True
                    request_id += 1
                    chunk_idx += 1

            selected: dict[str, Any] | None = None
            stale_drops = 0
            now = time.perf_counter()
            while pending_actions:
                candidate = pending_actions.pop(0)
                if now - candidate["observation_mono"] <= max_action_age_s:
                    selected = candidate
                    break
                stale_drops += 1
            if stale_drops:
                telemetry.record("stale_action_drop", control_step=sent_steps, dropped=stale_drops)

            queue_underrun = selected is None
            raw_action: np.ndarray | None
            if selected is None:
                raw_action = last_command.copy() if last_command is not None else None
            else:
                raw_action = np.asarray(selected["raw_action"], dtype=np.float64)

            if raw_action is not None:
                executed = post.apply(raw_action)
                sent = _action_dict(executed)
                if dry_run:
                    print(f"[real_backend] dry-run action {sent_steps}: {sent}", flush=True)
                else:
                    robot.send_action(sent)
                last_command = executed.copy()
            else:
                executed = None

            action_mono = time.perf_counter()
            sleep_info = limiter.sleep()
            telemetry.record(
                "action_step",
                mode="async",
                control_step=sent_steps,
                chunk_idx=selected["chunk_idx"] if selected else None,
                slot_idx=selected["slot_idx"] if selected else None,
                raw_action=raw_action,
                executed_action=executed,
                actual_state=actual_state,
                action_age_ms=(
                    (action_mono - selected["observation_mono"]) * 1000.0
                    if selected else None
                ),
                loop_dt_ms=(action_mono - last_loop_mono) * 1000.0,
                deadline_overrun_ms=sleep_info["overrun_s"] * 1000.0,
                queue_underrun=queue_underrun,
                pending_len=len(pending_actions),
                in_flight=in_flight,
            )
            last_loop_mono = action_mono
            sent_steps += 1

            if sent_steps % max(1, int(step_hz)) == 0:
                print(
                    f"[real_backend] async steps={sent_steps} pending={len(pending_actions)} "
                    f"in_flight={in_flight} failures={consecutive_failures}",
                    flush=True,
                )
    finally:
        stop_event.set()
        try:
            request_queue.put_nowait(None)
        except queue.Full:
            pass
        worker.join(timeout=2)
    return sent_steps


def _runtime_value(
    args_value: Any,
    runtime_cfg: dict[str, Any],
    key: str,
    default: Any,
) -> Any:
    return args_value if args_value is not None else runtime_cfg.get(key, default)


def run_real_backend(args: argparse.Namespace) -> int:
    cfg = _load_config(Path(args.config).expanduser().resolve())
    runtime_cfg = dict(cfg.get("runtime") or {})
    remote_cfg = dict(cfg.get("remote_policy") or {})
    prompt = args.prompt or cfg.get("prompt") or runtime_cfg.get("prompt")
    if not prompt:
        raise SystemExit("prompt must be provided by --prompt or real backend config")

    runtime = {
        "step_hz": _runtime_value(args.step_hz, runtime_cfg, "step_hz", 30),
        "action_horizon": _runtime_value(args.action_horizon, runtime_cfg, "action_horizon", 10),
        "max_steps": _runtime_value(args.max_steps, runtime_cfg, "max_steps", 0),
        "dry_run": bool(args.dry_run or runtime_cfg.get("dry_run", False)),
        "execution_mode": _runtime_value(args.execution_mode, runtime_cfg, "execution_mode", "sync"),
        "replan_steps": _runtime_value(args.replan_steps, runtime_cfg, "replan_steps", None),
        "max_action_age_ms": _runtime_value(args.max_action_age_ms, runtime_cfg, "max_action_age_ms", 250),
        "max_consecutive_failures": _runtime_value(
            args.max_consecutive_failures,
            remote_cfg,
            "max_consecutive_failures",
            2,
        ),
        "action_smoothing": _runtime_value(args.action_smoothing, runtime_cfg, "action_smoothing", "none"),
        "ema_tau_s": _runtime_value(args.ema_tau_s, runtime_cfg, "ema_tau_s", 0.12),
    }
    runtime["step_hz"] = float(runtime["step_hz"])
    runtime["action_horizon"] = int(runtime["action_horizon"])
    runtime["max_steps"] = int(runtime["max_steps"])
    runtime["replan_steps"] = int(runtime["replan_steps"] or runtime["action_horizon"])
    runtime["max_action_age_ms"] = float(runtime["max_action_age_ms"])
    runtime["max_consecutive_failures"] = int(runtime["max_consecutive_failures"])
    runtime["ema_tau_s"] = float(runtime["ema_tau_s"])

    if runtime["execution_mode"] not in {"sync", "async"}:
        raise SystemExit("--execution-mode must be sync or async")
    if runtime["replan_steps"] <= 0:
        raise SystemExit("replan_steps must be positive")

    policy = _make_policy(args, remote_cfg)
    print(f"[real_backend] server metadata: {policy.get_server_metadata()}", flush=True)

    robot = _build_robot(cfg)
    warmup_steps = int(runtime_cfg.get("warmup_steps", 2))
    debug_dir_raw = args.debug_dir or runtime_cfg.get("debug_dir")
    telemetry_dir_raw = args.telemetry_dir or runtime_cfg.get("telemetry_dir") or debug_dir_raw
    recorder = PolicyInputRecorder(
        Path(debug_dir_raw).expanduser().resolve() if debug_dir_raw else None
    )
    telemetry = TelemetryRecorder(
        Path(telemetry_dir_raw).expanduser().resolve() if telemetry_dir_raw else None
    )
    if telemetry.path is not None:
        print(f"[real_backend] telemetry: {telemetry.path}", flush=True)

    stop_event = threading.Event()

    def _stop(_signum, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    sent_steps = 0
    try:
        robot.connect(calibrate=bool(runtime_cfg.get("calibrate", True)))
        for _ in range(warmup_steps):
            robot.get_observation()

        telemetry.record("run_start", runtime=runtime, prompt=prompt)
        if runtime["execution_mode"] == "async":
            sent_steps = _run_async_loop(
                robot, policy, prompt, runtime, args, recorder, telemetry, stop_event
            )
        else:
            sent_steps = _run_sync_loop(
                robot, policy, prompt, runtime, args, recorder, telemetry, stop_event
            )
        telemetry.record("run_stop", sent_steps=sent_steps)
    finally:
        recorder.close()
        telemetry.close()
        policy.close()
        if robot.is_connected:
            robot.disconnect()
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="Real SO-101 openpi backend.")
    parser.add_argument("--config", required=True, help="YAML/JSON real backend config.")
    parser.add_argument("--policy-host", default="localhost")
    parser.add_argument("--policy-port", type=int, default=8000)
    parser.add_argument("--prompt", default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--step-hz", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--debug-dir", default=None)
    parser.add_argument("--telemetry-dir", default=None)
    parser.add_argument("--execution-mode", choices=["sync", "async"], default=None)
    parser.add_argument("--replan-steps", type=int, default=None)
    parser.add_argument("--request-timeout-ms", type=float, default=None)
    parser.add_argument("--max-action-age-ms", type=float, default=None)
    parser.add_argument("--max-consecutive-failures", type=int, default=None)
    parser.add_argument("--action-smoothing", choices=["none", "ema"], default=None)
    parser.add_argument("--ema-tau-s", type=float, default=None)
    parser.add_argument("--detect-cameras", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.detect_cameras:
        _print_detected_cameras()
        return
    sys.exit(run_real_backend(args))


if __name__ == "__main__":
    main()
