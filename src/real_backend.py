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
import signal
import sys
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

    def sleep(self) -> None:
        self.next_t += self.period_s
        delay = self.next_t - time.perf_counter()
        if delay > 0:
            time.sleep(delay)
        else:
            self.next_t = time.perf_counter()


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
    from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
    from lerobot.robots.so101_follower.so101_follower import SO101Follower

    robot_cfg = dict(cfg.get("robot") or {})
    port = robot_cfg.get("port")
    if not port:
        raise SystemExit("real backend config requires robot.port, e.g. /dev/ttyACM0")

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
        max_relative_target=robot_cfg.get("max_relative_target", 10),
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


def run_real_backend(args: argparse.Namespace) -> int:
    from openpi_client.websocket_client_policy import WebsocketClientPolicy

    cfg = _load_config(Path(args.config).expanduser().resolve())
    runtime_cfg = dict(cfg.get("runtime") or {})
    prompt = args.prompt or cfg.get("prompt") or runtime_cfg.get("prompt")
    if not prompt:
        raise SystemExit("prompt must be provided by --prompt or real backend config")

    policy = WebsocketClientPolicy(host=args.policy_host, port=args.policy_port)
    print(f"[real_backend] server metadata: {policy.get_server_metadata()}", flush=True)

    robot = _build_robot(cfg)
    step_hz = args.step_hz if args.step_hz is not None else runtime_cfg.get("step_hz", 30)
    action_horizon_raw = (
        args.action_horizon
        if args.action_horizon is not None
        else runtime_cfg.get("action_horizon", 10)
    )
    max_steps_raw = (
        args.max_steps if args.max_steps is not None else runtime_cfg.get("max_steps", 0)
    )
    limiter = RateLimiter(float(step_hz))
    action_horizon = int(action_horizon_raw)
    max_steps = int(max_steps_raw)
    warmup_steps = int(runtime_cfg.get("warmup_steps", 2))
    dry_run = bool(args.dry_run or runtime_cfg.get("dry_run", False))

    stop = False

    def _stop(_signum, _frame) -> None:
        nonlocal stop
        stop = True

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)

    sent_steps = 0
    try:
        robot.connect(calibrate=bool(runtime_cfg.get("calibrate", True)))
        for _ in range(warmup_steps):
            robot.get_observation()

        while not stop and (max_steps <= 0 or sent_steps < max_steps):
            obs = robot.get_observation()
            policy_obs = _build_policy_obs(obs, prompt)
            t0 = time.perf_counter()
            result = policy.infer(policy_obs)
            infer_ms = (time.perf_counter() - t0) * 1000.0
            chunk = np.asarray(result["actions"], dtype=np.float64)
            if chunk.ndim != 2 or chunk.shape[-1] != 6:
                raise RuntimeError(f"unexpected action chunk shape: {chunk.shape}")

            for action in chunk[:action_horizon]:
                if stop or (max_steps > 0 and sent_steps >= max_steps):
                    break
                sent = _action_dict(action)
                if dry_run:
                    print(f"[real_backend] dry-run action {sent_steps}: {sent}", flush=True)
                else:
                    robot.send_action(sent)
                sent_steps += 1
                limiter.sleep()
            print(
                f"[real_backend] chunk infer={infer_ms:.1f}ms sent_steps={sent_steps}",
                flush=True,
            )
    finally:
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
    parser.add_argument("--detect-cameras", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, force=True)
    if args.detect_cameras:
        _print_detected_cameras()
        return
    sys.exit(run_real_backend(args))


if __name__ == "__main__":
    main()
