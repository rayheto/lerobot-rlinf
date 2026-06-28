"""Real SO-101 evaluation and teleoperation data collection.

Two subcommands:

  python src/real.py eval    -- run a trained policy on a real follower arm
  python src/real.py collect -- teleoperate (leader→follower) and record demos

Both write lerobot v2.1 datasets identical in schema to eval_run.py output,
so src/diagnostics/ can compare real vs sim rollouts directly.

No Isaac Sim dependency — runs on any Linux machine with Feetech motors and
USB cameras.
"""

from __future__ import annotations

import argparse
import json
import os
import select
import signal
import sys
import time
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import websockets.sync.client
import yaml

from leisaac.devices.lerobot.common.motors import (
    FeetechMotorsBus,
    Motor,
    MotorCalibration,
    MotorNormMode,
    OperatingMode,
)
from leisaac.policy.openpi import image_tools, msgpack_numpy
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES

# ---------------------------------------------------------------------------
# Constants — duplicated from leisaac.assets.robots.lerobot to avoid pulling
# in isaaclab.sim via that module's top-level imports.
# ---------------------------------------------------------------------------

SO101_MOTOR_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-100.0, 100.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-100.0, 100.0),
    "wrist_flex": (-100.0, 100.0),
    "wrist_roll": (-100.0, 100.0),
    "gripper": (0.0, 100.0),
}

SO101_MOTORS: dict[str, Motor] = {
    "shoulder_pan": Motor(1, "sts3215", MotorNormMode.RANGE_M100_100),
    "shoulder_lift": Motor(2, "sts3215", MotorNormMode.RANGE_M100_100),
    "elbow_flex": Motor(3, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_flex": Motor(4, "sts3215", MotorNormMode.RANGE_M100_100),
    "wrist_roll": Motor(5, "sts3215", MotorNormMode.RANGE_M100_100),
    "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
}

PARQUET_SCHEMA = pa.schema([
    ("action", pa.list_(pa.float32(), 6)),
    ("observation.state", pa.list_(pa.float32(), 6)),
    ("timestamp", pa.float32()),
    ("frame_index", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("task_index", pa.int64()),
])

IMAGE_STATS_MAX_SAMPLES = 150

# ---------------------------------------------------------------------------
# Motor calibration helpers
# ---------------------------------------------------------------------------


def load_calibration(path: str) -> dict[str, MotorCalibration]:
    with open(path) as f:
        data = json.load(f)
    return {
        name: MotorCalibration(
            id=int(m["id"]),
            drive_mode=int(m["drive_mode"]),
            homing_offset=int(m["homing_offset"]),
            range_min=int(m["range_min"]),
            range_max=int(m["range_max"]),
        )
        for name, m in data.items()
    }


def save_calibration(calibration: dict[str, MotorCalibration], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data = {
        k: {
            "id": v.id,
            "drive_mode": v.drive_mode,
            "homing_offset": v.homing_offset,
            "range_min": v.range_min,
            "range_max": v.range_max,
        }
        for k, v in calibration.items()
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def run_calibration(port: str, calib_path: str) -> dict[str, MotorCalibration]:
    bus = FeetechMotorsBus(port=port, motors=dict(SO101_MOTORS))
    bus.connect()
    bus.disable_torque()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

    input("Move arm to the middle of its range of motion and press ENTER...")
    homing_offset = bus.set_half_turn_homings()
    print("Move all joints through their full range of motion.")
    print("Press ENTER to stop recording...")
    range_mins, range_maxes = bus.record_ranges_of_motion()

    calibration = {
        motor: MotorCalibration(
            id=bus.motors[motor].id,
            drive_mode=0,
            homing_offset=homing_offset[motor],
            range_min=range_mins[motor],
            range_max=range_maxes[motor],
        )
        for motor in bus.motors
    }
    bus.write_calibration(calibration)
    save_calibration(calibration, calib_path)
    print(f"Calibration saved to {calib_path}")
    bus.disconnect()
    return calibration


def connect_arm(
    port: str,
    calib_path: str,
    recalibrate: bool = False,
    enable_torque: bool = True,
) -> FeetechMotorsBus:
    if not os.path.exists(calib_path) or recalibrate:
        calibration = run_calibration(port, calib_path)
    else:
        calibration = load_calibration(calib_path)

    bus = FeetechMotorsBus(port=port, motors=dict(SO101_MOTORS), calibration=calibration)
    bus.connect()
    bus.disable_torque()
    bus.configure_motors()
    for motor in bus.motors:
        bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)
    if enable_torque:
        for motor in bus.motors:
            bus.write("Torque_Enable", motor, 1, normalize=False)
    return bus


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------


def _load_camera_calib(yaml_path: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if yaml_path is None:
        return None
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    camera_matrix = np.array(data["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.array(data["distortion_coefficients"], dtype=np.float64)
    return camera_matrix, dist_coeffs


class DualCamera:
    def __init__(
        self,
        front_index: int,
        wrist_index: int,
        width: int,
        height: int,
        front_calib_yaml: str | None = None,
        wrist_calib_yaml: str | None = None,
    ):
        self.front_index = front_index
        self.wrist_index = wrist_index
        self.width = width
        self.height = height
        self._front_cap: cv2.VideoCapture | None = None
        self._wrist_cap: cv2.VideoCapture | None = None

        self._front_undistort = _load_camera_calib(front_calib_yaml)
        self._wrist_undistort = _load_camera_calib(wrist_calib_yaml)
        self._front_new_cam_mtx: np.ndarray | None = None
        self._wrist_new_cam_mtx: np.ndarray | None = None

    def connect(self) -> None:
        self._front_cap = cv2.VideoCapture(self.front_index)
        self._front_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._front_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._wrist_cap = cv2.VideoCapture(self.wrist_index)
        self._wrist_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._wrist_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        if not self._front_cap.isOpened():
            raise RuntimeError(f"Cannot open front camera (index {self.front_index})")
        if not self._wrist_cap.isOpened():
            raise RuntimeError(f"Cannot open wrist camera (index {self.wrist_index})")

        if self._front_undistort is not None:
            mtx, dist = self._front_undistort
            self._front_new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (self.width, self.height), 1, (self.width, self.height),
            )
        if self._wrist_undistort is not None:
            mtx, dist = self._wrist_undistort
            self._wrist_new_cam_mtx, _ = cv2.getOptimalNewCameraMatrix(
                mtx, dist, (self.width, self.height), 1, (self.width, self.height),
            )
        print(
            f"Cameras connected: front={self.front_index} wrist={self.wrist_index} "
            f"({self.width}x{self.height})",
        )

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        ret_f, front = self._front_cap.read()
        ret_w, wrist = self._wrist_cap.read()
        if not ret_f:
            raise RuntimeError("Front camera read failed")
        if not ret_w:
            raise RuntimeError("Wrist camera read failed")
        front = cv2.cvtColor(front, cv2.COLOR_BGR2RGB)
        wrist = cv2.cvtColor(wrist, cv2.COLOR_BGR2RGB)
        if self._front_undistort is not None:
            mtx, dist = self._front_undistort
            front = cv2.undistort(front, mtx, dist, None, self._front_new_cam_mtx)
        if self._wrist_undistort is not None:
            mtx, dist = self._wrist_undistort
            wrist = cv2.undistort(wrist, mtx, dist, None, self._wrist_new_cam_mtx)
        return front, wrist

    def disconnect(self) -> None:
        if self._front_cap is not None:
            self._front_cap.release()
        if self._wrist_cap is not None:
            self._wrist_cap.release()
        print("Cameras disconnected.")


# ---------------------------------------------------------------------------
# Policy client (no radian conversion — real arm is already motor-degrees)
# ---------------------------------------------------------------------------


class RealPolicyClient:
    def __init__(self, host: str, port: int):
        self._uri = f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._ws: websockets.sync.client.ClientConnection | None = None

    def connect(self) -> None:
        print(f"Connecting to policy server at {self._uri}...")
        self._ws = websockets.sync.client.connect(
            self._uri, compression=None, max_size=None,
        )
        _metadata = msgpack_numpy.unpackb(self._ws.recv())
        print("Policy server connected.")

    def infer(
        self,
        state: np.ndarray,
        front_frame: np.ndarray,
        wrist_frame: np.ndarray,
        prompt: str,
    ) -> np.ndarray:
        obs = {
            "images/front": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(front_frame, 224, 224),
            ),
            "images/wrist": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(wrist_frame, 224, 224),
            ),
            "state": state.astype(np.float64),
            "prompt": prompt,
        }
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            raise RuntimeError(f"Policy server error:\n{response}")
        result = msgpack_numpy.unpackb(response)
        return np.asarray(result["actions"], dtype=np.float32)

    def disconnect(self) -> None:
        if self._ws is not None:
            self._ws.close()
            print("Policy client disconnected.")


# ---------------------------------------------------------------------------
# Dataset writing (lerobot v2.1 format — mirrors eval_run.py)
# ---------------------------------------------------------------------------


class ImageStatsAccumulator:
    def __init__(self, max_samples: int = IMAGE_STATS_MAX_SAMPLES):
        self.max_samples = max_samples
        self.reset()

    def reset(self) -> None:
        self._sum = np.zeros(3, dtype=np.float64)
        self._sum_sq = np.zeros(3, dtype=np.float64)
        self._min = np.full(3, np.inf, dtype=np.float64)
        self._max = np.full(3, -np.inf, dtype=np.float64)
        self._count = 0

    def add(self, frame: np.ndarray) -> None:
        f = frame.astype(np.float64) / 255.0
        flat = f.reshape(-1, 3)
        self._sum += flat.mean(axis=0)
        self._sum_sq += (flat**2).mean(axis=0)
        self._min = np.minimum(self._min, flat.min(axis=0))
        self._max = np.maximum(self._max, flat.max(axis=0))
        self._count += 1

    def finalize(self) -> dict:
        if self._count == 0:
            z = np.zeros(3, dtype=np.float64)
            return self._fmt(z, z, z, z, 0)
        mean = self._sum / self._count
        var = np.maximum(self._sum_sq / self._count - mean**2, 0.0)
        return self._fmt(self._min, self._max, mean, np.sqrt(var), self._count)

    @staticmethod
    def _fmt(mn, mx, mean, std, count) -> dict:
        def _s(v):
            return [[[float(x)]] for x in v]
        return {"min": _s(mn), "max": _s(mx), "mean": _s(mean), "std": _s(std), "count": [int(count)]}


def _vec_stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        d = arr.shape[1] if arr.ndim == 2 else 1
        z = [0.0] * d
        return {"min": z, "max": z, "mean": z, "std": z, "count": [0]}
    return {
        "min": arr.min(axis=0).astype(float).tolist(),
        "max": arr.max(axis=0).astype(float).tolist(),
        "mean": arr.mean(axis=0).astype(float).tolist(),
        "std": arr.std(axis=0).astype(float).tolist(),
        "count": [int(arr.shape[0])],
    }


def _scalar_stats(arr: np.ndarray) -> dict:
    if arr.size == 0:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


def _video_feature(h: int, w: int, fps: int) -> dict:
    vinfo = {
        "video.height": h, "video.width": w,
        "video.codec": "h264", "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False, "video.fps": fps,
        "video.channels": 3, "has_audio": False,
    }
    return {
        "dtype": "video", "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "video_info": {**vinfo, "video.fps": float(fps)},
        "info": vinfo,
    }


class DatasetWriter:
    def __init__(self, dataset_dir: str, fps: int, task_description: str):
        self.root = Path(dataset_dir).resolve()
        self.fps = fps
        self.task = task_description

        self._data_dir = self.root / "data" / "chunk-000"
        self._front_vid_dir = self.root / "videos" / "chunk-000" / "observation.images.front"
        self._wrist_vid_dir = self.root / "videos" / "chunk-000" / "observation.images.wrist"
        self._meta_dir = self.root / "meta"
        for d in (self._data_dir, self._front_vid_dir, self._wrist_vid_dir, self._meta_dir):
            d.mkdir(parents=True, exist_ok=True)

        self._ep_lengths: list[int] = []
        self._ep_outcomes: list[bool] = []
        self._ep_stats: list[dict] = []
        self._global_index = 0
        self._frame_h = 0
        self._frame_w = 0

        self._actions_buf: list[np.ndarray] = []
        self._states_buf: list[np.ndarray] = []
        self._front_stats = ImageStatsAccumulator()
        self._wrist_stats = ImageStatsAccumulator()
        self._front_writer = None
        self._wrist_writer = None
        self._cur_ep_idx = -1
        self._kept = 0

    def start_episode(self, episode_index: int) -> None:
        self._cur_ep_idx = episode_index
        tag = f"episode_{episode_index:06d}"
        self._front_writer = imageio.get_writer(
            str(self._front_vid_dir / f"{tag}.mp4"), fps=self.fps, codec="libx264",
        )
        self._wrist_writer = imageio.get_writer(
            str(self._wrist_vid_dir / f"{tag}.mp4"), fps=self.fps, codec="libx264",
        )
        self._actions_buf.clear()
        self._states_buf.clear()
        self._front_stats.reset()
        self._wrist_stats.reset()
        self._kept = 0

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        front: np.ndarray,
        wrist: np.ndarray,
    ) -> None:
        if self._frame_h == 0:
            self._frame_h, self._frame_w = front.shape[:2]
        self._front_writer.append_data(front)
        self._wrist_writer.append_data(wrist)
        self._states_buf.append(state.astype(np.float32))
        self._actions_buf.append(action.astype(np.float32))

        sample_now = self._kept < 30 or self._kept % 5 == 0
        if sample_now and self._front_stats._count < IMAGE_STATS_MAX_SAMPLES:
            self._front_stats.add(front)
            self._wrist_stats.add(wrist)
        self._kept += 1

    def end_episode(self, episode_index: int, success: bool) -> None:
        self._front_writer.close()
        self._wrist_writer.close()
        ep_len = len(self._actions_buf)
        actions_arr = np.stack(self._actions_buf) if ep_len else np.zeros((0, 6), np.float32)
        states_arr = np.stack(self._states_buf) if ep_len else np.zeros((0, 6), np.float32)
        timestamps = np.arange(ep_len, dtype=np.float32) / float(self.fps)
        frame_idx = np.arange(ep_len, dtype=np.int64)
        indices = np.arange(self._global_index, self._global_index + ep_len, dtype=np.int64)
        ep_idx_col = np.full(ep_len, episode_index, dtype=np.int64)
        task_idx_col = np.zeros(ep_len, dtype=np.int64)

        tag = f"episode_{episode_index:06d}"
        table = pa.Table.from_arrays(
            [
                pa.FixedSizeListArray.from_arrays(
                    pa.array(actions_arr.reshape(-1), type=pa.float32()), 6,
                ),
                pa.FixedSizeListArray.from_arrays(
                    pa.array(states_arr.reshape(-1), type=pa.float32()), 6,
                ),
                pa.array(timestamps, type=pa.float32()),
                pa.array(frame_idx, type=pa.int64()),
                pa.array(ep_idx_col, type=pa.int64()),
                pa.array(indices, type=pa.int64()),
                pa.array(task_idx_col, type=pa.int64()),
            ],
            schema=PARQUET_SCHEMA,
        )
        pq.write_table(table, self._data_dir / f"{tag}.parquet")

        self._ep_stats.append({
            "episode_index": episode_index,
            "stats": {
                "action": _vec_stats(actions_arr),
                "observation.state": _vec_stats(states_arr),
                "observation.images.front": self._front_stats.finalize(),
                "observation.images.wrist": self._wrist_stats.finalize(),
                "timestamp": _scalar_stats(timestamps),
                "frame_index": _scalar_stats(frame_idx),
                "episode_index": _scalar_stats(ep_idx_col),
                "index": _scalar_stats(indices),
                "task_index": _scalar_stats(task_idx_col),
            },
        })
        self._ep_lengths.append(ep_len)
        self._ep_outcomes.append(success)
        self._global_index += ep_len

    def discard_episode(self) -> None:
        self._front_writer.close()
        self._wrist_writer.close()
        tag = f"episode_{self._cur_ep_idx:06d}"
        for p in (
            self._front_vid_dir / f"{tag}.mp4",
            self._wrist_vid_dir / f"{tag}.mp4",
            self._data_dir / f"{tag}.parquet",
        ):
            p.unlink(missing_ok=True)
        self._actions_buf.clear()
        self._states_buf.clear()

    def write_meta(self) -> None:
        total_eps = len(self._ep_lengths)
        info = {
            "codebase_version": "v2.1",
            "robot_type": "so101_follower",
            "total_episodes": total_eps,
            "total_frames": self._global_index,
            "total_tasks": 1,
            "total_videos": total_eps * 2,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": self.fps,
            "splits": {"train": f"0:{total_eps}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32", "shape": [6],
                    "names": [f"{j}.pos" for j in SINGLE_ARM_JOINT_NAMES],
                },
                "observation.state": {
                    "dtype": "float32", "shape": [6],
                    "names": [f"{j}.pos" for j in SINGLE_ARM_JOINT_NAMES],
                },
                "observation.images.front": _video_feature(self._frame_h, self._frame_w, self.fps),
                "observation.images.wrist": _video_feature(self._frame_h, self._frame_w, self.fps),
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        (self._meta_dir / "info.json").write_text(json.dumps(info, indent=4))
        with (self._meta_dir / "tasks.jsonl").open("w") as f:
            f.write(json.dumps({"task_index": 0, "task": self.task}) + "\n")
        with (self._meta_dir / "episodes.jsonl").open("w") as f:
            for i, ln in enumerate(self._ep_lengths):
                f.write(json.dumps({"episode_index": i, "tasks": [self.task], "length": ln}) + "\n")
        with (self._meta_dir / "episodes_stats.jsonl").open("w") as f:
            for entry in self._ep_stats:
                f.write(json.dumps(entry) + "\n")
        (self.root / "results.json").write_text(json.dumps({
            "total_episodes": total_eps,
            "success_count": sum(self._ep_outcomes),
            "success_rate": sum(self._ep_outcomes) / max(1, total_eps),
            "outcomes": self._ep_outcomes,
        }, indent=2))
        print(f"Dataset meta written to {self._meta_dir}")


# ---------------------------------------------------------------------------
# Timing and keyboard helpers
# ---------------------------------------------------------------------------


def busy_wait(duration_s: float) -> None:
    end = time.perf_counter() + duration_s
    while time.perf_counter() < end:
        time.sleep(0.0005)


def poll_key() -> str | None:
    if select.select([sys.stdin], [], [], 0.0)[0]:
        return sys.stdin.readline().strip()[:1]
    return None


def _clamp_action(action: np.ndarray) -> np.ndarray:
    out = action.copy()
    for i, name in enumerate(SINGLE_ARM_JOINT_NAMES):
        lo, hi = SO101_MOTOR_LIMITS[name]
        out[i] = np.clip(out[i], lo, hi)
    return out


def _state_from_bus(bus: FeetechMotorsBus) -> np.ndarray:
    pos = bus.sync_read("Present_Position")
    return np.array([pos[j] for j in SINGLE_ARM_JOINT_NAMES], dtype=np.float32)


def _write_bus(bus: FeetechMotorsBus, action: np.ndarray) -> None:
    values = {j: float(action[i]) for i, j in enumerate(SINGLE_ARM_JOINT_NAMES)}
    bus.sync_write("Goal_Position", values)


# ---------------------------------------------------------------------------
# eval subcommand
# ---------------------------------------------------------------------------


def main_eval(args: argparse.Namespace) -> None:
    follower = connect_arm(
        args.follower_port, args.follower_calib,
        recalibrate=args.recalibrate, enable_torque=True,
    )
    cameras = DualCamera(
        args.front_camera_index, args.wrist_camera_index,
        args.camera_width, args.camera_height,
        front_calib_yaml=args.front_camera_calib,
        wrist_calib_yaml=args.wrist_camera_calib,
    )
    cameras.connect()
    policy = RealPolicyClient(args.policy_host, args.policy_port)
    policy.connect()
    writer = DatasetWriter(args.dataset_dir, args.fps, args.policy_language_instruction)

    def _shutdown(signum, frame):
        print("\nSIGNAL received, disabling torque...")
        follower.disable_torque()
        sys.exit(1)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    episode_index = 0
    success_count = 0

    try:
        while episode_index < args.eval_rounds:
            print(
                f"\n[eval] Episode {episode_index + 1}/{args.eval_rounds}  "
                f"(press 'n'=success, 'r'=fail, 'q'=quit)",
            )
            writer.start_episode(episode_index)
            max_frames = int(args.episode_length_s * args.fps)
            action_queue: list[np.ndarray] = []
            frame_count = 0
            ep_success = False
            user_ended = False

            while frame_count < max_frames:
                t0 = time.perf_counter()

                state = _state_from_bus(follower)
                front, wrist = cameras.capture()

                if len(action_queue) == 0:
                    chunk = policy.infer(state, front, wrist, args.policy_language_instruction)
                    action_queue = list(chunk)

                action = _clamp_action(action_queue.pop(0))
                _write_bus(follower, action)
                writer.add_frame(state, action, front, wrist)
                frame_count += 1

                key = poll_key()
                if key == "n":
                    ep_success = True
                    user_ended = True
                    break
                elif key == "r":
                    user_ended = True
                    break
                elif key == "q":
                    raise KeyboardInterrupt

                elapsed = time.perf_counter() - t0
                busy_wait(1.0 / args.fps - elapsed)

            writer.end_episode(episode_index, ep_success)
            if ep_success:
                success_count += 1
            status = "SUCCESS" if ep_success else ("FAIL (user)" if user_ended else "TIMEOUT")
            print(
                f"[eval] Episode {episode_index + 1}: {status}  "
                f"frames={frame_count}  rate={success_count}/{episode_index + 1}",
            )
            episode_index += 1

        writer.write_meta()
        print(
            f"\n[eval] Done. Success rate: {success_count}/{episode_index} "
            f"({success_count / max(1, episode_index):.1%})",
        )
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial dataset...")
        if episode_index > 0:
            writer.write_meta()
    finally:
        follower.disable_torque()
        follower.disconnect()
        cameras.disconnect()
        policy.disconnect()


# ---------------------------------------------------------------------------
# collect subcommand
# ---------------------------------------------------------------------------


def main_collect(args: argparse.Namespace) -> None:
    leader = connect_arm(
        args.leader_port, args.leader_calib,
        recalibrate=args.recalibrate, enable_torque=False,
    )
    follower = connect_arm(
        args.follower_port, args.follower_calib,
        recalibrate=args.recalibrate, enable_torque=True,
    )
    cameras = DualCamera(
        args.front_camera_index, args.wrist_camera_index,
        args.camera_width, args.camera_height,
        front_calib_yaml=args.front_camera_calib,
        wrist_calib_yaml=args.wrist_camera_calib,
    )
    cameras.connect()
    writer = DatasetWriter(args.dataset_dir, args.fps, args.task_description)

    def _shutdown(signum, frame):
        print("\nSIGNAL received, disabling torque...")
        follower.disable_torque()
        sys.exit(1)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    episode_index = 0

    try:
        while args.max_episodes == 0 or episode_index < args.max_episodes:
            print(
                f"\n[collect] Ready for episode {episode_index + 1}. "
                f"Press 'b'+ENTER to start recording.",
            )
            while True:
                key = poll_key()
                if key == "b":
                    break
                if key == "q":
                    raise KeyboardInterrupt
                time.sleep(0.05)

            print("[collect] Recording... ('n'+ENTER=save, 'r'+ENTER=discard, 'q'+ENTER=quit)")
            writer.start_episode(episode_index)
            frame_count = 0

            while True:
                t0 = time.perf_counter()

                leader_pos = _state_from_bus(leader)
                action = _clamp_action(leader_pos)
                _write_bus(follower, action)
                follower_state = _state_from_bus(follower)
                front, wrist = cameras.capture()

                writer.add_frame(follower_state, action, front, wrist)
                frame_count += 1

                key = poll_key()
                if key == "n":
                    writer.end_episode(episode_index, success=True)
                    print(f"[collect] Episode {episode_index + 1} saved (success). frames={frame_count}")
                    episode_index += 1
                    break
                elif key == "r":
                    writer.discard_episode()
                    print(f"[collect] Episode discarded. frames={frame_count}")
                    break
                elif key == "q":
                    writer.discard_episode()
                    raise KeyboardInterrupt

                elapsed = time.perf_counter() - t0
                busy_wait(1.0 / args.fps - elapsed)

        if episode_index > 0:
            writer.write_meta()
        print(f"\n[collect] Done. {episode_index} episodes saved to {writer.root}")
    except KeyboardInterrupt:
        print("\nInterrupted. Saving partial dataset...")
        if episode_index > 0:
            writer.write_meta()
    finally:
        follower.disable_torque()
        follower.disconnect()
        leader.disconnect()
        cameras.disconnect()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _add_common_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--follower_port", default="/dev/ttyACM1")
    p.add_argument("--follower_calib", default=".cache/so101_follower.json")
    p.add_argument("--front_camera_index", type=int, default=0)
    p.add_argument("--wrist_camera_index", type=int, default=2)
    p.add_argument("--camera_width", type=int, default=640)
    p.add_argument("--camera_height", type=int, default=480)
    p.add_argument("--front_camera_calib", default=None,
                    help="mono-calib YAML for front camera (camera_matrix + distortion_coefficients).")
    p.add_argument("--wrist_camera_calib", default=None,
                    help="mono-calib YAML for wrist camera.")
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--dataset_dir", required=True)
    p.add_argument("--recalibrate", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Real SO-101 evaluation and teleoperation data collection.",
    )
    subs = parser.add_subparsers(dest="command", required=True)

    # -- eval --
    p_eval = subs.add_parser("eval", help="Run policy on real follower arm and record rollouts.")
    _add_common_args(p_eval)
    p_eval.add_argument("--policy_host", default="localhost")
    p_eval.add_argument("--policy_port", type=int, default=8000)
    p_eval.add_argument("--policy_action_horizon", type=int, default=10)
    p_eval.add_argument("--policy_language_instruction", default="Grab orange and place into plate")
    p_eval.add_argument("--episode_length_s", type=float, default=60.0)
    p_eval.add_argument("--eval_rounds", type=int, default=20)

    # -- collect --
    p_col = subs.add_parser("collect", help="Teleoperate (leader→follower) and record demos.")
    _add_common_args(p_col)
    p_col.add_argument("--leader_port", default="/dev/ttyACM0")
    p_col.add_argument("--leader_calib", default=".cache/so101_leader.json")
    p_col.add_argument("--task_description", default="Grab orange and place into plate")
    p_col.add_argument("--max_episodes", type=int, default=0, help="0 = unlimited.")

    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "eval":
        main_eval(args)
    elif args.command == "collect":
        main_collect(args)


if __name__ == "__main__":
    main()
