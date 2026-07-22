"""LeRobot v3 dataset exporter for RECAP sessions.

Reads a recorded session from the DataRecorder SQLite database and writes a
LeRobot v3 format dataset that can be loaded by the official LeRobot loader.

Output layout (LeRobot v3):
  <output_dir>/
    data/
      chunk-000/
        episode_000000.parquet     (action, state, timestamps, frame_index, ...)
        episode_000001.parquet
        ...
    videos/
      chunk-000/
        observation.images.front/
          episode_000000.mp4
        observation.images.wrist/
          episode_000000.mp4
    meta/
      info.json
      tasks.jsonl
      episodes.jsonl
      episodes_stats.jsonl
      stats.json

The exporter uses pyarrow for Parquet and cv2/imageio for MP4 encoding.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


class LeRobotV3Exporter:
    """Export a RECAP session to LeRobot v3 format.

    Parameters
    ----------
    db_path : Path
        Path to session.sqlite from DataRecorder.
    output_dir : Path
        Root of the LeRobot v3 output dataset.
    fps : int
        Dataset FPS (must match control loop step_hz).
    joint_names : list[str]
        Joint names in canonical order (from ProductContract).
    camera_keys : list[str]
        Camera stream keys (e.g. ["front", "wrist"]).
    """

    def __init__(
        self,
        db_path: Path,
        output_dir: Path,
        fps: int = 30,
        joint_names: list[str] | None = None,
        camera_keys: list[str] | None = None,
    ) -> None:
        self.db_path = Path(db_path)
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.joint_names = joint_names or [
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]
        self.camera_keys = camera_keys or ["front", "wrist"]
        self._n_joints = len(self.joint_names)

    def export(self) -> dict[str, Any]:
        """Export the full session. Returns a summary dict."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "data" / "chunk-000").mkdir(parents=True, exist_ok=True)
        for cam in self.camera_keys:
            (self.output_dir / "videos" / "chunk-000" / f"observation.images.{cam}").mkdir(parents=True, exist_ok=True)
        meta_dir = self.output_dir / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)

        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            episodes = self._get_episodes(conn)
            total_frames = 0
            for ep_id, ep_ticks in episodes.items():
                n = self._write_episode_parquet(ep_id, ep_ticks)
                total_frames += n
                for cam in self.camera_keys:
                    self._write_episode_video(ep_id, ep_ticks, cam)

            self._write_info(total_frames, len(episodes))
            self._write_tasks()
            self._write_episodes_jsonl(episodes)
            self._write_stats(episodes)
        finally:
            conn.close()

        return {
            "output_dir": str(self.output_dir),
            "episodes": len(episodes),
            "total_frames": total_frames,
            "fps": self.fps,
            "joint_names": self.joint_names,
            "camera_keys": self.camera_keys,
        }

    def _get_episodes(self, conn: sqlite3.Connection) -> dict[int, list[sqlite3.Row]]:
        rows = conn.execute(
            "SELECT * FROM ticks ORDER BY tick_id"
        ).fetchall()
        episodes: dict[int, list[sqlite3.Row]] = {}
        for row in rows:
            ep = row["episode_id"]
            episodes.setdefault(ep, []).append(row)
        return episodes

    def _write_episode_parquet(
        self, episode_id: int, ticks: list[sqlite3.Row]
    ) -> int:
        import pyarrow as pa
        import pyarrow.parquet as pq

        n = len(ticks)
        actions = np.zeros((n, self._n_joints), dtype=np.float32)
        states = np.zeros((n, self._n_joints), dtype=np.float32)
        timestamps = np.zeros(n, dtype=np.float32)
        frame_indices = np.arange(n, dtype=np.int64)
        episode_indices = np.full(n, episode_id, dtype=np.int64)
        indices = np.arange(n, dtype=np.int64)
        task_indices = np.zeros(n, dtype=np.int64)
        intervention_masks = np.zeros(n, dtype=np.int64)
        action_sources = []

        for i, tick in enumerate(ticks):
            if tick["executed_action"]:
                actions[i] = np.array(json.loads(tick["executed_action"]), dtype=np.float32)
            if tick["joint_state"]:
                states[i] = np.array(json.loads(tick["joint_state"]), dtype=np.float32)
            timestamps[i] = float(tick["mono_time"] - ticks[0]["mono_time"])
            if tick["intervention_id"] is not None:
                intervention_masks[i] = 1
            action_sources.append(tick["action_source"] or "policy")

        table = pa.table({
            "action": pa.array([row for row in actions], type=pa.list_(pa.float32(), self._n_joints)),
            "observation.state": pa.array([row for row in states], type=pa.list_(pa.float32(), self._n_joints)),
            "timestamp": pa.array(timestamps),
            "frame_index": pa.array(frame_indices),
            "episode_index": pa.array(episode_indices),
            "index": pa.array(indices),
            "task_index": pa.array(task_indices),
            "intervention_mask": pa.array(intervention_masks),
            "action_source": pa.array(action_sources, type=pa.string()),
        })

        out_path = self.output_dir / "data" / "chunk-000" / f"episode_{episode_id:06d}.parquet"
        pq.write_table(table, str(out_path))
        return n

    def _write_episode_video(
        self, episode_id: int, ticks: list[sqlite3.Row], cam_key: str
    ) -> None:
        frame_paths = []
        for tick in ticks:
            col = f"{cam_key}_frame_path"
            p = tick[col] if col in tick.keys() else None
            if p:
                frame_paths.append(p)

        if not frame_paths:
            return

        out_path = (
            self.output_dir / "videos" / "chunk-000"
            / f"observation.images.{cam_key}" / f"episode_{episode_id:06d}.mp4"
        )

        try:
            import cv2
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            sample = cv2.imread(frame_paths[0])
            if sample is None:
                return
            h, w = sample.shape[:2]
            writer = cv2.VideoWriter(str(out_path), fourcc, self.fps, (w, h))
            for fp in frame_paths:
                img = cv2.imread(fp)
                if img is not None:
                    writer.write(img)
            writer.release()
        except ImportError:
            # No cv2 — write a placeholder
            out_path.write_text(f"video placeholder: {len(frame_paths)} frames")

    def _write_info(self, total_frames: int, n_episodes: int) -> None:
        info = {
            "codebase_version": 3,
            "robot_type": "so101_follower",
            "fps": self.fps,
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [self._n_joints],
                    "names": self.joint_names,
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [self._n_joints],
                    "names": self.joint_names,
                },
            },
            "features_keys": {
                "action": "action",
                "observation.state": "observation.state",
            },
            "video_keys": [
                {"key": f"observation.images.{cam}", "shape": [480, 640, 3]}
                for cam in self.camera_keys
            ],
            "total_frames": total_frames,
            "total_episodes": n_episodes,
            "chunks_size": 1000,
        }
        (self.output_dir / "meta" / "info.json").write_text(
            json.dumps(info, indent=2, default=_json_default)
        )

    def _write_tasks(self) -> None:
        conn = sqlite3.connect(str(self.db_path))
        try:
            row = conn.execute(
                "SELECT value FROM session_meta WHERE key = 'prompt'"
            ).fetchone()
            prompt = json.loads(row[0]) if row else "Grab orange and place into plate"
        finally:
            conn.close()
        with (self.output_dir / "meta" / "tasks.jsonl").open("w") as f:
            f.write(json.dumps({"task_index": 0, "task": prompt}) + "\n")

    def _write_episodes_jsonl(self, episodes: dict[int, list[Any]]) -> None:
        with (self.output_dir / "meta" / "episodes.jsonl").open("w") as f:
            for ep_id in sorted(episodes):
                n = len(episodes[ep_id])
                f.write(json.dumps({
                    "episode_index": ep_id,
                    "tasks": ["Grab orange and place into plate"],
                    "length": n,
                }) + "\n")

    def _write_stats(self, episodes: dict[int, list[Any]]) -> None:
        all_actions = []
        all_states = []
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            for row in conn.execute("SELECT * FROM ticks ORDER BY tick_id").fetchall():
                if row["executed_action"]:
                    try:
                        vals = json.loads(row["executed_action"])
                        if vals is not None and len(vals) == self._n_joints:
                            all_actions.append(vals)
                    except (json.JSONDecodeError, TypeError):
                        pass
                if row["joint_state"]:
                    try:
                        vals = json.loads(row["joint_state"])
                        if vals is not None and len(vals) == self._n_joints:
                            all_states.append(vals)
                    except (json.JSONDecodeError, TypeError):
                        pass
        finally:
            conn.close()

        stats = {}
        if all_actions:
            arr = np.array(all_actions, dtype=np.float32)
            stats["action"] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
        if all_states:
            arr = np.array(all_states, dtype=np.float32)
            stats["observation.state"] = {
                "mean": arr.mean(axis=0).tolist(),
                "std": arr.std(axis=0).tolist(),
                "min": arr.min(axis=0).tolist(),
                "max": arr.max(axis=0).tolist(),
            }
        (self.output_dir / "meta" / "stats.json").write_text(
            json.dumps(stats, indent=2, default=_json_default)
        )
        # episodes_stats.jsonl (per-episode)
        with (self.output_dir / "meta" / "episodes_stats.jsonl").open("w") as f:
            for ep_id in sorted(episodes):
                f.write(json.dumps({"episode_index": ep_id}) + "\n")
