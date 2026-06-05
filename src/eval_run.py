"""Headless inference loop for SO-101 pick-orange — runs inside rlinf-isaacsim-env.

Replaces leisaac/scripts/evaluation/policy_inference.py with three upgrades:

  * Offscreen video capture (one mp4 per episode, no live GUI window).
  * Async chunk prefetch (off by default — see eval.py for why).
  * Writes a lerobot v2.1-format dataset alongside the videos so eval
    rollouts can be compared 1:1 against the training data at
    EverNorif/leisaac-pick-orange. Layout:

        <dataset_dir>/
          data/chunk-000/episode_NNNNNN.parquet
          videos/chunk-000/observation.images.front/episode_NNNNNN.mp4
          videos/chunk-000/observation.images.wrist/episode_NNNNNN.mp4
          meta/{info.json, tasks.jsonl, episodes.jsonl, episodes_stats.jsonl}

    Parquet columns match the reference dataset exactly:
      action, observation.state    fixed_size_list<float32>[6]   (motor degrees)
      timestamp                    float32                       (seconds)
      frame_index/episode_index/index/task_index   int64

    Codec deviates: reference uses av1, we use libx264 (~10x faster encode,
    same container, lerobot's pyav reader is codec-agnostic).

Invoked by src/eval.py — see that script for the two-venv split.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if multiprocessing.get_start_method(allow_none=True) != "spawn":
    multiprocessing.set_start_method("spawn", force=True)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="SO-101 pick-orange eval with video + lerobot dataset output.")
parser.add_argument("--task", type=str, required=True)
parser.add_argument("--step_hz", type=int, default=60)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument("--episode_length_s", type=float, default=60.0)
parser.add_argument("--eval_rounds", type=int, default=20)
parser.add_argument("--policy_host", type=str, default="localhost")
parser.add_argument("--policy_port", type=int, default=8000)
parser.add_argument("--policy_action_horizon", type=int, default=10)
parser.add_argument("--policy_language_instruction", type=str, required=True)
parser.add_argument("--dataset_dir", type=str, required=True,
                    help="Root of the lerobot v2.1 output dataset.")
parser.add_argument("--dataset_fps", type=int, default=30,
                    help="Output dataset fps. Must divide sim_fps evenly "
                         "(60Hz sim / 30Hz dataset = subsample 2:1).")
parser.add_argument("--prefetch", action="store_true", default=True)
parser.add_argument("--no-prefetch", dest="prefetch", action="store_false")
parser.add_argument("--prefetch-latency-ms", type=float, default=120.0,
                    help="Wall-clock budget reserved for the in-flight infer; "
                    "fire next chunk this many ms before current chunk ends.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# Headless + cameras are non-negotiable for this loop.
args_cli.headless = True
args_cli.enable_cameras = True

app_launcher = AppLauncher(vars(args_cli))
simulation_app = app_launcher.app

# Imports below depend on the simulation app being live.
import gymnasium as gym  # noqa: E402
import imageio.v2 as imageio  # noqa: E402
import numpy as np  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402
import torch  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.sensors import Camera, TiledCamera  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import leisaac  # noqa: F401, E402
from leisaac.policy import OpenPIServicePolicyClient  # noqa: E402
from leisaac.utils.constant import SINGLE_ARM_JOINT_NAMES  # noqa: E402
from leisaac.utils.env_utils import dynamic_reset_gripper_effort_limit_sim, get_task_type  # noqa: E402
from leisaac.utils.robot_utils import convert_leisaac_action_to_lerobot  # noqa: E402


# Match reference dataset columns and dtype precisely.
PARQUET_SCHEMA = pa.schema([
    ("action", pa.list_(pa.float32(), 6)),
    ("observation.state", pa.list_(pa.float32(), 6)),
    ("timestamp", pa.float32()),
    ("frame_index", pa.int64()),
    ("episode_index", pa.int64()),
    ("index", pa.int64()),
    ("task_index", pa.int64()),
])

# How many frames per episode contribute to the per-channel image stats.
# Reference dataset shows ~146 samples for 774-frame episodes (~1-in-5).
IMAGE_STATS_MAX_SAMPLES = 150


class RateLimiter:
    """Wall-clock 60 Hz pacing, copied verbatim from leisaac's loop."""

    def __init__(self, hz: int):
        self.hz = hz
        self.last_time = time.time()
        self.sleep_duration = 1.0 / hz
        self.render_period = min(0.0166, self.sleep_duration)

    def sleep(self, env) -> None:
        next_wakeup_time = self.last_time + self.sleep_duration
        while time.time() < next_wakeup_time:
            time.sleep(self.render_period)
            env.sim.render()
        self.last_time = self.last_time + self.sleep_duration
        if self.last_time < time.time():
            while self.last_time < time.time():
                self.last_time += self.sleep_duration


def _snapshot_obs(obs_policy: dict, prompt: str) -> dict:
    """Clone tensor entries to CPU so the prefetch thread sees a stable view
    while the main thread keeps stepping the env in-place."""
    out: dict = {}
    for k, v in obs_policy.items():
        if torch.is_tensor(v):
            out[k] = v.detach().to("cpu", copy=True)
        else:
            out[k] = v
    out["task_description"] = prompt
    return out


def _video_frame(obs_policy: dict, key: str) -> np.ndarray:
    t = obs_policy[key]
    if not torch.is_tensor(t):
        raise RuntimeError(f"obs['policy'][{key!r}] is not a tensor: {type(t)}")
    if t.dim() == 4:
        t = t[0]
    arr = t.detach().cpu().numpy()
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


def _lerobot_state_from_obs(obs_policy: dict) -> np.ndarray:
    """Joint state in lerobot (motor-degree) convention, shape (6,) float32."""
    jp = obs_policy["joint_pos"]
    if torch.is_tensor(jp):
        jp = jp.detach().cpu()
    # convert_leisaac_action_to_lerobot expects (N, 6)
    return convert_leisaac_action_to_lerobot(jp).astype(np.float32)[0]


def _lerobot_action_from_leisaac(action: torch.Tensor) -> np.ndarray:
    """action: (1, 6) torch tensor in leisaac rad convention → (6,) float32 lerobot motor degrees."""
    return convert_leisaac_action_to_lerobot(action).astype(np.float32)[0]


class ImageStatsAccumulator:
    """Per-channel min/max/mean/std normalized to [0,1] over a sampled
    subset of frames. Matches the reference dataset's stats shape:
    (3, 1, 1) per stat, count = #sampled frames."""

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
        # Frame is (H, W, 3) uint8; reduce to per-channel scalar stats over pixels.
        f = frame.astype(np.float64) / 255.0
        per_ch_mean = f.reshape(-1, 3).mean(axis=0)
        per_ch_sq = (f.reshape(-1, 3) ** 2).mean(axis=0)
        per_ch_min = f.reshape(-1, 3).min(axis=0)
        per_ch_max = f.reshape(-1, 3).max(axis=0)
        # Accumulate the per-frame averages, mimicking lerobot's "one stat per
        # frame, averaged across frames" convention.
        self._sum += per_ch_mean
        self._sum_sq += per_ch_sq
        self._min = np.minimum(self._min, per_ch_min)
        self._max = np.maximum(self._max, per_ch_max)
        self._count += 1

    def finalize(self) -> dict:
        if self._count == 0:
            zeros = np.zeros(3, dtype=np.float64)
            return self._stats_dict(zeros, zeros, zeros, zeros, 0)
        mean = self._sum / self._count
        mean_sq = self._sum_sq / self._count
        var = np.maximum(mean_sq - mean ** 2, 0.0)
        std = np.sqrt(var)
        return self._stats_dict(self._min, self._max, mean, std, self._count)

    @staticmethod
    def _stats_dict(mn, mx, mean, std, count) -> dict:
        # Nested-list shape (3, 1, 1) matches the reference dataset.
        def _shape(v):
            return [[[float(x)]] for x in v]
        return {
            "min": _shape(mn),
            "max": _shape(mx),
            "mean": _shape(mean),
            "std": _shape(std),
            "count": [int(count)],
        }


def _vec_stats(arr: np.ndarray) -> dict:
    """Per-channel min/max/mean/std for an (N, D) array. count = [N]."""
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
    """Stats for a scalar 1-D column, returned as 1-element lists to match the reference."""
    if arr.size == 0:
        return {"min": [0.0], "max": [0.0], "mean": [0.0], "std": [0.0], "count": [0]}
    return {
        "min": [float(arr.min())],
        "max": [float(arr.max())],
        "mean": [float(arr.mean())],
        "std": [float(arr.std())],
        "count": [int(arr.size)],
    }


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    task_type = get_task_type(args_cli.task)
    env_cfg.use_teleop_device(task_type)
    env_cfg.seed = args_cli.seed if args_cli.seed is not None else int(time.time())
    env_cfg.episode_length_s = args_cli.episode_length_s
    if args_cli.eval_rounds <= 0 and hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    env_cfg.recorders = None

    env: ManagerBasedRLEnv = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    camera_keys = [
        k for k, s in env.scene.sensors.items() if isinstance(s, (Camera, TiledCamera))
    ]
    print(f"[eval_run] camera keys: {camera_keys}", flush=True)
    for needed in ("front", "wrist"):
        if needed not in camera_keys:
            raise RuntimeError(f"camera {needed!r} missing — got {camera_keys}")
    policy = OpenPIServicePolicyClient(
        host=args_cli.policy_host,
        port=args_cli.policy_port,
        camera_keys=camera_keys,
        task_type=task_type,
    )

    rate_limiter = RateLimiter(args_cli.step_hz)

    # sim_fps drives the rendered frame cadence; dataset_fps may subsample.
    sim_fps = int(round(1.0 / env.step_dt))
    if sim_fps % args_cli.dataset_fps != 0:
        raise RuntimeError(
            f"--dataset_fps={args_cli.dataset_fps} must divide sim_fps={sim_fps} evenly"
        )
    subsample = sim_fps // args_cli.dataset_fps

    dataset_dir = Path(args_cli.dataset_dir).resolve()
    data_dir = dataset_dir / "data" / "chunk-000"
    front_video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.front"
    wrist_video_dir = dataset_dir / "videos" / "chunk-000" / "observation.images.wrist"
    meta_dir = dataset_dir / "meta"
    for d in (data_dir, front_video_dir, wrist_video_dir, meta_dir):
        d.mkdir(parents=True, exist_ok=True)

    print(
        f"[eval_run] dataset_dir={dataset_dir}  sim_fps={sim_fps}  "
        f"dataset_fps={args_cli.dataset_fps}  subsample=1:{subsample}",
        flush=True,
    )

    step_period_ms = 1000.0 / args_cli.step_hz
    prefetch_steps_left = max(1, int(np.ceil(args_cli.prefetch_latency_ms / step_period_ms)))
    print(
        f"[eval_run] prefetch={args_cli.prefetch} step_hz={args_cli.step_hz} "
        f"horizon={args_cli.policy_action_horizon} fire when "
        f"{prefetch_steps_left} steps left in chunk",
        flush=True,
    )

    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="prefetch")

    obs_dict, _ = env.reset()
    success_count = 0
    episode_index = 0  # zero-indexed to match the reference dataset
    global_index = 0   # running row counter across all episodes
    episode_lengths: list[int] = []   # for meta/episodes.jsonl
    episode_outcomes: list[bool] = []
    episodes_stats: list[dict] = []   # for meta/episodes_stats.jsonl

    # Discover frame shape at first capture for info.json.
    sample_frame = _video_frame(obs_dict["policy"], "front")
    frame_h, frame_w, frame_c = sample_frame.shape
    print(
        f"[eval_run] camera resolution: {frame_w}x{frame_h}x{frame_c}",
        flush=True,
    )

    try:
        while episode_index < args_cli.eval_rounds:
            print(f"[Evaluation] Evaluating episode {episode_index + 1}...", flush=True)

            ep_tag = f"episode_{episode_index:06d}"
            front_path = front_video_dir / f"{ep_tag}.mp4"
            wrist_path = wrist_video_dir / f"{ep_tag}.mp4"
            front_writer = imageio.get_writer(
                str(front_path), fps=args_cli.dataset_fps, codec="libx264"
            )
            wrist_writer = imageio.get_writer(
                str(wrist_path), fps=args_cli.dataset_fps, codec="libx264"
            )

            actions_buf: list[np.ndarray] = []
            states_buf: list[np.ndarray] = []
            front_stats = ImageStatsAccumulator()
            wrist_stats = ImageStatsAccumulator()
            sim_step_in_ep = 0
            kept_in_ep = 0
            # Image stats are heavy — cap at ~1-in-5 frames after the first 30
            # (matches the reference dataset's ~146 samples on 774-frame eps).
            STATS_WARMUP = 30
            STATS_STRIDE = 5

            success = False
            time_out = False

            snap = _snapshot_obs(obs_dict["policy"], args_cli.policy_language_instruction)
            pending = executor.submit(policy.get_action, snap)

            with torch.inference_mode():
                while simulation_app.is_running():
                    actions = pending.result().to(env.device)
                    pending = None
                    n_steps = min(args_cli.policy_action_horizon, actions.shape[0])
                    fire_at = max(0, n_steps - prefetch_steps_left) if args_cli.prefetch else -1

                    for i in range(n_steps):
                        action = actions[i, :, :]
                        if env.cfg.dynamic_reset_gripper_effort_limit:
                            dynamic_reset_gripper_effort_limit_sim(env, task_type)
                        obs_dict, _, term, tout, _ = env.step(action)

                        # Subsample to dataset_fps. Keep step 0 of each chunk
                        # of `subsample` sim-steps. At sim_fps=60 / ds=30 this
                        # is every other env step.
                        if sim_step_in_ep % subsample == 0:
                            front_frame = _video_frame(obs_dict["policy"], "front")
                            wrist_frame = _video_frame(obs_dict["policy"], "wrist")
                            front_writer.append_data(front_frame)
                            wrist_writer.append_data(wrist_frame)

                            states_buf.append(_lerobot_state_from_obs(obs_dict["policy"]))
                            actions_buf.append(_lerobot_action_from_leisaac(action))

                            sample_now = kept_in_ep < STATS_WARMUP or (
                                kept_in_ep % STATS_STRIDE == 0
                            )
                            if sample_now and (
                                front_stats._count + wrist_stats._count
                            ) < 2 * IMAGE_STATS_MAX_SAMPLES:
                                front_stats.add(front_frame)
                                wrist_stats.add(wrist_frame)
                            kept_in_ep += 1

                        sim_step_in_ep += 1

                        if term[0]:
                            success = True
                            break
                        if tout[0]:
                            time_out = True
                            break

                        if args_cli.prefetch and pending is None and i == fire_at:
                            snap = _snapshot_obs(
                                obs_dict["policy"], args_cli.policy_language_instruction
                            )
                            pending = executor.submit(policy.get_action, snap)

                        rate_limiter.sleep(env)

                    if success or time_out:
                        break
                    if pending is None:
                        snap = _snapshot_obs(
                            obs_dict["policy"], args_cli.policy_language_instruction
                        )
                        pending = executor.submit(policy.get_action, snap)

            front_writer.close()
            wrist_writer.close()
            # Drain any in-flight prefetch so the next env.reset isn't racing it.
            if pending is not None:
                try:
                    pending.result(timeout=5.0)
                except Exception:
                    pass
                pending = None

            # ---- Persist parquet ----
            ep_len = len(actions_buf)
            if ep_len == 0:
                print(f"[eval_run] WARN: episode {episode_index} had 0 kept frames", flush=True)
            actions_arr = np.stack(actions_buf, axis=0) if ep_len else np.zeros((0, 6), np.float32)
            states_arr = np.stack(states_buf, axis=0) if ep_len else np.zeros((0, 6), np.float32)
            timestamps = (np.arange(ep_len, dtype=np.float32) / float(args_cli.dataset_fps))
            frame_idx = np.arange(ep_len, dtype=np.int64)
            indices = np.arange(global_index, global_index + ep_len, dtype=np.int64)
            ep_idx_col = np.full(ep_len, episode_index, dtype=np.int64)
            task_idx_col = np.zeros(ep_len, dtype=np.int64)

            parquet_path = data_dir / f"{ep_tag}.parquet"
            table = pa.Table.from_arrays(
                [
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(actions_arr.reshape(-1), type=pa.float32()), 6
                    ),
                    pa.FixedSizeListArray.from_arrays(
                        pa.array(states_arr.reshape(-1), type=pa.float32()), 6
                    ),
                    pa.array(timestamps, type=pa.float32()),
                    pa.array(frame_idx, type=pa.int64()),
                    pa.array(ep_idx_col, type=pa.int64()),
                    pa.array(indices, type=pa.int64()),
                    pa.array(task_idx_col, type=pa.int64()),
                ],
                schema=PARQUET_SCHEMA,
            )
            pq.write_table(table, parquet_path)

            # ---- Per-episode stats ----
            ep_stats = {
                "action": _vec_stats(actions_arr),
                "observation.state": _vec_stats(states_arr),
                "observation.images.front": front_stats.finalize(),
                "observation.images.wrist": wrist_stats.finalize(),
                "timestamp": _scalar_stats(timestamps),
                "frame_index": _scalar_stats(frame_idx),
                "episode_index": _scalar_stats(ep_idx_col),
                "index": _scalar_stats(indices),
                "task_index": _scalar_stats(task_idx_col),
            }
            episodes_stats.append({"episode_index": episode_index, "stats": ep_stats})
            episode_lengths.append(ep_len)
            episode_outcomes.append(success)
            global_index += ep_len

            if success:
                success_count += 1
                print(
                    f"[Evaluation] Episode {episode_index + 1} is successful!  "
                    f"frames={ep_len}  video={front_path}",
                    flush=True,
                )
            else:
                print(
                    f"[Evaluation] Episode {episode_index + 1} timed out!  "
                    f"frames={ep_len}  video={front_path}",
                    flush=True,
                )
            print(
                f"[Evaluation] now success rate: {success_count/(episode_index+1):.4f} "
                f" [{success_count}/{episode_index+1}]",
                flush=True,
            )
            episode_index += 1

        # ---- Write meta files ----
        prompt = args_cli.policy_language_instruction
        info = {
            "codebase_version": "v2.1",
            "robot_type": "so101_follower",
            "total_episodes": episode_index,
            "total_frames": global_index,
            "total_tasks": 1,
            "total_videos": episode_index * 2,
            "total_chunks": 1,
            "chunks_size": 1000,
            "fps": args_cli.dataset_fps,
            "splits": {"train": f"0:{episode_index}"},
            "data_path": "data/chunk-{episode_chunk:03d}/episode_{episode_index:06d}.parquet",
            "video_path": "videos/chunk-{episode_chunk:03d}/{video_key}/episode_{episode_index:06d}.mp4",
            "features": {
                "action": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [f"{j}.pos" for j in SINGLE_ARM_JOINT_NAMES],
                },
                "observation.state": {
                    "dtype": "float32",
                    "shape": [6],
                    "names": [f"{j}.pos" for j in SINGLE_ARM_JOINT_NAMES],
                },
                "observation.images.front": _video_feature(frame_h, frame_w, args_cli.dataset_fps),
                "observation.images.wrist": _video_feature(frame_h, frame_w, args_cli.dataset_fps),
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None},
            },
        }
        (meta_dir / "info.json").write_text(json.dumps(info, indent=4))
        with (meta_dir / "tasks.jsonl").open("w") as f:
            f.write(json.dumps({"task_index": 0, "task": prompt}) + "\n")
        with (meta_dir / "episodes.jsonl").open("w") as f:
            for i, ln in enumerate(episode_lengths):
                f.write(json.dumps({"episode_index": i, "tasks": [prompt], "length": ln}) + "\n")
        with (meta_dir / "episodes_stats.jsonl").open("w") as f:
            for entry in episodes_stats:
                f.write(json.dumps(entry) + "\n")

        # Side-channel summary file so eval.py can pick up the rate without re-parsing logs.
        (dataset_dir / "results.json").write_text(json.dumps({
            "total_episodes": episode_index,
            "success_count": success_count,
            "success_rate": success_count / max(1, episode_index),
            "outcomes": episode_outcomes,
        }, indent=2))

        print(
            f"[Evaluation] Final success rate: {success_count/args_cli.eval_rounds:.3f} "
            f" [{success_count}/{args_cli.eval_rounds}]",
            flush=True,
        )
        print(f"[Evaluation] Dataset written to {dataset_dir}", flush=True)
    finally:
        executor.shutdown(wait=False, cancel_futures=True)
        env.close()
        simulation_app.close()


def _video_feature(h: int, w: int, fps: int) -> dict:
    """info.json feature stanza for a video stream — matches reference layout."""
    vinfo = {
        "video.height": h,
        "video.width": w,
        # Reference dataset is av1; we encode libx264 for speed. The pyav reader
        # in lerobot is codec-agnostic so this is purely cosmetic.
        "video.codec": "h264",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": False,
        "video.fps": fps,
        "video.channels": 3,
        "has_audio": False,
    }
    return {
        "dtype": "video",
        "shape": [h, w, 3],
        "names": ["height", "width", "channels"],
        "video_info": {**vinfo, "video.fps": float(fps)},
        "info": vinfo,
    }


if __name__ == "__main__":
    main()
