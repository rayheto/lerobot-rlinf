"""Residual-action SO-101 PickOrange env for RLinf.

RLinf sees obs["states"] = concat(so101_state, frozen_pi05_base_action) and
outputs only a residual action. The wrapped IsaacLab task receives
base_action + residual_action in lerobot motor-degree units.
"""
from __future__ import annotations

import os
import pathlib
import socket
import time
from typing import Any

import torch

from src.rl.envs.isaaclab_pick_orange import IsaaclabPickOrangeEnv
from src.rl.simple._openpi_server import (
    kill_server,
    spawn_openpi_server,
    wait_for_port,
)
from src.rl.simple.policy import _Pi05JaxWrapper

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
RESIDUAL_TASK_ID = "LeIsaac-SO101-PickOrange-Residual-v0"


def _resolve_path(path: str) -> pathlib.Path:
    p = pathlib.Path(path).expanduser()
    if not p.is_absolute():
        p = _REPO_ROOT / p
    return p


def _listener_exists(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def _worker_gpu_rank(worker_info: Any) -> int:
    rank = getattr(worker_info, "accelerator_rank", -1) if worker_info is not None else -1
    if rank is not None and int(rank) >= 0:
        return int(rank)
    for key in ("LOCAL_ACCELERATOR_RANK", "LOCAL_RANK"):
        value = os.environ.get(key)
        if value is not None:
            try:
                return int(value)
            except ValueError:
                pass
    return 0


class IsaaclabPickOrangeResidualEnv(IsaaclabPickOrangeEnv):
    """PickOrange wrapper that consumes residual actions from RLinf PPO."""

    def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
        self._residual_cfg = cfg.get("residual", {})
        self._openpi_proc = None
        self._pi05: _Pi05JaxWrapper | None = None
        self._pending_base: torch.Tensor | None = None
        self._force_refill_mask: torch.Tensor | None = None
        self._last_executed_base: torch.Tensor | None = None
        self._last_executed_residual: torch.Tensor | None = None
        self._last_executed_full_action: torch.Tensor | None = None
        self._residual_l2_sum: torch.Tensor | None = None
        self._residual_l2_count: torch.Tensor | None = None
        self._residual_l2_max: torch.Tensor | None = None
        self._worker_gpu_rank = _worker_gpu_rank(worker_info)

        super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)
        self._start_or_reuse_openpi_server()
        self._pi05 = _Pi05JaxWrapper(
            host=str(self._residual_cfg.get("openpi_host", "127.0.0.1")),
            port=self._openpi_port(),
            prompt=self.task_description,
            chunk_horizon=int(self._residual_cfg.get("pi05_chunk_horizon", 10)),
            device=str(self.device),
        )

    def _openpi_port(self) -> int:
        base_port = int(self._residual_cfg.get("openpi_base_port", 8124))
        node_rank = int(getattr(self.worker_info, "cluster_node_rank", 0) or 0)
        gpus_per_node = int(self._residual_cfg.get("gpus_per_node", 8))
        return base_port + node_rank * gpus_per_node + self._worker_gpu_rank

    def _start_or_reuse_openpi_server(self) -> None:
        host = str(self._residual_cfg.get("openpi_host", "127.0.0.1"))
        port = self._openpi_port()
        if _listener_exists(host, port):
            print(f"[rlinf-residual] reusing openpi server {host}:{port}", flush=True)
            return

        ckpt = _resolve_path(str(self._residual_cfg["pi05_jax_ckpt_dir"]))
        if not ckpt.is_dir():
            raise RuntimeError(f"JAX ckpt dir missing: {ckpt}")
        if not (ckpt / "params").is_dir():
            raise RuntimeError(f"ckpt is not JAX/Orbax format (missing params/): {ckpt}")

        extra_env = {
            "CUDA_VISIBLE_DEVICES": str(self._worker_gpu_rank),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "XLA_PYTHON_CLIENT_MEM_FRACTION": str(
                self._residual_cfg.get("xla_mem_fraction", 0.25)
            ),
            "PYTHONUNBUFFERED": "1",
        }
        self._openpi_proc = spawn_openpi_server(
            ckpt=ckpt,
            config_name=str(self._residual_cfg["pi05_jax_config_name"]),
            prompt=self.task_description,
            port=port,
            extra_env=extra_env,
        )
        wait_for_port(
            host,
            port,
            float(self._residual_cfg.get("openpi_server_startup_s", 180.0)),
        )
        time.sleep(float(self._residual_cfg.get("openpi_handshake_grace_s", 3.0)))

    def _wrap_obs(self, obs):
        pi_obs = super()._wrap_obs(obs)
        if self._pi05 is None:
            return pi_obs

        base = self._pi05.base_action(pi_obs, reset_mask=self._force_refill_mask)
        self._force_refill_mask = None
        self._pending_base = base.detach().clone()

        pi_obs["base_actions"] = self._pending_base
        pi_obs["states"] = torch.cat(
            [pi_obs["states"].to(self.device), self._pending_base], dim=-1
        )
        return pi_obs

    def reset(self, seed=None, env_ids=None):
        if env_ids is None:
            self._force_refill_mask = torch.ones(
                self.num_envs, dtype=torch.bool, device=self.device
            )
            self._reset_residual_metrics()
        else:
            mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
            env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            mask[env_ids_t] = True
            self._force_refill_mask = mask
            self._reset_residual_metrics(env_ids_t)
        return super().reset(seed=seed, env_ids=env_ids)

    def _reset_residual_metrics(self, env_ids: torch.Tensor | None = None) -> None:
        if self._residual_l2_sum is None:
            self._residual_l2_sum = torch.zeros(self.num_envs, device=self.device)
            self._residual_l2_count = torch.zeros(self.num_envs, device=self.device)
            self._residual_l2_max = torch.zeros(self.num_envs, device=self.device)
        if env_ids is None:
            self._residual_l2_sum.zero_()
            self._residual_l2_count.zero_()
            self._residual_l2_max.zero_()
        else:
            self._residual_l2_sum[env_ids] = 0.0
            self._residual_l2_count[env_ids] = 0.0
            self._residual_l2_max[env_ids] = 0.0

    def _record_residual_metrics(self, residual: torch.Tensor, infos: dict) -> None:
        if self._residual_l2_sum is None:
            self._reset_residual_metrics()
        assert self._residual_l2_sum is not None
        assert self._residual_l2_count is not None
        assert self._residual_l2_max is not None

        l2 = torch.linalg.vector_norm(residual.detach(), dim=-1)
        self._residual_l2_sum += l2
        self._residual_l2_count += 1.0
        self._residual_l2_max = torch.maximum(self._residual_l2_max, l2)

        episode = infos.setdefault("episode", {})
        episode["residual_l2_mean"] = (
            self._residual_l2_sum / self._residual_l2_count.clamp_min(1.0)
        ).clone()
        episode["residual_l2_max"] = self._residual_l2_max.clone()

    def step(self, actions=None, auto_reset=True):
        if self._pending_base is None:
            raise RuntimeError("reset() must be called before residual step().")

        if actions is None:
            residual = torch.zeros_like(self._pending_base)
        else:
            residual = torch.as_tensor(
                actions, dtype=torch.float32, device=self._pending_base.device
            )
            if residual.ndim == 3 and residual.shape[1] == 1:
                residual = residual[:, 0]
            if residual.shape != self._pending_base.shape:
                raise ValueError(
                    f"residual shape {tuple(residual.shape)} does not match "
                    f"pending base {tuple(self._pending_base.shape)}"
                )

        clip = float(self._residual_cfg.get("residual_clip", 0.0))
        if clip > 0:
            residual = residual.clamp(-clip, clip)

        base = self._pending_base
        full_action = base + residual
        self._last_executed_base = base.detach().clone()
        self._last_executed_residual = residual.detach().clone()
        self._last_executed_full_action = full_action.detach().clone()

        obs, reward, terminations, truncations, infos = super().step(
            full_action, auto_reset=auto_reset
        )
        self._record_residual_metrics(residual, infos)
        infos["residual_action"] = self._last_executed_residual
        infos["base_action"] = self._last_executed_base
        infos["executed_action"] = self._last_executed_full_action
        return obs, reward, terminations, truncations, infos

    def close(self):
        try:
            super().close()
        finally:
            if self._openpi_proc is not None:
                kill_server(self._openpi_proc)
                self._openpi_proc = None


def register_env() -> None:
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    REGISTER_ISAACLAB_ENVS[RESIDUAL_TASK_ID] = IsaaclabPickOrangeResidualEnv
