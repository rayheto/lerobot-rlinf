"""RLinf PPO entrypoint for frozen pi05 + SO-101 residual MLP.

Usage:
    python -m src.rl.rlinf_residual.train --config-name pick_orange_residual_ppo
"""
from __future__ import annotations

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from src.rl.rlinf_residual.ext import register

mp.set_start_method("spawn", force=True)


def _setup_env_vars() -> None:
    import os
    import pathlib

    repo_root = pathlib.Path(__file__).resolve().parents[3]
    os.environ.setdefault(
        "LEISAAC_ASSETS_ROOT", str(repo_root / "third_party" / "leisaac" / "assets")
    )
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    os.environ.setdefault("RLINF_OPENPI_SO101_PATCH", "1")
    os.environ.setdefault("RLINF_EXT_MODULE", "src.rl.rlinf_residual.ext")
    os.environ.setdefault(
        "EMBODIED_PATH",
        str(repo_root / "third_party" / "RLinf" / "examples" / "embodiment"),
    )


def _register_project_extensions() -> None:
    register()


def _drop_null_component_placements(cfg) -> None:
    placements = cfg.get("cluster", {}).get("component_placement")
    if placements is None:
        return
    with open_dict(placements):
        for key, value in list(placements.items()):
            if value is None:
                del placements[key]


@hydra.main(version_base="1.1", config_path="config", config_name="pick_orange_residual_ppo")
def main(cfg) -> None:
    _setup_env_vars()
    _register_project_extensions()
    _drop_null_component_placements(cfg)

    from rlinf.config import validate_cfg
    from rlinf.runners.embodied_runner import EmbodiedRunner
    from rlinf.scheduler import Cluster
    from rlinf.utils.placement import HybridComponentPlacement
    from rlinf.workers.actor.fsdp_actor_worker import EmbodiedFSDPActor
    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    cfg = validate_cfg(cfg)
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(
        cluster_cfg=cfg.cluster,
        distributed_log_dir=cfg.runner.per_worker_log_path,
    )
    component_placement = HybridComponentPlacement(cfg, cluster)

    actor_group = EmbodiedFSDPActor.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=component_placement.get_strategy("actor"),
    )
    rollout_group = MultiStepRolloutWorker.create_group(cfg).launch(
        cluster,
        name=cfg.rollout.group_name,
        placement_strategy=component_placement.get_strategy("rollout"),
    )
    env_group = EnvWorker.create_group(cfg).launch(
        cluster,
        name=cfg.env.group_name,
        placement_strategy=component_placement.get_strategy("env"),
    )

    runner = EmbodiedRunner(
        cfg=cfg,
        actor=actor_group,
        rollout=rollout_group,
        env=env_group,
        reward=None,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
