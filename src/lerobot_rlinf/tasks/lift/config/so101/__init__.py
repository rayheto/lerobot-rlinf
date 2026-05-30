"""Gym registration for the SO-101 lift-cube env."""
import gymnasium as gym

gym.register(
    id="Isaac-Lift-Cube-SO101-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SO101CubeLiftEnvCfg",
    },
    disable_env_checker=True,
)

gym.register(
    id="Isaac-Lift-Cube-SO101-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    kwargs={
        "env_cfg_entry_point": f"{__name__}.joint_pos_env_cfg:SO101CubeLiftEnvCfg_PLAY",
    },
    disable_env_checker=True,
)
