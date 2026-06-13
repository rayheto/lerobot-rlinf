"""Single-source-of-truth config for the simple PPO framework.

Plain dataclass; no hydra. Override with argparse in train.py.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Config:
    # --------------------------- paths ---------------------------
    # Step 8: switched to JAX server for inference (matches SFT eval + dryrun_pi05_jax).
    # PyTorch 24999_pt path kept registered above for back-compat but unused at rollout.
    pi05_ckpt_dir: str = (
        "outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999_pt"
    )
    pi05_norm_stats_path: str = (
        "outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/"
        "assets/EverNorif/leisaac-pick-orange/norm_stats.json"
    )
    pi05_config_name: str = "pi05_isaaclab_so101_pick_orange"

    # JAX server config — drives rollout pi05 inference.
    pi05_jax_ckpt_dir: str = (
        "outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999"
    )
    pi05_jax_config_name: str = "pi05_lora_so101_pick_orange"  # SFT training config
    pi05_server_host: str = "127.0.0.1"
    pi05_server_port: int = 8124
    pi05_server_startup_s: float = 180.0
    pi05_chunk_horizon: int = 10  # action_horizon — server returns (10, 6) chunks
    demo_dataset_path: str = (
        "/home/hlei/.cache/huggingface/lerobot/EverNorif/leisaac-pick-orange"
    )
    log_dir: str = "logs/simple_ppo"

    # --------------------------- env -----------------------------
    env_task_id: str = "LeIsaac-SO101-PickOrange-v0"
    env_prompt: str = "Pick the orange and place it on the plate."
    env_num_envs: int = 4        # Step 8: parallel envs (smoke confirmed 4 fits)
    env_max_episode_steps: int = 900
    # Step 9: env_decimation 1→2 (30Hz native, matches SFT 30 fps).
    # 900 steps × decimation=2 / sim_fps=60 = 30s sim time, matches episode_length_s.
    env_episode_length_s: float = 30.0
    env_decimation: int = 2
    env_seed: int = 0
    env_wrist_cam_h: int = 224
    env_wrist_cam_w: int = 224
    env_front_cam_h: int = 224
    env_front_cam_w: int = 224
    # env-side sparse reward (Step 8: 3-orange task w/ rest bonus + fail-A)
    env_grasp_bonus: float = 10.0
    env_carry_speed_coef: float = 0.5
    env_place_bonus: float = 20.0
    env_drop_penalty: float = -5.0
    env_timeout_penalty: float = -2.0
    env_rest_bonus: float = 30.0   # all-3-placed + arm at rest pose
    env_fail_penalty: float = -5.0 # fail-A: orange off-table / out of xy box

    # --------------------------- PPO -----------------------------
    rollout_len: int = 900       # Step 8: full episode; success/fail terminate early
    total_iters: int = 50        # Step 8: 50 iter * 900 * 4env = 180k env steps
    update_epochs: int = 4
    minibatch_size: int = 128    # Step 8: was 16; rollout_len*num_envs=3600 supports it
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_clip: float = 0.2
    value_coef: float = 0.5
    ent_coef: float = 0.003
    max_grad_norm: float = 1.0
    target_kl: float = 0.05
    lr: float = 1.0e-4  # Step 9 F3: 3e-4→1e-4; after F1 (σ=0.37) iter-0 KL was still 1.21, lr was over-aggressive
    adam_eps: float = 1.0e-5
    weight_decay: float = 0.0

    # ----------------------- reward shaping ----------------------
    # Step 8e: OOD disabled — SFT already inside demo manifold; previous
    # ood_coef=0.01 produced -45~-75/ep, dominating grasp_bonus(+10) and
    # killing exploration.
    ood_coef: float = 0.0
    ood_k_neighbors: int = 5
    # Step 8e: relaxed from -0.005 to -0.001 (~-0.9/ep at 900 steps).
    # Long-term time-pressure term; was overwhelming sparse positive signal.
    survival_cost: float = -0.001

    # Dense shaping (Plan B) — see docs/dryrun_jax_24999_diagnostic.md.
    # SFT policy gets ee within 7 cm of orange but never inside the 5 cm grasp
    # gate. These two terms give RL a continuous gradient toward grasp.
    # dense_eo:   -dense_eo_coef * max(0, d_ee_orange - dense_eo_floor)
    # dense_lift: +dense_lift_coef * max(0, lift_dz)
    # Step 9: dense_eo DISABLED (was 1.0). Independent dense_eo term is
    # double-counting on top of grasp_bonus, and at SFT baseline d_eo≈0.136 m
    # it injects a state-correlated ~-77/ep stationary bias. Combined with
    # zero-init V head, iter-0 advantages are all-negative huge → PPO trust
    # region breaks (approx_kl=4.47, grad_norm=325 in step8f).
    # The correct fix is potential-based shaping with Φ(s)=-d_eo (γΦ(s')-Φ(s)
    # has expectation 0 → no bias) — deferred to Step 10. For now, just shut
    # off the term. See docs/dryrun_jax_step8f_crossanalysis.md §6.
    # dense_lift kept: only fires when orange001 is physically lifted (z>z0),
    # baseline lift=0 → no stationary bias, safe.
    dense_eo_coef: float = 0.0
    dense_eo_floor: float = 0.05
    dense_lift_coef: float = 2.0

    # -------------------------- BC anchor ------------------------
    # Step 9: DISABLED for diagnostic. BC anchor at coef=0.1 was a confounder
    # in iter-0 KL blowup: with σ init=0.135, even small μ shifts produce
    # large logp changes for demo actions → BC gradient amplifies. Re-enable
    # (start=0.05?) only after split-grad-clip + zero-init value head are
    # validated to keep iter-0 KL < target.
    bc_coef_start: float = 0.0
    bc_coef_end: float = 0.0
    bc_coef_warmup_iters: int = 80  # ~5k env steps at rollout_len=64
    bc_batch_size: int = 64
    bc_action_horizon: int = 1  # we only train residual on chunk[0]

    # ---------------------- policy / head ------------------------
    head_hidden: int = 256
    head_init_log_std: float = -1.0  # Step 9 F1: σ≈0.37 (was -2.0/σ=0.135); 1/σ² gradient amplification was ~55×→~7.4×, fixed iter-0 KL blowup
    log_std_min: float = -2.5  # Step 5→6: was -5.0, hard floor to prevent σ collapse
    log_std_max: float = 2.0
    # NOTE: hard-clipping the residual breaks PPO logp consistency between
    # rollout (logp before clip) and evaluate (logp of action - base, i.e. the
    # clipped residual). Keep <=0 to disable; rely on log_sigma_max + small
    # init std + grad clip to bound behavior. Set to a small positive value
    # only as an emergency physical safety cap when debugging an env-side
    # blowup.
    residual_clip: float = 0.0  # disabled
    deterministic_pi05: bool = True  # zero-noise sampling for reproducible base

    # ---------------------------- misc ---------------------------
    ckpt_every_iters: int = 50
    seed: int = 1234
    device: str = "cuda"

    # -------------------------- video --------------------------
    video_every_iters: int = 50  # record one rollout's main+wrist video every K iters; 0 disables
    video_fps: int = 30

    # ------------------------- env-cfg helper --------------------
    def build_env_cfg(self):
        """Build the OmegaConf-style DictConfig expected by IsaaclabPickOrangeEnv.

        Mirrors src/rl/config/pick_orange_ppo.yaml `_env_common` minus
        rlinf-specific keys. Returns an OmegaConf DictConfig.
        """
        from omegaconf import OmegaConf

        return OmegaConf.create(
            {
                "env_type": "isaaclab",
                "auto_reset": True,
                "ignore_terminations": False,
                "use_rel_reward": False,
                "seed": self.env_seed,
                "group_size": 1,
                "reward_coef": 1.0,
                "use_fixed_reset_state_ids": False,
                "max_steps_per_rollout_epoch": self.rollout_len,
                "max_episode_steps": self.env_max_episode_steps,
                "reward": {
                    "grasp_bonus": self.env_grasp_bonus,
                    "carry_speed_coef": self.env_carry_speed_coef,
                    "place_bonus": self.env_place_bonus,
                    "drop_penalty": self.env_drop_penalty,
                    "timeout_penalty": self.env_timeout_penalty,
                    "rest_bonus": self.env_rest_bonus,
                    "fail_penalty": self.env_fail_penalty,
                    "grasp_diff_threshold": 0.08,
                    "grasp_close_threshold": 0.60,
                    "grasp_lift_threshold": 0.06,
                },
                "init_params": {
                    "id": self.env_task_id,
                    "num_envs": self.env_num_envs,
                    "max_episode_steps": self.env_max_episode_steps,
                    "episode_length_s": self.env_episode_length_s,
                    "decimation": self.env_decimation,
                    "task_description": self.env_prompt,
                    "wrist_cam": {
                        "height": self.env_wrist_cam_h,
                        "width": self.env_wrist_cam_w,
                    },
                    "front_cam": {
                        "height": self.env_front_cam_h,
                        "width": self.env_front_cam_w,
                    },
                },
                "video_cfg": {
                    "save_video": False,
                    "info_on_video": False,
                    "fps": 30,
                    "video_base_dir": f"{self.log_dir}/video",
                },
            }
        )
