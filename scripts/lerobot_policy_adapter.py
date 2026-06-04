"""Wrap lerobot PI05Policy to expose RLinf's `predict_action_batch` interface.

Drop-in replacement for `rlinf.models.embodiment.openpi.get_model() → model`.
Usage in eval::

    from lerobot_policy_adapter import LerobotPolicyAdapter
    model = LerobotPolicyAdapter(ckpt_path, device="cuda:0")
    actions, _ = model.predict_action_batch(env_obs, mode="eval")

``env_obs`` is the RLinf-format dict from ``wrap_obs()`` with keys
``main_images`` (uint8 BHWC), ``wrist_images``, ``states`` (motor-deg),
``task_descriptions``, ``extra_view_images``.
"""

from __future__ import annotations

import torch
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors


class LerobotPolicyAdapter:
    """Wraps lerobot PI05Policy and its preprocessor chain, exposes
    RLinf-compatible ``predict_action_batch(env_obs, mode=..., compute_values=...)``.
    """

    def __init__(self, pretrained_path: str, device: str = "cuda:0"):
        self.device = device
        self.policy = PI05Policy.from_pretrained(pretrained_path).to(device).eval()
        self.preprocessor, self.postprocessor = make_pre_post_processors(
            self.policy.config, pretrained_path=pretrained_path
        )
        self._action_chunk = self.policy.config.chunk_size  # e.g. 50

    def predict_action_batch(self, env_obs: dict, mode: str = "eval",
                             compute_values: bool = False, **kwargs):
        """RLinf-compatible action prediction.

        Returns (actions, result_dict). ``actions`` is [B, chunk * 6] motor-deg,
        matching the original openpi output shape before env reshaping.
        """
        batch = self._env_obs_to_batch(env_obs)
        obs_processed = self.preprocessor(batch)

        # Standard lerobot inference → [B, chunk_size, action_dim] in NORMALIZED space
        action_chunk = self.policy.predict_action_chunk(obs_processed)
        action_chunk = action_chunk[:, :, :6]  # [B, chunk, 6] normalized

        # Unnormalize via lerobot postprocessor (quantile → motor-deg)
        b, chunk, dim = action_chunk.shape
        action_flat = action_chunk.reshape(b * chunk, dim)
        action_motor_deg = self.postprocessor(action_flat)  # [B*chunk, 6] on CPU
        if isinstance(action_motor_deg, torch.Tensor):
            actions_flat = action_motor_deg.reshape(b, chunk * dim).to(self.device)
        else:
            import numpy as np
            actions_flat = torch.from_numpy(
                np.asarray(action_motor_deg).reshape(b, chunk * dim)
            ).float().to(self.device)

        result = {
            "prev_logprobs": None,
            "prev_values": None,
            "forward_inputs": None,
        }
        return actions_flat.contiguous(), result

    # ------------------------------------------------------------------
    def _env_obs_to_batch(self, env_obs: dict) -> dict:
        """Convert RLinf env_obs → lerobot-format flat batch dict.

        Image keys use dataset-original names (front, wrist) so the
        rename_observations_processor inside the preprocessor maps them to
        ``base_0_rgb`` / ``right_wrist_0_rgb``.  Missing ``left_wrist_0_rgb``
        and ``empty_camera_0`` are auto-padded by ``_preprocess_images``.
        """
        num_envs = len(env_obs["task_descriptions"])

        def _to_float_chw(img_u8: torch.Tensor) -> torch.Tensor:
            if img_u8.dtype != torch.uint8:
                img_u8 = img_u8.to(torch.uint8)
            return (img_u8.float().div_(255.0).permute(0, 3, 1, 2).contiguous())

        front = _to_float_chw(env_obs["main_images"])    # [B,3,H,W] float [0,1]
        wrist = _to_float_chw(env_obs["wrist_images"])   # [B,3,H,W] float [0,1]
        state = env_obs["states"]                         # [B,6] motor-deg

        obs_batch = {
            "observation.images.front": front,
            "observation.images.wrist": wrist,
            "observation.state": state,
            "task": env_obs["task_descriptions"],
        }
        # Add batch dimension for the preprocessor (it expects [1, ...] per obs)
        # but our build_obs already has batch dim → handled by preprocessor's
        # AddBatchDimension step which is a no-op when batch dim exists.
        return obs_batch
