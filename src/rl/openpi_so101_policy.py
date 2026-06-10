# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""OpenPI input/output transforms for SO-101 sponge-bowl lift.

State and action are both 6-dim joint pos in degrees (URDF order), unlike
stack_cube which uses 7-dim EEF + binary gripper. We mirror the lerobot
SFT camera rename: overhead → base_0_rgb, wrist → right_wrist_0_rgb,
left_wrist_0_rgb padded with zeros (matches ``--policy.empty_cameras=1``).
"""
import dataclasses

import einops
import numpy as np
from openpi import transforms
from openpi.models import model as _model


def make_so101_lift_example() -> dict:
    return {
        "observation/state": np.random.rand(6).astype(np.float32),
        "observation/image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "observation/wrist_image": np.random.randint(
            256, size=(224, 224, 3), dtype=np.uint8
        ),
        "prompt": "Pick the blue sponge and place it in the bowl",
    }


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class So101LiftInputs(transforms.DataTransformFn):
    """SO-101 obs → OpenPI inputs.

    overhead → base_0_rgb (matches pi05_base camera 0 slot).
    wrist    → right_wrist_0_rgb (matches the SFT rename map).
    left_wrist_0_rgb is zero-padded with mask=False (empty_cameras=1).
    """

    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        base_image = _parse_image(data["observation/image"])
        wrist_image = _parse_image(data["observation/wrist_image"])

        # Slot layout MUST match the lerobot SFT training pipeline.
        # lerobot `_preprocess_images` puts present cameras first, then pads
        # missing ones — for SO-101 (front + wrist) that's
        #   slot0=front, slot1=wrist, slot2=pad, slot3=pad
        # PaliGemma's visual prefix is position-encoded, so putting wrist at
        # slot2 (the openpi-native default) makes the action expert read
        # wrist tokens at a position it was never trained on.
        inputs = {
            "state": data["observation/state"],
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": wrist_image,
                "right_wrist_0_rgb": np.zeros_like(base_image),
                "empty_camera_0": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.False_,
                "empty_camera_0": np.False_,
            },
        }

        if "actions" in data:
            inputs["actions"] = data["actions"]
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class So101LiftOutputs(transforms.DataTransformFn):
    """OpenPI outputs → SO-101 action (6-dim joint pos in degrees).

    No sign() on the gripper — SO-101 gripper is a continuous joint
    position, not a binary command.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :6])}
