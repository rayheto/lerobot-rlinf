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
import dataclasses
import logging
import pathlib

import openpi.models.model as _model
import openpi.transforms as _transforms
from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory
from typing_extensions import override

from rlinf.models.embodiment.openpi.policies import so101_lift_policy


def _patch_lerobot_revision_fallback() -> None:
    # Public LeRobot datasets (e.g. aswinkumar99/...) often lack the v3.0
    # codebase_version git tag that lerobot's get_safe_version() insists on,
    # which makes openpi's LeRobotDatasetMetadata(repo_id) blow up with
    # RevisionNotFoundError. Mirror the `--dataset.revision=main` escape
    # hatch used in the lerobot-train CLI path. Must patch both the source
    # module and lerobot_dataset's re-imported binding, since lerobot_dataset
    # does `from .utils import get_safe_version` at module load.
    from lerobot.common.datasets import utils as _ds_utils
    from lerobot.common.datasets import lerobot_dataset as _ds_module
    from huggingface_hub.errors import RevisionNotFoundError

    if getattr(_ds_utils.get_safe_version, "_rlinf_patched", False):
        return

    _original = _ds_utils.get_safe_version

    def _safe_version_with_main_fallback(repo_id, version):
        try:
            return _original(repo_id, version)
        except RevisionNotFoundError:
            logging.getLogger(__name__).warning(
                "get_safe_version(%s, %s) found no codebase tag; falling back to 'main'",
                repo_id,
                version,
            )
            return "main"

    _safe_version_with_main_fallback._rlinf_patched = True
    _ds_utils.get_safe_version = _safe_version_with_main_fallback
    _ds_module.get_safe_version = _safe_version_with_main_fallback


def _patch_openpi_prompt_from_lerobot_task() -> None:
    # lerobot >=0.4 stores meta.tasks as a pandas DataFrame indexed by task
    # string (column = task_index). openpi 0.1.0 expects the old dict[int,str]
    # and calls .get(int) on it, which DataFrames don't support → ValueError.
    # Convert at __post_init__ so the dict (not the DataFrame) is what gets
    # pickled to DataLoader workers — workers use spawn and won't re-run our
    # patches. The conversion must happen in the main process at the moment
    # PromptFromLeRobotTask is constructed.
    import openpi.transforms as _t

    if getattr(_t.PromptFromLeRobotTask, "_rlinf_patched", False):
        return

    _original_init = _t.PromptFromLeRobotTask.__init__

    def _patched_init(self, tasks):
        if hasattr(tasks, "iterrows") and hasattr(tasks, "index"):
            tasks = {int(row.task_index): str(idx) for idx, row in tasks.iterrows()}
        _original_init(self, tasks)

    _t.PromptFromLeRobotTask.__init__ = _patched_init
    _t.PromptFromLeRobotTask._rlinf_patched = True


_patch_lerobot_revision_fallback()
_patch_openpi_prompt_from_lerobot_task()


@dataclasses.dataclass(frozen=True)
class LeRobotIsaacLabSo101LiftDataConfig(DataConfigFactory):
    """OpenPI data config for aswinkumar99 SO-101 sponge-bowl dataset."""

    default_prompt: str | None = "Pick the blue sponge and place it in the bowl"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.overhead",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                so101_lift_policy.So101LiftInputs(model_type=model_config.model_type)
            ],
            outputs=[so101_lift_policy.So101LiftOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )


@dataclasses.dataclass(frozen=True)
class LeRobotIsaacLabSo101PickOrangeDataConfig(DataConfigFactory):
    """OpenPI data config for LightwheelAI leisaac-pick-orange SO-101 dataset.

    Same architecture as the sponge variant; only the lerobot dataset camera
    key is `front` (not `overhead`) and the prompt is the pick-orange task.
    """

    default_prompt: str | None = "Grab orange and place into plate"

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> DataConfig:
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation/image": "observation.images.front",
                        "observation/wrist_image": "observation.images.wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )

        data_transforms = _transforms.Group(
            inputs=[
                so101_lift_policy.So101LiftInputs(model_type=model_config.model_type)
            ],
            outputs=[so101_lift_policy.So101LiftOutputs()],
        )

        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(
            model_config
        )

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=("action",),
        )
