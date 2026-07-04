# Setup notes & gotchas

Chinese version: [notes.zh.md](notes.zh.md)

Issues we hit while bringing SO-101 up on Isaac Sim 5.1 + Isaac Lab. Capturing them
here so we don't pay the cost twice.

## SO-101 specifics

- **Zero pose IS the standing pose.** Counter-intuitive vs Franka (whose "home"
  uses non-zero joint values). Verified via `inspect_so101.py --pose` →
  `0 0 0 0 0 0` puts the arm in a sensible ready posture. Setting non-zero
  `init_state.joint_pos` (e.g. `shoulder_lift=-0.6, elbow_flex=1.0` copied from
  Franka intuition) flattens it.
- **Joint order**: `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex,
  wrist_roll, gripper` (6 DoF). EE body for command/reward = `gripper_frame_link`.
- **Joint limits** (from URDF parse, in joint order above):
  lower `[-1.920, -1.745, -1.690, -1.658, -2.744, -0.175]`,
  upper `[+1.920, +1.745, +1.690, +1.658, +2.841, +1.745]`.
- **Workspace radius ~20 cm**, vs Franka's ~80 cm. Tighten command/spawn ranges
  for any task ported from Franka or the cube/target ends up unreachable.

## PD tuning

- Franka's actuator gains (`stiffness=80, damping=4`) are **way too soft** for
  SO-101 — the arm sags under its own weight even at the zero pose.
- URDF importer default (`stiffness=1000, damping=50`) holds it fine. That's
  what we now use in `SO101_CFG.actuators`.
- **Why `inspect_so101.py` didn't reveal this**: it uses
  `robot.set_joint_positions()`, which teleports joints (writes both position
  and target), bypassing the PD loop. Lift env goes through
  `JointPositionActionCfg` → PD controller → exposes weak gains. Always test
  PD with an actual policy/zero-action loop, not teleport.

## Isaac Sim 5.1 / Isaac Lab gotchas

- **API renames from 4.x**: `omni.isaac.core` → `isaacsim.core.api`;
  `Articulation` class moved to `isaacsim.core.prims.SingleArticulation`;
  no `get_joint_limits()` on SingleArticulation (limits live in PhysX dof
  properties view, or just hardcode from URDF parse output).
- **`pxr` is unavailable until `SimulationApp` boots.** Any pure-Python
  `import isaaclab` test outside SimulationApp will fail at `from pxr import Usd`.
  Run import tests inside an `AppLauncher`/`SimulationApp` context.
- **`gym.make("Isaac-...")` takes a cfg dataclass, not a dict.** Use
  `isaaclab_tasks.utils.parse_env_cfg(task, device=..., num_envs=...)` first.
- **Argparse + SimulationApp argv collision.** SimulationApp forwards leftover
  `sys.argv` to Kit, which rejects unknown flags. Strip our flags before init:
  `sys.argv = sys.argv[:1]`.
- **Kit may swallow late stdout.** Use `python -u` + `flush=True` on prints
  (`functools.partial(print, flush=True)`) or you'll lose log lines and think
  the script silently exited.
- **`omni.ui.Window` visibility is flaky** under headless display tunneling.
  `position_x/y`, `dockPreference`, `window.visible=True` — none reliably
  surface the window on `DISPLAY=:110`. Use a stdin REPL on a worker thread
  (see `inspect_so101.py --pose`) instead of struggling with omni.ui.

## Install fiddles

`.venv-isaacsim` (uv-managed, `uv venv --python 3.11 .venv-isaacsim`) replaced the old
`rlinf-isaacsim-env` conda env — conda is not used anywhere in this repo anymore. Isaac
Sim 5.1 is plain pip-installable (`isaacsim[all,extscache]==5.1.0` from
`pypi.nvidia.com`), no binary installer or conda required. See README's "One-time
setup" for the full `uv pip install` sequence. Fiddles hit along the way:

- **uv's default index strategy breaks the multi-index resolve.** `torch`
  (download.pytorch.org) + `isaacsim` (pypi.nvidia.com) + PyPI all need to be
  searched for compatible versions of shared deps like `idna`; uv's default only
  looks at the *first* index that has any version of a package. Use
  `--index-strategy unsafe-best-match`.
- **`flatdict==4.0.1` (an `isaaclab` dep) fails to build**: `ModuleNotFoundError: No
  module named 'pkg_resources'`. Recent `setuptools` (81+) dropped `pkg_resources`
  entirely. Fix: `uv pip install "setuptools<81"` into the venv, then install
  `isaaclab` with `--no-build-isolation` so the build sees that setuptools.
- `pip install -e source/isaaclab*` (all 5 packages) bumps `psutil` to >=7 and
  breaks `isaacsim-kernel==5.1.0` which pins `psutil==5.9.8`; also bumps `click` past
  the `8.1.7` isaacsim-kernel wants.
  **Re-pin after**: `uv pip install "psutil==5.9.8" "click==8.1.7"`. (ipython /
  huggingface-hub / typer will complain but still work — same category as the
  fastapi/starlette mismatch below.)
- Installing `leisaac` with `--no-deps` (required — its `[isaaclab]` extra would
  re-pull isaacsim/isaaclab and risk a numpy>=2 bump) also skips its *real* direct
  deps. Install them explicitly: `deepdiff feetech-servo-sdk "pygame>=2.5.1,<2.7.0"
  pyserial`.
- **`eval_run.py` needs `msgpack`, `pydantic`, and `pyarrow`** — none of them come in
  transitively from the isaaclab/leisaac installs above. Without them, every eval
  shard dies instantly (client crashes on import right after connecting, server logs
  look like generic WebSocket handshake failures — `EOFError: stream ends after 0
  bytes` — which is misleading; the real error is the client-side
  `ModuleNotFoundError`/`ImportError` for these packages, only visible in the
  client's own stdout).
- **The `kitchen_with_orange` scene + `so101_follower.usd` robot are not in the
  `leisaac` submodule** (`third_party/leisaac/assets/{robots,scenes}/` ship with only
  `.gitkeep` placeholders). Download from `LightwheelAI/leisaac_env` on HuggingFace:
  ```python
  from huggingface_hub import snapshot_download
  snapshot_download(repo_id="LightwheelAI/leisaac_env",
      allow_patterns=["assets/robots/so101_follower.usd",
                       "assets/scenes/kitchen_with_orange/**"],
      local_dir="third_party/leisaac")
  ```
  (Public repo, no token needed.) Symptom without this: `pxr.Tf.ErrorException:
  ... Failed to open layer @.../scenes/kitchen_with_orange/scene.usd@`.
- `fastapi 0.115` wants `starlette<0.46`, isaaclab pulls 0.49 — ignore, fastapi
  isn't used by the training/rollout path.
- IsaacLab is editable-installed from `third_party/leisaac/dependencies/IsaacLab/`
  (bundled in the `leisaac` submodule). Re-running the `uv pip install -e` loop over
  `source/isaaclab*` after `git pull` is enough; no need for `isaaclab.sh --install`
  (its conda-activation-hook and `apt-get` side effects never come into play since we
  call `uv pip install -e` directly on each package dir).
- **Don't install `lerobot` (or `accelerate`) into `.venv-isaacsim`.**
  Both transitively pull numpy>=2, which breaks Isaac Sim 5.1 hard:
  hundreds of `isaacsim.*` extensions fail to load with
  `AttributeError: module 'numpy' has no attribute '_no_nep50_warning'`.
  `isaacsim-kernel==5.1.0.0` pins `numpy==1.26.0`. If you need dataset
  access from inside the Isaac env (e.g. for the replay sanity script),
  read the parquet shards directly via pyarrow from `HF_LEROBOT_HOME`;
  don't import `LeRobotDataset`. Recovery if already broken:
  `uv pip install "numpy==1.26.0"` — Isaac extensions reload on next run.
- **Running multiple `eval.py` processes in parallel (`src/eval.py --watch`
  per-checkpoint fan-out) needs distinct `--port` per shard** — and on a shared
  machine, don't assume the default `8000`/`8001`/... range is free. Watch mode uses
  `--base-port` and checks the shard ports before launch; override it if a range is
  busy.

## Cosmetic warnings (safe to ignore)

- `Unresolved reference prim path .../visuals/gripper_frame_link` — URDF
  importer doesn't write a visual sublayer for the fixed end-effector frame
  (it has no mesh). Physics fine, just no mesh at that prim.
- `getAttributeCount / getTypes called on non-existent path
  .../wrist_link/visuals/wrist_roll_pitch_so101_v2/node_STL_BINARY_` — same
  family, leftover STL conversion path.
- `omni.isaac.dynamic_control` / `omni.isaac.wheeled_robots` deprecation
  notices — third-party extensions in IsaacLab, not ours.
- `EULA prompt` on first Isaac Sim 5.1 import: `echo "Yes" | python ...` once
  to accept (note: `DISPLAY=:110 echo Yes | python` is wrong — env var binds
  to echo. Use `echo Yes | DISPLAY=:110 python ...`).

## LeRobot Pi 0.5 interface contract

Pulled from `huggingface/lerobot` main and the `lerobot/pi05_base` HF model
card on 2026-05-30. Capturing here so the next person doesn't re-derive
this when wiring W2.

- **Obs key constants** (`src/lerobot/utils/constants.py`):
  `OBS_STATE = "observation.state"`, `OBS_IMAGES = "observation.images"`
  (prefix; full key is `observation.images.<camera_name>`),
  `ACTION = "action"`.
- **Camera names are dataset-defined, not policy-pinned.** `pi05_base`
  ships with `observation.images.{base_0_rgb, left_wrist_0_rgb,
  right_wrist_0_rgb}` as example keys; an SO-101 finetune sets its own
  (typically `front` + `wrist`). Whatever the finetune dataset uses is
  what the policy expects.
- **Image format** (`modeling_pi0.py`): accepts `[B,C,H,W]` OR `[B,H,W,C]`
  — no need to permute. Values must be float in `[0,1]`. Policy resizes
  to `(224, 224)` internally and renormalizes to `[-1,1]` for SigLIP.
- **Action output**: chunk `[B, chunk_size=50, max_action_dim=32]` (padded
  to 32; real action_dim from finetune = 6 for SO-101). Pi 0.5 uses
  `NormalizationMode.MEAN_STD` for both STATE and ACTION; `select_action()`
  unnormalizes outputs back to **dataset units** before returning.
- **Action units are NOT radians.** They are whatever the SO-101 dataset
  captures, which for LeRobot Feetech-calibrated teleop is normalized
  degree-like motor values. Conversion to env-side joint targets must
  read the dataset's action stats — don't assume radians.

**Implication for the env**: do not pre-bake any of this into env cfg.
Bridging (key rename, dtype/scale, dataset→env action units) belongs in
a runtime wrapper, written when the concrete finetune is in hand.

## SFT smoke debugging (Pi 0.5 + lerobot 0.4.4)

Lessons from getting `scripts/sft_smoke.sh` to pass in the
`rlinf-lerobot-train` env. Save the next person hours.

- **`lerobot[pi]` extra is broken in 0.4.4.** Pip warns the extra doesn't
  exist; base install succeeds but is missing `transformers`. Install it
  yourself — and not vanilla.
- **Pi 0.5 needs a patched transformers fork.** Vanilla `transformers`
  lacks `transformers.models.siglip.check` which `modeling_pi05.py`
  imports at load. Install the lerobot-openpi branch:
  ```
  pip install --force-reinstall --no-deps \
    "transformers @ git+https://github.com/huggingface/transformers.git@fix/lerobot_openpi"
  ```
  Resolves to 4.53.3. `--no-deps` to keep hf-hub<0.36 (lerobot pin).
  Verify: `from transformers.models.siglip import check;
  check.check_whether_transformers_replace_is_installed_correctly()` → True.
- **First `pip install ... git+...` may silently no-op.** Pip sees the
  cached version and skips. `--force-reinstall` is mandatory the first
  time, not optional.
- **Dataset revision pin.** Public LeRobot datasets often lack the v3.0
  git tag the code checks for → `RevisionNotFoundError`. Pass
  `--dataset.revision=main` to bypass.
- **PaliGemma tokenizer is gated per HF account.** Pi 0.5 pulls
  `google/paligemma-3b-pt-224`; token alone isn't enough — accept the
  license at https://huggingface.co/google/paligemma-3b-pt-224 with the
  account whose token you're using. Verify with
  `huggingface_hub.hf_hub_download('google/paligemma-3b-pt-224', 'config.json')`.
- **`--policy.repo_id` is required even when not pushing.** lerobot
  validates it before the `push_to_hub` check. Pass a dummy:
  `--policy.push_to_hub=false --policy.repo_id=local/<run_name>`.
- **Camera-key bridge to pi05_base.** `pi05_base` declares 3 cameras
  (`base_0_rgb`, `left_wrist_0_rgb`, `right_wrist_0_rgb`); the
  aswinkumar99 SO-101 dataset has 2 (`overhead`, `wrist`). Bridge via
  CLI, no code change:
  ```
  --policy.empty_cameras=1
  --rename_map='{"observation.images.overhead":"observation.images.base_0_rgb",
                 "observation.images.wrist":"observation.images.right_wrist_0_rgb"}'
  ```
- **24GB VRAM recipe for 4B Pi 0.5.** All four together fit comfortably:
  `--policy.train_expert_only=true` (freezes the 2B VLM, trains only the
  300M action expert + projections), `--policy.freeze_vision_encoder=true`,
  `--policy.dtype=bfloat16`, `--policy.gradient_checkpointing=true`.
  Verified: 4B total / 693M trainable, ~1.7 steps/s at batch=1 on a 4090D.
- **Benign warning to ignore.** `Missing key(s) in state_dict:
  ...language_model.embed_tokens.weight` prints on load. Training
  proceeds normally — it's a weight-remap artifact in the Pi 0.5 load path.

## Simulation Settings panel (Isaac Sim UI)

For reference — what the docked panel in the viewport actually controls:

- **USD / Fabric CPU / Fabric GPU**: physics state backend. Fabric GPU is
  default and required for training perf. USD goes back to per-frame USD
  read/write — only useful when debugging USD prim state.
- **Reset Simulation on Stop**: whether `Stop` reverts to t=0. Irrelevant to
  RL training (we reset via env API), keep it on while authoring scenes.
