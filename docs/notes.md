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

- `pip install -e source/isaaclab*` (all 5 packages) bumps `psutil` to >=7 and
  breaks `isaacsim-kernel==5.1.0` which pins `psutil==5.9.8`.
  **Re-pin after**: `pip install "psutil==5.9.8"`. (ipython will complain but works.)
- `fastapi 0.115` wants `starlette<0.46`, isaaclab pulls 0.49 — ignore, fastapi
  isn't used by the training/rollout path.
- IsaacLab is editable-installed from `third_party/IsaacLab/` (gitignored).
  Re-running `pip install -e source/*` after `git pull` is enough; no need to
  rerun `isaaclab.sh -i`.

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

## Simulation Settings panel (Isaac Sim UI)

For reference — what the docked panel in the viewport actually controls:

- **USD / Fabric CPU / Fabric GPU**: physics state backend. Fabric GPU is
  default and required for training perf. USD goes back to per-frame USD
  read/write — only useful when debugging USD prim state.
- **Reset Simulation on Stop**: whether `Stop` reverts to t=0. Irrelevant to
  RL training (we reset via env API), keep it on while authoring scenes.
