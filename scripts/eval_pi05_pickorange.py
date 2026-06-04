"""Standalone eval driver: SFT'd Pi 0.5 (RLinf openpi) → LeIsaac SO-101
pick_orange env. Single process, no Ray.

Ported from `eval_pi05_liftcube.py`. Differences:
  * env ID  → LeIsaac-SO101-PickOrange-v0
  * obs     → uses real `wrist` cam (lift_cube had only `front`)
  * success → reuse env's built-in `success` DoneTerm (`mdp.task_done`,
              all 3 oranges on plate); `term[i]==True && step<max` → success
  * prompt  → matches leisaac PickOrangeEnvCfg.task_description

Run (headless):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/eval_pi05_pickorange.py
Run (GUI on :110):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/eval_pi05_pickorange.py --gui

Note: default `--model-path` still points at the sponge-trained ckpt
(pick_orange SFT hasn't been run yet). Expect SR≈0% — this run only
validates the plumbing on the new env.
"""
import argparse
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--num-envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=20, help="total episodes (must be multiple of num-envs)")
parser.add_argument("--max-steps", type=int, default=1500)
parser.add_argument(
    "--model-path",
    default="/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/checkpoints/034000/pretrained_model",
)
parser.add_argument("--num-action-chunks", type=int, default=50,
                    help="action chunk length; SFT trained with 50, RLinf default was 5")
parser.add_argument("--num-steps", type=int, default=10,
                    help="flow-matching denoise steps; lerobot default 10, RLinf default was 5")
parser.add_argument("--field-dump", action="store_true",
                    help="emit [FIELD_DUMP] lines at every alignment probe point (batch 0 only)")
args = parser.parse_args()

sys.argv = sys.argv[:1]
assert args.episodes % args.num_envs == 0, "episodes must be a multiple of num-envs"

_REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ["LEISAAC_ASSETS_ROOT"] = str(_REPO / "third_party" / "leisaac" / "assets")

sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

from isaaclab.app import AppLauncher

app_launcher = AppLauncher(headless=not args.gui, enable_cameras=True)
simulation_app = app_launcher.app

import functools
print = functools.partial(print, flush=True)  # noqa: A001

import gymnasium as gym
import torch
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

import leisaac  # noqa: F401
from leisaac.devices.action_process import init_action_cfg
from leisaac.utils.robot_utils import (
    convert_leisaac_action_to_lerobot,
    convert_lerobot_action_to_leisaac,
    is_so101_at_rest_pose,
)

DEVICE = "cuda:0"
ENV_ID = "LeIsaac-SO101-PickOrange-v0"
TASK_PROMPT = "Grab orange and place into plate"
# NOTE: pi05_isaaclab_so101_lift uses So101LiftOutputs (not IsaacLabOutputs);
# So101LiftOutputs already preserves the continuous gripper — no patch needed.

# --- field-dump (alignment with lerobot eval) -------------------------------
# Probe schema MIRRORS scripts/eval_pi05_pickorange_lerobot.py exactly so
# `diff` between the two FIELD_DUMP streams lands directly on the divergence.
_FD_FIRED = {"count": 0}

def _fd(label: str, val):
    if not args.field_dump:
        return
    if isinstance(val, torch.Tensor):
        t = val.detach()
        shape = tuple(t.shape)
        if t.dtype.is_floating_point:
            mn = float(t.float().min()); mx = float(t.float().max())
            sample = [round(float(x), 6) for x in t.float().flatten()[:8].tolist()]
            print(f"[FIELD_DUMP] {label}  dtype={t.dtype}  shape={shape}  "
                  f"range=[{mn:.6f},{mx:.6f}]  sample={sample}")
        else:
            mn = int(t.min()); mx = int(t.max())
            sample = t.flatten()[:20].tolist()
            print(f"[FIELD_DUMP] {label}  dtype={t.dtype}  shape={shape}  "
                  f"range=[{mn},{mx}]  sample={sample}")
    elif isinstance(val, list):
        head = val[:3]
        print(f"[FIELD_DUMP] {label}  type=list  len={len(val)}  head={head!r}")
    elif isinstance(val, str):
        print(f"[FIELD_DUMP] {label}  type=str  len={len(val)}  text={val!r}")
    elif hasattr(val, "shape") and hasattr(val, "dtype"):  # numpy array
        import numpy as _np
        arr = _np.asarray(val)
        shape = tuple(arr.shape)
        if arr.dtype.kind == "f":
            mn = float(arr.min()); mx = float(arr.max())
            sample = [round(float(x), 6) for x in arr.flatten()[:8].tolist()]
            print(f"[FIELD_DUMP] {label}  dtype={arr.dtype}  shape={shape}  "
                  f"range=[{mn:.6f},{mx:.6f}]  sample={sample}")
        else:
            mn = int(arr.min()); mx = int(arr.max())
            sample = arr.flatten()[:20].tolist()
            print(f"[FIELD_DUMP] {label}  dtype={arr.dtype}  shape={shape}  "
                  f"range=[{mn},{mx}]  sample={sample}")
    else:
        print(f"[FIELD_DUMP] {label}  type={type(val).__name__}  value={val!r}")


env_cfg = parse_env_cfg(ENV_ID, device=DEVICE, num_envs=args.num_envs)
init_action_cfg(env_cfg.actions, device="so101leader")
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.max_steps * 0.02 + 1.0)
env = gym.make(ENV_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
print(f"[EVAL] env ready  action_space={env.action_space.shape}  num_envs={args.num_envs}")

ROBOT_JOINT_NAMES = list(env.scene["robot"].data.joint_names)


print("[EVAL] loading PI05Policy (lerobot-native adapter) …")
from lerobot_policy_adapter import LerobotPolicyAdapter
model = LerobotPolicyAdapter(args.model_path, device=DEVICE)
print("[EVAL] model loaded (lerobot-native)")

def wrap_obs(raw_obs, num_envs):
    """leisaac pick_orange obs → RLinf openpi model input.

    SingleArmObservationsCfg gives us both `front` and `wrist` cams
    (each [B, 480, 640, 3] uint8) plus joint_pos in radians.

    State must be in lerobot **motor-deg** units (matches dataset stats
    + train-time normalizer); a plain rad→deg conversion gives joint-deg
    which is per-joint mis-scaled.  leisaac's canonical client uses
    `convert_leisaac_action_to_lerobot()` for this — same function, just
    applied to joint_pos instead of an action.
    """
    policy = raw_obs["policy"]
    front = policy["front"]
    wrist = policy["wrist"]
    joint_pos = policy["joint_pos"]
    states_motor_deg_np = convert_leisaac_action_to_lerobot(joint_pos)
    states = torch.from_numpy(states_motor_deg_np).float().to(joint_pos.device)
    if _FD_FIRED["count"] == 0:
        # Mirror lerobot probe labels exactly so `diff` is meaningful.
        _fd("raw.front_u8", front[0])
        _fd("raw.wrist_u8", wrist[0])
        _fd("raw.joint_pos_rad", joint_pos[0])
        _fd("raw.state_motor_deg", states[0])
    return {
        "main_images": front,
        "wrist_images": wrist,
        "states": states,
        "task_descriptions": [TASK_PROMPT] * num_envs,
        "extra_view_images": None,
    }


n_batches = args.episodes // args.num_envs
all_successes: list[bool] = []
# Aggregate stage-wise counters (filled per-episode after each rollout).
agg = {
    "grasp_per_orange": [0, 0, 0],         # ever_picked[k] across all episodes
    "place_per_orange": [0, 0, 0],         # ever_placed[k] across all episodes (sticky-OR)
    "grasp_count_at_least": [0, 0, 0, 0],  # episodes with ≥0,≥1,≥2,≥3 oranges ever picked
    "place_count_at_least": [0, 0, 0, 0],  # episodes with ≥0,≥1,≥2,≥3 oranges ever placed
    "ever_all3_simul": 0,                  # 3 oranges on plate at the SAME step (any step)
    "ever_rest_after_first_pick": 0,       # arm reached rest pose after first grasp (not the trivial start-state rest)
    "ever_all3_simul_and_rest": 0,         # all 3 placed AND at rest at the same step
}

with torch.inference_mode():
    for batch in range(n_batches):
        raw_obs, _ = env.reset()
        env_obs = wrap_obs(raw_obs, args.num_envs)
        done = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        # success := env's built-in `success` DoneTerm fired (term && !trunc).
        succ = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)

        # Track partial progress via env subtask obs: pick_orange00N + put_orange00N_to_plate.
        # Per-orange `ever_*` is OR-over-time (sticky). For "all 3 simultaneously" we
        # OR the cross-orange AND-at-this-step, which is what `success` actually requires.
        n_oranges = 3
        ever_picked = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
        ever_placed = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
        ever_all3_simul = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        # Rest pose: arm starts at rest, so naive `ever_at_rest` is trivially True.
        # Track rest only after at least one orange has been grasped — i.e. the
        # "return to rest after task" event the success criterion really wants.
        any_picked_yet = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_rest_after_first_pick = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_all3_simul_and_rest = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)

        def _update_subtasks(obs):
            sub = obs.get("subtask_terms")
            cur_placed = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
            cur_picked = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
            if sub is not None:
                for i in range(1, n_oranges + 1):
                    pk = sub.get(f"pick_orange00{i}")
                    pl = sub.get(f"put_orange00{i}_to_plate")
                    if pk is not None:
                        cur_picked[:, i - 1] = pk.to(DEVICE).bool()
                    if pl is not None:
                        cur_placed[:, i - 1] = pl.to(DEVICE).bool()
            ever_picked[:] = ever_picked | cur_picked
            ever_placed[:] = ever_placed | cur_placed
            all3_now = cur_placed.all(dim=1)
            ever_all3_simul[:] = ever_all3_simul | all3_now
            any_picked_yet[:] = any_picked_yet | cur_picked.any(dim=1)
            # rest pose from joint_pos (radians) — same path `task_done` uses.
            # IsaacLab auto-resets done envs and joint_pos returns to start (= rest pose).
            # Mask rest-pose updates by `~done` so post-success auto-reset doesn't
            # spuriously light up `ever_rest_after_first_pick`.
            jp = obs.get("policy", {}).get("joint_pos")
            if jp is not None:
                at_rest_now = is_so101_at_rest_pose(jp.to(DEVICE), ROBOT_JOINT_NAMES)
                active = ~done
                ever_rest_after_first_pick[:] = ever_rest_after_first_pick | (at_rest_now & any_picked_yet & active)
                ever_all3_simul_and_rest[:] = ever_all3_simul_and_rest | (all3_now & at_rest_now & active)

        _update_subtasks(raw_obs)

        step = 0
        chunk_idx = 0
        while step < args.max_steps and not done.all():
            actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
            actions = actions.view(args.num_envs, args.num_action_chunks, 6)

            if _FD_FIRED["count"] == 0:
                # `actions` here is already unnormalized (openpi output_transform
                # ran Unnormalize). Mirror lerobot's "model.action_motor_deg".
                _fd("model.action_motor_deg", actions[0, 0, :])
                _fd("model.action_chunk_first_step", actions[0, 0, :])
                a_env_dbg_np = convert_lerobot_action_to_leisaac(actions[:, 0, :])
                _fd("model.action_env_rad", torch.from_numpy(a_env_dbg_np)[0])
                _FD_FIRED["count"] += 1

            if batch == 0 and chunk_idx < 3:
                print(f"[DIAG] chunk {chunk_idx}  env0 state pre-chunk = {env_obs['states'][0].tolist()}")
                for t_dbg in range(actions.shape[1]):
                    print(f"[DIAG]   action[env0, t={t_dbg}] = {actions[0, t_dbg].tolist()}")
            chunk_idx += 1

            for t in range(actions.shape[1]):
                if step >= args.max_steps or done.all():
                    break
                a_t = actions[:, t, :].to(DEVICE)
                # Model output is in lerobot motor-degree units (dataset stats).
                # leisaac env consumes joint positions in radians.  Convert
                # motor-deg → joint-deg (linear remap via per-joint limits) →
                # radians.  Without this the env gets ~90 rad commands and
                # clamps to joint limit on the first action (root cause of
                # 0% SR across all SFT ckpts).
                a_env_np = convert_lerobot_action_to_leisaac(a_t)
                a_env = torch.from_numpy(a_env_np).float().to(DEVICE)
                raw_obs, _rew, term, trunc, _info = env.step(a_env)
                _update_subtasks(raw_obs)
                active = ~done
                succ = succ | (active & term & ~trunc)
                done = done | term | trunc
                step += 1

            env_obs = wrap_obs(raw_obs, args.num_envs)

        for i in range(args.num_envs):
            all_successes.append(bool(succ[i].item()))
            picked = ever_picked[i].tolist()
            placed = ever_placed[i].tolist()
            all3 = bool(ever_all3_simul[i].item())
            rest = bool(ever_rest_after_first_pick[i].item())
            all3_rest = bool(ever_all3_simul_and_rest[i].item())
            print(
                f"[EVAL] ep {batch * args.num_envs + i:3d}  "
                f"success={int(succ[i].item())}  steps={step}  "
                f"picked={sum(picked)}/3 {picked}  placed={sum(placed)}/3 {placed}  "
                f"all3_simul={int(all3)}  rest_after_pick={int(rest)}  all3+rest={int(all3_rest)}"
            )
            for k in range(n_oranges):
                if picked[k]:
                    agg["grasp_per_orange"][k] += 1
                if placed[k]:
                    agg["place_per_orange"][k] += 1
            n_picked = sum(picked)
            n_placed = sum(placed)
            for k in range(4):
                if n_picked >= k:
                    agg["grasp_count_at_least"][k] += 1
                if n_placed >= k:
                    agg["place_count_at_least"][k] += 1
            agg["ever_all3_simul"] += int(all3)
            agg["ever_rest_after_first_pick"] += int(rest)
            agg["ever_all3_simul_and_rest"] += int(all3_rest)

n_succ = sum(all_successes)
N = len(all_successes)
print(f"\n[EVAL] success rate: {n_succ}/{N} = {n_succ / N:.1%}")
print("[EVAL] success criterion: env's `success` DoneTerm (all 3 oranges on plate + arm at rest)")


def _pct(num: int) -> str:
    return f"{num}/{N} ({num / N:.1%})"


print("\n[EVAL] --- stage breakdown ---")
print("[EVAL] per-orange grasp ever:  "
      f"O1={_pct(agg['grasp_per_orange'][0])}  "
      f"O2={_pct(agg['grasp_per_orange'][1])}  "
      f"O3={_pct(agg['grasp_per_orange'][2])}")
print("[EVAL] per-orange place ever:  "
      f"O1={_pct(agg['place_per_orange'][0])}  "
      f"O2={_pct(agg['place_per_orange'][1])}  "
      f"O3={_pct(agg['place_per_orange'][2])}")
print("[EVAL] cumulative grasps:      "
      f"≥1={_pct(agg['grasp_count_at_least'][1])}  "
      f"≥2={_pct(agg['grasp_count_at_least'][2])}  "
      f"=3={_pct(agg['grasp_count_at_least'][3])}")
print("[EVAL] cumulative places:      "
      f"≥1={_pct(agg['place_count_at_least'][1])}  "
      f"≥2={_pct(agg['place_count_at_least'][2])}  "
      f"=3={_pct(agg['place_count_at_least'][3])}")
print(f"[EVAL] ever all 3 on plate simul:   {_pct(agg['ever_all3_simul'])}")
print(f"[EVAL] ever rest-pose after pick:   {_pct(agg['ever_rest_after_first_pick'])}")
print(f"[EVAL] ever all3 + rest simul:      {_pct(agg['ever_all3_simul_and_rest'])}")

# Conditional stage-survival rates (chained transitions). 0/0 → "-".
def _cond(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den else f"{num}/{den} (-)"
g1 = agg['grasp_count_at_least'][1]
g2 = agg['grasp_count_at_least'][2]
g3 = agg['grasp_count_at_least'][3]
p1 = agg['place_count_at_least'][1]
p2 = agg['place_count_at_least'][2]
p3 = agg['place_count_at_least'][3]
print("\n[EVAL] --- conditional survival (where does the chain break?) ---")
print(f"[EVAL] P(grasp≥2 | grasp≥1)  = {_cond(g2, g1)}")
print(f"[EVAL] P(grasp=3 | grasp≥2)  = {_cond(g3, g2)}")
print(f"[EVAL] P(place≥1 | grasp≥1)  = {_cond(p1, g1)}")
print(f"[EVAL] P(place≥2 | place≥1)  = {_cond(p2, p1)}")
print(f"[EVAL] P(place=3 | place≥2)  = {_cond(p3, p2)}")
print(f"[EVAL] P(all3_simul | place=3 ever) = {_cond(agg['ever_all3_simul'], p3)}   "
      f"# fruit getting knocked off before all 3 land")
print(f"[EVAL] P(success | all3_simul)      = {_cond(n_succ, agg['ever_all3_simul'])}   "
      f"# parking + holding all 3 till done")

if args.gui:
    print("[EVAL] rollout done — GUI staying open. Ctrl-C to exit.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

env.close()
simulation_app.close()
