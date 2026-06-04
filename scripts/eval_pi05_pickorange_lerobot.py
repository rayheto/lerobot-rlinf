"""Standalone eval driver: SFT'd Pi 0.5 via lerobot-native runtime → LeIsaac
SO-101 pick_orange env. Bypasses the openpi remap pipeline entirely.

Loads:
  * PI05Policy.from_pretrained(<ckpt>/pretrained_model)
  * policy_preprocessor.json + step_2_normalizer.safetensors
  * policy_postprocessor.json + step_0_unnormalizer.safetensors

Run (headless):
  /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \
      scripts/eval_pi05_pickorange_lerobot.py \
      --model-path outputs/sft_pi05_pickorange/checkpoints/028000/pretrained_model
Run (GUI):
  DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python \
      scripts/eval_pi05_pickorange_lerobot.py --gui --model-path ...

A/B target: if this also gets 0% SR, the SFT itself didn't learn the task
(no remap / norm_stats blame). If this works but eval_pi05_pickorange.py
(openpi path) doesn't, conversion is the bug.
"""
import argparse
import os
import pathlib
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true")
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--episodes", type=int, default=4, help="must be multiple of num-envs")
parser.add_argument("--max-steps", type=int, default=400)
parser.add_argument(
    "--n-action-steps",
    type=int,
    default=None,
    help="override policy.config.n_action_steps (closes the loop faster; chunk_size unchanged)",
)
parser.add_argument(
    "--hl-state-machine",
    action="store_true",
    help="hand-coded HL policy: per-stage language prompt switching by placed-count "
         "(0/1/2/3). Targets chain decay (per-orange grasp drop-off) AND the parking "
         "bottleneck simultaneously. All 4 prompts are OOD: training used a single "
         "static prompt 'Grab orange and place into plate'. Effect is not guaranteed.",
)
parser.add_argument("--hl-prompt-0", default="Pick up the first orange and place it on the plate",
                    help="prompt when placed==0")
parser.add_argument("--hl-prompt-1", default="Pick up the second orange and place it on the plate",
                    help="prompt when placed==1")
parser.add_argument("--hl-prompt-2", default="Pick up the third orange and place it on the plate",
                    help="prompt when placed==2")
parser.add_argument("--hl-rest-prompt", default="Move robot arm back to rest position",
                    help="prompt when placed==3 (parking phase)")
parser.add_argument(
    "--model-path",
    default="/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/checkpoints/028000/pretrained_model",
)
parser.add_argument("--field-dump", action="store_true",
                    help="emit [FIELD_DUMP] lines at alignment probe points (batch 0, first chunk only)")
args = parser.parse_args()

sys.argv = sys.argv[:1]
assert args.episodes % args.num_envs == 0, "episodes must be a multiple of num-envs"

_REPO = pathlib.Path(__file__).resolve().parents[1]
os.environ["LEISAAC_ASSETS_ROOT"] = str(_REPO / "third_party" / "leisaac" / "assets")

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

# --- lerobot native runtime -------------------------------------------------
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

DEVICE = "cuda:0"
ENV_ID = "LeIsaac-SO101-PickOrange-v0"
TASK_PROMPT = "Grab orange and place into plate"

# --- field-dump (alignment with RLinf openpi eval) --------------------------
# Probe points emit `[FIELD_DUMP] <label>  dtype=...  shape=...  range=[..,..]
# sample=[...]` lines.  Both lerobot eval and RLinf eval use the SAME label
# scheme so `diff` cuts straight to the divergence.  Gated on --field-dump
# and only fires on the first model call (batch 0, first chunk).
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
    else:
        print(f"[FIELD_DUMP] {label}  type={type(val).__name__}  value={val!r}")

# --- env --------------------------------------------------------------------
env_cfg = parse_env_cfg(ENV_ID, device=DEVICE, num_envs=args.num_envs)
init_action_cfg(env_cfg.actions, device="so101leader")
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.max_steps * 0.02 + 1.0)
env = gym.make(ENV_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
print(f"[EVAL] env ready  action_space={env.action_space.shape}  num_envs={args.num_envs}")

ROBOT_JOINT_NAMES = list(env.scene["robot"].data.joint_names)

# --- policy + processors ----------------------------------------------------
print(f"[EVAL] loading PI05Policy from {args.model_path}")
policy = PI05Policy.from_pretrained(args.model_path).to(DEVICE).eval()
preprocessor, postprocessor = make_pre_post_processors(
    policy.config, pretrained_path=args.model_path
)
if args.n_action_steps is not None:
    assert 1 <= args.n_action_steps <= policy.config.chunk_size, \
        f"n_action_steps must be in [1, chunk_size={policy.config.chunk_size}]"
    print(f"[EVAL] override n_action_steps: {policy.config.n_action_steps} → {args.n_action_steps}")
    policy.config.n_action_steps = args.n_action_steps
print(
    f"[EVAL] policy loaded  chunk_size={policy.config.chunk_size}  "
    f"n_action_steps={policy.config.n_action_steps}  state_dim={policy.config.max_state_dim}"
)

# Monkey-patch _preprocess_images: dump the per-camera tensors that actually
# enter the model (post resize_with_pad, post *2-1 normalize, with -1-padded
# empty cameras). Key field names mirror RLinf's image dict so diff aligns.
_IMG_FEAT_KEYS = list(policy.config.input_features.keys())
_IMG_FEAT_KEYS = [k for k in _IMG_FEAT_KEYS if k.startswith("observation.images.")]
_orig_preprocess_images = policy._preprocess_images

def _patched_preprocess_images(batch):
    images, img_masks = _orig_preprocess_images(batch)
    if _FD_FIRED["count"] == 0:
        for i, (img, msk) in enumerate(zip(images, img_masks)):
            cam = _IMG_FEAT_KEYS[i] if i < len(_IMG_FEAT_KEYS) else f"slot{i}"
            cam = cam.replace("observation.images.", "")
            _fd(f"pix.{cam}", img[0])
            _fd(f"pix.{cam}.mask", msk)
    return images, img_masks

policy._preprocess_images = _patched_preprocess_images
print(f"[EVAL] image feature keys (preprocess_images order): {_IMG_FEAT_KEYS}")
HL_STAGE_PROMPTS = [
    args.hl_prompt_0,
    args.hl_prompt_1,
    args.hl_prompt_2,
    args.hl_rest_prompt,
]
if args.hl_state_machine:
    print(f"[EVAL] hl_state_machine=ON")
    for k, p in enumerate(HL_STAGE_PROMPTS):
        print(f"[EVAL]   placed={k} → '{p}'")
else:
    print(f"[EVAL] hl_state_machine=OFF  (using static prompt '{TASK_PROMPT}')")


def _per_env_prompts(raw_obs, num_envs):
    """Hand-coded HL state machine: derive per-env prompt from `subtask_terms`.

    Switches by placed count (0/1/2/3). All 4 prompts are OOD w.r.t. training
    (training had a single static prompt), so this is an empirical probe of
    PaliGemma's prompt sensitivity.

    Falls back to the static training prompt if subtask_terms is missing.
    """
    if not args.hl_state_machine:
        return [TASK_PROMPT] * num_envs
    sub = raw_obs.get("subtask_terms") or {}
    p1 = sub.get("put_orange001_to_plate")
    p2 = sub.get("put_orange002_to_plate")
    p3 = sub.get("put_orange003_to_plate")
    prompts = []
    for i in range(num_envs):
        n = 0
        if p1 is not None and bool(p1[i].item()):
            n += 1
        if p2 is not None and bool(p2[i].item()):
            n += 1
        if p3 is not None and bool(p3[i].item()):
            n += 1
        prompts.append(HL_STAGE_PROMPTS[min(n, 3)])
    return prompts


def build_obs(raw_obs, num_envs):
    """leisaac raw obs → lerobot-format flat batch dict.

    Image keys use the dataset-original names (front, wrist) so the
    rename_observations_processor step inside the preprocessor maps them
    to {base_0_rgb, right_wrist_0_rgb}. left_wrist_0_rgb + empty_camera_0
    are missing → PI05Policy._preprocess_images auto-pads with -1 + zero
    mask, matching training (empty_cameras=1 + 2-cam dataset).

    Images: leisaac returns [B, 480, 640, 3] uint8.  PI05 expects float
    in [0, 1] (the policy's internal _preprocess_images does *2-1, no
    /255); convert here.

    State: leisaac joint_pos is in radians.  Dataset stats are in lerobot
    motor-degree units, same per-joint linear map used by the openpi
    eval driver (convert_leisaac_action_to_lerobot).
    """
    policy_obs = raw_obs["policy"]
    front_u8 = policy_obs["front"]
    wrist_u8 = policy_obs["wrist"]
    joint_pos = policy_obs["joint_pos"]

    # [B, H, W, 3] uint8 → [B, 3, H, W] float in [0, 1]
    def _to_float_chw(img_u8):
        if img_u8.dtype != torch.uint8:
            img_u8 = img_u8.to(torch.uint8)
        return (
            img_u8.to(torch.float32)
            .div_(255.0)
            .permute(0, 3, 1, 2)
            .contiguous()
        )

    front = _to_float_chw(front_u8)
    wrist = _to_float_chw(wrist_u8)

    states_motor_deg_np = convert_leisaac_action_to_lerobot(joint_pos)
    states = torch.from_numpy(states_motor_deg_np).float()  # [B, 6], CPU OK

    if _FD_FIRED["count"] == 0:
        _fd("raw.front_u8", front_u8[0])
        _fd("raw.wrist_u8", wrist_u8[0])
        _fd("raw.joint_pos_rad", joint_pos[0])
        _fd("raw.front_chw_float01", front[0])
        _fd("raw.wrist_chw_float01", wrist[0])
        _fd("raw.state_motor_deg", states[0])

    return {
        "observation.images.front": front,
        "observation.images.wrist": wrist,
        "observation.state": states,
        "task": _per_env_prompts(raw_obs, num_envs),
    }


n_batches = args.episodes // args.num_envs
total_placed = 0
n_oranges = 3
all_successes: list[bool] = []
# Aggregate stage-wise counters (filled per-episode after each rollout).
agg = {
    "grasp_per_orange": [0, 0, 0],
    "place_per_orange": [0, 0, 0],
    "grasp_count_at_least": [0, 0, 0, 0],
    "place_count_at_least": [0, 0, 0, 0],
    "ever_all3_simul": 0,
    "ever_rest_after_first_pick": 0,
    "ever_all3_simul_and_rest": 0,
    "ever_hl_fired": 0,
}

with torch.inference_mode():
    for batch_idx in range(n_batches):
        raw_obs, _ = env.reset()
        policy.reset()  # clear action queue at start of each episode batch

        done = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        # episode-level success := env's `success` DoneTerm fired (term && !trunc).
        succ = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_picked = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
        ever_placed = torch.zeros(args.num_envs, n_oranges, dtype=torch.bool, device=DEVICE)
        ever_all3_simul = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        any_picked_yet = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_rest_after_first_pick = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_all3_simul_and_rest = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        ever_hl_fired = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        prev_placed_count = torch.zeros(args.num_envs, dtype=torch.long, device=DEVICE)

        def _update_subtasks(obs, cur_step):
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
            cur_placed_count = cur_placed.sum(dim=1).long()
            if args.hl_state_machine:
                # Log every forward stage transition (downward changes after
                # auto-reset are suppressed). Each line = a prompt switch.
                advanced = cur_placed_count > prev_placed_count
                if advanced.any():
                    for ei in advanced.nonzero(as_tuple=False).flatten().tolist():
                        ep_id = batch_idx * args.num_envs + ei
                        prev_n = int(prev_placed_count[ei].item())
                        new_n = int(cur_placed_count[ei].item())
                        print(f"[HL] ep={ep_id:3d} env={ei} step={cur_step:4d}  "
                              f"placed {prev_n}→{new_n}  "
                              f"prompt='{HL_STAGE_PROMPTS[min(new_n, 3)]}'")
            prev_placed_count[:] = cur_placed_count
            ever_hl_fired[:] = ever_hl_fired | (cur_placed_count >= 3)
            any_picked_yet[:] = any_picked_yet | cur_picked.any(dim=1)
            # Arm starts at rest — gate rest-pose updates by `any_picked_yet & ~done`
            # so we capture "returned to rest after the task" not the trivial start
            # state or the post-success auto-reset state.
            jp = obs.get("policy", {}).get("joint_pos")
            if jp is not None:
                at_rest_now = is_so101_at_rest_pose(jp.to(DEVICE), ROBOT_JOINT_NAMES)
                active = ~done
                ever_rest_after_first_pick[:] = ever_rest_after_first_pick | (at_rest_now & any_picked_yet & active)
                ever_all3_simul_and_rest[:] = ever_all3_simul_and_rest | (all3_now & at_rest_now & active)

        _update_subtasks(raw_obs, 0)

        diag_dumped = 0
        step = 0
        while step < args.max_steps and not done.all():
            # Rebuild obs every step — cheap. select_action only re-invokes
            # the model when its internal queue empties (every chunk_size=50
            # steps); on other calls the batch arg is ignored.
            obs_batch = build_obs(raw_obs, args.num_envs)
            obs_processed = preprocessor(obs_batch)

            if _FD_FIRED["count"] == 0:
                if isinstance(obs_processed, dict):
                    for k in sorted(obs_processed.keys()):
                        v = obs_processed[k]
                        if isinstance(v, torch.Tensor) and v.ndim >= 1:
                            _fd(f"pre.{k}", v[0])
                        elif isinstance(v, list):
                            _fd(f"pre.{k}", v[0] if v else v)
                        else:
                            _fd(f"pre.{k}", v)

            # [B, 6] in normalized space
            action_norm = policy.select_action(obs_processed)
            # postprocessor: unnormalize (quantile) + move to cpu
            action_motor_deg = postprocessor(action_norm)  # [B, 6] on CPU

            if _FD_FIRED["count"] == 0:
                _fd("model.action_norm", action_norm[0])
                _fd("model.action_motor_deg", action_motor_deg[0])
                a_env_dbg_np = convert_lerobot_action_to_leisaac(action_motor_deg)
                _fd("model.action_env_rad", torch.from_numpy(a_env_dbg_np)[0])
                _FD_FIRED["count"] += 1

            if batch_idx == 0 and diag_dumped < 3 and step % policy.config.n_action_steps == 0:
                print(
                    f"[DIAG] step={step}  env0 state(motor-deg) = "
                    f"{obs_batch['observation.state'][0].tolist()}"
                )
                print(f"[DIAG]   action_norm[env0]   = {action_norm[0].tolist()}")
                print(f"[DIAG]   action_motor[env0] = {action_motor_deg[0].tolist()}")
                diag_dumped += 1

            # motor-deg → joint-deg → rad (matches openpi eval; without this
            # the env gets ~90 rad commands and clamps to joint limit).
            a_env_np = convert_lerobot_action_to_leisaac(action_motor_deg)
            a_env = torch.from_numpy(a_env_np).float().to(DEVICE)

            raw_obs, _rew, term, trunc, _info = env.step(a_env)
            _update_subtasks(raw_obs, step + 1)
            active = ~done
            succ = succ | (active & term & ~trunc)
            done = done | term | trunc
            step += 1

        for i in range(args.num_envs):
            placed = ever_placed[i].tolist()
            picked = ever_picked[i].tolist()
            all3 = bool(ever_all3_simul[i].item())
            rest = bool(ever_rest_after_first_pick[i].item())
            all3_rest = bool(ever_all3_simul_and_rest[i].item())
            ep_succ = bool(succ[i].item())
            total_placed += sum(placed)
            all_successes.append(ep_succ)
            print(
                f"[EVAL] ep {batch_idx * args.num_envs + i:3d}  "
                f"success={int(ep_succ)}  "
                f"placed={sum(placed)}/3 {placed}  picked={sum(picked)}/3 {picked}  "
                f"all3_simul={int(all3)}  rest_after_pick={int(rest)}  all3+rest={int(all3_rest)}  "
                f"steps={step}"
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
            agg["ever_hl_fired"] += int(ever_hl_fired[i].item())

total_oranges = args.episodes * n_oranges
n_succ = sum(all_successes)
N = len(all_successes)
# Keep legacy "SR" line so orchestrator's grep still parses (per-orange place rate).
print(f"\n[EVAL] SR: {total_placed}/{total_oranges} = {total_placed / total_oranges:.1%}  "
      f"(per-orange place rate — orchestrator-compatible)")
# Real per-episode success rate (all 3 placed simul + arm at rest).
print(f"[EVAL] full-task success rate: {n_succ}/{N} = {n_succ / N:.1%}  "
      f"(env DoneTerm: 3 oranges on plate + arm at rest)")
print("[EVAL] runtime: lerobot-native PI05Policy (no openpi remap)")


def _pct(num: int) -> str:
    return f"{num}/{N} ({num / N:.1%})"


def _cond(num: int, den: int) -> str:
    return f"{num}/{den} ({num / den:.1%})" if den else f"{num}/{den} (-)"


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
if args.hl_state_machine:
    print(f"[EVAL] HL state-machine fired in:   {_pct(agg['ever_hl_fired'])}   "
          f"# episodes where prompt switched to rest-pose")

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
      f"# fruit knocked off before all 3 land")
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
