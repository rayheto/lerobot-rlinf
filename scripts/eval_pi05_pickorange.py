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
parser.add_argument("--max-steps", type=int, default=600)
parser.add_argument(
    "--model-path",
    default="/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_sponge/openpi_remapped",
)
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
from omegaconf import OmegaConf

import leisaac  # noqa: F401
from leisaac.devices.action_process import init_action_cfg
from rlinf.models.embodiment.openpi import get_model

DEVICE = "cuda:0"
ENV_ID = "LeIsaac-SO101-PickOrange-v0"
TASK_PROMPT = "Pick three oranges and put them into the plate, then reset the arm to rest state."


env_cfg = parse_env_cfg(ENV_ID, device=DEVICE, num_envs=args.num_envs)
init_action_cfg(env_cfg.actions, device="so101leader")
env_cfg.episode_length_s = max(env_cfg.episode_length_s, args.max_steps * 0.02 + 1.0)
env = gym.make(ENV_ID, cfg=env_cfg, render_mode="rgb_array").unwrapped
print(f"[EVAL] env ready  action_space={env.action_space.shape}  num_envs={args.num_envs}")


cfg = OmegaConf.create({
    "model_path": args.model_path,
    "model_type": "openpi",
    "action_dim": 6,
    "num_action_chunks": 5,
    "num_steps": 5,
    "add_value_head": True,
    "precision": "bfloat16",
    "openpi": {
        "config_name": "pi05_isaaclab_so101_lift",
        "num_images_in_input": 2,
        "noise_level": 0.5,
        "joint_logprob": False,
        "num_steps": 5,
        "value_after_vlm": True,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": True,
        "action_chunk": 5,
        "action_dim": 32,
        "action_env_dim": 6,
        "add_value_head": True,
    },
})
print("[EVAL] loading model …")
model = get_model(cfg).to(DEVICE).eval()
print("[EVAL] model loaded")


def wrap_obs(raw_obs, num_envs):
    """leisaac pick_orange obs → RLinf openpi model input.

    SingleArmObservationsCfg gives us both `front` and `wrist` cams
    (each [B, 480, 640, 3] uint8) plus joint_pos in radians.
    """
    policy = raw_obs["policy"]
    front = policy["front"]
    wrist = policy["wrist"]
    states_deg = policy["joint_pos"] * (180.0 / torch.pi)
    return {
        "main_images": front,
        "wrist_images": wrist,
        "states": states_deg,
        "task_descriptions": [TASK_PROMPT] * num_envs,
        "extra_view_images": None,
    }


n_batches = args.episodes // args.num_envs
all_successes: list[bool] = []

with torch.inference_mode():
    for batch in range(n_batches):
        raw_obs, _ = env.reset()
        env_obs = wrap_obs(raw_obs, args.num_envs)
        done = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)
        # success := env's built-in `success` DoneTerm fired (term && !trunc).
        succ = torch.zeros(args.num_envs, dtype=torch.bool, device=DEVICE)

        step = 0
        chunk_idx = 0
        while step < args.max_steps and not done.all():
            actions, _ = model.predict_action_batch(env_obs, mode="eval", compute_values=False)
            actions = actions.view(args.num_envs, 5, 6)

            if batch == 0 and chunk_idx < 3:
                print(f"[DIAG] chunk {chunk_idx}  env0 state pre-chunk = {env_obs['states'][0].tolist()}")
                for t_dbg in range(actions.shape[1]):
                    print(f"[DIAG]   action[env0, t={t_dbg}] = {actions[0, t_dbg].tolist()}")
            chunk_idx += 1

            for t in range(actions.shape[1]):
                if step >= args.max_steps or done.all():
                    break
                a_t = actions[:, t, :].to(DEVICE)
                raw_obs, _rew, term, trunc, _info = env.step(a_t)
                active = ~done
                succ = succ | (active & term & ~trunc)
                done = done | term | trunc
                step += 1

            env_obs = wrap_obs(raw_obs, args.num_envs)

        for i in range(args.num_envs):
            all_successes.append(bool(succ[i].item()))
            print(
                f"[EVAL] ep {batch * args.num_envs + i:3d}  "
                f"success={int(succ[i].item())}  steps={step}"
            )

n_succ = sum(all_successes)
print(f"\n[EVAL] success rate: {n_succ}/{len(all_successes)} = {n_succ / len(all_successes):.1%}")
print("[EVAL] success criterion: env's `success` DoneTerm (all 3 oranges on plate)")

if args.gui:
    print("[EVAL] rollout done — GUI staying open. Ctrl-C to exit.")
    try:
        while simulation_app.is_running():
            simulation_app.update()
    except KeyboardInterrupt:
        pass

env.close()
simulation_app.close()
