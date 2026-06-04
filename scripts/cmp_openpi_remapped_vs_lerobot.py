"""Side-by-side diff: lerobot PI05Policy (baseline) vs openpi-native loading
the *converted* openpi_remapped checkpoint.

The earlier cmp scripts compared lerobot vs openpi-via-lerobot-adapter (both
loading from the lerobot ckpt → openpi short-circuits to the adapter, so the
native code path is never tested). This script forces openpi to load from
openpi_remapped/ which has no train_config.json → falls through to the
native OpenPi0ForRLActionPrediction path with Normalize/Unnormalize transforms.

Same raw env_obs goes into both pipelines. We print:
  - normalized state seen by the action expert (hooked from inside)
  - first 5 timesteps of the action chunk in motor-degree space
  - max |diff| of unnormalized actions across the chunk
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

LEROBOT_CKPT = "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/checkpoints/034000/pretrained_model"
OPENPI_REMAPPED = "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/openpi_remapped"

torch.manual_seed(42)
np.random.seed(42)
device = "cuda:0"
bsize = 1
PROMPT = "Grab orange and place into plate"

# Deterministic but realistic obs: random uint8 images, rest-pose-ish state.
front_u8 = torch.randint(0, 256, (bsize, 480, 640, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(1))
wrist_u8 = torch.randint(0, 256, (bsize, 480, 640, 3), dtype=torch.uint8, generator=torch.Generator().manual_seed(2))
# motor-space rest pose roughly: [0, -50, 50, 50, 0, 50]
state_motor_deg = torch.tensor([[0.0, -50.0, 50.0, 50.0, 0.0, 50.0]], dtype=torch.float32)

print("=" * 70)
print("Pipeline A: lerobot PI05Policy directly (baseline, ~80% grasp)")
print("=" * 70)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.factory import make_pre_post_processors

lerobot_policy = PI05Policy.from_pretrained(LEROBOT_CKPT).to(device).eval()
preprocessor, postprocessor = make_pre_post_processors(lerobot_policy.config, pretrained_path=LEROBOT_CKPT)

obs_batch = {
    "observation.images.front": front_u8.float().div(255.0).permute(0, 3, 1, 2).contiguous().to(device),
    "observation.images.wrist": wrist_u8.float().div(255.0).permute(0, 3, 1, 2).contiguous().to(device),
    "observation.state": state_motor_deg.to(device),
    "task": [PROMPT] * bsize,
}
obs_processed = preprocessor(obs_batch)
norm_state_lerobot = obs_processed["observation.state"]
print(f"[A] normalized state seen by model: {[round(float(x), 4) for x in norm_state_lerobot[0]]}")

images_l, masks_l = lerobot_policy._preprocess_images(obs_processed)
tokens_l = obs_processed["observation.language.tokens"]
attn_l = obs_processed["observation.language.attention_mask"]

torch.manual_seed(42)
with torch.inference_mode():
    raw_actions_l = lerobot_policy.model.sample_actions(images_l, masks_l, tokens_l, attn_l, num_steps=10)
print(f"[A] raw (normalized) action[0,0,:6]: {[round(float(x), 4) for x in raw_actions_l[0,0,:6]]}")

# postprocess: motor-space degrees
b, c, d = raw_actions_l[:, :, :6].shape
unnorm_l = postprocessor(raw_actions_l[:, :, :6].reshape(b * c, d).cpu())
if not isinstance(unnorm_l, torch.Tensor):
    unnorm_l = torch.from_numpy(np.asarray(unnorm_l)).float()
actions_l_deg = unnorm_l.reshape(b, c, d)
print(f"[A] motor-deg t=0..4 (joint 0): {[round(float(actions_l_deg[0, t, 0]), 2) for t in range(5)]}")
print(f"[A] motor-deg t=0..4 (joint 5/gripper): {[round(float(actions_l_deg[0, t, 5]), 2) for t in range(5)]}")
print(f"[A] motor-deg range: [{float(actions_l_deg.min()):.1f}, {float(actions_l_deg.max()):.1f}]")

del lerobot_policy
torch.cuda.empty_cache()

print()
print("=" * 70)
print("Pipeline B: openpi-native from openpi_remapped/ (the bug suspect)")
print("=" * 70)
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model

cfg = OmegaConf.create({
    "model_path": OPENPI_REMAPPED,
    "model_type": "openpi",
    "action_dim": 6,
    "num_action_chunks": 50,
    "num_steps": 10,
    "add_value_head": True,
    "precision": "bfloat16",
    "openpi": {
        "config_name": "pi05_isaaclab_so101_pick_orange",
        "discrete_state_input": True,
        "num_images_in_input": 2,
        "noise_level": 0.5,
        "joint_logprob": False,
        "num_steps": 10,
        "value_after_vlm": True,
        "value_vlm_mode": "mean_token",
        "detach_critic_input": True,
        "action_chunk": 50,
        "action_dim": 32,
        "action_env_dim": 6,
        "add_value_head": True,
    },
})
openpi_model = get_model(cfg).to(device).eval()

# Hook: capture state going in AND raw (pre-unnormalize) action coming out.
captured = {}
orig_sample = openpi_model.sample_actions
def _hook(obs, *args, **kwargs):
    captured["state"] = obs.state.detach().clone()
    out = orig_sample(obs, *args, **kwargs)
    captured["raw_actions"] = out["actions"].detach().clone()
    return out
openpi_model.sample_actions = _hook

env_obs = {
    "main_images": front_u8.to(device),
    "wrist_images": wrist_u8.to(device),
    "states": state_motor_deg.to(device),
    "task_descriptions": [PROMPT] * bsize,
    "extra_view_images": None,
}

torch.manual_seed(42)
with torch.inference_mode():
    op_flat, _ = openpi_model.predict_action_batch(env_obs, mode="eval", compute_values=False)
    actions_o_deg = op_flat.view(bsize, 50, 6).float().cpu()

norm_state_openpi = captured["state"][0].float().cpu()
raw_actions_o = captured["raw_actions"][0].float().cpu()  # [chunk, action_dim]
print(f"[B] normalized state seen by model (padded to 32): first 6 dims = {[round(float(x), 4) for x in norm_state_openpi[:6]]}")
print(f"[B] raw (pre-unnorm) action[0,:6]: {[round(float(x), 4) for x in raw_actions_o[0, :6]]}")
print(f"[B] motor-deg t=0..4 (joint 0): {[round(float(actions_o_deg[0, t, 0]), 2) for t in range(5)]}")
print(f"[B] motor-deg t=0..4 (joint 5/gripper): {[round(float(actions_o_deg[0, t, 5]), 2) for t in range(5)]}")
print(f"[B] motor-deg range: [{float(actions_o_deg.min()):.1f}, {float(actions_o_deg.max()):.1f}]")

print()
print("=" * 70)
print("DIFF")
print("=" * 70)
state_diff = (norm_state_lerobot[0].cpu() - norm_state_openpi[:6]).abs()
raw_act_diff = (raw_actions_l[0, 0, :6].cpu() - raw_actions_o[0, :6]).abs()
action_diff = (actions_l_deg - actions_o_deg).abs()
print(f"normalized state |A - B|:        max={float(state_diff.max()):.4f}  mean={float(state_diff.mean()):.4f}")
print(f"raw (pre-unnorm) action |A - B|: max={float(raw_act_diff.max()):.4f}  mean={float(raw_act_diff.mean()):.4f}")
print(f"  A raw: {[round(float(x), 4) for x in raw_actions_l[0, 0, :6]]}")
print(f"  B raw: {[round(float(x), 4) for x in raw_actions_o[0, :6]]}")
print(f"  A: {[round(float(x), 4) for x in norm_state_lerobot[0]]}")
print(f"  B: {[round(float(x), 4) for x in norm_state_openpi[:6]]}")
print(f"motor-deg action |A - B|:        max={float(action_diff.max()):.2f}  mean={float(action_diff.mean()):.2f}")

if state_diff.max() > 0.05:
    print("\n>>> STATE NORMALIZATION MISMATCH — openpi feeds the model a different state value.")
if action_diff.max() > 5.0:
    print(">>> ACTION OUTPUT DIVERGES — likely cascades from state mismatch and/or output unnormalize.")
