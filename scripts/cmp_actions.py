"""Compare openpi vs adapter action outputs — eliminate RNG divergence.

Feeds the SAME noise to both models' sample_actions to isolate
preprocessing vs sampling differences.
"""
import sys, torch
sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

CKPT = "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/checkpoints/034000/pretrained_model"

# --- load lerobot ---
print("Loading lerobot PI05Policy ...")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
lerobot_policy = PI05Policy.from_pretrained(CKPT).eval()

# --- load openpi ---
print("Loading openpi model ...")
from omegaconf import OmegaConf
from rlinf.models.embodiment.openpi import get_model
cfg = OmegaConf.create({
    "model_path": CKPT, "model_type": "openpi", "action_dim": 6,
    "num_action_chunks": 50, "num_steps": 10, "add_value_head": True, "precision": "float32",
    "openpi": {"config_name": "pi05_isaaclab_so101_lift", "discrete_state_input": True,
               "num_images_in_input": 2, "noise_level": 0.5, "joint_logprob": False,
               "num_steps": 10, "value_after_vlm": True, "value_vlm_mode": "mean_token",
               "detach_critic_input": True, "action_chunk": 50, "action_dim": 32,
               "action_env_dim": 6, "add_value_head": True},
})
openpi_model = get_model(cfg).eval()

# --- build identical preprocessed inputs ---
from lerobot.policies.factory import make_pre_post_processors
preprocessor, postprocessor = make_pre_post_processors(lerobot_policy.config, pretrained_path=CKPT)

bsize = 2
front_u8 = torch.randint(0, 256, (bsize, 480, 640, 3), dtype=torch.uint8)
wrist_u8 = torch.randint(0, 256, (bsize, 480, 640, 3), dtype=torch.uint8)
state_motor_deg = torch.zeros(bsize, 6, dtype=torch.float32)
state_motor_deg[0, 2] = 5.26

obs_batch = {
    "observation.images.front": front_u8.float().div(255.0).permute(0, 3, 1, 2).contiguous(),
    "observation.images.wrist": wrist_u8.float().div(255.0).permute(0, 3, 1, 2).contiguous(),
    "observation.state": state_motor_deg,
    "task": ["Grab orange and place into plate"] * bsize,
}
obs_processed = preprocessor(obs_batch)
images, img_masks = lerobot_policy._preprocess_images(obs_processed)
tokens = obs_processed["observation.language.tokens"]
masks = obs_processed["observation.language.attention_mask"]

# --- test 1: same noise, lerobot sample_actions vs openpi sample_actions ---
print("\n=== Test 1: Same preprocessed input + same noise → raw sample_actions ===")
noise = torch.randn(bsize, 50, 32)
noise_openpi = noise.clone()

with torch.inference_mode():
    lo = lerobot_policy.model.sample_actions(images, img_masks, tokens, masks, noise=noise, num_steps=10)

# For openpi, bypass predict_action_batch → go through _preprocess_observation then sample_actions directly
from openpi.models import model as _model
state_padded = torch.zeros(bsize, 32)
obs = _model.Observation(
    state=state_padded,
    images={k: img.permute(0, 2, 3, 1) for k, img in zip(
        ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "empty_camera_0"], images)},
    image_masks={k: m for k, m in zip(
        ["base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb", "empty_camera_0"], img_masks)},
    tokenized_prompt=tokens,
    tokenized_prompt_mask=masks,
)

with torch.inference_mode():
    oo_result = openpi_model.sample_actions(obs, noise=noise_openpi, mode="eval", compute_values=False)
    oo = oo_result["actions"]

lo6 = lo[:, :, :6]
oo6 = oo[:, :, :6]
# unnormalize both
b, c, d = lo6.shape
lo_unnorm = postprocessor(lo6.reshape(b * c, d))
if not isinstance(lo_unnorm, torch.Tensor):
    import numpy as np; lo_unnorm = torch.from_numpy(np.asarray(lo_unnorm)).float()
lo_deg = lo_unnorm.reshape(b, c, d)

# openpi output is already unnormalized by sample_actions → output_transform
oo_deg = oo6.float()

diff_same_noise = (lo_deg - oo_deg).abs()
print(f"lerobot t=0 env0: {[round(float(x), 2) for x in lo_deg[0, 0]]}")
print(f"openpi  t=0 env0: {[round(float(x), 2) for x in oo_deg[0, 0]]}")
print(f"lerobot step-step delta max: {float((lo_deg[0,1:]-lo_deg[0,:-1]).abs().max()):.1f}")
print(f"openpi  step-step delta max: {float((oo_deg[0,1:]-oo_deg[0,:-1]).abs().max()):.1f}")
print(f"max diff: {float(diff_same_noise.max()):.2f}, mean diff: {float(diff_same_noise.mean()):.2f}")

if diff_same_noise.max() < 0.1:
    print("SAME NOISE → IDENTICAL OUTPUT — issue is RNG divergence in preprocessing")
else:
    print(f"SAME NOISE → DIVERGE (max={float(diff_same_noise.max()):.2f}) — architecture/code path differs")

# --- test 2: predict_action_batch (full pipeline) ---
print("\n=== Test 2: full predict_action_batch pipeline ===")
env_obs = {
    "main_images": front_u8, "wrist_images": wrist_u8,
    "states": state_motor_deg,
    "task_descriptions": ["Grab orange and place into plate"] * bsize,
    "extra_view_images": None,
}

torch.manual_seed(42)
with torch.inference_mode():
    op_flat, _ = openpi_model.predict_action_batch(env_obs, mode="eval", compute_values=False)
    op_deg2 = op_flat.view(bsize, 50, 6).float()

print(f"predict_action_batch t=0 env0: {[round(float(x), 2) for x in op_deg2[0, 0]]}")
print(f"predict_action_batch step-step delta max: {float((op_deg2[0,1:]-op_deg2[0,:-1]).abs().max()):.1f}")

# --- test 3: test noise_level sensitivity ---
print("\n=== Test 3: noise_level impact ===")
for nl in [0.0, 0.1, 0.5, 1.0]:
    nl_cfg = OmegaConf.create({**cfg, "openpi": {**cfg.openpi, "noise_level": nl}})
    nl_model = get_model(nl_cfg).eval()
    torch.manual_seed(42)
    with torch.inference_mode():
        nl_flat, _ = nl_model.predict_action_batch(env_obs, mode="eval", compute_values=False)
        nl_deg = nl_flat.view(bsize, 50, 6).float()
    delta = float((nl_deg[0,1:]-nl_deg[0,:-1]).abs().max())
    print(f"  noise_level={nl}: step-step delta max={delta:.1f} deg, range=[{float(nl_deg.min()):.1f}, {float(nl_deg.max()):.1f}]")
