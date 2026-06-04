"""Cross-model comparison: feed same preprocessed inputs to both lerobot PI05Policy
and openpi model, diff the raw output.

Usage (isaacsim env):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/diff_model_outputs.py
"""
import sys
import torch
import numpy as np

sys.path.insert(0, "/home/hlei/RLinf")
sys.path.insert(0, "/home/hlei/RLinf/.venv/lib/python3.11/site-packages")

LEROBOT_CKPT = "/home/hlei/robotic/lerobot-rlinf/outputs/sft_pi05_pickorange/checkpoints/034000/pretrained_model"

torch.manual_seed(42)
np.random.seed(42)
device = "cuda:0"

# ---- step 1: load lerobot PI05Policy ----
print("=== LEROBOT PATH ===")
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # noqa: E402

lerobot_policy = PI05Policy.from_pretrained(LEROBOT_CKPT)
lerobot_policy.eval()
print(f"lerobot loaded, device={lerobot_policy.config.device}")

# Create deterministic dummy inputs matching post-_preprocess_images format
bsize = 1
dummy_img_real = torch.randn(bsize, 3, 224, 224, device=device)
dummy_img_pad = torch.full((bsize, 3, 224, 224), -1.0, device=device, dtype=torch.float32)

images = [dummy_img_real.clone(), dummy_img_pad.clone(),
          dummy_img_pad.clone(), dummy_img_pad.clone()]
img_masks = [
    torch.ones(bsize, dtype=torch.bool, device=device),
    torch.zeros(bsize, dtype=torch.bool, device=device),
    torch.zeros(bsize, dtype=torch.bool, device=device),
    torch.zeros(bsize, dtype=torch.bool, device=device),
]

tokens = torch.full((bsize, 200), 0, device=device, dtype=torch.long)
tokens[0, 0] = 2  # BOS
masks = torch.zeros(bsize, 200, dtype=torch.bool, device=device)
masks[0, 0] = True

with torch.inference_mode():
    lerobot_out = lerobot_policy.model.sample_actions(images, img_masks, tokens, masks, num_steps=10)
print(f"lerobot actions shape: {lerobot_out.shape}")
print(f"lerobot actions[0,0,:6]: {[round(float(x), 6) for x in lerobot_out[0,0,:6]]}")
print(f"lerobot actions range: [{float(lerobot_out.min()):.4f}, {float(lerobot_out.max()):.4f}]")

# ---- step 2: load openpi model ----
print("\n=== OPENPI PATH ===")
from omegaconf import OmegaConf  # noqa: E402
from rlinf.models.embodiment.openpi import get_model  # noqa: E402
from openpi.models import model as _model  # noqa: E402

cfg = OmegaConf.create({
    "model_path": LEROBOT_CKPT,
    "model_type": "openpi",
    "action_dim": 6,
    "num_action_chunks": 50,
    "num_steps": 10,
    "add_value_head": True,
    "precision": "bfloat16",
    "openpi": {
        "config_name": "pi05_isaaclab_so101_lift",
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
print("openpi model loaded")

# Feed same inputs through openpi — provide CHW format (what from_dict produces)
state = torch.zeros(bsize, 32, device=device, dtype=torch.float32)
torch.manual_seed(42)
obs = _model.Observation(
    state=state,
    images={
        "base_0_rgb": images[0],
        "left_wrist_0_rgb": images[1],
        "right_wrist_0_rgb": images[2],
        "empty_camera_0": images[3],
    },
    image_masks={
        "base_0_rgb": img_masks[0],
        "left_wrist_0_rgb": img_masks[1],
        "right_wrist_0_rgb": img_masks[2],
        "empty_camera_0": img_masks[3],
    },
    tokenized_prompt=tokens,
    tokenized_prompt_mask=masks,
)

with torch.inference_mode():
    result = openpi_model.sample_actions(obs, mode="eval", compute_values=False)
    openpi_out = result["actions"]

print(f"openpi actions shape: {openpi_out.shape}")
print(f"openpi actions[0,0,:6]: {[round(float(x), 6) for x in openpi_out[0,0,:6]]}")
print(f"openpi actions range: [{float(openpi_out.min()):.4f}, {float(openpi_out.max()):.4f}]")

# ---- step 3: compare ----
print("\n=== COMPARISON ===")
diff = (lerobot_out.float() - openpi_out.float()).abs()
print(f"lerobot range: [{float(lerobot_out.min()):.4f}, {float(lerobot_out.max()):.4f}]")
print(f"openpi  range: [{float(openpi_out.min()):.4f}, {float(openpi_out.max()):.4f}]")
print(f"max  abs diff: {float(diff.max()):.6f}")
print(f"mean abs diff: {float(diff.mean()):.6f}")

if diff.max() < 1e-3:
    print("VERDICT: Models IDENTICAL — issue is in preprocessing/image pipeline")
elif diff.max() < 0.5:
    print(f"VERDICT: Models CLOSE (max diff={float(diff.max()):.4f}) — likely bf16/fp32 precision")
else:
    print(f"VERDICT: Models DIVERGE (max diff={float(diff.max()):.2f}) — architecture or weight mismatch")
