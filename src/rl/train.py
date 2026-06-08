"""SO-101 PickOrange PPO training entrypoint.

Wraps RLinf's ``examples/embodiment/train_embodied_agent.main`` with:

1. Runtime registry patch that injects our ``IsaaclabPickOrangeEnv`` into
   ``rlinf.envs.isaaclab.REGISTER_ISAACLAB_ENVS`` — no third_party edits.
2. Hydra ``--config-dir`` override so our YAML in ``src/rl/config/`` is the
   primary search path; RLinf's own ``examples/embodiment/config/`` is added
   as a secondary search path so we can still ``defaults: -`` reference its
   shared components (``model/pi0_5``, ``training_backend/fsdp`` etc.).

Usage (single node):

    bash src/rl/run.sh pick_orange_ppo

Or directly:

    EMBODIED_PATH=third_party/RLinf/examples/embodiment \\
    PYTHONPATH=$PWD:third_party/RLinf:$PYTHONPATH \\
    python -m src.rl.train --config-name=pick_orange_ppo \\
        runner.logger.log_path=./logs/$(date +%s)-pick_orange_ppo

Pre-requisite: convert the JAX/Orbax SFT checkpoint to PyTorch safetensors:

    python third_party/openpi/examples/convert_jax_model_to_pytorch.py \\
        --checkpoint_dir outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999/params \\
        --output_path outputs/pi05_lora_so101_pick_orange/so101_pick_orange_lora_v0/24999_pt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

print("[rl/train] entering main, this may take 3-5 min...", flush=True)

# Step 1: ensure RLinf is importable.
_REPO = Path(__file__).resolve().parents[2]
_RLINF = _REPO / "third_party" / "RLinf"
for p in (str(_REPO), str(_RLINF)):
    if p not in sys.path:
        sys.path.insert(0, p)

# Step 2: monkey-patch the IsaacLab env registry. Must run BEFORE the RLinf
# entry is imported (which itself triggers env worker initialization).
from src.rl.envs import registry_patch  # noqa: E402

registry_patch.patch()
assert registry_patch.is_patched(), "registry patch failed — IsaaclabPickOrangeEnv not registered"

# Step 3: ensure RLinf's hydra searchpath env var resolves so its defaults
# (env/behavior_r1pro, model/pi0_5 etc.) still load.
os.environ.setdefault("EMBODIED_PATH", str(_RLINF / "examples" / "embodiment"))

# Step 4: re-export RLinf's hydra main. We import it lazily so the patch has
# already run by the time hydra calls into the worker module that creates
# the env.
from examples.embodiment.train_embodied_agent import main  # noqa: E402


if __name__ == "__main__":
    # Inject --config-dir if the user hasn't provided one — point Hydra at
    # our src/rl/config/.
    cfg_dir = str(Path(__file__).resolve().parent / "config")
    if not any(arg.startswith(("--config-dir", "--config-path")) for arg in sys.argv[1:]):
        sys.argv.insert(1, f"--config-path={cfg_dir}")
    main()
