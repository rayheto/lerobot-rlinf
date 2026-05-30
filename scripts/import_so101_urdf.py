"""Import SO-101 URDF into Isaac Sim and save as USD.

Target: Isaac Sim 5.1 (pip install in conda env `rlinf-isaacsim-env`).
NOTE: 5.1 renamed the URDF importer API — the body below uses the 4.x API
and will be rewritten against `isaacsim.asset.importer.urdf` once the
5.1 install is verified.
"""
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from omni.isaac.urdf import _urdf  # TODO(5.1): migrate to isaacsim.asset.importer.urdf
from omni.isaac.core.utils.stage import save_stage

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_PATH = REPO_ROOT / "assets/so_arm100/SO101/so101_new_calib.urdf"
USD_OUT = REPO_ROOT / "assets/so_arm100/SO101/so101_new_calib.usd"

urdf_interface = _urdf.acquire_urdf_interface()
cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.make_default_prim = True
cfg.self_collision_enabled = False
cfg.create_physics_scene = True
cfg.default_drive_type = _urdf.UrdfJointTargetType.JOINT_DRIVE_POSITION
cfg.default_drive_strength = 1000.0
cfg.default_position_drive_damping = 50.0

result, prim_path = urdf_interface.parse_urdf(str(URDF_PATH.parent), URDF_PATH.name, cfg)
urdf_interface.import_robot(str(URDF_PATH.parent), URDF_PATH.name, result, cfg, str(USD_OUT))

print(f"[ok] imported to {prim_path}, saved {USD_OUT}")
app.close()
