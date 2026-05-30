"""Import SO-101 URDF into Isaac Sim and save as USD.

Run inside Isaac Sim 4.2 Python environment:
    ${ISAAC_SIM}/python.sh scripts/import_so101_urdf.py
"""
from pathlib import Path

from omni.isaac.kit import SimulationApp

app = SimulationApp({"headless": True})

from omni.isaac.urdf import _urdf
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
