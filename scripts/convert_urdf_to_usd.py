"""Convert SO-101 URDF to USD via Isaac Sim 5.1 URDF importer.

Run:
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/convert_urdf_to_usd.py
"""
from pathlib import Path

from isaacsim import SimulationApp

app = SimulationApp({"headless": True})

from isaacsim.core.utils.extensions import enable_extension
enable_extension("isaacsim.asset.importer.urdf")

from isaacsim.asset.importer.urdf import _urdf

REPO_ROOT = Path(__file__).resolve().parent.parent
URDF_DIR = REPO_ROOT / "assets/so_arm100/SO101"
URDF_NAME = "so101_new_calib.urdf"
USD_OUT = URDF_DIR / "so101_new_calib.usd"

iface = _urdf.acquire_urdf_interface()
cfg = _urdf.ImportConfig()
cfg.merge_fixed_joints = False
cfg.fix_base = True
cfg.make_default_prim = True
cfg.self_collision = False
cfg.create_physics_scene = True
cfg.default_drive_type = _urdf.JOINT_DRIVE_POSITION
cfg.default_drive_strength = 1000.0
cfg.default_position_drive_damping = 50.0
cfg.import_inertia_tensor = True
cfg.distance_scale = 1.0
cfg.density = 0.0  # use URDF-provided masses

robot = iface.parse_urdf(str(URDF_DIR), URDF_NAME, cfg)
print(f"[urdf] parsed: links={len(robot.links)} joints={len(robot.joints)}")
for jname, joint in robot.joints.items():
    print(f"  joint  {jname:25s} type={joint.type} limit=[{joint.limit.lower:+.3f},{joint.limit.upper:+.3f}]")

prim_path = iface.import_robot(str(URDF_DIR), URDF_NAME, robot, cfg, str(USD_OUT))
print(f"[urdf] imported -> prim={prim_path}  usd={USD_OUT}")

app.close()
