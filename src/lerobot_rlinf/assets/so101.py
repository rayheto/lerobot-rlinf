"""ArticulationCfg for the SO-ARM100 / SO-101 6-DoF arm.

Joint order (URDF): shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper.
USD is produced by `scripts/convert_urdf_to_usd.py` from the vendored URDF + meshes.
"""
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_REPO_ROOT = Path(__file__).resolve().parents[3]
SO101_USD_PATH = str(_REPO_ROOT / "assets/so_arm100/SO101/so101_new_calib.usd")

# Canonical URDF joint order. Used by both observation.state and action vectors so
# they match the convention LeRobot SO-101 teleop datasets capture.
SO101_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

# Per-joint Feetech ↔ URDF-radian calibration. LeRobot real-robot teleop captures
# action/state as normalized [-100, 100]% of each Feetech motor's calibrated
# extremes. To mimic that interface in sim, we map norm ↔ rad via:
#     rad = scale * norm + offset
#     norm = (rad - offset) / scale
# Values derived from URDF joint limits: scale = (upper-lower)/200,
# offset = (upper+lower)/2.
SO101_FEETECH_SCALE = {
    "shoulder_pan": 0.01920,    # ±1.920 rad
    "shoulder_lift": 0.01745,   # ±1.745 rad
    "elbow_flex": 0.01690,      # ±1.690 rad
    "wrist_flex": 0.01658,      # ±1.658 rad
    "wrist_roll": 0.027925,     # [-2.744, +2.841] rad — asymmetric
    "gripper": 0.00960,         # [-0.175, +1.745] rad — asymmetric
}
SO101_FEETECH_OFFSET = {
    "shoulder_pan": 0.0,
    "shoulder_lift": 0.0,
    "elbow_flex": 0.0,
    "wrist_flex": 0.0,
    "wrist_roll": 0.0485,
    "gripper": 0.785,
}


SO101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=SO101_USD_PATH,
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.8,
        },
    ),
    actuators={
        # PD tuned to hold SO-101 against gravity at zero pose. Franka's 80/4 was way
        # too soft — arm collapsed onto the table. URDF importer default is 1000/50.
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit_sim=10.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""SO-101 6-DoF (5 arm joints + 1 gripper). Effort/stiffness are placeholders — tune on hardware."""
