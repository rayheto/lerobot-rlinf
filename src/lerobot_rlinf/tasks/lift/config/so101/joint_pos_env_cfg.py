"""SO-101 cube-lift env config — LeRobot Pi 0.5 compatible obs/action.

Conventions (match real SO-101 teleop datasets captured by lerobot):
- `observation.state`: [B, 6] Feetech-normalized joint positions in [-100, 100],
  URDF joint order (`SO101_JOINT_NAMES`).
- `observation.images.{front, wrist}`: [B, H, W, 3] uint8 RGB at 224×224.
- action: [B, 6] Feetech-normalized [-100, 100], same URDF joint order. Mapped
  to radians per-joint by IsaacLab's JointPositionActionCfg.

IsaacLab group naming (`policy` / `images`) is preserved; renaming to LeRobot's
`observation.state` / `observation.images.*` happens at the wrapper layer.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from lerobot_rlinf.assets.so101 import (
    SO101_CFG,
    SO101_FEETECH_OFFSET,
    SO101_FEETECH_SCALE,
    SO101_JOINT_NAMES,
)
from lerobot_rlinf.tasks.lift import mdp
from lerobot_rlinf.tasks.lift.lift_env_cfg import LiftEnvCfg


# End-effector body for command/reward tracking. `gripper_frame_link` is the fixed
# frame at the jaw tip; falls between the two fingers when the gripper is centered.
EE_BODY_NAME = "gripper_frame_link"

# Split arm vs gripper to fit base ActionsCfg (arm_action + gripper_action). The
# concatenated action vector still totals 6 in URDF joint order.
_ARM_JOINTS = SO101_JOINT_NAMES[:5]   # shoulder_pan ... wrist_roll
_GRIPPER_JOINTS = SO101_JOINT_NAMES[5:]  # gripper


@configclass
class StateCfg(ObsGroup):
    """observation.state — 6-DoF Feetech-normalized joint pos, URDF order."""

    joint_pos = ObsTerm(
        func=mdp.joint_pos_feetech,
        params={
            "asset_name": "robot",
            "joint_names": SO101_JOINT_NAMES,
            "scale": SO101_FEETECH_SCALE,
            "offset": SO101_FEETECH_OFFSET,
        },
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class ImagesCfg(ObsGroup):
    """observation.images — two RGB views matching LeRobot SO-101 teleop naming."""

    front = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("front_cam"), "data_type": "rgb", "normalize": False},
    )
    wrist = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SO101CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Action: Feetech-normalized [-100, 100] → radians via per-joint scale+offset.
        # IsaacLab's JointPositionActionCfg computes `target_rad = scale * action + offset`.
        # Split into two terms (arm + gripper) to fit base ActionsCfg structure;
        # ActionManager concatenates them in declaration order so the full input is
        # still [B, 6] in URDF joint order.
        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_ARM_JOINTS,
            scale={n: SO101_FEETECH_SCALE[n] for n in _ARM_JOINTS},
            offset={n: SO101_FEETECH_OFFSET[n] for n in _ARM_JOINTS},
            use_default_offset=False,
            preserve_order=True,
        )
        self.actions.gripper_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=_GRIPPER_JOINTS,
            scale={n: SO101_FEETECH_SCALE[n] for n in _GRIPPER_JOINTS},
            offset={n: SO101_FEETECH_OFFSET[n] for n in _GRIPPER_JOINTS},
            use_default_offset=False,
            preserve_order=True,
        )

        # Replace the base class's rich policy obs (joint_pos_rel, joint_vel_rel,
        # object_position, command, last_action) with LeRobot-format 6-dim joint pos.
        # Privileged sim state (object pos, target) is still available to rewards via
        # scene entities; if we later need it for an RL critic, add a separate group.
        self.observations.policy = StateCfg()
        self.observations.images = ImagesCfg()

        self.commands.object_pose.body_name = EE_BODY_NAME

        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.3, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.5, 0.5, 0.5),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY_NAME,
                    name="end_effector",
                ),
            ],
        )

        # SO-101 is much smaller than Franka — tighten command range so targets stay reachable.
        self.commands.object_pose.ranges.pos_x = (0.15, 0.35)
        self.commands.object_pose.ranges.pos_y = (-0.15, 0.15)
        self.commands.object_pose.ranges.pos_z = (0.10, 0.25)

        # Overhead RGB ("front" in LeRobot SO-101 convention), world-fixed per env.
        # ROS: cam +Z = forward; rot=(0,1,0,0) is 180° about X → forward points -Z (down).
        self.scene.front_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/front_cam",
            update_period=0.0,
            width=224,
            height=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.7), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        )
        # Green debug sphere at front_cam pose. Placeholder for future real camera USD mesh.
        self.scene.front_cam_viz = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/front_cam/viz",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        )

        # Wrist RGB, mounted on gripper_frame_link (jaw tip). URDF flips this frame
        # 180° about Y relative to gripper_link, so its +Z points OUT of the jaws.
        # Offset tuned visually with the red debug sphere.
        self.scene.wrist_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY_NAME + "/wrist_cam",
            update_period=0.0,
            width=224,
            height=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 2.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.09, 0.0, -0.13), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )
        # Red debug sphere at wrist_cam pose (follows the gripper).
        self.scene.wrist_cam_viz = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY_NAME + "/wrist_cam/viz",
            spawn=sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )


@configclass
class SO101CubeLiftEnvCfg_PLAY(SO101CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
