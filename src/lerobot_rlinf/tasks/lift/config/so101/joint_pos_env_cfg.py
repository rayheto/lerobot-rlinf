"""SO-101 cube-lift env config (joint-position action space)."""

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.sensors import CameraCfg, FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from lerobot_rlinf.assets.so101 import SO101_CFG
from lerobot_rlinf.tasks.lift import mdp
from lerobot_rlinf.tasks.lift.lift_env_cfg import LiftEnvCfg


# URDF joint limit for "gripper" is [-0.175, +1.745]. Use the wide-open and closed extremes.
GRIPPER_OPEN = 1.745
GRIPPER_CLOSED = -0.175

# End-effector body for command/reward tracking. `gripper_frame_link` is the fixed frame at
# the jaw tip; falls between the two fingers when the gripper is centered.
EE_BODY_NAME = "gripper_frame_link"


@configclass
class ImagesCfg(ObsGroup):
    """RGB camera obs kept as a dict — do NOT concatenate with state."""

    cam_high = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("cam_high"), "data_type": "rgb", "normalize": False},
    )
    cam_wrist = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("cam_wrist"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SO101CubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = SO101_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.actions.arm_action = mdp.JointPositionActionCfg(
            asset_name="robot",
            joint_names=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            scale=0.5,
            use_default_offset=True,
        )
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["gripper"],
            open_command_expr={"gripper": GRIPPER_OPEN},
            close_command_expr={"gripper": GRIPPER_CLOSED},
        )

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

        # Overhead RGB, world-fixed per env. ROS convention: cam +Z = forward.
        # rot=(0, 1, 0, 0) is 180deg about X → forward points to world -Z (straight down).
        self.scene.cam_high = CameraCfg(
            prim_path="{ENV_REGEX_NS}/cam_high",
            update_period=0.0,
            width=224,
            height=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(0.3, 0.0, 0.7), rot=(0.0, 1.0, 0.0, 0.0), convention="ros"),
        )
        # Green debug sphere at cam_high pose.
        self.scene.cam_high_viz = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/cam_high/viz",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        )

        # Wrist RGB, attached to gripper_frame_link (jaw tip). URDF flips this frame 180°
        # about Y relative to gripper_link, so its +Z points OUT of the jaws.
        # Offset: pull -Z by 13cm (back toward gripper body), bump +Y by 8cm as a guess for "up".
        # Tune empirically by watching the red debug sphere.
        self.scene.cam_wrist = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY_NAME + "/cam_wrist",
            update_period=0.0,
            width=224,
            height=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.02, 2.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.09, 0.0, -0.13), rot=(1.0, 0.0, 0.0, 0.0), convention="ros"),
        )
        # Red debug sphere at cam_wrist pose (follows the gripper).
        self.scene.cam_wrist_viz = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Robot/" + EE_BODY_NAME + "/cam_wrist/viz",
            spawn=sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        )

        self.observations.images = ImagesCfg()


@configclass
class SO101CubeLiftEnvCfg_PLAY(SO101CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
