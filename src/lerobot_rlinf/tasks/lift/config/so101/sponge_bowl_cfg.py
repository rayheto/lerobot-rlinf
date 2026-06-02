"""SO-101 sponge-pick-place-bowl env — visually aligned to
aswinkumar99/LeRobot-SO101-task1-single-sponge-no-distractors-random-locations.

What this variant changes vs SO101CubeLiftEnvCfg (zero touch to base):
- Object: DexCube → small blue cuboid (sponge proxy, ~3×4×2.5cm).
- Camera names: front/wrist → overhead/wrist (matches dataset feature keys
  exactly, so the LeRobot wrapper does no key renaming on the image side).
- Camera pose: top-down → front-elevated with downtilt (matches dataset's
  "overhead" view, which is actually an eye-level front camera).
- Target: random pose command → freeze command to bowl's drop-in position,
  so the existing object_goal_tracking reward chain becomes "place in bowl"
  with zero reward refactor.
- Bowl: added as a static AssetBaseCfg (serving_bowl.usd from Isaac Sim
  extscache test assets — local-only, no Nucleus required).

Action/state semantics (degrees, URDF order) inherited unchanged.
"""

import math
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import CameraCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass

from lerobot_rlinf.tasks.lift import mdp

from .joint_pos_env_cfg import SO101CubeLiftEnvCfg


# Bowl USD shipped inside the isaacsim extension cache (omni.tools.array test
# data). Local-only; no Nucleus pull needed. Path is conda-env specific —
# replace with a vendored copy if we ever package the project.
_SERVING_BOWL_USD = (
    "/home/hlei/miniconda3/envs/rlinf-isaacsim-env/lib/python3.11/"
    "site-packages/isaacsim/extscache/omni.tools.array-107.0.0/"
    "omni/tools/array/tests/data/Collected_serving_bowl/serving_bowl.usd"
)
assert Path(_SERVING_BOWL_USD).exists(), f"serving_bowl.usd not found at {_SERVING_BOWL_USD}"

# Bowl pose in robot-base frame. Right of arm, in front, on table.
# Matches dataset overhead view: bowl on the right edge of workspace.
_BOWL_POS = (0.30, -0.20, 0.0)

# Sponge spawn pose (initial; per-env randomization handled by base EventCfg).
_SPONGE_POS = (0.25, 0.10, 0.025)


@configclass
class OverheadWristImagesCfg(ObsGroup):
    """observation.images.{overhead,wrist} — names match dataset keys 1:1."""

    overhead = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("overhead_cam"), "data_type": "rgb", "normalize": False},
    )
    wrist = ObsTerm(
        func=mdp.image,
        params={"sensor_cfg": SceneEntityCfg("wrist_cam"), "data_type": "rgb", "normalize": False},
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class SO101SpongeBowlEnvCfg(SO101CubeLiftEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # --- Robot home pose: match dataset frame-0 distribution ---
        # Mean of `observation.state[frame_index==0]` across 64 episodes of
        # aswinkumar99/.../single-sponge-no-distractors. URDF zero pose is
        # OOD (gripper 45.8° >> dataset q99=29°, shoulder_lift 0° vs -103°)
        # so resetting there causes the policy to spend the first ~10 steps
        # just driving the arm into its trained home before any grasp action.
        # Note: shoulder_lift clamped to -99.5° (URDF limit is ±100°, dataset
        # operates slightly outside the URDF-stated limit — real hardware
        # range extends further than the kinematic model declares).
        _d2r = math.pi / 180.0
        self.scene.robot.init_state.joint_pos = {
            "shoulder_pan":   -3.16 * _d2r,
            "shoulder_lift":  -99.5 * _d2r,
            "elbow_flex":     96.02 * _d2r,
            "wrist_flex":     79.35 * _d2r,
            "wrist_roll":     -9.70 * _d2r,
            "gripper":         3.27 * _d2r,
        }

        # --- Object: DexCube → blue sponge proxy (small cuboid) ---
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=list(_SPONGE_POS), rot=[1, 0, 0, 0]),
            spawn=sim_utils.CuboidCfg(
                size=(0.04, 0.03, 0.025),  # ~real blue sponge in dataset
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
                mass_props=sim_utils.MassPropertiesCfg(mass=0.01),  # sponge is light
                collision_props=sim_utils.CollisionPropertiesCfg(),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.20, 0.45, 0.85)),
            ),
        )

        # --- Bowl: static USD asset on the right of workspace ---
        self.scene.bowl = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/Bowl",
            init_state=AssetBaseCfg.InitialStateCfg(pos=list(_BOWL_POS), rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(usd_path=_SERVING_BOWL_USD, scale=(1.0, 1.0, 1.0)),
        )

        # --- Camera rename + repose: front (top-down) → overhead (front-elevated) ---
        del self.scene.front_cam
        del self.scene.front_cam_viz
        # Camera behind+above the arm, looking forward-down (~35° downtilt).
        # Pose tuned visually to match dataset overhead view; refine after smoke.
        # ROS convention (cam +Z = forward). Quat = rotate +Z by 125° about Y:
        # (w,x,y,z) = (cos(62.5°), 0, sin(62.5°), 0) ≈ (0.462, 0, 0.887, 0).
        self.scene.overhead_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/overhead_cam",
            update_period=0.0,
            width=224,
            height=224,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.05, 5.0)
            ),
            offset=CameraCfg.OffsetCfg(pos=(-0.35, 0.0, 0.45), rot=(0.462, 0.0, 0.887, 0.0), convention="ros"),
        )
        self.scene.overhead_cam_viz = AssetBaseCfg(
            prim_path="{ENV_REGEX_NS}/overhead_cam/viz",
            spawn=sim_utils.SphereCfg(
                radius=0.02,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        )

        # --- Swap images group: now keyed overhead/wrist instead of front/wrist ---
        self.observations.images = OverheadWristImagesCfg()

        # --- Freeze the goal command to bowl drop-in pose (above bowl center) ---
        # Existing object_goal_tracking reward chain (std=0.3 + std=0.05 fine)
        # now scores "place at bowl" without any reward-side change.
        drop_x, drop_y, drop_z = _BOWL_POS[0], _BOWL_POS[1], _BOWL_POS[2] + 0.08
        self.commands.object_pose.ranges.pos_x = (drop_x, drop_x)
        self.commands.object_pose.ranges.pos_y = (drop_y, drop_y)
        self.commands.object_pose.ranges.pos_z = (drop_z, drop_z)

        # --- Disable command goal-pose viz (debug triad + red sphere) ---
        # Default is True in lift_env_cfg; with overhead_cam looking at the
        # bowl area, those markers get baked into the RGB obs and the policy
        # learns "go where the green arrow points" — classic visual shortcut.
        self.commands.object_pose.debug_vis = False

        # --- Tighten sponge spawn range to avoid clipping into bowl ---
        # Base EventCfg samples object pose ± from init; constrain to LEFT half.
        self.events.reset_object_position.params["pose_range"] = {
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.15),  # bias positive-Y (left of arm) → away from bowl
            "z": (0.0, 0.0),
        }


@configclass
class SO101SpongeBowlEnvCfg_PLAY(SO101SpongeBowlEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.scene.env_spacing = 1.5
        self.observations.policy.enable_corruption = False
