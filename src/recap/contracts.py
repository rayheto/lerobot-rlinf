"""Single source of truth for joint names, units, limits, and product definitions.

Both lerobot-rlinf and Rebot-Arm reference this contract so that joint names,
units, and limits are never hardcoded in three places (control loop, UI, recorder).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class JointUnit(str, Enum):
    RADIAN = "rad"
    DEGREE = "deg"
    METER = "m"


@dataclass(frozen=True)
class JointDef:
    """Definition of a single controllable joint."""

    name: str
    label: str
    min: float
    max: float
    home: float
    unit: JointUnit = JointUnit.RADIAN

    def clamp(self, value: float) -> float:
        return max(self.min, min(self.max, value))

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "min": self.min,
            "max": self.max,
            "home": self.home,
            "unit": self.unit.value,
        }


@dataclass(frozen=True)
class CameraDef:
    """Definition of a camera stream."""

    key: str
    label: str
    width: int = 640
    height: int = 480
    fps: int = 30

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "label": self.label,
            "width": self.width,
            "height": self.height,
            "fps": self.fps,
        }


@dataclass(frozen=True)
class ProductContract:
    """Full product definition: joints, cameras, URDF, gripper, defaults.

    This is the contract that the control loop, the UI, and the recorder all
    reference.  Adding a new product means adding a new instance here (or
    registering it via the Rebot-Arm product registry JSON).
    """

    product_id: str
    name: str
    joints: tuple[JointDef, ...]
    cameras: tuple[CameraDef, ...] = ()
    urdf_path: str | None = None
    mesh_dir: str | None = None
    gripper_joint: str | None = None
    gripper_closed: float = 0.0
    gripper_open: float = 0.09
    default_camera_view: dict[str, float] = field(
        default_factory=lambda: {"az": -0.5, "el": 0.5, "dist": 1.2}
    )

    @property
    def joint_names(self) -> tuple[str, ...]:
        return tuple(j.name for j in self.joints)

    @property
    def joint_limits(self) -> dict[str, tuple[float, float]]:
        return {j.name: (j.min, j.max) for j in self.joints}

    def joint_by_name(self, name: str) -> JointDef | None:
        for j in self.joints:
            if j.name == name:
                return j
        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "product_id": self.product_id,
            "name": self.name,
            "joints": [j.to_dict() for j in self.joints],
            "cameras": [c.to_dict() for c in self.cameras],
            "urdf_path": self.urdf_path,
            "mesh_dir": self.mesh_dir,
            "gripper_joint": self.gripper_joint,
            "gripper_closed": self.gripper_closed,
            "gripper_open": self.gripper_open,
            "default_camera_view": dict(self.default_camera_view),
        }


# ---------------------------------------------------------------------------
# SO-101 (LeRobot motor-degree convention, 6 DOF + gripper)
# ---------------------------------------------------------------------------

_SO101_MOTOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

_SO101_JOINT_LIMITS_DEG = {
    "shoulder_pan": (-90.0, 90.0),
    "shoulder_lift": (-180.0, 0.0),
    "elbow_flex": (-150.0, 0.0),
    "wrist_flex": (-90.0, 90.0),
    "wrist_roll": (-180.0, 180.0),
    "gripper": (0.0, 100.0),
}

SO101_PRODUCT = ProductContract(
    product_id="so101",
    name="SO-101",
    joints=tuple(
        JointDef(
            name=n,
            label=n.replace("_", " ").title(),
            min=_SO101_JOINT_LIMITS_DEG[n][0],
            max=_SO101_JOINT_LIMITS_DEG[n][1],
            home=0.0,
            unit=JointUnit.DEGREE,
        )
        for n in _SO101_MOTOR_NAMES
    ),
    cameras=(
        CameraDef(key="front", label="Front", width=640, height=480, fps=30),
        CameraDef(key="wrist", label="Wrist", width=640, height=480, fps=30),
    ),
    urdf_path=None,
    mesh_dir=None,
    gripper_joint="gripper",
    gripper_closed=0.0,
    gripper_open=100.0,
    default_camera_view={"az": -0.6, "el": 0.35, "dist": 1.0},
)

# ---------------------------------------------------------------------------
# B601-DM (Rebot-Arm, radian convention, 6 DOF + gripper)
# Joint limits match rebot-sim.js jointDefs.
# ---------------------------------------------------------------------------

B601_PRODUCT = ProductContract(
    product_id="b601_dm",
    name="reBot Arm B601-DM",
    joints=(
        JointDef("joint1", "J1 Base Yaw", -2.8, 2.8, 0.0),
        JointDef("joint2", "J2 Shoulder", -3.14, 0.0, 0.0),
        JointDef("joint3", "J3 Elbow", -3.14, 0.0, 0.0),
        JointDef("joint4", "J4 Wrist Pitch", -1.87, 1.57, 0.0),
        JointDef("joint5", "J5 Wrist Yaw", -1.57, 1.57, 0.0),
        JointDef("joint6", "J6 Tool Roll", -3.14, 3.14, 0.0),
        JointDef("gripper", "J7 Gripper", 0.0, 0.09, 0.0, JointUnit.METER),
    ),
    cameras=(
        CameraDef(key="front", label="Front", width=640, height=480, fps=30),
        CameraDef(key="wrist", label="Wrist", width=640, height=480, fps=30),
    ),
    urdf_path="/api/urdf",
    mesh_dir="/api/description/meshes",
    gripper_joint="gripper",
    gripper_closed=0.0,
    gripper_open=0.09,
    default_camera_view={"az": -0.5, "el": 0.5, "dist": 1.2},
)

PRODUCT_REGISTRY: dict[str, ProductContract] = {
    SO101_PRODUCT.product_id: SO101_PRODUCT,
    B601_PRODUCT.product_id: B601_PRODUCT,
}


def get_product(product_id: str) -> ProductContract | None:
    """Look up a product by id. Returns None if not found."""
    return PRODUCT_REGISTRY.get(product_id)


def all_products() -> list[ProductContract]:
    return list(PRODUCT_REGISTRY.values())
