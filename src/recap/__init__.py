"""RECAP: Real-robot inference verification, human correction, and data export.

Submodules:
  contracts            -- single source of truth for joints, units, products
  state_machine        -- session/episode/pause/freeze/intervention authority
  data_recorder        -- SQLite index + raw data recording
  hook_server          -- WebSocket server for direct Hook link to Rebot-Arm
  lerobot_v3_exporter  -- LeRobot v3 dataset export from recorded sessions
  fake_robot           -- fake SO-101 robot + cameras + policy for testing
"""
from __future__ import annotations

from .contracts import (
    B601_PRODUCT,
    SO101_PRODUCT,
    CameraDef,
    JointDef,
    JointUnit,
    ProductContract,
    PRODUCT_REGISTRY,
    all_products,
    get_product,
)
from .state_machine import (
    ControlSource,
    FreezeState,
    FreezeTarget,
    InterventionWindow,
    SessionState,
    StateEvent,
    StateMachine,
)
from .data_recorder import DataRecorder, TickRecord
from .hook_server import HookServer, PROTOCOL_VERSION
from .lerobot_v3_exporter import LeRobotV3Exporter
from .fake_robot import FakeRobot, FakeCamera, FakePolicy, build_fake_observation

__all__ = [
    "B601_PRODUCT",
    "SO101_PRODUCT",
    "CameraDef",
    "JointDef",
    "JointUnit",
    "ProductContract",
    "PRODUCT_REGISTRY",
    "all_products",
    "get_product",
    "ControlSource",
    "FreezeState",
    "FreezeTarget",
    "InterventionWindow",
    "SessionState",
    "StateEvent",
    "StateMachine",
    "DataRecorder",
    "TickRecord",
    "HookServer",
    "PROTOCOL_VERSION",
    "LeRobotV3Exporter",
    "FakeRobot",
    "FakeCamera",
    "FakePolicy",
    "build_fake_observation",
]
