"""End-to-end RECAP test with fake robot/camera/policy.

Exercises the full pipeline:
  fake robot + cameras -> control loop -> state machine -> data recorder
  -> pause/freeze/intervention -> LeRobot v3 export -> readback

This test does NOT require any hardware and runs in the openpi venv.
"""
import json
import sqlite3
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recap import (
    DataRecorder,
    FakeCamera,
    FakePolicy,
    FakeRobot,
    LeRobotV3Exporter,
    SO101_PRODUCT,
    StateMachine,
    TickRecord,
    build_fake_observation,
)
from recap.contracts import JointUnit
from recap.state_machine import ControlSource, FreezeTarget, SessionState


class TestFakeEndToEnd(unittest.TestCase):
    """Full pipeline test with fake components."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.sm = StateMachine()
        self.recorder = DataRecorder(self.tmpdir / "session_data")
        self.robot = FakeRobot(n_joints=6, step_hz=30)
        self.front_cam = FakeCamera("front")
        self.wrist_cam = FakeCamera("wrist")
        self.policy = FakePolicy(action_horizon=10, n_joints=6)
        self.session_id = "test_session_001"

    def tearDown(self):
        self.recorder.close()
        self.policy.close()
        if self.robot.is_connected:
            self.robot.disconnect()

    def _record_tick(self, tick_id, state, raw_action, executed_action,
                     front_frame, wrist_frame, **kwargs):
        rec = TickRecord(
            tick_id=tick_id,
            session_id=self.session_id,
            episode_id=self.sm.episode_id,
            mono_time=time.perf_counter(),
            wall_time=time.time(),
            joint_state=state,
            raw_action=raw_action,
            executed_action=executed_action,
            front_frame=front_frame,
            wrist_frame=wrist_frame,
            **kwargs,
        )
        self.recorder.record_tick(rec)

    def test_full_session_with_pause_and_intervention(self):
        """Run a fake session with pause, freeze, and intervention."""
        self.robot.connect(calibrate=True)
        evs = self.sm.start_session(self.session_id, "Grab orange and place into plate")
        for ev in evs:
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        self.recorder.set_meta("prompt", "Grab orange and place into plate")
        self.recorder.set_meta("product", "so101")

        tick = 0
        # Phase 1: normal policy running (5 ticks)
        for _ in range(5):
            obs = self.robot.get_observation()
            state = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
            front = self.front_cam.read()
            wrist = self.wrist_cam.read()
            result = self.policy.infer({"state": state})
            chunk = np.asarray(result["action"])
            action = chunk[0]
            self.robot.send_action({
                f"{n}.pos": float(v) for n, v in zip(SO101_PRODUCT.joint_names, action)
            })
            self.sm.advance_tick()
            self._record_tick(tick, state, action, action, front, wrist,
                              action_source="policy", chunk_idx=0, slot_idx=0)
            tick += 1

        self.assertEqual(self.policy.request_count, 5)
        self.assertEqual(self.robot.action_count, 5)

        # Phase 2: pause inference (3 ticks, no requests)
        self.sm.pause_inference()
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        for _ in range(3):
            obs = self.robot.get_observation()
            state = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
            front = self.front_cam.read()
            wrist = self.wrist_cam.read()
            self.sm.advance_tick()
            self._record_tick(tick, state, None, None, front, wrist,
                              action_source="none", inference_paused=True)
            tick += 1
        # No new policy requests during pause
        self.assertEqual(self.policy.request_count, 5)

        # Phase 3: resume inference (3 ticks)
        self.sm.resume_inference()
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        for _ in range(3):
            obs = self.robot.get_observation()
            state = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
            front = self.front_cam.read()
            wrist = self.wrist_cam.read()
            result = self.policy.infer({"state": state})
            chunk = np.asarray(result["action"])
            action = chunk[0]
            self.robot.send_action({
                f"{n}.pos": float(v) for n, v in zip(SO101_PRODUCT.joint_names, action)
            })
            self.sm.advance_tick()
            self._record_tick(tick, state, action, action, front, wrist,
                              action_source="policy")
            tick += 1
        self.assertEqual(self.policy.request_count, 8)

        # Phase 4: intervention (human takes control, 4 ticks)
        self.sm.start_intervention()
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        iv_id = self.sm.current_intervention.intervention_id
        self.recorder.record_intervention({
            "intervention_id": iv_id,
            "session_id": self.session_id,
            "start_tick": self.sm.tick,
            "start_mono": time.perf_counter(),
            "shadow_policy": False,
        })
        for _ in range(4):
            obs = self.robot.get_observation()
            state = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
            front = self.front_cam.read()
            wrist = self.wrist_cam.read()
            human_action = np.random.uniform(-10, 10, 6)
            self.robot.send_action({
                f"{n}.pos": float(v) for n, v in zip(SO101_PRODUCT.joint_names, human_action)
            })
            self.sm.record_human_action()
            self.sm.advance_tick()
            self._record_tick(tick, state, human_action, human_action, front, wrist,
                              action_source="human", intervention_id=iv_id)
            tick += 1

        # End intervention (does not auto-resume)
        self.sm.end_intervention()
        iv = self.sm.interventions[-1]
        self.recorder.record_intervention({
            "intervention_id": iv.intervention_id,
            "session_id": self.session_id,
            "start_tick": iv.start_tick,
            "start_mono": iv.start_mono,
            "end_tick": iv.end_tick,
            "end_mono": iv.end_mono,
            "confirmed": False,
            "shadow_policy": iv.shadow_policy,
            "human_action_count": iv.human_action_count,
            "policy_action_count": iv.policy_action_count,
        })
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        self.assertEqual(self.sm.state, SessionState.RESUME_PENDING)

        # Confirm resume
        self.sm.confirm_resume()
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)
        self.assertEqual(self.sm.state, SessionState.RUNNING)

        # Phase 5: post-intervention policy running (3 ticks)
        for _ in range(3):
            obs = self.robot.get_observation()
            state = np.asarray(obs["agent"]["qpos"], dtype=np.float64)
            front = self.front_cam.read()
            wrist = self.wrist_cam.read()
            result = self.policy.infer({"state": state})
            chunk = np.asarray(result["action"])
            action = chunk[0]
            self.robot.send_action({
                f"{n}.pos": float(v) for n, v in zip(SO101_PRODUCT.joint_names, action)
            })
            self.sm.advance_tick()
            self._record_tick(tick, state, action, action, front, wrist,
                              action_source="policy")
            tick += 1

        # Phase 6: stop session
        self.sm.stop_session()
        for ev in self.sm.drain_events():
            self.recorder.record_event(ev.event, ev.tick, ev.mono_time, ev.wall_time,
                                       self.session_id, ev.payload)

        # Wait for recorder to flush
        time.sleep(1.0)
        self.recorder.close()

        # Verify data integrity
        self.assertEqual(self.recorder.tick_count(), tick)
        self.assertGreater(self.recorder.event_count(), 0)
        self.assertEqual(self.recorder.dropped_count, 0)
        self.assertEqual(self.recorder.error_count, 0)

        # Verify intervention window query
        window = self.recorder.query_intervention_window(iv_id, before=2, after=2)
        self.assertEqual(window["intervention_id"], iv_id)
        self.assertGreater(len(window["before"]), 0)
        self.assertGreater(len(window["during"]), 0)
        self.assertGreater(len(window["after"]), 0)

        # Export to LeRobot v3
        exporter = LeRobotV3Exporter(
            db_path=self.tmpdir / "session_data" / "session.sqlite",
            output_dir=self.tmpdir / "lerobot_v3",
            fps=30,
            joint_names=list(SO101_PRODUCT.joint_names),
            camera_keys=["front", "wrist"],
        )
        summary = exporter.export()
        self.assertEqual(summary["episodes"], 1)
        self.assertEqual(summary["total_frames"], tick)
        self.assertEqual(summary["fps"], 30)

        # Verify LeRobot v3 output structure
        v3dir = self.tmpdir / "lerobot_v3"
        self.assertTrue((v3dir / "meta" / "info.json").exists())
        self.assertTrue((v3dir / "meta" / "tasks.jsonl").exists())
        self.assertTrue((v3dir / "meta" / "episodes.jsonl").exists())
        self.assertTrue((v3dir / "data" / "chunk-000" / "episode_000000.parquet").exists())

        # Read back the Parquet
        import pyarrow.parquet as pq
        table = pq.read_table(str(v3dir / "data" / "chunk-000" / "episode_000000.parquet"))
        self.assertEqual(table.num_rows, tick)
        self.assertIn("action", table.column_names)
        self.assertIn("observation.state", table.column_names)
        self.assertIn("intervention_mask", table.column_names)
        self.assertIn("action_source", table.column_names)

        # Verify intervention mask is set during intervention ticks
        iv_mask = table.column("intervention_mask").to_pylist()
        self.assertEqual(sum(iv_mask), 4)  # 4 intervention ticks

        # Verify action sources
        sources = table.column("action_source").to_pylist()
        self.assertIn("human", sources)
        self.assertIn("policy", sources)
        self.assertIn("none", sources)


class TestFreezeDuringSession(unittest.TestCase):
    """Test camera freeze during a fake session."""

    def setUp(self):
        self.tmpdir = Path(tempfile.mkdtemp())
        self.sm = StateMachine()
        self.recorder = DataRecorder(self.tmpdir / "session_data")
        self.front_cam = FakeCamera("front")
        self.wrist_cam = FakeCamera("wrist")
        self.session_id = "test_freeze_001"

    def tearDown(self):
        self.recorder.close()

    def test_front_freeze_during_session(self):
        self.sm.start_session(self.session_id)
        # Capture first frame
        frame1 = self.front_cam.read()
        ok, _ = self.sm.freeze_camera(FreezeTarget.FRONT, frame1, "f1")
        self.assertTrue(ok)

        # Subsequent frames should be frozen
        for _ in range(3):
            live = self.front_cam.read()
            frozen, is_frozen = self.sm.get_camera_frame(FreezeTarget.FRONT, live, "f2")
            self.assertTrue(is_frozen)
            np.testing.assert_array_equal(frozen, frame1)

        # Wrist should not be frozen
        wrist_live = self.wrist_cam.read()
        _, wrist_frozen = self.sm.get_camera_frame(FreezeTarget.WRIST, wrist_live, "w1")
        self.assertFalse(wrist_frozen)

        # Unfreeze
        self.sm.unfreeze_camera(FreezeTarget.FRONT)
        live = self.front_cam.read()
        _, is_frozen = self.sm.get_camera_frame(FreezeTarget.FRONT, live, "f3")
        self.assertFalse(is_frozen)


class TestDataIntegrityFlags(unittest.TestCase):
    """Test drop/error/incomplete flags are explicit, no silent success."""

    def test_queue_full_drops_recorded(self):
        tmpdir = Path(tempfile.mkdtemp())
        # Very small queue to force drops
        recorder = DataRecorder(tmpdir / "data", max_queue=2)
        sm = StateMachine()
        sm.start_session("s1")
        # Flood with ticks
        for i in range(100):
            rec = TickRecord(
                tick_id=i,
                session_id="s1",
                episode_id=0,
                mono_time=float(i),
                wall_time=time.time(),
                joint_state=np.zeros(6),
            )
            recorder.record_tick(rec)
        time.sleep(0.5)
        recorder.close()
        self.assertGreater(recorder.dropped_count, 0)


if __name__ == "__main__":
    unittest.main()
