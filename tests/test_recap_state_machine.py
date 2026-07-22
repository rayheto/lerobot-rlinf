"""Tests for the RECAP state machine: pause/freeze/intervention authority."""
import unittest

import numpy as np

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from recap.state_machine import (
    ControlSource,
    FreezeTarget,
    SessionState,
    StateMachine,
)


class TestSessionLifecycle(unittest.TestCase):
    def test_start_and_stop(self):
        sm = StateMachine()
        self.assertEqual(sm.state, SessionState.IDLE)
        evs = sm.start_session("s1", "prompt")
        self.assertEqual(len(evs), 1)
        self.assertEqual(evs[0].event, "session_start")
        self.assertEqual(sm.state, SessionState.RUNNING)
        self.assertEqual(sm.control_source, ControlSource.POLICY)
        evs = sm.stop_session()
        self.assertEqual(sm.state, SessionState.STOPPED)
        self.assertTrue(any(e.event == "session_stop" for e in evs))

    def test_double_start_ignored(self):
        sm = StateMachine()
        sm.start_session("s1")
        evs = sm.start_session("s2")
        self.assertEqual(evs, [])
        self.assertEqual(sm.session_id, "s1")

    def test_next_episode(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.advance_tick()
        evs = sm.next_episode()
        self.assertEqual(evs[0].event, "episode_start")
        self.assertEqual(sm.episode_id, 1)
        self.assertEqual(sm.tick, 0)


class TestInferencePause(unittest.TestCase):
    def test_pause_stops_requests(self):
        sm = StateMachine()
        sm.start_session("s1")
        self.assertTrue(sm.should_request_inference())
        sm.pause_inference()
        self.assertFalse(sm.should_request_inference())
        self.assertEqual(sm.state, SessionState.PAUSED)

    def test_resume_replans(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.pause_inference()
        evs = sm.resume_inference()
        self.assertEqual(evs[0].event, "inference_resume")
        self.assertTrue(evs[0].payload["replan"])
        self.assertEqual(sm.state, SessionState.RUNNING)
        self.assertTrue(sm.should_request_inference())

    def test_paused_response_ignored(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.pause_inference()
        self.assertFalse(sm.should_execute_policy_action())
        evs = sm.mark_response_ignored(42, 7)
        self.assertEqual(evs[0].event, "response_ignored")
        self.assertEqual(evs[0].payload["request_id"], 42)

    def test_double_pause_idempotent(self):
        sm = StateMachine()
        sm.start_session("s1")
        evs1 = sm.pause_inference()
        self.assertEqual(len(evs1), 1)
        evs2 = sm.pause_inference()
        self.assertEqual(evs2, [])


class TestCameraFreeze(unittest.TestCase):
    def test_freeze_with_frame(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        ok, evs = sm.freeze_camera(FreezeTarget.FRONT, frame, "frame_001")
        self.assertTrue(ok)
        self.assertEqual(evs[0].event, "freeze_start")
        self.assertEqual(evs[0].payload["source_frame_id"], "frame_001")
        self.assertTrue(sm.is_frozen(FreezeTarget.FRONT))

    def test_freeze_returns_captured_frame(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame1 = np.ones((480, 640, 3), dtype=np.uint8) * 10
        frame2 = np.ones((480, 640, 3), dtype=np.uint8) * 20
        sm.freeze_camera(FreezeTarget.FRONT, frame1, "f1")
        out, is_frozen = sm.get_camera_frame(FreezeTarget.FRONT, frame2, "f2")
        self.assertTrue(is_frozen)
        np.testing.assert_array_equal(out, frame1)

    def test_wrist_freeze_independent(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sm.freeze_camera(FreezeTarget.FRONT, frame, "f1")
        self.assertTrue(sm.is_frozen(FreezeTarget.FRONT))
        self.assertFalse(sm.is_frozen(FreezeTarget.WRIST))
        out_w, frozen_w = sm.get_camera_frame(FreezeTarget.WRIST, frame, "f2")
        self.assertFalse(frozen_w)

    def test_freeze_without_frame_nack(self):
        sm = StateMachine()
        sm.start_session("s1")
        ok, evs = sm.freeze_camera(FreezeTarget.FRONT, None, "")
        self.assertFalse(ok)
        self.assertEqual(evs[0].event, "freeze_nack")
        self.assertEqual(evs[0].payload["reason"], "no_first_frame")

    def test_unfreeze(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sm.freeze_camera(FreezeTarget.FRONT, frame, "f1")
        evs = sm.unfreeze_camera(FreezeTarget.FRONT)
        self.assertEqual(evs[0].event, "freeze_stop")
        self.assertFalse(sm.is_frozen(FreezeTarget.FRONT))

    def test_both_freeze_simultaneous(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sm.freeze_camera(FreezeTarget.FRONT, frame, "f1")
        sm.freeze_camera(FreezeTarget.WRIST, frame, "f2")
        self.assertTrue(sm.is_frozen(FreezeTarget.FRONT))
        self.assertTrue(sm.is_frozen(FreezeTarget.WRIST))
        sm.unfreeze_camera(FreezeTarget.FRONT)
        self.assertFalse(sm.is_frozen(FreezeTarget.FRONT))
        self.assertTrue(sm.is_frozen(FreezeTarget.WRIST))


class TestIntervention(unittest.TestCase):
    def test_intervention_switches_authority(self):
        sm = StateMachine()
        sm.start_session("s1")
        evs = sm.start_intervention()
        self.assertEqual(sm.state, SessionState.INTERVENING)
        self.assertEqual(sm.control_source, ControlSource.HUMAN)
        self.assertFalse(sm.should_execute_policy_action())
        self.assertEqual(evs[-1].event, "intervention_start")
        self.assertEqual(evs[-1].payload["intervention_id"], 1)

    def test_intervention_does_not_auto_resume(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.start_intervention()
        evs = sm.end_intervention()
        self.assertEqual(sm.state, SessionState.RESUME_PENDING)
        self.assertEqual(evs[0].event, "intervention_end")
        # Without confirm, should not request inference
        self.assertFalse(sm.should_request_inference())

    def test_confirm_resume(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.start_intervention()
        sm.end_intervention()
        evs = sm.confirm_resume()
        self.assertEqual(sm.state, SessionState.RUNNING)
        self.assertEqual(evs[0].event, "resume_confirmed")
        self.assertTrue(sm.should_request_inference())

    def test_intervention_ids_continuous(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.start_intervention()
        iid1 = sm.current_intervention.intervention_id
        sm.end_intervention()
        sm.confirm_resume()
        sm.start_intervention()
        iid2 = sm.current_intervention.intervention_id
        self.assertEqual(iid2, iid1 + 1)

    def test_shadow_policy_records_not_executes(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.start_intervention(shadow_policy=True)
        # Shadow mode: inference continues but actions not executed
        self.assertFalse(sm.should_execute_policy_action())
        sm.record_shadow_policy_action()
        iv = sm.current_intervention
        self.assertEqual(iv.policy_action_count, 1)

    def test_human_action_counted(self):
        sm = StateMachine()
        sm.start_session("s1")
        sm.start_intervention()
        sm.record_human_action()
        sm.record_human_action()
        iv = sm.current_intervention
        self.assertEqual(iv.human_action_count, 2)

    def test_intervention_window_before_during_after(self):
        sm = StateMachine()
        sm.start_session("s1")
        for _ in range(5):
            sm.advance_tick()
        sm.start_intervention()
        for _ in range(3):
            sm.advance_tick()
            sm.record_human_action()
        sm.end_intervention()
        for _ in range(2):
            sm.advance_tick()
        iv = sm.interventions[0]
        self.assertEqual(iv.start_tick, 5)
        self.assertEqual(iv.end_tick, 8)
        self.assertEqual(iv.human_action_count, 3)
        self.assertFalse(iv.confirmed)


class TestSnapshot(unittest.TestCase):
    def test_snapshot_roundtrip(self):
        sm = StateMachine()
        sm.start_session("s1", "grab")
        sm.pause_inference()
        snap = sm.snapshot()
        self.assertEqual(snap["state"], "paused")
        self.assertEqual(snap["session_id"], "s1")
        self.assertTrue(snap["inference_paused"])
        self.assertEqual(snap["control_source"], "policy")


class TestFreezePlusPause(unittest.TestCase):
    def test_freeze_and_pause_independent(self):
        sm = StateMachine()
        sm.start_session("s1")
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        sm.freeze_camera(FreezeTarget.FRONT, frame, "f1")
        sm.pause_inference()
        self.assertTrue(sm.is_frozen(FreezeTarget.FRONT))
        self.assertFalse(sm.should_request_inference())
        sm.resume_inference()
        self.assertTrue(sm.is_frozen(FreezeTarget.FRONT))
        sm.unfreeze_camera(FreezeTarget.FRONT)
        self.assertFalse(sm.is_frozen(FreezeTarget.FRONT))


if __name__ == "__main__":
    unittest.main()
