import unittest
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from real_backend import ActionPostProcessor, _action_dict, _stale_prefix_steps


class RealBackendControlHelpersTest(unittest.TestCase):
    def test_stale_prefix_steps_floor_and_clamp(self):
        self.assertEqual(_stale_prefix_steps(0.0, 1.0 / 30.0, 10), 0)
        self.assertEqual(_stale_prefix_steps(0.099, 1.0 / 30.0, 10), 2)
        self.assertEqual(_stale_prefix_steps(9.0, 1.0 / 30.0, 10), 10)

    def test_action_dict_requires_six_dof(self):
        action = np.arange(6, dtype=np.float64)
        result = _action_dict(action)
        self.assertEqual(set(result), {
            "shoulder_pan.pos",
            "shoulder_lift.pos",
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos",
        })
        with self.assertRaises(RuntimeError):
            _action_dict(np.arange(5, dtype=np.float64))

    def test_ema_smoothing_moves_toward_latest_action(self):
        smoother = ActionPostProcessor(mode="ema", step_hz=30.0, ema_tau_s=0.12)
        first = smoother.apply(np.zeros(6, dtype=np.float64))
        second = smoother.apply(np.ones(6, dtype=np.float64))
        self.assertTrue(np.allclose(first, np.zeros(6)))
        self.assertTrue(np.all(second > 0.0))
        self.assertTrue(np.all(second < 1.0))


if __name__ == "__main__":
    unittest.main()
