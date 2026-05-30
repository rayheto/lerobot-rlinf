"""Inspect SO-101 USD: load as Articulation, sweep joints, optional viewport preview.

Run (headless sanity):
    /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/inspect_so101.py
Run (GUI preview on display :110):
    DISPLAY=:110 /home/hlei/miniconda3/envs/rlinf-isaacsim-env/bin/python scripts/inspect_so101.py --gui
"""
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--gui", action="store_true", help="Open viewport window")
parser.add_argument("--hold", type=float, default=2.0,
                    help="Seconds to dwell at each sweep target (GUI only)")
parser.add_argument("--pose", action="store_true",
                    help="Skip sweep; open a joint-slider panel to dial in a ready pose (GUI only)")
args = parser.parse_args()

# SimulationApp forwards leftover sys.argv to Kit, which rejects unknown flags.
# Strip our own argparse flags so Kit only sees an empty argv tail.
sys.argv = sys.argv[:1]

from isaacsim import SimulationApp

app = SimulationApp({"headless": not args.gui, "width": 1280, "height": 720})

import numpy as np
from isaacsim.core.api import World
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.stage import add_reference_to_stage

REPO_ROOT = Path(__file__).resolve().parent.parent
USD_PATH = REPO_ROOT / "assets/so_arm100/SO101/so101_new_calib.usd"
PRIM_PATH = "/World/so101"

world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()
add_reference_to_stage(usd_path=str(USD_PATH), prim_path=PRIM_PATH)
robot = world.scene.add(SingleArticulation(prim_path=PRIM_PATH, name="so101"))

world.reset()

print(f"[verify] dof_names ({robot.num_dof}):")
for i, name in enumerate(robot.dof_names):
    print(f"  [{i}] {name}")

# Joint limits sourced from the URDF parse (5.1 SingleArticulation has no
# direct get_joint_limits — limits live in the PhysX dof_properties view).
lower = np.array([-1.920, -1.745, -1.690, -1.658, -2.744, -0.175], dtype=np.float32)
upper = np.array([+1.920, +1.745, +1.690, +1.658, +2.841, +1.745], dtype=np.float32)
mid = (lower + upper) / 2

physics_dt = world.get_physics_dt()
hold_frames = max(60, int(args.hold / physics_dt)) if args.gui else 60


if args.pose and args.gui:
    import threading

    dof_names = list(robot.dof_names)
    current = np.asarray(robot.get_joint_positions()).flatten().astype(np.float32)
    state_lock = threading.Lock()
    stop_flag = threading.Event()

    def _stdin_loop():
        """Read poses from stdin on a worker thread; main thread drives the sim."""
        print("\n=== SO-101 pose REPL ===", flush=True)
        print(f"DoF order: {dof_names}", flush=True)
        print("Type 6 floats (space-separated), or one of:", flush=True)
        print("  print     — dump current pose as joint_pos dict", flush=True)
        print("  reset     — back to zeros", flush=True)
        print("  q / quit  — exit\n", flush=True)
        for lo, hi, n in zip(lower, upper, dof_names):
            print(f"  {n:15s}  [{lo:+.3f}, {hi:+.3f}]", flush=True)
        print("", flush=True)
        while not stop_flag.is_set():
            try:
                line = input("pose> ").strip()
            except EOFError:
                stop_flag.set()
                return
            if not line:
                continue
            if line in ("q", "quit", "exit"):
                stop_flag.set()
                return
            if line == "print":
                with state_lock:
                    vals = [round(float(v), 4) for v in current]
                pairs = ", ".join(f'"{n}": {v}' for n, v in zip(dof_names, vals))
                print(f"[pose] joint_pos={{{pairs}}}", flush=True)
                continue
            if line == "reset":
                with state_lock:
                    current[:] = 0.0
                continue
            try:
                vals = [float(x) for x in line.split()]
            except ValueError:
                print("[pose] parse error — expected 6 floats", flush=True)
                continue
            if len(vals) != 6:
                print(f"[pose] expected 6 values, got {len(vals)}", flush=True)
                continue
            clipped = np.clip(np.array(vals, dtype=np.float32), lower, upper)
            with state_lock:
                current[:] = clipped
            print(f"[pose] applied {clipped.tolist()}", flush=True)

    t = threading.Thread(target=_stdin_loop, daemon=True)
    t.start()
    try:
        while app.is_running() and not stop_flag.is_set():
            with state_lock:
                tgt = current.copy().reshape(1, -1)
            robot.set_joint_positions(tgt)
            world.step(render=True)
    except KeyboardInterrupt:
        stop_flag.set()
else:
    print(f"[verify] sweep ({hold_frames} frames per target, render={args.gui})")
    for label, target in [("mid", mid), ("lower", lower), ("upper", upper), ("mid", mid)]:
        tgt = target.reshape(1, -1)
        robot.set_joint_positions(tgt)
        for _ in range(hold_frames):
            world.step(render=args.gui)
        achieved = np.asarray(robot.get_joint_positions()).flatten()
        err = float(np.abs(achieved - target).max())
        print(f"  {label:5s} target={np.round(target,2)} achieved={np.round(achieved,2)} max_err={err:.4f}")

    if args.gui:
        print("[verify] idle render loop — Ctrl+C to exit")
        try:
            while app.is_running():
                world.step(render=True)
        except KeyboardInterrupt:
            pass

print("[verify] OK")
app.close()
