# SO-ARM100 Assets

Robot description files for SO-100 and SO-101 vendored from upstream.

## Source

- Upstream: https://github.com/TheRobotStudio/SO-ARM100
- Path: `Simulation/SO100`, `Simulation/SO101`
- Sync date: 2026-05-30

## Layout

```
SO101/
├── so101_new_calib.urdf   ← recommended (zero at mid-range)
├── so101_old_calib.urdf
├── so101_new_calib.xml    ← MuJoCo MJCF
├── so101_old_calib.xml
├── scene.xml              ← MuJoCo scene (for sim2sim reference)
├── joints_properties.xml
└── assets/                ← STL meshes

SO100/                     ← legacy variant, kept for reference
```

## Calibration

- **new_calib** (default): each joint zero = middle of its range
- **old_calib**: each joint zero = fully extended horizontal pose

Project default: `so101_new_calib.urdf`.

## Gripper note

Upstream URDFs do NOT yet reflect the LeRobot gripper convention:

- LeRobot: `0` = fully closed, `100` = fully open (linear joint)
- URDF: raw revolute joint, no normalization

The mapping is handled in our action processing layer (`actors/` or `envs/`), not in the URDF.

## URDF → USD conversion

Use `scripts/import_so101_urdf.py` inside Isaac Sim 4.2 to convert to USD. Output USD is gitignored (regenerate locally).

## License

Upstream license applies — see https://github.com/TheRobotStudio/SO-ARM100.
