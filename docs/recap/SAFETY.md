# Real-Robot Safety and Failure Modes

## Conservative Defaults

- All RECAP features are OFF by default. Existing real_backend works unchanged.
- `max_relative_target: 10` (from existing config) clips large jumps.
- `calibrate: false` in example config (user must explicitly enable).
- Hook server binds to `0.0.0.0` but only when `recap.enabled: true`.

## Failure Modes

| Failure | Behavior | Mitigation |
|---------|----------|------------|
| Policy server disconnect | Control loop stops, robot holds last position | `max_consecutive_failures` |
| Hook server crash | Control loop continues, UI shows offline | Separate thread, bounded queue |
| UI disconnect | Control loop continues | Latest-value, no blocking |
| Disk full | Recorder drops + flags, control continues | Bounded queue, drop counter |
| Camera freeze without frame | NACK returned, no freeze | Explicit `freeze_nack` event |
| Intervention without confirm | Robot stays in RESUME_PENDING | No auto-resume |

## Real-Robot Verification Checklist

1. Test with `dry_run: true` first - no commands sent to arm.
2. Verify `max_relative_target` clips unexpected jumps.
3. Test pause/resume with hand near e-stop.
4. Test intervention with low speeds.
5. Verify Hook server does not block control loop (check `loop_dt_ms`).
6. Verify LeRobot v3 export matches recording (frame count, timestamps).
