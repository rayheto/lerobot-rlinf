# RECAP Architecture: Cross-Repo Real-Robot Inference Verification & Correction

## 1. Current State Audit

### lerobot-rlinf (branch: main, HEAD: 22d8863)
- `real_backend.py` (1070 lines): SO-101 control, front/wrist OpenCV cameras,
  sync/async action-chunk execution, WebSocket policy client, JSONL telemetry,
  policy-input JPEG recording. User has uncommitted `_warmup_policy` addition (preserved).
- `eval.py` / `eval_run.py`: Isaac Sim two-venv orchestrator; writes LeRobot **v2.1** layout.
- LeRobot **0.4.4** installed in openpi venv (supports Dataset v3 reader/writer).
- `tests/test_real_backend.py`: 3 unit tests (stale-prefix, action_dict, EMA).
- **Gaps**: no session/episode model, no pause/freeze, no control-authority
  arbitration, no human-correction, no raw wire-response recording, no per-tick
  state capture, no LeRobot v3 real-robot export, no direct-Hook server.

### Rebot-Arm (branch: main, HEAD includes `4d19f06`)
- `server.js`: Node.js static file server, product registry endpoints, B601-DM
  model routes, and SO-101 lazy model cache.
- `/api/config?product=...`: product definition lookup for B601-DM and SO-101.
- `/api/so101/urdf`: ensures the real SO-101 model cache exists, then serves
  `so101_new_calib.urdf`.
- `/api/so101/assets/...`: serves cached SO-101 STL mesh assets.
- `rebot-sim.js`: Three.js + URDFLoader, product selector, product-specific
  joint sliders, model reload, TCP drag, teach record/replay, forward-sim
  command dispatch.
- `direct-hook-client.js` / `direct-hook-ui.js`: Direct Hook client and RECAP
  controls for pause/resume, camera freeze, intervention, and session status.
- `rebot-ros-client.js` / `rebot-ros-ui.js`: rosbridge WebSocket client.
- CSS: dark workbench, teal `#33d6b0`, amber `#f2a541`, red `#ef5a4d`,
  360 px right panel, 7-8 px radius.
- Runtime cache: `reBotArm_simulator/cache/so101-model/` stores downloaded
  SO-101 URDF/STL assets from `TheRobotStudio/SO-ARM100`; it is ignored by Git.
- **Remaining gaps**: no Direct Hook session query/history browser in the UI.

## 2. Cross-Repo Architecture

See data flow below. lerobot-rlinf is the real-time control + data source.
Rebot-Arm is the product model, visualization, human operation, and session query entry.

## 3. Data Flow

1. **Observation**: robot.get_observation() -> joint state + front/wrist frames.
2. **Policy request**: policy_obs built from obs -> WebSocket infer -> raw wire response.
3. **Recording**: every tick records (tick_id, monotonic_ts, joint_state, raw_action,
   executed_action, front_frame, wrist_frame, policy_obs_mapping, wire_response,
   state_event) into SQLite index + binary blobs.
4. **Hook broadcast**: HookServer pushes latest joint state + camera frames +
   session status to connected Rebot-Arm clients (bounded queue, latest-value).
5. **Intervention**: human takes control via Rebot-Arm -> HookServer receives
   human actions -> state machine switches authority -> policy actions shadowed
   (recorded, not executed) -> human actions executed + flagged.
6. **Export**: DataRecorder -> LeRobot v3 exporter -> Parquet + MP4 + meta,
   readable by official LeRobot loader.

## 4. Key Design Decisions

- **SQLite index + raw blobs**: SQLite for queryable metadata (intervention
  windows, tick alignment, drop/error flags); raw frames as MP4 (video) and
  Parquet (arrays) for space efficiency. See `DATA_ADR.md`.
- **Single contract**: `recap/contracts.py` is the single source of truth for
  joint names, units, limits, and product definitions. Both repos reference it.
- **Bounded queues**: all Hook/IO paths use bounded queues with latest-value
  eviction to never block the 30 Hz control loop.
- **Conservative defaults**: new features OFF by default; existing real_backend
  and B601 page work unchanged when features are disabled.
- **Lazy robot assets**: SO-101 real visualization assets are runtime data. The
  browser blocks on the existing Loading screen while the local Node server
  downloads missing URDF/STL files, then later page loads use the local cache.
