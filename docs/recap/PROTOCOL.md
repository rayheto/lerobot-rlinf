# Direct-Realtime Protocol v1

## Overview

The direct-realtime protocol connects the lerobot-rlinf real backend (control
source) to the Rebot-Arm web UI (visualization + human operation) over a single
WebSocket connection.  It does NOT depend on ROS.

Protocol version: `direct-realtime/v1`

## Transport

- WebSocket (ws:// or wss://)
- Default port: 8765 (configurable via `recap.hook_port`)
- Max message size: 10 MB (for JPEG frames)
- Binary messages: camera JPEG frames
- Text messages: JSON

## Message Types

### Server -> Client

`{"type": "hello", "protocol": "direct-realtime/v1", "timestamp": 1234567890.0}`
Sent on connect.  Client must verify protocol version.

`{"type": "snapshot", "state": "running", "session_id": "recap_abc123", ...}`
Full state snapshot sent on connect and on explicit request.

`{"type": "state", "state": "running", ...}`
State update broadcast at ~30 Hz.

`{"type": "joints", "joints": {"shoulder_pan": 10.5, ...}, ...}`
Joint state broadcast at ~30 Hz.

Binary: JPEG frames (front/wrist cameras, alternating).

### Client -> Server (Commands)

All commands return an ACK with the result and updated state.

Commands:
- `pause_inference` - stop sending policy requests (connections stay alive)
- `resume_inference` - resume; next action from latest observation
- `freeze` - freeze a camera (target: front or wrist)
- `unfreeze` - release camera freeze
- `start_intervention` - human takes control (shadow_policy: bool)
- `end_intervention` - end correction (does NOT auto-resume)
- `confirm_resume` - confirm resumption after intervention
- `next_episode` - advance episode counter
- `get_snapshot` - request full state snapshot
- `human_action` - record a human action during intervention

## Version Strategy

- v1 is the initial protocol. Breaking changes require a new version number.
- The `hello` message includes the protocol version; clients must verify.
- New optional fields may be added without a version bump.
