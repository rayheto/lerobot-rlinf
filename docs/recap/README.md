# RECAP: Real-Robot Inference Verification, Correction, and Data Export

## Quick Start

### 1. Run the real backend with RECAP enabled

```bash
# In the openpi venv
third_party/openpi/.venv/Scripts/python.exe src/real_backend.py \
  --config configs/real_so101.example.yaml \
  --policy-host localhost --policy-port 8000 \
  --recap --recap-data-dir outputs/recap/test_session
```

### 2. Start the Rebot-Arm visualizer

```bash
cd third_party/Rebot-Arm/reBotArm_simulator
npm start
# Open http://localhost:3001
```

### 3. Connect the Direct Hook

In the Rebot-Arm web UI, find the 'Direct Hook (RECAP)' section in the right
panel, enter `ws://localhost:8765`, and click Connect.

## Configuration

Add to `configs/real_so101.example.yaml`:

```yaml
recap:
  enabled: true
  data_dir: outputs/recap/my_session
  hook_host: 0.0.0.0
  hook_port: 8765
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the cross-repo diagram and data flow.

## Protocol

See [PROTOCOL.md](PROTOCOL.md) for the direct-realtime WebSocket protocol.

## Data Recording

See [DATA_ADR.md](DATA_ADR.md) for the SQLite + raw blobs design.

## Safety

See [SAFETY.md](SAFETY.md) for failure modes and verification checklist.

## RECAP Schema

See [RECAP_SCHEMA.md](RECAP_SCHEMA.md) for intervention and annotation schema.

## Testing

```bash
# Run all tests (28 tests)
third_party/openpi/.venv/Scripts/python.exe -m unittest discover -s tests -p 'test_*.py' -v
```

## Key Files

| File | Description |
|------|-------------|
| `src/recap/contracts.py` | Single source of truth for joints, units, products |
| `src/recap/state_machine.py` | Session/episode/pause/freeze/intervention state machine |
| `src/recap/data_recorder.py` | SQLite index + raw data recording |
| `src/recap/hook_server.py` | WebSocket server for Rebot-Arm direct Hook |
| `src/recap/lerobot_v3_exporter.py` | LeRobot v3 dataset export |
| `src/recap/fake_robot.py` | Fake robot/camera/policy for testing |
| `src/real_backend.py` | Integrated control loop with RECAP |
| `tests/test_recap_state_machine.py` | 22 state machine tests |
| `tests/test_recap_e2e.py` | 3 end-to-end pipeline tests |
