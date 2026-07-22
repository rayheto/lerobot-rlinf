# ADR: Data Recording Architecture

## Status
Accepted

## Context

The RECAP system needs to record real-robot inference sessions for:
1. Debugging and replay
2. Human correction (intervention) annotation
3. LeRobot v3 training data export

Data includes: per-tick joint states, raw/executed actions, two camera streams,
policy wire responses, state machine events, and intervention metadata.

## Decision

Use a **hybrid storage** approach:

| Data type | Storage | Rationale |
|-----------|---------|-----------|
| Tick metadata (state, action, timing) | SQLite | Queryable, ACID, fast point lookups |
| Camera frames | Per-tick JPEG files | Lossless alignment, survives export failure |
| Wire responses | MsgPack files | Raw bytes, no re-encoding |
| Parquet (action/state arrays) | Generated at export time | Columnar, compact, LeRobot-compatible |
| MP4 (video) | Generated at export time | Compressed, LeRobot-compatible |
| Events and interventions | SQLite | Queryable by tick range |

### Why not one storage?

- **All-SQLite**: BLOBs for video would bloat the DB and slow queries.
- **All-Parquet**: No efficient point queries for intervention windows.
- **All-MP4**: Cannot query individual ticks without decoding.

### Why per-tick JPEG instead of direct MP4?

- Per-tick JPEG ensures every tick is aligned even if the export step fails.
- MP4 is generated at export time from the JPEGs, so a crash during recording
  does not lose the entire video.
- Trade-off: more files, but filesystems handle this well.

## Consequences

- SQLite DB is the index; raw files are the blobs.
- Export is a separate step (not real-time).
- Disk usage is higher during recording (JPEGs), but export compresses.
- Query intervention windows via SQL.
