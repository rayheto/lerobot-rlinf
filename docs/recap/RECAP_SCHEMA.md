# RECAP Intervention and Annotation Schema

## State Machine

```
IDLE -> RUNNING -> PAUSED -> RUNNING
                 -> INTERVENING -> RESUME_PENDING -> RUNNING
                 -> STOPPED
```

## Intervention Schema

Each intervention has:
- `intervention_id`: monotonic integer, continuous across session
- `start_tick` / `end_tick`: tick range
- `start_mono` / `end_mono`: monotonic timestamps
- `confirmed`: whether resume was confirmed
- `shadow_policy`: whether policy was shadowed (recorded, not executed)
- `human_action_count`: actions executed by human
- `policy_action_count`: policy actions shadowed

## Annotation Fields in LeRobot v3 Export

- `intervention_mask`: int64, 1 if tick is during intervention, 0 otherwise
- `action_source`: string, policy | human | none

## Before/During/After Windows

SQLite query `query_intervention_window(id, before, after)` returns ticks
before, during, and after an intervention for training data context.
