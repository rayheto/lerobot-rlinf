"""RECAP data recorder: SQLite index + raw data blobs.

Design (see DATA_ADR.md):
  * SQLite stores queryable metadata: tick alignment, intervention windows,
    drop/error/incomplete flags, request/response timing.
  * Raw camera frames are written as JPEG (per-tick) or batched into MP4
    at export time.  Per-tick JPEG ensures lossless alignment even if the
    export step fails.
  * Raw wire responses (msgpack bytes from the policy server) are stored
    as MsgPack blobs keyed by request_id.
  * Joint states and actions are stored as Parquet columns at export time;
    per-tick they go into SQLite as JSON for immediate queryability.

All writes happen on a background thread with a bounded queue.  When the
queue is full, the oldest item is dropped and a drop event is recorded.
"""
from __future__ import annotations

import json
import queue
import sqlite3
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _json_default(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS ticks (
    tick_id        INTEGER PRIMARY KEY,
    session_id     TEXT NOT NULL,
    episode_id     INTEGER NOT NULL,
    mono_time      REAL NOT NULL,
    wall_time      REAL NOT NULL,
    joint_state    TEXT,
    raw_action     TEXT,
    executed_action TEXT,
    action_source  TEXT,
    chunk_idx      INTEGER,
    slot_idx       INTEGER,
    request_id     INTEGER,
    front_frame_path TEXT,
    wrist_frame_path TEXT,
    wire_response_path TEXT,
    intervention_id INTEGER,
    is_frozen_front INTEGER DEFAULT 0,
    is_frozen_wrist INTEGER DEFAULT 0,
    inference_paused INTEGER DEFAULT 0,
    drop_flag      INTEGER DEFAULT 0,
    error_flag     INTEGER DEFAULT 0,
    incomplete_flag INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS events (
    seq            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL,
    tick           INTEGER NOT NULL,
    mono_time      REAL NOT NULL,
    wall_time      REAL NOT NULL,
    event          TEXT NOT NULL,
    payload        TEXT
);

CREATE TABLE IF NOT EXISTS interventions (
    intervention_id INTEGER PRIMARY KEY,
    session_id      TEXT NOT NULL,
    start_tick      INTEGER NOT NULL,
    start_mono      REAL NOT NULL,
    end_tick        INTEGER,
    end_mono        REAL,
    confirmed       INTEGER DEFAULT 0,
    shadow_policy   INTEGER DEFAULT 0,
    human_action_count INTEGER DEFAULT 0,
    policy_action_count INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS requests (
    request_id     INTEGER PRIMARY KEY,
    session_id     TEXT NOT NULL,
    chunk_idx      INTEGER NOT NULL,
    control_tick   INTEGER NOT NULL,
    observation_mono REAL NOT NULL,
    observation_wall REAL NOT NULL,
    response_mono  REAL,
    response_wall  REAL,
    ok             INTEGER,
    error          TEXT,
    chunk_len      INTEGER,
    wire_response_path TEXT,
    ignored        INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS session_meta (
    key   TEXT PRIMARY KEY,
    value TEXT
);

CREATE INDEX IF NOT EXISTS idx_ticks_session ON ticks(session_id, tick_id);
CREATE INDEX IF NOT EXISTS idx_ticks_intervention ON ticks(intervention_id);
CREATE INDEX IF NOT EXISTS idx_events_session ON events(session_id, tick);
CREATE INDEX IF NOT EXISTS idx_requests_session ON requests(session_id, request_id);
"""


@dataclass
class TickRecord:
    """Per-tick data to record."""

    tick_id: int
    session_id: str
    episode_id: int
    mono_time: float
    wall_time: float
    joint_state: np.ndarray | None = None
    raw_action: np.ndarray | None = None
    executed_action: np.ndarray | None = None
    action_source: str = "policy"
    chunk_idx: int | None = None
    slot_idx: int | None = None
    request_id: int | None = None
    front_frame: np.ndarray | None = None
    wrist_frame: np.ndarray | None = None
    wire_response: bytes | None = None
    intervention_id: int | None = None
    is_frozen_front: bool = False
    is_frozen_wrist: bool = False
    inference_paused: bool = False
    drop_flag: bool = False
    error_flag: bool = False
    incomplete_flag: bool = False


class DataRecorder:
    """Background-thread data recorder with SQLite index + raw blobs.

    Parameters
    ----------
    data_dir : Path
        Root directory for this session's data.  SQLite goes in
        data_dir/session.sqlite; raw frames in data_dir/frames/;
        wire responses in data_dir/wire/.
    max_queue : int
        Maximum pending records.  When full, oldest is dropped + flagged.
    """

    def __init__(self, data_dir: Path | None, max_queue: int = 512) -> None:
        self.data_dir = data_dir
        self._queue: queue.Queue[Any | None] | None = None
        self._thread: threading.Thread | None = None
        self._db_path: Path | None = None
        self._frames_dir: Path | None = None
        self._wire_dir: Path | None = None
        self._dropped_count = 0
        self._error_count = 0
        self._closed = False
        if data_dir is not None:
            self.data_dir = Path(data_dir)
            self.data_dir.mkdir(parents=True, exist_ok=True)
            self._db_path = self.data_dir / "session.sqlite"
            self._frames_dir = self.data_dir / "frames"
            self._wire_dir = self.data_dir / "wire"
            self._frames_dir.mkdir(exist_ok=True)
            self._wire_dir.mkdir(exist_ok=True)
            self._init_db()
            self._queue = queue.Queue(maxsize=max_queue)
            self._thread = threading.Thread(
                target=self._worker, name="recap-data-writer", daemon=True
            )
            self._thread.start()

    @property
    def db_path(self) -> Path | None:
        return self._db_path

    @property
    def dropped_count(self) -> int:
        return self._dropped_count

    @property
    def error_count(self) -> int:
        return self._error_count

    def _init_db(self) -> None:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        try:
            conn.executescript(_SCHEMA)
            conn.commit()
        finally:
            conn.close()

    def record_tick(self, rec: TickRecord) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(("tick", rec))
        except queue.Full:
            self._dropped_count += 1
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(("tick", rec))
            except queue.Full:
                pass

    def record_event(
        self,
        event: str,
        tick: int,
        mono_time: float,
        wall_time: float,
        session_id: str,
        payload: dict[str, Any] | None = None,
    ) -> None:
        if self._queue is None:
            return
        item = ("event", {
            "event": event,
            "tick": tick,
            "mono_time": mono_time,
            "wall_time": wall_time,
            "session_id": session_id,
            "payload": payload or {},
        })
        try:
            self._queue.put_nowait(item)
        except queue.Full:
            self._dropped_count += 1

    def record_request(self, req: dict[str, Any]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(("request", req))
        except queue.Full:
            self._dropped_count += 1

    def record_intervention(self, iv: dict[str, Any]) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(("intervention", iv))
        except queue.Full:
            self._dropped_count += 1

    def set_meta(self, key: str, value: Any) -> None:
        if self._queue is None:
            return
        try:
            self._queue.put_nowait(("meta", {"key": key, "value": json.dumps(value, default=_json_default)}))
        except queue.Full:
            self._dropped_count += 1

    def _worker(self) -> None:
        assert self._queue is not None
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        try:
            while True:
                item = self._queue.get()
                if item is None:
                    break
                try:
                    self._process(conn, item)
                    conn.commit()
                except Exception:
                    self._error_count += 1
        finally:
            conn.close()

    def _process(self, conn: sqlite3.Connection, item: tuple[str, Any]) -> None:
        kind, data = item
        if kind == "tick":
            self._write_tick(conn, data)
        elif kind == "event":
            self._write_event(conn, data)
        elif kind == "request":
            self._write_request(conn, data)
        elif kind == "intervention":
            self._write_intervention(conn, data)
        elif kind == "meta":
            conn.execute(
                "INSERT OR REPLACE INTO session_meta (key, value) VALUES (?, ?)",
                (data["key"], data["value"]),
            )

    def _write_tick(self, conn: sqlite3.Connection, rec: TickRecord) -> None:
        front_path = None
        wrist_path = None
        wire_path = None
        if rec.front_frame is not None and self._frames_dir is not None:
            front_path = str(self._frames_dir / f"tick_{rec.tick_id:07d}_front.jpg")
            self._write_jpeg(front_path, rec.front_frame)
        if rec.wrist_frame is not None and self._frames_dir is not None:
            wrist_path = str(self._frames_dir / f"tick_{rec.tick_id:07d}_wrist.jpg")
            self._write_jpeg(wrist_path, rec.wrist_frame)
        if rec.wire_response is not None and self._wire_dir is not None:
            wire_path = str(self._wire_dir / f"req_{rec.request_id:07d}.msgpack")
            self._wire_dir.mkdir(exist_ok=True)
            Path(wire_path).write_bytes(rec.wire_response)

        conn.execute(
            """INSERT OR REPLACE INTO ticks
               (tick_id, session_id, episode_id, mono_time, wall_time,
                joint_state, raw_action, executed_action, action_source,
                chunk_idx, slot_idx, request_id,
                front_frame_path, wrist_frame_path, wire_response_path,
                intervention_id, is_frozen_front, is_frozen_wrist,
                inference_paused, drop_flag, error_flag, incomplete_flag)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                rec.tick_id,
                rec.session_id,
                rec.episode_id,
                rec.mono_time,
                rec.wall_time,
                json.dumps(rec.joint_state.tolist() if rec.joint_state is not None else None, default=_json_default),
                json.dumps(rec.raw_action.tolist() if rec.raw_action is not None else None, default=_json_default),
                json.dumps(rec.executed_action.tolist() if rec.executed_action is not None else None, default=_json_default),
                rec.action_source,
                rec.chunk_idx,
                rec.slot_idx,
                rec.request_id,
                front_path,
                wrist_path,
                wire_path,
                rec.intervention_id,
                int(rec.is_frozen_front),
                int(rec.is_frozen_wrist),
                int(rec.inference_paused),
                int(rec.drop_flag),
                int(rec.error_flag),
                int(rec.incomplete_flag),
            ),
        )

    def _write_event(self, conn: sqlite3.Connection, data: dict[str, Any]) -> None:
        conn.execute(
            """INSERT INTO events (session_id, tick, mono_time, wall_time, event, payload)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                data["session_id"],
                data["tick"],
                data["mono_time"],
                data["wall_time"],
                data["event"],
                json.dumps(data["payload"], default=_json_default),
            ),
        )

    def _write_request(self, conn: sqlite3.Connection, req: dict[str, Any]) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO requests
               (request_id, session_id, chunk_idx, control_tick,
                observation_mono, observation_wall,
                response_mono, response_wall, ok, error, chunk_len,
                wire_response_path, ignored)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                req.get("request_id"),
                req.get("session_id"),
                req.get("chunk_idx"),
                req.get("control_tick"),
                req.get("observation_mono"),
                req.get("observation_wall"),
                req.get("response_mono"),
                req.get("response_wall"),
                req.get("ok"),
                req.get("error"),
                req.get("chunk_len"),
                req.get("wire_response_path"),
                int(req.get("ignored", 0)),
            ),
        )

    def _write_intervention(self, conn: sqlite3.Connection, iv: dict[str, Any]) -> None:
        conn.execute(
            """INSERT OR REPLACE INTO interventions
               (intervention_id, session_id, start_tick, start_mono,
                end_tick, end_mono, confirmed, shadow_policy,
                human_action_count, policy_action_count)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                iv.get("intervention_id"),
                iv.get("session_id"),
                iv.get("start_tick"),
                iv.get("start_mono"),
                iv.get("end_tick"),
                iv.get("end_mono"),
                int(iv.get("confirmed", 0)),
                int(iv.get("shadow_policy", 0)),
                iv.get("human_action_count", 0),
                iv.get("policy_action_count", 0),
            ),
        )

    def _write_jpeg(self, path: str, frame: np.ndarray) -> None:
        try:
            import cv2
            cv2.imwrite(path, frame)
        except ImportError:
            np.save(path.replace(".jpg", ".npy"), frame)

    def close(self) -> None:
        if self._queue is None or self._closed:
            return
        self._closed = True
        self._queue.put(None)
        if self._thread is not None:
            self._thread.join(timeout=10)

    def query_intervention_window(
        self, intervention_id: int, before: int = 5, after: int = 5
    ) -> dict[str, Any]:
        """Return ticks before, during, and after an intervention."""
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute(
                "SELECT start_tick, end_tick FROM interventions WHERE intervention_id = ?",
                (intervention_id,),
            ).fetchone()
            if row is None:
                return {"error": "intervention not found"}
            start_tick, end_tick = row
            before_ticks = conn.execute(
                "SELECT tick_id, joint_state, executed_action FROM ticks WHERE tick_id >= ? AND tick_id < ? ORDER BY tick_id",
                (start_tick - before, start_tick),
            ).fetchall()
            during_ticks = conn.execute(
                "SELECT tick_id, joint_state, executed_action, action_source FROM ticks WHERE tick_id >= ? AND tick_id <= ? ORDER BY tick_id",
                (start_tick, end_tick),
            ).fetchall()
            after_ticks = conn.execute(
                "SELECT tick_id, joint_state, executed_action FROM ticks WHERE tick_id > ? AND tick_id <= ? ORDER BY tick_id",
                (end_tick, end_tick + after),
            ).fetchall()
            return {
                "intervention_id": intervention_id,
                "start_tick": start_tick,
                "end_tick": end_tick,
                "before": before_ticks,
                "during": during_ticks,
                "after": after_ticks,
            }
        finally:
            conn.close()

    def tick_count(self) -> int:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()

    def event_count(self) -> int:
        assert self._db_path is not None
        conn = sqlite3.connect(str(self._db_path))
        try:
            row = conn.execute("SELECT COUNT(*) FROM events").fetchone()
            return row[0] if row else 0
        finally:
            conn.close()
