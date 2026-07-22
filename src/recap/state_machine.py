"""RECAP state machine: session/episode/pause/freeze/intervention authority.

This module implements the control-authority state machine that governs the
real-robot inference loop.  It is deliberately pure-Python with no I/O so it
can be unit-tested exhaustively without hardware.

State transitions
-----------------
Session lifecycle:  IDLE -> RUNNING -> STOPPED
Inference pause:    RUNNING -> PAUSED -> RUNNING  (requests stop, connections stay alive)
Camera freeze:      front / wrist independently frozen (latest frame replayed)
Intervention:       RUNNING -> INTERVENING -> RUNNING_PENDING_CONFIRM -> RUNNING
                     (human takes control; policy shadowed; confirm to resume)

Key invariants
--------------
* Pausing inference does NOT disconnect the policy server or close Hook sockets.
* In-flight responses received while paused are marked `ignored` and not executed.
* Resuming inference replans from the latest observation (no stale chunk replay).
* Freeze without a captured first frame returns an explicit NACK.
* Intervention has a continuous `intervention_id` and clear before/during/after windows.
* Ending intervention does NOT auto-resume; explicit confirmation is required.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class SessionState(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    INTERVENING = "intervening"
    RESUME_PENDING = "resume_pending"
    STOPPED = "stopped"


class FreezeTarget(str, Enum):
    FRONT = "front"
    WRIST = "wrist"


class ControlSource(str, Enum):
    POLICY = "policy"
    HUMAN = "human"
    NONE = "none"


@dataclass
class FreezeState:
    """Per-camera freeze state with first-frame capture."""

    frozen: bool = False
    first_frame: Any = None  # np.ndarray or bytes
    first_frame_mono: float = 0.0
    source_frame_id: str | None = None

    def capture(self, frame: Any, mono: float, frame_id: str) -> bool:
        """Capture the first frame. Returns False if already frozen."""
        if self.frozen:
            return False
        self.frozen = True
        self.first_frame = frame
        self.first_frame_mono = mono
        self.source_frame_id = frame_id
        return True

    def release(self) -> None:
        self.frozen = False
        self.first_frame = None
        self.first_frame_mono = 0.0
        self.source_frame_id = None

    def get_frame(self, live_frame: Any, mono: float, frame_id: str) -> tuple[Any, bool]:
        """Return (frame, is_frozen_copy). If frozen, returns the captured frame."""
        if self.frozen and self.first_frame is not None:
            return self.first_frame, True
        return live_frame, False


@dataclass
class InterventionWindow:
    """Metadata for a single human-correction episode."""

    intervention_id: int
    start_tick: int
    start_mono: float
    end_tick: int | None = None
    end_mono: float | None = None
    confirmed: bool = False
    shadow_policy: bool = False
    human_action_count: int = 0
    policy_action_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "intervention_id": self.intervention_id,
            "start_tick": self.start_tick,
            "start_mono": self.start_mono,
            "end_tick": self.end_tick,
            "end_mono": self.end_mono,
            "confirmed": self.confirmed,
            "shadow_policy": self.shadow_policy,
            "human_action_count": self.human_action_count,
            "policy_action_count": self.policy_action_count,
        }


@dataclass
class StateEvent:
    """An event emitted by the state machine for recording."""

    event: str
    tick: int
    mono_time: float
    wall_time: float
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event": self.event,
            "tick": self.tick,
            "mono_time": self.mono_time,
            "wall_time": self.wall_time,
            **self.payload,
        }


class StateMachine:
    """Thread-safe RECAP control-authority state machine.

    All public methods are safe to call from the control loop, the Hook server,
    or the UI.  Methods return a list of StateEvent that the caller should
    record and broadcast.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._state: SessionState = SessionState.IDLE
        self._session_id: str | None = None
        self._episode_id: int = 0
        self._tick: int = 0
        self._inference_paused: bool = False
        self._freezes: dict[FreezeTarget, FreezeState] = {
            FreezeTarget.FRONT: FreezeState(),
            FreezeTarget.WRIST: FreezeState(),
        }
        self._control_source: ControlSource = ControlSource.NONE
        self._interventions: list[InterventionWindow] = []
        self._next_intervention_id: int = 1
        self._current_intervention: InterventionWindow | None = None
        self._shadow_policy: bool = False
        self._events: list[StateEvent] = []

    # -- properties ---------------------------------------------------------

    @property
    def state(self) -> SessionState:
        with self._lock:
            return self._state

    @property
    def session_id(self) -> str | None:
        with self._lock:
            return self._session_id

    @property
    def episode_id(self) -> int:
        with self._lock:
            return self._episode_id

    @property
    def tick(self) -> int:
        with self._lock:
            return self._tick

    @property
    def inference_paused(self) -> bool:
        with self._lock:
            return self._inference_paused

    @property
    def control_source(self) -> ControlSource:
        with self._lock:
            return self._control_source

    @property
    def is_intervening(self) -> bool:
        with self._lock:
            return self._state == SessionState.INTERVENING

    @property
    def current_intervention(self) -> InterventionWindow | None:
        with self._lock:
            return self._current_intervention

    @property
    def interventions(self) -> list[InterventionWindow]:
        with self._lock:
            return list(self._interventions)

    def is_frozen(self, target: FreezeTarget) -> bool:
        with self._lock:
            return self._freezes[target].frozen

    def get_freeze(self, target: FreezeTarget) -> FreezeState:
        with self._lock:
            return self._freezes[target]

    # -- session lifecycle --------------------------------------------------

    def start_session(self, session_id: str, prompt: str = "") -> list[StateEvent]:
        with self._lock:
            if self._state != SessionState.IDLE:
                return []
            self._session_id = session_id
            self._episode_id = 0
            self._tick = 0
            self._state = SessionState.RUNNING
            self._control_source = ControlSource.POLICY
            self._inference_paused = False
            ev = self._emit("session_start", {"session_id": session_id, "prompt": prompt})
            return [ev]

    def stop_session(self) -> list[StateEvent]:
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPED):
                return []
            # Close any open intervention
            events: list[StateEvent] = []
            if self._current_intervention is not None:
                events.extend(self._end_intervention_locked())
            self._state = SessionState.STOPPED
            self._control_source = ControlSource.NONE
            events.append(self._emit("session_stop", {"session_id": self._session_id}))
            return events

    def next_episode(self) -> list[StateEvent]:
        with self._lock:
            if self._state == SessionState.STOPPED:
                return []
            self._episode_id += 1
            self._tick = 0
            self._state = SessionState.RUNNING
            self._control_source = ControlSource.POLICY
            self._inference_paused = False
            for fz in self._freezes.values():
                fz.release()
            return [self._emit("episode_start", {"episode_id": self._episode_id})]

    # -- tick ---------------------------------------------------------------

    def advance_tick(self) -> int:
        with self._lock:
            self._tick += 1
            return self._tick

    # -- inference pause/resume ---------------------------------------------

    def pause_inference(self) -> list[StateEvent]:
        with self._lock:
            if self._inference_paused or self._state not in (
                SessionState.RUNNING,
                SessionState.RESUME_PENDING,
            ):
                return []
            self._inference_paused = True
            if self._state == SessionState.RUNNING:
                self._state = SessionState.PAUSED
            return [self._emit("inference_pause", {})]

    def resume_inference(self) -> list[StateEvent]:
        """Resume inference. Next action comes from latest observation."""
        with self._lock:
            if not self._inference_paused:
                return []
            self._inference_paused = False
            if self._state == SessionState.PAUSED:
                self._state = SessionState.RUNNING
            elif self._state == SessionState.RESUME_PENDING:
                self._state = SessionState.RUNNING
            return [self._emit("inference_resume", {"replan": True})]

    def should_request_inference(self) -> bool:
        """Whether the control loop should send a new policy request this tick."""
        with self._lock:
            if self._inference_paused:
                return False
            if self._state in (SessionState.INTERVENING, SessionState.STOPPED, SessionState.IDLE):
                return False
            return True

    def should_execute_policy_action(self) -> bool:
        """Whether a returned policy action should be executed (not shadowed)."""
        with self._lock:
            if self._inference_paused:
                return False
            if self._state == SessionState.INTERVENING:
                return False
            return True

    def mark_response_ignored(self, request_id: int, chunk_idx: int) -> list[StateEvent]:
        """Mark an in-flight response that arrived while paused as ignored."""
        with self._lock:
            return [self._emit(
                "response_ignored",
                {"request_id": request_id, "chunk_idx": chunk_idx},
            )]

    # -- camera freeze ------------------------------------------------------

    def freeze_camera(
        self, target: FreezeTarget, frame: Any, frame_id: str
    ) -> tuple[bool, list[StateEvent]]:
        """Freeze a camera stream. Returns (ok, events).

        Returns (False, [nack_event]) if no first frame is available.
        """
        with self._lock:
            fz = self._freezes[target]
            if frame is None:
                return False, [self._emit(
                    "freeze_nack",
                    {"target": target.value, "reason": "no_first_frame"},
                )]
            ok = fz.capture(frame, time.perf_counter(), frame_id)
            if ok:
                return True, [self._emit(
                    "freeze_start",
                    {"target": target.value, "source_frame_id": frame_id},
                )]
            return True, []  # already frozen, idempotent

    def unfreeze_camera(self, target: FreezeTarget) -> list[StateEvent]:
        with self._lock:
            fz = self._freezes[target]
            if not fz.frozen:
                return []
            fz.release()
            return [self._emit("freeze_stop", {"target": target.value})]

    def get_camera_frame(
        self, target: FreezeTarget, live_frame: Any, frame_id: str
    ) -> tuple[Any, bool]:
        """Return (frame, is_frozen_copy)."""
        with self._lock:
            fz = self._freezes[target]
            return fz.get_frame(live_frame, time.perf_counter(), frame_id)

    # -- intervention -------------------------------------------------------

    def start_intervention(self, shadow_policy: bool = False) -> list[StateEvent]:
        """Begin human correction. Policy actions stop executing.

        If shadow_policy is True, policy requests continue but are only recorded.
        """
        with self._lock:
            if self._state in (SessionState.IDLE, SessionState.STOPPED):
                return []
            if self._state == SessionState.INTERVENING:
                return []
            events: list[StateEvent] = []
            # Pause inference if not already
            if not self._inference_paused and not shadow_policy:
                self._inference_paused = True
                events.append(self._emit("inference_pause", {"reason": "intervention"}))
            iid = self._next_intervention_id
            self._next_intervention_id += 1
            self._current_intervention = InterventionWindow(
                intervention_id=iid,
                start_tick=self._tick,
                start_mono=time.perf_counter(),
                shadow_policy=shadow_policy,
            )
            self._interventions.append(self._current_intervention)
            self._state = SessionState.INTERVENING
            self._control_source = ControlSource.HUMAN
            self._shadow_policy = shadow_policy
            events.append(self._emit("intervention_start", {
                "intervention_id": iid,
                "shadow_policy": shadow_policy,
                "start_tick": self._tick,
            }))
            return events

    def record_human_action(self) -> None:
        """Called when a human action is executed during intervention."""
        with self._lock:
            if self._current_intervention is not None:
                self._current_intervention.human_action_count += 1

    def record_shadow_policy_action(self) -> None:
        """Called when a policy action is shadowed (recorded, not executed)."""
        with self._lock:
            if self._current_intervention is not None:
                self._current_intervention.policy_action_count += 1

    def end_intervention(self) -> list[StateEvent]:
        """End human correction. Does NOT auto-resume; requires confirmation."""
        with self._lock:
            return self._end_intervention_locked()

    def _end_intervention_locked(self) -> list[StateEvent]:
        if self._current_intervention is None:
            return []
        iv = self._current_intervention
        iv.end_tick = self._tick
        iv.end_mono = time.perf_counter()
        self._state = SessionState.RESUME_PENDING
        self._control_source = ControlSource.NONE
        events = [self._emit("intervention_end", iv.to_dict())]
        self._current_intervention = None
        return events

    def confirm_resume(self) -> list[StateEvent]:
        """Confirm resumption after intervention. Replans from latest obs."""
        with self._lock:
            if self._state != SessionState.RESUME_PENDING:
                return []
            self._state = SessionState.RUNNING
            self._control_source = ControlSource.POLICY
            self._inference_paused = False
            self._shadow_policy = False
            return [self._emit("resume_confirmed", {"replan": True})]

    # -- snapshot -----------------------------------------------------------

    def snapshot(self) -> dict[str, Any]:
        """Return a full state snapshot for UI recovery after reconnect."""
        with self._lock:
            return {
                "state": self._state.value,
                "session_id": self._session_id,
                "episode_id": self._episode_id,
                "tick": self._tick,
                "inference_paused": self._inference_paused,
                "control_source": self._control_source.value,
                "freezes": {
                    t.value: {"frozen": fz.frozen, "source_frame_id": fz.source_frame_id}
                    for t, fz in self._freezes.items()
                },
                "current_intervention": (
                    self._current_intervention.to_dict()
                    if self._current_intervention
                    else None
                ),
                "intervention_count": len(self._interventions),
                "shadow_policy": self._shadow_policy,
            }

    def drain_events(self) -> list[StateEvent]:
        """Return and clear pending events."""
        with self._lock:
            evs = self._events
            self._events = []
            return evs

    # -- internal -----------------------------------------------------------

    def _emit(self, event: str, payload: dict[str, Any]) -> StateEvent:
        ev = StateEvent(
            event=event,
            tick=self._tick,
            mono_time=time.perf_counter(),
            wall_time=time.time(),
            payload=payload,
        )
        self._events.append(ev)
        return ev
