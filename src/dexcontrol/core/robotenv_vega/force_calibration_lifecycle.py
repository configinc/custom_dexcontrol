"""Server-owned lifecycle and durable storage for force calibration.

This module deliberately contains no robot motion or fitting code.  A robot
adapter owns those operations and reports phase completion here.  Keeping this
state machine pure makes the bilateral safety contract independently testable.
"""

from __future__ import annotations

import json
import math
import os
import threading
import time
import uuid
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Callable, Mapping
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
SCHEMA_VERSION = 1
ARM_NAMES = frozenset({"left", "right"})
TERMINAL_STATES = frozenset({"SUCCEEDED", "FAILED", "CANCELLED"})
ACTIVE_STATES = frozenset(
    {
        "READY_FOR_STAGE",
        "MOVING",
        "SETTLING",
        "COLLECTING",
        "FITTING",
        "RETURNING",
        "CANCELLING",
        "READY_TO_APPLY",
    }
)
PERSISTED_RECORD_ERRORS = (
    OSError,
    ValueError,
    TypeError,
    KeyError,
    ArithmeticError,
    RecursionError,
    json.JSONDecodeError,
)


class LifecycleError(RuntimeError):
    """A deterministic request rejection with a gRPC-compatible code name."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _non_empty_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    if value != value.strip():
        raise ValueError(f"{name} must not contain surrounding whitespace")
    return value


def _strict_json_object(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be a JSON object")
    result = dict(value)
    encoded = json.dumps(
        result,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    decoded = json.loads(encoded, parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)))
    if not isinstance(decoded, dict):  # pragma: no cover - guarded above
        raise ValueError(f"{name} must be a JSON object")
    return decoded


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key {key!r}")
        result[key] = value
    return result


def _unix_us() -> int:
    return time.time_ns() // 1_000


def _next_kst_boundary_us(timestamp_us: int) -> int:
    seconds, microseconds = divmod(timestamp_us, 1_000_000)
    instant = datetime.fromtimestamp(seconds, tz=timezone.utc).replace(
        microsecond=microseconds
    ).astimezone(KST)
    boundary = instant.replace(hour=7, minute=0, second=0, microsecond=0)
    if instant >= boundary:
        boundary += timedelta(days=1)
    utc_boundary = boundary.astimezone(timezone.utc)
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    delta = utc_boundary - epoch
    return (
        delta.days * 86_400_000_000
        + delta.seconds * 1_000_000
        + delta.microseconds
    )


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if temporary.exists():
            temporary.unlink()


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(
            handle,
            object_pairs_hook=_reject_duplicate_keys,
            parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
        )
    if not isinstance(value, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return value


def _durable_unlink(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        return
    directory_fd = os.open(path.parent, os.O_RDONLY)
    try:
        os.fsync(directory_fd)
    finally:
        os.close(directory_fd)


@dataclass(frozen=True)
class CalibrationStage:
    stage_id: str
    arm_payload: dict[str, Any]


@dataclass(frozen=True)
class BilateralCalibrationPlan:
    plan_id: str
    plan_sha256: str
    arm_name: str
    stages: tuple[CalibrationStage, ...]
    raw_bytes: bytes

    @classmethod
    def load(cls, path: str | Path, arm_name: str) -> "BilateralCalibrationPlan":
        arm = _non_empty_string(arm_name, "arm_name").lower()
        if arm not in ARM_NAMES:
            raise ValueError(f"arm_name must be one of {sorted(ARM_NAMES)}")
        raw = Path(path).read_bytes()
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ValueError("calibration plan must be UTF-8") from exc
        try:
            document = json.loads(
                text,
                object_pairs_hook=_reject_duplicate_keys,
                parse_constant=lambda token: (_ for _ in ()).throw(ValueError(token)),
            )
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError("calibration plan must be strict JSON") from exc
        if not isinstance(document, dict) or document.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported calibration plan schema_version")
        plan_id = _non_empty_string(document.get("plan_id"), "plan_id")
        raw_stages = document.get("stages")
        if not isinstance(raw_stages, list) or not raw_stages:
            raise ValueError("calibration plan needs at least one stage")
        stages: list[CalibrationStage] = []
        seen: set[str] = set()
        for index, raw_stage in enumerate(raw_stages):
            if not isinstance(raw_stage, dict):
                raise ValueError(f"stage {index} must be an object")
            stage_id = _non_empty_string(raw_stage.get("stage_id"), f"stage {index} stage_id")
            if stage_id in seen:
                raise ValueError(f"duplicate stage_id {stage_id!r}")
            seen.add(stage_id)
            arms = raw_stage.get("arms")
            if not isinstance(arms, dict) or set(arms) != ARM_NAMES:
                raise ValueError(f"stage {stage_id!r} must define left and right exactly once")
            checked_arms = {
                name: _strict_json_object(
                    arms[name], f"stage {stage_id!r} {name} payload"
                )
                for name in ARM_NAMES
            }
            stages.append(
                CalibrationStage(
                    stage_id=stage_id,
                    arm_payload=checked_arms[arm],
                )
            )
        return cls(
            plan_id=plan_id,
            plan_sha256=sha256(raw).hexdigest(),
            arm_name=arm,
            stages=tuple(stages),
            raw_bytes=raw,
        )


@dataclass(frozen=True)
class CalibrationStatusSnapshot:
    state: str
    run_id: str
    current_stage_index: int
    completed_stages: int
    total_stages: int
    message: str
    arm_name: str
    plan_id: str
    plan_sha256: str
    stage_id: str
    server_instance_id: str
    cache_reused: bool
    command_deadline_timestamp_us: int


class ForceCalibrationLifecycle:
    """Thread-safe state machine shared by a robot server and its worker."""

    def __init__(
        self,
        *,
        plan: BilateralCalibrationPlan,
        server_identity: Mapping[str, Any],
        storage_directory: str | Path,
        calibration_kind: str,
        reference_point_id: str,
        command_timeout_seconds: float = 30.0,
        now_us: Callable[[], int] = _unix_us,
        validate_parameters: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> None:
        if not math.isfinite(command_timeout_seconds) or command_timeout_seconds <= 0:
            raise ValueError("command_timeout_seconds must be finite and positive")
        self.plan = plan
        self.server_identity = _strict_json_object(server_identity, "server_identity")
        self.storage_directory = Path(storage_directory)
        self.calibration_kind = _non_empty_string(calibration_kind, "calibration_kind")
        self.reference_point_id = _non_empty_string(reference_point_id, "reference_point_id")
        self.command_timeout_us = int(command_timeout_seconds * 1_000_000)
        self.now_us = now_us
        self.validate_parameters = validate_parameters
        self.server_instance_id = uuid.uuid4().hex
        self._lock = threading.RLock()
        self._state = "IDLE"
        self._run_id = ""
        self._force_recalibrate = False
        self._current_stage_index = 0
        self._completed_stages = 0
        self._message = ""
        self._cache_reused = False
        self._deadline_us = 0
        self._restore_persistent_state()

    @property
    def active_cache_path(self) -> Path:
        return self.storage_directory / "active.json"

    @property
    def staged_candidate_path(self) -> Path:
        return self.storage_directory / "staged.json"

    @property
    def terminal_state_path(self) -> Path:
        return self.storage_directory / "terminal.json"

    @property
    def latest_start_path(self) -> Path:
        return self.storage_directory / "latest_start.json"

    def _history_path(self, run_id: str) -> Path:
        return self.storage_directory / "results" / f"{sha256(run_id.encode('utf-8')).hexdigest()}.json"

    def _terminal_result_path(self, run_id: str) -> Path:
        return self.storage_directory / "terminal_results" / f"{sha256(run_id.encode('utf-8')).hexdigest()}.json"

    def _deadline(self) -> int:
        return self.now_us() + self.command_timeout_us

    def _stage_id(self) -> str:
        if self._state in {"READY_FOR_STAGE", "MOVING", "SETTLING", "COLLECTING", "RETURNING"} and (
            0 <= self._current_stage_index < len(self.plan.stages)
        ):
            return self.plan.stages[self._current_stage_index].stage_id
        return ""

    def status(self) -> CalibrationStatusSnapshot:
        with self._lock:
            self._expire_command_deadline_locked()
            return CalibrationStatusSnapshot(
                state=self._state,
                run_id=self._run_id,
                current_stage_index=self._current_stage_index,
                completed_stages=self._completed_stages,
                total_stages=len(self.plan.stages),
                message=self._message,
                arm_name=self.plan.arm_name,
                plan_id=self.plan.plan_id,
                plan_sha256=self.plan.plan_sha256,
                stage_id=self._stage_id(),
                server_instance_id=self.server_instance_id,
                cache_reused=self._cache_reused,
                command_deadline_timestamp_us=self._deadline_us,
            )

    def _base_record(
        self,
        *,
        run_id: str,
        source_run_id: str,
        calibrated_at_us: int,
        parameters: Mapping[str, Any],
        quality: Mapping[str, Any],
        was_committed: bool,
    ) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "run_id": run_id,
            "source_run_id": source_run_id,
            "server_instance_id": self.server_instance_id,
            "force_recalibrate": self._force_recalibrate,
            "arm_name": self.plan.arm_name,
            "plan_id": self.plan.plan_id,
            "plan_sha256": self.plan.plan_sha256,
            "server_identity": self.server_identity,
            "calibrated_at_timestamp_us": calibrated_at_us,
            "expires_at_timestamp_us": _next_kst_boundary_us(calibrated_at_us),
            "calibration_kind": self.calibration_kind,
            "reference_point_id": self.reference_point_id,
            "parameters": _strict_json_object(parameters, "parameters"),
            "quality": _strict_json_object(quality, "quality"),
            "was_committed": was_committed,
        }

    def _validate_record(
        self,
        value: Mapping[str, Any],
        *,
        require_current: bool,
        allow_future_timestamp: bool = False,
    ) -> dict[str, Any]:
        record = dict(value)
        if record.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported calibration cache schema")
        for key in (
            "run_id",
            "source_run_id",
            "arm_name",
            "plan_id",
            "plan_sha256",
            "calibration_kind",
            "reference_point_id",
        ):
            _non_empty_string(record.get(key), key)
        if record["arm_name"] != self.plan.arm_name:
            raise ValueError("calibration arm identity mismatch")
        if record["plan_id"] != self.plan.plan_id or record["plan_sha256"] != self.plan.plan_sha256:
            raise ValueError("calibration plan identity mismatch")
        if record.get("server_identity") != self.server_identity:
            raise ValueError("calibration server identity mismatch")
        if record["calibration_kind"] != self.calibration_kind:
            raise ValueError("calibration kind mismatch")
        if record["reference_point_id"] != self.reference_point_id:
            raise ValueError("calibration reference point mismatch")
        calibrated = record.get("calibrated_at_timestamp_us")
        expires = record.get("expires_at_timestamp_us")
        if isinstance(calibrated, bool) or not isinstance(calibrated, int) or calibrated <= 0:
            raise ValueError("invalid calibrated_at_timestamp_us")
        if isinstance(expires, bool) or not isinstance(expires, int):
            raise ValueError("invalid expires_at_timestamp_us")
        now = self.now_us()
        if calibrated > now and not allow_future_timestamp:
            raise ValueError("future calibration cache")
        if expires != _next_kst_boundary_us(calibrated):
            raise ValueError("calibration cache expiry does not match 07:00 KST boundary")
        if require_current and now >= expires:
            raise ValueError("calibration cache expired")
        record["parameters"] = _strict_json_object(record.get("parameters"), "parameters")
        if self.validate_parameters is not None:
            self.validate_parameters(record["parameters"])
        record["quality"] = _strict_json_object(record.get("quality"), "quality")
        if record["quality"].get("passed") is not True:
            raise ValueError("calibration quality gate did not pass")
        if not isinstance(record.get("was_committed"), bool):
            raise ValueError("was_committed must be boolean")
        if not isinstance(record.get("force_recalibrate"), bool):
            raise ValueError("force_recalibrate must be boolean")
        _non_empty_string(record.get("server_instance_id"), "server_instance_id")
        return record

    def _load_valid_active(self) -> dict[str, Any] | None:
        try:
            record = self._validate_record(_read_json(self.active_cache_path), require_current=True)
        except PERSISTED_RECORD_ERRORS:
            return None
        return record if record["was_committed"] else None

    def _persist_start_marker(self, run_id: str, force_recalibrate: bool) -> None:
        _atomic_write_json(
            self.latest_start_path,
            {
                "schema_version": SCHEMA_VERSION,
                "run_id": run_id,
                "force_recalibrate": force_recalibrate,
                "plan_id": self.plan.plan_id,
                "plan_sha256": self.plan.plan_sha256,
                "arm_name": self.plan.arm_name,
                "server_identity": self.server_identity,
                "calibration_kind": self.calibration_kind,
                "reference_point_id": self.reference_point_id,
            },
        )

    def _read_latest_start_marker(self) -> dict[str, Any] | None:
        try:
            marker = _read_json(self.latest_start_path)
        except PERSISTED_RECORD_ERRORS:
            return None
        if not (
            marker.get("schema_version") == SCHEMA_VERSION
            and isinstance(marker.get("run_id"), str)
            and bool(marker["run_id"])
            and isinstance(marker.get("force_recalibrate"), bool)
            and marker.get("plan_id") == self.plan.plan_id
            and marker.get("plan_sha256") == self.plan.plan_sha256
            and marker.get("arm_name") == self.plan.arm_name
            and marker.get("server_identity") == self.server_identity
            and marker.get("calibration_kind") == self.calibration_kind
            and marker.get("reference_point_id") == self.reference_point_id
        ):
            return None
        return marker

    def _latest_start_matches(self, run_id: str, force_recalibrate: bool) -> bool:
        marker = self._read_latest_start_marker()
        return (
            marker is not None
            and marker["run_id"] == run_id
            and marker["force_recalibrate"] is force_recalibrate
        )

    def _recover_successful_start_retry(
        self, run_id: str, force_recalibrate: bool
    ) -> CalibrationStatusSnapshot | None:
        if not self._latest_start_matches(run_id, force_recalibrate):
            return None
        try:
            history = self._validate_record(
                _read_json(self._history_path(run_id)),
                require_current=False,
                allow_future_timestamp=True,
            )
            active = self._validate_record(
                _read_json(self.active_cache_path),
                require_current=False,
                allow_future_timestamp=True,
            )
        except PERSISTED_RECORD_ERRORS:
            return None
        if (
            history != active
            or history["run_id"] != run_id
            or history["force_recalibrate"] is not force_recalibrate
            or not history["was_committed"]
        ):
            return None
        self._run_id = run_id
        self._force_recalibrate = history["force_recalibrate"]
        self._current_stage_index = 0
        self._completed_stages = 0
        self._cache_reused = history["source_run_id"] != run_id
        message = "successful Start retry recovered"
        try:
            self._persist_terminal("SUCCEEDED", message)
        except OSError:
            self._run_id = ""
            self._force_recalibrate = False
            self._cache_reused = False
            raise
        self._state = "SUCCEEDED"
        self._deadline_us = 0
        self._message = message
        return self.status()

    def _persist_terminal(self, state: str, message: str) -> None:
        if state not in TERMINAL_STATES:
            raise ValueError("terminal state required")
        payload = {
            "schema_version": SCHEMA_VERSION,
            "state": state,
            "run_id": self._run_id,
            "force_recalibrate": self._force_recalibrate,
            "current_stage_index": self._current_stage_index,
            "completed_stages": self._completed_stages,
            "cache_reused": self._cache_reused,
            "message": message,
            "plan_id": self.plan.plan_id,
            "plan_sha256": self.plan.plan_sha256,
            "arm_name": self.plan.arm_name,
            "server_identity": self.server_identity,
            "calibration_kind": self.calibration_kind,
            "reference_point_id": self.reference_point_id,
        }
        if state in {"FAILED", "CANCELLED"}:
            _atomic_write_json(self._terminal_result_path(self._run_id), payload)
        _atomic_write_json(self.terminal_state_path, payload)

    def _repair_partial_commit(self, run_id: str) -> bool:
        try:
            active = self._validate_record(
                _read_json(self.active_cache_path),
                require_current=False,
                allow_future_timestamp=True,
            )
        except PERSISTED_RECORD_ERRORS:
            return False
        if active["run_id"] != run_id or active["was_committed"] is not True:
            return False
        _atomic_write_json(self._history_path(run_id), active)
        message = "partially committed calibration recovered"
        self._persist_terminal("SUCCEEDED", message)
        self._state = "SUCCEEDED"
        self._deadline_us = 0
        self._message = message
        _durable_unlink(self.staged_candidate_path)
        return True

    def _read_valid_terminal(
        self, path: Path, *, expected_run_id: str | None = None
    ) -> dict[str, Any] | None:
        try:
            value = _read_json(path)
        except PERSISTED_RECORD_ERRORS:
            return None
        state = value.get("state")
        if (
            value.get("schema_version") != SCHEMA_VERSION
            or not isinstance(state, str)
            or state not in TERMINAL_STATES
            or value.get("plan_id") != self.plan.plan_id
            or value.get("plan_sha256") != self.plan.plan_sha256
            or value.get("arm_name") != self.plan.arm_name
            or value.get("server_identity") != self.server_identity
            or value.get("calibration_kind") != self.calibration_kind
            or value.get("reference_point_id") != self.reference_point_id
            or not isinstance(value.get("force_recalibrate"), bool)
            or not isinstance(value.get("cache_reused"), bool)
        ):
            return None
        try:
            run_id = _non_empty_string(value.get("run_id"), "run_id")
            _non_empty_string(value.get("message"), "message")
        except ValueError:
            return None
        if expected_run_id is not None and run_id != expected_run_id:
            return None
        current = value.get("current_stage_index")
        completed = value.get("completed_stages")
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in (current, completed)):
            return None
        if current > len(self.plan.stages) or completed > len(self.plan.stages):
            return None
        return value

    def _restore_terminal_path(self, path: Path) -> bool:
        value = self._read_valid_terminal(path)
        if value is None:
            return False
        self._state = value["state"]
        self._run_id = value["run_id"]
        self._force_recalibrate = value["force_recalibrate"]
        self._current_stage_index = value["current_stage_index"]
        self._completed_stages = value["completed_stages"]
        self._cache_reused = value["cache_reused"]
        self._message = value["message"]
        self._deadline_us = 0
        return True

    def _restore_terminal(self) -> bool:
        return self._restore_terminal_path(self.terminal_state_path)

    def _restore_persistent_state(self) -> None:
        if self._restore_terminal():
            return
        try:
            record = self._validate_record(
                _read_json(self.staged_candidate_path),
                require_current=False,
                allow_future_timestamp=True,
            )
        except PERSISTED_RECORD_ERRORS:
            return
        if record["was_committed"]:
            return
        if self._restore_terminal_path(self._terminal_result_path(record["run_id"])):
            return
        self._run_id = record["run_id"]
        self._force_recalibrate = record["force_recalibrate"]
        self._state = "READY_TO_APPLY"
        self._current_stage_index = len(self.plan.stages)
        self._completed_stages = len(self.plan.stages)
        self._deadline_us = 0
        self._message = "staged calibration candidate recovered"

    def start(self, run_id: str, *, force_recalibrate: bool) -> CalibrationStatusSnapshot:
        run = _non_empty_string(run_id, "run_id")
        if not isinstance(force_recalibrate, bool):
            raise LifecycleError("INVALID_ARGUMENT", "force_recalibrate must be boolean")
        with self._lock:
            if run == self._run_id:
                if force_recalibrate != self._force_recalibrate:
                    raise LifecycleError("FAILED_PRECONDITION", "conflicting Start retry")
                return self.status()
            if self._state in ACTIVE_STATES:
                raise LifecycleError("FAILED_PRECONDITION", "another calibration run owns the robot")
            marker = self._read_latest_start_marker()
            if (
                marker is not None
                and marker["run_id"] == run
                and marker["force_recalibrate"] is not force_recalibrate
            ):
                raise LifecycleError("FAILED_PRECONDITION", "conflicting Start retry")
            if self._history_path(run).exists():
                recovered = self._recover_successful_start_retry(run, force_recalibrate)
                if recovered is not None:
                    return recovered
                raise LifecycleError("FAILED_PRECONDITION", "run_id already has an immutable result")
            if self._terminal_result_path(run).exists():
                raise LifecycleError("FAILED_PRECONDITION", "run_id already has an immutable result")
            cached = None if force_recalibrate else self._load_valid_active()
            self._persist_start_marker(run, force_recalibrate)
            _durable_unlink(self.staged_candidate_path)
            _durable_unlink(self.terminal_state_path)
            self._run_id = run
            self._force_recalibrate = force_recalibrate
            self._current_stage_index = 0
            self._completed_stages = 0
            self._cache_reused = cached is not None
            if cached is not None:
                alias = dict(cached)
                alias["run_id"] = run
                alias["server_instance_id"] = self.server_instance_id
                alias["force_recalibrate"] = False
                message = "valid calibration cache reused"
                try:
                    _atomic_write_json(self.active_cache_path, alias)
                    _atomic_write_json(self._history_path(run), alias)
                    self._persist_terminal("SUCCEEDED", message)
                except OSError:
                    self._state = "IDLE"
                    self._run_id = ""
                    self._force_recalibrate = False
                    self._current_stage_index = 0
                    self._completed_stages = 0
                    self._cache_reused = False
                    self._deadline_us = 0
                    self._message = ""
                    raise
                self._state = "SUCCEEDED"
                self._deadline_us = 0
                self._message = message
            else:
                self._state = "READY_FOR_STAGE"
                self._deadline_us = self._deadline()
                self._message = "calibration prepared"
            return self.status()

    def advance(self, run_id: str, stage_index: int, stage_id: str) -> tuple[CalibrationStatusSnapshot, dict[str, Any] | None]:
        run = _non_empty_string(run_id, "run_id")
        stage = _non_empty_string(stage_id, "stage_id")
        if isinstance(stage_index, bool) or not isinstance(stage_index, int) or stage_index < 0:
            raise LifecycleError("INVALID_ARGUMENT", "stage_index must be a non-negative integer")
        with self._lock:
            self._expire_command_deadline_locked()
            if run != self._run_id:
                raise LifecycleError("FAILED_PRECONDITION", "stale calibration run_id")
            if stage_index < self._completed_stages:
                expected = self.plan.stages[stage_index].stage_id
                if stage != expected:
                    raise LifecycleError("FAILED_PRECONDITION", "completed stage_id mismatch")
                return self.status(), None
            if (
                stage_index == self._current_stage_index
                and stage_index < len(self.plan.stages)
                and self._state in {"MOVING", "SETTLING", "COLLECTING", "RETURNING"}
            ):
                expected = self.plan.stages[stage_index].stage_id
                if stage != expected:
                    raise LifecycleError("FAILED_PRECONDITION", "active stage_id mismatch")
                return self.status(), None
            if self._state != "READY_FOR_STAGE" or stage_index != self._current_stage_index:
                raise LifecycleError("FAILED_PRECONDITION", "stage is not ready for release")
            expected_stage = self.plan.stages[stage_index]
            if stage != expected_stage.stage_id:
                raise LifecycleError("FAILED_PRECONDITION", "stage_id mismatch")
            self._state = "MOVING"
            self._deadline_us = 0
            self._message = f"stage {stage_index} accepted"
            return self.status(), deepcopy(expected_stage.arm_payload)

    def set_phase(self, run_id: str, state: str, message: str = "") -> CalibrationStatusSnapshot:
        if state not in {"MOVING", "SETTLING", "COLLECTING", "FITTING", "RETURNING"}:
            raise ValueError("unsupported calibration worker phase")
        with self._lock:
            allowed = {
                "MOVING": {"MOVING", "SETTLING", "COLLECTING", "RETURNING"},
                "SETTLING": {"SETTLING", "COLLECTING"},
                "COLLECTING": {"COLLECTING", "RETURNING"},
                "FITTING": {"FITTING", "RETURNING"},
                "RETURNING": {"RETURNING"},
            }
            if run_id != self._run_id or state not in allowed.get(self._state, set()):
                raise LifecycleError("FAILED_PRECONDITION", "stale calibration worker")
            self._state = state
            self._deadline_us = 0
            self._message = message
            return self.status()

    def complete_stage(self, run_id: str, stage_index: int) -> CalibrationStatusSnapshot:
        with self._lock:
            if run_id != self._run_id or stage_index != self._current_stage_index:
                raise LifecycleError("FAILED_PRECONDITION", "stale stage completion")
            if self._state not in {"COLLECTING", "RETURNING"}:
                raise LifecycleError("FAILED_PRECONDITION", "stage is not executing")
            self._completed_stages += 1
            self._current_stage_index += 1
            if self._completed_stages < len(self.plan.stages):
                self._state = "READY_FOR_STAGE"
                self._deadline_us = self._deadline()
                self._message = "next stage ready"
            else:
                self._state = "FITTING"
                self._deadline_us = 0
                self._message = "all stages collected"
            return self.status()

    def stage_candidate(
        self,
        run_id: str,
        *,
        parameters: Mapping[str, Any],
        quality: Mapping[str, Any],
        calibrated_at_timestamp_us: int | None = None,
    ) -> CalibrationStatusSnapshot:
        with self._lock:
            if run_id != self._run_id or self._completed_stages != len(self.plan.stages):
                raise LifecycleError("FAILED_PRECONDITION", "calibration collection is incomplete")
            if self._state not in {"FITTING", "RETURNING"}:
                raise LifecycleError("FAILED_PRECONDITION", "candidate cannot be staged in this state")
            calibrated = self.now_us() if calibrated_at_timestamp_us is None else calibrated_at_timestamp_us
            if isinstance(calibrated, bool) or not isinstance(calibrated, int) or calibrated <= 0:
                raise LifecycleError("INVALID_ARGUMENT", "invalid calibration timestamp")
            if calibrated > self.now_us():
                raise LifecycleError("INVALID_ARGUMENT", "calibration timestamp is in the future")
            checked_quality = _strict_json_object(quality, "quality")
            if checked_quality.get("passed") is not True:
                raise LifecycleError("FAILED_PRECONDITION", "calibration quality gate did not pass")
            record = self._base_record(
                run_id=run_id,
                source_run_id=run_id,
                calibrated_at_us=calibrated,
                parameters=parameters,
                quality=checked_quality,
                was_committed=False,
            )
            if self.validate_parameters is not None:
                self.validate_parameters(record["parameters"])
            _atomic_write_json(self.staged_candidate_path, record)
            self._state = "READY_TO_APPLY"
            self._deadline_us = 0
            self._message = "candidate durably staged"
            return self.status()

    def commit(self, run_id: str, plan_sha256: str) -> CalibrationStatusSnapshot:
        run = _non_empty_string(run_id, "run_id")
        digest = _non_empty_string(plan_sha256, "plan_sha256")
        with self._lock:
            if run != self._run_id or digest != self.plan.plan_sha256:
                raise LifecycleError("FAILED_PRECONDITION", "stale Commit request")
            if self._state == "SUCCEEDED":
                return self.status()
            if self._state != "READY_TO_APPLY":
                raise LifecycleError("FAILED_PRECONDITION", "candidate is not ready to apply")
            try:
                record = self._validate_record(
                    _read_json(self.staged_candidate_path),
                    require_current=False,
                    allow_future_timestamp=True,
                )
            except PERSISTED_RECORD_ERRORS as exc:
                raise LifecycleError("FAILED_PRECONDITION", "staged candidate is unavailable") from exc
            if record["run_id"] != run:
                raise LifecycleError("FAILED_PRECONDITION", "staged candidate run mismatch")
            record["was_committed"] = True
            _atomic_write_json(self.active_cache_path, record)
            _atomic_write_json(self._history_path(run), record)
            message = "calibration committed"
            self._persist_terminal("SUCCEEDED", message)
            self._state = "SUCCEEDED"
            self._deadline_us = 0
            self._message = message
            _durable_unlink(self.staged_candidate_path)
            return self.status()

    def get_staged_candidate(self, run_id: str) -> dict[str, Any]:
        """Return the validated current candidate used by Commit."""
        run = _non_empty_string(run_id, "run_id")
        with self._lock:
            if run != self._run_id or self._state != "READY_TO_APPLY":
                raise LifecycleError("FAILED_PRECONDITION", "candidate is not ready to apply")
            try:
                record = self._validate_record(
                    _read_json(self.staged_candidate_path),
                    require_current=False,
                    allow_future_timestamp=True,
                )
            except PERSISTED_RECORD_ERRORS as exc:
                raise LifecycleError(
                    "FAILED_PRECONDITION", "staged candidate is unavailable"
                ) from exc
            if record["run_id"] != run or record["was_committed"]:
                raise LifecycleError("FAILED_PRECONDITION", "staged candidate mismatch")
            return deepcopy(record)

    def repair_partial_commit(self, run_id: str) -> CalibrationStatusSnapshot | None:
        """Finish a commit whose authoritative active record reached disk."""
        run = _non_empty_string(run_id, "run_id")
        with self._lock:
            if run != self._run_id:
                raise LifecycleError("NOT_FOUND", "unknown calibration run_id")
            return self.status() if self._repair_partial_commit(run) else None

    def request_cancel(self, run_id: str) -> CalibrationStatusSnapshot:
        run = _non_empty_string(run_id, "run_id")
        with self._lock:
            if run != self._run_id:
                raise LifecycleError("NOT_FOUND", "unknown calibration run_id")
            if self._repair_partial_commit(run):
                return self.status()
            if self._state in TERMINAL_STATES:
                return self.status()
            if self._state == "READY_TO_APPLY":
                message = "staged candidate discarded"
                self._persist_terminal("CANCELLED", message)
                self._state = "CANCELLED"
                self._deadline_us = 0
                self._message = message
                _durable_unlink(self.staged_candidate_path)
            else:
                self._state = "CANCELLING"
                self._deadline_us = 0
                self._message = "cancellation requested"
            return self.status()

    def finish_cancel(self, run_id: str, *, message: str = "calibration cancelled") -> CalibrationStatusSnapshot:
        checked_message = _non_empty_string(message, "message")
        with self._lock:
            if run_id != self._run_id or self._state != "CANCELLING":
                raise LifecycleError("FAILED_PRECONDITION", "calibration is not cancelling")
            self._persist_terminal("CANCELLED", checked_message)
            self._state = "CANCELLED"
            self._deadline_us = 0
            self._message = checked_message
            return self.status()

    def fail(self, run_id: str, message: str) -> CalibrationStatusSnapshot:
        checked_message = _non_empty_string(message, "message")
        with self._lock:
            if run_id != self._run_id or self._state in TERMINAL_STATES:
                raise LifecycleError("FAILED_PRECONDITION", "stale calibration worker")
            if self._repair_partial_commit(run_id):
                return self.status()
            self._persist_terminal("FAILED", checked_message)
            self._state = "FAILED"
            self._deadline_us = 0
            self._message = checked_message
            _durable_unlink(self.staged_candidate_path)
            return self.status()

    def _expire_command_deadline_locked(self) -> None:
        if (
            self._state == "READY_FOR_STAGE"
            and self._deadline_us
            and self.now_us() >= self._deadline_us
        ):
            self._state = "CANCELLING"
            self._deadline_us = 0
            self._message = "coordinator command deadline expired"

    def expire_command_deadline(self) -> CalibrationStatusSnapshot:
        with self._lock:
            self._expire_command_deadline_locked()
            return self.status()

    def get_result(self, run_id: str = "") -> dict[str, Any]:
        with self._lock:
            if not run_id:
                record = self._load_valid_active()
                if record is None:
                    return {"available": False, "run_id": "", "message": "no valid applied calibration"}
                marker = self._read_latest_start_marker()
                commit_incomplete = self._state != "SUCCEEDED" and (
                    (bool(self._run_id) and record["run_id"] == self._run_id)
                    or (marker is not None and record["run_id"] == marker["run_id"])
                )
                if commit_incomplete:
                    # active.json is the authoritative repair marker and is
                    # intentionally written before history/terminal state. It
                    # is not externally current until the commit transaction
                    # reaches SUCCEEDED (and the manager can atomically install
                    # the matching runtime applier under its publication lock).
                    return {
                        "available": False,
                        "run_id": "",
                        "message": "calibration commit is incomplete",
                    }
                return {"available": True, **record, "message": ""}
            if run_id == self._run_id and self._state not in {"SUCCEEDED"}:
                return {"available": False, "run_id": run_id, "message": self._message}
            try:
                record = self._validate_record(
                    _read_json(self._history_path(run_id)),
                    require_current=False,
                    allow_future_timestamp=True,
                )
            except PERSISTED_RECORD_ERRORS as exc:
                terminal = self._read_valid_terminal(
                    self._terminal_result_path(run_id), expected_run_id=run_id
                )
                if terminal is None or terminal["state"] not in {"FAILED", "CANCELLED"}:
                    raise LifecycleError("NOT_FOUND", "unknown calibration result") from exc
                return {
                    "available": False,
                    "run_id": run_id,
                    "message": terminal["message"],
                }
            return {"available": True, **record, "message": ""}


def bilateral_results_compatible(
    left: Mapping[str, Any],
    right: Mapping[str, Any],
    *,
    now_us: int | None = None,
) -> bool:
    """Return whether two current-result snapshots may enable inference."""
    if left.get("arm_name") != "left" or right.get("arm_name") != "right":
        return False
    if not left.get("available") or not right.get("available"):
        return False
    if left.get("was_committed") is not True or right.get("was_committed") is not True:
        return False
    for key in (
        "run_id",
        "source_run_id",
        "plan_id",
        "plan_sha256",
        "calibration_kind",
        "reference_point_id",
    ):
        value = left.get(key)
        if not isinstance(value, str) or not value or right.get(key) != value:
            return False
    current = _unix_us() if now_us is None else now_us
    for result in (left, right):
        expires = result.get("expires_at_timestamp_us")
        if isinstance(expires, bool) or not isinstance(expires, int) or current >= expires:
            return False
    return True
