"""Expose one bimanual Vega as a Loop Robot Node.

This is the Node-Graph migration of the previous ``LoopRobotClient`` bridge. It
keeps the device-side behavior in this repository: one shared physical Vega,
two per-arm RobotEnv services, one pre-action observation snapshot, and the
same per-arm ``Step`` calls.

- ``_LockedStepService`` is the upstream ``VegaRobotEnvService`` plus one fix: it
  serializes ``Step`` on the upstream ``_cmd_lock`` (upstream guards only ``Reset``),
  so the bus action lane can't race a Reset on shared IK/filter state.
- ``VegaRobotNode`` publishes one typed bimanual observation, receives one typed
  bimanual action, and dispatches each arm's slice to that arm's Step.

A bimanual robot is ONE ``Robot`` exposing both arms; two per-arm services share it
(``VegaRobot``/service take an injected ``robot``), reusing every per-arm
gain/frame/interpolation/IK/gripper path verbatim.
"""

from __future__ import annotations

import contextlib
import logging
import signal
import threading
import time
from collections.abc import Callable, Iterator, Mapping
from typing import Any, Literal, Sequence, cast

from loop_sdk import (
    LifecycleState,
    NodeConnectionConfig,
    ReceivedMessage,
    RobotCommand,
    RobotNode,
    TensorValue,
)
from pydantic import BaseModel, ConfigDict, Field, model_validator

from dexcontrol.core.robotenv_vega import server as _vega_server
from loop_bridge.contracts import (
    LEFT_ARM,
    RIGHT_ARM,
    ROBOT_ACTION_CONTRACT,
    ROBOT_OBSERVATION_CONTRACT,
    action_info_shape,
    float64_tensor,
    float64_values,
    is_action_info_scalar,
)
from loop_bridge.obs_publisher import merge_observations
from loop_bridge.robot_obs import observation_state

LOGGER = logging.getLogger("loop_bridge.vega")

DEFAULT_ACTION_SPACE = "target_cartesian_delta"


class VegaRobotNodeConfig(BaseModel):
    """Hardware settings supplied by the Cell Config before Start."""

    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    frame_type: Literal["vega-1-pro_torso_frame_v1", "vega-1-pro_torso_frame_v2"] = (
        Field(
            default="vega-1-pro_torso_frame_v1",
            description="Arm startup and Home pose preset; does not rotate the coordinate frame.",
        )
    )
    control_hz: int = Field(default=20, gt=0)
    observation_frequency_hz: float = Field(default=20.0, gt=0)
    gripper_type: Literal["default", "robotiq", "sr_gripper"] = "robotiq"
    left_gripper_device: str = Field(
        default="/dev/ttyUSB0",
        min_length=1,
        description="Left Robotiq serial path or SR EtherCAT interface; ignored for built-in grippers.",
    )
    right_gripper_device: str = Field(
        default="/dev/ttyUSB1",
        min_length=1,
        description="Right Robotiq serial path or SR EtherCAT interface; ignored for built-in grippers.",
    )

    @model_validator(mode="after")
    def _distinct_gripper_devices(self) -> VegaRobotNodeConfig:
        if (
            self.gripper_type != "default"
            and self.left_gripper_device == self.right_gripper_device
        ):
            raise ValueError(
                "Left and right grippers must use different devices/interfaces"
            )
        if self.gripper_type == "sr_gripper" and any(
            "/" in device
            for device in (self.left_gripper_device, self.right_gripper_device)
        ):
            raise ValueError(
                "SR grippers require EtherCAT interface names, not serial device paths"
            )
        return self


# Preserve the deployed controller tuning; only Node Config fields vary per cell.
_SERVICE_DEFAULTS: dict[str, Any] = {
    "robot_model": "vega_1",
    "ik_solver_type": "pink",
    "use_velocity_feedforward": True,
    "interpolation_method": "linear",
    "interpolation_history": 3,
    "control_loop_hz": 200,
    "filter_type": "none",
    "vel_smoothing_alpha": 1.0,
    "hw_correction_alpha": 0.5,
    "max_delta_scale": 3.0,
    "max_jerk": 0.0,
    "rot_sensitivity": 2.0,
    "vel_ratio": 1.0,
}


class _BusStepContext:
    """Minimal gRPC servicer context for replaying Step in-process.

    The Vega ``Step`` happy path never touches the context; we still surface an
    ``abort`` (which it would only call on a hard error) as an exception so the
    action lane skips that tick rather than silently succeeding.
    """

    def set_code(self, code: Any) -> None:
        self.code = code

    def set_details(self, details: str) -> None:
        self.details = details

    def abort(self, code: Any, details: str) -> None:
        raise RuntimeError(f"Step aborted: {code} {details}")


def _encode_pre_action_state(
    obs: Mapping[str, Any],
) -> dict[str, Any]:
    """Encode a loop-side obs dict into ``StepRequest.pre_action_state`` proto values.

    This is the bridge boundary where loop's ``obs`` becomes the Vega server's
    ``pre_action_state``. Mirrors ``VegaRobotEnvService._to_proto_value`` — list /
    ndarray → ``FloatArray``; int / bool / float → ``float_value``; otherwise
    stringify. Skips keys whose value is ``None`` so an absent reading doesn't
    spill onto the wire as a zero.
    """
    pb2: Any = _vega_server.robotenv_pb2
    encoded: dict[str, Any] = {}
    for key, value in obs.items():
        if value is None:
            continue
        if isinstance(value, (list, tuple)):
            encoded[key] = pb2.Value(
                float_array=pb2.FloatArray(values=[float(v) for v in value])
            )
            continue
        if isinstance(value, bool):
            # bool is int in Python — must precede the int branch to encode correctly.
            encoded[key] = pb2.Value(float_value=float(value))
            continue
        if isinstance(value, (int, float)):
            encoded[key] = pb2.Value(float_value=float(value))
            continue
        # ndarray path — mirror the Vega server's fallback shape.
        try:
            encoded[key] = pb2.Value(
                float_array=pb2.FloatArray(values=[float(v) for v in value])
            )
        except TypeError:
            encoded[key] = pb2.Value(string_value=str(value))
    return encoded


def _decode_action_info(
    action_info: Mapping[str, Any],
) -> dict[str, Any]:
    """Decode ``StepResponse.action_info`` proto values into a plain dict.

    The server computes these (desired_velocity, delta_action, resolved cartesian,
    ...) from the action against the pre-apply state and returns them per Step; the
    reverse of ``_encode_pre_action_state``. Drops the ``state.*`` entries the server
    flattens in — they duplicate the obs snapshot loop already publishes for this tick.
    """
    decoded: dict[str, Any] = {}
    for key, value in action_info.items():
        if key.startswith("state."):
            continue
        kind = value.WhichOneof("kind")
        if kind == "float_array":
            decoded[key] = [float(v) for v in value.float_array.values]
            continue
        if kind == "float_value":
            decoded[key] = float(value.float_value)
            continue
        if kind == "int_value":
            decoded[key] = int(value.int_value)
            continue
        if kind == "string_value":
            decoded[key] = value.string_value
    return decoded


class _StepApplier:
    """Adapts one arm service's ``Step`` to the action consumer's ``step(...)`` seam."""

    def __init__(self, service: Any) -> None:
        self._service = service

    def step(
        self,
        action: list[float],
        action_space: str,
        gripper_action_space: str,
        *,
        pre_apply_obs: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Step one arm and return the server-computed action info for this tick.

        The returned dict holds the auxiliary values the server derived while
        applying the action (desired_velocity, delta_action, resolved cartesian, ...),
        keyed by their bare names; the caller namespaces them per arm before merging
        onto the published robot-obs. Empty if the server sent no action info.
        """
        pb2: Any = _vega_server.robotenv_pb2
        request = pb2.StepRequest(
            action=list(action),
            action_space=action_space,
            gripper_action_space=gripper_action_space,
        )
        # Both arms apply against the snapshot captured before this action.
        if pre_apply_obs is not None:
            for key, value in _encode_pre_action_state(pre_apply_obs).items():
                request.pre_action_state[key].CopyFrom(value)
        response = self._service.Step(request, _BusStepContext())
        # Step swallows hardware faults into a non-SUCCESS StepResponse.status rather
        # than raising, so the action lane would otherwise treat a stalled command
        # (joint limit, IK failure, comms) as success. Surface it.
        status = getattr(response, "status", "") or "UNKNOWN"
        if status != "SUCCESS":
            LOGGER.warning(
                "robot-action Step returned %s: %s",
                status,
                getattr(response, "message", ""),
            )
            raise RuntimeError(
                f"robot-action Step returned {status}: {getattr(response, 'message', '')}"
            )
        return _decode_action_info(response.action_info)

    def home(self) -> None:
        """Home this arm: ``Reset(mode="home")`` — moves it to its home pose.

        The operational counterpart to ``step``; surfaces a non-SUCCESS reset the
        same way (the bus would otherwise treat a stalled home as success).
        """
        pb2: Any = _vega_server.robotenv_pb2
        request = pb2.ResetRequest(mode="home", params={})
        context = _BusStepContext()
        response = self._service.Reset(request, context)
        status = getattr(response, "status", "") or "UNKNOWN"
        if status != "SUCCESS":
            details = getattr(response, "message", "") or getattr(
                context, "details", ""
            )
            raise RuntimeError(f"home Reset returned {status}: {details}")


class _LockedStepService(_vega_server.VegaRobotEnvService):
    """Serialize arm commands and let the Node own the control-loop lifecycle."""

    def __init__(self, **kwargs: Any) -> None:
        try:
            super().__init__(auto_start_control_loop=False, **kwargs)
        except BaseException:
            robot = getattr(self, "_robot", None)
            if robot is not None:
                robot.close()
            raise
        self.control_error: Exception | None = None
        self._paused = False

    def Step(self, request, context):
        with self._cmd_lock:
            return super().Step(request, context)

    def resume(self) -> None:
        self._paused = False
        self.control_error = None
        self._robot.reset_filter_state()
        self._robot._start_gripper_worker()
        if self._robot.interpolation_enabled and self._control_loop_hz > 0:
            self._start_control_loop()

    def pause(self) -> None:
        if self._paused:
            return
        self._stop_control_loop()
        with self._cmd_lock:
            self._robot.pause_commands()
        self._paused = True

    def close(self) -> None:
        self._stop_control_loop()
        self._robot.close()

    def _control_loop_run(self) -> None:
        period_s = 1.0 / self._control_loop_hz
        while not self._control_loop_stop.is_set():
            started = time.perf_counter()
            try:
                sent = self._robot.execute_interpolated_tick()
                if sent and not self._robot._prev_command_successful:
                    raise RuntimeError(f"{self.arm_side} interpolated command failed")
            except Exception as error:
                self.control_error = error
                self._control_loop_stop.set()
                return
            self._control_loop_stop.wait(
                max(0.0, period_s - (time.perf_counter() - started))
            )


ArmServices = Sequence[tuple[str, Any]]
ServiceFactory = Callable[
    [VegaRobotNodeConfig], contextlib.AbstractContextManager[ArmServices]
]


class VegaRobotNode(RobotNode[VegaRobotNodeConfig]):
    """One Robot Node; hardware opens on Start and stays connected across Stop."""

    def __init__(
        self,
        arm_services: ArmServices | None = None,
        *,
        service_factory: ServiceFactory | None = None,
    ) -> None:
        if (arm_services is None) == (service_factory is None):
            raise ValueError("Provide arm services or a service factory")
        self._service_factory = service_factory
        self._config: VegaRobotNodeConfig | None = None
        self._opened_config: VegaRobotNodeConfig | None = None
        self._resources = contextlib.ExitStack()
        self._arm_services: tuple[tuple[str, Any], ...] = ()
        self._appliers: dict[str, _StepApplier] = {}
        if arm_services is not None:
            self._set_services(arm_services)
            for _arm, service in arm_services:
                self._resources.callback(service.close)
        self._device_lock = threading.RLock()
        self._observation_stop = threading.Event()
        self._observation_thread: threading.Thread | None = None
        self._observation_hz = 20.0
        self._action_space = DEFAULT_ACTION_SPACE
        self._gripper_action_space = "position"
        self._latest_action_info: dict[str, Any] = {}
        self._active = False
        super().__init__(
            config_type=VegaRobotNodeConfig,
            observation_contract=ROBOT_OBSERVATION_CONTRACT,
            action_contract=ROBOT_ACTION_CONTRACT,
            action_handler=self._apply_action,
            command_handler=self._handle_command,
            input_capacity=1,
            on_configure=self._on_configure,
            on_start=self._on_start,
            on_stop=self._on_stop,
            on_reset_fault=self._on_reset_fault,
            on_shutdown=self._on_shutdown,
        )

    def _set_services(self, services: ArmServices) -> None:
        services = tuple(services)
        arms = tuple(arm for arm, _service in services)
        if arms != (LEFT_ARM, RIGHT_ARM):
            raise ValueError(
                f"VegaRobotNode requires ordered left/right arm services, received {arms}"
            )
        self._arm_services = services
        self._appliers = {arm: _StepApplier(service) for arm, service in services}

    def _on_configure(self, config: VegaRobotNodeConfig) -> None:
        self._config = config
        self._observation_hz = config.observation_frequency_hz

    def _on_start(self) -> None:
        config = self._config
        if config is None:
            raise RuntimeError("Configure must succeed before Start")
        try:
            with self._device_lock:
                if (
                    self._service_factory is not None
                    and self._opened_config is not None
                ):
                    previous = self._opened_config.model_dump(
                        exclude={"observation_frequency_hz"}
                    )
                    current = config.model_dump(exclude={"observation_frequency_hz"})
                    if previous != current:
                        self._close_services()
                if not self._arm_services:
                    if self._service_factory is None:
                        raise RuntimeError("Arm services are closed")
                    self._set_services(
                        self._resources.enter_context(self._service_factory(config))
                    )
                self._opened_config = config
                self._latest_action_info = {}
                for _arm, service in self._arm_services:
                    service.resume()
                self._observation_stop.clear()
                self._active = True
                self._read_and_publish()
                self._observation_thread = threading.Thread(
                    target=self._observation_loop,
                    name="vega-robot-observation",
                    daemon=True,
                )
                self._observation_thread.start()
        except BaseException:
            self._on_shutdown()
            raise

    def _on_stop(self) -> None:
        with self._device_lock:
            self._active = False
            self._observation_stop.set()
            thread = self._observation_thread
            self._observation_thread = None
        errors: list[Exception] = []
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=2.0)
            if thread.is_alive():
                errors.append(RuntimeError("Observation thread did not stop"))
        with self._device_lock:
            for arm, service in self._arm_services:
                try:
                    service.pause()
                except Exception as error:
                    errors.append(RuntimeError(f"{arm}: {error}"))
        if errors:
            raise RuntimeError("; ".join(map(str, errors)))

    def _on_reset_fault(self) -> None:
        self._on_stop()

    def _close_services(self) -> None:
        try:
            self._resources.close()
        finally:
            self._arm_services = ()
            self._appliers = {}
            self._opened_config = None

    def _on_shutdown(self) -> None:
        try:
            self._on_stop()
        finally:
            self._close_services()

    def _fault(self, code: str, error: Exception) -> None:
        self.report_fault(code, str(error))
        try:
            self._on_stop()
        except Exception:
            LOGGER.exception("Failed to pause Vega after %s", code)

    def check_health(self) -> None:
        """Called by the host loop so control-thread faults stop both arms."""
        with self._device_lock:
            if not self._active:
                return
            error = next(
                (
                    service.control_error
                    for _, service in self._arm_services
                    if service.control_error is not None
                ),
                None,
            )
        if error is not None:
            self._fault("robot_control_failed", error)
        elif self.status.lifecycle is LifecycleState.FAULT:
            self._on_stop()

    def _observation_loop(self) -> None:
        period_s = 1.0 / self._observation_hz
        while not self._observation_stop.wait(period_s):
            try:
                with self._device_lock:
                    if not self._active:
                        return
                    self._read_and_publish()
            except Exception as error:
                self._fault("robot_observation_failed", error)
                return

    def _read_observations(self) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
        observations: dict[str, Mapping[str, Any]] = {}
        for arm, service in self._arm_services:
            observation, _sample_timestamp_us = service._create_observation()
            observations[arm] = observation
        return observations, merge_observations(observations)

    def _read_and_publish(self) -> None:
        _observations, payload = self._read_observations()
        payload.update(self._latest_action_info)
        self._publish_observation(payload)

    def _publish_observation(self, payload: Mapping[str, Any]) -> None:
        # Samples carry wall-clock capture time; scheduling uses a monotonic clock.
        self.publish_observation(payload, timestamp_ns=time.time_ns())

    def _apply_action(self, message: ReceivedMessage) -> None:
        """Apply against one pre-action state; retain diagnostics for the next observation."""
        try:
            with self._device_lock:
                if not self._active:
                    return
                observations, _payload = self._read_observations()
                payload: dict[str, Any] = {}
                actions = _decode_bimanual_action(message)
                payload["received_action"] = float64_tensor(
                    (*actions[LEFT_ARM], *actions[RIGHT_ARM]), shape=(14,)
                )
                for arm, applier in self._appliers.items():
                    action_info = applier.step(
                        actions[arm],
                        self._action_space,
                        self._gripper_action_space,
                        pre_apply_obs=observation_state(observations[arm]),
                    )
                    payload.update(_action_info_payload(arm, action_info))
                self._latest_action_info = payload
        except Exception as error:
            self._fault("robot_action_failed", error)
            raise

    def _handle_command(self, command: RobotCommand) -> None:
        if command is not RobotCommand.HOME:
            raise ValueError(f"unsupported robot command: {command.value!r}")
        try:
            with self._device_lock:
                if not self._active:
                    raise RuntimeError("Vega Robot Node is not active")
                for applier in self._appliers.values():
                    applier.home()
                self._latest_action_info = {}
        except Exception as error:
            self._fault("robot_home_failed", error)
            raise


def _decode_bimanual_action(message: ReceivedMessage) -> dict[str, list[float]]:
    actions: dict[str, list[float]] = {}
    for arm in (LEFT_ARM, RIGHT_ARM):
        field = f"{arm}.target_cartesian_delta"
        delta = float64_values(
            cast(TensorValue, message.payload[field]),
            field_name=field,
            shape=(6,),
        )
        gripper_value = message.payload[f"{arm}.gripper_position"]
        if isinstance(gripper_value, bool) or not isinstance(
            gripper_value, (int, float)
        ):
            raise TypeError(f"{arm}.gripper_position must be a scalar")
        gripper = float(gripper_value)
        actions[arm] = [*delta, gripper]
    return actions


def _action_info_payload(arm: str, action_info: Mapping[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for field, value in action_info.items():
        shape = action_info_shape(field)
        if shape is not None:
            payload[f"{arm}.action.{field}"] = float64_tensor(value, shape=shape)
            continue
        if is_action_info_scalar(field):
            payload[f"{arm}.action.{field}"] = float(value)
    return payload


@contextlib.contextmanager
def _open_arm_services(config: VegaRobotNodeConfig) -> Iterator[ArmServices]:
    service_kwargs = {
        **_SERVICE_DEFAULTS,
        "frame_type": config.frame_type,
        "gripper_type": config.gripper_type,
        "control_hz": config.control_hz,
    }
    left_port = config.left_gripper_device
    right_port = config.right_gripper_device
    with contextlib.ExitStack() as resources:
        left = _LockedStepService(
            arm_side="left", **{**service_kwargs, "robotiq_comport": left_port}
        )
        resources.callback(left.close)
        right = _LockedStepService(
            arm_side="right",
            robot=left._robot.robot,
            **{**service_kwargs, "robotiq_comport": right_port},
        )
        resources.callback(right.close)
        yield [(LEFT_ARM, left), (RIGHT_ARM, right)]


def serve_dual_arm(
    *,
    node_id: str,
    connection: NodeConnectionConfig,
) -> None:
    """Register in IDLE; the first Graph Start opens one shared Vega connection."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    node = VegaRobotNode(service_factory=_open_arm_services)
    stop_requested = threading.Event()

    def request_stop(_signum: int, _frame: Any) -> None:
        stop_requested.set()

    previous_handlers = {
        sig: signal.signal(sig, request_stop) for sig in (signal.SIGINT, signal.SIGTERM)
    }
    try:
        node.start(node_id=node_id, connection=connection)
        LOGGER.info(
            "Vega Robot Node ready: node_id=%s; waiting for Loop Start", node_id
        )
        while node.is_running and not stop_requested.is_set():
            node.process_control(timeout_s=0.1)
            node.check_health()
    finally:
        try:
            if (
                node.is_running
                and node.status.lifecycle is not LifecycleState.FINALIZED
            ):
                node.shutdown()
        finally:
            try:
                node._on_shutdown()
            finally:
                try:
                    node.close()
                finally:
                    for sig, handler in previous_handlers.items():
                        signal.signal(sig, handler)
