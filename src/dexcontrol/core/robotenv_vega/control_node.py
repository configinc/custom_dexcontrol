"""Vega lifecycle and actions above the existing arm/body mode calculations."""

import time
from contextlib import contextmanager

import numpy as np

from ..control.actions import ActionRunner
from ..control.config import load_config
from ..control.fsm import ControlConflict, Node, State
from ..control.motion_safety import MotionSafety
from ..control.safety import Safety
from ..control.torque_limit import ArmTorqueLimiter
from .body_control import BodyControl


class VegaControlNode(BodyControl):
    """A single lifecycle and final output decision for the complete robot."""

    def __init__(self, hardware, config=None, clock=time.monotonic, model_worker=None):
        self.lifecycle = Node()
        self.config = config or load_config()
        self.safety = Safety(self.config.safety)
        self.gate = hardware.gate
        self.arm_torque = ArmTorqueLimiter(
            self.gate.devices,
            self.config.arm_torque,
            self.config.safety["arm_torque"].enabled,
            1.0 / hardware.control_hz,
            clock,
        )
        collision = (
            hardware.create_collision_model(self.config)
            if any(
                self.config.safety[name].enabled
                for name in ("self_collision", "static_stability")
            )
            else None
        )
        self.motion_safety = MotionSafety(
            self.gate.devices, self.config, collision, clock, model_worker
        )
        self.normal_owner = object()
        self.hold_owner = object()
        self.actions = ActionRunner(self.gate)
        self.gate.select(self.hold_owner)
        with self.gate.producing(self.hold_owner):
            super().__init__(hardware, clock)
        self.TIMEOUT = self.config.body_timeout_s
        self.PERIOD = 1.0 / hardware.control_hz
        self._next_body_tick = self.clock()
        self.gate.commit(self._check_hold_commands)

    def state(self):
        with self.lock:
            state = super().state()
            state["limits"]["watchdog_seconds"] = self.TIMEOUT
            fault = self.lifecycle.fault
            if fault is not None:
                state["error"] = fault.message
            state.update(
                control_state=self.lifecycle.state.value,
                fault_reason=fault.reason if fault else "",
                fault_action=fault.action if fault else "",
                action=self.actions.snapshot(),
                collision=self.motion_safety.snapshot("self_collision"),
                stability=self.motion_safety.snapshot("static_stability"),
                arm_torque=self.arm_torque.snapshot(),
            )
            return state

    def start(self):
        with self.lock:
            self.hardware.validate_state()
            self._require_action_finished()
            self.lifecycle.start()
            self.hardware.capture_head_reference()
            return self.state()

    def _require_action_finished(self):
        if self.actions.busy:
            raise ControlConflict("Previous action is still running or cancelling")

    def _check_transition(self):
        self.lifecycle.require(State.RUNNING)
        if self.mode == "action":
            raise ControlConflict("Cancel the current action before changing mode")
        self._require_action_finished()

    def select(self, mode):
        if mode not in ("body", "prepare_arms"):
            raise ValueError("Expected body or prepare_arms")
        try:
            with self.gate.producing(self.hold_owner):
                return super().select(mode)
        except ControlConflict:
            raise
        except Exception as exc:
            with self.lock:
                self.fault("execution_error", str(exc))
            raise

    def _hold(self, error=None):
        self.arm_torque.reset()
        self.motion_safety.reset()
        self.gate.select(self.hold_owner)
        with self.gate.producing(self.hold_owner):
            super()._hold(error)
        self.gate.commit(self._check_hold_commands)

    def enable_arms(self):
        with self.lock:
            self.lifecycle.require(State.RUNNING)
            try:
                result = super().enable_arms()
            except ControlConflict:
                raise
            except Exception as exc:
                self.fault("execution_error", str(exc))
                raise
            self.gate.select(self.normal_owner)
            return result

    def set_command(self, values, shoulder_target=None):
        self.lifecycle.require(State.RUNNING)
        super().set_command(values, shoulder_target)

    def accept_arm(self, side):
        self.lifecycle.require(State.RUNNING)
        super().accept_arm(side)

    def halt(self):
        with self.lock:
            self.actions.cancel()
            self.hardware.capture_head_reference()
            self.gate.select(self.hold_owner)
            self._hold()
            return self.state()

    def stop(self):
        with self.lock:
            self.halt()
            if self.lifecycle.state in (State.IDLE, State.RUNNING):
                self.lifecycle.stop()
            # Stop never clears a latched fault.
            return self.state()

    def fault(self, reason, message):
        rule = self.safety.rules[reason]
        if self.safety.report(reason, str(message), self.clock()) != "fault":
            return False
        self.lifecycle.fail(reason, message, rule.fault_action)
        self.actions.cancel()
        self.hardware.capture_head_reference()
        self._hold(str(message))
        return True

    def reset_fault(self):
        with self.lock:
            self.lifecycle.require(State.FAULT)
            if self.actions.busy:
                raise ControlConflict("Action cancellation is still in progress")
            self.hardware.reset_fault()
            self.hardware.validate_state()
            if not self.hardware.stationary():
                raise ControlConflict("Robot must be stopped before resetting a fault")
            self.hardware.capture_head_reference()
            self._hold()
            self.lifecycle.reset_fault()
            return self.state()

    def close(self):
        self.motion_safety.close()

    def _check_hold_commands(self, commands, effects=None):
        # A hold remains possible inside either safety margin, including FAULT.
        self._check_commands(commands, effects, check_model=False)

    def _check_commands(self, commands, effects=None, check_model=True):
        # Check the complete batch before publishing any component or gripper.
        for name, command in commands.items():
            device = self.gate.devices[name]
            for key, value in command.items():
                if (
                    value.shape != (len(device.joint_name),)
                    or not np.isfinite(value).all()
                ):
                    raise ValueError(f"Invalid {key} command for {name}")
        if check_model:
            if self.lifecycle.state == State.RUNNING and self.mode in (
                "arms",
                "body",
                "action",
            ):
                for message in self.arm_torque.filter(commands):
                    self.safety.report("arm_torque", message, self.clock())
            event = self.motion_safety.filter(
                commands,
                effects if effects is not None else {},
                self.gate.owner,
                arm_mode=self.mode == "arms",
            )
            guard = self.motion_safety.guard
            if guard and guard.guarded and guard.active is None:
                self.arm_torque.suspend()
            if event:
                if event == "rejected":
                    status = self.motion_safety.snapshot()
                    for name in status["violations"]:
                        self.safety.report(name, status["reason"], self.clock())
                # Clear producer buffers after committing the measured hold.
                self._motion_event = event
        self.arm_torque.record_output(commands)

    def _handle_motion_event(self):
        event, self._motion_event = getattr(self, "_motion_event", ""), ""
        if not event:
            return
        self.arm_torque.reset()
        self.hardware.clear_motion_targets(rejected=event == "rejected")
        if event == "rejected" and self.mode == "action":
            # Actions cannot wait indefinitely at a rejected waypoint. Cancel
            # them; ordinary teleoperation keeps its mode for recovery input.
            self.halt()

    @contextmanager
    def arm_input(self, side):
        self.accept_arm(side)
        with self.gate.producing(self.normal_owner):
            yield

    def start_action(self, name, operation=None):
        with self.lock:
            self.lifecycle.require(State.RUNNING)
            if self.mode == "action":
                raise ControlConflict("An action already owns robot commands")
            self._require_action_finished()
            if operation is None:
                operation = self.hardware.action(name)
            try:
                self._hold()
                self.mode = "action"
                self.actions.start(name, operation)
            except Exception as exc:
                self.fault("action_error", str(exc))
                raise
            return self.state()

    def finish_action(self):
        with self.lock:
            if self.mode != "action":
                return False
            status = self.actions.snapshot()
            if status["state"] == "failed":
                self.fault("action_error", status["error"])
            elif status["state"] in ("succeeded", "cancelled"):
                self._hold()
            else:
                return False
            return True

    def claim_calibration(self, service):
        """Join the same action from either arm's existing calibration RPC."""
        with self.lock:
            self.lifecycle.require(State.RUNNING)
            if self.mode == "action" and self.actions.can_join("force_calibration"):
                return self.actions.add_participant(
                    service.arm_side, service._cancel_force_calibration_motion
                )
            if self.mode != "arms":
                return None
            self._require_action_finished()
            self._hold()
            self.mode = "action"
            self.actions.begin(
                "force_calibration", output_tick=self.hardware.calibration_tick
            )
            return self.actions.add_participant(
                service.arm_side, service._cancel_force_calibration_motion
            )

    def tick(self):
        with self.lock:
            try:
                now = self.clock()
                if self.lifecycle.state == State.FAULT:
                    with self.gate.producing(self.hold_owner):
                        self.hardware.maintain_hold()
                    self.gate.commit(self._check_hold_commands)
                    self.motion_safety.observe(self.hold_owner)
                    return
                self.hardware.validate_state()
                if now >= self.deadline:
                    if self.fault("input_timeout", "Control input expired"):
                        return
                if self.mode == "action":
                    if self.finish_action():
                        return
                    self.actions.tick()
                elif self.mode == "arms":
                    with self.gate.producing(self.normal_owner):
                        self.hardware.arm_tick()
                elif self.mode == "body":
                    if self.gate.owner is not self.normal_owner:
                        self.gate.select(self.normal_owner)
                    if now >= self._next_body_tick:
                        with self.gate.producing(self.normal_owner):
                            self.hardware.move_body(
                                self.command, now - self._last_body_time
                            )
                        self._last_body_time = now
                        self._next_body_tick = now + 0.02
                else:
                    with self.gate.producing(self.hold_owner):
                        self.hardware.maintain_hold()
                tracking_head = False
                if self.lifecycle.state == State.RUNNING and self.mode in (
                    "arms",
                    "body",
                    "action",
                    "hold",
                ):
                    with self.gate.producing(self.gate.owner):
                        tracking_head = self.hardware.head_tick()
                check = (
                    self._check_commands
                    if self.lifecycle.state == State.RUNNING
                    and (self.mode in ("arms", "body", "action") or tracking_head)
                    else self._check_hold_commands
                )
                self.gate.commit(check)
                if check == self._check_hold_commands:
                    self.motion_safety.observe(self.hold_owner)
                self._handle_motion_event()
            except Exception as exc:
                self.fault("execution_error", str(exc))
                raise
