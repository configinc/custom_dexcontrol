"""Nonblocking output selection around the shared model safety worker."""

from dataclasses import dataclass

import numpy as np

from .model_worker import ModelWorker


def copy_commands(commands):
    return {
        name: {key: value.copy() for key, value in command.items()}
        for name, command in commands.items()
    }


@dataclass
class PendingCheck:
    sampled_at: float
    owner: object
    commands: dict | None
    effects: dict
    valid: bool = True


class MotionGuard:
    def __init__(self, devices, model, config, clock, worker=None):
        self.devices, self.model, self.config, self.clock = (
            devices,
            model,
            config.collision,
            clock,
        )
        required = set(model.joint_names)
        self.modeled = {
            name
            for name, device in devices.items()
            if required.intersection(device.joint_name)
        }
        if required - {j for name in self.modeled for j in devices[name].joint_name}:
            raise ValueError("Missing model safety joint devices")
        self.rules = config.safety
        self.stability_config = config.stability
        self.worker = worker if worker is not None else ModelWorker(model, config)
        self.request = None
        self.result = None
        self.owner = None
        self.arm_mode = False
        self.stop_event = ""
        self.last_check_ms = 0.0
        self.max_check_ms = 0.0
        self.reset()

    def reset(self):
        """Revoke motion on every ownership transition, including a late reply."""
        if self.request is not None:
            self.request.valid = False
            self.request.commands = None
            self.request.effects.clear()
        self.paused = True
        self.guarded = True
        self.active = None
        self.output = None
        self.desired, self.desired_effects = {}, {}
        self.hold = None
        self.last_sample = None
        self.started_at = self.clock()
        self.next_monitor = self.next_approval = 0.0
        self.reason = "Waiting for motion safety monitor"
        self.rejected_reasons = {}
        self.stop_event = ""

    def close(self):
        self.worker.close()

    def snapshot(self, check=None):
        if check and not self.rules[check].enabled:
            return {"state": "disabled"}
        state = (
            "paused"
            if self.paused
            else "monitoring"
            if not self.guarded
            else "guarded"
            if self.active is not None
            else "blocked"
        )
        status = {
            "state": state,
            "blocked": not self.paused and self.guarded and self.active is None,
            "reason": self.reason if self.guarded else "",
            "result_age_s": max(0.0, self.clock() - self.last_sample)
            if self.last_sample is not None
            else None,
            "check_ms": self.last_check_ms,
        }
        if check is None:
            status["violations"] = list(self.result.violations) if self.result else []
            return status
        decision = (
            (
                self.result.collision
                if check == "self_collision"
                else self.result.stability
            )
            if self.result
            else None
        )
        if decision is not None:
            status["reason"] = self.rejected_reasons.get(check, decision.reason)
            if not self.paused and not decision.guarded:
                status.update(state="monitoring", blocked=False)
        if check == "self_collision":
            status.update(
                clearance_m=decision.clearance_m if decision else None,
                pair=list(decision.pair) if decision else [],
                required_margin_m=self.config.margin_m,
            )
        else:
            status.update(
                margin_m=decision.margin_m if decision else None,
                target_margin_m=decision.target_margin_m if decision else None,
                com_m=decision.com_m if decision else [],
                mass_kg=decision.mass_kg if decision else None,
                required_margin_m=self.stability_config.margin_m,
            )
        return status

    def _measured(self):
        holds, named = {}, {}
        for name, device in self.devices.items():
            q = np.asarray(device.get_joint_pos(), dtype=float)
            if q.shape != (len(device.joint_name),) or not np.isfinite(q).all():
                raise ValueError(f"Invalid model safety feedback for {name}")
            holds[name] = (
                {"vel": np.zeros_like(q)}
                if name == "drive"
                else {
                    "pos": self._bounded_position(name, q),
                    "vel": np.zeros_like(q),
                }
            )
            named.update(zip(device.joint_name, q))
        return self.model.positions(named), holds

    def _bounded_position(self, name, target):
        # Match the SDK's output clamp when constructing approval endpoints.
        # Encoder noise at a joint bound is not a separate fault condition.
        limits = self.devices[name].joint_pos_limit
        return (
            target.copy()
            if limits is None
            else np.clip(target, limits[:, 0], limits[:, 1])
        )

    def _target(self, commands, holds):
        batch = copy_commands(commands)
        named = {}
        period = 1.0 / self.config.approval_hz
        for name in self.modeled:
            q = holds[name]["pos"]
            command = batch.get(name, holds[name])
            target = (
                command["pos"].copy()
                if "pos" in command
                else q + command.get("vel", np.zeros_like(q)) * period
            )
            # Guarded motion uses position interpolation, with no arm velocity
            # feedforward that could keep driving beyond the approved endpoint.
            # Torso/head velocity fields are actuator speeds, not feedforward.
            velocity = (
                command.get("vel", np.zeros_like(q)).copy()
                if name in ("torso", "head")
                else np.zeros_like(q)
            )
            target = self._bounded_position(name, target)
            batch[name] = {"pos": target, "vel": velocity}
            named.update(zip(self.devices[name].joint_name, target))
        return batch, self.model.positions(named)

    def observe(self, owner):
        """Refresh measured diagnostics while stopped; never approve a target."""
        now = self.clock()
        if not self.paused or owner is not self.owner or self.arm_mode:
            self.reset()
            self.owner = owner
            self.arm_mode = False
            self.result = None
        decision = self.worker.poll()
        if decision is not None:
            request, self.request = self.request, None
            if request is None:
                raise RuntimeError(
                    "Motion safety worker returned an unsolicited result"
                )
            if (
                request.valid
                and now - request.sampled_at <= self.config.result_timeout_s
            ):
                self.result, self.last_sample = decision, request.sampled_at
                self.last_check_ms = decision.elapsed_ms
                self.max_check_ms = max(self.max_check_ms, decision.elapsed_ms)
        # Hold/IK preparation does not authorize motion. Discard stale replies
        # and sample again; filter() still requires fresh results before moving.
        if self.request is None and now >= self.next_monitor:
            measured, _ = self._measured()
            self.worker.submit(measured, None, False, False, False)
            self.request = PendingCheck(now, owner, None, {})
            self.next_monitor = now + 1.0 / self.config.monitor_hz

    def filter(self, commands, effects, owner, arm_mode=False):
        now = self.clock()
        self.stop_event = ""
        if self.paused or owner is not self.owner or arm_mode != self.arm_mode:
            self.reset()
            self.owner, self.paused, self.started_at = owner, False, now
            self.arm_mode = arm_mode
            self.result = None
        measured, holds = self._measured()
        self.desired.update(copy_commands(commands))
        self.desired_effects.update(effects)
        if self.hold is None:
            self.hold = copy_commands(holds)
        decision = self.worker.poll()
        if decision is not None:
            request, self.request = self.request, None
            if request is None:
                raise RuntimeError(
                    "Motion safety worker returned an unsolicited result"
                )
            if (
                request.valid
                and now - request.sampled_at <= self.config.result_timeout_s
            ):
                previous = self.guarded
                self.result, self.last_sample = decision, request.sampled_at
                self.last_check_ms = decision.elapsed_ms
                self.max_check_ms = max(self.max_check_ms, decision.elapsed_ms)
                self.guarded = decision.guarded
                if self.guarded and not previous:
                    self.active = None
                    self.hold = copy_commands(holds)
                    self.output = None
                    self.reason = "Entered motion safety approach region"
                    self.stop_event = "entered"
                    self.desired.clear()
                    self.desired_effects.clear()
                if not self.guarded:
                    self.rejected_reasons.clear()
                    if previous:
                        # Commands/effects submitted once during startup or
                        # guarded operation must survive the initial hold.
                        # Normal monitoring now permits the latest inputs.
                        commands.update(copy_commands(self.desired))
                        effects.update(request.effects)
                        effects.update(self.desired_effects)
                    self.active, self.reason = None, ""
                elif request.owner is owner and request.commands is not None:
                    if decision.accepted:
                        self.rejected_reasons.clear()
                        self.active = (
                            now,
                            copy_commands(
                                self.output if self.output is not None else holds
                            ),
                            request.commands,
                            request.effects,
                        )
                        self.reason = ""
                    else:
                        self.active = None
                        self.hold = copy_commands(holds)
                        self.output = None
                        self.reason = decision.reason
                        self.rejected_reasons = {
                            name: check.reason
                            for name, check in (
                                ("self_collision", decision.collision),
                                ("static_stability", decision.stability),
                            )
                            if check is not None and check.accepted is False
                        }
                        self.stop_event = "rejected"
                        self.desired.clear()
                        self.desired_effects.clear()
        last = self.last_sample if self.last_sample is not None else self.started_at
        if now - last > self.config.result_timeout_s:
            raise TimeoutError("Motion safety monitor result expired")
        # Never queue commands: the next request samples the latest producer
        # output. Commands captured before a stop/transition cannot reappear.
        if self.request is None and now >= self.next_monitor:
            batch, target, saved_effects = None, None, {}
            if (
                self.guarded
                and (self.desired or self.desired_effects)
                and not self.stop_event
                and now >= self.next_approval
            ):
                batch, target = self._target(self.desired, holds)
                saved_effects = dict(self.desired_effects)
                self.desired_effects.clear()
                self.next_approval = now + 1.0 / self.config.approval_hz
            base_motion = bool(
                batch is not None and np.any(batch.get("drive", {}).get("vel", 0))
            )
            self.worker.submit(
                measured, target, bool(saved_effects), base_motion, self.arm_mode
            )
            self.request = PendingCheck(now, owner, batch, saved_effects)
            self.next_monitor = now + 1.0 / self.config.monitor_hz
        if not self.guarded:
            self.desired_effects.clear()
            self.hold = None
            return
        commands.clear()
        effects.clear()
        if self.active is not None:
            started, start, goal, approved_effects = self.active
            fraction = min(1.0, (now - started) * self.config.approval_hz)
            commands.update(copy_commands(goal))
            for name in self.modeled:
                commands[name]["pos"] = start[name]["pos"] + fraction * (
                    goal[name]["pos"] - start[name]["pos"]
                )
            effects.update(approved_effects)
            approved_effects.clear()
            self.output = copy_commands(commands)
            return
        commands.update(copy_commands(self.hold))
