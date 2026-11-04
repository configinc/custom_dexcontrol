"""Vega SDK composition for a single checked command writer."""

import logging
from time import monotonic

import numpy as np

from ..control.actions import ActionCancelled
from ..control.collision import CoalCollisionModel
from ..control.output import OutputGate
from .body_hardware import VegaHardware


class ManagedVegaHardware(VegaHardware):
    def __init__(self, services, config):
        self.config = config
        self.gate = OutputGate()
        self.control_hz = services[0]._control_loop_hz or 200
        if any((s._control_loop_hz or 200) != self.control_hz for s in services):
            raise ValueError("Managed arms must share one output frequency")
        super().__init__(services)
        components = {
            "left_arm": self.robot.left_arm,
            "right_arm": self.robot.right_arm,
            "torso": self.torso,
            "steering": self.chassis.chassis_steer,
            "drive": self.chassis.chassis_drive,
        }
        if getattr(self.robot, "head", None) is not None:
            components["head"] = self.robot.head
        for service in services:
            hand = service._robot.hand
            if hand is not None and hasattr(hand, "bind_command_sink"):
                components[f"{service.arm_side}_hand"] = hand
        for name, device in components.items():
            self.gate.attach(name, device)
        for service in services:
            service._robot._stop_gripper_worker()
            service._robot.command_gate = self.gate
        self.capture_head_reference()

    def capture_head_reference(self):
        """Keep the current viewing direction when control starts or stops."""
        head = getattr(self.robot, "head", None)
        if head is not None:
            with self.gate.lock:
                self.head_reference = np.asarray(head.get_joint_pos()).copy()
                self.head_reference -= self.robot.compensate_torso_pitch(
                    np.zeros(3), "head"
                )

    def head_tick(self):
        """Track measured torso pitch through the normal checked output batch."""
        head = getattr(self.robot, "head", None)
        if head is None:
            return False
        target = np.asarray(
            self.robot.compensate_torso_pitch(self.head_reference, "head")
        )
        if head.joint_pos_limit is not None:
            target = np.clip(
                target, head.joint_pos_limit[:, 0], head.joint_pos_limit[:, 1]
            )
        head.set_joint_pos(target)
        return True

    def create_collision_model(self, config):
        collision = config.collision
        model = CoalCollisionModel.for_robot(
            self.robot._robot_info.urdf_path,
            collision.collision_urdf,
            collision.srdf,
            collision.excluded_links,
            collision.max_outer_error_m,
            model_config=config.model,
            robotiq_sides=tuple(
                s.arm_side for s in self.services if s.gripper_type == "robotiq"
            ),
        )
        logging.getLogger(__name__).warning(
            "Coal self-collision coverage: %s. Unconfigured tools, cables and payloads "
            "are not covered. Explicitly excluded links: %s. "
            "Model: %s; %d solid shapes, %d shape pairs.",
            ", ".join(model.covered_links),
            ", ".join(model.excluded_links),
            model.urdf_path,
            len(model.shape_meshes),
            len(model.shape_pairs),
        )
        return model

    def start_arms(self):
        # The common tick owns output; RPC/IK work remains on the RPC threads.
        for service in self.services:
            service._cancel_move.clear()
            service._robot.reset_filter_state()

    def arm_tick(self):
        for service in self.services:
            service._robot.execute_interpolated_tick(raise_errors=True)

    def calibration_tick(self):
        for service in self.services:
            with self.gate.lock:
                if (
                    service._force_calibration_owns_robot.is_set()
                    and not service._calibration_direct_reset
                ):
                    service._robot.execute_interpolated_tick(raise_errors=True)

    def clear_motion_targets(self, rejected):
        """Drop buffered motion after the writer has published a measured hold."""
        for service in self.services:
            robot = service._robot
            with robot._interp_lock:
                robot.reset_filter_state()
                robot._latest_hand_action = None
                robot._latest_gripper_action = 0.0
                robot._clear_gripper_command_queue()
            if service.gripper_type == "robotiq":
                robot.hand.stop_motion()
        self.wheel_command[:] = 0
        if rejected:
            # Retain the accepted UI goal, but cancel its execution. A new
            # target from the operator is required to choose a recovery motion.
            self.torso_path = None
            self.torso_command = np.asarray(self.torso.get_joint_pos()).copy()
            self.torso_elapsed = 0.0

    def validate_state(self):
        for name, device in self.gate.devices.items():
            q = np.asarray(device.get_joint_pos())
            dq = np.asarray(device.get_joint_vel())
            if (
                q.shape != (len(device.joint_name),)
                or dq.shape != q.shape
                or not np.isfinite(q).all()
                or not np.isfinite(dq).all()
            ):
                raise ValueError(f"Invalid measured state for {name}")
        for service in self.services:
            if service.gripper_type == "robotiq":
                service._robot.hand.check_error()

    def reset_fault(self):
        for service in self.services:
            if service.gripper_type == "robotiq":
                service._robot.hand.reset_error()

    def action(self, name):
        actions = {"initial_pose": self.initial_pose, "fold": self.fold}
        if name not in actions:
            raise ValueError(
                f"Unknown action: {name!r}; available: {', '.join(actions)}"
            )
        if name == "fold":
            self._fold_targets()  # Reject invalid endpoints before claiming output.
        return actions[name]

    def _fold_targets(self):
        targets = {}
        for name, values in self.config.poses["folded"].items():
            device = self.gate.devices[name]
            target = np.asarray(values, dtype=float)
            limits = device.joint_pos_limit
            if (
                target.shape != (len(device.joint_name),)
                or not np.isfinite(target).all()
            ):
                raise ValueError(f"Invalid folded joint positions for {name}")
            if (
                limits is None
                or np.any(target < limits[:, 0])
                or np.any(target > limits[:, 1])
            ):
                raise ValueError(f"Folded {name} exceeds joint limits")
            self._fold_speeds(name)
            targets[name] = target.copy()
        return targets

    def _fold_speeds(self, name):
        device = self.gate.devices[name]
        limits = np.asarray(device.joint_vel_limit, dtype=float)
        if (
            limits.shape != (len(device.joint_name),)
            or not np.isfinite(limits).all()
            or np.any(limits <= 0)
        ):
            raise ValueError(f"Fold requires valid velocity limits for {name}")
        settings = self.config.actions["fold"]
        speed = (
            settings.torso_joint_velocity_rad_s
            if name == "torso"
            else settings.arm_joint_velocity_rad_s
        )
        speeds = np.minimum(limits, speed)
        if name == "torso":
            speeds = np.minimum(speeds, self.torso_model.velocity_limits)
        return speeds

    def fold(self, cancelled):
        """Fold left arm, right arm, then torso; keep the current grasp."""
        targets = self._fold_targets()
        held = {
            name: np.asarray(
                self.gate.devices[name].get_joint_pos(), dtype=float
            ).copy()
            for name in targets
        }
        for stage, moving in (
            ("left_arm", ("left_arm",)),
            ("right_arm", ("right_arm",)),
            ("torso", ("torso",)),
        ):
            if cancelled.is_set():
                raise ActionCancelled("Fold cancelled")
            if stage == "torso":
                self.torso_target = targets["torso"].copy()
                self.shoulder_target = self.torso_model.forward(self.torso_target)
            self._fold_stage(stage, moving, targets, held, cancelled)
            held.update({name: targets[name].copy() for name in moving})
        self.prepare_arms()

    def _fold_stage(self, stage, moving, targets, held, cancelled):
        settings = self.config.actions["fold"]
        devices = {name: self.gate.devices[name] for name in held}
        start = np.concatenate([devices[name].get_joint_pos() for name in moving])
        goal = np.concatenate([targets[name] for name in moving])
        speeds = np.concatenate([self._fold_speeds(name) for name in moving])
        delta = goal - start
        duration = float(np.max(np.abs(delta) / speeds))
        lengths = [len(devices[name].joint_name) for name in moving]
        offsets = np.cumsum([0, *lengths])
        elapsed, settled_at = 0.0, None
        started = last = monotonic()
        while True:
            if cancelled.is_set():
                raise ActionCancelled(f"Fold {stage} cancelled")
            now = monotonic()
            if now - started >= settings.stage_timeout_s:
                raise TimeoutError(f"Fold {stage} did not reach its target")
            measured = np.concatenate(
                [devices[name].get_joint_pos() for name in moving]
            )
            velocity = np.concatenate(
                [devices[name].get_joint_vel() for name in moving]
            )
            if (
                measured.shape != goal.shape
                or velocity.shape != goal.shape
                or not np.isfinite([measured, velocity]).all()
            ):
                raise ValueError(f"Invalid fold {stage} feedback")
            if duration:
                # Reuse the body's measured-progress/100 ms lead behavior, so
                # a delayed actuator cannot let the reference race to the goal.
                progress = (
                    float(
                        np.clip(
                            np.dot(measured - start, delta) / np.dot(delta, delta), 0, 1
                        )
                    )
                    * duration
                )
                elapsed = min(
                    elapsed + now - last, max(elapsed, progress + self.TORSO_LOOKAHEAD)
                )
                reference = start + min(1.0, elapsed / duration) * delta
            else:
                reference = goal
            last = now
            commands = {name: pos.copy() for name, pos in held.items()}
            commands.update(
                {
                    name: reference[offsets[i] : offsets[i + 1]]
                    for i, name in enumerate(moving)
                }
            )
            # Stage the complete arm/torso batch together. The normal writer
            # still owns collision/stability approval and I/O.
            with self.gate.lock:
                for name, pos in commands.items():
                    speed = (
                        self._fold_speeds(name)
                        if name == "torso" and name in moving
                        else np.zeros_like(pos)
                    )
                    devices[name].set_joint_pos_vel(pos, speed)
                self.torso_command = commands["torso"].copy()
            reached = (
                elapsed >= duration
                and np.max(np.abs(measured - goal)) <= settings.position_tolerance_rad
                and np.max(np.abs(velocity)) <= settings.velocity_tolerance_rad_s
            )
            settled_at = (
                (now if settled_at is None else settled_at) if reached else None
            )
            if settled_at is not None and now - settled_at >= settings.settle_time_s:
                return
            cancelled.wait(1.0 / self.control_hz)

    def initial_pose(self, cancelled):
        # The existing arm waypoint order is retained; nothing runs at startup.
        for service in self.services:
            if cancelled.is_set():
                raise ActionCancelled("Initial pose cancelled")
            self.initialize_gripper(service, cancelled)
            service._cancel_move.clear()
            with service._cmd_lock:
                service._execute_reset_sequence(
                    service.reset_joints,
                    cancelled=cancelled,
                    interruptible_gripper=True,
                )
            if cancelled.is_set():
                raise ActionCancelled("Initial pose cancelled")
        head = getattr(self.robot, "head", None)
        if head is not None:
            # Head motion continues on the common tick, including after this
            # action completes. Head tracking error is not an action failure.
            with self.gate.lock:
                if cancelled.is_set():
                    raise ActionCancelled("Initial pose cancelled")
                target = self.robot._head_init_pos
                if target is None:
                    target = head.get_predefined_pose("home")
                self.head_reference = np.asarray(target).copy()
        self.prepare_arms()

    def initialize_gripper(self, service, cancelled):
        if service.gripper_type == "robotiq":
            service._robot.hand.initialize(
                cancelled,
                lambda operation: self.gate.defer(
                    f"{service.arm_side}_gripper_activation", operation
                ),
            )

    def hold(self):
        super().hold()
        head = getattr(self.robot, "head", None)
        if head is not None:
            self.head_hold = np.asarray(head.get_joint_pos()).copy()
            head.set_joint_pos(self.head_hold)
        for service in self.services:
            hand = service._robot.hand
            if service.gripper_type == "robotiq":
                self.gate.defer(f"{service.arm_side}_gripper_stop", hand.stop_motion)
            elif hand is not None:
                hand.set_joint_pos(hand.get_joint_pos())

    def maintain_hold(self):
        super().maintain_hold()
        head = getattr(self.robot, "head", None)
        if head is not None:
            head.set_joint_pos(self.head_hold)
