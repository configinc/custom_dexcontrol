"""SDK adapter for exclusive arm/body control."""
import numpy as np
from .body_control import BodyControl
from .torso_kinematics import TorsoKinematics


class VegaHardware:
    # A 20 ms / 0.004 rad reference can remain inside the torso drive's
    # observed startup deadband. A 100 ms lead starts motion while measured
    # progress still prevents the reference from running away from hardware.
    TORSO_LOOKAHEAD = .1

    def __init__(self, services):
        self.services = services
        self.robot = services[0]._robot.robot
        self.chassis = self.robot.chassis
        self.torso = self.robot.torso
        self.torso_limits = self.torso.joint_pos_limit
        if self.torso_limits is None or self.torso.joint_vel_limit is None:
            raise ValueError("Torso joint limits are required")
        self.torso_model = TorsoKinematics.from_urdf(self.robot._robot_info.urdf_path,
                                                    self.torso_limits, self.torso.joint_vel_limit)
        self.torso_limits = self.torso_model.limits
        for service in self.services:
            service._robot.body_torso_kinematics = self.torso_model
        self.command_limits = np.r_[min(.15, self.chassis.max_lin_vel), min(.15, self.chassis.max_lin_vel),
                                    min(.3, self.chassis.max_ang_vel), .02, .02, .05]
        self.wheel_command = np.zeros(2)
        self.hold_joints = []
        self.stop_timestamps = {}
        self.reset_torso_target()

    def limits(self):
        return {"steering": self.chassis.steering_limits.tolist(),
                "torso": self.torso_limits.tolist(), "command_velocity": self.command_limits.tolist(),
                "torso_joint_velocity": self.torso_model.velocity_limits.tolist(),
                "shoulder_frame": "arm_center", "reference_frame": "base",
                "watchdog_seconds": BodyControl.TIMEOUT}

    def components(self):
        return [self.torso, self.chassis.chassis_steer, self.chassis.chassis_drive,
                *(s._robot.arm for s in self.services)]

    def snapshot(self):
        torso = self.torso.get_joint_pos()
        return {"torso": torso.tolist(),
                "shoulder": self.torso_model.forward(torso).tolist(),
                "torso_target": self.torso_target.tolist(),
                "shoulder_target": self.shoulder_target.tolist(),
                "steering": self.chassis.steering_angle.tolist(),
                "wheel_velocity": self.chassis.chassis_drive.get_joint_vel().tolist()}

    def hold(self):
        # Revoke buffered arm/gripper targets, retaining the physical grasp.
        self.hold_joints = [None] * len(self.services)
        errors = []
        # Brake first; a gripper worker may take time to finish an I/O call.
        try:
            self.wheel_command[:] = 0
            self.chassis.set_motion_state(self.chassis.steering_angle, np.zeros(2))
        except Exception as exc:
            errors.append(str(exc))
        try:
            self.reset_torso_target()
            self.torso.set_joint_pos_vel(self.torso_command, np.zeros(3), wait_time=0)
        except Exception as exc:
            errors.append(str(exc))
        for i, service in enumerate(self.services):
            robot = service._robot
            try:
                service._stop_control_loop()
                with service._cmd_lock:
                    robot.reset_filter_state()
                    q = np.asarray(robot.arm.get_joint_pos(), dtype=float)
                    self.hold_joints[i] = q
                    robot.arm.set_joint_pos_vel(q, np.zeros_like(q))
                    robot._stop_gripper_worker()
            except Exception as exc:
                errors.append(str(exc))
        self.stop_timestamps = {id(c): c.get_timestamp_ns() for c in self.components()}
        if errors:
            raise RuntimeError("; ".join(errors))

    def maintain_hold(self):
        if any(q is None for q in self.hold_joints):
            raise RuntimeError("Arm hold state is incomplete")
        self.chassis.set_motion_state(self.chassis.steering_angle, np.zeros(2))
        self.torso.set_joint_pos_vel(self.torso_command, np.zeros(3), wait_time=0)
        for service, q in zip(self.services, self.hold_joints):
            service._robot.arm.set_joint_pos_vel(q, np.zeros_like(q))

    def stationary(self):
        return all(c.get_timestamp_ns() != self.stop_timestamps.get(id(c))
                   and np.max(np.abs(c.get_joint_vel())) < .01 for c in self.components())

    def prepare_arms(self):
        # One stopped hardware snapshot anchors both reduced IK models. This
        # changes model geometry only; it does not send a motion command.
        joints = self.robot.get_joint_pos_dict(component=["left_arm", "right_arm", "torso"])
        for service in self.services:
            service._robot.ik_controller.refresh_fixed_torso(joints)

    def reset_torso_target(self):
        self.torso_command = np.asarray(self.torso.get_joint_pos(), dtype=float)
        self.torso_target = self.torso_command.copy()
        self.shoulder_target = self.torso_model.forward(self.torso_target)
        self.torso_path = None
        self.torso_elapsed = 0.

    def set_shoulder_target(self, pose):
        pose = np.asarray(pose, dtype=float)
        if np.array_equal(pose, self.shoulder_target):
            return
        # Build everything before committing: a rejected target leaves motion intact.
        path = self.torso_model.plan(self.torso_command, pose)
        self.shoulder_target = pose.copy()
        self.torso_target = path.points[-1].copy()
        self.torso_path = path
        self.torso_elapsed = 0.

    def move_body(self, command, dt):
        vx, vy, wz = command[:3]
        a, b = self.chassis._center_to_wheel_axis_dist, self.chassis._half_wheels_dist
        vectors = ([vx - wz*b, vy + wz*a], [vx + wz*b, vy + wz*a])
        current = self.chassis.steering_angle
        targets = [self.chassis._compute_wheel_control(np.asarray(v), current[i], self.chassis.steering_limits[i])
                   if np.linalg.norm(v) > 1e-6 else (current[i], 0.) for i, v in enumerate(vectors)]
        angles, speeds = np.asarray(targets).T
        speeds *= min(1., self.chassis.max_lin_vel / max(1e-9, np.max(np.abs(speeds))))
        if np.max(np.abs(angles - current)) > .05:
            speeds[:] = 0
            # Brake before steering; do not turn the wheels while still rolling.
            if np.max(np.abs(self.chassis.chassis_drive.get_joint_vel())) > .01:
                angles = current
            self.wheel_command[:] = 0
        elif np.all(command[:3] == 0):
            self.wheel_command[:] = 0
        else:
            self.wheel_command += np.clip(speeds - self.wheel_command, -.2*dt, .2*dt)
        self.chassis.set_motion_state(angles, self.wheel_command, wait_time=0)
        if self.torso_path is None:
            joint_speed = np.zeros(3)
        else:
            # Advance along the validated path with measured joint progress.
            # A slow actuator slows the reference instead of letting the clock
            # run all the way to a distant goal while hardware is still behind.
            progress = self.torso_path.time_at(self.torso.get_joint_pos())
            self.torso_elapsed = float(np.clip(progress + max(dt, self.TORSO_LOOKAHEAD),
                                               self.torso_elapsed,
                                               self.torso_elapsed + dt))
            self.torso_command, _ = self.torso_path.sample(self.torso_elapsed)
            # Torso velocity specifies a positive movement speed, unlike arm
            # velocity feedforward. Keep it available at the final position so
            # delayed joints can finish moving after the planned time elapses.
            joint_speed = self.torso_model.velocity_limits
        self.torso.set_joint_pos_vel(self.torso_command, joint_speed, wait_time=0)
        for service, q in zip(self.services, self.hold_joints):
            service._robot.arm.set_joint_pos_vel(q, np.zeros_like(q))

    def start_arms(self):
        # Use the existing independent per-arm output loops and error handling.
        for service in self.services:
            service._robot._start_gripper_worker()
            service._start_control_loop()
