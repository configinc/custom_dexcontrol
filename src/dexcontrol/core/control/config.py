"""Load explicit policies before opening a robot connection."""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .safety import SafetyRule
from .torque_limit import ArmTorqueConfig, TorqueLimitConfig


@dataclass(frozen=True)
class ControlConfig:
    safety: dict
    body_timeout_s: float
    collision: "CollisionConfig"
    stability: "StabilityConfig"
    model: dict
    poses: dict
    actions: dict
    arm_torque: TorqueLimitConfig


@dataclass(frozen=True)
class JointMoveConfig:
    arm_joint_velocity_rad_s: float
    torso_joint_velocity_rad_s: float
    position_tolerance_rad: float
    velocity_tolerance_rad_s: float
    settle_time_s: float
    stage_timeout_s: float

    def __post_init__(self):
        for name, value in vars(self).items():
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < float("inf")
            ):
                raise ValueError(f"Action {name} must be positive and finite")
        if self.settle_time_s >= self.stage_timeout_s:
            raise ValueError(
                "Action settling time must be shorter than the stage timeout"
            )


@dataclass(frozen=True)
class StabilityConfig:
    margin_m: float
    approach_m: float
    release_m: float

    def __post_init__(self):
        for name in ("margin_m", "approach_m", "release_m"):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 <= value < float("inf")
            ):
                raise ValueError(f"Stability {name} must be nonnegative and finite")
        if not self.margin_m < self.approach_m < self.release_m:
            raise ValueError("Stability distances require margin < approach < release")


@dataclass(frozen=True)
class CollisionConfig:
    margin_m: float
    approach_m: float
    release_m: float
    monitor_hz: float
    approval_hz: float
    result_timeout_s: float
    max_outer_error_m: float
    excluded_links: tuple[str, ...]
    arms_mode_excluded_pairs: tuple[tuple[str, str], ...] = ()
    collision_urdf: str | None = None
    srdf: str | None = None

    def __post_init__(self):
        for pair in self.arms_mode_excluded_pairs:
            if (
                not isinstance(pair, (list, tuple))
                or len(pair) != 2
                or any(not isinstance(link, str) or not link for link in pair)
                or pair[0] == pair[1]
            ):
                raise ValueError("Arm-mode collision exclusions require link pairs")
        for name in (
            "margin_m",
            "approach_m",
            "release_m",
            "monitor_hz",
            "approval_hz",
            "result_timeout_s",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not 0 < value < float("inf")
            ):
                raise ValueError(f"{name} must be positive and finite")
        if not self.margin_m < self.approach_m < self.release_m:
            raise ValueError("Collision distances require margin < approach < release")
        if not 0 < self.approval_hz <= self.monitor_hz <= 200:
            raise ValueError("Collision rates require approval <= monitor <= 200 Hz")
        if self.result_timeout_s <= 1 / self.monitor_hz:
            raise ValueError("Collision result timeout must exceed the monitor period")
        if (
            isinstance(self.max_outer_error_m, bool)
            or not isinstance(self.max_outer_error_m, (float, int))
            or not 0 <= self.max_outer_error_m < float("inf")
        ):
            raise ValueError("Collision outer error must be nonnegative and finite")
        for name in ("collision_urdf", "srdf"):
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str) or not Path(value).is_absolute()
            ):
                raise ValueError(f"{name} must be an absolute path or null")
        if (
            not isinstance(self.excluded_links, (list, tuple))
            or any(
                not isinstance(name, str) or not name for name in self.excluded_links
            )
            or len(set(self.excluded_links)) != len(self.excluded_links)
        ):
            raise ValueError("excluded_links must contain unique link names")


def _validate_model_config(raw):
    required = {
        "robotiq_mount_xyz_m",
        "robotiq_mount_yaw_deg",
        "robotiq_mass_kg",
        "robotiq_com_sensor_m",
        "support_points_xy_m",
        "ground_z_m",
    }
    if set(raw) != required:
        raise ValueError(f"Model config must contain {sorted(required)}")
    for key in ("robotiq_mount_xyz_m", "robotiq_com_sensor_m"):
        value = np.asarray(raw[key], dtype=float)
        if value.shape != (3,) or not np.isfinite(value).all():
            raise ValueError(f"{key} requires three finite values")
    yaw = raw["robotiq_mount_yaw_deg"]
    if set(yaw) != {"left", "right"} or not np.isfinite(list(yaw.values())).all():
        raise ValueError("Gripper yaw requires finite left and right angles")
    mass = raw["robotiq_mass_kg"]
    if (
        isinstance(mass, bool)
        or not isinstance(mass, (int, float))
        or not 0 < mass < float("inf")
    ):
        raise ValueError("Gripper mass must be positive and finite")
    points = np.asarray(raw["support_points_xy_m"], dtype=float)
    if points.shape != (3, 2) or not np.isfinite(points).all():
        raise ValueError("Three finite fixed wheel support points are required")
    if not np.isfinite(raw["ground_z_m"]):
        raise ValueError("Ground Z must be finite")
    return raw


def load_config(path=None):
    path = Path(path) if path else Path(__file__).with_name("vega_control.json")
    raw = json.loads(path.read_text())
    if set(raw) != {
        "safety",
        "body_timeout_s",
        "collision",
        "stability",
        "model",
        "poses",
        "actions",
        "arm_torque",
    }:
        raise ValueError(
            "Control config requires safety, body_timeout_s, collision, stability, model, poses, actions and arm_torque"
        )
    required = {
        "input_timeout",
        "execution_error",
        "action_error",
        "self_collision",
        "static_stability",
        "arm_torque",
    }
    if set(raw["safety"]) != required:
        raise ValueError(f"Safety rules must be exactly {sorted(required)}")
    rules = {name: SafetyRule(**value) for name, value in raw["safety"].items()}
    if rules["arm_torque"].on_violation != "warn":
        raise ValueError("arm_torque corrects position and uses the warn policy")
    for name in ("self_collision", "static_stability"):
        if rules[name].on_violation != "block":
            raise ValueError(f"{name} uses the block policy")
    if any(
        rule.on_violation == "block"
        for name, rule in rules.items()
        if name not in ("self_collision", "static_stability")
    ):
        raise ValueError(
            "Only self-collision and static stability support the block policy"
        )
    for rule in rules.values():
        if rule.fault_action not in (None, "hold"):
            raise ValueError("Vega currently supports fault_action: hold")
    for name in ("execution_error", "action_error"):
        if not rules[name].enabled or rules[name].on_violation != "fault":
            raise ValueError(
                f"{name} must fault; execution cannot continue after an error"
            )
    timeout = raw["body_timeout_s"]
    if not isinstance(timeout, (int, float)) or not 0 < timeout < float("inf"):
        raise ValueError("body_timeout_s must be positive and finite")
    if set(raw["poses"]) != {"folded"} or set(raw["poses"]["folded"]) != {
        "left_arm",
        "right_arm",
        "torso",
    }:
        raise ValueError("poses.folded requires left_arm, right_arm and torso")
    for name, size in (("left_arm", 7), ("right_arm", 7), ("torso", 3)):
        values = np.asarray(raw["poses"]["folded"][name], dtype=float)
        if values.shape != (size,) or not np.isfinite(values).all():
            raise ValueError(f"Folded {name} requires {size} finite joint positions")
    if set(raw["actions"]) != {"fold"}:
        raise ValueError("Action configuration requires fold")
    if set(raw["arm_torque"]) != {"feedback_timeout_s", "arms"}:
        raise ValueError("arm_torque requires feedback_timeout_s and arms")
    return ControlConfig(
        safety=rules,
        body_timeout_s=float(timeout),
        collision=CollisionConfig(**raw["collision"]),
        stability=StabilityConfig(**raw["stability"]),
        model=_validate_model_config(raw["model"]),
        poses=raw["poses"],
        actions={"fold": JointMoveConfig(**raw["actions"]["fold"])},
        arm_torque=TorqueLimitConfig(
            feedback_timeout_s=raw["arm_torque"]["feedback_timeout_s"],
            arms={
                name: ArmTorqueConfig(**settings)
                for name, settings in raw["arm_torque"]["arms"].items()
            },
        ),
    )
