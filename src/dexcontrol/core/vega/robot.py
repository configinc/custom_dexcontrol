"""Vega robot wrapper compatible with RobotEnv service."""

from __future__ import annotations

import logging
import json
import os
import sys
import threading
import time as _time
from queue import Empty, Full, Queue
from pathlib import Path
from typing import Any, Optional

_logger = logging.getLogger("robotenv_vega")

import numpy as np
from scipy.spatial.transform import Rotation as R

from dexcontrol.utils.trajectory_interpolator import TrajectoryInterpolator
from dexcontrol.utils.filters import MultiChannelFilter

# Ensure local sibling repositories are importable in workspace deployments.
# robot.py lives at: <repo>/src/dexcontrol/core/vega/robot.py
#   parents: [0]=vega, [1]=core, [2]=dexcontrol, [3]=src, [4]=<repo>
_REPO_ROOT = Path(__file__).resolve().parents[4]
_DEXCONTROL_SRC = _REPO_ROOT / "src"
_DEXCONTROL_TELEOP = _REPO_ROOT / "examples" / "teleop"
for _path in (_DEXCONTROL_SRC, _DEXCONTROL_TELEOP):
    if _path.exists() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from dexcontrol import Robot
from dexbot_utils.configs import get_robot_config


class _RobotWithCustomHeadPose(Robot):
    """Robot subclass that skips the default head home pose on startup.

    head_init_pos is passed via __init__ so that _set_default_state() moves
    the head directly to the desired position instead of the predefined "home"
    pose.
    """

    def __init__(self, *args, head_init_pos: np.ndarray | None = None, **kwargs):
        self._head_init_pos = head_init_pos
        kwargs.setdefault("auto_shutdown", False)
        super().__init__(*args, **kwargs)

    def _set_default_state(self) -> None:
        estop = getattr(self, "estop", None)
        if estop is not None and estop.is_software_estop_enabled():
            _logger.warning(
                "Software E-Stop is active. "
                "Head cannot be enabled and control features are not functional. "
                "Call robot.estop.deactivate() to release the software E-Stop "
                "before controlling the robot."
            )
            return

        for arm in ["left_arm", "right_arm"]:
            component = getattr(self, arm, None)
            if component is not None:
                component.set_modes(["position"] * 7)

        head = getattr(self, "head", None)
        if head is not None:
            head.set_mode("enable")
            if self._head_init_pos is not None:
                init_pos = self.compensate_torso_pitch(
                    np.asarray(self._head_init_pos, dtype=np.float32).copy(), "head"
                )
                _logger.info("Setting custom head init pos: %s", init_pos.tolist())
            else:
                init_pos = self.compensate_torso_pitch(
                    head.get_predefined_pose("home"), "head"
                )
            init_pos = np.asarray(init_pos, dtype=np.float64)
            lower = np.array([-1.483, -2.792, -1.378], dtype=np.float64)
            upper = np.array([1.483, 2.792, 1.483], dtype=np.float64)
            if (
                init_pos.shape != (3,)
                or not np.isfinite(init_pos).all()
                or np.any(init_pos < lower)
                or np.any(init_pos > upper)
            ):
                raise ValueError(
                    "torso-compensated head startup pose exceeds Vega joint limits"
                )
            head.set_joint_pos(init_pos)

_base_arm_teleop_error = None
try:
    from base_arm_teleop import BaseIKController
except ImportError as e:
    BaseIKController = None
    _base_arm_teleop_error = e

try:
    from dexmotion.utils import robot_utils
except ImportError:
    robot_utils = None


SUPPORTED_ACTION_SPACES = (
    "joint_position",
    "joint_velocity",
    "joint_delta",
    "cartesian_velocity",
    "cartesian_delta",
    "target_cartesian_delta",
)

# Gains applied by Oculus controller to raw deltas before emitting
# cartesian_velocity.  Used to recover the original (gain-free) delta.
_TELEOP_POS_ACTION_GAIN = 5.0
_TELEOP_ROT_ACTION_GAIN = 2.0


class JointLimitExceededError(RuntimeError):
    """Raised when commanded joints exceed robot joint limits."""


class IKFailedError(RuntimeError):
    """Raised when IK cannot solve a requested Cartesian command."""


class CommunicationFailedError(RuntimeError):
    """Raised when arm/hand command transmission fails."""


def _resample_smoothing_alpha(
    alpha: float, source_hz: float, target_hz: float
) -> float:
    """Preserve a filter's wall-clock response when its tick rate changes."""
    clipped = float(np.clip(alpha, 0.0, 0.99))
    if clipped <= 0.0 or float(source_hz) == float(target_hz):
        return clipped
    omega = clipped * float(source_hz) / (1.0 - clipped)
    target_dt = 1.0 / float(target_hz)
    return float((omega * target_dt) / (1.0 + omega * target_dt))


def _resample_first_order_alpha(
    alpha: float, source_hz: float, target_hz: float
) -> float:
    """Preserve first-order EMA wall-clock decay across tick rates."""
    clipped = float(np.clip(alpha, 0.0, 1.0))
    if clipped <= 0.0 or clipped >= 1.0 or source_hz == target_hz:
        return clipped
    return float(1.0 - (1.0 - clipped) ** (float(source_hz) / float(target_hz)))


class VegaRobot:
    _FEEDFORWARD_MAX_AGE_INPUT_PERIODS = 2.0
    """Thin wrapper around dexcontrol Robot for single-arm control."""

    @staticmethod
    def _retry_failed_constructor_cleanup(
        robot: object | None,
        custom_hand: object | None,
    ) -> None:
        """Drain acquired hardware before propagating constructor failure.

        Once a custom hand worker or the base Robot exists, losing the wrapper
        object would also lose the only retry handle.  Keep retrying the same
        strict hand-before-base shutdown contract used by service shutdown.
        """
        for label, resource in (("custom hand", custom_hand), ("base robot", robot)):
            if resource is None:
                continue
            shutdown = getattr(resource, "shutdown", None)
            if not callable(shutdown):
                continue
            while True:
                try:
                    shutdown()
                    break
                except BaseException as exc:  # pylint: disable=broad-except
                    _logger.error(
                        "Vega constructor cleanup for %s is not yet safe; "
                        "retrying: %s",
                        label,
                        exc,
                    )
                    _time.sleep(0.1)

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        control_hz: int = 20,
        gripper_type: str = "default",
        ik_solver_type: str = "pink",
        use_velocity_feedforward: bool = False,
        robotiq_comport: str = "/dev/ttyUSB0",
        wuji_sn: str | None = None,
        wuji_effort_limit: float | None = None,
        ema_alpha: float = 0.0,
        interpolation_method: str = "none",
        interpolation_history: int = 4,
        interpolation_control_hz: int | None = None,
        filter_type: str = "none",
        filter_cutoff_freq: float = 10.0,
        filter_order: int = 2,
        filter_ema_alpha: float = 0.1,
        vel_smoothing_alpha: float = 0.3,
        hw_correction_alpha: float = 0.7,
        max_delta_scale: float = 1.0,
        max_jerk: float = 0.25,
        vel_ratio: float = 1.0,
        vel_damp_thresh: float = 0.05,
        head_init_pos: tuple[float, ...] | list[float] = (1.48, 0.0, -0.3),
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type
        # Legacy 5-finger names, pre-v1/v2 rename. An unmigrated caller (saved
        # shell history, an old unit row) must not fall through to the
        # catch-all below: that starts a server which passes its health check
        # while _hand_dof() collapses to 1 and every hand action is rejected.
        _legacy_gripper_types = {"wuji_hand": "wuji_hand_v1",
                                 "wuji_hand_2": "wuji_hand_v2"}
        if gripper_type in _legacy_gripper_types:
            _canonical = _legacy_gripper_types[gripper_type]
            _logger.warning(
                "gripper_type '%s' is a legacy name — using '%s'. Update the "
                "caller/unit config to the canonical value.",
                gripper_type, _canonical,
            )
            gripper_type = _canonical
        self._robotiq_comport = robotiq_comport
        self._ik_solver_type = ik_solver_type

        if arm_side not in ("left", "right"):
            raise ValueError(f"arm_side must be 'left' or 'right', got: {arm_side}")
        if BaseIKController is None:
            msg = (
                "BaseIKController not found. Ensure custom_dexcontrol/examples/teleop is available "
                "and dependencies are installed (e.g. pip install pytransform3d dexmotion dualsense_controller)."
            )
            if _base_arm_teleop_error is not None:
                msg += f" Original error: {_base_arm_teleop_error}"
            raise ImportError(msg) from _base_arm_teleop_error

        self.robot_model = robot_model
        self.arm_side = arm_side
        self.control_hz = int(control_hz)
        self.interpolation_control_hz = int(
            interpolation_control_hz
            if interpolation_control_hz is not None
            else control_hz
        )
        self.gripper_type = gripper_type
        self.use_velocity_feedforward = bool(use_velocity_feedforward)

        prepared_head_init_pos = None
        if head_init_pos is not None:
            head64 = np.asarray(head_init_pos, dtype=np.float64)
            head_lower = np.array([-1.483, -2.792, -1.378], dtype=np.float64)
            head_upper = np.array([1.483, 2.792, 1.483], dtype=np.float64)
            if (
                head64.shape != (3,)
                or not np.isfinite(head64).all()
                or np.any(head64 < head_lower)
                or np.any(head64 > head_upper)
            ):
                raise ValueError("head_init_pos exceeds Vega head joint limits")
            prepared_head_init_pos = head64.astype(np.float32)
            if not np.isfinite(prepared_head_init_pos).all():
                raise ValueError("head_init_pos is not finite in hardware format")

        configs = get_robot_config(robot_model)
        self.robot = _RobotWithCustomHeadPose(
            configs=configs,
            head_init_pos=prepared_head_init_pos,
        )
        custom_hand = None
        try:
            self.arm = getattr(self.robot, f"{arm_side}_arm")
            hand_component = f"{arm_side}_hand"
            if gripper_type == "robotiq":
                from dexcontrol.core.robotiq_gripper import RobotiqGripper  # lazy import
                self.hand = RobotiqGripper(comport=self._robotiq_comport)
                custom_hand = self.hand
            elif gripper_type == "sr_gripper":
                from dexcontrol.core.sr_gripper import SrGripperAdapter  # lazy import
                self.hand = SrGripperAdapter(comport=self._robotiq_comport)
                custom_hand = self.hand
            elif gripper_type in ("wuji_hand_v1", "wuji_hand_v2"):
                from dexcontrol.core.wuji_hand import WujiHandAdapter  # lazy import
                if wuji_effort_limit is None:
                    wuji_effort_limit = 1.7 if gripper_type == "wuji_hand_v2" else 1.5
                self.hand = WujiHandAdapter(
                    handedness=arm_side,
                    sn=wuji_sn,
                    effort_limit=wuji_effort_limit,
                )
                custom_hand = self.hand
                # The adapter connects whichever wuji generation is attached,
                # but RCI builds its retargeter from the configured type.
                if self.hand.model != gripper_type:
                    raise RuntimeError(
                        f"Unit config says gripper_type={gripper_type!r} but the "
                        f"attached hand is a {self.hand.model!r} "
                        f"(sn={self.hand.serial_number}). Fix the unit config in "
                        f"data-studio or attach the matching hand, then restart."
                    )
            else:
                self.hand = (
                    getattr(self.robot, hand_component)
                    if self.robot.has_component(hand_component)
                    else None
                )

            custom_ik_cfg = self._build_ik_config()
            self.ik_controller = BaseIKController(
                bot=self.robot, visualize=False,
                ik_solver_type=self._ik_solver_type,
                custom_local_ik_config=custom_ik_cfg,
            )
        except BaseException:
            self._retry_failed_constructor_cleanup(self.robot, custom_hand)
            raise
        try:
            self._finish_initialization(
                arm_side=arm_side,
                control_hz=control_hz,
                ema_alpha=ema_alpha,
                interpolation_method=interpolation_method,
                interpolation_history=interpolation_history,
                interpolation_control_hz=self.interpolation_control_hz,
                filter_type=filter_type,
                filter_cutoff_freq=filter_cutoff_freq,
                filter_order=filter_order,
                filter_ema_alpha=filter_ema_alpha,
                vel_smoothing_alpha=vel_smoothing_alpha,
                hw_correction_alpha=hw_correction_alpha,
                max_delta_scale=max_delta_scale,
                max_jerk=max_jerk,
                vel_ratio=vel_ratio,
                vel_damp_thresh=vel_damp_thresh,
            )
        except BaseException:
            self._retry_failed_constructor_cleanup(self.robot, custom_hand)
            raise

    def _finish_initialization(self, **config) -> None:
        """Complete all fallible setup while constructor rollback owns hardware."""
        arm_side = config["arm_side"]
        control_hz = config["control_hz"]
        self._arm_joint_names = [f"{arm_side[0].upper()}_arm_j{i + 1}" for i in range(7)]
        self.reset_joints = np.asarray(
            self.arm.get_predefined_pose("L_shape"), dtype=np.float64
        )
        self.safe_transit_pose = self.reset_joints.copy()
        self._gripper_open_pos, self._gripper_close_pos = self._init_gripper_reference()
        self._gripper_state_lock = threading.Lock()
        self._gripper_io_lock = threading.Lock()
        self._gripper_poll_interval_s = 1.0 / max(1, self.control_hz)
        self._gripper_position = 0.0
        self._gripper_joint_pos = np.asarray(self._gripper_open_pos, dtype=np.float64).copy()
        self._gripper_command_queue: Queue[float] | None = None
        self._gripper_stop_event = threading.Event()
        self._gripper_worker: threading.Thread | None = None
        self._gripper_worker_restart_required = False
        self._hand_shutdown_completed = False
        self._robot_shutdown_completed = False
        self._prev_command_successful = True
        self._prev_gripper_command_successful = True
        self._prev_controller_latency_ms = 0.0
        self._prev_joint_vel: np.ndarray | None = None
        self._vel_smoothing_alpha = float(np.clip(config["vel_smoothing_alpha"], 0.0, 1.0))
        self._interpolation_vel_smoothing_alpha = _resample_first_order_alpha(
            self._vel_smoothing_alpha,
            float(control_hz),
            float(config["interpolation_control_hz"]),
        )
        self._last_cmd_joint_pos: np.ndarray | None = None
        self._prev_cmd_delta: np.ndarray | None = None
        self._HW_CORRECTION_ALPHA = float(np.clip(config["hw_correction_alpha"], 0.0, 1.0))
        self._INTERPOLATION_HW_CORRECTION_ALPHA = _resample_first_order_alpha(
            self._HW_CORRECTION_ALPHA,
            float(control_hz),
            float(config["interpolation_control_hz"]),
        )
        self._HW_CORRECTION_OUTLIER_THRESH = 0.5
        self._max_delta_scale = float(max(0.1, config["max_delta_scale"]))
        self._MOTOR_MAX_JERK_RAD = float(max(0.0, config["max_jerk"]))
        self._vel_ratio = float(max(0.0, config["vel_ratio"]))
        self._vel_damp_thresh = float(max(0.001, config["vel_damp_thresh"]))

        vel_log_path = os.environ.get("VEL_LOG_PATH")
        self._vel_log_file = open(vel_log_path, "a") if vel_log_path else None
        if self._vel_log_file is not None:
            self._vel_log_file.write(
                f"timestamp,{','.join(f'j{i+1}_vel' for i in range(7))}\n"
            )
            _logger.info("Velocity logging enabled: %s", vel_log_path)

        self._ema_alpha = float(np.clip(config["ema_alpha"], 0.0, 0.99))
        self._interpolation_ema_alpha = _resample_smoothing_alpha(
            self._ema_alpha,
            float(control_hz),
            float(config["interpolation_control_hz"]),
        )
        self._filter_pos: np.ndarray | None = None
        self._filter_vel: np.ndarray | None = None
        self._interpolation_method = config["interpolation_method"]
        if self._interpolation_method != "none":
            self._interpolator = TrajectoryInterpolator(
                method=self._interpolation_method,
                history_size=config["interpolation_history"],
            )
        else:
            self._interpolator = None

        filter_type = config["filter_type"]
        if filter_type != "none":
            filter_cfg = {"default": {"type": filter_type}}
            if filter_type == "butterworth":
                filter_cfg["default"].update(
                    cutoff_freq=config["filter_cutoff_freq"],
                    order=config["filter_order"],
                )
            elif filter_type == "ema":
                filter_cfg["default"]["alpha"] = _resample_first_order_alpha(
                    config["filter_ema_alpha"],
                    float(control_hz),
                    float(config["interpolation_control_hz"]),
                )
            self._output_filter = MultiChannelFilter(
                filter_config=filter_cfg,
                control_rate=float(config["interpolation_control_hz"]),
            )
        else:
            self._output_filter = None

        self._latest_target_joint_pos: np.ndarray | None = None
        self._latest_arm_feedforward_velocity: np.ndarray | None = None
        self._latest_command_monotonic: float | None = None
        self._latest_gripper_action = 0.0
        self._latest_hand_action: np.ndarray | None = None
        self._latest_gripper_action_space = "position"
        self._interp_lock = threading.Lock()
        if self.hand is not None:
            self._refresh_gripper_state()
            self._start_gripper_worker()

    _AGENT_DEBUG_LOG_PATH = "/home/dexmate/.cursor/debug-daf7f0.log"
    _AGENT_DEBUG_SESSION_ID = "daf7f0"

    def _agent_debug_log(self, run_id: str, hypothesis_id: str, location: str, message: str, data: dict) -> None:
        payload = {
            "sessionId": self._AGENT_DEBUG_SESSION_ID,
            "runId": run_id,
            "hypothesisId": hypothesis_id,
            "location": location,
            "message": message,
            "data": data,
            "timestamp": int(_time.time() * 1000),
        }
        try:
            with open(self._AGENT_DEBUG_LOG_PATH, "a", encoding="utf-8") as f:
                f.write(json.dumps(payload, ensure_ascii=True, default=float) + "\n")
        except Exception as exc:
            _logger.error(
                "[AgentDebugLogError] run_id=%s message=%s path=%s err=%s",
                run_id, message, self._AGENT_DEBUG_LOG_PATH, exc,
            )

    def _build_ik_config(self):
        """Build an optimized IK config for real-time control."""
        if self._ik_solver_type != "pink":
            return None
        try:
            from dexmotion.configs.ik.ik_config import LocalPinkIKConfig, IKDampingWeightsConfig
            return LocalPinkIKConfig(
                solver_name="local_pink",
                target_frames=["L_ee", "R_ee"],
                dt=1.0 / max(1, self.control_hz),
                avoid_self_collisions=False,
                collision_pairs=0,
                collision_margin=0.01,
                qp_solver="proxqp",
                safety_buffer=0.2,
                damping_weights=IKDampingWeightsConfig(
                    default=1e-3,
                    override={
                        "torso_j1": 30000.0,
                        "torso_j2": 30000.0,
                        "torso_j3": 30000.0,
                        "R_arm_j1": 50,
                        "L_arm_j1": 50,
                        "R_arm_j2": 75,
                        "L_arm_j2": 75,
                        "R_arm_j3": 50.0,
                        "L_arm_j3": 50.0,
                        "R_arm_j4": 50.0,
                        "L_arm_j4": 50.0,
                        "R_arm_j5": 120.0,
                        "L_arm_j5": 120.0,
                        "R_arm_j6": 120.0,
                        "L_arm_j6": 120.0,
                        "R_arm_j7": 120.0,
                        "L_arm_j7": 120.0,
                    },
                ),
            )
        except Exception:
            return None

    def reset_filter_state(self) -> None:
        """Reset smoothing filter and command-tracking state.

        Call before/after large discontinuous motions (e.g. reset sequences)
        to prevent the filter from ramping between old and new positions.
        """
        self._filter_pos = None
        self._filter_vel = None
        self._last_cmd_joint_pos = None
        self._prev_cmd_delta = None
        self._prev_joint_vel = None
        if self._interpolator is not None:
            self._interpolator.clear()
        if self._output_filter is not None:
            self._output_filter.reset()
        self._latest_target_joint_pos = None
        self._latest_arm_feedforward_velocity = None
        self._latest_command_monotonic = None
        self._latest_hand_action = None
        self._latest_gripper_action = 0.0
        self._latest_gripper_action_space = "position"

    def reset_command_state(self) -> None:
        """Drain asynchronous hand work and discard every buffered command.

        Reset/current is a command boundary, not merely a filter boundary.  A
        scalar gripper command may already have left the interpolation loop and
        be waiting in the one-element worker queue (or be executing).  Stop and
        join that worker before clearing interpolation state, then create a
        fresh queue so no pre-reset command can execute afterward.
        """
        worker = getattr(self, "_gripper_worker", None)
        # A retained but no-longer-live handle means a previous drain timed
        # out and is now retryable. Finalize that stale worker/queue through
        # the same stop path, then restore a fresh worker as originally
        # intended.
        restart_worker = bool(
            getattr(self, "_gripper_worker_restart_required", False)
            or worker is not None
        )
        if worker is not None:
            self._gripper_worker_restart_required = True
            self._stop_gripper_worker()
        else:
            self._clear_gripper_command_queue()
        hand_reset = getattr(getattr(self, "hand", None), "reset_command_state", None)
        if callable(hand_reset):
            hand_reset()
        if getattr(self, "hand", None) is not None and not self._refresh_gripper_state():
            raise RuntimeError(
                "fresh hand state is unavailable after command-boundary hold"
            )
        self._prev_gripper_command_successful = True
        self.reset_filter_state()
        if restart_worker:
            self._start_gripper_worker()
            self._gripper_worker_restart_required = False

    def launch_robot(self) -> None:
        """Validate robot readiness and default control mode."""
        self.arm.set_modes(["position"] * 7)
        if self.arm.joint_pos_limit is None:
            raise RuntimeError("Arm joint position limits are unavailable")
        self.reset_filter_state()
        self.sync_motion_manager_with_arm(self.reset_joints)

    def _start_gripper_worker(self) -> None:
        if self.hand is None:
            return
        if self._gripper_worker is not None and self._gripper_worker.is_alive():
            return
        self._gripper_command_queue = Queue(maxsize=1)
        self._gripper_stop_event.clear()
        self._gripper_worker = threading.Thread(
            target=self._gripper_worker_loop,
            name=f"vega-gripper-{self.arm_side}",
            daemon=True,
        )
        self._gripper_worker.start()

    def _stop_gripper_worker(self) -> None:
        self._gripper_stop_event.set()
        self._clear_gripper_command_queue()
        worker = self._gripper_worker
        if worker is not None and worker.is_alive():
            worker.join(timeout=1.0)
            if worker.is_alive():
                # Preserve the handle and queue so a later shutdown retry can
                # prove the worker has exited.  Closing the hand transport
                # underneath an in-flight command is unsafe.
                raise RuntimeError("Vega gripper worker did not stop within 1 second")
        self._gripper_worker = None
        self._gripper_command_queue = None

    def _gripper_worker_loop(self) -> None:
        while not self._gripper_stop_event.is_set():
            latest_target: float | None = None
            if self._gripper_command_queue is not None:
                try:
                    latest_target = self._gripper_command_queue.get(
                        timeout=self._gripper_poll_interval_s
                    )
                    while True:
                        latest_target = self._gripper_command_queue.get_nowait()
                except Empty:
                    pass

            if latest_target is not None:
                if self._gripper_stop_event.is_set():
                    break
                if (
                    not self._prev_gripper_command_successful
                    or not self.hand_command_path_healthy()
                ):
                    # A sticky adapter failure is a command boundary.  Do not
                    # allow an already-dequeued target to reach the adapter
                    # after the server has entered recovery; only the strict
                    # reset/hold path may transmit while recovering.
                    self._prev_gripper_command_successful = False
                else:
                    self._execute_gripper_command(
                        latest_target, wait_time=0.0, raise_on_error=False
                    )

            if self._gripper_stop_event.is_set():
                break
            self._refresh_gripper_state()

    def _clear_gripper_command_queue(self) -> None:
        if self._gripper_command_queue is None:
            return
        while True:
            try:
                self._gripper_command_queue.get_nowait()
            except Empty:
                return

    def _gripper_target_to_joint_pos(self, target: float) -> np.ndarray:
        return self._gripper_open_pos + target * (self._gripper_close_pos - self._gripper_open_pos)

    def _refresh_gripper_state(self) -> bool:
        if self.hand is None:
            return False
        try:
            with self._gripper_io_lock:
                hand_joint_pos = np.asarray(self.hand.get_joint_pos(), dtype=np.float64)
            gripper_position = float(self._normalize_gripper_position(hand_joint_pos))
            with self._gripper_state_lock:
                self._gripper_joint_pos = hand_joint_pos.copy()
                self._gripper_position = gripper_position
            return True
        except Exception:
            self._prev_gripper_command_successful = False
            return False

    def _execute_gripper_command(
        self,
        target: float,
        wait_time: float,
        raise_on_error: bool,
    ) -> bool:
        if self.hand is None:
            return True
        if not np.isfinite(target):
            raise ValueError("Refusing to send non-finite scalar hand target")
        target = float(np.clip(target, 0.0, 1.0))
        target_joint_pos = self._gripper_target_to_joint_pos(target)
        try:
            with self._gripper_io_lock:
                if target <= 1e-3:
                    self.hand.open_hand(wait_time=wait_time)
                elif target >= 1.0 - 1e-3:
                    self.hand.close_hand(wait_time=wait_time)
                else:
                    self.hand.set_joint_pos(target_joint_pos, wait_time=wait_time)
            # State is refreshed from real measurements by
            # `_refresh_gripper_state` only — do not write the commanded
            # target into `_gripper_position` here. Conflating commanded
            # and measured produces logs/plots where state tracks the
            # action and snaps back to reality on contact stalls.
            return True
        except Exception:
            self._prev_gripper_command_successful = False
            if raise_on_error:
                raise
            return False

    def _enqueue_gripper_target(self, target: float) -> None:
        if not np.isfinite(target):
            raise ValueError("Refusing to enqueue non-finite scalar hand target")
        target = float(np.clip(target, 0.0, 1.0))
        if self._gripper_command_queue is None:
            self._execute_gripper_command(target, wait_time=0.0, raise_on_error=False)
            return
        try:
            self._gripper_command_queue.put_nowait(target)
        except Full:
            try:
                self._gripper_command_queue.get_nowait()
            except Empty:
                pass
            try:
                self._gripper_command_queue.put_nowait(target)
            except Full:
                pass

    def get_cached_gripper_position(self) -> float:
        if self.hand is None:
            return 0.0
        with self._gripper_state_lock:
            return float(self._gripper_position)

    def get_cached_gripper_joint_pos(self) -> np.ndarray:
        if self.hand is None:
            return np.zeros(1, dtype=np.float64)
        with self._gripper_state_lock:
            return np.asarray(self._gripper_joint_pos, dtype=np.float64).copy()

    def hand_command_path_healthy(self) -> bool:
        if self.hand is None:
            return True
        outer_before = bool(self._prev_gripper_command_successful)
        health = getattr(self.hand, "command_path_healthy", True)
        adapter_healthy = bool(health() if callable(health) else health)
        # Re-read the outer sticky state after querying the adapter.  This
        # makes the combined predicate a single conservative snapshot: an
        # outer failure concurrent with the adapter read cannot hide between
        # two separately sampled fields at the service boundary.
        outer_after = bool(self._prev_gripper_command_successful)
        return outer_before and adapter_healthy and outer_after

    # gripper_types that expose the end-effector as a multi-DoF joint vector.
    # Everything else — INCLUDING gripper_type "default" with an integrated
    # dexbot F5D6 hand attached — keeps the scalar [0,1] contract exactly as
    # main: reclassifying the dexbot hand by its joint_name alone would change
    # its observation schema and add crash paths for deployments that never
    # opted in.
    _MULTI_DOF_GRIPPER_TYPES = ("vega_hand", "wuji_hand_v1", "wuji_hand_v2")

    def _hand_dof(self) -> int:
        """End-effector DoF: N for an opted-in multi-finger hand, else 1.

        Gated on gripper_type first (see _MULTI_DOF_GRIPPER_TYPES). For opted-in
        hands, prefer the component's joint_name list (cached at init): it is
        the reliable DoF (6 for F5D6, 20 for Wuji) and does NOT depend on live
        hardware state — unlike get_joint_pos(), which may raise until the
        first state arrives and would make this fall back to 1 mid-run,
        rejecting the multi-DoF hand action.
        """
        if self.hand is None or self.gripper_type not in self._MULTI_DOF_GRIPPER_TYPES:
            return 1
        try:
            names = getattr(self.hand, "joint_name", None)
            if names:
                n = len(names)
                return n if n > 1 else 1
        except Exception:
            pass
        try:
            n = int(np.asarray(self.hand.get_joint_pos()).size)
        except Exception:
            n = 1
        return n if n > 1 else 1

    def _split_arm_hand_action(self, action: np.ndarray, arm_dof: int, action_space: str):
        """Split a flat action into (arm_action, hand_action).

        Accepts both the legacy scalar-gripper length (arm_dof+1) and the
        multi-finger length (arm_dof+hand_dof). Returns hand_action as a 1-D
        array (length 1 for scalar gripper, N for a multi-DoF hand).
        """
        hand_dof = self._hand_dof()
        expected = arm_dof + hand_dof
        n = action.shape[0]
        if n == expected:
            return action[:arm_dof], action[arm_dof:expected]
        # Back-compat: a scalar-gripper command (arm_dof+1) is always allowed.
        if n == arm_dof + 1:
            return action[:arm_dof], action[arm_dof:arm_dof + 1]
        raise ValueError(
            f"{action_space} expects {expected} values "
            f"(arm {arm_dof} + hand {hand_dof}), got {n}"
        )

    def _apply_hand_action(
        self,
        hand_action: np.ndarray,
        gripper_action_space: str,
        blocking: bool,
        execution_hz: float | None = None,
    ) -> None:
        """Send the end-effector command.

        Scalar (length 1) → existing normalized gripper pipeline.
        Multi-DoF (length>1) → direct joint-position command on the hand,
        bypassing the [0,1] scalar collapse. Wuji targets are passed through;
        other hands use soft limits when the driver exposes them.
        """
        if hand_action.size <= 1:
            self.update_gripper(
                float(hand_action.reshape(-1)[0]) if hand_action.size else 0.0,
                velocity=(gripper_action_space == "velocity"),
                blocking=blocking,
                execution_hz=execution_hz,
            )
            return
        if self.hand is None:
            return
        target = np.asarray(hand_action, dtype=np.float64)
        try:
            if gripper_action_space == "velocity":
                # Integrate joint velocities (rad/s) into absolute targets —
                # treating a velocity command as absolute radians would snap
                # the hand toward the zero pose.
                current = np.asarray(self.hand.get_joint_pos(), dtype=np.float64).reshape(-1)
                if current.size != target.size:
                    raise ValueError(
                        f"hand velocity command size {target.size} != state size {current.size}"
                    )
                tick_hz = self.control_hz if execution_hz is None else execution_hz
                target = current + target * (1.0 / max(1, tick_hz))
            # Check before clipping: infinities must fail rather than turn
            # into finite boundary commands that the adapter would accept.
            if not np.isfinite(target).all():
                raise ValueError("Hand target contains non-finite values")
            limits = getattr(self.hand, "joint_limits", None)
            is_wuji = getattr(self.hand, "model", None) in ("wuji_hand_v1", "wuji_hand_v2")
            if not is_wuji and limits is not None and np.asarray(limits).shape == (target.size, 2):
                limits = np.asarray(limits, dtype=np.float64)
                target = np.clip(target, limits[:, 0], limits[:, 1])
            tick_hz = self.control_hz if execution_hz is None else execution_hz
            wait_time = (1.0 / max(1, tick_hz)) if blocking else 0.0
            with self._gripper_io_lock:
                self.hand.set_joint_pos(target, wait_time=wait_time)
        except Exception:
            self._prev_gripper_command_successful = False
            raise

    def _prepare_direct_hand_action(
        self, hand_action: np.ndarray, gripper_action_space: str
    ) -> tuple[str, np.ndarray]:
        """Resolve and validate a direct-path hand target before arm dispatch."""
        command = np.asarray(hand_action, dtype=np.float64).reshape(-1)
        if not np.isfinite(command).all():
            raise ValueError("Hand command contains non-finite values")
        if self.hand is None:
            return "none", command
        if command.size <= 1:
            value = float(command[0]) if command.size else 0.0
            if gripper_action_space == "velocity":
                current = self.get_cached_gripper_position()
                if not np.isfinite(current):
                    raise ValueError("Scalar hand state is non-finite")
                value = current + value / max(1, self.control_hz)
            target = np.array([np.clip(value, 0.0, 1.0)], dtype=np.float64)
            return "scalar", target

        target = command.copy()
        if gripper_action_space == "velocity":
            current = np.asarray(self.hand.get_joint_pos(), dtype=np.float64).reshape(-1)
            if current.size != target.size or not np.isfinite(current).all():
                raise ValueError("Multi-DoF hand state is invalid")
            target = current + target / max(1, self.control_hz)
        if not np.isfinite(target).all():
            raise ValueError("Resolved hand target contains non-finite values")
        limits = getattr(self.hand, "joint_limits", None)
        is_wuji = getattr(self.hand, "model", None) in ("wuji_hand_v1", "wuji_hand_v2")
        if not is_wuji and limits is not None and np.asarray(limits).shape == (target.size, 2):
            limits = np.asarray(limits, dtype=np.float64)
            if not np.isfinite(limits).all():
                raise ValueError("Hand joint limits contain non-finite values")
            target = np.clip(target, limits[:, 0], limits[:, 1])
        return "multi", target

    def _dispatch_prepared_direct_hand_action(
        self, kind: str, target: np.ndarray, blocking: bool
    ) -> None:
        if kind == "none":
            return
        if kind == "scalar":
            if blocking:
                self._clear_gripper_command_queue()
                self._execute_gripper_command(
                    float(target[0]),
                    wait_time=1.0 / max(1, self.control_hz),
                    raise_on_error=True,
                )
            else:
                self._enqueue_gripper_target(float(target[0]))
            return
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0
        try:
            with self._gripper_io_lock:
                self.hand.set_joint_pos(target, wait_time=wait_time)
        except Exception:
            self._prev_gripper_command_successful = False
            raise

    def update_command(
        self,
        command: np.ndarray,
        action_space: str,
        gripper_action_space: str = "position",
        blocking: bool = False,
    ) -> None:
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space '{action_space}'")
        if gripper_action_space not in ("position", "velocity"):
            raise ValueError(f"Unsupported gripper_action_space '{gripper_action_space}'")

        action = np.asarray(command, dtype=np.float64).reshape(-1)
        if not np.isfinite(action).all():
            raise ValueError("Command contains non-finite values")
        dt = 1.0 / max(1, self.control_hz)
        state_dict, _ = self.get_robot_state()
        current_joint_pos = np.asarray(state_dict["joint_positions"], dtype=np.float64)
        if current_joint_pos.shape != (7,) or not np.isfinite(current_joint_pos).all():
            raise ValueError("Measured arm joint state is invalid")

        arm_dof = 7 if action_space.startswith("joint") else 6
        arm_action, hand_action = self._split_arm_hand_action(action, arm_dof, action_space)

        if action_space == "joint_position":
            target_joint_pos = arm_action
        elif action_space == "joint_velocity":
            target_joint_pos = current_joint_pos + arm_action * dt
        elif action_space == "joint_delta":
            target_joint_pos = current_joint_pos + arm_action
        elif action_space == "cartesian_velocity":
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3] * dt, arm_action[3:6] * dt)
        else:  # cartesian_delta
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3], arm_action[3:6])
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape != (7,) or not np.isfinite(target_joint_pos).all():
            raise ValueError("Resolved arm target is invalid")
        prepared_hand_kind, prepared_hand_target = self._prepare_direct_hand_action(
            hand_action, gripper_action_space
        )

        # Log per-joint delta from IK (before smoothing/clipping)
        ik_delta = target_joint_pos - current_joint_pos
        if np.max(np.abs(ik_delta)) > 0.005:
            _logger.info(
                "[IKDelta] space=%s delta=%s cart_in=%s",
                action_space,
                np.round(ik_delta, 4).tolist(),
                np.round(arm_action[:6], 4).tolist() if not action_space.startswith("joint") else "n/a",
            )

        # Critically damped 2nd-order smoothing filter.
        # Exact discrete solution — unconditionally stable, zero overshoot.
        # omega (natural frequency) is derived from ema_alpha so the CLI
        # parameter retains a similar "responsiveness" feel.
        if self._ema_alpha > 0.0:
            if self._filter_pos is None:
                self._filter_pos = target_joint_pos.copy()
                self._filter_vel = np.zeros_like(target_joint_pos)
            else:
                alpha = self._ema_alpha
                omega = alpha / (dt * (1.0 - alpha))

                exp_term = np.exp(-omega * dt)
                err = self._filter_pos - target_joint_pos
                c = self._filter_vel + omega * err

                self._filter_pos = target_joint_pos + (err + c * dt) * exp_term
                self._filter_vel = (self._filter_vel - c * omega * dt) * exp_term

                target_joint_pos = self._filter_pos.copy()

        try:
            self.update_joints(target_joint_pos, velocity=False, blocking=blocking)
            self._dispatch_prepared_direct_hand_action(
                prepared_hand_kind, prepared_hand_target, blocking
            )
            self._prev_command_successful = True
        except (JointLimitExceededError, IKFailedError):
            self._prev_command_successful = False
            raise
        except Exception as exc:
            self._prev_command_successful = False
            raise CommunicationFailedError(str(exc)) from exc

    # ------------------------------------------------------------------
    # Interpolation-based API (input-rate / control-rate split)
    # ------------------------------------------------------------------

    def add_command_point(
        self,
        command: np.ndarray,
        action_space: str,
        gripper_action_space: str = "position",
    ) -> None:
        """Buffer a command at input rate for later interpolation.

        Called by the gRPC Step() handler at input rate.  The actual
        motor command is dispatched by the control-loop thread calling
        ``execute_interpolated_tick()``.

        When interpolation is disabled this falls back to the
        synchronous ``update_command()`` path.
        """
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space '{action_space}'")
        if gripper_action_space not in ("position", "velocity"):
            raise ValueError(
                f"Unsupported gripper_action_space '{gripper_action_space}'"
            )
        if self._interpolator is None:
            # No interpolation — direct execution (legacy path).
            self.update_command(command, action_space, gripper_action_space, blocking=False)
            return

        # Resolve action → joint-space target (same logic as update_command).
        action = np.asarray(command, dtype=np.float64).reshape(-1)
        dt = 1.0 / max(1, self.control_hz)
        state_dict, _ = self.get_robot_state()
        current_joint_pos = np.asarray(state_dict["joint_positions"], dtype=np.float64)

        arm_dof = 7 if action_space.startswith("joint") else 6
        arm_action, hand_action = self._split_arm_hand_action(action, arm_dof, action_space)

        if action_space == "joint_position":
            target_joint_pos = arm_action
            arm_feedforward_velocity = None
        elif action_space == "joint_velocity":
            target_joint_pos = current_joint_pos + arm_action * dt
            arm_feedforward_velocity = arm_action.copy()
        elif action_space == "joint_delta":
            target_joint_pos = current_joint_pos + arm_action
            arm_feedforward_velocity = None
        elif action_space == "cartesian_velocity":
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3] * dt, arm_action[3:6] * dt)
            arm_feedforward_velocity = (target_joint_pos - current_joint_pos) / dt
        else:  # cartesian_delta
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3], arm_action[3:6])
            arm_feedforward_velocity = None

        with self._interp_lock:
            resolved_hand_action = np.asarray(
                hand_action, dtype=np.float64
            ).copy()
            if not np.isfinite(resolved_hand_action).all():
                raise ValueError("Hand action contains non-finite values")
            if not np.isfinite(target_joint_pos).all():
                raise ValueError("Arm target contains non-finite values")
            hand = getattr(self, "hand", None)
            is_wuji = getattr(hand, "model", None) in (
                "wuji_hand_v1",
                "wuji_hand_v2",
            )
            resolved_gripper_action_space = gripper_action_space
            if gripper_action_space == "velocity":
                previous_target = self._latest_hand_action
                if (
                    is_wuji
                    or previous_target is None
                    or self._latest_gripper_action_space != "position"
                    or np.asarray(previous_target).size != resolved_hand_action.size
                ):
                    if resolved_hand_action.size <= 1:
                        previous_target = np.array(
                            [self.get_cached_gripper_position()], dtype=np.float64
                        )
                    elif hand is not None:
                        with self._gripper_io_lock:
                            previous_target = np.asarray(
                                hand.get_joint_pos(), dtype=np.float64
                            ).reshape(-1)
                    else:
                        previous_target = np.zeros_like(resolved_hand_action)
                resolved_hand_action = (
                    np.asarray(previous_target, dtype=np.float64).reshape(-1)
                    + resolved_hand_action * dt
                )
                resolved_gripper_action_space = "position"
            if resolved_hand_action.size <= 1:
                resolved_hand_action = np.clip(resolved_hand_action, 0.0, 1.0)
            elif hand is not None and not is_wuji:
                limits = getattr(hand, "joint_limits", None)
                if np.asarray(limits).shape == (resolved_hand_action.size, 2):
                    limits = np.asarray(limits, dtype=np.float64)
                    resolved_hand_action = np.clip(
                        resolved_hand_action, limits[:, 0], limits[:, 1]
                    )
            if not np.isfinite(resolved_hand_action).all():
                raise ValueError("Resolved hand target contains non-finite values")
            # Publish arm and hand together only after all fallible preparation.
            timestamp = _time.perf_counter()
            previous_hand = self._latest_hand_action
            if (
                previous_hand is not None
                and np.asarray(previous_hand).size != resolved_hand_action.size
            ):
                # Interpolator history must have one stable vector schema.
                self._interpolator.clear()
            combined_target = np.concatenate(
                [target_joint_pos, resolved_hand_action]
            )
            self._interpolator.add_point(timestamp, combined_target)
            self._latest_target_joint_pos = target_joint_pos.copy()
            self._latest_command_monotonic = timestamp
            self._latest_arm_feedforward_velocity = (
                None
                if arm_feedforward_velocity is None
                else np.asarray(arm_feedforward_velocity, dtype=np.float64).copy()
            )
            self._latest_hand_action = resolved_hand_action
            self._latest_gripper_action = (
                float(resolved_hand_action.reshape(-1)[0])
                if resolved_hand_action.size
                else 0.0
            )
            self._latest_gripper_action_space = resolved_gripper_action_space

    def execute_interpolated_tick(self) -> bool:
        """Run one control-loop tick using interpolated position.

        Called at control rate (e.g. 100 Hz) by the background control
        thread.  Returns ``True`` if a command was sent, ``False`` if
        there is no data yet.
        """
        if getattr(self, "hand", None) is not None and (
            not self._prev_gripper_command_successful
            or not self.hand_command_path_healthy()
        ):
            raise CommunicationFailedError(
                "previous asynchronous hand dispatch or state refresh failed"
            )
        with self._interp_lock:
            if self._latest_target_joint_pos is None:
                return False

            now = _time.perf_counter()
            query_time = now - (1.0 / max(1, self.control_hz))
            combined_pos, combined_vel = self._interpolator.interpolate(
                query_time,
                compute_velocity=self.use_velocity_feedforward,
            )

            if combined_pos is None:
                # Not enough data for interpolation — use latest raw target.
                latest_hand = self._latest_hand_action
                if latest_hand is None:
                    latest_hand = np.array(
                        [self._latest_gripper_action], dtype=np.float64
                    )
                combined_pos = np.concatenate(
                    [self._latest_target_joint_pos, latest_hand]
                )
                combined_vel = None

            combined_pos = np.asarray(combined_pos, dtype=np.float64).reshape(-1)
            if combined_pos.size < 8:
                raise RuntimeError("Interpolated command is missing its hand target")
            pos = combined_pos[:7].copy()
            hand_action = combined_pos[7:].copy()
            arm_feedforward_velocity = (
                None
                if combined_vel is None
                else np.asarray(combined_vel, dtype=np.float64).reshape(-1)[:7].copy()
            )
            latest_command_monotonic = self._latest_command_monotonic
            if (
                arm_feedforward_velocity is None
                and self._latest_arm_feedforward_velocity is not None
                and latest_command_monotonic is not None
                and now - latest_command_monotonic
                <= self._FEEDFORWARD_MAX_AGE_INPUT_PERIODS
                / max(1, self.control_hz)
            ):
                arm_feedforward_velocity = self._latest_arm_feedforward_velocity.copy()
            if arm_feedforward_velocity is None:
                arm_feedforward_velocity = np.zeros_like(
                    self._latest_target_joint_pos
                )
            gripper_vel = self._latest_gripper_action_space == "velocity"
            feedforward_deadman = (
                latest_command_monotonic is None
                or query_time >= latest_command_monotonic
            )

        # Apply the output filter (Butterworth / EMA) after interpolation.
        if self._output_filter is not None:
            pos = self._output_filter.apply(pos)

        # Apply existing EMA 2nd-order smoothing on top if enabled.
        if self._interpolation_ema_alpha > 0.0:
            dt = 1.0 / max(1, self.interpolation_control_hz)
            if self._filter_pos is None:
                self._filter_pos = pos.copy()
                self._filter_vel = np.zeros_like(pos)
            else:
                alpha = self._interpolation_ema_alpha
                omega = alpha / (dt * (1.0 - alpha))
                exp_term = np.exp(-omega * dt)
                err = self._filter_pos - pos
                c = self._filter_vel + omega * err
                self._filter_pos = pos + (err + c * dt) * exp_term
                self._filter_vel = (self._filter_vel - c * omega * dt) * exp_term
                pos = self._filter_pos.copy()

        try:
            self.update_joints(
                pos,
                velocity=False,
                blocking=False,
                execution_hz=self.interpolation_control_hz,
                feedforward_velocity=arm_feedforward_velocity,
                force_zero_feedforward=feedforward_deadman,
            )
            self._apply_hand_action(
                hand_action,
                "velocity" if gripper_vel else "position",
                blocking=False,
                execution_hz=self.interpolation_control_hz,
            )
            self._prev_command_successful = True
        except (JointLimitExceededError, IKFailedError):
            self._prev_command_successful = False
            raise
        except Exception as exc:
            self._prev_command_successful = False
            _logger.error("Interpolated tick failed: %s", exc)
            raise

        return True

    @property
    def interpolation_enabled(self) -> bool:
        """True when trajectory interpolation is active."""
        return self._interpolator is not None

    # ------------------------------------------------------------------

    # Per-joint maximum delta per control step (radians).
    # j1-j3 (shoulder/base) are kept conservative; j4-j7 (elbow/wrist)
    # get larger limits because the IK solver assigns them larger deltas
    # due to their lower damping weights.
    _MOTOR_MAX_DELTA_RAD_BASE = np.array([
        0.35,   # j1 – base rotation
        0.35,   # j2 – shoulder
        0.35,   # j3 – shoulder
        0.5,    # j4 – elbow
        0.6,    # j5 – wrist
        0.6,    # j6 – wrist
        0.6,    # j7 – wrist
    ], dtype=np.float64)

    @property
    def _MOTOR_MAX_DELTA_RAD(self) -> np.ndarray:
        return self._MOTOR_MAX_DELTA_RAD_BASE * self._max_delta_scale

    _JOINT_LIMIT_TOLERANCE_RAD = 0.01  # ~0.57 deg tolerance for IK numerical precision

    def update_joints(
        self,
        joint_pos_command: np.ndarray,
        velocity: bool = False,
        blocking: bool = False,
        execution_hz: float | None = None,
        feedforward_velocity: np.ndarray | None = None,
        force_zero_feedforward: bool = False,
    ) -> None:
        run_id = f"joint-diagnostics-{self.arm_side}"
        tick_hz = self.control_hz if execution_hz is None else execution_hz
        interpolation_tick = execution_hz is not None
        rate_ratio = float(self.control_hz) / float(tick_hz)
        target_joint_pos = np.asarray(joint_pos_command, dtype=np.float64)
        if target_joint_pos.shape != (7,) or not np.isfinite(target_joint_pos).all():
            raise ValueError("Joint command must contain seven finite values")
        if velocity:
            dt = 1.0 / max(1, tick_hz)
            current_joint_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            if current_joint_pos.shape != (7,) or not np.isfinite(current_joint_pos).all():
                raise ValueError("Measured arm joint state is invalid")
            target_joint_pos = current_joint_pos + target_joint_pos * dt

        # Clip small IK numerical errors within tolerance before hard validation
        limits = self.arm.joint_pos_limit
        if limits is not None:
            low = limits[:, 0].astype(np.float64)
            high = limits[:, 1].astype(np.float64)
            tol = self._JOINT_LIMIT_TOLERANCE_RAD
            target_joint_pos = np.where(
                (target_joint_pos < low) & (target_joint_pos >= low - tol),
                low, target_joint_pos,
            )
            target_joint_pos = np.where(
                (target_joint_pos > high) & (target_joint_pos <= high + tol),
                high, target_joint_pos,
            )

        self.validate_joint_limits(target_joint_pos)
        if limits is not None:
            low = limits[:, 0].astype(np.float64)
            high = limits[:, 1].astype(np.float64)
            margin_low = target_joint_pos - low
            margin_high = high - target_joint_pos
            min_margin = float(np.minimum(margin_low, margin_high).min())
            if min_margin < 0.08:
                # #region agent log
                self._agent_debug_log(
                    run_id=run_id,
                    hypothesis_id="H8",
                    location="dexcontrol/core/vega/robot.py:update_joints",
                    message="near_joint_limit",
                    data={
                        "arm_side": self.arm_side,
                        "min_margin_rad": min_margin,
                        "margin_low_min_rad": float(margin_low.min()),
                        "margin_high_min_rad": float(margin_high.min()),
                        "target_joint_pos": np.round(target_joint_pos, 6).tolist(),
                    },
                )
                # #endregion

        # Gradual correction: use _last_cmd_joint_pos as the base for delta
        # clipping, blending a fraction of hw feedback error each step to
        # prevent drift without causing discontinuous jumps.
        hw_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        if hw_pos.shape != (7,) or not np.isfinite(hw_pos).all():
            raise ValueError("Measured arm joint state is invalid")
        if self._last_cmd_joint_pos is not None:
            raw_error = hw_pos - self._last_cmd_joint_pos
            # Per-joint outlier rejection: skip correction for joints where
            # hw feedback jumps beyond threshold (e.g. glitchy readings).
            outlier_mask = np.abs(raw_error) > self._HW_CORRECTION_OUTLIER_THRESH
            raw_error[outlier_mask] = 0.0
            correction_alpha = (
                self._INTERPOLATION_HW_CORRECTION_ALPHA
                if interpolation_tick
                else self._HW_CORRECTION_ALPHA
            )
            current = self._last_cmd_joint_pos + raw_error * correction_alpha
        else:
            current = hw_pos
        diff = target_joint_pos - current
        raw_diff = diff.copy()
        max_delta = self._MOTOR_MAX_DELTA_RAD * rate_ratio
        clipped_mask = np.abs(diff) > max_delta
        clipped_count = int(np.count_nonzero(clipped_mask))
        if np.any(clipped_mask):
            clipped = np.clip(diff, -max_delta, max_delta)
            clipped_indices = np.where(clipped_mask)[0]
            # #region agent log
            self._agent_debug_log(
                run_id=run_id,
                hypothesis_id="H6",
                location="dexcontrol/core/vega/robot.py:update_joints",
                message="delta_clip_applied",
                data={
                    "arm_side": self.arm_side,
                    "indices": clipped_indices.tolist(),
                    "raw_delta": np.round(diff[clipped_mask], 6).tolist(),
                    "clip_limit": max_delta[clipped_mask].tolist(),
                },
            )
            # #endregion
            _logger.info(
                "[DeltaClip] joints=%s raw=%s limit=%s",
                clipped_indices.tolist(),
                np.round(diff[clipped_mask], 4).tolist(),
                max_delta[clipped_mask].tolist(),
            )
            target_joint_pos = current + clipped
            diff = clipped

        # Jerk limit: restrict how fast the per-step delta can change between
        # consecutive steps.  Smooths acceleration/deceleration to prevent
        # the abrupt velocity changes that cause oscillation on stop.
        jerk_count = 0
        if self._prev_cmd_delta is not None and self._MOTOR_MAX_JERK_RAD > 0:
            accel = diff - self._prev_cmd_delta
            jerk_limit = self._MOTOR_MAX_JERK_RAD * rate_ratio * rate_ratio
            jerk_mask = np.abs(accel) > jerk_limit
            jerk_count = int(np.count_nonzero(jerk_mask))
            if np.any(jerk_mask):
                _logger.debug(
                    "[JerkLimit] joints=%s raw_accel=%s limit=%.6f",
                    np.where(jerk_mask)[0].tolist(),
                    np.round(accel[jerk_mask], 6).tolist(),
                    float(jerk_limit),
                )
                accel = np.clip(accel, -jerk_limit, jerk_limit)
                diff = self._prev_cmd_delta + accel
                target_joint_pos = current + diff
        self._prev_cmd_delta = diff.copy()

        # #region agent log
        self._agent_debug_log(
            run_id=run_id,
            hypothesis_id="H9",
            location="dexcontrol/core/vega/robot.py:update_joints",
            message="joint_update_metrics",
            data={
                "arm_side": self.arm_side,
                "raw_diff_l2": float(np.linalg.norm(raw_diff)),
                "final_diff_l2": float(np.linalg.norm(diff)),
                "clipped_count": clipped_count,
                "jerk_count": jerk_count,
                "min_joint_margin_rad": min_margin if limits is not None else None,
            },
        )
        # #endregion

        if not np.isfinite(target_joint_pos).all():
            raise ValueError("Refusing to send non-finite joint position to hardware")
        if self.use_velocity_feedforward and not blocking:
            dt = 1.0 / max(1, tick_hz)
            dispatched_velocity = (target_joint_pos - current) / dt
            if feedforward_velocity is None:
                raw_vel = dispatched_velocity
            else:
                requested_velocity = np.asarray(
                    feedforward_velocity, dtype=np.float64
                )
                same_direction = requested_velocity * dispatched_velocity > 0.0
                raw_vel = np.where(
                    same_direction,
                    np.sign(requested_velocity)
                    * np.minimum(
                        np.abs(requested_velocity),
                        np.abs(dispatched_velocity),
                    ),
                    0.0,
                )
            if force_zero_feedforward:
                target_joint_vel = np.zeros_like(raw_vel)
            elif self._prev_joint_vel is not None:
                a = (
                    self._interpolation_vel_smoothing_alpha
                    if interpolation_tick
                    else self._vel_smoothing_alpha
                )
                target_joint_vel = a * raw_vel + (1.0 - a) * self._prev_joint_vel
            else:
                target_joint_vel = raw_vel
            # Per-joint velocity damping: scale down velocity when delta is small
            # to help joints settle near target without overshoot.
            pos_err = np.abs(target_joint_pos - current)
            damp_thresh = self._vel_damp_thresh
            damp_scale = np.clip(pos_err / damp_thresh, 0.0, 1.0)
            with np.errstate(over="ignore", invalid="ignore"):
                target_joint_vel = target_joint_vel * damp_scale * self._vel_ratio
            if not np.isfinite(target_joint_vel).all():
                raise ValueError(
                    "Refusing to send non-finite joint velocity to hardware"
                )
            self._prev_joint_vel = target_joint_vel.copy()
            if self._vel_log_file is not None:
                self._vel_log_file.write(
                    f"{_time.time()},{','.join(f'{v:.6f}' for v in target_joint_vel)}\n"
                )
            self.arm.set_joint_pos_vel(target_joint_pos, target_joint_vel, relative=False)
        elif blocking:
            wait_time = 1.0 / max(1, tick_hz)
            self.arm.set_joint_pos(target_joint_pos, wait_time=wait_time)
        else:
            self.arm._send_position_command(target_joint_pos)
        self._last_cmd_joint_pos = target_joint_pos.copy()
        self.sync_motion_manager_with_arm(target_joint_pos)

    def update_gripper(
        self,
        command: float,
        velocity: bool = True,
        blocking: bool = False,
        execution_hz: float | None = None,
    ) -> None:
        if self.hand is None:
            self._prev_gripper_command_successful = True
            return
        current = self.get_cached_gripper_position()
        if not np.isfinite(command) or not np.isfinite(current):
            raise ValueError("Scalar hand command/state must be finite")
        tick_hz = self.control_hz if execution_hz is None else execution_hz
        if velocity:
            target = current + float(command) * (1.0 / max(1, tick_hz))
        else:
            target = float(command)

        target = float(np.clip(target, 0.0, 1.0))
        wait_time = (1.0 / max(1, tick_hz)) if blocking else 0.0

        if blocking:
            self._clear_gripper_command_queue()
            self._execute_gripper_command(target, wait_time=wait_time, raise_on_error=True)
            self._refresh_gripper_state()
            return

        self._enqueue_gripper_target(target)

    def get_robot_state(self) -> tuple[dict[str, Any], dict[str, int]]:
        joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        joint_velocities = np.asarray(self.arm.get_joint_vel(), dtype=np.float64)
        try:
            joint_torques = np.asarray(self.arm.get_joint_torque(), dtype=np.float64)
        except ValueError:
            joint_torques = np.zeros(7, dtype=np.float64)
        # End-effector state under the historical key "gripper_position".
        # Scalar gripper → float in [0,1]. Multi-finger hand → the full joint
        # vector (radians), so recorded data preserves all DoF.
        if self.hand is None:
            gripper_position = 0.0
        elif self._hand_dof() > 1:
            gripper_position = np.asarray(
                self.hand.get_joint_pos(), dtype=np.float64
            ).tolist()
        else:
            gripper_position = self.get_cached_gripper_position()

        # Tactile: preferred driver accessor (WujiHandAdapter.get_hand_tactile),
        # with the legacy F5D6_V2 fingertip-force gate kept for the dexbot
        # hand. Gated on the multi-DoF opt-in — a "default"-gripper deployment
        # with an F5D6V2 attached must stay byte-identical to main (no new key,
        # and no per-step get_finger_tip_force warnings when the touch sensor
        # is unconfigured). Scalar grippers have neither method anyway.
        hand_tactile = None
        if self._hand_dof() > 1:
            if hasattr(self.hand, "get_hand_tactile"):
                _force = self.hand.get_hand_tactile()
                if _force is not None:
                    hand_tactile = np.asarray(_force, dtype=np.float64).tolist()
            elif hasattr(self.hand, "get_finger_tip_force"):
                _force = self.hand.get_finger_tip_force()
                if _force is not None:
                    hand_tactile = np.asarray(_force, dtype=np.float64).tolist()

        cartesian_position = self._get_cartesian_pose(joint_positions=joint_positions)

        timestamp_ns = int(self.arm.get_timestamp_ns())
        timestamp_dict = {
            "robot_timestamp_seconds": int(timestamp_ns // 1_000_000_000),
            "robot_timestamp_nanos": int(timestamp_ns % 1_000_000_000),
            "robot_timestamp_us": int(timestamp_ns // 1_000),
        }

        state_dict = {
            "joint_positions": joint_positions,
            "joint_velocities": joint_velocities,
            "joint_torques_computed": joint_torques,
            "prev_joint_torques_computed": joint_torques.copy(),
            "prev_joint_torques_computed_safened": joint_torques.copy(),
            "motor_torques_measured": joint_torques.copy(),
            "gripper_position": gripper_position,
            "cartesian_position": cartesian_position,
            "prev_controller_latency_ms": float(self._prev_controller_latency_ms),
            "prev_command_successful": bool(self._prev_command_successful),
            "prev_gripper_command_successful": bool(
                self._prev_gripper_command_successful
                and self.hand_command_path_healthy()
            ),
        }
        for field_name, component_name in (
            ("torso_joint_positions", "torso"),
            ("head_joint_positions", "head"),
        ):
            component = getattr(getattr(self, "robot", None), component_name, None)
            if component is not None:
                values = np.asarray(component.get_joint_pos(), dtype=np.float64)
                if values.shape == (3,) and np.isfinite(values).all():
                    state_dict[field_name] = values
        # Only present for hands with a tactile stream; scalar grippers never
        # add this key.
        if hand_tactile is not None:
            state_dict["hand_tactile"] = hand_tactile
        return state_dict, timestamp_dict

    def validate_joint_limits(self, target_joint_pos: np.ndarray) -> None:
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape != (7,) or not np.isfinite(target_joint_pos).all():
            raise ValueError("Joint target must contain seven finite values")
        limits = self.arm.joint_pos_limit
        if limits is None:
            return
        low = limits[:, 0].astype(np.float64)
        high = limits[:, 1].astype(np.float64)
        violates = np.logical_or(target_joint_pos < low, target_joint_pos > high)
        if np.any(violates):
            raise JointLimitExceededError(
                f"Joint limit exceeded for indices {np.where(violates)[0].tolist()}"
            )

    def move_to_joints_planned(
        self,
        target_arm_joints: np.ndarray,
        control_frequency: float = 100.0,
    ) -> None:
        """Plan and execute a smooth trajectory from current to target joints."""
        from dexmotion.utils import robot_utils

        mm = self.ik_controller.motion_manager
        actual = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        self.sync_motion_manager_with_arm(actual)

        start_dict = dict(mm.get_joint_pos_dict())
        goal_dict = dict(start_dict)
        for i, name in enumerate(self._arm_joint_names):
            if name in goal_dict:
                goal_dict[name] = float(target_arm_joints[i])

        waypoints = mm.plan_to_configuration(start_dict, goal_dict)
        if waypoints is None or len(waypoints) == 0:
            self.arm.set_joint_pos(target_arm_joints, wait_time=0.0)
            return

        wp_rows = []
        for wp in waypoints:
            if isinstance(wp, dict):
                wp_rows.append(robot_utils.get_qpos_from_joint_dict(mm.pin_robot, wp))
            else:
                wp_rows.append(np.asarray(wp, dtype=np.float64))
        wp_array = np.vstack(wp_rows)

        traj = mm.smooth_trajectory(
            wp_array, method="auto", control_frequency=control_frequency,
        )
        positions = traj.get("positions", wp_array)
        dt = 1.0 / control_frequency

        joint_names_full = list(mm.get_joint_pos_dict().keys())
        arm_indices = [joint_names_full.index(n) for n in self._arm_joint_names if n in joint_names_full]

        for pos in positions:
            arm_pos = pos[arm_indices] if arm_indices else pos[:7]
            self.arm.set_joint_pos(np.asarray(arm_pos, dtype=np.float64), wait_time=0.0)
            _time.sleep(dt)

        mm.set_joint_pos(goal_dict)

    def sync_motion_manager_with_arm(self, arm_joint_pos: np.ndarray) -> None:
        try:
            motion_manager = self.ik_controller.motion_manager
            qpos_dict = motion_manager.get_joint_pos_dict()
            for i, joint_name in enumerate(self._arm_joint_names):
                if joint_name in qpos_dict:
                    qpos_dict[joint_name] = float(arm_joint_pos[i])
            # Clip to joint limits so motion manager accepts the configuration
            # even when the real arm is marginally outside limits (IK numerical error)
            if robot_utils is not None:
                qpos_dict = robot_utils.clip_joint_positions_to_limits(
                    motion_manager.pin_robot, qpos_dict
                )
            motion_manager.set_joint_pos(qpos_dict)
        except Exception:
            pass

    def _solve_cartesian_delta(self, delta_xyz: np.ndarray, delta_rpy: np.ndarray) -> np.ndarray:
        # Gradual correction: use last command + outlier-rejected hw correction as IK start
        hw_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        if self._last_cmd_joint_pos is not None:
            raw_error = hw_pos - self._last_cmd_joint_pos
            outlier_mask = np.abs(raw_error) > self._HW_CORRECTION_OUTLIER_THRESH
            raw_error[outlier_mask] = 0.0
            actual_joints = self._last_cmd_joint_pos + raw_error * self._HW_CORRECTION_ALPHA
        else:
            actual_joints = hw_pos
        self.sync_motion_manager_with_arm(actual_joints)
        try:
            target_joint_pos = self.ik_controller.move_delta_cartesian(
                delta_xyz=np.asarray(delta_xyz, dtype=np.float64),
                delta_rpy=np.asarray(delta_rpy, dtype=np.float64),
                side=self.arm_side,
            )
        except Exception as exc:
            raise IKFailedError(f"IK solver failed: {exc}") from exc

        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
        if target_joint_pos.shape[0] != 7:
            raise IKFailedError(f"IK returned invalid joint vector shape: {target_joint_pos.shape}")
        return target_joint_pos

    def _init_gripper_reference(self) -> tuple[np.ndarray, np.ndarray]:
        if self.hand is None:
            return np.array([0.0], dtype=np.float64), np.array([1.0], dtype=np.float64)
        try:
            open_pos = np.asarray(self.hand.get_predefined_pose("open"), dtype=np.float64)
        except Exception:
            open_pos = np.asarray(self.hand.get_joint_pos(), dtype=np.float64)

        try:
            close_pos = np.asarray(self.hand.get_predefined_pose("close"), dtype=np.float64)
        except Exception:
            close_pos = open_pos.copy()

        if open_pos.shape != close_pos.shape:
            close_pos = np.resize(close_pos, open_pos.shape)
        return open_pos, close_pos

    def _normalize_gripper_position(self, hand_joint_pos: np.ndarray) -> float:
        open_pos = np.asarray(self._gripper_open_pos, dtype=np.float64)
        close_pos = np.asarray(self._gripper_close_pos, dtype=np.float64)
        hand_joint_pos = np.asarray(hand_joint_pos, dtype=np.float64)
        if hand_joint_pos.shape != open_pos.shape:
            hand_joint_pos = np.resize(hand_joint_pos, open_pos.shape)

        denom = close_pos - open_pos
        valid = np.abs(denom) > 1e-6
        if not np.any(valid):
            return float(np.clip(np.mean(hand_joint_pos), 0.0, 1.0))

        ratio = np.zeros_like(hand_joint_pos, dtype=np.float64)
        ratio[valid] = (hand_joint_pos[valid] - open_pos[valid]) / denom[valid]
        return float(np.clip(np.mean(ratio[valid]), 0.0, 1.0))

    def _forward_kinematics(self, joint_positions: np.ndarray) -> np.ndarray:
        motion_manager = self.ik_controller.motion_manager
        qpos_dict = motion_manager.get_joint_pos_dict()
        for i, joint_name in enumerate(self._arm_joint_names):
            if joint_name in qpos_dict:
                qpos_dict[joint_name] = float(joint_positions[i])

        if robot_utils is not None:
            qpos = robot_utils.get_qpos_from_joint_dict(motion_manager.pin_robot, qpos_dict)
        else:
            self.sync_motion_manager_with_arm(np.asarray(joint_positions, dtype=np.float64))
            qpos = motion_manager.get_joint_pos()

        ee_pose_dict = motion_manager.fk(
            frame_names=motion_manager.target_frames,
            qpos=qpos,
            update_robot_state=False,
        )
        key = "L_ee" if self.arm_side == "left" else "R_ee"
        if key not in ee_pose_dict:
            key = sorted(ee_pose_dict.keys())[0]

        ee_pose = ee_pose_dict[key]
        pose_mat = ee_pose.np if hasattr(ee_pose, "np") else np.asarray(ee_pose)
        xyz = pose_mat[:3, 3]
        rpy = R.from_matrix(pose_mat[:3, :3]).as_euler("xyz", degrees=False)
        return np.concatenate([xyz, rpy]).astype(np.float64)

    def _get_cartesian_pose(self, joint_positions: Optional[np.ndarray] = None) -> np.ndarray:
        if joint_positions is None:
            joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        return self._forward_kinematics(joint_positions)

    @staticmethod
    def _velocity_to_motor_delta(
        velocity: np.ndarray,
        max_lin_delta: float,
        max_rot_delta: float,
    ) -> np.ndarray:
        """Replicate server ``_cartesian_velocity_to_delta`` logic.

        Normalizes linear/rotation components independently (if norm > 1)
        then scales by the per-step delta limits.
        """
        lin_vel = velocity[:3].copy()
        rot_vel = velocity[3:6].copy()
        lin_norm = float(np.linalg.norm(lin_vel))
        rot_norm = float(np.linalg.norm(rot_vel))
        if lin_norm > 1.0:
            lin_vel /= lin_norm
        if rot_norm > 1.0:
            rot_vel /= rot_norm
        return np.concatenate([lin_vel * max_lin_delta, rot_vel * max_rot_delta])

    def create_action_dict(
        self,
        action: np.ndarray,
        action_space: str,
        gripper_action_space: Optional[str] = None,
        robot_state: Optional[dict] = None,
        max_lin_delta: Optional[float] = None,
        max_rot_delta: Optional[float] = None,
    ) -> dict:
        """Convert action to comprehensive action dict with all representations.

        Used by policy_runner for action logging and data collection.

        Parameters
        ----------
        max_lin_delta, max_rot_delta:
            Per-step Cartesian delta limits used by the server's
            ``_cartesian_velocity_to_delta``.  When provided the dict's
            ``delta_action`` will match the motor command exactly.
        """
        if action_space not in SUPPORTED_ACTION_SPACES:
            raise ValueError(f"Unsupported action_space: {action_space}")

        if robot_state is None:
            robot_state, _ = self.get_robot_state()

        action = np.asarray(action, dtype=np.float64)
        dt = 1.0 / max(1, self.control_hz)

        action_dict = {"robot_state": robot_state}

        # Determine gripper action space
        velocity = "velocity" in action_space
        if gripper_action_space is None:
            gripper_action_space = "velocity" if velocity else "position"

        # Extract end-effector action
        arm_dof = 7 if action_space.startswith("joint") else 6
        _, hand_action = self._split_arm_hand_action(action, arm_dof, action_space)
        is_multi_hand = hand_action.size > 1

        # End-effector conversions. Names (gripper_position / gripper_delta) are
        # kept so state/action_info/get_info all agree; values become N-vectors
        # for the multi-finger hand. Element-wise so the scalar float() path
        # doesn't blow up on an N-vector.
        current_gripper = np.asarray(robot_state["gripper_position"], dtype=np.float64).reshape(-1)
        if is_multi_hand:
            # Absolute joint-angle command (no [0,1] normalization for the hand).
            gripper_position = np.asarray(hand_action, dtype=np.float64).reshape(-1)
            if current_gripper.shape != gripper_position.shape:
                current_gripper = np.resize(current_gripper, gripper_position.shape)
            action_dict["gripper_position"] = gripper_position.tolist()
            action_dict["gripper_delta"] = (gripper_position - current_gripper).tolist()
            gripper_action = gripper_position  # used below for concat
        else:
            gripper_action_s = float(hand_action.reshape(-1)[0]) if hand_action.size else 0.0
            cur = float(current_gripper[0]) if current_gripper.size else 0.0
            if gripper_action_space == "velocity":
                gripper_pos_s = float(np.clip(cur + gripper_action_s * dt, 0.0, 1.0))
            else:
                gripper_pos_s = float(np.clip(gripper_action_s, 0.0, 1.0))
            action_dict["gripper_position"] = gripper_pos_s
            action_dict["gripper_delta"] = float(gripper_pos_s - cur)
            gripper_position = gripper_pos_s
            gripper_action = gripper_action_s

        # Process action based on space
        current_joint_pos = np.asarray(robot_state["joint_positions"], dtype=np.float64)
        current_cart_pos = np.asarray(robot_state["cartesian_position"], dtype=np.float64)

        if action_space.startswith("cartesian") or action_space == "target_cartesian_delta":
            cart_action = action[:6]

            if action_space == "target_cartesian_delta":
                # target_cartesian_delta → recover velocity by multiplying gains
                cart_velocity = np.empty(6, dtype=np.float64)
                cart_velocity[:3] = cart_action[:3] * _TELEOP_POS_ACTION_GAIN
                cart_velocity[3:6] = cart_action[3:6] * _TELEOP_ROT_ACTION_GAIN
                action_dict["cartesian_velocity"] = cart_velocity.tolist()
                action_dict["target_cartesian_delta"] = np.concatenate(
                    [cart_action, np.atleast_1d(gripper_position)]
                ).tolist()
                # Compute motor delta using the same logic as the server
                if max_lin_delta is not None and max_rot_delta is not None:
                    cartesian_delta = self._velocity_to_motor_delta(
                        cart_velocity, max_lin_delta, max_rot_delta,
                    )
                else:
                    cartesian_delta = cart_velocity * dt
            elif velocity:
                # cartesian_velocity
                action_dict["cartesian_velocity"] = cart_action.tolist()
                # Use the same motor-delta conversion as the server when the
                # caller provides max_*_delta limits, so the logged
                # cartesian_delta matches what the robot actually executes.
                # Fall back to vel*dt for callers that do not pass limits.
                if max_lin_delta is not None and max_rot_delta is not None:
                    cartesian_delta = self._velocity_to_motor_delta(
                        cart_action, max_lin_delta, max_rot_delta,
                    )
                else:
                    cartesian_delta = cart_action * dt
            else:
                # cartesian_delta
                cartesian_delta = cart_action
                cartesian_velocity = cart_action / dt
                action_dict["cartesian_velocity"] = cartesian_velocity.tolist()

            action_dict["delta_action"] = np.concatenate(
                [cartesian_delta, np.atleast_1d(gripper_action)]
            ).tolist()

            # target_cartesian_delta (add for all cartesian input spaces)
            if "target_cartesian_delta" not in action_dict:
                vel_for_tcd = (
                    cart_action if velocity
                    else action_dict["cartesian_velocity"]
                )
                gp = np.atleast_1d(gripper_position)
                tcd = np.empty(6 + gp.size, dtype=np.float64)
                tcd[:3] = np.asarray(vel_for_tcd[:3]) / _TELEOP_POS_ACTION_GAIN
                tcd[3:6] = np.asarray(vel_for_tcd[3:6]) / _TELEOP_ROT_ACTION_GAIN
                tcd[6:] = gp
                action_dict["target_cartesian_delta"] = tcd.tolist()

            # Compute target cartesian position (using simple pose addition)
            target_cart = current_cart_pos.copy()
            target_cart[:3] += cartesian_delta[:3]  # xyz
            target_cart[3:6] += cartesian_delta[3:6]  # rpy
            action_dict["cartesian_position"] = target_cart.tolist()

            # Use IK to get joint positions
            try:
                target_joints = self._solve_cartesian_delta(cartesian_delta[:3], cartesian_delta[3:6])
                action_dict["joint_position"] = target_joints.tolist()
                joint_delta = target_joints - current_joint_pos
                action_dict["joint_velocity"] = (joint_delta / dt).tolist()
            except Exception:
                # IK failed, use current positions
                action_dict["joint_position"] = current_joint_pos.tolist()
                action_dict["joint_velocity"] = [0.0] * 7

        elif action_space.startswith("joint"):
            joint_action = action[:7]

            if action_space == "joint_position":
                target_joints = joint_action
                joint_delta = target_joints - current_joint_pos
                joint_velocity = joint_delta / dt
            elif action_space == "joint_velocity":
                joint_velocity = joint_action
                joint_delta = joint_velocity * dt
                target_joints = current_joint_pos + joint_delta
            else:  # joint_delta
                joint_delta = joint_action
                target_joints = current_joint_pos + joint_delta
                joint_velocity = joint_delta / dt

            action_dict["joint_position"] = target_joints.tolist()
            action_dict["joint_velocity"] = joint_velocity.tolist()
            action_dict["joint_delta"] = joint_delta.tolist()

            # Compute cartesian via FK
            try:
                target_cart = self._get_cartesian_pose(target_joints)
                action_dict["cartesian_position"] = target_cart.tolist()
            except Exception:
                action_dict["cartesian_position"] = current_cart_pos.tolist()

        return action_dict

    def stop(self) -> None:
        """Stop robot motion (emergency stop compatibility)."""
        try:
            # Stop arm motion by sending current position as target
            current_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
            self.arm.set_joint_pos(current_pos, wait_time=0.0)
        except Exception:
            pass
        try:
            # Stop chassis wheels
            self.robot.chassis.stop()
        except Exception:
            pass

    def close(self) -> None:
        if self._vel_log_file is not None:
            self._vel_log_file.close()
            self._vel_log_file = None
        self._stop_gripper_worker()
        errors = []
        if (
            not getattr(self, "_hand_shutdown_completed", False)
            and self.gripper_type
            in ("robotiq", "sr_gripper", "wuji_hand_v1", "wuji_hand_v2")
            and self.hand is not None
        ):
            try:
                self.hand.shutdown()
                self._hand_shutdown_completed = True
            except Exception as exc:
                errors.append(f"hand shutdown failed: {exc}")
        # The hand adapters share lower-level process/device resources with the
        # robot runtime.  Never close the base transport until required hand
        # teardown has succeeded; otherwise a retry cannot finish the hand.
        hand_shutdown_required = (
            self.gripper_type
            in ("robotiq", "sr_gripper", "wuji_hand_v1", "wuji_hand_v2")
            and self.hand is not None
        )
        if (
            (not hand_shutdown_required or self._hand_shutdown_completed)
            and not getattr(self, "_robot_shutdown_completed", False)
        ):
            try:
                self.robot.shutdown()
                self._robot_shutdown_completed = True
            except Exception as exc:
                errors.append(f"robot shutdown failed: {exc}")
        if errors:
            raise RuntimeError("; ".join(errors))
