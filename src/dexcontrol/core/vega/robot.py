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


class VegaRobot:
    """Thin wrapper around dexcontrol Robot for single-arm control."""

    def __init__(
        self,
        robot_model: str = "vega_1",
        arm_side: str = "left",
        control_hz: int = 20,
        gripper_type: str = "default",
        ik_solver_type: str = "pink",
        use_velocity_feedforward: bool = False,
        robotiq_comport: str = "/dev/ttyUSB0",
        ema_alpha: float = 0.0,
        interpolation_method: str = "none",
        interpolation_history: int = 4,
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
        head_init_pos: tuple[float, ...] | list[float] = (2.0, 0.0, -0.3),
        **kwargs,
    ):
        hand_type = kwargs.pop("hand_type", None)
        if kwargs:
            unexpected = ", ".join(sorted(kwargs.keys()))
            raise TypeError(f"Unexpected keyword arguments: {unexpected}")
        if hand_type is not None and gripper_type == "default":
            gripper_type = hand_type
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
        self.gripper_type = gripper_type
        self.use_velocity_feedforward = bool(use_velocity_feedforward)

        configs = get_robot_config(robot_model)
        self.robot = _RobotWithCustomHeadPose(
            configs=configs,
            head_init_pos=np.asarray(head_init_pos, dtype=np.float32) if head_init_pos is not None else None,
        )
        self.arm = getattr(self.robot, f"{arm_side}_arm")
        hand_component = f"{arm_side}_hand"

        if gripper_type == "robotiq":
            from dexcontrol.core.robotiq_gripper import RobotiqGripper  # lazy import
            self.hand = RobotiqGripper(comport=self._robotiq_comport)
        elif gripper_type == "sr_gripper":
            from dexcontrol.core.sr_gripper import SrGripperAdapter  # lazy import
            self.hand = SrGripperAdapter(comport=self._robotiq_comport)
        else:
            self.hand = getattr(self.robot, hand_component) if self.robot.has_component(hand_component) else None

        custom_ik_cfg = self._build_ik_config()
        self.ik_controller = BaseIKController(
            bot=self.robot, visualize=False,
            ik_solver_type=self._ik_solver_type,
            custom_local_ik_config=custom_ik_cfg,
        )
        self._arm_joint_names = [f"{arm_side[0].upper()}_arm_j{i + 1}" for i in range(7)]

        # Use Vega unfold pose (L_shape) as default reset, same as fold_robot.py unfold.
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
        self._prev_command_successful = True
        self._prev_gripper_command_successful = True
        self._prev_controller_latency_ms = 0.0
        self._prev_joint_vel: np.ndarray | None = None
        self._vel_smoothing_alpha = float(np.clip(vel_smoothing_alpha, 0.0, 1.0))
        self._last_cmd_joint_pos: np.ndarray | None = None  # Track last sent command for delta clipping
        self._prev_cmd_delta: np.ndarray | None = None  # Previous step delta for jerk limiting
        self._HW_CORRECTION_ALPHA = float(np.clip(hw_correction_alpha, 0.0, 1.0))
        self._HW_CORRECTION_OUTLIER_THRESH = 0.5  # If |hw - cmd| > this, skip correction for that joint entirely
        self._max_delta_scale = float(max(0.1, max_delta_scale))
        self._MOTOR_MAX_JERK_RAD = float(max(0.0, max_jerk))
        self._vel_ratio = float(max(0.0, vel_ratio))
        self._vel_damp_thresh = float(max(0.001, vel_damp_thresh))

        # Velocity logging for diagnostics.
        vel_log_path = os.environ.get("VEL_LOG_PATH")
        if vel_log_path:
            self._vel_log_file = open(vel_log_path, "a")
            header = f"timestamp,{','.join(f'j{i+1}_vel' for i in range(7))}\n"
            self._vel_log_file.write(header)
            _logger.info("Velocity logging enabled: %s", vel_log_path)
        else:
            self._vel_log_file = None

        # Critically damped 2nd-order smoothing filter for joint commands.
        # alpha controls responsiveness (0.0 = disabled, 0.3~0.8 = typical).
        self._ema_alpha = float(np.clip(ema_alpha, 0.0, 0.99))
        self._filter_pos: np.ndarray | None = None
        self._filter_vel: np.ndarray | None = None

        # --- Trajectory interpolation (input-rate → control-rate upsampling) ---
        # When interpolation_method != "none", incoming commands are buffered
        # via add_command_point() and the control loop calls
        # get_interpolated_command() at a higher rate for smooth output.
        self._interpolation_method = interpolation_method
        if interpolation_method != "none":
            self._interpolator = TrajectoryInterpolator(
                method=interpolation_method,
                history_size=interpolation_history,
            )
            _logger.info(
                "Trajectory interpolation enabled: method=%s history=%d",
                interpolation_method,
                interpolation_history,
            )
        else:
            self._interpolator = None

        # --- Output filter (applied after interpolation / EMA) ---
        if filter_type != "none":
            filter_cfg = {"default": {"type": filter_type}}
            if filter_type == "butterworth":
                filter_cfg["default"]["cutoff_freq"] = filter_cutoff_freq
                filter_cfg["default"]["order"] = filter_order
            elif filter_type == "ema":
                filter_cfg["default"]["alpha"] = filter_ema_alpha
            self._output_filter = MultiChannelFilter(
                filter_config=filter_cfg,
                control_rate=float(control_hz),
            )
            _logger.info(
                "Output filter enabled: type=%s (cutoff=%.1f order=%d ema_alpha=%.2f)",
                filter_type,
                filter_cutoff_freq,
                filter_order,
                filter_ema_alpha,
            )
        else:
            self._output_filter = None

        # Latest raw command stored by add_command_point for the control loop
        self._latest_target_joint_pos: np.ndarray | None = None
        self._latest_gripper_action: float = 0.0
        self._latest_hand_action: np.ndarray | None = None
        self._latest_gripper_action_space: str = "position"
        self._interp_lock = threading.Lock()

        if self.hand is not None:
            self._refresh_gripper_state()
            self._start_gripper_worker()

        # #region agent log
        self._agent_debug_log(
            run_id=f"joint-diagnostics-{self.arm_side}",
            hypothesis_id="H10",
            location="dexcontrol/core/vega/robot.py:__init__",
            message="agent_debug_probe",
            data={
                "pid": int(os.getpid()),
                "module_file": str(Path(__file__).resolve()),
                "log_path": self._AGENT_DEBUG_LOG_PATH,
                "arm_side": self.arm_side,
                "control_hz": self.control_hz,
                "ik_solver_type": self._ik_solver_type,
            },
        )
        # #endregion

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
        if self._interpolator is not None:
            self._interpolator.clear()
        if self._output_filter is not None:
            self._output_filter.reset()
        self._latest_target_joint_pos = None

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
        if self._gripper_worker is not None and self._gripper_worker.is_alive():
            self._gripper_worker.join(timeout=1.0)
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
                self._execute_gripper_command(
                    latest_target, wait_time=0.0, raise_on_error=False
                )

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
            self._prev_gripper_command_successful = True
            return True
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
            self._prev_gripper_command_successful = True
            return True
        except Exception:
            self._prev_gripper_command_successful = False
            if raise_on_error:
                raise
            return False

    def _enqueue_gripper_target(self, target: float) -> None:
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

    def _hand_dof(self) -> int:
        """End-effector DoF: 6 for the F5D6 multi-finger hand, else 1 (scalar gripper).

        Prefer the component's joint_name list (cached from RobotInfo at init):
        it is the reliable DoF for dexbot Hand components (6 for F5D6) and does
        NOT depend on live hardware state — unlike get_joint_pos(), which raises
        ServiceUnavailableError until the first Zenoh state arrives and would
        make this fall back to 1 mid-run, rejecting the 6-DoF hand action.
        Scalar grippers (Robotiq/SR) have no joint_name, so fall back to
        get_joint_pos().size for them.
        """
        if self.hand is None:
            if not getattr(self, "_hand_dof_logged", False):
                _logger.info("[HandDoF] self.hand is None -> dof=1 (gripper_type=%s)", self.gripper_type)
                self._hand_dof_logged = True
            return 1
        # --- diagnostics: log what we actually see, once ---
        _names = None
        _gjp_repr = None
        try:
            _names = getattr(self.hand, "joint_name", None)
        except Exception as e:
            _names = f"<joint_name raised {e!r}>"
        try:
            _g = np.asarray(self.hand.get_joint_pos())
            _gjp_repr = f"shape={_g.shape} size={_g.size}"
        except Exception as e:
            _gjp_repr = f"<get_joint_pos raised {e!r}>"
        if not getattr(self, "_hand_dof_logged", False):
            _logger.info(
                "[HandDoF] hand_type=%s gripper_type=%s joint_name=%s get_joint_pos(%s)",
                type(self.hand).__name__, self.gripper_type, _names, _gjp_repr,
            )
            self._hand_dof_logged = True
        # --- end diagnostics ---
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
        _hn = getattr(self.hand, "joint_name", None) if self.hand is not None else None
        _logger.info(
            "[Split] space=%s len=%d arm_dof=%d hand_dof=%d expected=%d hand=%s joint_name_len=%s",
            action_space, n, arm_dof, hand_dof, expected,
            type(self.hand).__name__ if self.hand is not None else None,
            (len(_hn) if _hn else _hn),
        )
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
    ) -> None:
        """Send the end-effector command.

        Scalar (length 1) → existing normalized gripper pipeline.
        Multi-DoF (length>1) → direct joint-position command on the hand,
        bypassing the [0,1] scalar collapse.
        """
        if hand_action.size <= 1:
            self.update_gripper(
                float(hand_action.reshape(-1)[0]) if hand_action.size else 0.0,
                velocity=(gripper_action_space == "velocity"),
                blocking=blocking,
            )
            return
        if self.hand is None:
            return
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0
        with self._gripper_io_lock:
            self.hand.set_joint_pos(np.asarray(hand_action, dtype=np.float64), wait_time=wait_time)

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
        dt = 1.0 / max(1, self.control_hz)
        state_dict, _ = self.get_robot_state()
        current_joint_pos = np.asarray(state_dict["joint_positions"], dtype=np.float64)

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
            self._apply_hand_action(hand_action, gripper_action_space, blocking=blocking)
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
        elif action_space == "joint_velocity":
            target_joint_pos = current_joint_pos + arm_action * dt
        elif action_space == "joint_delta":
            target_joint_pos = current_joint_pos + arm_action
        elif action_space == "cartesian_velocity":
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3] * dt, arm_action[3:6] * dt)
        else:  # cartesian_delta
            target_joint_pos = self._solve_cartesian_delta(arm_action[:3], arm_action[3:6])

        timestamp = _time.perf_counter()
        with self._interp_lock:
            self._interpolator.add_point(timestamp, target_joint_pos)
            self._latest_target_joint_pos = target_joint_pos.copy()
            # Store the full end-effector action (1 scalar gripper, or N-DoF hand).
            self._latest_hand_action = np.asarray(hand_action, dtype=np.float64).copy()
            self._latest_gripper_action = float(hand_action.reshape(-1)[0]) if hand_action.size else 0.0
            self._latest_gripper_action_space = gripper_action_space

    def execute_interpolated_tick(self) -> bool:
        """Run one control-loop tick using interpolated position.

        Called at control rate (e.g. 100 Hz) by the background control
        thread.  Returns ``True`` if a command was sent, ``False`` if
        there is no data yet.
        """
        with self._interp_lock:
            if self._latest_target_joint_pos is None:
                return False

            now = _time.perf_counter()
            pos, vel = self._interpolator.interpolate(now, compute_velocity=self.use_velocity_feedforward)

            if pos is None:
                # Not enough data for interpolation — use latest raw target.
                pos = self._latest_target_joint_pos.copy()
                vel = None

            hand_action = getattr(self, "_latest_hand_action", None)
            if hand_action is None:
                hand_action = np.array([self._latest_gripper_action], dtype=np.float64)
            else:
                hand_action = hand_action.copy()
            gripper_vel = self._latest_gripper_action_space == "velocity"

        # Apply the output filter (Butterworth / EMA) after interpolation.
        if self._output_filter is not None:
            pos = self._output_filter.apply(pos)

        # Apply existing EMA 2nd-order smoothing on top if enabled.
        if self._ema_alpha > 0.0:
            dt = 1.0 / max(1, self.control_hz)
            if self._filter_pos is None:
                self._filter_pos = pos.copy()
                self._filter_vel = np.zeros_like(pos)
            else:
                alpha = self._ema_alpha
                omega = alpha / (dt * (1.0 - alpha))
                exp_term = np.exp(-omega * dt)
                err = self._filter_pos - pos
                c = self._filter_vel + omega * err
                self._filter_pos = pos + (err + c * dt) * exp_term
                self._filter_vel = (self._filter_vel - c * omega * dt) * exp_term
                pos = self._filter_pos.copy()

        try:
            self.update_joints(pos, velocity=False, blocking=False)
            self._apply_hand_action(
                hand_action,
                "velocity" if gripper_vel else "position",
                blocking=False,
            )
            self._prev_command_successful = True
        except (JointLimitExceededError, IKFailedError):
            self._prev_command_successful = False
        except Exception as exc:
            self._prev_command_successful = False
            _logger.error("Interpolated tick failed: %s", exc)

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
    ) -> None:
        run_id = f"joint-diagnostics-{self.arm_side}"
        target_joint_pos = np.asarray(joint_pos_command, dtype=np.float64)
        if velocity:
            dt = 1.0 / max(1, self.control_hz)
            current_joint_pos = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
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
        if self._last_cmd_joint_pos is not None:
            raw_error = hw_pos - self._last_cmd_joint_pos
            # Per-joint outlier rejection: skip correction for joints where
            # hw feedback jumps beyond threshold (e.g. glitchy readings).
            outlier_mask = np.abs(raw_error) > self._HW_CORRECTION_OUTLIER_THRESH
            raw_error[outlier_mask] = 0.0
            current = self._last_cmd_joint_pos + raw_error * self._HW_CORRECTION_ALPHA
        else:
            current = hw_pos
        diff = target_joint_pos - current
        raw_diff = diff.copy()
        clipped_mask = np.abs(diff) > self._MOTOR_MAX_DELTA_RAD
        clipped_count = int(np.count_nonzero(clipped_mask))
        if np.any(clipped_mask):
            clipped = np.clip(diff, -self._MOTOR_MAX_DELTA_RAD, self._MOTOR_MAX_DELTA_RAD)
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
                    "clip_limit": self._MOTOR_MAX_DELTA_RAD[clipped_mask].tolist(),
                },
            )
            # #endregion
            _logger.info(
                "[DeltaClip] joints=%s raw=%s limit=%s",
                clipped_indices.tolist(),
                np.round(diff[clipped_mask], 4).tolist(),
                self._MOTOR_MAX_DELTA_RAD[clipped_mask].tolist(),
            )
            target_joint_pos = current + clipped
            diff = clipped

        # Jerk limit: restrict how fast the per-step delta can change between
        # consecutive steps.  Smooths acceleration/deceleration to prevent
        # the abrupt velocity changes that cause oscillation on stop.
        jerk_count = 0
        if self._prev_cmd_delta is not None and self._MOTOR_MAX_JERK_RAD > 0:
            accel = diff - self._prev_cmd_delta
            jerk_limit = self._MOTOR_MAX_JERK_RAD
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

        if self.use_velocity_feedforward and not blocking:
            dt = 1.0 / max(1, self.control_hz)
            raw_vel = (target_joint_pos - current) / dt
            if self._prev_joint_vel is not None:
                a = self._vel_smoothing_alpha
                target_joint_vel = a * raw_vel + (1.0 - a) * self._prev_joint_vel
            else:
                target_joint_vel = raw_vel
            # Per-joint velocity damping: scale down velocity when delta is small
            # to help joints settle near target without overshoot.
            pos_err = np.abs(target_joint_pos - current)
            damp_thresh = self._vel_damp_thresh
            damp_scale = np.clip(pos_err / damp_thresh, 0.0, 1.0)
            target_joint_vel = target_joint_vel * damp_scale * self._vel_ratio
            self._prev_joint_vel = target_joint_vel.copy()
            if self._vel_log_file is not None:
                self._vel_log_file.write(
                    f"{_time.time()},{','.join(f'{v:.6f}' for v in target_joint_vel)}\n"
                )
            self.arm.set_joint_pos_vel(target_joint_pos, target_joint_vel, relative=False)
        elif blocking:
            wait_time = 1.0 / max(1, self.control_hz)
            self.arm.set_joint_pos(target_joint_pos, wait_time=wait_time)
        else:
            self.arm._send_position_command(target_joint_pos)
        self._last_cmd_joint_pos = target_joint_pos.copy()
        self.sync_motion_manager_with_arm(target_joint_pos)

    def update_gripper(self, command: float, velocity: bool = True, blocking: bool = False) -> None:
        if self.hand is None:
            self._prev_gripper_command_successful = True
            return
        current = self.get_cached_gripper_position()
        if velocity:
            target = current + float(command) * (1.0 / max(1, self.control_hz))
        else:
            target = float(command)

        target = float(np.clip(target, 0.0, 1.0))
        wait_time = (1.0 / max(1, self.control_hz)) if blocking else 0.0

        if blocking:
            self._clear_gripper_command_queue()
            self._execute_gripper_command(target, wait_time=wait_time, raise_on_error=True)
            self._refresh_gripper_state()
            return

        self._enqueue_gripper_target(target)
        self._prev_gripper_command_successful = True

    def get_robot_state(self) -> tuple[dict[str, Any], dict[str, int]]:
        joint_positions = np.asarray(self.arm.get_joint_pos(), dtype=np.float64)
        joint_velocities = np.asarray(self.arm.get_joint_vel(), dtype=np.float64)
        try:
            joint_torques = np.asarray(self.arm.get_joint_torque(), dtype=np.float64)
        except ValueError:
            joint_torques = np.zeros(7, dtype=np.float64)
        # End-effector state under the historical key "gripper_position".
        # Scalar gripper → float in [0,1]. Multi-finger F5D6 hand → the full
        # 6-DoF joint vector (radians), so recorded data preserves all DoF.
        if self.hand is None:
            gripper_position = 0.0
        elif self._hand_dof() > 1:
            gripper_position = np.asarray(
                self.hand.get_joint_pos(), dtype=np.float64
            ).tolist()
        else:
            gripper_position = self.get_cached_gripper_position()

        # Fingertip force (tactile), F5D6_V2 hand only. get_finger_tip_force()
        # exists solely on HandF5D6V2, so the hasattr gate keeps scalar grippers
        # (Robotiq/SR, which lack the method) completely untouched — no new key.
        hand_tactile = None
        if hasattr(self.hand, "get_finger_tip_force"):
            _force = self.hand.get_finger_tip_force()
            if _force is not None:
                hand_tactile = np.asarray(_force, dtype=np.float64).tolist()

        wrench_state = np.zeros(6, dtype=np.float64)
        if getattr(self.arm, "wrench_sensor", None) is not None:
            wrench_state = np.asarray(self.arm.wrench_sensor.get_wrench_state(), dtype=np.float64)

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
            "wrench_state": wrench_state,
            "prev_controller_latency_ms": float(self._prev_controller_latency_ms),
            "prev_command_successful": bool(self._prev_command_successful),
            "prev_gripper_command_successful": bool(self._prev_gripper_command_successful),
        }
        # Only present for the F5D6_V2 hand; scalar grippers never add this key.
        if hand_tactile is not None:
            state_dict["hand_tactile"] = hand_tactile
        return state_dict, timestamp_dict

    def validate_joint_limits(self, target_joint_pos: np.ndarray) -> None:
        limits = self.arm.joint_pos_limit
        if limits is None:
            return
        target_joint_pos = np.asarray(target_joint_pos, dtype=np.float64)
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

        # Extract gripper action
        arm_dof = 7 if action_space.startswith("joint") else 6
        _, hand_action = self._split_arm_hand_action(action, arm_dof, action_space)
        is_multi_hand = hand_action.size > 1

        # End-effector conversions. Names (gripper_position / gripper_delta) are
        # kept so state/action_info/get_info all agree; values become N-vectors
        # for the multi-finger hand. Element-wise so the scalar float() path
        # doesn't blow up on a 6-vector.
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
        if self.gripper_type in ("robotiq", "sr_gripper") and self.hand is not None:
            try:
                self.hand.shutdown()
            except Exception:
                pass
        try:
            self.robot.shutdown()
        except Exception:
            pass
