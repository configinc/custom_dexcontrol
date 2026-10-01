"""Torso state + torso reset in the robotenv_vega server (fake robot, no hardware).

    python tests/test_robotenv_vega_torso.py
"""
import sys
import threading
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.modules.setdefault("dualsense_controller", mock.MagicMock())  # needs hidapi at import

import dexcontrol.core.robotenv_vega.server as srv_mod  # noqa: E402
from proto import robotenv_pb2  # noqa: E402

LIMITS = np.array([[0.0, 1.570], [0.0, 3.141], [-1.570, 1.570]])
V1 = [0.78, 1.57, 0.0]


class FakeTorso:
    def __init__(self, pos, reach=True):
        self.pos = np.asarray(pos, dtype=np.float64)
        self.reach = reach
        self.joint_pos_limit = LIMITS
        self.calls = []

    def get_joint_pos(self):
        return self.pos.copy()

    def get_joint_pos_dict(self):
        return {f"torso_j{i + 1}": v for i, v in enumerate(self.pos)}

    @property
    def pitch_angle(self):
        return self.pos[0] + self.pos[2] + np.pi / 2 - self.pos[1]

    def set_idle_mode(self, enabled):
        self.calls.append(("idle", enabled))

    def set_joint_target(self, pos, scale=None, tracked=False):
        self.calls.append(("target", np.asarray(pos).tolist(), scale))
        if self.reach:
            self.pos = np.asarray(pos, dtype=np.float64)
        return mock.MagicMock()

    def is_joint_pos_reached(self, pos, tolerance):
        return bool(np.max(np.abs(self.pos - pos)) <= tolerance)


def make_server(torso):
    srv = object.__new__(srv_mod.VegaRobotEnvService)
    srv.robot_model, srv.control_hz, srv.arm_side = "vega_1", 90, "right"
    srv.gripper_type, srv.frame_type = "robotiq", "vega-1-pro_torso_frame"
    srv._cancel_move, srv._cmd_lock = threading.Event(), threading.Lock()
    srv._control_loop_thread = None
    srv.order = []
    srv._execute_reset_sequence = lambda target: srv.order.append("arm")
    srv._create_observation = lambda: ({}, 0)
    robot = mock.MagicMock()
    robot.arm.joint_pos_limit = None
    robot.arm.get_joint_pos.return_value = np.zeros(7)
    robot.robot.torso = torso
    robot.sync_motion_manager_with_arm.side_effect = lambda q: srv.order.append("sync")
    srv._robot = robot
    return srv


def reset_request(torso=None):
    params = {"joint_positions": robotenv_pb2.Value(float_array=robotenv_pb2.FloatArray(values=[0.0] * 7))}
    if torso is not None:
        params["torso_joint_positions"] = robotenv_pb2.Value(float_array=robotenv_pb2.FloatArray(values=torso))
    return robotenv_pb2.ResetRequest(mode="target", params=params)


class TorsoResetTest(unittest.TestCase):
    def test_extract(self):
        srv = make_server(FakeTorso(V1))
        self.assertIsNone(srv._extract_torso_target(reset_request()))
        np.testing.assert_allclose(srv._extract_torso_target(reset_request([0.78, 1.57, 0.14])), [0.78, 1.57, 0.14])
        with self.assertRaises(ValueError):
            srv._extract_torso_target(reset_request([0.78, 1.57]))
        with self.assertRaises(ValueError):
            srv._extract_torso_target(reset_request([0.78, 1.57, 2.0]))

    def test_already_at_target_skips(self):
        torso = FakeTorso([0.785, 1.57, 0.0])
        srv = make_server(torso)
        resp = srv.Reset(reset_request(V1), mock.MagicMock())
        self.assertEqual(resp.status, "SUCCESS")
        self.assertEqual(torso.calls, [])
        self.assertEqual(srv.order, ["arm"])

    def test_moves_after_arm(self):
        torso = FakeTorso([0.83, 1.57, 0.0])
        srv = make_server(torso)
        resp = srv.Reset(reset_request(V1), mock.MagicMock())
        self.assertEqual(resp.status, "SUCCESS")
        self.assertEqual(torso.calls, [("idle", False), ("target", V1, 0.3)])
        self.assertEqual(srv.order, ["arm", "sync"])  # arm -> torso -> IK sync
        np.testing.assert_allclose(resp.observation["torso_joint_positions"].float_array.values, V1)

    def test_not_reached_is_error(self):
        torso = FakeTorso([0.83, 1.57, 0.0], reach=False)
        resp = make_server(torso).Reset(reset_request(V1), mock.MagicMock())
        self.assertEqual(resp.status, "ERROR")

    def test_no_key_arm_only(self):
        torso = FakeTorso([0.5, 1.0, 0.0])
        srv = make_server(torso)
        resp = srv.Reset(reset_request(), mock.MagicMock())
        self.assertEqual(resp.status, "SUCCESS")
        self.assertEqual(torso.calls, [])
        self.assertEqual(srv.order, ["arm"])

    def test_bad_torso_rejected_before_arm_moves(self):
        srv = make_server(FakeTorso(V1))
        ctx = mock.MagicMock()
        srv.Reset(reset_request([0.78, 1.57, 9.0]), ctx)
        ctx.set_code.assert_called_once()
        self.assertEqual(srv.order, [])


class TorsoStateTest(unittest.TestCase):
    def test_get_config_metadata(self):
        md = make_server(FakeTorso(V1)).GetConfig(None, None).metadata
        self.assertEqual(md["torso_joint_pos"], "0.7800,1.5700,0.0000")
        self.assertEqual(md["torso_pitch"], "0.7808")
        self.assertEqual(md["control_hz"], "90")

    def test_get_config_without_torso(self):
        srv = make_server(None)  # torso.get_joint_pos -> AttributeError
        md = srv.GetConfig(None, None).metadata
        self.assertNotIn("torso_joint_pos", md)
        self.assertEqual(md["arm_side"], "right")

    def test_sync_writes_torso(self):
        from core.vega.robot import VegaRobot  # same module object the server uses

        qpos = {f"R_arm_j{i + 1}": 0.0 for i in range(7)} | {"torso_j1": 0.0, "torso_j2": 0.0, "torso_j3": 0.0}
        mm = mock.MagicMock()
        mm.get_joint_pos_dict.return_value = dict(qpos)
        bot = object.__new__(VegaRobot)
        bot.ik_controller = mock.MagicMock(motion_manager=mm)
        bot._arm_joint_names = [f"R_arm_j{i + 1}" for i in range(7)]
        bot.robot = mock.MagicMock()
        bot.robot.torso = FakeTorso([0.78, 1.57, 0.14])
        with mock.patch("core.vega.robot.robot_utils", None):
            bot.sync_motion_manager_with_arm(np.full(7, 0.1))
        sent = mm.set_joint_pos.call_args[0][0]
        self.assertEqual([sent[f"torso_j{i}"] for i in (1, 2, 3)], [0.78, 1.57, 0.14])
        self.assertEqual(sent["R_arm_j1"], 0.1)


if __name__ == "__main__":
    unittest.main()
