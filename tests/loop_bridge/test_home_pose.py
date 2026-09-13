"""Verify pose selection before startup motion and on subsequent Home commands."""

from types import SimpleNamespace

import numpy as np
import pytest

from dexcontrol.core.robotenv_vega import server


@pytest.mark.parametrize(
    ("arm", "v1", "v2"),
    [
        (
            "left",
            [-1.4234, 1.3524, 2.8707, -1.981, 0.6751, -0.1662, 0.068],
            [1.2373, 0.2848, 0.2404, -1.5499, 1.5265, -0.0526, 0.3908],
        ),
        (
            "right",
            [1.4234, -1.3524, -2.8707, -1.981, -0.1515, 0.1662, -0.068],
            [-1.2373, -0.2848, -0.2404, -1.5499, -1.0265, 0.0526, -0.3908],
        ),
    ],
)
def test_startup_and_home_use_selected_pose_without_changing_legacy_default(
    monkeypatch, arm, v1, v2
):
    robot = SimpleNamespace(
        launch_robot=lambda: None,
        safe_transit_pose=np.zeros(7),
        interpolation_enabled=False,
    )
    monkeypatch.setattr(server.VegaRobot, "build", lambda **kwargs: robot)
    monkeypatch.setattr(
        server.VegaRobotEnvService, "_fetch_firmware_version", lambda self: "test"
    )
    monkeypatch.setattr(
        server.VegaRobotEnvService, "_create_observation", lambda self: ({}, 0)
    )
    motions = []
    monkeypatch.setattr(
        server.VegaRobotEnvService,
        "_execute_reset_sequence",
        lambda self, target: motions.append(target.copy()),
    )

    for frame_type, target in [
        ("vega-1-pro_torso_frame_v2", v2),
        ("vega-1-pro_torso_frame_v1", v1),
        ("vega_mobile_base", v1),
    ]:
        motions.clear()
        service = server.VegaRobotEnvService(arm_side=arm, frame_type=frame_type)
        assert len(motions) == 1
        np.testing.assert_array_equal(motions[0], target)

        response = service.Reset(server.robotenv_pb2.ResetRequest(mode="home"), None)
        assert response.status == "SUCCESS"
        assert len(motions) == 2
        np.testing.assert_array_equal(motions[1], target)
