"""Consume ``robot-action`` off Loop's Source Bus and execute it on the robot.

The write-side counterpart to the obs poll: where that loop READS state and
publishes ``robot-obs``, this one SUBSCRIBES to ``robot-action`` and drives the
robot's ``Step`` path — the call that moves the arm. The applier is injected (the
in-process bridge passes one that forwards to the RobotEnv service's ``Step``), so
this module depends only on loop-sdk and is testable with a fake applier.

``run()`` is a blocking subscribe loop meant to own a thread; ``close()`` cancels
the in-flight subscription from another thread so the loop unwinds.
"""

from __future__ import annotations

import logging
from typing import Any

from loop_sdk import RobotFrame

from loop_bridge.robot_action import DEFAULT_ACTION_SPACE, action_from_state
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

LOGGER = logging.getLogger("loop_bridge.vega.action")


class RobotActionConsumer:
    """Subscribes ``robot-action`` and applies each frame's action via the applier.

    ``applier`` is any object with ``step(action, action_space, gripper_action_space)``.
    """

    def __init__(
        self,
        consumer: Any,
        applier: Any,
        *,
        source_id: str = "robot-action",
        arm_prefix: str = DEFAULT_ARM_PREFIX,
        action_space: str = DEFAULT_ACTION_SPACE,
        gripper_action_space: str = "",
    ) -> None:
        self._consumer = consumer
        self._applier = applier
        self._source_id = source_id
        self._arm_prefix = arm_prefix
        self._action_space = action_space
        self._gripper_action_space = gripper_action_space
        self._closed = False

    def run(self) -> None:
        """Subscribe and apply actions until the source ends or ``close()`` is called.

        Blocks the calling thread. A clean end of stream returns; a subscription
        fault after ``close()`` (the cancelled in-flight call) is swallowed; any
        other fault is logged and re-raised so the owner can react.
        """
        try:
            for frame in self._consumer.subscribe(self._source_id):
                if self._closed:
                    return
                self._apply(frame)
        except Exception:
            if self._closed:
                return  # subscription was cancelled by close() — expected
            LOGGER.exception("robot-action subscription faulted")
            raise

    def _apply(self, frame: Any) -> None:
        """Decode one frame and execute it; never let a single bad frame stop the loop."""
        if not isinstance(frame, RobotFrame):
            return  # robot-action carries RobotFrames; ignore anything else
        try:
            action = action_from_state(
                frame.state, self._arm_prefix, self._action_space
            )
        except ValueError as error:
            LOGGER.warning("dropping malformed robot-action frame: %s", error)
            return
        if action is None:
            return  # no command for this arm this tick — the loop is holding
        try:
            self._applier.step(action, self._action_space, self._gripper_action_space)
        except Exception as error:
            # A single failed Step (transient hardware/transport error) skips this
            # tick — it must NOT tear down the subscription (that path is reserved
            # for genuine subscription faults in run()).
            LOGGER.warning("robot-action Step failed; skipping tick: %s", error)

    def close(self) -> None:
        """Cancel the in-flight subscription so ``run()`` unwinds. Idempotent."""
        self._closed = True
        self._consumer.close()
