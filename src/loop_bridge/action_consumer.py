"""Consume ``robot-action`` off Loop's Source Bus and execute it on the robot.

The write-side counterpart to the obs poll: it SUBSCRIBES to the single
``robot-action`` source and, per frame, dispatches each arm's slice to that arm's
``Step`` applier — the call that moves the arm. One robot may have several arm
backends (a bimanual Vega = two per-arm services over one shared hardware unit);
each decodes its own ``robotN.`` namespace from the shared frame.

The appliers are injected (each forwards to a RobotEnv service's ``Step``), so this
module depends only on loop-sdk and is testable with fake appliers. ``run()`` is a
blocking subscribe loop meant to own a thread; ``close()`` cancels the in-flight
subscription from another thread so the loop unwinds.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from loop_sdk import RobotFrame

from loop_bridge.robot_action import DEFAULT_ACTION_SPACE, action_from_state
from loop_bridge.robot_obs import DEFAULT_ARM_PREFIX

LOGGER = logging.getLogger("loop_bridge.vega.action")


class ArmActionBackend:
    """One arm's action lane: decode its ``robotN.`` slice and execute it via ``Step``.

    ``applier`` is any object with ``step(action, action_space, gripper_action_space)``.
    """

    def __init__(
        self,
        applier: Any,
        *,
        arm_prefix: str = DEFAULT_ARM_PREFIX,
        action_space: str = DEFAULT_ACTION_SPACE,
        gripper_action_space: str = "",
    ) -> None:
        self._applier = applier
        self._arm_prefix = arm_prefix
        self._action_space = action_space
        self._gripper_action_space = gripper_action_space

    def apply(self, state: Any) -> None:
        """Decode this arm's action from one frame and execute it; never raise on a tick.

        Returns silently if the frame carries no command for this arm (it holds). A
        malformed slice or a failed Step skips this tick — it must NOT tear down the
        shared subscription.
        """
        try:
            action = action_from_state(state, self._arm_prefix, self._action_space)
        except ValueError as error:
            LOGGER.warning(
                "dropping malformed robot-action for %s: %s", self._arm_prefix, error
            )
            return
        if action is None:
            return  # no command for this arm this tick
        try:
            self._applier.step(action, self._action_space, self._gripper_action_space)
        except Exception as error:
            LOGGER.warning(
                "robot-action Step failed for %s; skipping tick: %s",
                self._arm_prefix,
                error,
            )


class RobotActionConsumer:
    """Subscribes the single ``robot-action`` and applies it to every arm backend."""

    def __init__(
        self,
        consumer: Any,
        backends: Sequence[ArmActionBackend],
        *,
        source_id: str = "robot-action",
    ) -> None:
        self._consumer = consumer
        self._backends = tuple(backends)
        self._source_id = source_id
        self._closed = False

    def run(self) -> None:
        """Subscribe and apply actions until the source ends or ``close()`` is called.

        Blocks the calling thread. A fault after ``close()`` (the cancelled in-flight
        call) is swallowed; any other fault is logged and re-raised so the owner can
        react.
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
        if not isinstance(frame, RobotFrame):
            return  # robot-action carries RobotFrames; ignore anything else
        for backend in self._backends:
            backend.apply(frame.state)

    def close(self) -> None:
        """Cancel the in-flight subscription so ``run()`` unwinds. Idempotent."""
        self._closed = True
        self._consumer.close()
