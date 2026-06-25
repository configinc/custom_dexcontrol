"""The two bus I/O lanes the Vega bridge runs as background threads.

Kept dexcontrol-free (only loop-sdk + the bridge's own bus modules) so both lanes
are unit-testable with injected seams, without importing the heavy in-process
robot stack:

  - ``run_obs_poll`` — read the current observation and publish it as ``robot-obs``
    at a fixed rate. Vega publishes obs only when ``_create_observation`` runs, so
    the bridge must DRIVE it on a clock; otherwise teleop (which needs obs to
    compute a delta) and obs (driven by the resulting action's Step) deadlock at
    startup. This lane breaks that cycle.
  - ``run_action_lane`` — subscribe ``robot-action`` and apply each frame via the
    injected applier (the in-process RobotEnv ``Step`` path), retrying until the
    source opens, with an interruptible backoff and clean cancel-on-shutdown.
"""

from __future__ import annotations

import contextlib
import logging
import threading
import time
from typing import Any, Callable, Optional, Tuple

from loop_sdk import SourceConsumer

from loop_bridge.action_consumer import RobotActionConsumer

LOGGER = logging.getLogger("loop_bridge.vega")

# Backoff between action-lane (re)subscribe attempts. ``robot-action`` only exists
# once the in-loop RCI engine is running, so the bridge may start before it.
_ACTION_RETRY_S = 2.0

# (observation map, timestamp_us) for one sample.
ReadObservation = Callable[[], Tuple[Any, int]]
Publish = Callable[[Any, int], Any]


def run_obs_poll(
    *,
    stop_event: threading.Event,
    read_observation: ReadObservation,
    publish: Publish,
    period_s: float,
    sleep: Optional[Callable[[float], None]] = None,
) -> None:
    """Read + publish one observation per ``period_s`` until ``stop_event``.

    A failed read/publish on one tick is logged and skipped — a transient hiccup
    must not kill the lane. Backoff defaults to the stop event's wait so shutdown
    wakes it immediately; tests inject a non-blocking sleep.
    """
    wait = sleep if sleep is not None else (lambda seconds: stop_event.wait(seconds))
    while not stop_event.is_set():
        try:
            observation, timestamp_us = read_observation()
            publish(observation, timestamp_us)
        except Exception:
            LOGGER.exception("robot-obs poll failed; skipping tick")
        if stop_event.is_set():
            return
        wait(period_s)


def run_action_lane(
    *,
    stop_event: threading.Event,
    loop_addr: str,
    applier: Any,
    action_source_id: str,
    arm_prefix: str,
    action_space: str,
    gripper_action_space: str,
    register_consumer: Callable[[Optional[RobotActionConsumer]], None],
    sleep: Optional[Callable[[float], None]] = None,
) -> None:
    """Subscribe ``robot-action`` and apply it via ``applier``, retrying until stop.

    ``register_consumer`` publishes the live consumer to the shutdown path so it
    can be cancelled mid-subscribe; it is cleared between attempts. The applier is
    owned by the caller (the in-process service) and is NOT closed here.
    """
    backoff = sleep if sleep is not None else (lambda seconds: stop_event.wait(seconds))
    while not stop_event.is_set():
        consumer: Optional[RobotActionConsumer] = None
        try:
            consumer = RobotActionConsumer(
                SourceConsumer.connect(loop_addr),
                applier,
                source_id=action_source_id,
                arm_prefix=arm_prefix,
                action_space=action_space,
                gripper_action_space=gripper_action_space,
            )
            # Register BEFORE the stop re-check so a stop that lands now can find
            # and cancel this consumer; then bail before blocking in run().
            register_consumer(consumer)
            if stop_event.is_set():
                return
            consumer.run()
        except (
            Exception
        ) as exc:  # connect/subscription faulted (e.g. source not open yet)
            if stop_event.is_set():
                return
            LOGGER.warning(
                "robot-action lane error; retrying in %.1fs: %s", _ACTION_RETRY_S, exc
            )
        finally:
            register_consumer(None)
            if consumer is not None:
                with contextlib.suppress(Exception):
                    consumer.close()
        if stop_event.is_set():
            return
        backoff(_ACTION_RETRY_S)


def now_us() -> int:
    """Epoch microseconds, for obs samples whose source carries no timestamp."""
    return time.time_ns() // 1_000
