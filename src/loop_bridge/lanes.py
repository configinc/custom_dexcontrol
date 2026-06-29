"""The obs-poll lane the Vega bridge runs as a background thread.

Kept dexcontrol-free (only the bridge's own callbacks) so it is unit-testable with
injected seams. Vega publishes obs only when ``_create_observation`` runs, so the
bridge must DRIVE it on a clock; otherwise teleop (which needs obs to compute a
delta) and obs (driven by the resulting action's Step) deadlock at startup. This
lane breaks that cycle: each tick reads + publishes the observation and applies any
freshest pulled action. The action subscription itself lives in loop-sdk's
``LoopRobotClient`` (its own thread); this lane just pulls + decodes + applies.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Optional, Tuple

LOGGER = logging.getLogger("loop_bridge.vega")

# (observation map, timestamp_us) for one sample.
ReadObservation = Callable[[], Tuple[Any, int]]
Publish = Callable[[Any, int], Any]


def run_obs_poll(
    *,
    stop_event: threading.Event,
    read_observation: ReadObservation,
    publish: Publish,
    period_s: float | Callable[[], float],
    apply_actions: Optional[Callable[[], None]] = None,
    sleep: Optional[Callable[[float], None]] = None,
) -> None:
    """Read + publish one observation per ``period_s`` (and apply pulled actions).

    A failed read/publish on one tick is logged and skipped — a transient hiccup
    must not kill the lane. ``period_s`` may be a callable so the rate can be
    re-paced live (config negotiation). ``apply_actions`` (optional) is called each
    tick after the publish to Step the freshest pulled action. Backoff defaults to
    the stop event's wait so shutdown wakes it immediately; tests inject a sleep.
    """
    wait = sleep if sleep is not None else (lambda seconds: stop_event.wait(seconds))
    while not stop_event.is_set():
        try:
            observation, timestamp_us = read_observation()
            publish(observation, timestamp_us)
            if apply_actions is not None:
                apply_actions()
        except Exception:
            LOGGER.exception("robot-obs poll failed; skipping tick")
        if stop_event.is_set():
            return
        wait(period_s() if callable(period_s) else period_s)


def now_us() -> int:
    """Epoch microseconds, for obs samples whose source carries no timestamp."""
    return time.time_ns() // 1_000
