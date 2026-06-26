"""Tests for the bridge's obs-poll lane (read + publish + apply pulled actions)."""

from __future__ import annotations

import threading

from conftest import make_observation

from loop_bridge import lanes


def test_obs_poll_reads_and_publishes_each_tick():
    obs = make_observation()
    published = []
    stop = threading.Event()
    ticks = {"n": 0}

    def read():
        return obs, 1000 + ticks["n"]

    def publish(observation, ts):
        published.append((observation, ts))

    def sleep(_):
        ticks["n"] += 1
        if ticks["n"] >= 3:
            stop.set()

    lanes.run_obs_poll(
        stop_event=stop,
        read_observation=read,
        publish=publish,
        period_s=0.0,
        sleep=sleep,
    )

    assert len(published) == 3
    assert [ts for _, ts in published] == [1000, 1001, 1002]


def test_obs_poll_applies_actions_each_tick():
    stop = threading.Event()
    applied = {"n": 0}
    ticks = {"n": 0}

    def apply_actions():
        applied["n"] += 1

    def sleep(_):
        ticks["n"] += 1
        if ticks["n"] >= 2:
            stop.set()

    lanes.run_obs_poll(
        stop_event=stop,
        read_observation=lambda: (make_observation(), 1),
        publish=lambda *_: None,
        period_s=0.0,
        apply_actions=apply_actions,
        sleep=sleep,
    )

    assert applied["n"] == 2  # actions pulled + applied once per tick


def test_obs_poll_returns_immediately_when_stop_already_set():
    stop = threading.Event()
    stop.set()
    published = []

    def read():
        raise AssertionError("must not read when already stopped")

    lanes.run_obs_poll(
        stop_event=stop,
        read_observation=read,
        publish=lambda *_: published.append(_),
        period_s=0.0,
        sleep=lambda _: None,
    )
    assert published == []


def test_obs_poll_skips_failed_tick_but_keeps_going():
    stop = threading.Event()
    calls = {"n": 0}

    def read():
        calls["n"] += 1
        raise RuntimeError("transient read failure")

    def sleep(_):
        if calls["n"] >= 2:
            stop.set()

    # Must not raise despite read() always failing; stops after 2 attempts.
    lanes.run_obs_poll(
        stop_event=stop,
        read_observation=read,
        publish=lambda *_: None,
        period_s=0.0,
        sleep=sleep,
    )
    assert calls["n"] == 2
