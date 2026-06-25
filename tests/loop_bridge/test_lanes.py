"""Tests for the bridge's bus I/O lanes (obs poll + action lane), with fakes."""

from __future__ import annotations

import threading

from conftest import FakeApplier, FakeConsumer, make_observation

from loop_bridge import lanes
from loop_bridge.action_consumer import ArmActionBackend


def _backends():
    return [
        ArmActionBackend(
            FakeApplier(), arm_prefix="robot0", action_space="target_cartesian_delta"
        )
    ]


# --- run_obs_poll ---


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


# --- run_action_lane ---


def _patch_consumer(monkeypatch, *, fault=None, frames=None):
    consumers: list[FakeConsumer] = []

    class _ConsumerConn:
        @staticmethod
        def connect(addr):
            consumer = FakeConsumer(frames=frames, fault=fault)
            consumers.append(consumer)
            return consumer

    monkeypatch.setattr(lanes, "SourceConsumer", _ConsumerConn)
    return consumers


def _lane_kwargs(stop, register, sleep):
    return dict(
        stop_event=stop,
        loop_addr="loop:1",
        backends=_backends(),
        action_source_id="robot-action",
        register_consumer=register,
        sleep=sleep,
    )


def test_action_lane_returns_immediately_when_stop_already_set(monkeypatch):
    consumers = _patch_consumer(monkeypatch)
    stop = threading.Event()
    stop.set()
    registered = []

    lanes.run_action_lane(**_lane_kwargs(stop, registered.append, lambda _: None))

    assert consumers == []  # never connected
    assert registered == []


def test_action_lane_retries_on_fault_then_stops(monkeypatch):
    consumers = _patch_consumer(monkeypatch, fault=RuntimeError("source not open yet"))
    stop = threading.Event()
    registered = []

    lanes.run_action_lane(**_lane_kwargs(stop, registered.append, lambda _: stop.set()))

    assert len(consumers) == 1
    assert consumers[0].subscribed == ["robot-action"]
    assert registered[0] is not None and registered[-1] is None
    assert consumers[0].closed is True  # underlying consumer cleaned up


def test_action_lane_bails_before_run_if_stop_lands_during_setup(monkeypatch):
    consumers = _patch_consumer(monkeypatch, frames=[])
    stop = threading.Event()

    def register(consumer):
        if consumer is not None:
            stop.set()

    lanes.run_action_lane(**_lane_kwargs(stop, register, lambda _: None))

    assert len(consumers) == 1
    assert consumers[0].subscribed == []  # run()/subscribe never reached
    assert consumers[0].closed is True
