"""Offline worker/endpoint and 200 Hz writer overhead; never connects to a robot."""

import argparse
import json
import platform
import time
from collections import Counter
from importlib.metadata import version
from pathlib import Path

import numpy as np

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.model_worker import ModelPolicy
from dexcontrol.core.control.motion_safety import MotionSafety


class RecordedDevice:
    def __init__(self, joint_names, limits=None):
        self.joint_name = joint_names
        self.joint_pos_limit = limits
        self.q = np.zeros(len(joint_names))
        self.dq = np.zeros(len(joint_names))

    def get_joint_pos(self):
        return self.q.copy()

    def get_joint_vel(self):
        return self.dq.copy()


def percentiles(values):
    return dict(
        zip(
            ("p50_ms", "p95_ms", "p99_ms", "max_ms"),
            np.percentile(values, [50, 95, 99, 100]).tolist(),
        )
    )


def writer_timing(model, config, q, seconds):
    device = RecordedDevice(
        model.joint_names,
        np.column_stack(
            (
                model.model.lowerPositionLimit[model.q_indices],
                model.model.upperPositionLimit[model.q_indices],
            )
        ),
    )
    device.q = q.copy()
    safety = MotionSafety({"robot": device}, config, model)
    owner = object()
    timings, lateness, states, worker_ms, ages = [], [], Counter(), [], []
    next_tick = time.monotonic()
    try:
        for _ in range(round(seconds * 200)):
            now = time.monotonic()
            lateness.append(max(0, now - next_tick) * 1000)
            started = time.perf_counter()
            commands = {"robot": {"pos": q.copy(), "vel": np.zeros_like(q)}}
            safety.filter(commands, {}, owner)
            timings.append((time.perf_counter() - started) * 1000)
            status = safety.snapshot()
            states[status["state"]] += 1
            worker_ms.append(status["check_ms"])
            if status["result_age_s"] is not None:
                ages.append(status["result_age_s"] * 1000)
            next_tick += 0.005
            time.sleep(max(0.0, next_tick - time.monotonic()))
        return {
            "writer_compute": percentiles(timings),
            "scheduling_lateness": percentiles(lateness),
            "worker_compute": percentiles(worker_ms),
            "result_age": percentiles(ages),
            "states": dict(states),
            "ticks": len(timings),
        }
    finally:
        safety.close()


def main():
    from benchmark_collision_hulls import presets
    from dexbot_utils.configs import get_robot_config

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-urdf", type=Path)
    parser.add_argument("--control-config", type=Path)
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument(
        "--writer-seconds",
        type=float,
        default=5.0,
        help="Seconds per far/near writer run",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or not 0.1 <= args.writer_seconds <= 300:
        parser.error("Positive iterations and 0.1–300 writer seconds required")
    robot = get_robot_config("vega_1")
    config = load_config(args.control_config)
    c = config.collision
    model = CoalCollisionModel.for_robot(
        args.robot_urdf or robot.urdf_path,
        c.collision_urdf,
        c.srdf,
        c.excluded_links,
        c.max_outer_error_m,
        robotiq_sides=("left", "right"),
        model_config=config.model,
    )
    policy = ModelPolicy(model, config)
    rng = np.random.default_rng(args.seed)
    lower, upper = (
        model.model.lowerPositionLimit[model.q_indices],
        model.model.upperPositionLimit[model.q_indices],
    )
    monitor_ms, approval_ms, stability_ms, outcomes = [], [], [], Counter()
    for _ in range(args.iterations):
        q = rng.uniform(lower, upper)
        target = np.clip(q + rng.uniform(-0.1, 0.1, len(q)), lower, upper)
        started = time.perf_counter()
        policy.evaluate(q)
        monitor_ms.append((time.perf_counter() - started) * 1000)
        started = time.perf_counter()
        result = policy.evaluate(q, target)
        approval_ms.append((time.perf_counter() - started) * 1000)
        outcomes["accepted" if result.accepted else "blocked"] += 1
        if policy.stability:
            started = time.perf_counter()
            policy.stability.model.evaluate(dict(zip(model.joint_names, q)))
            stability_ms.append((time.perf_counter() - started) * 1000)
    poses = presets(robot, model, config)
    report = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            name: version(name)
            for name in ("pin", "coal", "trimesh", "numpy", "dexmate-urdf")
        },
        "iterations": args.iterations,
        "seed": args.seed,
        "shapes": len(model.shape_meshes),
        "shape_pairs": len(model.shape_pairs),
        "link_pairs": len(model.link_pairs),
        "monitor": percentiles(monitor_ms),
        "endpoint_approval": percentiles(approval_ms),
        "static_stability_only": percentiles(stability_ms) if stability_ms else None,
        "outcomes": dict(outcomes),
        "far_writer": writer_timing(
            model, config, poses["Initial arms / upright torso"], args.writer_seconds
        ),
        "near_writer": writer_timing(
            model, config, poses["Folded"], args.writer_seconds
        ),
        "scope": "Offline local CPU, 1 mm hulls, two full-stroke Robotiq envelopes, fixed-support static stability. Writer includes feedback copies, IPC and output selection; excludes SDK I/O, IK, motor tracking and physical braking. Endpoint checks only, no path or dynamic tipping guarantee.",
    }
    text = json.dumps(report, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n")


if __name__ == "__main__":
    main()
