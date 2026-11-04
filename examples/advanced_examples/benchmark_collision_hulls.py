"""Compare bounded outer hulls on identical offline Vega safety inputs.

Generates standalone collision URDFs and a
JSON report without modifying live configuration or opening robot connections.
"""

import argparse
import json
import os
import platform
from collections import Counter
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from benchmark_control_safety import percentiles
from collision_hull_tools import export_collision_urdf, verify_outer_hull
from dexbot_utils.configs import get_robot_config

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.model_worker import EndpointPolicy


def presets(robot_config, model, config):
    groups = {
        "torso": [f"torso_j{i}" for i in range(1, 4)],
        "left_arm": [f"L_arm_j{i}" for i in range(1, 8)],
        "right_arm": [f"R_arm_j{i}" for i in range(1, 8)],
        "head": [f"head_j{i}" for i in range(1, 4)],
    }
    result = {}
    for name, torso, arms, head in (
        ("Folded", "folded", "folded", "tucked"),
        ("Folded arms / upright torso", "home", "folded", "home"),
        ("Initial arms / upright torso", "home", "L_shape", "home"),
        ("Lifted arms / upright torso", "home", "lift_up", "home"),
    ):
        named = {}
        for component, joints in groups.items():
            pose = (
                torso if component == "torso" else head if component == "head" else arms
            )
            values = (
                config.poses["folded"][component]
                if pose == "folded"
                else robot_config.components[component].pose_pool[pose]
            )
            named.update(zip(joints, values))
        result[name] = np.clip(
            model.positions(named),
            model.model.lowerPositionLimit[model.q_indices],
            model.model.upperPositionLimit[model.q_indices],
        )
    return result


def verify_reload(original, outer, error_m):
    if (
        original.joint_names != outer.joint_names
        or original.link_pairs != outer.link_pairs
    ):
        raise ValueError("Generated URDF changed joints or checked link pairs")
    if original.shape_links != outer.shape_links:
        raise ValueError("Generated URDF changed connected-component coverage")
    np.testing.assert_array_equal(original.shape_pairs, outer.shape_pairs)
    result = []
    for a, b, first, second in zip(
        original.geometry_model.geometryObjects,
        outer.geometry_model.geometryObjects,
        original.shape_meshes,
        outer.shape_meshes,
    ):
        np.testing.assert_allclose(
            a.placement.homogeneous, b.placement.homogeneous, atol=1e-12, rtol=0
        )
        result.append(verify_outer_hull(first, second, error_m))
    return result


def sample_inputs(model, config, count, seed):
    rng = np.random.default_rng(seed)
    lower = model.model.lowerPositionLimit[model.q_indices]
    upper = model.model.upperPositionLimit[model.q_indices]
    if model.continuous.any() or not np.isfinite([lower, upper]).all():
        raise ValueError("Sampling requires finite revolute joint limits")
    starts = []
    for _ in range(count * 100):
        q = rng.uniform(lower, upper)
        if model.violation(q, config.collision.margin_m) is None:
            starts.append(q)
            if len(starts) == count:
                break
    if len(starts) != count:
        raise RuntimeError("Insufficient clear start configurations")
    return [
        (
            q,
            rng.uniform(-1, 1, len(q)),
            np.clip(q + rng.uniform(-0.2, 0.2, len(q)), lower, upper),
            rng.uniform(-1, 1, len(q)),
        )
        for q in starts
    ]


def compare(models, config, inputs):
    guards = [EndpointPolicy(m, config.collision) for m in models]
    timings = [dict(clearance=[], endpoint_approval=[]) for _ in models]
    outcomes = [[] for _ in models]
    shifts = [[] for _ in models]
    max_increase = np.zeros(len(models))
    for guard in guards:
        for _ in range(100):
            guard.evaluate(inputs[0][0], inputs[0][2])
    for i, (q, dq, target, velocity) in enumerate(inputs):
        all_distances = [None] * len(models)
        labels = [None] * len(models)
        # Rotate variant order so one variant is not always measured first.
        for j in np.roll(np.arange(len(models)), i % len(models)):
            guard, m = guards[j], models[j]
            begun = perf_counter()
            all_distances[j] = m.clearances(q)
            timings[j]["clearance"].append((perf_counter() - begun) * 1000)
            begun = perf_counter()
            result = guard.evaluate(q, target)
            timings[j]["endpoint_approval"].append((perf_counter() - begun) * 1000)
            labels[j] = "accepted" if result.accepted else result.reason
            outcomes[j].append(labels[j])
        for j in range(len(models)):
            # Starts are clear in the original; all original distances are
            # positive, so the two-sided distance/error comparison is meaningful.
            change = all_distances[0] - all_distances[j]
            shifts[j].append(float(np.max(change)))
            max_increase[j] = max(max_increase[j], float(-np.min(change)))
        if (i + 1) % 500 == 0:
            print(f"Measured {i + 1}/{len(inputs)} identical inputs", flush=True)
    accepted_everywhere = np.all(np.array(outcomes) == "accepted", axis=0)
    result = []
    for j, m in enumerate(models):
        transitions = Counter(zip(outcomes[0], outcomes[j]))
        result.append(
            {
                "vertices": sum(len(mesh.vertices) for mesh in m.shape_meshes),
                **{name: percentiles(values) for name, values in timings[j].items()},
                "shared_accepted_count": int(np.sum(accepted_everywhere)),
                "shared_accepted_approval": percentiles(
                    np.array(timings[j]["endpoint_approval"])[accepted_everywhere]
                )
                if np.any(accepted_everywhere)
                else None,
                "checks_over_5_ms": int(
                    np.sum(np.array(timings[j]["endpoint_approval"]) > 5)
                ),
                "outcomes": dict(Counter(outcomes[j])),
                "transitions_from_original": [
                    {"original": a, "variant": b, "count": n}
                    for (a, b), n in transitions.items()
                ],
                "max_distance_reduction_m": max(shifts[j]),
                "max_distance_increase_m": float(max_increase[j]),
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--errors-mm", type=float, nargs="+", default=[0.5, 1.0, 2.0])
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations < 1 or any(not np.isfinite(e) or e <= 0 for e in args.errors_mm):
        parser.error("Iterations and errors must be positive and finite")
    if len(set(args.errors_mm)) != len(args.errors_mm):
        parser.error("Error levels must be distinct")
    directory = args.output_dir.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    robot = get_robot_config("vega_1")
    config = load_config()
    c = config.collision
    baseline = CoalCollisionModel.for_robot(
        robot.urdf_path, c.collision_urdf, c.srdf, c.excluded_links
    )
    models = [baseline]
    variants = [{"label": "original", "error_m": 0.0}]
    srdf = c.srdf or Path(robot.urdf_path).with_suffix(".srdf")
    for error_mm in args.errors_mm:
        print(f"Building {error_mm:g} mm outer hulls", flush=True)
        started = perf_counter()
        path, shapes = export_collision_urdf(
            baseline, directory / f"outer_{error_mm:g}mm", error_mm / 1000
        )
        model = CoalCollisionModel.for_robot(
            robot.urdf_path, path, srdf, c.excluded_links
        )
        verified = verify_reload(baseline, model, error_mm / 1000)
        variants.append(
            {
                "label": f"{error_mm:g} mm",
                "error_m": error_mm / 1000,
                "collision_urdf": str(path),
                "build_and_verify_s": perf_counter() - started,
                "shapes": [dict(a, **b) for a, b in zip(shapes, verified)],
            }
        )
        models.append(model)
        print(
            f"Verified {sum(len(m.vertices) for m in model.shape_meshes)} vertices",
            flush=True,
        )
    poses = presets(robot, baseline, config)
    for model, variant in zip(models, variants):
        variant["presets"] = {}
        for name, q in poses.items():
            distances = model.clearances(q)
            nearest = int(np.argmin(distances))
            variant["presets"][name] = {
                "clearance_m": float(distances[nearest]),
                "pair": model.link_pairs[nearest],
                "above_margin": bool(distances[nearest] > c.margin_m),
            }
    print("Sampling original-clear starts", flush=True)
    inputs = sample_inputs(baseline, config, args.iterations, args.seed)
    np.savez_compressed(directory / "inputs.npz", values=np.array(inputs))
    measurements = compare(models, config, inputs)
    for variant, measurement in zip(variants, measurements):
        variant.update(measurement)
        if variant["max_distance_increase_m"] > 5e-6:
            raise AssertionError(
                "An outer hull reported greater clearance than the original"
            )
        if variant["max_distance_reduction_m"] > 2 * variant["error_m"] + 5e-6:
            raise AssertionError(
                "Clearance change exceeded the combined outer error bound"
            )
    result = {
        "platform": platform.platform(),
        "python": platform.python_version(),
        "packages": {
            n: version(n)
            for n in (
                "coal",
                "pin",
                "numpy",
                "scipy",
                "trimesh",
                "rtree",
                "dexmate-urdf",
            )
        },
        "thread_environment": {
            n: os.environ.get(n) for n in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS")
        },
        "iterations": args.iterations,
        "seed": args.seed,
        "original_urdf": str(baseline.urdf_path),
        "srdf": str(srdf),
        "excluded_links": c.excluded_links,
        "margin_m": c.margin_m,
        "policy": "pairwise endpoint approval only; no path subdivision",
        "shape_pairs": len(baseline.shape_pairs),
        "variants": variants,
        "scope": "Offline CPU comparison on identical inputs; excludes robot I/O, IK, writer scheduling and braking. No live configuration changes.",
    }
    report = directory / "report.json"
    report.write_text(json.dumps(result, indent=2) + "\n")
    print(f"Report: {report}", flush=True)
    for v in variants:
        print(
            json.dumps(
                {
                    k: v[k]
                    for k in (
                        "label",
                        "vertices",
                        "clearance",
                        "endpoint_approval",
                        "shared_accepted_approval",
                        "checks_over_5_ms",
                    )
                }
            )
        )


if __name__ == "__main__":
    main()
