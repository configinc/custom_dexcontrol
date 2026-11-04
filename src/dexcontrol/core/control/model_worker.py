"""A model-only process for periodic monitoring and endpoint approval.

There is at most one request in flight. The parent retains the corresponding
command batch; a result never grants permission to an unrelated newer target.
No SDK objects, callbacks, or robot connections are passed to the child.
"""

import multiprocessing as mp
import time
from dataclasses import dataclass

import numpy as np

from .collision import CoalCollisionModel
from .stability import StabilityDecision, StabilityPolicy, StaticStability


@dataclass(frozen=True)
class CollisionDecision:
    guarded: bool
    clearance_m: float
    pair: tuple[str, str]
    accepted: bool | None
    reason: str
    elapsed_ms: float


class EndpointPolicy:
    """Hysteresis plus pairwise endpoint approval, without path subdivision."""

    def __init__(self, model, config):
        self.model, self.config = model, config
        self.guarded = True
        excluded = {frozenset(pair) for pair in config.arms_mode_excluded_pairs}
        self.arm_pairs = np.array(
            [
                i
                for i, pair in enumerate(model.link_pairs)
                if frozenset(pair) not in excluded
            ],
            dtype=int,
        )
        self.all_pairs = np.arange(len(model.link_pairs))

    def evaluate(
        self, measured, target=None, non_position_motion=False, arm_mode=False
    ):
        started = time.perf_counter()
        pairs = self.arm_pairs if arm_mode else self.all_pairs
        current = self.model.clearances(measured)[pairs]
        index = int(np.argmin(current))
        threshold = self.config.release_m if self.guarded else self.config.approach_m
        self.guarded = bool(current[index] < threshold)
        accepted, reason = None, ""
        if target is not None:
            goal = self.model.clearances(target)[pairs]
            # Inside the margin, recovery may remain inside but must not worsen
            # ANY offending pair. Other pairs must retain the configured margin.
            inside = current < self.config.margin_m
            floor = np.where(inside, current, self.config.margin_m)
            tolerance = 2 * self.model.DISTANCE_TOLERANCE_M
            bad = np.where(inside, goal < floor - tolerance, goal < floor)
            if np.any(bad):
                index = int(np.argmin(np.where(bad, goal - floor, np.inf)))
                accepted, reason = (
                    False,
                    "Target reduces recovery clearance or enters the collision margin",
                )
            elif (
                np.any(inside)
                and (non_position_motion or not np.array_equal(measured, target))
                and not np.any(goal[inside] > current[inside] + tolerance)
            ):
                accepted, reason = False, "Target does not increase recovery clearance"
            else:
                accepted = True
        return CollisionDecision(
            self.guarded,
            float(current[index]),
            self.model.link_pairs[pairs[index]],
            accepted,
            reason,
            (time.perf_counter() - started) * 1000,
        )


@dataclass(frozen=True)
class ModelDecision:
    guarded: bool
    accepted: bool | None
    reason: str
    violations: tuple[str, ...]
    collision: CollisionDecision | None
    stability: StabilityDecision | None
    elapsed_ms: float


class ModelPolicy:
    """One approval covers both collision and static support for the same target."""

    def __init__(self, model, config):
        self.model = model
        self.collision = (
            EndpointPolicy(model, config.collision)
            if config.safety["self_collision"].enabled
            else None
        )
        self.stability = (
            StabilityPolicy(
                StaticStability(model.urdf_path, config.model), config.stability
            )
            if config.safety["static_stability"].enabled
            else None
        )

    def evaluate(
        self,
        measured,
        target=None,
        non_position_motion=False,
        base_motion=False,
        arm_mode=False,
    ):
        started = time.perf_counter()
        collision = (
            self.collision.evaluate(
                measured, target, non_position_motion or base_motion, arm_mode
            )
            if self.collision
            else None
        )
        stability = (
            self.stability.evaluate(
                dict(zip(self.model.joint_names, measured)),
                None if target is None else dict(zip(self.model.joint_names, target)),
                non_position_motion,
                base_motion,
            )
            if self.stability
            else None
        )
        checks = {"self_collision": collision, "static_stability": stability}
        violations = tuple(
            name
            for name, check in checks.items()
            if check is not None and check.accepted is False
        )
        return ModelDecision(
            guarded=any(
                check.guarded for check in checks.values() if check is not None
            ),
            accepted=None if target is None else not violations,
            reason="; ".join(
                check.reason
                for check in checks.values()
                if check is not None and check.reason
            ),
            violations=violations,
            collision=collision,
            stability=stability,
            elapsed_ms=(time.perf_counter() - started) * 1000,
        )


def _run(connection, model_args, config):
    try:
        model = CoalCollisionModel(*model_args)
        policy = ModelPolicy(model, config)
        connection.send(("ready", None))
        while True:
            request = connection.recv()
            if request is None:
                break
            connection.send(("result", policy.evaluate(*request)))
    except EOFError:
        pass  # The controller closed its endpoint during shutdown.
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        connection.close()


class ModelWorker:
    def __init__(self, model, config):
        context = mp.get_context("spawn")
        self.connection, child = context.Pipe()
        model_args = (
            str(model.urdf_path),
            str(model.srdf_path),
            model.excluded_links,
            model.max_outer_error_m,
        )
        self.process = context.Process(
            target=_run,
            args=(child, model_args, config),
            name="vega-motion-safety",
            daemon=True,
        )
        self.pending = False
        self.closed = False
        self.process.start()
        child.close()
        try:
            if not self.connection.poll(60):
                raise TimeoutError("Motion safety worker did not initialize")
            kind, value = self.connection.recv()
            if kind != "ready":
                raise RuntimeError(
                    f"Motion safety worker initialization failed: {value}"
                )
        except Exception:
            self.close()
            raise

    def submit(
        self,
        measured,
        target=None,
        non_position_motion=False,
        base_motion=False,
        arm_mode=False,
    ):
        if self.pending:
            raise RuntimeError(
                "Motion safety worker already has an outstanding request"
            )
        self.connection.send(
            (measured, target, non_position_motion, base_motion, arm_mode)
        )
        self.pending = True

    def poll(self):
        if self.connection.poll():
            kind, value = self.connection.recv()
            self.pending = False
            if kind != "result":
                raise RuntimeError(f"Motion safety worker failed: {value}")
            return value
        if not self.process.is_alive():
            raise RuntimeError("Motion safety worker exited")
        return None

    def close(self):
        if self.closed:
            return
        self.closed = True
        if self.process.is_alive():
            try:
                self.connection.send(None)
            except (BrokenPipeError, EOFError):
                pass  # The child already closed its endpoint.
            self.process.join(2)
            if self.process.is_alive():
                self.process.terminate()
                self.process.join(2)
        self.connection.close()
