"""Static center of mass and signed clearance to a fixed support triangle."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np
from scipy.spatial import ConvexHull


class StaticStability:
    """Level-ground estimate; no steering/caster motion, payload or acceleration."""

    def __init__(self, urdf_path, settings):
        import pinocchio as pin

        self.pin = pin
        self.model = pin.buildModelFromUrdf(str(urdf_path))
        self.data = self.model.createData()
        self.q = pin.neutral(self.model)
        self.joints = {
            self.model.names[i]: self.model.joints[i]
            for i in range(1, self.model.njoints)
        }
        self.frames, self.masses, self.centers = [], [], []
        for link in ET.parse(urdf_path).getroot().findall("link"):
            inertial = link.find("inertial")
            if inertial is None:
                continue  # Fixed reference frames have no physical mass.
            mass = float(inertial.find("mass").get("value"))
            origin = inertial.find("origin")
            center = np.fromstring(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            if (
                not np.isfinite(mass)
                or mass < 0
                or center.shape != (3,)
                or not np.isfinite(center).all()
            ):
                raise ValueError(f"Invalid inertia for {link.get('name')}")
            self.frames.append(self.model.getFrameId(link.get("name")))
            self.masses.append(mass)
            self.centers.append(center)
        self.masses = np.asarray(self.masses)
        self.centers = np.asarray(self.centers)
        self.mass = float(self.masses.sum())
        if self.mass <= 0:
            raise ValueError("Robot mass must be positive")
        points = np.asarray(settings["support_points_xy_m"], dtype=float)
        self.support = points[
            ConvexHull(points).vertices
        ]  # CCW, reject degenerate support.
        self.ground_z = float(settings["ground_z_m"])

    def evaluate(self, named_positions):
        # Fixed wheels are intentional. Only supplied articulated joints change.
        q = self.q.copy()
        for name, value in named_positions.items():
            if name not in self.joints:
                raise ValueError(f"Unknown stability joint: {name}")
            joint = self.joints[name]
            if joint.nq == 1:
                q[joint.idx_q] = value
            elif joint.nq == 2 and joint.nv == 1:
                q[joint.idx_q : joint.idx_q + 2] = [np.cos(value), np.sin(value)]
            else:
                raise ValueError(f"Unsupported stability joint: {name}")
        if not np.isfinite(q).all():
            raise ValueError("Nonfinite stability joint state")
        self.pin.framesForwardKinematics(self.model, self.data, q)
        positions = np.array(
            [
                self.data.oMf[frame].act(center)
                for frame, center in zip(self.frames, self.centers)
            ]
        )
        # Sum XML link masses explicitly, including the fixed root/base. A
        # fixed-base dynamics model's moving-joint CoM can omit this mass.
        com = np.sum(positions * self.masses[:, None], axis=0) / self.mass
        edge = np.roll(self.support, -1, axis=0) - self.support
        offset = com[:2] - self.support
        gaps = (edge[:, 0] * offset[:, 1] - edge[:, 1] * offset[:, 0]) / np.linalg.norm(
            edge, axis=1
        )
        margin = float(gaps.min())
        return {
            "inside_support": margin >= 0,
            "margin_m": margin,
            "com_m": com.tolist(),
            "projection_m": [float(com[0]), float(com[1]), self.ground_z],
            "mass_kg": self.mass,
            "support_xy_m": self.support.tolist(),
            "edge_margins_m": gaps.tolist(),
            "ground_z_m": self.ground_z,
            "basis": "Static, level ground, fixed wheel support points, no payload",
        }


@dataclass(frozen=True)
class StabilityDecision:
    guarded: bool
    accepted: bool | None
    reason: str
    margin_m: float
    target_margin_m: float | None
    com_m: list[float]
    mass_kg: float


class StabilityPolicy:
    """Approve endpoints using every fixed support edge, including recovery."""

    TOLERANCE_M = 1e-9

    def __init__(self, model, config):
        self.model, self.config = model, config
        self.guarded = True

    def evaluate(
        self, measured, target=None, non_position_motion=False, base_motion=False
    ):
        current = self.model.evaluate(measured)
        edges = np.asarray(current["edge_margins_m"])
        threshold = self.config.release_m if self.guarded else self.config.approach_m
        self.guarded = bool(current["margin_m"] < threshold)
        accepted, reason, target_margin = None, "", None
        if target is not None:
            goal = self.model.evaluate(target)
            target_margin = goal["margin_m"]
            goal_edges = np.asarray(goal["edge_margins_m"])
            inside = edges < self.config.margin_m
            floor = np.where(inside, edges, self.config.margin_m)
            bad = np.where(
                inside, goal_edges < floor - self.TOLERANCE_M, goal_edges < floor
            )
            if np.any(inside) and base_motion:
                accepted, reason = (
                    False,
                    "Base motion is blocked below the static stability margin",
                )
            elif np.any(bad):
                accepted, reason = (
                    False,
                    "Target reduces recovery support clearance or enters the static stability margin",
                )
            elif (
                np.any(inside)
                and (non_position_motion or measured != target)
                and not np.any(goal_edges[inside] > edges[inside] + self.TOLERANCE_M)
            ):
                accepted, reason = (
                    False,
                    "Target does not increase recovery support clearance",
                )
            else:
                accepted = True
        return StabilityDecision(
            self.guarded,
            accepted,
            reason,
            current["margin_m"],
            target_margin,
            current["com_m"],
            current["mass_kg"],
        )
