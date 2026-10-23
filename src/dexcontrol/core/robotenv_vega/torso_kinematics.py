"""Planar shoulder X/Z/pitch IK, using the running robot's URDF.

Pitch is rotation about base +Y: zero is an upright arm_center frame,
positive tilts its +Z axis forward. It is NOT Torso.pitch_angle (from horizontal).
"""
from itertools import product
import xml.etree.ElementTree as ET

import numpy as np


def rotate(vector, angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([c*vector[0]-s*vector[1], s*vector[0]+c*vector[1]])


class TorsoKinematics:
    def __init__(self, origins, limits, velocity_limits):
        self.origins = np.asarray(origins, dtype=float)
        self.limits = np.asarray(limits, dtype=float)
        self.velocity_limits = np.asarray(velocity_limits, dtype=float)
        if (self.origins.shape != (4, 2) or self.limits.shape != (3, 2)
                or self.velocity_limits.shape != (3,) or not np.all(np.isfinite(self.origins))
                or not np.all(np.isfinite(self.limits)) or not np.all(np.isfinite(self.velocity_limits))
                or np.any(self.limits[:, 0] >= self.limits[:, 1]) or np.any(self.velocity_limits <= 0)):
            raise ValueError("Invalid torso geometry or limits")

    @classmethod
    def from_urdf(cls, path, component_limits, component_velocity_limits):
        root = ET.parse(path).getroot()
        joints = {j.attrib["name"]: j for j in root.findall("joint")}
        origins, limits, velocities = [], [], []
        parent = "base"
        for i, name in enumerate(("torso_j1", "torso_j2", "torso_j3", "arm_center_j0")):
            joint = joints[name]
            if joint.find("parent").attrib["link"] != parent:
                raise ValueError("Unsupported torso chain: expected base to arm_center")
            parent = joint.find("child").attrib["link"]
            origin = joint.find("origin")
            xyz = np.fromstring(origin.attrib["xyz"], sep=" ")
            rpy = np.fromstring(origin.attrib.get("rpy", "0 0 0"), sep=" ")
            if xyz.shape != (3,) or abs(xyz[1]) > 1e-8 or not np.allclose(rpy, 0):
                raise ValueError("Unsupported non-planar torso URDF")
            origins.append(xyz[[0, 2]])
            if i < 3:
                axis = np.fromstring(joint.find("axis").attrib["xyz"], sep=" ")
                if joint.attrib["type"] != "revolute" or not np.allclose(axis, [0, (-1, 1, -1)[i], 0]):
                    raise ValueError("Unsupported torso joint axis")
                limit = joint.find("limit").attrib
                limits.append([max(float(limit["lower"]), component_limits[i][0]),
                               min(float(limit["upper"]), component_limits[i][1])])
                velocities.append(min(float(limit["velocity"]), component_velocity_limits[i], .2))
            elif joint.attrib["type"] != "fixed" or parent != "arm_center":
                raise ValueError("Expected fixed shoulder center frame")
        return cls(origins, limits, velocities)

    def points(self, q):
        q1, q2, q3 = q
        points = [self.origins[0].copy()]
        for offset, angle in zip(self.origins[1:], (q1, q1-q2, q1-q2+q3)):
            points.append(points[-1] + rotate(offset, angle))
        return np.asarray(points)

    def forward(self, q):
        x, z = self.points(q)[-1]
        return np.array([x, z, -q[0]+q[1]-q[2]])

    def inverse(self, pose, reference):
        pose, reference = np.asarray(pose, dtype=float), np.asarray(reference, dtype=float)
        if pose.shape != (3,) or reference.shape != (3,) or not np.all(np.isfinite([pose, reference])):
            raise ValueError("Expected finite shoulder pose and joint angles")
        theta = -pose[2]
        wrist = pose[:2] - self.origins[0] - rotate(self.origins[3], theta)
        lengths = np.linalg.norm(self.origins[1:3], axis=1)
        offsets = np.arctan2(self.origins[1:3, 1], self.origins[1:3, 0])
        cosine = (wrist @ wrist - lengths @ lengths) / (2*np.prod(lengths))
        if abs(cosine) >= 1 - 1e-6:
            raise ValueError("Shoulder target unreachable or near a torso singularity")
        candidates = []
        for delta in (np.arccos(cosine), -np.arccos(cosine)):
            q1 = np.arctan2(wrist[1], wrist[0]) - np.arctan2(lengths[1]*np.sin(delta), lengths[0]+lengths[1]*np.cos(delta)) - offsets[0]
            q2 = offsets[1] - offsets[0] - delta
            for turn1, turn2 in product((-1, 0, 1), repeat=2):
                a, b = q1+turn1*2*np.pi, q2+turn2*2*np.pi
                q = np.array([a, b, theta-a+b])
                if np.all(q >= self.limits[:, 0]) and np.all(q <= self.limits[:, 1]):
                    candidates.append(q)
        if not candidates:
            raise ValueError("Shoulder target exceeds torso joint limits")
        return min(candidates, key=lambda q: np.linalg.norm(q-reference))

    def plan(self, q, pose):
        """Validate a Cartesian path, then time its joint segments by speed limits.

        Shoulder waypoints are at most 1 mm / 0.002 rad apart. Joint interpolation
        between valid waypoints stays inside joint bounds and respects every
        joint's velocity limit. No target or path is silently clipped.
        """
        q, pose = np.asarray(q, dtype=float), np.asarray(pose, dtype=float)
        self.inverse(pose, q)  # Reject invalid endpoints before allocating a path.
        start = self.forward(q)
        count = max(1, int(np.ceil(np.max(np.abs(pose-start) / [.001, .001, .002]))))
        points = [q.copy()]
        for fraction in np.linspace(0., 1., count+1)[1:]:
            points.append(self.inverse(start + (pose-start)*fraction, points[-1]))
        return TorsoTrajectory(points, self.velocity_limits)


class TorsoTrajectory:
    def __init__(self, points, velocity_limits):
        self.points = np.asarray(points, dtype=float)
        durations = np.max(np.abs(np.diff(self.points, axis=0)) / velocity_limits, axis=1)
        self.times = np.r_[0., np.cumsum(durations)]
        self.duration = float(self.times[-1])

    def time_at(self, measured):
        """Project measured joints onto the path to track physical progress."""
        starts = self.points[:-1]
        segments = np.diff(self.points, axis=0)
        lengths_squared = np.sum(segments*segments, axis=1)
        fractions = np.divide(
            np.sum((np.asarray(measured)-starts)*segments, axis=1),
            lengths_squared, out=np.zeros_like(lengths_squared),
            where=lengths_squared > 0)
        fractions = np.clip(fractions, 0., 1.)
        projected = starts + fractions[:, None]*segments
        i = int(np.argmin(np.sum((projected-measured)**2, axis=1)))
        return float(self.times[i] + fractions[i]*(self.times[i+1]-self.times[i]))

    def sample(self, elapsed):
        if elapsed >= self.duration:
            return self.points[-1].copy(), np.zeros(3)
        i = min(int(np.searchsorted(self.times, elapsed, side="right")-1), len(self.points)-2)
        duration = self.times[i+1]-self.times[i]
        velocity = (self.points[i+1]-self.points[i]) / duration
        return self.points[i] + velocity*(elapsed-self.times[i]), velocity
