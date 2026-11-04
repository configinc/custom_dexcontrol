"""Coal self-clearance queries, independent of robot I/O and safety policy."""

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ClearanceViolation:
    pair: tuple[str, str]
    distance_m: float
    detail: str = "clearance below margin"

    def message(self, margin):
        return (
            f"{self.pair[0]} / {self.pair[1]}: {self.detail}; "
            f"clearance={self.distance_m:.4f} m, margin={margin:.4f} m"
        )


def _solid_shapes(geometry, max_outer_error_m=0.0):
    """Represent every connected mesh component by its conservative convex hull."""
    import coal
    import trimesh

    if isinstance(geometry, coal.BVHModelBase):
        faces = np.array(
            [
                [geometry.tri_indices(i)[j] for j in range(3)]
                for i in range(geometry.num_tris)
            ]
        )
        mesh = trimesh.Trimesh(vertices=geometry.vertices(), faces=faces, process=True)
        parts = mesh.split(only_watertight=False)
        if not len(parts) or any(not part.is_watertight for part in parts):
            raise ValueError("Collision mesh must contain closed solids")
        shapes = []
        for part in parts:
            hull = part.convex_hull
            if max_outer_error_m:
                from .hulls import simplify_outer_hull

                hull, _ = simplify_outer_hull(hull, max_outer_error_m)
            vertices, triangles = coal.StdVec_Vec3s(), coal.StdVec_Triangle()
            vertices.extend(hull.vertices)
            triangles.extend([coal.Triangle(*map(int, face)) for face in hull.faces])
            shapes.append((coal.Convex(vertices, triangles), hull))
        return shapes
    if isinstance(geometry, coal.Sphere):
        mesh = trimesh.creation.icosphere(subdivisions=2, radius=geometry.radius)
    elif isinstance(geometry, coal.Box):
        mesh = trimesh.creation.box(extents=2 * geometry.halfSide)
    elif isinstance(geometry, coal.Cylinder):
        mesh = trimesh.creation.cylinder(
            radius=geometry.radius, height=2 * geometry.halfLength
        )
    elif isinstance(geometry, coal.Capsule):
        mesh = trimesh.creation.capsule(
            radius=geometry.radius, height=2 * geometry.halfLength
        )
    else:
        raise ValueError(f"Unsupported collision geometry: {type(geometry).__name__}")
    return [(geometry, mesh)]


class CoalCollisionModel:
    """Pinocchio kinematics and Coal signed distances between solid shapes.

    Meshes use connected-component convex hulls, an explicit conservative policy,
    not a fallback on failed queries. Spaces between separate pieces are retained.
    Primitives remain analytic Coal shapes. Mutable FK/query data has one writer.
    """

    DISTANCE_TOLERANCE_M = 1e-6

    def __init__(self, urdf_path, srdf_path, excluded_links=(), max_outer_error_m=0.0):
        import coal
        import pinocchio as pin

        self.pin = pin
        self.urdf_path = Path(urdf_path)
        self.srdf_path = Path(srdf_path)
        self.max_outer_error_m = max_outer_error_m
        if (
            isinstance(max_outer_error_m, bool)
            or not np.isfinite(max_outer_error_m)
            or max_outer_error_m < 0
        ):
            raise ValueError("Outer hull error must be nonnegative and finite")
        root = ET.parse(self.urdf_path).getroot()
        links = {link.attrib["name"] for link in root.findall("link")}
        self.excluded_links = tuple(excluded_links)
        if set(excluded_links) - links:
            raise ValueError(f"Unknown excluded links: {set(excluded_links) - links}")
        disabled = set()
        for pair in ET.parse(srdf_path).getroot().findall("disable_collisions"):
            names = (pair.attrib["link1"], pair.attrib["link2"])
            if not set(names) <= links:
                raise ValueError(f"SRDF references unknown links: {names}")
            disabled.add(frozenset(names))
        if any(j.find("mimic") is not None for j in root.findall("joint")):
            raise ValueError(
                "Collision model requires explicit joints, not mimic joints"
            )
        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        self.data = self.model.createData()
        loaded = pin.buildGeomFromUrdf(
            self.model,
            str(self.urdf_path),
            pin.GeometryType.COLLISION,
            package_dirs=[str(self.urdf_path.parent)],
        )
        self.geometry_model = pin.GeometryModel()
        self.shape_links, self.shape_meshes = [], []
        mesh_cache = {}
        for obj in loaded.geometryObjects:
            link = self.model.frames[obj.parentFrame].name
            if link in excluded_links:
                continue
            if isinstance(obj.geometry, coal.BVHModelBase):
                key = (obj.meshPath, tuple(obj.meshScale))
                if key not in mesh_cache:
                    mesh_cache[key] = _solid_shapes(obj.geometry, max_outer_error_m)
                shapes = mesh_cache[key]
            else:
                shapes = _solid_shapes(obj.geometry)
            for i, (shape, display_mesh) in enumerate(shapes):
                shape.computeLocalAABB()
                bounds = np.array([shape.aabb_local.min_, shape.aabb_local.max_])
                if not np.isfinite(bounds).all() or np.any(bounds[1] <= bounds[0]):
                    raise ValueError(f"Invalid collision geometry bounds: {obj.name}")
                self.geometry_model.addGeometryObject(
                    pin.GeometryObject(
                        f"{obj.name}_part_{i}",
                        obj.parentJoint,
                        obj.parentFrame,
                        obj.placement,
                        shape,
                    )
                )
                self.shape_links.append(link)
                self.shape_meshes.append(display_mesh)
        objects = self.geometry_model.geometryObjects
        if not len(objects):
            raise ValueError("Collision model has no included geometry")
        self.covered_links = tuple(dict.fromkeys(self.shape_links))
        self.uncovered_links = tuple(sorted(links - set(self.covered_links)))
        local_min = np.array([obj.geometry.aabb_local.min_ for obj in objects])
        local_max = np.array([obj.geometry.aabb_local.max_ for obj in objects])
        self._box_centers = (local_min + local_max) * 0.5
        self._box_extents = (local_max - local_min) * 0.5
        ancestors = []
        for obj in objects:
            chain = []
            joint = obj.parentJoint
            while joint:
                chain.append(joint)
                joint = int(self.model.parents[joint])
            ancestors.append(chain)
        ids = sorted({j for chain in ancestors for j in chain})
        if not ids or any(
            self.model.joints[j].nv != 1 or self.model.joints[j].nq not in (1, 2)
            for j in ids
        ):
            raise ValueError("Collision model requires scalar articulated joints")
        self.joint_names = tuple(self.model.names[j] for j in ids)
        joint_types = {
            j.attrib["name"]: j.attrib["type"] for j in root.findall("joint")
        }
        if any(
            joint_types[name] not in ("revolute", "continuous")
            for name in self.joint_names
        ):
            raise ValueError(
                "Collision sweep bounds currently support rotational joints"
            )
        self.q_indices = np.array([self.model.joints[j].idx_q for j in ids])
        self.continuous = np.array([self.model.joints[j].nq == 2 for j in ids])
        self._q = pin.neutral(self.model)

        # Bound movement of EVERY point, including rotation of elongated shapes.
        # A sphere-center bound would be insufficient for boxes/capsules/meshes.
        reach = np.zeros((len(objects), len(ids)))
        influence = np.zeros_like(reach, dtype=bool)
        for i, (obj, chain) in enumerate(zip(objects, ancestors)):
            bounds = obj.geometry.aabb_local
            corners = (
                np.array(np.meshgrid(*zip(bounds.min_, bounds.max_))).reshape(3, -1).T
            )
            points = corners @ obj.placement.rotation.T + obj.placement.translation
            radius = float(np.max(np.linalg.norm(points, axis=1)))
            for joint in chain:
                k = ids.index(joint)
                reach[i, k], influence[i, k] = radius, True
                radius += np.linalg.norm(self.model.jointPlacements[joint].translation)
        groups = {}
        for i, first in enumerate(self.shape_links):
            for j in range(i + 1, len(objects)):
                second = self.shape_links[j]
                if first == second or frozenset((first, second)) in disabled:
                    continue
                if objects[i].parentJoint == objects[j].parentJoint:
                    continue
                groups.setdefault((first, second), []).append((i, j))
        if not groups:
            raise ValueError("Collision model has no enabled pairs")
        self.link_pairs = tuple(groups)
        self.group_starts = np.cumsum([0] + [len(p) for p in groups.values()])[:-1]
        self.shape_pairs = np.array(
            [pair for group in groups.values() for pair in group]
        )
        self._pair_groups = np.repeat(
            np.arange(len(groups)), [len(p) for p in groups.values()]
        )
        for i, j in self.shape_pairs:
            self.geometry_model.addCollisionPair(pin.CollisionPair(int(i), int(j)))
        a, b = self.shape_pairs.T
        bounds = (reach[a] + reach[b]) * (influence[a] ^ influence[b])
        self.sweep_bounds = np.maximum.reduceat(bounds, self.group_starts, axis=0)
        self.geometry_data = pin.GeometryData(self.geometry_model)
        for request in self.geometry_data.distanceRequests:
            request.enable_signed_distance = True
            request.gjk_tolerance = self.DISTANCE_TOLERANCE_M
            request.epa_tolerance = self.DISTANCE_TOLERANCE_M
        self.shape_distances = np.full(len(self.shape_pairs), np.nan)

    @classmethod
    def for_robot(
        cls,
        robot_urdf,
        collision_urdf=None,
        srdf=None,
        excluded_links=(),
        max_outer_error_m=0.0,
        robotiq_sides=(),
        model_config=None,
    ):
        robot_urdf = Path(robot_urdf)
        collision_urdf = Path(collision_urdf) if collision_urdf else robot_urdf
        srdf = Path(srdf) if srdf else robot_urdf.with_suffix(".srdf")

        def joints(path):
            return {
                j.attrib["name"]: (
                    j.attrib["type"],
                    tuple(
                        (tag, tuple(sorted(j.find(tag).attrib.items())))
                        for tag in ("parent", "child", "origin", "axis")
                        if j.find(tag) is not None
                    ),
                )
                for j in ET.parse(path).getroot().findall("joint")
            }

        if joints(robot_urdf) != joints(collision_urdf):
            raise ValueError("Robot and collision URDF kinematics do not match")
        if robotiq_sides:
            from .robot_model import attach_robotiq

            if model_config is None:
                raise ValueError(
                    "Robotiq attachment requires model settings from control config"
                )
            directory, collision_urdf, srdf = attach_robotiq(
                collision_urdf,
                srdf,
                robotiq_sides,
                model_config,
            )
            try:
                model = cls(collision_urdf, srdf, excluded_links, max_outer_error_m)
            except Exception:
                directory.cleanup()
                raise
            model._asset_directory = directory
            return model
        return cls(collision_urdf, srdf, excluded_links, max_outer_error_m)

    def positions(self, named_positions):
        missing = set(self.joint_names) - named_positions.keys()
        if missing:
            raise ValueError(f"Missing collision joint feedback: {sorted(missing)}")
        q = np.array([named_positions[name] for name in self.joint_names], dtype=float)
        if q.shape != (len(self.joint_names),) or not np.isfinite(q).all():
            raise ValueError("Invalid collision joint feedback")
        return q

    def _place(self, positions):
        q = np.asarray(positions, dtype=float)
        if q.shape != self.q_indices.shape or not np.isfinite(q).all():
            raise ValueError("Expected finite positions in collision joint_names order")
        self._q[self.q_indices] = q
        indices = self.q_indices[self.continuous]
        self._q[indices], self._q[indices + 1] = (
            np.cos(q[self.continuous]),
            np.sin(q[self.continuous]),
        )
        self.pin.updateGeometryPlacements(
            self.model, self.data, self.geometry_model, self.geometry_data, self._q
        )

    def clearances(self, positions):
        """Minimum signed solid distance for each link pair, in meters."""
        self._place(positions)
        self.pin.computeDistances(self.geometry_model, self.geometry_data)
        self.shape_distances = np.array(
            [r.min_distance for r in self.geometry_data.distanceResults]
        )
        if not np.isfinite(self.shape_distances).all():
            raise ValueError("Coal produced nonfinite clearances")
        return np.minimum.reduceat(self.shape_distances, self.group_starts)

    def _bounded_clearances(self, positions, thresholds):
        """Exact near distances, conservative lower bounds beyond each threshold.

        World AABBs enclose the complete transformed solids. Only pairs whose
        box distance already exceeds the requested clearance can skip Coal.
        Web/export queries use clearances() for exact values for every pair.
        """
        self._place(positions)
        poses = self.geometry_data.oMg
        rotations = np.array([pose.rotation for pose in poses])
        translations = np.array([pose.translation for pose in poses])
        centers = np.einsum("nij,nj->ni", rotations, self._box_centers) + translations
        extents = np.einsum("nij,nj->ni", np.abs(rotations), self._box_extents)
        a, b = self.shape_pairs.T
        distances = np.linalg.norm(
            np.maximum(np.abs(centers[a] - centers[b]) - extents[a] - extents[b], 0),
            axis=1,
        )
        required = np.broadcast_to(thresholds, (len(self.link_pairs),))[
            self._pair_groups
        ]
        for index in np.flatnonzero(distances <= required):
            result = self.pin.computeDistance(
                self.geometry_model, self.geometry_data, int(index)
            )
            distances[index] = result.min_distance
        if not np.isfinite(distances).all():
            raise ValueError("Coal produced nonfinite clearances")
        return np.minimum.reduceat(distances, self.group_starts)

    def nearest_points(self, pair_index):
        """World-space witnesses from the last clearance query for a link pair."""
        start = self.group_starts[pair_index]
        end = (
            self.group_starts[pair_index + 1]
            if pair_index + 1 < len(self.group_starts)
            else len(self.shape_pairs)
        )
        index = int(start + np.argmin(self.shape_distances[start:end]))
        result = self.geometry_data.distanceResults[index]
        points = np.array([result.getNearestPoint1(), result.getNearestPoint2()])
        if not np.isfinite(points).all():
            raise ValueError("Coal produced nonfinite nearest points")
        return points

    def violation(self, positions, margin):
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Collision margin must be nonnegative and finite")
        distances = self._bounded_clearances(
            positions, margin + self.DISTANCE_TOLERANCE_M
        )
        i = int(np.argmin(distances))
        if distances[i] <= margin:
            return ClearanceViolation(self.link_pairs[i], float(distances[i]))
        return None

    def check_path(self, start, target, margin, max_checks=32):
        return self.check_segments([(start, target)], margin, max_checks)

    def check_segments(self, segments, margin, max_checks=32):
        """Certify joint-linear segments using conservative whole-shape bounds.

        Subdivide when midpoint distance cannot certify an interval. Exhausting
        the evaluation budget rejects the path, never passes an unchecked segment.
        This bounds model geometry, not physical braking or drive tracking.
        """
        if not np.isfinite(margin) or margin < 0:
            raise ValueError("Collision margin must be nonnegative and finite")
        if type(max_checks) is not int or max_checks < 1:
            raise ValueError("Collision query budget must be a positive integer")
        pending = [
            (np.asarray(start), np.asarray(target))
            for start, target in reversed(segments)
        ]
        checks = 0
        while pending:
            low, high = pending.pop()
            midpoint = (low + high) * 0.5
            bounds = self.sweep_bounds @ (np.abs(high - low) * 0.5)
            distances = self._bounded_clearances(
                midpoint, margin + bounds + self.DISTANCE_TOLERANCE_M
            )
            checks += 1
            i = int(np.argmin(distances))
            if distances[i] <= margin:
                return ClearanceViolation(self.link_pairs[i], float(distances[i]))
            if np.all(distances - bounds > margin + self.DISTANCE_TOLERANCE_M):
                if pending and checks >= max_checks:
                    return ClearanceViolation(
                        self.link_pairs[i],
                        float(distances[i]),
                        "cannot certify all paths within evaluation budget",
                    )
                continue
            if checks + len(pending) + 2 > max_checks:
                i = int(np.argmin(distances - bounds))
                return ClearanceViolation(
                    self.link_pairs[i],
                    float(distances[i]),
                    "cannot certify path clearance within evaluation budget",
                )
            pending.extend(((midpoint, high), (low, midpoint)))
        return None
