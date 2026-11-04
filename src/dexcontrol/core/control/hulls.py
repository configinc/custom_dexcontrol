"""Conservative convex hull simplification performed only at model startup."""

import itertools

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection


def verify_outer_hull(original, outer, error_m):
    """Check containment and a vertex-based Hausdorff bound for convex solids.

    Containment follows from every original vertex satisfying every outer face
    halfspace. Distance to a convex set is convex, so bounding the distance of
    every outer vertex bounds the entire outer solid, not just sampled surfaces.
    All comparisons use meters with a 1 nm containment tolerance.
    """
    if not np.isfinite(error_m) or error_m <= 0:
        raise ValueError("Outer error must be positive and finite")
    for mesh in (original, outer):
        if not mesh.is_volume or not mesh.is_convex:
            raise ValueError("Expected closed, consistently oriented convex solids")
    normals = outer.face_normals
    offsets = np.einsum("ij,ij->i", normals, outer.triangles[:, 0])
    outside = max(
        float(np.max(chunk @ normals.T - offsets))
        for chunk in np.array_split(
            original.vertices, max(1, len(original.vertices) // 256)
        )
    )
    if outside > 1e-9:
        raise ValueError(f"Simplification shrinks the original by {outside:.9g} m")
    _, distances, _ = original.nearest.on_surface(outer.vertices)
    error = float(np.max(distances))
    if not np.isfinite(error) or error > error_m:
        raise ValueError(f"Outer error {error:.9g} m exceeds {error_m:.9g} m")
    return {
        "original_vertices": len(original.vertices),
        "outer_vertices": len(outer.vertices),
        "max_outer_error_m": error,
        "max_containment_violation_m": max(0.0, outside),
    }


def simplify_outer_hull(mesh, error_m):
    """Construct a conservative approximation within the requested error.

    Support-plane offsets are maxima over ALL original vertices. Refinement
    adds directions toward offending outer vertices, rather than deleting mesh
    vertices. A small outward numerical padding survives OBJ/Coal round trips.
    The final geometry is independently checked before it can be exported.
    """
    if not np.isfinite(error_m) or error_m <= 0:
        raise ValueError("Outer error must be positive and finite")
    center = mesh.vertices.mean(axis=0)
    original = trimesh.Trimesh(mesh.vertices - center, mesh.faces, process=False)
    directions = np.array(
        [v for v in itertools.product((-1.0, 0.0, 1.0), repeat=3) if any(v)]
    )
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    for _ in range(32):
        offsets = np.max(original.vertices @ directions.T, axis=0) + 1e-9
        points = HalfspaceIntersection(
            np.column_stack((directions, -offsets)), np.zeros(3)
        ).intersections
        nearest, distances, _ = original.nearest.on_surface(points)
        refine = distances > error_m * 0.95
        if not np.any(refine):
            # Snap numerical slivers to a 1 um grid before triangulating. This
            # may move vertices inward; the support-based rescaling below must
            # restore containment, and final verification bounds the total error.
            hull = ConvexHull(np.unique(np.round(points, 6), axis=0))
            faces = hull.simplices.copy()
            triangles = hull.points[faces]
            crosses = np.cross(
                triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
            )
            inward = np.einsum("ij,ij->i", crosses, hull.equations[:, :3]) < 0
            faces[inward] = faces[inward, ::-1]
            outer = trimesh.Trimesh(hull.points, faces, process=False)
            outer.remove_unreferenced_vertices()
            normals = outer.face_normals
            support = np.einsum("ij,ij->i", normals, outer.triangles[:, 0])
            if np.any(support <= 0):
                raise ValueError("Hull origin must be strictly inside every face")
            required = np.max(original.vertices @ normals.T, axis=0)
            scale = max(1.0, float(np.max(required / support)))
            scale += 1e-7 / float(np.min(support))
            outer.vertices *= scale
            outer.vertices += center
            return outer, verify_outer_hull(mesh, outer, error_m)
        normals = points[refine] - nearest[refine]
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        directions = np.unique(np.round(np.vstack((directions, normals)), 12), axis=0)
    raise RuntimeError("Outer hull did not meet the error bound within 32 refinements")
