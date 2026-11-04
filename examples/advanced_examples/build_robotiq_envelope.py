"""Build bundled 2F-85 geometry from ROS-Industrial's expanded URDF.

Expand robotiq_arg2f_85_model.xacro with xacro first. Pass its URDF and the
robotiq_2f_85_gripper_visualization package directory. No robot I/O is used.
"""

import argparse
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import trimesh
from scipy.spatial import ConvexHull, HalfspaceIntersection
from scipy.spatial.transform import Rotation


def origin(element):
    result = np.eye(4)
    if element is not None:
        result[:3, 3] = np.fromstring(element.get("xyz", "0 0 0"), sep=" ")
        result[:3, :3] = Rotation.from_euler(
            "xyz", np.fromstring(element.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
    return result


def build(urdf, package, output):
    root = ET.parse(urdf).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {link.get("name"): link for link in root.findall("link")}
    base = next(name for name in links if name not in joints)

    def fk(name, angle):
        if name == base:
            return np.eye(4)
        joint = joints[name]
        pose = origin(joint.find("origin"))
        if joint.get("type") != "fixed":
            mimic = joint.find("mimic")
            q = (
                angle
                if mimic is None
                else angle * float(mimic.get("multiplier", 1))
                + float(mimic.get("offset", 0))
            )
            rotation = np.eye(4)
            axis = np.fromstring(joint.find("axis").get("xyz"), sep=" ")
            rotation[:3, :3] = Rotation.from_rotvec(q * axis).as_matrix()
            pose = pose @ rotation
        return fk(joint.find("parent").get("link"), angle) @ pose

    def mesh(element):
        geometry = element.find("geometry")
        path = geometry.find("mesh")
        if path is not None:
            filename = path.get("filename").replace(
                "package://robotiq_2f_85_gripper_visualization", str(package)
            )
            value = trimesh.load(filename, force="mesh")
            value.apply_scale(np.fromstring(path.get("scale", "1 1 1"), sep=" "))
        else:
            value = trimesh.creation.box(
                np.fromstring(geometry.find("box").get("size"), sep=" ")
            )
        value.apply_transform(origin(element.find("origin")))
        return value

    output.mkdir(parents=True, exist_ok=True)
    visual, hulls, speed_bounds = [], {"base": [], "left": [], "right": []}, {}
    step = 0.001
    angles = np.linspace(0.0, 0.8, 801)
    for name, link in links.items():
        for item in link.findall("visual"):
            value = mesh(item)
            value.apply_transform(fk(name, 0))
            color = item.find("material/color")
            rgba = (
                np.fromstring(color.get("rgba"), sep=" ")
                if color is not None
                else np.array([0.2, 0.2, 0.2, 1])
            )
            value.visual.vertex_colors = (rgba * 255).astype(np.uint8)
            visual.append(value)
        for item in link.findall("collision"):
            value = mesh(item).convex_hull
            group = (
                "base"
                if name == base
                else "left"
                if name.startswith("left")
                else "right"
            )
            # Bound |dp/dq| by summing each moving ancestor's maximum radius.
            # Distance to a joint is bounded by all downstream link lengths.
            radius = np.linalg.norm(value.vertices, axis=1).max()
            velocity_bound = 0.0
            child = name
            while child in joints:
                joint = joints[child]
                mimic = joint.find("mimic")
                multiplier = (
                    float(mimic.get("multiplier", 1)) if mimic is not None else 1.0
                )
                if joint.get("type") != "fixed":
                    velocity_bound += radius * abs(multiplier)
                radius += np.linalg.norm(origin(joint.find("origin"))[:3, 3])
                child = joint.find("parent").get("link")
            speed_bounds[group] = max(speed_bounds.get(group, 0.0), velocity_bound)
            cloud = []
            for angle in angles if group != "base" else [0.0]:
                pose = fk(name, angle)
                cloud.append(value.vertices @ pose[:3, :3].T + pose[:3, 3])
            hulls[group].append(np.concatenate(cloud))
    trimesh.util.concatenate(visual).export(output / "open.glb")
    metadata = {
        "source_commit": "45196f6558fe8ba9d89bc8a105396c68c3e7e892",
        "driver_angle_rad": [0.0, 0.8],
        "step_rad": step,
        "envelope_padding_m": {},
    }
    for group, pieces in hulls.items():
        hull = ConvexHull(np.concatenate(pieces))
        padding = speed_bounds[group] * step / 2 + 1e-7
        # Moving by at most padding from a sample remains inside every shifted
        # support plane. Their intersection conservatively covers the stroke.
        planes = np.unique(np.round(hull.equations, 10), axis=0)
        planes[:, 3] -= padding
        center = hull.points[hull.vertices].mean(axis=0)
        vertices = HalfspaceIntersection(planes, center).intersections
        # Halfspace intersections contain near-duplicate vertices. Quantize
        # before triangulation, then expand by a proven rounding bound so the
        # exported, watertight hull still contains the unrounded polytope.
        result = trimesh.convex.convex_hull(np.unique(np.round(vertices, 6), axis=0))
        center = result.vertices.mean(axis=0)
        planes = ConvexHull(result.vertices).equations
        inradius = np.min(-(planes[:, :3] @ center + planes[:, 3]))
        result.vertices = center + (result.vertices - center) * (1 + 1e-6 / inradius)
        if not result.is_watertight:
            raise ValueError(f"Non-watertight {group} envelope")
        result.export(output / f"{group}_envelope.obj")
        metadata["envelope_padding_m"][group] = float(padding)
    (output / "geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("urdf", type=Path)
    parser.add_argument("package", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    build(args.urdf, args.package, args.output)
