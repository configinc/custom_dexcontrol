"""Vega tool geometry and fixed support points, shared by runtime and Pose Lab."""

import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

ASSETS = Path(__file__).with_name("assets") / "robotiq_2f85"


def attach_robotiq(urdf_path, srdf_path, sides, settings):
    """Add fixed full-stroke collision envelopes and open-gripper visuals.

    Geometry covers the entire nominal parallel-jaw stroke, so runtime safety
    needs neither serial gripper reads nor a guessed instantaneous jaw position.
    Mutual gripper collisions remain in the model; runtime policy selects pairs.
    """
    if not sides or len(set(sides)) != len(sides) or set(sides) - {"left", "right"}:
        raise ValueError("Expected unique left/right Robotiq attachment sides")
    root, srdf = ET.parse(urdf_path).getroot(), ET.parse(srdf_path).getroot()
    links = {link.get("name") for link in root.findall("link")}
    for mesh in root.findall(".//mesh"):
        filename = Path(mesh.get("filename"))
        if not filename.is_absolute():
            filename = Path(urdf_path).parent / filename
        if not filename.is_file():
            raise ValueError(f"Cannot resolve robot mesh: {filename}")
        mesh.set("filename", str(filename.resolve()))
    for side in sides:
        prefix = "L" if side == "left" else "R"
        parent, name = f"{prefix}_arm_l8", f"{prefix}_robotiq"
        if parent not in links or name in links:
            raise ValueError(f"Cannot attach {name} to {parent}")
        xyz = np.asarray(settings["robotiq_mount_xyz_m"])
        yaw = np.deg2rad(settings["robotiq_mount_yaw_deg"][side])
        rotation = Rotation.from_euler("z", yaw).as_matrix()
        com = rotation.T @ (np.asarray(settings["robotiq_com_sensor_m"]) - xyz)
        link = ET.SubElement(root, "link", name=name)
        inertial = ET.SubElement(link, "inertial")
        ET.SubElement(inertial, "origin", xyz=" ".join(map(str, com)))
        ET.SubElement(inertial, "mass", value=str(settings["robotiq_mass_kg"]))
        # Only the first moment is used. This model does not estimate tool
        # rotational dynamics; zero inertia is an explicit point-mass model.
        ET.SubElement(
            inertial, "inertia", ixx="0", ixy="0", ixz="0", iyy="0", iyz="0", izz="0"
        )
        visual = ET.SubElement(link, "visual")
        ET.SubElement(
            ET.SubElement(visual, "geometry"), "mesh", filename=str(ASSETS / "open.glb")
        )
        for part in ("base", "left", "right"):
            collision = ET.SubElement(link, "collision")
            ET.SubElement(
                ET.SubElement(collision, "geometry"),
                "mesh",
                filename=str(ASSETS / f"{part}_envelope.obj"),
            )
        joint = ET.SubElement(root, "joint", name=f"{name}_mount", type="fixed")
        ET.SubElement(joint, "origin", xyz=" ".join(map(str, xyz)), rpy=f"0 0 {yaw}")
        ET.SubElement(joint, "parent", link=parent)
        ET.SubElement(joint, "child", link=name)
    directory = tempfile.TemporaryDirectory(prefix="vega-model-")
    path = Path(directory.name) / "vega_robotiq.urdf"
    ET.indent(root)
    ET.indent(srdf)
    ET.ElementTree(root).write(path, encoding="utf-8", xml_declaration=True)
    ET.ElementTree(srdf).write(
        path.with_suffix(".srdf"), encoding="utf-8", xml_declaration=True
    )
    return directory, path, path.with_suffix(".srdf")
