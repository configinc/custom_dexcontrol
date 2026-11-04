"""Offline outer-hull construction and verification; no robot I/O.

Intersect supporting halfspaces, then refine vertices that are too far from the
original convex solid. The result contains the original, with a bounded outward
Euclidean error. This is not ordinary mesh decimation, which can shrink solids.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import trimesh

from dexcontrol.core.control.hulls import simplify_outer_hull, verify_outer_hull


def export_collision_urdf(model, directory, error_m):
    """Write per-component OBJ solids and a kinematically unchanged URDF.

    This experiment accepts mesh-only models. It does not approximate native
    analytic shapes by their display meshes. Production models/config are never
    edited. The caller must reload and revalidate the resulting URDF.
    """
    import coal

    if any(
        not isinstance(obj.geometry, coal.ConvexBase)
        for obj in model.geometry_model.geometryObjects
    ):
        raise ValueError("This offline experiment requires convex mesh geometry")
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=False)
    root = ET.parse(model.urdf_path).getroot()
    # Keep vendor visual and excluded-link assets resolvable from the new URDF.
    for element in root.findall(".//mesh"):
        filename = element.attrib["filename"]
        if "://" in filename:
            raise ValueError("This exporter requires filesystem mesh paths")
        element.set("filename", str((model.urdf_path.parent / filename).resolve()))
    links = {link.attrib["name"]: link for link in root.findall("link")}
    for name in model.covered_links:
        for collision in links[name].findall("collision"):
            links[name].remove(collision)
    statistics = []
    for i, (obj, mesh, name) in enumerate(
        zip(model.geometry_model.geometryObjects, model.shape_meshes, model.shape_links)
    ):
        outer, info = simplify_outer_hull(mesh, error_m)
        mesh_path = directory / f"shape_{i:02d}.obj"
        mesh_path.write_text(trimesh.exchange.obj.export_obj(outer, digits=17))
        collision = ET.SubElement(links[name], "collision", name=f"outer_{i}")
        # Geometry placement is relative to the parent joint; URDF origins are
        # relative to the link frame, including any intervening fixed joints.
        frame = model.model.frames[obj.parentFrame]
        placement = frame.placement.inverse() * obj.placement
        ET.SubElement(
            collision,
            "origin",
            xyz=" ".join(map(str, placement.translation)),
            rpy=" ".join(map(str, model.pin.rpy.matrixToRpy(placement.rotation))),
        )
        geometry = ET.SubElement(collision, "geometry")
        ET.SubElement(geometry, "mesh", filename=str(mesh_path))
        statistics.append({"link": name, "shape": obj.name, **info})
    ET.indent(root)
    path = directory / "vega_collision.urdf"
    ET.ElementTree(root).write(path, encoding="unicode", xml_declaration=True)
    return path, statistics
