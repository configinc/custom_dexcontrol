"""Interactive offline Vega pose editor with the managed server's collision model.

No Robot, SDK transport, motor commands, or live robot feedback is used.
"""

import argparse
import json
import threading
from pathlib import Path
from time import perf_counter

import numpy as np
import viser
import viser.transforms as vtf
from dexbot_utils.configs import get_robot_config
from PIL import Image, ImageDraw
from viser.extras import ViserUrdf

from dexcontrol.core.control.collision import CoalCollisionModel
from dexcontrol.core.control.config import load_config
from dexcontrol.core.control.stability import StaticStability

GROUPS = {
    "Torso": ("torso", [f"torso_j{i}" for i in range(1, 4)]),
    "Left arm": ("left_arm", [f"L_arm_j{i}" for i in range(1, 8)]),
    "Right arm": ("right_arm", [f"R_arm_j{i}" for i in range(1, 8)]),
    "Head": ("head", [f"head_j{i}" for i in range(1, 4)]),
}
PRESETS = {
    "Folded": ("folded", "folded", "tucked"),
    "Folded arms / upright torso": ("home", "folded", "home"),
    "Folded with closed-hand wrist angles": ("folded", "folded_closed_hand", "tucked"),
    "Initial arms / upright torso": ("home", "L_shape", "home"),
    "Initial arms / raised torso": ("crouch45_high", "L_shape", "home"),
    "Lifted arms / upright torso": ("home", "lift_up", "home"),
}


class PoseModel:
    """Model-only pose operations, with no viewer or robot connection."""

    def __init__(self, robot_config, config):
        collision_config = config.collision
        self.robot_config = robot_config
        self.poses = config.poses
        self.urdf_path = Path(robot_config.urdf_path)
        self.model_config = config.model
        self.collision = CoalCollisionModel.for_robot(
            self.urdf_path,
            collision_config.collision_urdf,
            collision_config.srdf,
            collision_config.excluded_links,
            collision_config.max_outer_error_m,
            robotiq_sides=("left", "right"),
            model_config=self.model_config,
        )
        self.urdf_path = self.collision.urdf_path
        self.stability = StaticStability(self.urdf_path, self.model_config)
        self.stability_config = config.stability
        self.stability_enabled = config.safety["static_stability"].enabled
        expected = {name for _, names in GROUPS.values() for name in names}
        if set(self.collision.joint_names) != expected:
            raise ValueError("Pose Lab expects the 20 Vega arm, torso and head joints")
        self.joint_names = self.collision.joint_names
        self.lower = self.collision.model.lowerPositionLimit[self.collision.q_indices]
        self.upper = self.collision.model.upperPositionLimit[self.collision.q_indices]
        self.indices = {name: i for i, name in enumerate(self.joint_names)}

    def preset(self, name):
        torso, arms, head = PRESETS[name]
        named = {}
        for component, joints in GROUPS.values():
            pose_name = (
                torso if component == "torso" else head if component == "head" else arms
            )
            pose = (
                self.poses["folded"][component]
                if pose_name == "folded"
                else self.robot_config.components[component].pose_pool[pose_name]
            )
            if pose_name == "folded":
                indices = [self.indices[joint] for joint in joints]
                if np.any(np.asarray(pose) < self.lower[indices]) or np.any(
                    np.asarray(pose) > self.upper[indices]
                ):
                    raise ValueError(
                        f"Configured folded {component} exceeds URDF limits"
                    )
            named.update(zip(joints, pose))
        requested = self.collision.positions(named)
        applied = np.clip(requested, self.lower, self.upper)
        changes = [
            f"{joint}: {requested[i]:.5f} → {applied[i]:.5f} rad"
            for i, joint in enumerate(self.joint_names)
            if requested[i] != applied[i]
        ]
        return applied, changes

    def evaluate(self, q, margin_m):
        q = np.asarray(q, dtype=float)
        if q.shape != self.lower.shape or not np.isfinite(q).all():
            raise ValueError("Expected 20 finite joint angles")
        if np.any(q < self.lower) or np.any(q > self.upper):
            raise ValueError("Joint angle exceeds URDF limits")
        if not np.isfinite(margin_m) or margin_m < 0:
            raise ValueError("Collision margin must be nonnegative and finite")
        started = perf_counter()
        distances = self.collision.clearances(q)
        elapsed_ms = (perf_counter() - started) * 1000
        pairs = self.collision.link_pairs
        order = np.argsort(distances)
        violation_count = int(np.count_nonzero(distances <= margin_m))
        overlap_count = int(np.count_nonzero(distances < 0))
        stability = self.stability.evaluate(dict(zip(self.joint_names, q)))
        stability.update(
            safety_enabled=self.stability_enabled,
            required_margin_m=self.stability_config.margin_m,
            approach_m=self.stability_config.approach_m,
            release_m=self.stability_config.release_m,
        )
        return {
            "stability": stability,
            "backend": "Coal",
            "geometry": "connected-component outer convex hulls",
            "max_outer_error_m": self.collision.max_outer_error_m,
            "status": "OVERLAP"
            if overlap_count
            else "BELOW MARGIN"
            if violation_count
            else "CLEAR",
            "min_clearance_m": float(distances[order[0]]),
            "margin_m": float(margin_m),
            "violation_count": violation_count,
            "overlap_count": overlap_count,
            "check_ms": elapsed_ms,
            "nearest_points_m": self.collision.nearest_points(int(order[0])).tolist(),
            "pairs": [
                {
                    "links": list(pairs[i]),
                    "clearance_m": float(distances[i]),
                    "violates_margin": bool(distances[i] <= margin_m),
                }
                for i in order
            ],
        }

    def world_bounds(self):
        points = []
        for mesh, pose in zip(
            self.collision.shape_meshes, self.collision.geometry_data.oMg
        ):
            corners = np.array(np.meshgrid(*mesh.bounds.T)).reshape(3, -1).T
            points.extend(corners @ pose.rotation.T + pose.translation)
        return np.min(points, axis=0), np.max(points, axis=0)

    def export(self, q, result):
        named = dict(zip(self.joint_names, map(float, q)))
        return {
            "schema": "vega-pose-lab/v1",
            "units": "rad",
            "simulation_only": True,
            "robot_model": self.urdf_path.name,
            "joint_positions": named,
            "components": {
                component: [named[joint] for joint in names]
                for component, names in GROUPS.values()
            },
            "self_collision": result,
            "coverage": "Coal outer hulls + Robotiq 2F-85 full-stroke envelopes. Mutual gripper collisions included. Wheel links excluded; adapters, cables, payloads not checked.",
            "model_config": self.model_config,
            "excluded_links": self.collision.excluded_links,
            "trajectory_checked": False,
        }


class PoseLab:
    def __init__(self, server, model, margin_m, initial_pose=None):
        self.server, self.model = server, model
        self.lock = threading.RLock()
        self.q, changes = model.preset("Folded")
        if initial_pose is not None:
            self.q = model.collision.positions(initial_pose)
            model.evaluate(self.q, margin_m)
            changes = []
        self.sliders = {}
        self.result = None
        server.scene.set_up_direction("+z")
        server.gui.configure_theme(
            control_width="large",
            show_logo=False,
            show_share_button=False,
            brand_color=(35, 128, 115),
        )
        server.gui.main_panel.dock_right()
        server.scene.add_grid(
            "/floor",
            width=4,
            height=4,
            cell_size=0.1,
            section_size=0.5,
            plane="xy",
            position=(0.0, 0.0, model.stability.ground_z),
        )
        server.scene.add_frame("/base_axes", axes_length=0.18, axes_radius=0.004)
        self.visual = ViserUrdf(server, model.urdf_path, root_node_name="/robot")
        self.collision_shapes = [
            server.scene.add_mesh_simple(
                f"/collision/shape_{i}",
                vertices=mesh.vertices,
                faces=mesh.faces,
                color=(220, 55, 70),
                opacity=0.35,
                wireframe=False,
                cast_shadow=False,
                receive_shadow=False,
                visible=False,
            )
            for i, mesh in enumerate(model.collision.shape_meshes)
        ]
        self.clearance_line = server.scene.add_line_segments(
            "/nearest_pair",
            np.zeros((1, 2, 3), dtype=np.float32),
            colors=(235, 80, 70),
            thickness=3,
            thickness_units="screen",
        )
        support = model.stability.support
        support3 = np.column_stack((support, np.full(3, model.stability.ground_z)))
        server.scene.add_mesh_simple(
            "/stability/support",
            support3,
            np.array([[0, 1, 2]]),
            color=(60, 170, 230),
            opacity=0.25,
            side="double",
        )
        server.scene.add_line_segments(
            "/stability/edges",
            np.stack((support3, np.roll(support3, -1, axis=0)), axis=1).astype(
                np.float32
            ),
            colors=(60, 170, 230),
            thickness=3,
            thickness_units="screen",
        )
        self.com_marker = server.scene.add_icosphere(
            "/stability/com", radius=0.018, color=(255, 190, 40)
        )
        self.com_projection = server.scene.add_icosphere(
            "/stability/projection", radius=0.012, color=(50, 220, 130)
        )
        self.com_line = server.scene.add_line_segments(
            "/stability/com_line",
            np.zeros((1, 2, 3), dtype=np.float32),
            colors=(255, 190, 40),
            thickness=2,
            thickness_units="screen",
        )
        self.com_label = server.scene.add_label("/stability/label", "Center of mass")
        self.pair_label = server.scene.add_label(
            "/nearest_label", "", font_screen_scale=0.8
        )
        server.gui.add_markdown(
            "# Vega Pose Lab\n**Offline pose editor** · shared browser session\n\n"
            "Drag to orbit · scroll to zoom · right-drag to pan.\n\n"
            "The pose shown here is a simulation, not the live robot posture."
        )
        self.status = server.gui.add_markdown("Loading model…")
        self.stability_status = server.gui.add_markdown("")
        self.support_image = server.gui.add_image(
            np.full((160, 300, 3), 250, dtype=np.uint8),
            label="Fixed support · top view",
        )
        self.preset = server.gui.add_dropdown(
            "Pose preset", list(PRESETS), initial_value="Folded"
        )
        self.preset.on_update(self._select_preset)
        server.gui.add_button("Reload selected preset").on_click(self._select_preset)
        self.notice = server.gui.add_markdown(self._preset_notice(changes))
        self.margin = server.gui.add_number(
            "Collision margin [mm]",
            initial_value=margin_m * 1000,
            min=0.0,
            max=100.0,
            step=1.0,
        )
        self.margin.on_update(self._changed)
        self.overlay = server.gui.add_dropdown(
            "Collision geometry", ("Violations only", "All", "Hidden")
        )
        self.overlay.on_update(self._changed)
        self.show_visual = server.gui.add_checkbox(
            "Show robot meshes", initial_value=True
        )
        self.show_visual.on_update(self._changed)
        self.show_pair = server.gui.add_checkbox(
            "Show closest pair", initial_value=True
        )
        self.show_pair.on_update(self._changed)
        server.gui.add_button("Reset camera").on_click(
            lambda event: self.reset_camera(event.client)
        )
        with server.gui.add_folder("Closest link pairs", expand_by_default=False):
            self.pair_table = server.gui.add_markdown("")
        tabs = server.gui.add_tab_group()
        for label, (_, names) in GROUPS.items():
            with tabs.add_tab(label):
                server.gui.add_markdown(
                    "Joint angles in **radians**. Type a value or drag a slider."
                )
                for name in names:
                    i = model.indices[name]
                    slider = server.gui.add_slider(
                        name,
                        min=float(model.lower[i]),
                        max=float(model.upper[i]),
                        step=0.001,
                        initial_value=float(self.q[i]),
                        hint=f"URDF limits: {model.lower[i]:.4f} to {model.upper[i]:.4f} rad",
                    )
                    self.sliders[name] = slider
                    slider.on_update(self._changed)
        with server.gui.add_folder("Save / restore pose", expand_by_default=False):
            server.gui.add_markdown(
                "Download includes all 20 joint angles and all checked pair distances. "
                "Copy the joint JSON below to reuse a pose; editing the text does not move the model until Apply."
            )
            server.gui.add_button("Download pose JSON").on_click(self._download)
            self.pose_text = server.gui.add_text(
                "Joint positions JSON [rad]", initial_value="{}", multiline=True
            )
            server.gui.add_button("Apply joint JSON").on_click(self._apply_json)
            self.import_status = server.gui.add_markdown("")
        with server.gui.add_folder("Model coverage", expand_by_default=False):
            server.gui.add_markdown(
                f"**{model.urdf_path.stem}** · {len(model.joint_names)} joints · "
                f"{len(model.collision.shape_meshes)} solid shapes · {len(model.collision.link_pairs)} checked link pairs\n\n"
                "Uses Coal and the same collision geometry and SRDF exclusions as managed body control. "
                "Each connected mesh component uses an outer hull with at most 1 mm simplification error. Red shapes overlap; "
                "orange shapes are within the selected margin.\n\n"
                "**Robotiq 2F-85:** open-jaw visual; collision envelopes cover the full nominal parallel stroke. "
                "Left +90° / right −90° about sensor Z, 18.5 mm mounting offset. "
                "Mutual gripper and gripper-to-arm/body collisions are included. "
                "**Not modelled:** adapter geometry, cables, payloads and adaptive fingertip wrapping. "
                "Wheel links are explicitly excluded from collision checking. "
                "This checks the displayed posture only, not the path to it or physical stopping distance."
            )
        self.render()
        server.on_client_connect(self.reset_camera)

    @staticmethod
    def _preset_notice(changes):
        return (
            "Preset adjusted to URDF limits: " + "; ".join(changes)
            if changes
            else "Preset loaded from the Vega configuration."
        )

    def reset_camera(self, client):
        if client is not None:
            with self.lock:
                lower, upper = self.model.world_bounds()
                center = (lower + upper) / 2
                extent = max(float(np.max(upper - lower)), 0.5)
                direction = np.array([1.0, 0.9, 0.55])
                direction /= np.linalg.norm(direction)
                client.camera.position = center + direction * extent * 2.0
                client.camera.look_at = center
                client.camera.up_direction = (0.0, 0.0, 1.0)

    def _select_preset(self, event):
        if event.client is None:
            return
        with self.lock, self.server.atomic():
            self.q, changes = self.model.preset(self.preset.value)
            for name, slider in self.sliders.items():
                slider.value = float(self.q[self.model.indices[name]])
            self.notice.content = self._preset_notice(changes)
            self.render()
            self.reset_camera(event.client)

    def _changed(self, event):
        if event.client is None:
            return
        with self.lock, self.server.atomic():
            self.q = np.array(
                [self.sliders[name].value for name in self.model.joint_names]
            )
            if event.target in self.sliders.values():
                self.notice.content = "Custom pose · adjusted with joint sliders."
            self.render()

    def render(self):
        result = self.model.evaluate(self.q, self.margin.value / 1000)
        self.result = result
        named = dict(zip(self.model.joint_names, map(float, self.q)))
        self.visual.update_cfg(named)
        self.visual.show_visual = self.show_visual.value
        nearest = result["pairs"][0]
        self.status.content = (
            f"## {result['status']}\n"
            f"**Minimum clearance: {result['min_clearance_m'] * 1000:+.2f} mm** "
            f"/ margin {result['margin_m'] * 1000:.1f} mm\n\n"
            f"{result['overlap_count']} overlapping pairs · {result['violation_count']} pairs below margin\n\n"
            f"Closest: `{nearest['links'][0]}` ↔ `{nearest['links'][1]}`\n\n"
            f"Coal + FK: {result['check_ms']:.2f} ms (server computation)"
        )
        stability = result["stability"]
        inside = stability["inside_support"]
        margin_ok = stability["margin_m"] >= stability["required_margin_m"]
        self.com_marker.position = tuple(stability["com_m"])
        self.com_projection.position = tuple(stability["projection_m"])
        self.com_projection.color = (
            (50, 220, 130) if margin_ok else (240, 153, 40) if inside else (235, 60, 60)
        )
        self.com_line.points = np.array(
            [[stability["com_m"], stability["projection_m"]]], dtype=np.float32
        )
        self.com_label.position = tuple(stability["com_m"])
        self.com_label.text = (
            f"COM · support margin {stability['margin_m'] * 1000:+.1f} mm"
        )
        self.stability_status.content = (
            f"### {'INSIDE' if inside else 'OUTSIDE'} SUPPORT TRIANGLE\n"
            f"**Static tipping margin: {stability['margin_m'] * 1000:+.1f} mm**\n\n"
            f"Static protection: {'enabled' if stability['safety_enabled'] else 'disabled'} · "
            f"required {stability['required_margin_m'] * 1000:.0f} mm · "
            f"{'margin satisfied' if margin_ok else 'below safety margin'}\n\n"
            f"Total model mass: {stability['mass_kg']:.2f} kg · COM [m]: "
            f"`{stability['com_m'][0]:.3f}, {stability['com_m'][1]:.3f}, {stability['com_m'][2]:.3f}`\n\n"
            "Blue: fixed support · yellow: COM · projection: green above margin, orange below margin, red outside.\n\n"
            "Level ground, no payload. Steering/caster motion and dynamic tipping are not included."
        )
        self.support_image.image = self.support_diagram(stability)
        rows = ["| Link pair | Gap [mm] |", "| :-- | --: |"]
        for pair in result["pairs"][:10]:
            label = " ↔ ".join(pair["links"])
            value = f"{pair['clearance_m'] * 1000:+.2f}"
            rows.append(
                f"| **{label}** | **{value}** |"
                if pair["violates_margin"]
                else f"| {label} | {value} |"
            )
        self.pair_table.content = "\n".join(rows)
        self.pose_text.value = json.dumps(named, indent=2)
        model = self.model.collision
        gaps = model.shape_distances
        bad = model.shape_pairs[gaps <= result["margin_m"]]
        overlap = set(model.shape_pairs[gaps < 0].ravel().tolist())
        affected = set(bad.ravel().tolist())
        for i, shape in enumerate(self.collision_shapes):
            pose = model.geometry_data.oMg[i]
            shape.position = pose.translation
            shape.wxyz = vtf.SO3.from_matrix(pose.rotation).wxyz
            shape.color = (
                (220, 55, 70)
                if i in overlap
                else (240, 153, 40)
                if i in affected
                else (45, 161, 151)
            )
            shape.opacity = 0.4 if i in affected else 0.12
            shape.visible = self.overlay.value == "All" or (
                self.overlay.value == "Violations only" and i in affected
            )
        points = np.array(result["nearest_points_m"])
        self.clearance_line.points = points[None].astype(np.float32)
        self.clearance_line.visible = self.show_pair.value
        self.pair_label.position = points.mean(axis=0)
        self.pair_label.text = f"{nearest['links'][0]} / {nearest['links'][1]}\n{gaps.min() * 1000:+.2f} mm"
        self.pair_label.visible = self.show_pair.value

    @staticmethod
    def support_diagram(stability):
        points = np.array(stability["support_xy_m"])
        com = np.array(stability["com_m"][:2])
        all_points = np.vstack((points, com))
        center = (all_points.max(axis=0) + all_points.min(axis=0)) / 2
        scale = min(
            240 / max(0.1, np.ptp(all_points[:, 0])),
            120 / max(0.1, np.ptp(all_points[:, 1])),
        )

        def pixel(point):
            xy = (point - center) * scale * [1, -1] + [150, 80]
            return tuple(map(float, xy))

        image = Image.new("RGB", (300, 160), (248, 250, 252))
        draw = ImageDraw.Draw(image)
        draw.polygon(
            [pixel(p) for p in points],
            fill=(222, 240, 249),
            outline=(45, 135, 200),
            width=2,
        )
        for point in points:
            x, y = pixel(point)
            draw.ellipse((x - 3, y - 3, x + 3, y + 3), fill=(45, 135, 200))
        x, y = pixel(com)
        color = (
            (35, 170, 100)
            if stability["margin_m"] >= stability["required_margin_m"]
            else (240, 153, 40)
            if stability["inside_support"]
            else (225, 55, 65)
        )
        draw.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
        draw.text((x + 7, y - 5), "COM", fill=(40, 50, 65))
        draw.text((8, 8), "+Y up / +X right", fill=(65, 80, 95))
        return np.asarray(image)

    def _download(self, event):
        with self.lock:
            content = json.dumps(
                self.model.export(self.q, self.result), indent=2
            ).encode()
        event.client.send_file_download(
            "vega-pose.json", content, save_immediately=True
        )

    def _apply_json(self, event):
        with self.lock:
            try:
                named = json.loads(self.pose_text.value)
                if not isinstance(named, dict) or set(named) != set(
                    self.model.joint_names
                ):
                    raise ValueError("JSON must contain exactly the 20 named joints")
                q = self.model.collision.positions(named)
                self.model.evaluate(q, self.margin.value / 1000)
            except (ValueError, TypeError) as exc:
                self.import_status.content = f"**Not applied:** {exc}"
                return
            with self.server.atomic():
                self.q = q
                for name, slider in self.sliders.items():
                    slider.value = float(q[self.model.indices[name]])
                self.notice.content = "Custom pose · loaded from joint JSON."
                self.import_status.content = "Joint JSON applied."
                self.render()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument(
        "--control-config",
        type=Path,
        help="Shared Vega control and model configuration",
    )
    parser.add_argument(
        "--margin-mm",
        type=float,
        help="Override the display margin; defaults to collision.margin_m in the control config",
    )
    parser.add_argument(
        "--pose-json",
        type=Path,
        help="Restore exported pose or named joint JSON at startup",
    )
    args = parser.parse_args()
    config = load_config(args.control_config)
    margin_mm = (
        args.margin_mm
        if args.margin_mm is not None
        else config.collision.margin_m * 1000
    )
    if not np.isfinite(margin_mm) or not 0 < margin_mm <= 100:
        parser.error("Display margin must be greater than zero and at most 100 mm")
    model = PoseModel(get_robot_config("vega_1"), config)
    server = viser.ViserServer(host=args.host, port=args.port, label="Vega Pose Lab")
    try:
        if server.get_port() != args.port:
            raise RuntimeError(f"Requested port {args.port} is already in use")
        initial_pose = None
        if args.pose_json:
            saved = json.loads(args.pose_json.read_text())
            initial_pose = saved.get("joint_positions", saved)
        PoseLab(server, model, margin_mm / 1000, initial_pose)
        print(
            f"POSE_LAB_READY http://{args.host}:{server.get_port()} (offline only)",
            flush=True,
        )
        threading.Event().wait()
    except KeyboardInterrupt:
        pass
    finally:
        server.stop()


if __name__ == "__main__":
    main()
