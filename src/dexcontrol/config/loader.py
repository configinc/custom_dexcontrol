"""Control parameter loader for custom_dexcontrol."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import yaml

_logger = logging.getLogger("robotenv_vega")
_DEFAULT_PATH = Path(__file__).parent / "control_params.yaml"


def load_control_params(path: str | Path | None = None) -> dict:
    """Load control_params.yaml. Returns {} if file not found."""
    p = Path(path) if path else _DEFAULT_PATH
    if not p.exists():
        _logger.warning("control_params.yaml not found at %s; using hardcoded defaults", p)
        return {}
    with p.open() as f:
        return yaml.safe_load(f) or {}


def merge_args(args_ns: argparse.Namespace, cfg: dict) -> dict:
    """Merge argparse namespace (None = unset) with yaml config.

    Priority: CLI explicit value > yaml value > (caller supplies hardcoded default).
    Returns a flat dict of resolved parameter values.
    """
    def _cli(attr, yaml_val):
        v = getattr(args_ns, attr, None)
        return v if v is not None else yaml_val

    teleop = cfg.get("teleop") or {}
    mf = cfg.get("motion_filter") or {}
    ik = cfg.get("ik_damping") or {}

    return {
        # init/middle joints — kept as dict; caller selects arm_side key
        "init_joints":   cfg.get("init_joints"),
        "middle_joints": cfg.get("middle_joints"),

        # head
        "head_init_pos": _cli("head_init_pos", cfg.get("head_init_pos")),

        # teleop gains
        "teleop_pos_gain": teleop.get("pos_action_gain", 5.0),
        "teleop_rot_gain": teleop.get("rot_action_gain", 2.0),

        # motion filter
        "ema_alpha":           _cli("ema_alpha",           mf.get("ema_alpha", 0.0)),
        "filter_type":         _cli("filter_type",         mf.get("filter_type", "none")),
        "filter_cutoff_freq":  _cli("filter_cutoff_freq",  mf.get("filter_cutoff_freq", 10.0)),
        "filter_order":        _cli("filter_order",        mf.get("filter_order", 2)),
        "filter_ema_alpha":    _cli("filter_ema_alpha",    mf.get("filter_ema_alpha", 0.1)),
        "vel_smoothing_alpha": _cli("vel_smoothing_alpha", mf.get("vel_smoothing_alpha", 0.3)),
        "hw_correction_alpha": _cli("hw_correction_alpha", mf.get("hw_correction_alpha", 0.7)),
        "max_delta_scale":     _cli("max_delta_scale",     mf.get("max_delta_scale", 1.0)),
        "max_accel_delta":     _cli("max_accel_delta",     mf.get("max_accel_delta", 0.25)),
        "vel_ratio":           _cli("vel_ratio",           mf.get("vel_ratio", 1.0)),
        "vel_damp_thresh":     _cli("vel_damp_thresh",     mf.get("vel_damp_thresh", 0.05)),

        # IK damping
        "ik_damping_default":  ik.get("default", 1e-3),
        "ik_damping_override": ik.get("override", {}),

        # motor max delta
        "motor_max_delta_rad": cfg.get("motor_max_delta_rad"),
    }
