"""Explicit violation policy. Checks and robot fault actions belong to callers."""

import logging
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class SafetyRule:
    enabled: bool
    on_violation: str
    fault_action: str | None = None
    log_interval_s: float = 1.0

    def __post_init__(self):
        if type(self.enabled) is not bool or self.on_violation not in (
            "warn",
            "block",
            "fault",
        ):
            raise ValueError(
                "Safety rules require enabled and on_violation: warn, block or fault"
            )
        if (self.on_violation == "fault") != bool(self.fault_action):
            raise ValueError("Only fault rules must specify a fault_action")
        if not math.isfinite(self.log_interval_s) or self.log_interval_s < 0:
            raise ValueError("Safety log interval must be finite and nonnegative")


class Safety:
    def __init__(self, rules):
        self.rules = dict(rules)
        self.last_log = {}

    def report(self, name, message, now):
        rule = self.rules[name]
        if not rule.enabled:
            return None
        if rule.on_violation in ("warn", "block"):
            if now - self.last_log.get(name, -math.inf) >= rule.log_interval_s:
                logging.getLogger(__name__).warning("%s: %s", name, message)
                self.last_log[name] = now
        return rule.on_violation
