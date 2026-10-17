"""Read-only summaries of Hand 2 frames already consumed by the hand worker.

Peaks cover the worker's sampled frames, not every native device frame.
This helper opens no SDK connection and sends no commands or configuration.
"""
import math


def _finite(value):
    try:
        value = float(value)
    except (ValueError, TypeError):
        return None
    return value if math.isfinite(value) else None


class CurrentSummary:
    def __init__(self, side, limit, kp, kd):
        self.config = dict(side=side, configured_limit_a=limit,
                           configured_kp=kp, configured_kd=kd)
        self.started = None
        self.frames = 0
        self.rows = {}

    def sample(self, entries, joint_index, sent_target, now):
        if self.started is None:
            self.started = now
        self.frames += 1
        for entry in entries:
            idx = joint_index(entry.nid)
            if not 0 <= idx < 20:
                continue
            current = _finite(getattr(entry, "effort", None))
            position = _finite(entry.position)
            target = None if sent_target is None else _finite(sent_target[idx])
            previous = self.rows.get(idx, {})
            peak = previous.get("peak_abs_a")
            if current is not None:
                peak = max(abs(current), peak if peak is not None else 0.)
            self.rows[idx] = dict(
                joint=idx, nid=int(entry.nid), current_a=current, peak_abs_a=peak,
                position_rad=position, sent_target_rad=target,
                sent_error_rad=(None if target is None or position is None
                                else target - position),
                sample_time=now,
            )
        if now - self.started < 1.0:
            return None
        rows = []
        for idx in sorted(self.rows):
            row = self.rows[idx].copy()
            row["age_s"] = now - row.pop("sample_time")
            rows.append(row)
        result = dict(self.config, window_s=now - self.started,
                      sampled_frames=self.frames, joints=rows)
        self.started, self.frames, self.rows = now, 0, {}
        return result
