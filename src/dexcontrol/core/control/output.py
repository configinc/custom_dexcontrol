"""Stage local SDK commands and commit only after a complete batch is checked."""

import threading
from contextlib import contextmanager
from contextvars import ContextVar

import numpy as np

from .fsm import ControlConflict

_PRODUCER = ContextVar("control_producer", default=None)


class OutputGate:
    def __init__(self):
        self.lock = threading.RLock()
        self.owner = None
        self.pending = {}
        self.devices = {}
        self.effects = {}

    def attach(self, name, device):
        if name in self.devices:
            raise ValueError(f"Duplicate output component: {name}")
        self.devices[name] = device
        device.bind_command_sink(lambda command: self.submit(name, command))

    def select(self, owner):
        with self.lock:
            self.owner = owner
            self.pending.clear()
            self.effects.clear()

    @contextmanager
    def producing(self, owner):
        token = _PRODUCER.set(owner)
        try:
            yield
        finally:
            _PRODUCER.reset(token)

    def _admit(self):
        if self.owner is None or _PRODUCER.get() is not self.owner:
            raise ControlConflict("Command producer no longer owns robot output")

    def submit(self, name, command):
        with self.lock:
            self._admit()
            values = {key: np.asarray(value).copy() for key, value in command.items()}
            if not all(np.isfinite(value).all() for value in values.values()):
                raise ValueError(f"Nonfinite command for {name}")
            self.pending[name] = values

    def defer(self, name, operation):
        with self.lock:
            self._admit()
            self.effects[name] = operation

    def clear(self):
        with self.lock:
            self.pending.clear()
            self.effects.clear()

    def commit(self, check):
        with self.lock:
            # Never retain rejected targets or hand commands for a later tick.
            commands, effects = self.pending, self.effects
            self.pending, self.effects = {}, {}
            check(commands, effects)
            for name, command in commands.items():
                self.devices[name].publish_staged_command(command)
            for operation in effects.values():
                operation()
            return commands
