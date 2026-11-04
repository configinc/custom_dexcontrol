"""Explicit lifecycle transitions; no SDK calls or controller calculations."""

from dataclasses import dataclass
from enum import Enum


class ControlConflict(RuntimeError):
    pass


class State(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    FAULT = "fault"


@dataclass(frozen=True)
class Fault:
    reason: str
    message: str
    action: str


class Node:
    def __init__(self):
        self.state = State.IDLE
        self.fault = None

    def require(self, *states):
        if self.state not in states:
            raise ControlConflict(
                f"Operation requires {', '.join(s.value for s in states)}; state is {self.state.value}"
            )

    def start(self):
        self.require(State.IDLE)
        self.state = State.RUNNING

    def stop(self):
        self.require(State.RUNNING, State.IDLE)
        self.state = State.IDLE

    def fail(self, reason, message, action):
        if self.fault is None:
            self.fault = Fault(reason, str(message), action)
        self.state = State.FAULT

    def reset_fault(self):
        self.require(State.FAULT)
        self.fault = None
        self.state = State.IDLE
