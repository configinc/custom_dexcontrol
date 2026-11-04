"""Exclusive actions with cancellable workers and optional concurrent participants."""

import threading
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

from .fsm import ControlConflict


class ActionCancelled(RuntimeError):
    pass


@dataclass
class ActionStatus:
    name: str = ""
    state: str = "idle"
    error: str = ""


class _ActionWorker(threading.Thread):
    """Count registered jobs until completion, including the pre-start window."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.done = threading.Event()

    def start(self):
        try:
            super().start()
        except BaseException:
            self.done.set()
            raise

    def run(self):
        try:
            super().run()
        finally:
            self.done.set()


class ActionParticipant:
    """One producer in an action; completion includes all of its worker cleanup."""

    def __init__(self, runner, key, on_cancel):
        self.runner = runner
        self.key = key
        self.on_cancel = on_cancel
        self.owner = runner.owner
        self.cancelled = runner.cancelled
        self.workers = []
        self.finished = False
        self.outcome = "succeeded"
        self.error = ""

    @property
    def busy(self):
        return not self.finished or any(
            not worker.done.is_set() for worker in self.workers
        )

    @contextmanager
    def commands(self):
        if self.finished:
            raise ControlConflict("Action participant no longer owns robot output")
        with self.runner.gate.producing(self.owner):
            yield

    def worker(self, *, target, args=(), name=None, daemon=True):
        """Register before starting, so cancellation also waits for cleanup jobs."""
        with self.runner.lock:
            if self.finished:
                raise ControlConflict("Action participant has already finished")

            def run():
                try:
                    with self.runner.gate.producing(self.owner):
                        target(*args)
                except BaseException as exc:
                    self.finish(
                        error=""
                        if self.cancelled.is_set()
                        else (str(exc) or type(exc).__name__),
                        cancelled=self.cancelled.is_set(),
                    )

            worker = _ActionWorker(target=run, name=name, daemon=daemon)
            self.workers.append(worker)
            return worker

    def finish(self, *, error=None, cancelled=False):
        with self.runner.lock:
            if self.finished:
                return
            self.finished = True
            self.error = error or ""
            self.outcome = (
                "failed" if error else "cancelled" if cancelled else "succeeded"
            )


class ActionRunner:
    def __init__(self, gate):
        self.gate = gate
        self.status = ActionStatus()
        self.cancelled = threading.Event()
        self.worker = None
        self.owner = None
        self.participants = {}
        self.output_tick = None
        self.lock = threading.RLock()

    @property
    def busy(self):
        with self.lock:
            return any(participant.busy for participant in self.participants.values())

    def snapshot(self):
        with self.lock:
            if self.participants:
                failed = next(
                    (p for p in self.participants.values() if p.outcome == "failed"),
                    None,
                )
                if failed is not None:
                    self.status.state, self.status.error = "failed", failed.error
                elif self.busy:
                    self.status.state = (
                        "cancelling" if self.cancelled.is_set() else "running"
                    )
                else:
                    self.status.state = (
                        "cancelled"
                        if self.cancelled.is_set()
                        or any(
                            p.outcome == "cancelled" for p in self.participants.values()
                        )
                        else "succeeded"
                    )
            return asdict(self.status)

    def begin(self, name, *, output_tick=None):
        with self.lock:
            if self.busy:
                raise ControlConflict("Previous action has not finished cancelling")
            self.cancelled = threading.Event()
            self.owner = object()
            self.status = ActionStatus(name, "running")
            self.participants = {}
            self.worker = None
            self.output_tick = output_tick
            self.gate.select(self.owner)

    def can_join(self, name):
        with self.lock:
            return (
                self.status.name == name
                and self.snapshot()["state"] == "running"
                and bool(self.participants)
            )

    def add_participant(self, key, on_cancel=None):
        with self.lock:
            if self.snapshot()["state"] != "running" or key in self.participants:
                raise ControlConflict("Action is not accepting this participant")
            participant = ActionParticipant(self, key, on_cancel)
            self.participants[key] = participant
            return participant

    def start(self, name, operation):
        with self.lock:
            self.begin(name)
            participant = self.add_participant(name)

            def run():
                operation(participant.cancelled)
                participant.finish(cancelled=participant.cancelled.is_set())

            self.worker = participant.worker(target=run, name=f"vega-action-{name}")
            try:
                self.worker.start()
            except BaseException as exc:
                participant.finish(error=str(exc))
                raise

    def tick(self):
        if self.output_tick is not None and self.snapshot()["state"] == "running":
            with self.gate.producing(self.owner):
                self.output_tick()

    def cancel(self):
        with self.lock:
            if not self.busy or self.cancelled.is_set():
                return
            self.cancelled.set()
            callbacks = [
                p.on_cancel
                for p in self.participants.values()
                if p.on_cancel is not None and not p.finished
            ]
        # Callbacks signal cancellation and schedule cleanup; they must not wait
        # for a participant's motion or command lock on the control thread.
        for callback in callbacks:
            callback()

    def join(self, timeout=2.0):
        deadline = time.monotonic() + timeout
        while self.busy:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise RuntimeError("Action did not stop after cancellation")
            with self.lock:
                workers = [
                    w
                    for p in self.participants.values()
                    for w in p.workers
                    if w.is_alive()
                ]
            if workers:
                workers[0].join(min(remaining, 0.01))
            else:
                time.sleep(min(remaining, 0.01))
