"""Circular file handler that overwrites the oldest log entries when the file exceeds SOFT_LIMIT."""

from __future__ import annotations

import logging
import os
import struct
import threading

SOFT_LIMIT = 500 * 1024 * 1024  # 500MB — threshold for wrapping
MAX_BYTES = 505 * 1024 * 1024   # 505MB — actual file size (margin to prevent record truncation)


class CircularFileHandler(logging.Handler):

    def __init__(self, filepath: str):
        super().__init__()
        self._filepath = filepath
        self._metapath = filepath + ".pos"
        self._lock = threading.Lock()
        self._write_pos = self._load_pos()
        existed = os.path.exists(filepath) and os.path.getsize(filepath) >= MAX_BYTES
        self._file = open(filepath, "r+b" if existed else "w+b")
        if not existed:
            self._file.write(b"\x00" * MAX_BYTES)
            self._file.flush()

    def _load_pos(self) -> int:
        if os.path.exists(self._metapath):
            with open(self._metapath, "rb") as f:
                data = f.read(8)
                if len(data) == 8:
                    return struct.unpack(">Q", data)[0]
        return 0

    def _save_pos(self) -> None:
        with open(self._metapath, "wb") as f:
            f.write(struct.pack(">Q", self._write_pos))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            encoded = (self.format(record) + "\n").encode("utf-8", errors="replace")
            with self._lock:
                if self._write_pos >= SOFT_LIMIT:
                    self._write_pos = 0
                self._file.seek(self._write_pos)
                self._file.write(encoded)
                self._write_pos += len(encoded)
                self._file.flush()
                self._save_pos()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._lock:
            self._file.close()
        super().close()
