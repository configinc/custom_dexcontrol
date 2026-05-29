import logging
import queue
import socket
import struct
import threading
import time
import tyro

import numpy as np
from dexbot_utils import RobotInfo
from dexcomm import Node

from dexcontrol.core.head import Head
from dexcontrol.core.torso import Torso

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

_HEAD_PRESET = np.array([-2.0, 0.0, 0.17])  # yaw, pitch, roll (radians)

_STALE_THRESHOLD_S = 0.5        # frames older than this threshold (from receive time) are dropped
_MAX_FRAME_BYTES   = 960 * 600 * 3  # upper bound on raw frame size, used for SO_SNDBUF calculation
_SNDBUF_MAX_FPS    = 60.0           # SO_SNDBUF sized for worst-case fps; actual rate follows dexsensor

_TOPIC_MAP = {
    "left_rgb":  "sensors/head_camera/left_rgb",
    "right_rgb": "sensors/head_camera/right_rgb",
}


def _sender_thread(sock, frame_queue: queue.Queue, stats: dict, stop_event: threading.Event):
    """Dedicated thread for TCP sendall so Zenoh receive loop is never blocked by network I/O."""
    while not stop_event.is_set():
        try:
            packet = frame_queue.get(timeout=0.05)
        except queue.Empty:
            continue
        try:
            sock.sendall(packet)
            stats["sent"] += 1
            stats["bytes"] += len(packet)
        except OSError as e:
            logging.warning(f"Send failed: {e}")
            stop_event.set()


def stream_frames(host: str, port: int, camera_key: str = "right_rgb"):
    topic = _TOPIC_MAP.get(camera_key)
    if topic is None:
        raise ValueError(f"Unsupported camera_key: {camera_key!r}. Valid values: {list(_TOPIC_MAP)}")

    # Buffer sized for worst-case fps; actual send rate is dexsensor-driven.
    sndbuf = int(_MAX_FRAME_BYTES * _SNDBUF_MAX_FPS * _STALE_THRESHOLD_S)

    node = Node("head_camera_stream_node")
    # decoder=None → returns raw bytes as serialized by dexcomm's NumpyArrayCodec
    sub = node.create_subscriber(topic=topic, decoder=None, buffer_size=1)

    logging.info(f"Zenoh subscription started: {topic}")
    logging.info(f"Connecting to recorder PC {host}:{port}...")

    while True:
        sock = None
        stop_event = threading.Event()
        sender = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
            sock.connect((host, port))
            logging.info(f"Connected → {host}:{port}  SO_SNDBUF≈{sndbuf//1024}KB")

            # Queue depth 1: Zenoh loop always drops the old frame and puts the newest one.
            # This prevents sendall backpressure from building up a backlog of stale frames.
            frame_queue: queue.Queue = queue.Queue(maxsize=1)
            stats = {"sent": 0, "bytes": 0}
            sender = threading.Thread(
                target=_sender_thread,
                args=(sock, frame_queue, stats, stop_event),
                daemon=True,
            )
            sender.start()

            stat_skipped = 0
            stat_t = time.time()

            while not stop_event.is_set():
                t0 = time.time()

                if t0 - stat_t >= 1.0:
                    elapsed_s = t0 - stat_t
                    sent = stats["sent"]
                    logging.info(
                        f"[stats] {sent / elapsed_s:.1f} fps  |  "
                        f"{stats['bytes'] / elapsed_s / 1024:.0f} KB/s  |  "
                        f"sent {sent} frames  |  skipped {stat_skipped} frames"
                    )
                    stats["sent"] = 0
                    stats["bytes"] = 0
                    stat_skipped = 0
                    stat_t = t0

                jpeg_bytes = sub.get_latest()
                if jpeg_bytes is None:
                    # No new frame yet — poll at 1ms to follow dexsensor rate naturally.
                    time.sleep(0.001)
                    continue

                rx_ns = sub.get_receive_time_ns()
                age = (time.time_ns() - rx_ns) / 1e9
                if age > _STALE_THRESHOLD_S:
                    logging.debug(f"Stale frame skipped (age={age:.3f}s)")
                    stat_skipped += 1
                    continue

                cap_ns = rx_ns
                send_ns = time.time_ns()
                # header: data_len(4B) + cap_ns(8B) + send_ns(8B) = 20 bytes
                header = struct.pack(">IQQ", len(jpeg_bytes), cap_ns, send_ns)
                packet = header + bytes(jpeg_bytes)

                # Drop the queued frame if sender hasn't caught up yet (always keep latest).
                try:
                    frame_queue.put_nowait(packet)
                except queue.Full:
                    try:
                        frame_queue.get_nowait()
                    except queue.Empty:
                        pass
                    frame_queue.put_nowait(packet)
                    stat_skipped += 1

        except (ConnectionRefusedError, OSError) as e:
            logging.warning(f"Connection failed: {e} — retrying in 3s")
            time.sleep(3.0)
        finally:
            stop_event.set()
            if sender is not None:
                sender.join(timeout=1.0)
            if sock is not None:
                sock.close()


def _move_head_to_preset() -> None:
    try:
        robot_info = RobotInfo()
        torso = Torso(name="torso", robot_info=robot_info)
        head = Head(name="head", robot_info=robot_info)
        head.set_mode("enable")

        target = _HEAD_PRESET.copy()
        target[0] += torso.pitch_angle - np.pi / 2

        limits = head.get_joint_pos_limit()
        if limits is not None:
            target = np.clip(target, limits[:, 0], limits[:, 1])
        head.set_joint_pos(target, wait_time=2.0)

        torso.shutdown()
        head.shutdown()
        logging.info("Head moved to preset: yaw=%.2f pitch=%.2f roll=%.2f", *_HEAD_PRESET)
    except Exception as exc:
        logging.warning("Failed to move head to preset (non-fatal): %s", exc)


def main(host: str = "192.168.5.17", port: int = 9876,
         camera_key: str = "right_rgb"):
    _move_head_to_preset()
    stream_frames(host=host, port=port, camera_key=camera_key)


if __name__ == "__main__":
    tyro.cli(main)