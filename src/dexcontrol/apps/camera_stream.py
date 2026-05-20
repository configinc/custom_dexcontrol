import logging
import socket
import struct
import time
import tyro

from dexcomm import Node

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

_STALE_THRESHOLD_S = 0.5        # frames older than this threshold (from receive time) are dropped
_MAX_FRAME_BYTES   = 960 * 600 * 3  # upper bound on raw frame size, used for SO_SNDBUF calculation

_TOPIC_MAP = {
    "left_rgb":  "sensors/head_camera/left_rgb",
    "right_rgb": "sensors/head_camera/right_rgb",
}


def _make_sndbuf(fps: float) -> int:
    return int(_MAX_FRAME_BYTES * fps * _STALE_THRESHOLD_S)


def stream_frames(host: str, port: int, fps: float, camera_key: str = "right_rgb"):
    topic = _TOPIC_MAP.get(camera_key)
    if topic is None:
        raise ValueError(f"Unsupported camera_key: {camera_key!r}. Valid values: {list(_TOPIC_MAP)}")

    interval = 1.0 / fps
    sndbuf = _make_sndbuf(fps)

    node = Node("head_camera_stream_node")
    # decoder=None → returns raw bytes as serialized by dexcomm's NumpyArrayCodec
    sub = node.create_subscriber(topic=topic, decoder=None, buffer_size=1)

    logging.info(f"Zenoh subscription started: {topic}")
    logging.info(f"Connecting to recorder PC {host}:{port}...")

    while True:
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, sndbuf)
            sock.connect((host, port))
            logging.info(f"Connected → {host}:{port}  SO_SNDBUF≈{sndbuf//1024}KB")

            stat_frames = 0
            stat_bytes = 0
            stat_skipped = 0
            stat_t = time.time()

            while True:
                t0 = time.time()

                if t0 - stat_t >= 1.0:
                    elapsed_s = t0 - stat_t
                    logging.info(
                        f"[stats] {stat_frames / elapsed_s:.1f} fps  |  "
                        f"{stat_bytes / elapsed_s / 1024:.0f} KB/s  |  "
                        f"sent {stat_frames} frames  |  skipped {stat_skipped} frames"
                    )
                    stat_frames = 0
                    stat_bytes = 0
                    stat_skipped = 0
                    stat_t = t0

                jpeg_bytes = sub.get_latest()
                if jpeg_bytes is None:
                    time.sleep(0.005)
                    continue

                rx_ns = sub.get_receive_time_ns()
                age = (time.time_ns() - rx_ns) / 1e9
                if age > _STALE_THRESHOLD_S:
                    logging.debug(f"Stale frame skipped (age={age:.3f}s)")
                    stat_skipped += 1
                    time.sleep(interval)
                    continue

                cap_ns = rx_ns
                send_ns = time.time_ns()
                # header: data_len(4B) + cap_ns(8B) + send_ns(8B) = 20 bytes
                header = struct.pack(">IQQ", len(jpeg_bytes), cap_ns, send_ns)
                try:
                    sock.sendall(header + bytes(jpeg_bytes))
                    stat_frames += 1
                    stat_bytes += len(jpeg_bytes)
                except OSError as e:
                    logging.warning(f"Send failed: {e}")
                    break

                elapsed = time.time() - t0
                sleep = interval - elapsed
                if sleep > 0:
                    time.sleep(sleep)

        except (ConnectionRefusedError, OSError) as e:
            logging.warning(f"Connection failed: {e} — retrying in 3s")
            time.sleep(3.0)
        finally:
            if sock is not None:
                sock.close()


def main(host: str = "192.168.5.17", port: int = 9876, fps: float = 30.0,
         camera_key: str = "right_rgb"):
    stream_frames(host=host, port=port, fps=fps, camera_key=camera_key)


if __name__ == "__main__":
    tyro.cli(main)
