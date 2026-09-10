"""Discover the two gripper ports used by the deployment launcher."""

import argparse
import contextlib
from pathlib import Path


def discover(gripper_type: str) -> list[str]:
    if gripper_type == "robotiq":
        return sorted(str(path) for path in Path("/dev").glob("ttyUSB*"))

    import pysoem

    ports = []
    for adapter in pysoem.find_adapters():
        if adapter.name.startswith(("lo", "wl", "tailscale", "docker", "veth", "br-")):
            continue
        master = pysoem.Master()
        try:
            master.open(adapter.name)
            if (
                master.config_init()
                and master.slaves[0].man == 0x534543
                and master.slaves[0].id == 0x1002
            ):
                ports.append(adapter.name)
        except Exception:
            continue
        finally:
            with contextlib.suppress(Exception):
                master.close()
    return sorted(ports)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gripper_type", choices=["robotiq", "sr_gripper"])
    args = parser.parse_args()
    ports = discover(args.gripper_type)
    if len(ports) != 2:
        parser.error(
            "Expected two gripper ports; configure explicit per-arm ports instead."
        )
    print("\n".join(ports))


if __name__ == "__main__":
    main()
