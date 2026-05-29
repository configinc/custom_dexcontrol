#!/bin/bash
# Remote dexsensor launcher with ZED preflight cleanup (avoids camera lock retry loop).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/../ansible/inventory.ini"
NANO_HOST="dexmate-nano@192.168.50.22"

DEXMATE_PASS="$(grep -m1 'dexmate_pass=' "$INVENTORY" | cut -d= -f2)"
if [ -z "$DEXMATE_PASS" ]; then
    echo "ERROR: dexmate_pass not found in $INVENTORY"
    exit 1
fi

SSH_OPTS=(
    -o StrictHostKeyChecking=no
    -o ConnectTimeout=10
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
)

run_remote_launch() {
    sshpass -p "$DEXMATE_PASS" ssh "${SSH_OPTS[@]}" "$NANO_HOST" bash -s <<'REMOTE_EOF'
set -e
pkill -f "dexsensor launch" 2>/dev/null || true
sleep 2
zed_state() {
  ZED_Explorer -a 2>&1 | sed -n 's/.*State *: *"\(.*\)".*/\1/p' | head -1
}
STATE="$(zed_state)"
if [ "$STATE" != "AVAILABLE" ]; then
  for _ in 1 2 3 4 5; do
    sleep 2
    STATE="$(zed_state)"
    [ "$STATE" = "AVAILABLE" ] && break
  done
fi
echo "[nano] ZED state before launch: ${STATE:-UNKNOWN}"
if [ "$STATE" != "AVAILABLE" ]; then
  echo "[nano] Aborting: camera not AVAILABLE"
  exit 1
fi
exec dexsensor launch --sensor head_camera \
  --set head_camera.resolution=SVGA \
  --set head_camera.rate=30 \
  --set head_camera.left_rgb_transport.quality=75 \
  --set head_camera.right_rgb_transport.quality=75
REMOTE_EOF
}

attempt=0
while true; do
    attempt=$((attempt + 1))
    echo "[nano] Connecting to dexmate-nano (attempt $attempt)..."

    set +e
    run_remote_launch
    exit_code=$?
    set -e

    if [ "$exit_code" -eq 0 ]; then
        echo "[nano] dexsensor exited cleanly (exit 0)"
        break
    fi

    echo "[nano] Session ended (exit $exit_code), retrying in 15s..."
    sleep 15
done
