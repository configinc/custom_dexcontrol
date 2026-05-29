#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/ansible/inventory.ini"

DEXMATE_PASS="$(grep -m1 'dexmate_pass=' "$INVENTORY" | cut -d= -f2)"

if [ -z "$DEXMATE_PASS" ]; then
    echo "ERROR: dexmate_pass not found in $INVENTORY"
    exit 1
fi

# === 1. nano 세션: SSH 접속 후 dexsensor 실행 (실패 시 재시도) ===
tmux kill-session -t nano 2>/dev/null || true
tmux new-session -d -s nano

tmux send-keys -t nano "
while true; do
    echo \"[nano] Connecting to dexmate-nano...\"
    sshpass -p '$DEXMATE_PASS' ssh -o StrictHostKeyChecking=no \
        -o ConnectTimeout=10 \
        -o ServerAliveInterval=5 \
        -o ServerAliveCountMax=3 \
        dexmate-nano@192.168.50.22 \
        'echo [nano] SSH connected; dexsensor launch --sensor head_camera \
          --set head_camera.resolution=SVGA \
          --set head_camera.rate=30 \
          --set head_camera.left_rgb_transport.quality=75 \
          --set head_camera.right_rgb_transport.quality=75'
    echo \"[nano] Session ended (exit \$?), retrying in 5s...\"
    sleep 5
done
" Enter

# === 2. camera 세션: head 이동 후 camera_stream 실행 (실패 시 재시도) ===
tmux kill-session -t camera 2>/dev/null || true
tmux new-session -d -s camera

tmux send-keys -t camera "
while true; do
    echo \"[camera] Starting camera_stream...\"
    python ~/custom_dexcontrol/src/dexcontrol/apps/camera_stream.py
    echo \"[camera] Process ended (exit \$?), retrying in 5s...\"
    sleep 5
done
" Enter

echo "Sessions started:"
echo "  tmux attach -t nano    -> SSH + dexsensor"
echo "  tmux attach -t camera  -> head control + camera stream"
