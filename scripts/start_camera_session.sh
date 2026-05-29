#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INVENTORY="$SCRIPT_DIR/../ansible/inventory.ini"

DEXMATE_PASS="$(grep -m1 'dexmate_pass=' "$INVENTORY" | cut -d= -f2)"

if [ -z "$DEXMATE_PASS" ]; then
    echo "ERROR: dexmate_pass not found in $INVENTORY"
    exit 1
fi

export DEXMATE_PASS

# === 1. nano 세션: SSH 접속 후 dexsensor 실행 (실패 시 재시도) ===
tmux kill-session -t nano 2>/dev/null || true
tmux new-session -d -s nano

tmux send-keys -t nano "bash $SCRIPT_DIR/nano_dexsensor_loop.sh" Enter

# === 2. camera 세션: head 이동 후 camera_stream 실행 (실패 시 재시도) ===
tmux kill-session -t camera 2>/dev/null || true
tmux new-session -d -s camera

CAMERA_SCRIPT="$SCRIPT_DIR/../src/dexcontrol/apps/camera_stream.py"

tmux send-keys -t camera "
while true; do
    echo '[camera] Starting camera_stream...'
    python $CAMERA_SCRIPT
    echo '[camera] Process ended, retrying in 5s...'
    sleep 5
done
" Enter

echo "Sessions started:"
echo "  tmux attach -t nano    -> SSH + dexsensor"
echo "  tmux attach -t camera  -> head control + camera stream"