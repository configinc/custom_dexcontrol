#!/usr/bin/env bash
set -euo pipefail
service_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
# An interrupted first installation may have copied source before writing config.
if [[ ! -e "$service_dir/.env.loop-service" && "${1:-}" == check-stopped ]]; then
    exit 0
fi
if [[ ! -r "$service_dir/.env.loop-service" ]]; then
    echo 'Run Deploy to configure this Loop service first.' >&2
    exit 1
fi
source "$service_dir/.env.loop-service"

remote_robot() {
    python3 "$service_dir/ansible/files/vega_ssh.py" "$connection_file" \
        bash -s -- "$robot_project_dir" "$robot_start_script" "$robot_session" "$@" <<'ROBOT'
set -euo pipefail
project=$1 script=$2 session=$3 action=$4
shift 4
source "$project/ansible/files/service_control.sh"
case "$action" in
    preflight) check_install "$project" "$1" "$script" .venv/bin/dexcontrol-loop-robot-node ;;
    start) start_session "$session" "$project" "$script" --loop-endpoint "$1" ;;
    stop) stop_session "$session" ;;
    check-stopped) require_stopped "$session" ;;
esac
ROBOT
}

resolve_loop_endpoint() {
    loop_endpoint=${loop_endpoint:-${LOOP_NODE_GRAPH_NODE_ENDPOINT:-}}
    if [[ ! "$loop_endpoint" =~ ^tcp/([a-zA-Z0-9_.-]+):([0-9]+)$ ]]; then
        echo 'Set LOOP_NODE_GRAPH_NODE_ENDPOINT on Teleop to tcp/TELEOP_VEGA_IP:PORT, or set dexcontrol_loop_endpoint when deploying.' >&2
        return 1
    fi
}

case "${1:-}" in
    preflight)
        for tool in sshpass python3; do command -v "$tool" >/dev/null; done
        resolve_loop_endpoint
        remote_robot preflight "${2:?Expected deployed commit is required}"
        ;;
    start)
        resolve_loop_endpoint
        remote_robot check-stopped
        if ! remote_robot start "$loop_endpoint"; then
            remote_robot stop || true
            exit 1
        fi
        ;;
    stop)
        remote_robot stop
        ;;
    check-stopped)
        remote_robot check-stopped
        ;;
    *) echo "Usage: $0 {start|stop|check-stopped|preflight COMMIT}" >&2; exit 2 ;;
esac
