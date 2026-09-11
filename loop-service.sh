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
source "$service_dir/ansible/files/service_control.sh"

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

relay_target() {
    local endpoint=${loop_endpoint:-${LOOP_NODE_GRAPH_NODE_ENDPOINT:-}}
    if [[ ! "$endpoint" =~ ^tcp/([a-zA-Z0-9_.-]+):([0-9]+)$ ]]; then
        echo 'Set LOOP_NODE_GRAPH_NODE_ENDPOINT on the Teleop PC to tcp/GPU_HOST:PORT.' >&2
        return 1
    fi
    relay_destination="${BASH_REMATCH[1]}:${BASH_REMATCH[2]}"
}

relay_address() {
    if [[ -z "$relay_bind_address" ]]; then
        local host
        host=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["host"])' "$connection_file")
        relay_bind_address=$(ip -o -4 route get "$host" | awk '{for(i=1;i<=NF;i++) if($i=="src") {print $(i+1); exit}}')
    fi
    [[ -n "$relay_bind_address" ]] || { echo 'Cannot determine the Teleop address facing Vega.' >&2; return 1; }
}

case "${1:-}" in
    preflight)
        for tool in socat tmux sshpass python3 ip; do command -v "$tool" >/dev/null; done
        relay_target
        relay_address
        remote_robot preflight "${2:?Expected deployed commit is required}"
        ;;
    start)
        relay_target
        relay_address
        require_stopped "$relay_session"
        remote_robot check-stopped
        # The robot connects to this listener; the destination comes from the Teleop environment.
        relay_command=(socat -d -d -b 65536
            "TCP4-LISTEN:$relay_port,bind=$relay_bind_address,reuseaddr,fork,nodelay"
            "TCP4-CONNECT:$relay_destination,nodelay,connect-timeout=5")
        tmux new-session -d -s "$relay_session" -c "$service_dir" \
            "exec $(printf '%q ' "${relay_command[@]}")"
        sleep 1
        has_session "$relay_session" || { echo 'Vega relay failed to start.' >&2; exit 1; }
        if ! remote_robot start "tcp/$relay_bind_address:$relay_port"; then
            remote_robot stop || true
            stop_session "$relay_session"
            exit 1
        fi
        ;;
    stop)
        failed=0
        remote_robot stop || failed=1
        stop_session "$relay_session" || failed=1
        exit "$failed"
        ;;
    check-stopped)
        require_stopped "$relay_session"
        remote_robot check-stopped
        ;;
    *) echo "Usage: $0 {start|stop|check-stopped|preflight COMMIT}" >&2; exit 2 ;;
esac
