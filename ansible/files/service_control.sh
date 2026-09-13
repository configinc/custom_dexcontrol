#!/usr/bin/env bash
# Local tmux/container helpers for this repository's Loop service entrypoint.
set -euo pipefail

has_session() {
    command -v tmux >/dev/null && tmux has-session -t "=$1" 2>/dev/null
}

require_stopped() {
    local session
    for session in "$@"; do
        if has_session "$session"; then
            echo "$session is running. Stop it before continuing." >&2
            return 1
        fi
    done
}

stop_session() {
    local session=$1 pane commands attempt
    has_session "$session" || return 0
    while read -r pane; do
        tmux send-keys -t "$pane" C-c 2>/dev/null || true
    done < <(tmux list-panes -s -t "=$session" -F '#{pane_id}')
    for attempt in {1..30}; do
        has_session "$session" || return 0
        commands=$(tmux list-panes -s -t "=$session" -F '#{pane_current_command}' 2>/dev/null) || continue
        # Legacy sessions retain their interactive shell after Ctrl+C.
        if ! printf '%s\n' "$commands" | grep -qEv '^(bash|sh|zsh|fish)$'; then
            tmux kill-session -t "=$session" 2>/dev/null || ! has_session "$session"
            return $?
        fi
        sleep 1
    done
    echo "$session did not finish shutdown; no replacement was started." >&2
    return 1
}

check_install() {
    local project=$1 commit=$2 script=$3 executable=$4
    if ! test -x "$script" || ! test -x "$project/$executable" ||
       ! grep -qxF "commit: $commit" "$project/version_info.txt"; then
        echo "Install $project through Deploy before restarting Robot/UTI (expected $commit)." >&2
        return 1
    fi
}

start_session() {
    local session=$1 project=$2 script=$3
    shift 3
    require_stopped "$session" || return
    test -x "$script" || return
    tmux new-session -d -s "$session" -c "$project" \
        -e "LOOP_NODE_GRAPH_NODE_ENDPOINT=${LOOP_NODE_GRAPH_NODE_ENDPOINT:-}" \
        -e "ROBOT_NAME=${ROBOT_NAME:-}" -e "ZENOH_CONFIG=${ZENOH_CONFIG:-}" \
        "exec $(printf '%q ' "$script" "$@")" || return
    sleep 2
    has_session "$session"
}
