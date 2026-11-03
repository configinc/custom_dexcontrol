#!/bin/bash
# (Re)start the Vega body/arm robot server in tmux "robot-server" — same launch as
# custom_dexcontrol ansible restart.yml, plus a local head init pose for the ZED head camera.
# Usage: ~/start_robot_server.sh [J1 J2 J3]   (default: $HEAD_INIT_POS or "0.45 0.0 0.0")
HEAD="${*:-${HEAD_INIT_POS:-0.45 0.0 0.0}}"
if tmux has-session -t robot-server 2>/dev/null; then
  tmux kill-session -t robot-server            # SIGHUP -> server runs its stop/hold cleanup
  for i in $(seq 1 30); do pgrep -f "^python -u -m dexcontrol.core.robotenv_vega.body_server" >/dev/null || break; sleep 1; done
fi
mkdir -p ~/.local/state/dexcontrol
tmux new-session -d -s robot-server "bash -il"
tmux set-option -t robot-server remain-on-exit on
tmux pipe-pane -o -t robot-server 'cat >> ~/.local/state/dexcontrol/body_server.log'
tmux send-keys -t robot-server "conda activate robot && cd ~/custom_dexcontrol && exec python -u -m dexcontrol.core.robotenv_vega.body_server --host 0.0.0.0 --left-port 50061 --right-port 50063 --body-port 50065 --gripper-type robotiq --action-hz 20 --control-loop-hz 200 --use-velocity-feedforward --interpolation-method linear --interpolation-history 3 --vel-smoothing-alpha 1.0 --hw-correction-alpha 0.5 --max-delta-scale 3.0 --max-jerk 0 --rot-sensitivity 2.0 --vel-ratio 1.0 --head-init-pos $HEAD" C-m
echo "robot-server starting with --head-init-pos $HEAD"
