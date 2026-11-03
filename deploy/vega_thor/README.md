# Vega Thor onboard (branch vega-thor-onboard)

`body_server.py` gains `--head-init-pos J1 J2 J3` (omitted = unchanged default 2.0 0.0 -0.3).
`deploy/vega_thor/start_robot_server.sh` launches the server like ansible restart.yml plus a head pose
(0004-08: 0.45 0 0, upright enough for the ZED X Mini). Setup guide: robot-control-interface@vega-thor-onboard
`deploy/vega_thor/README.md`.
