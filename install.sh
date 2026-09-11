#!/usr/bin/env bash
set -euo pipefail

project_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-2}"
source "$project_dir/scripts/dependencies.sh"
prepare_dependency loop-sdk
prepare_dependency loop

for gripper in robotiq_2f_85_controller sr_gripper_controller; do
    if [[ ! -f "$project_dir/$gripper/pyproject.toml" ]]; then
        if [[ ! -e "$project_dir/.git" ]]; then
            printf 'Missing %s source. Initialize submodules before copying this checkout.\n' "$gripper" >&2
            exit 1
        fi
        git -C "$project_dir" submodule update --init "$gripper"
    fi
done
UV_PYTHON="${UV_PYTHON:-3.12}" uv sync --project "$project_dir" --locked --extra loop --no-default-groups "$@"
