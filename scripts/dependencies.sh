#!/usr/bin/env bash

dexcontrol_repo_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)

fetch_source() {
    local name=$1 url=$2 revision=$3 sparse_path=${4:-}
    local destination="$dexcontrol_repo_dir/third_party/$name"

    if [[ ! -e "$destination" ]]; then
        mkdir -p -- "$destination"
        git init --quiet "$destination"
        git -C "$destination" remote add origin "$url"
        git -C "$destination" config remote.origin.promisor true
        git -C "$destination" config remote.origin.partialclonefilter blob:none
    fi

    if [[ ! -d "$destination/.git" ]] ||
        [[ $(git -C "$destination" config --get remote.origin.url) != "$url" ]]; then
        printf 'Expected a checkout of %s at %s\n' "$url" "$destination" >&2
        return 1
    fi
    if [[ $(git -C "$destination" rev-parse --verify HEAD 2>/dev/null || true) == "$revision" ]]; then
        return 0
    fi
    if [[ -n $(git -C "$destination" status --porcelain --untracked-files=no) ]]; then
        printf 'Commit or stash changes in %s before changing its revision.\n' "$destination" >&2
        return 1
    fi

    if [[ "$name" == loop-sdk ]]; then
        # hatch-vcs needs tags and commit history to calculate the SDK version.
        git -C "$destination" fetch --filter=blob:none --tags origin "$revision"
    else
        git -C "$destination" fetch --filter=blob:none --depth=1 --no-tags origin "$revision"
    fi
    if [[ -n "$sparse_path" ]]; then
        # Exclude Loop's root workspace metadata as well as unrelated packages.
        git -C "$destination" sparse-checkout set --no-cone "/$sparse_path/"
    fi
    git -C "$destination" checkout --quiet --detach "$revision"
}

prepare_dependency() {
    case "$1" in
        loop-sdk)
            fetch_source loop-sdk git@github.com:configinc/loop-sdk.git \
                cab0a54b3cb3d1b00b9676d5742eb5ca1db57ea5
            ;;
        loop)
            fetch_source loop git@github.com:configinc/loop.git \
                a3ef49573c7c58ed98c80b908995bc15a208ca27 packages/loop-node
            ;;
        *)
            printf 'Unknown dependency: %s\n' "$1" >&2
            return 1
            ;;
    esac
}
