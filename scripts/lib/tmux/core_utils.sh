#!/bin/bash

resolve_real_path() {
  local input_path="$1"

  if command -v python3 >/dev/null 2>&1; then
    python3 - "$input_path" <<'PY'
import os
import sys

print(os.path.realpath(sys.argv[1]))
PY
    return 0
  fi

  if command -v perl >/dev/null 2>&1; then
    perl -MCwd=realpath -e 'print realpath($ARGV[0])' "$input_path"
    return 0
  fi

  printf '%s\n' "$input_path"
}

is_repo_root() {
  local candidate="$1"

  [[ -d "$candidate" ]] || return 1
  [[ -d "$candidate/scripts" ]] || return 1
  [[ -d "$candidate/ros2_ws" ]] || return 1
  [[ -f "$candidate/scripts/tmux.sh" ]] || return 1
}

find_repo_root_from() {
  local current="$1"

  [[ -n "$current" ]] || return 1

  if [[ -f "$current" ]]; then
    current="$(dirname "$current")"
  fi

  [[ -d "$current" ]] || return 1

  while true; do
    if is_repo_root "$current"; then
      (cd "$current" && pwd)
      return 0
    fi

    [[ "$current" == "/" ]] && break
    current="$(dirname "$current")"
  done

  return 1
}

resolve_repo_root() {
  local candidate

  for candidate in \
    "${INITIAL_PWD}" \
    "${SCRIPT_PATH}" \
    "$(dirname "${SCRIPT_PATH}")" \
    "$(dirname "$(dirname "${SCRIPT_PATH}")")" \
    /workspaces \
    /workspace; do
    [[ -n "$candidate" ]] || continue
    if find_repo_root_from "$candidate"; then
      return 0
    fi
  done

  (cd "$(dirname "${SCRIPT_PATH}")/.." && pwd)
}

resolve_existing_dir() {
  local candidate

  for candidate in "$@"; do
    if [[ -n "$candidate" && -d "$candidate" ]]; then
      printf '%s\n' "$candidate"
      return 0
    fi
  done

  return 1
}

resolve_setup_command() {
  local candidate

  for candidate in \
    "${TMUX_WORKSPACE_SETUP:-}" \
    "${REPO_ROOT}/install/setup.bash" \
    "/workspaces/install/setup.bash" \
    "install/setup.bash"; do
    if [[ -n "$candidate" && -f "$candidate" ]]; then
      printf 'source %s\n' "$candidate"
      return 0
    fi
  done

  printf '\n'
}

build_init_cmd() {
  local dir="$1"
  local setup="$2"
  local cmd=""

  if [[ -n "$dir" ]]; then
    cmd="cd $dir"
  fi

  if [[ -n "$setup" ]]; then
    if [[ -n "$cmd" ]]; then
      cmd="$cmd && $setup"
    else
      cmd="$setup"
    fi
  fi

  echo "$cmd"
}

