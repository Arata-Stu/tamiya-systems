#!/bin/bash

discover_map_candidates() {
  local metadata
  local map_dir
  local -a discovered=()

  MAP_CANDIDATES=()
  if [[ ! -d "$MAP_SEARCH_ROOT" ]]; then
    return 1
  fi

  while IFS= read -r metadata; do
    [[ -z "$metadata" ]] && continue
    map_dir="$(dirname "$metadata")"
    discovered+=("$map_dir")
  done < <(find "$MAP_SEARCH_ROOT" -type f -name '*.yaml' ! -path '*/cuvslam_map/*' 2>/dev/null)

  if [[ "${#discovered[@]}" -eq 0 ]]; then
    return 1
  fi

  while IFS= read -r map_dir; do
    [[ -n "$map_dir" ]] && MAP_CANDIDATES+=("$map_dir")
  done < <(printf '%s\n' "${discovered[@]}" | sort -u)

  [[ "${#MAP_CANDIDATES[@]}" -gt 0 ]]
}

build_command() {
  local cmd=("ros2" "launch" "system_launch" "system.launch.xml")
  local key

  for key in "${BOOL_KEYS[@]}"; do
    cmd+=("${key}:=$(get_arg "$key")")
  done
  cmd+=("bag_manager_param:=${ARG_bag_manager_param}")

  if [ "${#EXTRA_ARGS[@]}" -gt 0 ]; then
    cmd+=("${EXTRA_ARGS[@]}")
  fi

  printf '%q ' "${cmd[@]}"
  echo
}

source_setup_if_available() {
  if [[ "$SETUP_SOURCED" == "true" ]]; then
    return 0
  fi

  local nounset_was_enabled=0
  if [[ $- == *u* ]]; then
    nounset_was_enabled=1
    set +u
  fi

  if [[ -n "${SYSTEM_LAUNCH_SETUP:-}" ]]; then
    # shellcheck source=/dev/null
    source "$SYSTEM_LAUNCH_SETUP"
  elif [[ -f "/workspaces/install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "/workspaces/install/setup.bash"
  elif [[ -f "install/setup.bash" ]]; then
    # shellcheck source=/dev/null
    source "install/setup.bash"
  fi

  if [[ ${nounset_was_enabled} -eq 1 ]]; then
    set -u
  fi

  SETUP_SOURCED="true"
}

resolve_package_share_dir() {
  local package_name="$1"
  local share_dir=""
  local prefix=""

  if [[ "$package_name" == "system_launch" && -d "$SYSTEM_LAUNCH_SOURCE_SHARE" ]]; then
    printf '%s\n' "$SYSTEM_LAUNCH_SOURCE_SHARE"
    return 0
  fi

  source_setup_if_available

  if ! command -v ros2 >/dev/null 2>&1; then
    return 1
  fi

  share_dir="$(ros2 pkg prefix --share "$package_name" 2>/dev/null || true)"
  if [[ -n "$share_dir" ]]; then
    printf '%s\n' "$share_dir"
    return 0
  fi

  prefix="$(ros2 pkg prefix "$package_name" 2>/dev/null || true)"
  if [[ -n "$prefix" ]]; then
    printf '%s\n' "${prefix%/}/share/$package_name"
    return 0
  fi

  return 1
}

resolve_find_pkg_share_value() {
  local value="$1"
  local token
  local package_name
  local share_dir

  while [[ "$value" =~ \$\(find-pkg-share[[:space:]]+([[:alnum:]_]+)\) ]]; do
    token="${BASH_REMATCH[0]}"
    package_name="${BASH_REMATCH[1]}"

    if ! share_dir="$(resolve_package_share_dir "$package_name")"; then
      echo "Failed to resolve package share for: $package_name" >&2
      exit 1
    fi

    value="${value/$token/$share_dir}"
  done

  printf '%s\n' "$value"
}

resolve_launch_arg_assignment() {
  local arg="$1"
  local key
  local value

  case "$arg" in
    *:=*)
      key="${arg%%:=*}"
      value="${arg#*:=}"
      printf '%s:=%s\n' "$key" "$(resolve_find_pkg_share_value "$value")"
      ;;
    *=*)
      key="${arg%%=*}"
      value="${arg#*=}"
      printf '%s=%s\n' "$key" "$(resolve_find_pkg_share_value "$value")"
      ;;
    *)
      printf '%s\n' "$(resolve_find_pkg_share_value "$arg")"
      ;;
  esac
}

resolve_launch_paths() {
  local resolved_extra_args=()
  local arg

  ARG_bag_manager_param="$(resolve_find_pkg_share_value "$ARG_bag_manager_param")"

  for arg in "${EXTRA_ARGS[@]}"; do
    resolved_extra_args+=("$(resolve_launch_arg_assignment "$arg")")
  done

  EXTRA_ARGS=("${resolved_extra_args[@]}")
}

