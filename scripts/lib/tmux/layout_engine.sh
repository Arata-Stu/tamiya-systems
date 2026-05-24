#!/bin/bash

reset_panes() {
  PANE_WINDOWS=()
  PANE_DIRS=()
  PANE_SETUPS=()
  PANE_PREPARES=()
}

add_pane() {
  local window="$1"
  local dir="${2:-}"
  local setup="${3:-}"
  local prepare="${4:-}"

  PANE_WINDOWS+=("$window")
  PANE_DIRS+=("$dir")
  PANE_SETUPS+=("$setup")
  PANE_PREPARES+=("$prepare")
}

pane_count_for_window() {
  local window="$1"
  local count=0
  local idx

  for idx in "${!PANE_WINDOWS[@]}"; do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      count=$((count + 1))
    fi
  done

  echo "$count"
}

window_exists_in_specs() {
  local window="$1"
  local idx

  for idx in "${!PANE_WINDOWS[@]}"; do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      return 0
    fi
  done

  return 1
}

create_window_panes() {
  local window="$1"
  local pane_count
  local idx

  pane_count="$(pane_count_for_window "$window")"
  for ((idx = 1; idx < pane_count; idx++)); do
    tmux split-window -v -t "$SESSION_NAME":"$window".0
  done
  tmux select-layout -t "$SESSION_NAME":"$window" tiled >/dev/null
}

pane_index_for_spec() {
  local spec_idx="$1"
  local window="${PANE_WINDOWS[$spec_idx]}"
  local pane_index=0
  local idx

  for ((idx = 0; idx < spec_idx; idx++)); do
    if [[ "${PANE_WINDOWS[$idx]}" == "$window" ]]; then
      pane_index=$((pane_index + 1))
    fi
  done

  echo "$pane_index"
}

init_pane() {
  local target="$1"
  local cmd="$2"
  [[ -z "$cmd" ]] && return
  tmux send-keys -t "$target" "$cmd" C-m
}

prepare_cmd() {
  local target="$1"
  local cmd="$2"

  tmux send-keys -t "$target" C-l
  sleep 0.2

  if [[ -n "$cmd" ]]; then
    tmux send-keys -t "$target" "$cmd"
  fi
}

create_layout_from_panes() {
  local select_window="$1"
  local select_pane="${2:-0}"
  local idx
  local window
  local pane_index
  local init_cmd
  local created_windows=" "

  if [[ "${#PANE_WINDOWS[@]}" -eq 0 ]]; then
    echo "No pane specs configured" >&2
    exit 1
  fi

  tmux new-session -d -x 250 -y 80 -s "$SESSION_NAME" -n "${PANE_WINDOWS[0]}"

  for window in "${PANE_WINDOWS[@]}"; do
    if [[ "$created_windows" == *" $window "* ]]; then
      continue
    fi

    if [[ "$window" != "${PANE_WINDOWS[0]}" ]]; then
      tmux new-window -t "$SESSION_NAME:" -n "$window"
    fi

    created_windows="$created_windows$window "
  done

  created_windows=" "
  for window in "${PANE_WINDOWS[@]}"; do
    if [[ "$created_windows" == *" $window "* ]]; then
      continue
    fi

    create_window_panes "$window"
    created_windows="$created_windows$window "
  done

  for idx in "${!PANE_WINDOWS[@]}"; do
    window="${PANE_WINDOWS[$idx]}"
    pane_index="$(pane_index_for_spec "$idx")"

    init_cmd="$(build_init_cmd "${PANE_DIRS[$idx]}" "${PANE_SETUPS[$idx]}")"
    init_pane "$SESSION_NAME":"$window"."$pane_index" "$init_cmd"
  done

  sleep 2.0

  for idx in "${!PANE_WINDOWS[@]}"; do
    window="${PANE_WINDOWS[$idx]}"
    pane_index="$(pane_index_for_spec "$idx")"

    prepare_cmd "$SESSION_NAME":"$window"."$pane_index" "${PANE_PREPARES[$idx]}"
  done

  if window_exists_in_specs "$select_window"; then
    tmux select-window -t "$SESSION_NAME":"$select_window"
    tmux select-pane -t "$SESSION_NAME":"$select_window"."$select_pane"
  fi
}

