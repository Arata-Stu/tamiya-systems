#!/bin/bash

# Lightweight bash TUI helpers.
# Source this file from scripts that want checkbox-style option editing.

tui_clear_screen() {
  printf '\033[H\033[J'
}

tui_array_length() {
  local array_name="$1"
  eval "printf '%s\n' \"\${#${array_name}[@]}\""
}

tui_array_get() {
  local array_name="$1"
  local index="$2"
  eval "printf '%s\n' \"\${${array_name}[$index]}\""
}

tui_read_key() {
  local key
  local rest

  IFS= read -rsn1 key || return 1

  if [[ "$key" == $'\033' ]]; then
    IFS= read -rsn2 -t 0.1 rest || true
    key+="$rest"
  fi

  printf '%s' "$key"
}

tui_render_checkbox_menu() {
  local title="$1"
  local keys_array="$2"
  local get_value_func="$3"
  local extra_render_func="$4"
  local cursor="$5"
  local extra_menu_label="${6:-}"
  local set_value_label="${7:-}"
  local secondary_menu_label="${8:-}"
  local length
  local idx
  local key
  local value
  local marker
  local checked
  local help="j/k or ↑/↓: move  space: toggle"

  length="$(tui_array_length "$keys_array")"

  tui_clear_screen
  echo "$title"
  echo ""
  echo "Options:"

  for ((idx = 0; idx < length; idx++)); do
    key="$(tui_array_get "$keys_array" "$idx")"
    value="$("$get_value_func" "$key")"
    marker=" "
    checked=" "

    if [[ "$idx" -eq "$cursor" ]]; then
      marker=">"
    fi

    if [[ "$value" == "true" ]]; then
      checked="x"
    fi

    printf " %s [%s] %-34s %s\n" "$marker" "$checked" "$key" "$value"
  done

  if [[ -n "$extra_render_func" ]]; then
    echo ""
    "$extra_render_func"
  fi

  echo ""
  if [[ -n "$extra_menu_label" ]]; then
    help="$help  b: $extra_menu_label"
  fi
  if [[ -n "$set_value_label" ]]; then
    help="$help  s: $set_value_label"
  fi
  if [[ -n "$secondary_menu_label" ]]; then
    help="$help  m: $secondary_menu_label"
  fi
  echo "$help  enter: done  q: quit"
}

tui_checkbox_menu() {
  local title="$1"
  local keys_array="$2"
  local get_value_func="$3"
  local toggle_value_func="$4"
  local extra_render_func="${5:-}"
  local extra_menu_func="${6:-}"
  local set_value_func="${7:-}"
  local extra_menu_label="${8:-}"
  local set_value_label="${9:-}"
  local secondary_menu_func="${10:-}"
  local secondary_menu_label="${11:-}"
  local cursor=0
  local length
  local key
  local current

  length="$(tui_array_length "$keys_array")"
  if [[ "$length" -eq 0 ]]; then
    echo "No checkbox options configured" >&2
    return 1
  fi

  while true; do
    tui_render_checkbox_menu "$title" "$keys_array" "$get_value_func" "$extra_render_func" "$cursor" "$extra_menu_label" "$set_value_label" "$secondary_menu_label"
    key="$(tui_read_key)"

    case "$key" in
      ""|$'\n'|$'\r')
        tui_clear_screen
        return 0
        ;;
      q|Q)
        tui_clear_screen
        echo "Canceled." >&2
        return 130
        ;;
      j|$'\033[B')
        cursor=$(((cursor + 1) % length))
        ;;
      k|$'\033[A')
        cursor=$(((cursor + length - 1) % length))
        ;;
      " ")
        current="$(tui_array_get "$keys_array" "$cursor")"
        "$toggle_value_func" "$current"
        ;;
      b|B)
        if [[ -n "$extra_menu_func" ]]; then
          tui_clear_screen
          "$extra_menu_func"
        fi
        ;;
      s|S)
        if [[ -n "$set_value_func" ]]; then
          "$set_value_func"
        fi
        ;;
      m|M)
        if [[ -n "$secondary_menu_func" ]]; then
          tui_clear_screen
          "$secondary_menu_func"
        fi
        ;;
    esac
  done
}

TUI_PATH_SELECT_KEYS=()
TUI_PATH_SELECT_VALUES=()
TUI_PATH_SELECT_PATHS=()

tui_path_select_get_value() {
  local key="$1"
  local idx

  for idx in "${!TUI_PATH_SELECT_KEYS[@]}"; do
    if [[ "${TUI_PATH_SELECT_KEYS[$idx]}" == "$key" ]]; then
      echo "${TUI_PATH_SELECT_VALUES[$idx]}"
      return 0
    fi
  done

  echo "false"
}

tui_path_select_toggle_value() {
  local key="$1"
  local idx

  for idx in "${!TUI_PATH_SELECT_KEYS[@]}"; do
    if [[ "${TUI_PATH_SELECT_KEYS[$idx]}" == "$key" ]]; then
      if [[ "${TUI_PATH_SELECT_VALUES[$idx]}" == "true" ]]; then
        TUI_PATH_SELECT_VALUES[$idx]="false"
      else
        TUI_PATH_SELECT_VALUES[$idx]="true"
      fi
      return 0
    fi
  done
}

tui_path_relative_label() {
  local path="$1"
  local base="${2:-}"

  if [[ -n "$base" ]]; then
    base="${base%/}"
    if [[ "$path" == "$base"/* ]]; then
      printf '%s\n' "${path#"$base"/}"
      return 0
    fi
  fi

  printf '%s\n' "$path"
}

tui_select_paths() {
  local title="$1"
  local paths_array="$2"
  local output_array="$3"
  local base_dir="${4:-}"
  local length
  local idx
  local path
  local label
  local selected=()

  if [[ ! -t 0 ]]; then
    return 2
  fi

  length="$(tui_array_length "$paths_array")"
  if [[ "$length" -eq 0 ]]; then
    eval "$output_array=()"
    return 0
  fi

  TUI_PATH_SELECT_KEYS=()
  TUI_PATH_SELECT_VALUES=()
  TUI_PATH_SELECT_PATHS=()

  for ((idx = 0; idx < length; idx++)); do
    path="$(tui_array_get "$paths_array" "$idx")"
    label="$(tui_path_relative_label "$path" "$base_dir")"
    TUI_PATH_SELECT_KEYS+=("$(printf "%02d %s" "$((idx + 1))" "$label")")
    TUI_PATH_SELECT_VALUES+=("false")
    TUI_PATH_SELECT_PATHS+=("$path")
  done

  tui_checkbox_menu "$title" TUI_PATH_SELECT_KEYS tui_path_select_get_value tui_path_select_toggle_value

  for idx in "${!TUI_PATH_SELECT_VALUES[@]}"; do
    if [[ "${TUI_PATH_SELECT_VALUES[$idx]}" == "true" ]]; then
      selected+=("${TUI_PATH_SELECT_PATHS[$idx]}")
    fi
  done

  eval "$output_array=()"
  for path in "${selected[@]}"; do
    eval "$output_array+=(\"\$path\")"
  done
}
