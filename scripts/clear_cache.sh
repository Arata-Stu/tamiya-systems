#!/usr/bin/env bash
set -euo pipefail

confirm=true
do_swap=false
aggressive=false

for arg in "$@"; do
  case "$arg" in
    --yes|-y) confirm=false ;;
    --swap) do_swap=true ;;
    --aggressive) aggressive=true ;;
  esac
done

get_mem_info() {
  free | awk '/Mem:/ {used=$3; free=$4; total=$2; print used, free, total}'
}

read used_before free_before total_before < <(get_mem_info)
perc_before=$((100 * free_before / total_before))

echo "=== Before ==="
free -h
echo "Free: $((free_before/1024)) MB (${perc_before}%)"
echo

if $confirm; then
  read -r -p "Proceed? [y/N]: " ans
  case "${ans:-N}" in y|Y) ;; *) exit 0 ;; esac
fi

sync
$aggressive && sync
printf '3\n' | sudo tee /proc/sys/vm/drop_caches > /dev/null

if $do_swap; then
  if swapon --show | awk 'NR>1{exit 0} END{exit 1}'; then
    echo "No active swap, skip."
  else
    sudo swapoff -a
    sudo swapon -a
  fi
fi

read used_after free_after total_after < <(get_mem_info)
perc_after=$((100 * free_after / total_after))

freed_kb=$((free_after - free_before))
freed_mb=$((freed_kb / 1024))
diff_perc=$((perc_after - perc_before))

echo
echo "=== After ==="
free -h
echo "Freed: ${freed_mb} MB (+${diff_perc}%)"
echo "Free: $((free_after/1024)) MB (${perc_after}%)"
