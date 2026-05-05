#!/bin/bash

echo "=== Interactive SCP checkpoint transfer (multi-select, 2-level dirs, multi-IP) ==="

# ==========================================
# Defaults
# ==========================================
DEFAULT_BASE_DIR="./ckpts/train/"
DEFAULT_REMOTE_USER="tamiya"
DEFAULT_REMOTE_IPS=("10.42.0.1" "192.168.55.1" "192.168.11.190")
DEFAULT_REMOTE_DIR="/home/tamiya/workspace/tamiya-systems/python_ws/ckpts/pilotnet_trajectory/"
# ==========================================

read -p "Base directory (Enter for '${DEFAULT_BASE_DIR}'): " BASE_DIR
BASE_DIR=${BASE_DIR:-$DEFAULT_BASE_DIR}
BASE_DIR="${BASE_DIR%/}/"

if [ ! -d "$BASE_DIR" ]; then
    echo "Error: directory '$BASE_DIR' was not found."
    exit 1
fi

IFS=$'\n' read -r -d '' -a dirs < <(find "$BASE_DIR" -mindepth 2 -maxdepth 2 -type d -print0 | sort -z)

if [ ${#dirs[@]} -eq 0 ]; then
    echo "Error: no second-level checkpoint directories were found in '$BASE_DIR'."
    exit 1
fi

echo ""
echo "Select directories to transfer:"
i=1
for d in "${dirs[@]}"; do
    rel_path="${d#$BASE_DIR}"
    echo "  $i) $rel_path"
    ((i++))
done
echo ""

read -p "Enter numbers separated by spaces (example: 1 3 4): " DIR_CHOICES

if [ -z "$DIR_CHOICES" ]; then
    echo "Error: no directory was selected."
    exit 1
fi

SELECTED_DIRS=()
for choice in $DIR_CHOICES; do
    if [[ "$choice" =~ ^[0-9]+$ ]] && [ "$choice" -ge 1 ] && [ "$choice" -le "${#dirs[@]}" ]; then
        SELECTED_DIRS+=("${dirs[$((choice-1))]}")
    else
        echo "Warning: invalid selection '$choice' was skipped."
    fi
done

if [ ${#SELECTED_DIRS[@]} -eq 0 ]; then
    echo "Error: no valid directories were selected."
    exit 1
fi

echo ""
read -p "Remote user (Enter for '${DEFAULT_REMOTE_USER}'): " REMOTE_USER
REMOTE_USER=${REMOTE_USER:-$DEFAULT_REMOTE_USER}

echo ""
echo "Select a remote IP address, or type one directly:"
i=1
for ip in "${DEFAULT_REMOTE_IPS[@]}"; do
    if [ $i -eq 1 ]; then
        echo "  $i) $ip (default)"
    else
        echo "  $i) $ip"
    fi
    ((i++))
done
echo ""

read -p "Number or IP (Enter for '${DEFAULT_REMOTE_IPS[0]}'): " IP_CHOICE

if [ -z "$IP_CHOICE" ]; then
    REMOTE_IP="${DEFAULT_REMOTE_IPS[0]}"
elif [[ "$IP_CHOICE" =~ ^[0-9]+$ ]] && [ "$IP_CHOICE" -ge 1 ] && [ "$IP_CHOICE" -le "${#DEFAULT_REMOTE_IPS[@]}" ]; then
    REMOTE_IP="${DEFAULT_REMOTE_IPS[$((IP_CHOICE-1))]}"
else
    REMOTE_IP="$IP_CHOICE"
fi

echo ""
read -p "Remote directory (Enter for '${DEFAULT_REMOTE_DIR}'): " REMOTE_DIR
REMOTE_DIR=${REMOTE_DIR:-$DEFAULT_REMOTE_DIR}

echo ""
echo "================ Transfer Summary ================"
echo "Source directories (${#SELECTED_DIRS[@]}):"
for d in "${SELECTED_DIRS[@]}"; do
    echo "  - ${d#$BASE_DIR}"
done
echo "Destination: ${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"
echo "=================================================="

read -p "Start transfer? (Y/n, Enter for yes): " CONFIRM
CONFIRM=${CONFIRM:-y}

if [[ "$CONFIRM" =~ ^[Yy]$ ]]; then
    echo "Starting transfer..."
    scp -r "${SELECTED_DIRS[@]}" "${REMOTE_USER}@${REMOTE_IP}:${REMOTE_DIR}"

    if [ $? -eq 0 ]; then
        echo "Transfer completed successfully."
    else
        echo "Transfer failed."
    fi
else
    echo "Transfer cancelled."
fi
