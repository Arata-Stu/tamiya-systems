#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
PIP_BIN="${PIP_BIN:-pip3}"
OPTIMIZER_ROOT="${SCRIPT_DIR}/global_racetrajectory_optimization"
FORCE_COMPAT=false
FORCE_MODERN=false
NO_FALLBACK=false
CHECK_LOG_FILE=""

usage() {
    cat <<'EOF'
Usage: bash setup_global_opt_env.sh [options]

Install and verify the optional global raceline optimizer dependencies.
If the standard install hits the known quadprog ABI issue, this script can
retry with the compatibility workaround automatically.

Options:
  --optimizer-root PATH  Path to global_racetrajectory_optimization checkout
  --force-compat         Skip the default install and apply the quadprog 0.1.6 workaround
  --force-modern         Skip other paths and install the upstream master stack
  --no-fallback          Fail immediately if the default install check fails
  -h, --help             Show this help
EOF
}

cleanup() {
    if [ -n "${CHECK_LOG_FILE}" ] && [ -f "${CHECK_LOG_FILE}" ]; then
        rm -f "${CHECK_LOG_FILE}"
    fi
}

run_check() {
    cleanup
    CHECK_LOG_FILE="$(mktemp)"
    set +e
    "${PYTHON_BIN}" "${SCRIPT_DIR}/data_analysis/check_global_opt_env.py" \
        --optimizer-root "${OPTIMIZER_ROOT}" >"${CHECK_LOG_FILE}" 2>&1
    local status=$?
    set -e
    cat "${CHECK_LOG_FILE}"
    return "${status}"
}

install_default_stack() {
    "${PIP_BIN}" install -r "${SCRIPT_DIR}/requirements_global_opt.txt"
}

install_compat_stack() {
    "${PIP_BIN}" uninstall -y trajectory-planning-helpers quadprog || true
    "${PIP_BIN}" install --no-cache-dir "Cython<3" wheel
    if ! "${PIP_BIN}" install --no-cache-dir --no-build-isolation "quadprog==0.1.6"; then
        cat >&2 <<'EOF'
[global-opt] Failed to build quadprog==0.1.6.
If the error mentions Python.h or missing compiler headers, install python3-dev/build-essential in the image first.
EOF
        return 1
    fi
    "${PIP_BIN}" install --no-cache-dir "trajectory-planning-helpers==0.79" --no-deps
}

install_modern_stack() {
    "${PIP_BIN}" uninstall -y trajectory-planning-helpers quadprog || true
    "${PIP_BIN}" install --no-cache-dir "quadprog>=0.1.11"
    "${PIP_BIN}" install --no-cache-dir \
        "git+https://github.com/TUMFTM/trajectory_planning_helpers.git@master" --no-deps
}

trap cleanup EXIT

while [ "$#" -gt 0 ]; do
    case "$1" in
        --optimizer-root)
            if [ "$#" -lt 2 ]; then
                echo "Missing value for --optimizer-root" >&2
                exit 2
            fi
            OPTIMIZER_ROOT="$2"
            shift 2
            ;;
        --force-compat)
            FORCE_COMPAT=true
            shift
            ;;
        --force-modern)
            FORCE_MODERN=true
            shift
            ;;
        --no-fallback)
            NO_FALLBACK=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [ ! -d "${OPTIMIZER_ROOT}" ]; then
    echo "optimizer root not found: ${OPTIMIZER_ROOT}" >&2
    exit 1
fi

if [ "${FORCE_COMPAT}" = true ]; then
    echo "[global-opt] Installing compatibility stack" >&2
    install_compat_stack
    run_check
    exit $?
fi

if [ "${FORCE_MODERN}" = true ]; then
    echo "[global-opt] Installing upstream master stack" >&2
    install_modern_stack
    run_check
    exit $?
fi

echo "[global-opt] Installing default dependency set" >&2
install_default_stack
if run_check; then
    exit 0
fi

if [ "${NO_FALLBACK}" = true ]; then
    echo "[global-opt] Dependency check failed and fallback is disabled." >&2
    exit 1
fi

if grep -q "quadprog" "${CHECK_LOG_FILE}" && grep -q "undefined symbol" "${CHECK_LOG_FILE}"; then
    echo "[global-opt] Detected quadprog ABI mismatch; retrying compatibility install" >&2
    install_compat_stack
    if run_check; then
        exit 0
    fi

    if grep -q "quadprog" "${CHECK_LOG_FILE}" && grep -q "undefined symbol" "${CHECK_LOG_FILE}"; then
        echo "[global-opt] Compatibility stack still fails; trying upstream master stack" >&2
        install_modern_stack
        run_check
        exit $?
    fi

    exit 1
fi

echo "[global-opt] Dependency check failed for a reason other than the known quadprog ABI issue." >&2
exit 1
