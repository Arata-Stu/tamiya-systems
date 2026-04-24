#!/usr/bin/env bash
set -euo pipefail

# ==============================================================================
# MAGP RL -> ONNX -> TensorRT -> Triton deployment helper
# ==============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

EXPORT_SCRIPT="${PROJECT_ROOT}/export_onnx.py"
CHECKPOINT_BASE_DIR="${PROJECT_ROOT}/ckpts/train"

AGENT="sac"
MODEL_NAME="magp_rl_policy"
TRITON_MODEL_REPO="/workspaces/isaac_ros_assets/models"
INPUT_ONNX_PATH=""
OUTPUT_ONNX_PATH=""
CHECKPOINT_DIR=""
STEP=""

OBS_DIM=320
ACTION_DIM=2
INPUT_LAYOUT="scan"          # scan | flat
SCAN_POINTS=""               # default: obs_dim
NORMALIZE_INPUT=true
MAX_LIDAR_RANGE=12.0
LIDAR_PROFILE="custom"       # custom | hokuyo | t_mini_plus
LIDAR_FOV_RAD=""
INPUT_NAME="scan_input"
OUTPUT_NAME="control_output"
SAC_OUTPUT="deterministic"

PRECISION="fp16"             # fp16 | fp32
MAX_BATCH_SIZE=1
TRTEXEC_BIN="${TRTEXEC_BIN:-}"
YES=false
FORCE_OBS_DIM_MISMATCH=false

OBS_DIM_SET=false
SCAN_POINTS_SET=false
MAX_LIDAR_RANGE_SET=false
LIDAR_FOV_SET=false

usage() {
  cat <<'EOF'
Usage:
  bash scripts/deploy_isaac_triton.sh [options]

Options:
  --onnx PATH                 Use existing ONNX and skip export_onnx.py
  --checkpoint-dir PATH       Checkpoint run dir or checkpoint_XXX dir
  --checkpoint-base PATH      Base dir to search runs (default: ./ckpts/train)
  --step N                    Checkpoint step for export_onnx.py
  --output-onnx PATH          Output ONNX path (default: auto in checkpoint run)
  --agent {sac|ppo|td3}       Agent type (default: sac)
  --model-name NAME           Triton model name (default: magp_rl_policy)
  --triton-model-repo PATH    Triton model repository root

  --obs-dim N                 Training obs dim (320 or 1080)
  --action-dim N              Action dim (default: 2)
  --input-layout {scan|flat}  ONNX input layout (default: scan)
  --scan-points N             Input scan points for scan layout
  --normalize-input           Enable in-graph clip/div normalization
  --no-normalize-input        Disable in-graph normalization
  --max-lidar-range F         Normalization divisor / clamp max
  --lidar-profile NAME        custom | hokuyo | t_mini_plus
  --lidar-fov-rad F           Optional metadata only (default from profile)
  --input-name NAME           ONNX/Triton input tensor name
  --output-name NAME          ONNX/Triton output tensor name
  --sac-output MODE           deterministic | mean_logstd | all (default: deterministic)

  --precision {fp16|fp32}     trtexec precision (default: fp16)
  --max-batch-size N          trtexec max batch (default: 1)
  --trtexec-bin PATH          trtexec binary path
  --yes                       Non-interactive (auto-select latest run/checkpoint)
  --force-obs-dim-mismatch    Allow proceed when inferred checkpoint obs_dim and --obs-dim differ
  -h, --help                  Show this help

Examples:
  # Hokuyo 1080
  bash scripts/deploy_isaac_triton.sh \
    --model-name magp_sac_hokuyo \
    --lidar-profile hokuyo \
    --input-layout scan --normalize-input \
    --input-name scan_input --output-name control_output \
    --yes

  # T-mini 320
  bash scripts/deploy_isaac_triton.sh \
    --model-name magp_sac_tmini \
    --lidar-profile t_mini_plus \
    --input-layout scan --normalize-input \
    --input-name scan_input --output-name control_output \
    --yes
EOF
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

infer_checkpoint_obs_dim_hint() {
  local ckpt_dir="$1"
  local agent="$2"
  local step="$3"

  local dense_in_dim=""
  dense_in_dim="$(
    python3 - "${EXPORT_SCRIPT}" "${ckpt_dir}" "${agent}" "${step}" <<'PY'
import importlib.util
import sys

export_script = sys.argv[1]
ckpt_dir = sys.argv[2]
agent = sys.argv[3]
step_raw = sys.argv[4]
step = int(step_raw) if step_raw else None

try:
    import jax  # noqa: F401
    from flax.training import checkpoints
except Exception:
    print("")
    raise SystemExit(0)

spec = importlib.util.spec_from_file_location("magp_rl_export_onnx", export_script)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

try:
    # Restore target shape does not need to match checkpoint exactly for reading params.
    target = mod._make_restore_target(agent, obs_dim=1080, action_dim=2)
    restored = checkpoints.restore_checkpoint(ckpt_dir=ckpt_dir, target=target, step=step)
    actor_tree = mod._extract_actor_param_tree(restored)

    if agent == "ppo":
        dense_keys = mod._sorted_layer_keys(actor_tree["actor_mlp"], "Dense")
        first_dense = actor_tree["actor_mlp"][dense_keys[0]]
    else:
        dense_keys = mod._sorted_layer_keys(actor_tree, "Dense")
        first_dense = actor_tree[dense_keys[0]]

    in_dim = int(mod._as_numpy(first_dense["kernel"]).shape[0])
    print(str(in_dim))
except Exception:
    print("")
PY
  )"
  dense_in_dim="$(echo "${dense_in_dim}" | tr -d '[:space:]')"

  case "${dense_in_dim}" in
    640) echo "320" ;;
    2176) echo "1080" ;;
    *) echo "" ;;
  esac
}

confirm_obs_dim_preflight() {
  local selected_ckpt_dir="$1"
  local inferred_obs_dim="$2"
  local effective_scan_points="${SCAN_POINTS:-${OBS_DIM}}"

  echo "---------------------------------------------------"
  echo "Preflight: obs_dim confirmation"
  echo "Checkpoint       : ${selected_ckpt_dir}"
  echo "Configured       : obs_dim=${OBS_DIM}, scan_points=${effective_scan_points}, lidar_profile=${LIDAR_PROFILE}"
  if [[ -n "${inferred_obs_dim}" ]]; then
    echo "Checkpoint hint  : obs_dim=${inferred_obs_dim}"
  else
    echo "Checkpoint hint  : unknown"
  fi

  if [[ -n "${inferred_obs_dim}" && "${inferred_obs_dim}" != "${OBS_DIM}" ]]; then
    local msg="checkpoint hint is obs_dim=${inferred_obs_dim}, but --obs-dim=${OBS_DIM}"
    if [[ "${YES}" == true && "${FORCE_OBS_DIM_MISMATCH}" == false ]]; then
      die "obs_dim mismatch (${msg}). Re-run with --obs-dim ${inferred_obs_dim} or --force-obs-dim-mismatch."
    fi
    if [[ "${YES}" == false ]]; then
      echo "WARNING: ${msg}"
      local ans_mismatch
      read -r -p "Continue anyway? [y/N]: " ans_mismatch
      [[ "${ans_mismatch}" =~ ^[Yy]$ ]] || die "Canceled by user."
    fi
  fi

  if [[ "${YES}" == false ]]; then
    [[ -t 0 ]] || die "Interactive confirmation requires TTY. Use --yes for non-interactive mode."

    local ans_obs
    read -r -p "Type '${OBS_DIM}' to confirm model obs_dim: " ans_obs
    [[ "${ans_obs}" == "${OBS_DIM}" ]] || die "obs_dim confirmation failed."

    if [[ "${effective_scan_points}" != "${OBS_DIM}" ]]; then
      local ans_ds
      read -r -p "scan_points(${effective_scan_points}) != obs_dim(${OBS_DIM}). Type 'downsample-ok' to continue: " ans_ds
      [[ "${ans_ds}" == "downsample-ok" ]] || die "Canceled (downsample not confirmed)."
    fi
  fi
}

find_trtexec() {
  if [[ -n "${TRTEXEC_BIN}" ]]; then
    [[ -x "${TRTEXEC_BIN}" ]] || die "trtexec not executable: ${TRTEXEC_BIN}"
    echo "${TRTEXEC_BIN}"
    return
  fi
  if command -v trtexec >/dev/null 2>&1; then
    command -v trtexec
    return
  fi
  if [[ -x "/usr/src/tensorrt/bin/trtexec" ]]; then
    echo "/usr/src/tensorrt/bin/trtexec"
    return
  fi
  die "trtexec not found. Use --trtexec-bin PATH."
}

list_run_dirs() {
  find "${CHECKPOINT_BASE_DIR}" -mindepth 2 -maxdepth 2 -type d | sort
}

latest_run_dir() {
  local run
  run="$(list_run_dirs | tail -n 1 || true)"
  [[ -n "${run}" ]] || die "No run directories found in ${CHECKPOINT_BASE_DIR}"
  echo "${run}"
}

latest_ckpt_in_run() {
  local run_dir="$1"
  local ckpt
  ckpt="$(find "${run_dir}" -mindepth 1 -maxdepth 1 -type d -name 'checkpoint_*' | sort -V | tail -n 1 || true)"
  [[ -n "${ckpt}" ]] || die "No checkpoint_* directories found in ${run_dir}"
  echo "${ckpt}"
}

select_run_dir_interactive() {
  mapfile -t runs < <(list_run_dirs)
  [[ ${#runs[@]} -gt 0 ]] || die "No run directories found in ${CHECKPOINT_BASE_DIR}"

  local options=()
  local i
  for ((i = 0; i < ${#runs[@]}; i++)); do
    options+=("${runs[$i]#${CHECKPOINT_BASE_DIR}/}")
  done
  options+=("Quit")

  echo "=== Select checkpoint run directory ===" >&2
  PS3="Enter number: "
  select opt in "${options[@]}"; do
    if [[ "${opt}" == "Quit" ]]; then
      exit 1
    fi
    if [[ -n "${opt}" ]]; then
      local idx=$((REPLY - 1))
      echo "${runs[$idx]}"
      return
    fi
  done
}

print_parameters() {
  local effective_scan_points="${SCAN_POINTS:-${OBS_DIM}}"
  echo "==================================================="
  echo "MAGP RL Deployment Configuration"
  echo "==================================================="
  echo "AGENT              : ${AGENT}"
  echo "MODEL_NAME         : ${MODEL_NAME}"
  echo "TRITON_MODEL_REPO  : ${TRITON_MODEL_REPO}"
  echo "INPUT_ONNX_PATH    : ${INPUT_ONNX_PATH}"
  echo "OBS_DIM            : ${OBS_DIM}"
  echo "ACTION_DIM         : ${ACTION_DIM}"
  echo "INPUT_LAYOUT       : ${INPUT_LAYOUT}"
  echo "SCAN_POINTS        : ${effective_scan_points}"
  echo "LIDAR_PROFILE      : ${LIDAR_PROFILE}"
  echo "LIDAR_FOV_RAD      : ${LIDAR_FOV_RAD:-n/a}"
  echo "NORMALIZE_INPUT    : ${NORMALIZE_INPUT}"
  echo "MAX_LIDAR_RANGE    : ${MAX_LIDAR_RANGE}"
  echo "INPUT_NAME         : ${INPUT_NAME}"
  echo "OUTPUT_NAME        : ${OUTPUT_NAME}"
  echo "PRECISION          : ${PRECISION}"
  echo "MAX_BATCH_SIZE     : ${MAX_BATCH_SIZE}"
  echo "==================================================="
}

apply_lidar_profile_defaults() {
  case "${LIDAR_PROFILE}" in
    custom)
      ;;
    hokuyo)
      if [[ "${OBS_DIM_SET}" == false ]]; then OBS_DIM=1080; fi
      if [[ "${SCAN_POINTS_SET}" == false ]]; then SCAN_POINTS=1080; fi
      if [[ "${MAX_LIDAR_RANGE_SET}" == false ]]; then MAX_LIDAR_RANGE=30.0; fi
      if [[ "${LIDAR_FOV_SET}" == false ]]; then LIDAR_FOV_RAD=4.7; fi
      ;;
    t_mini_plus)
      if [[ "${OBS_DIM_SET}" == false ]]; then OBS_DIM=320; fi
      if [[ "${SCAN_POINTS_SET}" == false ]]; then SCAN_POINTS=320; fi
      if [[ "${MAX_LIDAR_RANGE_SET}" == false ]]; then MAX_LIDAR_RANGE=12.0; fi
      if [[ "${LIDAR_FOV_SET}" == false ]]; then LIDAR_FOV_RAD=4.7; fi
      ;;
    *)
      die "Unsupported --lidar-profile: ${LIDAR_PROFILE}"
      ;;
  esac
}

write_triton_config() {
  local model_root="$1"
  local input_dims
  local output_dims
  local tensor_dtype="TYPE_FP32"
  output_dims="${ACTION_DIM}"

  if [[ "${PRECISION}" == "fp16" ]]; then
    tensor_dtype="TYPE_FP16"
  fi

  if [[ "${INPUT_LAYOUT}" == "scan" ]]; then
    local points="${SCAN_POINTS:-${OBS_DIM}}"
    input_dims="1, ${points}"
  else
    input_dims="${OBS_DIM}"
  fi

  cat > "${model_root}/config.pbtxt" <<EOF
name: "${MODEL_NAME}"
platform: "tensorrt_plan"
max_batch_size: ${MAX_BATCH_SIZE}
default_model_filename: "model.plan"
input [
  {
    name: "${INPUT_NAME}"
    data_type: ${tensor_dtype}
    dims: [ ${input_dims} ]
  }
]
output [
  {
    name: "${OUTPUT_NAME}"
    data_type: ${tensor_dtype}
    dims: [ ${output_dims} ]
  }
]
instance_group [
  {
    kind: KIND_GPU
    count: 1
  }
]
EOF
}

build_trt_engine() {
  local model_root="$1"
  local version_path="$2"
  local trtexec
  trtexec="$(find_trtexec)"

  [[ "${PRECISION}" == "fp16" || "${PRECISION}" == "fp32" ]] || die "Unsupported precision: ${PRECISION}"
  local precision_args=()
  if [[ "${PRECISION}" == "fp16" ]]; then
    precision_args+=(--fp16)
  fi

  local min_shape opt_shape max_shape
  if [[ "${INPUT_LAYOUT}" == "scan" ]]; then
    local points="${SCAN_POINTS:-${OBS_DIM}}"
    min_shape="${INPUT_NAME}:1x1x${points}"
    opt_shape="${INPUT_NAME}:1x1x${points}"
    max_shape="${INPUT_NAME}:${MAX_BATCH_SIZE}x1x${points}"
  else
    min_shape="${INPUT_NAME}:1x${OBS_DIM}"
    opt_shape="${INPUT_NAME}:1x${OBS_DIM}"
    max_shape="${INPUT_NAME}:${MAX_BATCH_SIZE}x${OBS_DIM}"
  fi

  "${trtexec}" \
    --onnx="${version_path}/model.onnx" \
    --saveEngine="${version_path}/model.plan" \
    --minShapes="${min_shape}" \
    --optShapes="${opt_shape}" \
    --maxShapes="${max_shape}" \
    "${precision_args[@]}" \
    --verbose

  write_triton_config "${model_root}"
}

export_onnx_if_needed() {
  if [[ -n "${INPUT_ONNX_PATH}" ]]; then
    [[ -f "${INPUT_ONNX_PATH}" ]] || die "ONNX file not found: ${INPUT_ONNX_PATH}"
    return
  fi

  [[ -f "${EXPORT_SCRIPT}" ]] || die "export_onnx.py not found: ${EXPORT_SCRIPT}"

  local selected_ckpt_dir=""
  local run_dir=""

  if [[ -n "${CHECKPOINT_DIR}" ]]; then
    if [[ "$(basename "${CHECKPOINT_DIR}")" == checkpoint_* ]]; then
      selected_ckpt_dir="${CHECKPOINT_DIR}"
      run_dir="$(dirname "${CHECKPOINT_DIR}")"
    else
      run_dir="${CHECKPOINT_DIR}"
      selected_ckpt_dir="$(latest_ckpt_in_run "${run_dir}")"
    fi
  else
    if [[ "${YES}" == true ]]; then
      run_dir="$(latest_run_dir)"
    else
      run_dir="$(select_run_dir_interactive)"
    fi
    selected_ckpt_dir="$(latest_ckpt_in_run "${run_dir}")"
  fi

  [[ -d "${selected_ckpt_dir}" ]] || die "Checkpoint directory not found: ${selected_ckpt_dir}"

  local inferred_obs_dim=""
  inferred_obs_dim="$(infer_checkpoint_obs_dim_hint "${selected_ckpt_dir}" "${AGENT}" "${STEP}")"
  confirm_obs_dim_preflight "${selected_ckpt_dir}" "${inferred_obs_dim}"

  if [[ -z "${OUTPUT_ONNX_PATH}" ]]; then
    OUTPUT_ONNX_PATH="${run_dir}/${MODEL_NAME}.onnx"
  fi
  mkdir -p "$(dirname "${OUTPUT_ONNX_PATH}")"

  local cmd=(
    python3 "${EXPORT_SCRIPT}"
    --agent "${AGENT}"
    --checkpoint-dir "${selected_ckpt_dir}"
    --output "${OUTPUT_ONNX_PATH}"
    --lidar-profile "${LIDAR_PROFILE}"
    --obs-dim "${OBS_DIM}"
    --action-dim "${ACTION_DIM}"
    --input-layout "${INPUT_LAYOUT}"
    --max-lidar-range "${MAX_LIDAR_RANGE}"
    --input-name "${INPUT_NAME}"
    --output-name "${OUTPUT_NAME}"
    --sac-output "${SAC_OUTPUT}"
  )

  if [[ -n "${STEP}" ]]; then
    cmd+=(--step "${STEP}")
  fi
  if [[ "${INPUT_LAYOUT}" == "scan" ]]; then
    cmd+=(--scan-points "${SCAN_POINTS:-${OBS_DIM}}")
  fi
  if [[ -n "${LIDAR_FOV_RAD}" ]]; then
    cmd+=(--lidar-fov-rad "${LIDAR_FOV_RAD}")
  fi
  if [[ "${NORMALIZE_INPUT}" == true ]]; then
    cmd+=(--normalize-input)
  fi

  echo "Exporting ONNX from checkpoint: ${selected_ckpt_dir}"
  "${cmd[@]}"
  INPUT_ONNX_PATH="${OUTPUT_ONNX_PATH}"
}

deploy_to_triton() {
  [[ -f "${INPUT_ONNX_PATH}" ]] || die "ONNX file not found: ${INPUT_ONNX_PATH}"

  local model_root="${TRITON_MODEL_REPO}/${MODEL_NAME}"
  mkdir -p "${model_root}"

  local version=1
  while [[ -d "${model_root}/${version}" ]]; do
    version=$((version + 1))
  done

  local version_path="${model_root}/${version}"
  mkdir -p "${version_path}"

  echo "Deploy destination: ${version_path}"
  cp "${INPUT_ONNX_PATH}" "${version_path}/model.onnx"

  echo "Building TensorRT engine (.plan)..."
  build_trt_engine "${model_root}" "${version_path}"

  echo "==================================================="
  echo "Deployment complete"
  echo "Model name  : ${MODEL_NAME}"
  echo "Version     : ${version}"
  echo "Model root  : ${model_root}"
  echo "ONNX        : ${version_path}/model.onnx"
  echo "TensorRT    : ${version_path}/model.plan"
  echo "Triton conf : ${model_root}/config.pbtxt"
  echo "==================================================="
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --onnx) INPUT_ONNX_PATH="$2"; shift 2 ;;
    --checkpoint-dir) CHECKPOINT_DIR="$2"; shift 2 ;;
    --checkpoint-base) CHECKPOINT_BASE_DIR="$2"; shift 2 ;;
    --step) STEP="$2"; shift 2 ;;
    --output-onnx) OUTPUT_ONNX_PATH="$2"; shift 2 ;;
    --agent) AGENT="$2"; shift 2 ;;
    --model-name) MODEL_NAME="$2"; shift 2 ;;
    --triton-model-repo) TRITON_MODEL_REPO="$2"; shift 2 ;;
    --obs-dim) OBS_DIM="$2"; OBS_DIM_SET=true; shift 2 ;;
    --action-dim) ACTION_DIM="$2"; shift 2 ;;
    --input-layout) INPUT_LAYOUT="$2"; shift 2 ;;
    --scan-points) SCAN_POINTS="$2"; SCAN_POINTS_SET=true; shift 2 ;;
    --normalize-input) NORMALIZE_INPUT=true; shift ;;
    --no-normalize-input) NORMALIZE_INPUT=false; shift ;;
    --max-lidar-range) MAX_LIDAR_RANGE="$2"; MAX_LIDAR_RANGE_SET=true; shift 2 ;;
    --lidar-profile) LIDAR_PROFILE="$2"; shift 2 ;;
    --lidar-fov-rad) LIDAR_FOV_RAD="$2"; LIDAR_FOV_SET=true; shift 2 ;;
    --input-name) INPUT_NAME="$2"; shift 2 ;;
    --output-name) OUTPUT_NAME="$2"; shift 2 ;;
    --sac-output) SAC_OUTPUT="$2"; shift 2 ;;
    --precision) PRECISION="$2"; shift 2 ;;
    --max-batch-size) MAX_BATCH_SIZE="$2"; shift 2 ;;
    --trtexec-bin) TRTEXEC_BIN="$2"; shift 2 ;;
    --yes) YES=true; shift ;;
    --force-obs-dim-mismatch) FORCE_OBS_DIM_MISMATCH=true; shift ;;
    -h|--help) usage; exit 0 ;;
    *) die "Unknown option: $1 (use --help)" ;;
  esac
done

[[ "${AGENT}" == "sac" || "${AGENT}" == "ppo" || "${AGENT}" == "td3" ]] || die "agent must be sac, ppo, or td3"
[[ "${INPUT_LAYOUT}" == "scan" || "${INPUT_LAYOUT}" == "flat" ]] || die "input-layout must be scan or flat"
[[ "${LIDAR_PROFILE}" == "custom" || "${LIDAR_PROFILE}" == "hokuyo" || "${LIDAR_PROFILE}" == "t_mini_plus" ]] || die "invalid --lidar-profile"
[[ "${SAC_OUTPUT}" == "deterministic" || "${SAC_OUTPUT}" == "mean_logstd" || "${SAC_OUTPUT}" == "all" ]] || die "invalid --sac-output"
if [[ "${AGENT}" == "sac" ]]; then
  [[ "${SAC_OUTPUT}" == "deterministic" ]] || die "Deployment expects deterministic control output. Use --sac-output deterministic."
fi

apply_lidar_profile_defaults
export_onnx_if_needed
print_parameters
deploy_to_triton
