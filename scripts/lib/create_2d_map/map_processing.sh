#!/bin/bash

prompt_map_name_interactive() {
    while :; do
        read -r -p "作成する map 名を入力してください: " MAP_NAME
        MAP_NAME="${MAP_NAME#"${MAP_NAME%%[![:space:]]*}"}"
        MAP_NAME="${MAP_NAME%"${MAP_NAME##*[![:space:]]}"}"
        if [ -z "${MAP_NAME}" ]; then
            echo "map名が空です。"
            continue
        fi
        if [[ "${MAP_NAME}" == *"/"* ]]; then
            echo "map名に '/' は使えません。"
            continue
        fi
        return 0
    done
}

convert_pgm_to_png() {
    local pgm_path="$1"
    local png_path="$2"

    if command -v magick >/dev/null 2>&1; then
        magick "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v convert >/dev/null 2>&1; then
        convert "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v ffmpeg >/dev/null 2>&1; then
        ffmpeg -y -loglevel error -i "${pgm_path}" "${png_path}"
        return $?
    fi

    if command -v python3 >/dev/null 2>&1; then
        python3 - "${pgm_path}" "${png_path}" <<'PY'
import sys

try:
    from PIL import Image
except Exception as exc:
    raise SystemExit(f"Pillow not available: {exc}")

src, dst = sys.argv[1], sys.argv[2]
Image.open(src).save(dst, format="PNG")
PY
        return $?
    fi

    return 1
}

update_yaml_image_path() {
    local yaml_path="$1"
    local image_path="$2"
    local tmp_yaml_path

    if [ ! -f "${yaml_path}" ]; then
        return 1
    fi

    tmp_yaml_path="${yaml_path}.tmp.$$"

    if ! awk -v image_path="${image_path}" '
        BEGIN { updated = 0 }
        /^[[:space:]]*image:[[:space:]]*/ && updated == 0 {
            match($0, /^[[:space:]]*/)
            indent = substr($0, 1, RLENGTH)
            print indent "image: " image_path
            updated = 1
            next
        }
        { print }
        END {
            if (updated == 0) {
                exit 2
            }
        }
    ' "${yaml_path}" > "${tmp_yaml_path}"; then
        rm -f "${tmp_yaml_path}" 2>/dev/null || true
        return 1
    fi

    mv "${tmp_yaml_path}" "${yaml_path}"
}

resolve_centerline_script() {
    if [ -n "${CENTERLINE_SCRIPT_PATH}" ]; then
        if [ -f "${CENTERLINE_SCRIPT_PATH}" ]; then
            echo "${CENTERLINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/generate_centerline.py"
}

resolve_raceline_script() {
    if [ -n "${RACELINE_SCRIPT_PATH}" ]; then
        if [ -f "${RACELINE_SCRIPT_PATH}" ]; then
            echo "${RACELINE_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/generate_raceline.py"
}

resolve_line_preview_script() {
    if [ -n "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
        if [ -f "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
            echo "${LINE_PREVIEW_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "data_analysis/visualize_race_lines.py"
}

resolve_map_edit_script() {
    if [ -n "${MAP_EDIT_SCRIPT_PATH}" ]; then
        if [ -f "${MAP_EDIT_SCRIPT_PATH}" ]; then
            echo "${MAP_EDIT_SCRIPT_PATH}"
            return 0
        fi
        return 1
    fi

    resolve_python_ws_file "map_section_editor/map_cleanup_editor.py"
}

resolve_section_editor_script() {
    resolve_python_ws_file "map_section_editor/section_editor.py"
}

prompt_map_edit() {
    local edit_choice

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        MAP_EDIT_ENABLED=false
        return 0
    fi

    if [ "${VSLAM_LANDMARK_TRACE_COMPLETED}" = true ]; then
        MAP_EDIT_ENABLED=false
        return 0
    fi

    case "${MAP_EDIT_MODE}" in
        always)
            MAP_EDIT_ENABLED=true
            ;;
        never)
            MAP_EDIT_ENABLED=false
            ;;
        auto)
            MAP_EDIT_ENABLED=false
            if [ ! -t 0 ]; then
                return 0
            fi
            echo ""
            read -r -p "centerline前に map を手修正しますか？ (y/N, Enterでスキップ): " edit_choice
            if [[ "${edit_choice:-n}" =~ ^[Yy]$ ]]; then
                MAP_EDIT_ENABLED=true
            fi
            ;;
        *)
            echo "Invalid --map-edit-mode: ${MAP_EDIT_MODE}" >&2
            exit 1
            ;;
    esac
}

run_map_edit() {
    local input_map_path="$1"
    local map_edit_script_path
    local -a map_edit_cmd

    if [ "${MAP_EDIT_ENABLED}" != true ]; then
        echo "[prep] Skip GUI map cleanup"
        return 0
    fi

    echo "[prep] Launch GUI map cleanup"

    if [ ! -f "${input_map_path}" ]; then
        echo "Warning: map input not found for cleanup: ${input_map_path}" >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip GUI map cleanup." >&2
        return 0
    fi

    if ! map_edit_script_path="$(resolve_map_edit_script)"; then
        if [ -n "${MAP_EDIT_SCRIPT_PATH}" ]; then
            echo "Warning: map cleanup editor not found: ${MAP_EDIT_SCRIPT_PATH}" >&2
        else
            echo "Warning: map_cleanup_editor.py not found. Skip GUI map cleanup." >&2
        fi
        return 0
    fi

    if [ -z "${MAP_EDIT_OUTPUT_PATH}" ]; then
        MAP_EDIT_OUTPUT_PATH="${MAP_STEM}_centerline_input.png"
    fi

    map_edit_cmd=(
        python3 "${map_edit_script_path}"
        --input "${input_map_path}"
        --output "${MAP_EDIT_OUTPUT_PATH}"
    )

    if ! "${map_edit_cmd[@]}"; then
        echo "Warning: map cleanup editor failed. Keep original map for centerline." >&2
        return 0
    fi

    if [ -f "${MAP_EDIT_OUTPUT_PATH}" ]; then
        CENTERLINE_INPUT_MAP="${MAP_EDIT_OUTPUT_PATH}"
        echo "  - cleaned map: ${MAP_EDIT_OUTPUT_PATH}"
    else
        echo "Warning: cleaned map was not saved. Keep original map for centerline." >&2
    fi
}

run_section_edit() {
    local section_editor_script_path
    local -a section_editor_cmd

    echo "[post] Launch section editor"

    if [ ! -f "${MAP_YAML_PATH}" ]; then
        echo "Warning: map yaml not found for section edit: ${MAP_YAML_PATH}" >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip section edit." >&2
        return 0
    fi

    if ! section_editor_script_path="$(resolve_section_editor_script)"; then
        echo "Warning: section_editor.py not found. Skip section edit." >&2
        return 0
    fi

    section_editor_cmd=(
        python3 "${section_editor_script_path}"
        --map-yaml "${MAP_YAML_PATH}"
        --output "${SECTION_OUTPUT_PATH}"
    )

    if ! "${section_editor_cmd[@]}"; then
        echo "Warning: section editor failed." >&2
        return 0
    fi

    if [ -f "${SECTION_OUTPUT_PATH}" ]; then
        echo "  - sections: ${SECTION_OUTPUT_PATH}"
    else
        echo "Warning: section CSV was not saved." >&2
    fi

    if [ -f "${SECTION_GATE_OUTPUT_PATH}" ]; then
        echo "  - gates: ${SECTION_GATE_OUTPUT_PATH}"
    fi
}

generate_centerline() {
    local input_map_path="$1"
    local centerline_script_path
    local -a centerline_cmd

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        echo "[5/8] Skip centerline generation"
        return 0
    fi

    echo "[5/8] Generate centerline"

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip centerline generation." >&2
        return 0
    fi

    if ! centerline_script_path="$(resolve_centerline_script)"; then
        if [ -n "${CENTERLINE_SCRIPT_PATH}" ]; then
            echo "Warning: centerline script not found: ${CENTERLINE_SCRIPT_PATH}" >&2
        else
            echo "Warning: generate_centerline.py not found. Skip centerline generation." >&2
        fi
        return 0
    fi

    if [ "${CENTERLINE_DEBUG}" = true ] && [ -z "${CENTERLINE_DEBUG_DIR}" ]; then
        CENTERLINE_DEBUG_PATH="${MAP_STEM}_centerline_debug"
    else
        CENTERLINE_DEBUG_PATH="${CENTERLINE_DEBUG_DIR}"
    fi

    centerline_cmd=(
        python3 "${centerline_script_path}"
        --map "${input_map_path}"
        --output "${CENTERLINE_OUTPUT_PATH}"
        --yaml "${MAP_YAML_PATH}"
        --preset "${CENTERLINE_PRESET}"
        --direction "${CENTERLINE_DIRECTION}"
    )
    if [ -n "${CENTERLINE_DEBUG_PATH}" ]; then
        centerline_cmd+=(--debug-dir "${CENTERLINE_DEBUG_PATH}")
    fi

    if ! "${centerline_cmd[@]}"; then
        echo "Warning: centerline generation failed. Skip centerline output." >&2
        return 0
    fi

    CENTERLINE_CREATED=true
    echo "  - ${CENTERLINE_OUTPUT_PATH}"
    if [ "${CENTERLINE_DIRECTION}" = "both" ]; then
        echo "  - ${CENTERLINE_OUTPUT_PATH%.*}_reverse.${CENTERLINE_OUTPUT_PATH##*.}"
    fi
    if [ -n "${CENTERLINE_DEBUG_PATH}" ]; then
        echo "  - ${CENTERLINE_DEBUG_PATH}/"
    fi
}

generate_raceline() {
    local centerline_path="$1"
    local raceline_script_path
    local -a raceline_cmd

    if [ "${ENABLE_RACELINE}" != true ]; then
        echo "[6/8] Skip raceline generation"
        return 0
    fi

    echo "[6/8] Generate raceline"

    if [ "${CENTERLINE_CREATED}" != true ] || [ ! -f "${centerline_path}" ]; then
        echo "Warning: centerline CSV not found. Skip raceline generation." >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip raceline generation." >&2
        return 0
    fi

    if ! raceline_script_path="$(resolve_raceline_script)"; then
        if [ -n "${RACELINE_SCRIPT_PATH}" ]; then
            echo "Warning: raceline script not found: ${RACELINE_SCRIPT_PATH}" >&2
        else
            echo "Warning: generate_raceline.py not found. Skip raceline generation." >&2
        fi
        return 0
    fi

    raceline_cmd=(
        python3 "${raceline_script_path}"
        --preset "${RACELINE_PRESET}"
        --backend "${RACELINE_BACKEND}"
        --opt-type "${RACELINE_OPT_TYPE}"
        --centerline "${centerline_path}"
        --output "${RACELINE_OUTPUT_PATH}"
        --direction "${RACELINE_DIRECTION}"
    )
    if [ -n "${GLOBAL_OPTIMIZER_ROOT}" ]; then
        raceline_cmd+=(--optimizer-root "${GLOBAL_OPTIMIZER_ROOT}")
    fi

    if ! "${raceline_cmd[@]}"; then
        echo "Warning: raceline generation failed. Skip raceline output." >&2
        return 0
    fi

    RACELINE_CREATED=true
    echo "  - ${RACELINE_OUTPUT_PATH}"
    if [ "${RACELINE_DIRECTION}" = "both" ]; then
        echo "  - ${RACELINE_OUTPUT_PATH%.*}_reverse.${RACELINE_OUTPUT_PATH##*.}"
    fi
}

generate_line_preview() {
    local input_map_path="$1"
    local preview_script_path
    local -a preview_cmd

    if [ "${ENABLE_LINE_PREVIEW}" != true ]; then
        echo "[7/8] Skip line preview generation"
        return 0
    fi

    echo "[7/8] Generate line preview"

    if [ "${CENTERLINE_CREATED}" != true ] && [ "${RACELINE_CREATED}" != true ]; then
        echo "Warning: no centerline/raceline CSV found. Skip line preview generation." >&2
        return 0
    fi

    if ! command -v python3 >/dev/null 2>&1; then
        echo "Warning: python3 not found. Skip line preview generation." >&2
        return 0
    fi

    if ! preview_script_path="$(resolve_line_preview_script)"; then
        if [ -n "${LINE_PREVIEW_SCRIPT_PATH}" ]; then
            echo "Warning: line preview script not found: ${LINE_PREVIEW_SCRIPT_PATH}" >&2
        else
            echo "Warning: visualize_race_lines.py not found. Skip line preview generation." >&2
        fi
        return 0
    fi

    preview_cmd=(
        python3 "${preview_script_path}"
        --map "${input_map_path}"
        --yaml "${MAP_YAML_PATH}"
        --output "${LINE_PREVIEW_OUTPUT_PATH}"
    )
    if [ "${CENTERLINE_CREATED}" = true ] && [ -f "${CENTERLINE_OUTPUT_PATH}" ]; then
        preview_cmd+=(--centerline "${CENTERLINE_OUTPUT_PATH}")
    fi
    if [ "${RACELINE_CREATED}" = true ] && [ -f "${RACELINE_OUTPUT_PATH}" ]; then
        preview_cmd+=(--raceline "${RACELINE_OUTPUT_PATH}")
    fi

    if ! "${preview_cmd[@]}"; then
        echo "Warning: line preview generation failed. Skip line preview output." >&2
        return 0
    fi

    LINE_PREVIEW_CREATED=true
    echo "  - ${LINE_PREVIEW_OUTPUT_PATH}"
}

prompt_centerline_generation() {
    local generate_choice
    local debug_choice

    if [ "${ENABLE_CENTERLINE}" != true ]; then
        return 0
    fi

    echo ""
    read -r -p "centerlineを生成しますか？ (Y/n, Enterで生成): " generate_choice
    generate_choice=${generate_choice:-y}

    if [[ ! "${generate_choice}" =~ ^[Yy]$ ]]; then
        ENABLE_CENTERLINE=false
        return 0
    fi

    echo ""
    read -r -p "debug画像も保存しますか？ (Y/n, Enterで保存): " debug_choice
    debug_choice=${debug_choice:-y}

    if [[ "${debug_choice}" =~ ^[Nn]$ ]]; then
        CENTERLINE_DEBUG=false
    else
        CENTERLINE_DEBUG=true
    fi
}

prompt_raceline_generation() {
    local generate_choice

    if [ "${ENABLE_CENTERLINE}" != true ] || [ "${ENABLE_RACELINE}" != true ]; then
        return 0
    fi

    echo ""
    read -r -p "racelineも生成しますか？ (Y/n, Enterで生成): " generate_choice
    generate_choice=${generate_choice:-y}

    if [[ ! "${generate_choice}" =~ ^[Yy]$ ]]; then
        ENABLE_RACELINE=false
    fi
}

convert_pbstream_to_map() {
    local success_label="$1"
    local png_filename
    local png_created=false
    local yaml_image_updated=false

    if ! ros2 run cartographer_ros cartographer_pbstream_to_ros_map \
        -pbstream_filename "${PBSTREAM_PATH}" \
        -map_filestem "${MAP_STEM}" \
        -resolution 0.05; then
        echo "pbstream generated: ${PBSTREAM_PATH}" >&2
        echo "Failed to convert pbstream to occupancy map." >&2
        return 1
    fi

    png_filename="$(basename "${MAP_PNG_PATH}")"
    if convert_pgm_to_png "${MAP_PGM_PATH}" "${MAP_PNG_PATH}"; then
        png_created=true
        if update_yaml_image_path "${MAP_YAML_PATH}" "${png_filename}"; then
            yaml_image_updated=true
        else
            echo "Warning: PNG was generated, but failed to update image path in ${MAP_YAML_PATH}." >&2
        fi
    else
        echo "Warning: PNG conversion skipped. (Need one of: magick/convert/ffmpeg/python3+Pillow)" >&2
    fi

    echo ""
    echo "${success_label}"
    echo "  - ${MAP_YAML_PATH}"
    if [ "${yaml_image_updated}" = true ]; then
        echo "    image: ${png_filename}"
    fi
    echo "  - ${MAP_PGM_PATH}"
    if [ "${png_created}" = true ]; then
        echo "  - ${MAP_PNG_PATH}"
    fi
    echo "  - ${PBSTREAM_PATH}"
    if [ -d "${VSLAM_MAP_DIR}" ]; then
        echo "  - ${VSLAM_MAP_DIR}/"
    fi
    if [ -f "${VSLAM_REFERENCE_SNAPSHOT_PATH}" ]; then
        echo "  - ${VSLAM_REFERENCE_SNAPSHOT_PATH}"
    fi
    if [ "${OFFLINE_ODOM_BAG_CREATED}" = true ]; then
        echo "  - ${OFFLINE_ODOM_BAG_DIR}/"
    fi
}

