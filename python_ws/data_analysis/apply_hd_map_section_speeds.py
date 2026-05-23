#!/usr/bin/env python3
"""Apply HD map section speed overrides to a raceline CSV."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml


def detect_delimiter(line: str) -> str:
    counts = {",": line.count(","), ";": line.count(";"), "\t": line.count("\t")}
    return max(counts, key=counts.get) if max(counts.values()) > 0 else ","


def parse_float(token: str) -> Optional[float]:
    try:
        return float(token)
    except ValueError:
        return None


def read_raceline_csv(path: Path) -> tuple[list[str], list[list[float]], str, bool]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Raceline CSV is empty: {path}")

    delimiter = detect_delimiter(lines[0].lstrip("#").strip())
    header: list[str] = []
    rows: list[list[float]] = []
    header_was_comment = False

    for raw in lines:
        is_comment = raw.startswith("#")
        content = raw[1:].strip() if is_comment else raw
        tokens = [token.strip() for token in content.split(delimiter)]
        numeric = [parse_float(token) for token in tokens]
        if any(value is None for value in numeric):
            if not header:
                header = tokens
                header_was_comment = is_comment
            continue
        if is_comment:
            continue
        rows.append([float(value) for value in numeric if value is not None])

    if not rows:
        raise RuntimeError(f"Raceline CSV has no numeric rows: {path}")
    if not header:
        header = default_header_for_column_count(len(rows[0]))
    return header, rows, delimiter, header_was_comment


def default_header_for_column_count(column_count: int) -> list[str]:
    if column_count >= 7:
        return ["s_m", "x_m", "y_m", "psi_rad", "kappa_radpm", "vx_mps", "ax_mps2"][:column_count]
    if column_count == 3:
        return ["x_m", "y_m", "vx_mps"]
    return ["x_m", "y_m"] + [f"col_{i}" for i in range(2, column_count)]


def normalized_header_index(header: Sequence[str]) -> dict[str, int]:
    return {name.strip().lower(): index for index, name in enumerate(header)}


def find_index(header: Sequence[str], names: Sequence[str], fallback: int = -1) -> int:
    index = normalized_header_index(header)
    for name in names:
        if name.lower() in index:
            return index[name.lower()]
    return fallback


def ensure_column(header: list[str], rows: list[list[float]], name: str, default: float) -> int:
    index = find_index(header, [name], -1)
    if index >= 0:
        return index
    header.append(name)
    for row in rows:
        row.append(default)
    return len(header) - 1


def ensure_s_column(header: list[str], rows: list[list[float]]) -> int:
    s_index = find_index(header, ["s_m", "s"], -1)
    if s_index >= 0:
        return s_index
    x_index = find_index(header, ["x_m", "x"], 0)
    y_index = find_index(header, ["y_m", "y"], 1)
    if x_index < 0 or y_index < 0:
        raise RuntimeError("Cannot infer x/y columns to compute s_m.")
    header.insert(0, "s_m")
    cumulative = 0.0
    prev_x = rows[0][x_index]
    prev_y = rows[0][y_index]
    for row_index, row in enumerate(rows):
        if row_index > 0:
            cumulative += math.hypot(row[x_index] - prev_x, row[y_index] - prev_y)
            prev_x = row[x_index]
            prev_y = row[y_index]
        row.insert(0, cumulative)
    return 0


def load_speed_sections(hd_map_yaml: Path, lane_id: str) -> list[dict[str, Any]]:
    data = yaml.safe_load(hd_map_yaml.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise RuntimeError(f"HD map YAML root must be a mapping: {hd_map_yaml}")
    sections = []
    for raw in data.get("sections", []):
        if not isinstance(raw, dict):
            continue
        speed = raw.get("speed_override_mps")
        if speed is None or speed == "":
            continue
        section_lane = str(raw.get("lane_id", ""))
        if lane_id and section_lane and section_lane != lane_id:
            continue
        try:
            sections.append(
                {
                    "id": str(raw.get("id", "section")),
                    "lane_id": section_lane,
                    "start": float(raw["start_s_m"]),
                    "end": float(raw["end_s_m"]),
                    "speed": float(speed),
                    "wrap": bool(raw.get("wrap", False)),
                }
            )
        except (KeyError, TypeError, ValueError):
            continue
    return sections


def section_contains_s(section: dict[str, Any], s_value: float) -> bool:
    start = float(section["start"])
    end = float(section["end"])
    if bool(section.get("wrap", False)) or start > end:
        return s_value >= start or s_value < end
    return start <= s_value < end


def apply_overrides(rows: list[list[float]], s_index: int, speed_index: int, sections: Sequence[dict[str, Any]]) -> int:
    changed = 0
    for row in rows:
        s_value = row[s_index]
        for section in sections:
            if section_contains_s(section, s_value):
                row[speed_index] = float(section["speed"])
                changed += 1
                break
    return changed


def write_raceline_csv(path: Path, header: Sequence[str], rows: Sequence[Sequence[float]], delimiter: str, header_was_comment: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        if header_was_comment:
            file.write("# " + delimiter.join(header) + "\n")
            writer = csv.writer(file, delimiter=delimiter)
        else:
            writer = csv.writer(file, delimiter=delimiter)
            writer.writerow(header)
        for row in rows:
            writer.writerow([f"{value:.9g}" for value in row])


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raceline", required=True, help="Input raceline CSV.")
    parser.add_argument("--hd-map", required=True, help="HD map YAML with sections.")
    parser.add_argument("--output", default="", help="Output CSV. Default: <raceline_stem>_section_speeds.csv")
    parser.add_argument("--lane-id", default="", help="Only apply sections for this lane_id. Empty accepts all sections.")
    parser.add_argument("--in-place", action="store_true", help="Overwrite --raceline.")
    return parser


def main() -> int:
    args = build_arg_parser().parse_args()
    raceline_path = Path(args.raceline).expanduser().resolve()
    hd_map_path = Path(args.hd_map).expanduser().resolve()
    output_path = (
        raceline_path
        if args.in_place
        else Path(args.output).expanduser().resolve()
        if args.output
        else raceline_path.with_name(f"{raceline_path.stem}_section_speeds.csv")
    )

    header, rows, delimiter, header_was_comment = read_raceline_csv(raceline_path)
    s_index = ensure_s_column(header, rows)
    speed_index = ensure_column(header, rows, "vx_mps", 0.0)
    sections = load_speed_sections(hd_map_path, str(args.lane_id))
    changed = apply_overrides(rows, s_index, speed_index, sections)
    write_raceline_csv(output_path, header, rows, delimiter, header_was_comment)
    print(f"[INFO] section_speed_overrides={len(sections)} changed_rows={changed} output={output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
