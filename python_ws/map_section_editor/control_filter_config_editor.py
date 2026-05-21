#!/usr/bin/env python3
"""
Interactive editor for section-based control_filter configs.

Loads section names from sections_pixels.csv, lets you assign a class per
section, edits class/default filter parameters, and writes a YAML file that
control_filter_node can load directly.
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional

try:
    import yaml  # type: ignore
except ModuleNotFoundError:
    yaml = None


FILTER_CHOICES = ("slew_rate", "average", "none")
KNOWN_PARAM_KEYS = (
    "section_names",
    "section_classes",
    "filter_type",
    "window_size",
    "max_speed_slew_rate",
    "max_steer_slew_rate",
    "use_scale_filter",
    "straight_steer_threshold",
    "straight_speed_scale_ratio",
    "cornering_speed_scale_ratio",
    "steer_scale_ratio",
)


@dataclass
class FilterProfile:
    filter_type: str = "slew_rate"
    window_size: int = 5
    max_speed_slew_rate: float = 2.0
    max_steer_slew_rate: float = 1.5
    use_scale_filter: bool = True
    straight_steer_threshold: float = 0.20
    straight_speed_scale_ratio: float = 1.0
    cornering_speed_scale_ratio: float = 0.6
    steer_scale_ratio: float = 1.0


@dataclass
class EditorState:
    sections: list[str]
    assignments: dict[str, str]
    default_profile: FilterProfile
    class_profiles: dict[str, FilterProfile]
    output_path: Path
    source_config_path: Optional[Path]


def clear_screen() -> None:
    print("\033[H\033[J", end="")


def sanitize_class_name(name: str) -> str:
    cleaned = name.strip().replace(",", "_").replace(" ", "_")
    allowed: list[str] = []
    for ch in cleaned:
        if ch.isalnum() or ch in ("_", "-", "."):
            allowed.append(ch)
    return "".join(allowed)


def quote_yaml_string(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text if text else "0"
    return quote_yaml_string(str(value))


def load_sections_csv(path: Path) -> list[str]:
    sections: list[str] = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row:
                continue
            key = row[0].strip()
            if not key or key.startswith("#") or key != "section":
                continue
            if len(row) >= 2:
                name = row[1].strip()
                if name:
                    sections.append(name)
    if not sections:
        raise ValueError(f"no sections found in {path}")
    return sections


def profile_from_mapping(mapping: dict[str, Any], fallback: Optional[FilterProfile] = None) -> FilterProfile:
    base = replace(fallback) if fallback is not None else FilterProfile()
    if "filter_type" in mapping:
        base.filter_type = str(mapping["filter_type"])
    if "window_size" in mapping:
        base.window_size = int(mapping["window_size"])
    if "max_speed_slew_rate" in mapping:
        base.max_speed_slew_rate = float(mapping["max_speed_slew_rate"])
    if "max_steer_slew_rate" in mapping:
        base.max_steer_slew_rate = float(mapping["max_steer_slew_rate"])
    if "use_scale_filter" in mapping:
        base.use_scale_filter = bool(mapping["use_scale_filter"])
    if "straight_steer_threshold" in mapping:
        base.straight_steer_threshold = float(mapping["straight_steer_threshold"])
    if "straight_speed_scale_ratio" in mapping:
        base.straight_speed_scale_ratio = float(mapping["straight_speed_scale_ratio"])
    if "cornering_speed_scale_ratio" in mapping:
        base.cornering_speed_scale_ratio = float(mapping["cornering_speed_scale_ratio"])
    if "steer_scale_ratio" in mapping:
        base.steer_scale_ratio = float(mapping["steer_scale_ratio"])
    return base


def load_existing_config(path: Optional[Path]) -> tuple[FilterProfile, dict[str, str], dict[str, FilterProfile]]:
    if path is None or not path.is_file():
        return FilterProfile(), {}, {}
    if yaml is None:
        print(
            f"warning: PyYAML unavailable, skipping existing config load: {path}",
            file=sys.stderr,
        )
        return FilterProfile(), {}, {}

    with path.open("r", encoding="utf-8") as f:
        root = yaml.safe_load(f) or {}

    if not isinstance(root, dict):
        return FilterProfile(), {}, {}

    params = root.get("/**", {})
    if isinstance(params, dict):
        params = params.get("ros__parameters", {})
    if not isinstance(params, dict):
        return FilterProfile(), {}, {}

    default_profile = profile_from_mapping(params)
    section_names = [str(x) for x in params.get("section_names", []) if str(x).strip()]
    section_classes = [str(x) for x in params.get("section_classes", []) if str(x).strip()]
    assignments: dict[str, str] = {}
    for section_name, class_name in zip(section_names, section_classes):
        if class_name and class_name != "default":
            assignments[section_name] = class_name

    class_profiles: dict[str, FilterProfile] = {}
    for key, value in params.items():
        if key in KNOWN_PARAM_KEYS:
            continue
        if not isinstance(key, str) or not isinstance(value, dict):
            continue
        class_profiles[key] = profile_from_mapping(value, default_profile)

    for class_name in assignments.values():
        class_profiles.setdefault(class_name, replace(default_profile))

    return default_profile, assignments, class_profiles


def assigned_classes_in_order(state: EditorState) -> list[str]:
    ordered: list[str] = []
    for section_name in state.sections:
        class_name = state.assignments.get(section_name, "")
        if not class_name or class_name == "default":
            continue
        if class_name not in ordered:
            ordered.append(class_name)
    return ordered


def build_yaml_text(state: EditorState) -> str:
    mapped_sections: list[str] = []
    mapped_classes: list[str] = []
    for section_name in state.sections:
        class_name = state.assignments.get(section_name, "")
        if not class_name or class_name == "default":
            continue
        mapped_sections.append(section_name)
        mapped_classes.append(class_name)

    lines = [
        "/**:",
        "  ros__parameters:",
    ]

    if mapped_sections:
        lines.append("    section_names:")
        for section_name in mapped_sections:
            lines.append(f"      - {quote_yaml_string(section_name)}")
        lines.append("    section_classes:")
        for class_name in mapped_classes:
            lines.append(f"      - {quote_yaml_string(class_name)}")
    else:
        lines.append("    section_names: []")
        lines.append("    section_classes: []")

    lines.extend(
        [
            f"    filter_type: {yaml_scalar(state.default_profile.filter_type)}",
            f"    window_size: {yaml_scalar(state.default_profile.window_size)}",
            f"    max_speed_slew_rate: {yaml_scalar(state.default_profile.max_speed_slew_rate)}",
            f"    max_steer_slew_rate: {yaml_scalar(state.default_profile.max_steer_slew_rate)}",
            f"    use_scale_filter: {yaml_scalar(state.default_profile.use_scale_filter)}",
            f"    straight_steer_threshold: {yaml_scalar(state.default_profile.straight_steer_threshold)}",
            f"    straight_speed_scale_ratio: {yaml_scalar(state.default_profile.straight_speed_scale_ratio)}",
            f"    cornering_speed_scale_ratio: {yaml_scalar(state.default_profile.cornering_speed_scale_ratio)}",
            f"    steer_scale_ratio: {yaml_scalar(state.default_profile.steer_scale_ratio)}",
        ]
    )

    for class_name in assigned_classes_in_order(state):
        profile = state.class_profiles.get(class_name, replace(state.default_profile))
        lines.append(f"    {class_name}:")
        lines.extend(
            [
                f"      filter_type: {yaml_scalar(profile.filter_type)}",
                f"      window_size: {yaml_scalar(profile.window_size)}",
                f"      max_speed_slew_rate: {yaml_scalar(profile.max_speed_slew_rate)}",
                f"      max_steer_slew_rate: {yaml_scalar(profile.max_steer_slew_rate)}",
                f"      use_scale_filter: {yaml_scalar(profile.use_scale_filter)}",
                f"      straight_steer_threshold: {yaml_scalar(profile.straight_steer_threshold)}",
                f"      straight_speed_scale_ratio: {yaml_scalar(profile.straight_speed_scale_ratio)}",
                f"      cornering_speed_scale_ratio: {yaml_scalar(profile.cornering_speed_scale_ratio)}",
                f"      steer_scale_ratio: {yaml_scalar(profile.steer_scale_ratio)}",
            ]
        )

    return "\n".join(lines) + "\n"


def print_summary(state: EditorState) -> None:
    clear_screen()
    used_classes = assigned_classes_in_order(state)
    mapped = sum(1 for section_name in state.sections if state.assignments.get(section_name, ""))
    print("Section Control Filter Config Editor")
    print("")
    print(f"sections : {len(state.sections)}")
    print(f"mapped   : {mapped}")
    print(f"classes  : {', '.join(used_classes) if used_classes else '(default only)'}")
    print(f"output   : {state.output_path}")
    if state.source_config_path is not None:
        print(f"base     : {state.source_config_path}")
    print("")
    print("Sections:")
    for index, section_name in enumerate(state.sections, start=1):
        class_name = state.assignments.get(section_name, "") or "default"
        print(f"  {index:>2}) {section_name:<24} -> {class_name}")
    print("")
    print("Commands:")
    print("  a : assign class to sections")
    print("  c : edit class parameters")
    print("  d : edit default parameters")
    print("  p : preview YAML")
    print("  s : save")
    print("  q : quit")


def parse_section_selection(text: str, sections: list[str]) -> list[str]:
    raw = text.strip()
    if not raw or raw.lower() in ("all", "*"):
        return list(sections)

    selected: list[str] = []
    seen: set[str] = set()
    for token in raw.split(","):
        piece = token.strip()
        if not piece:
            continue
        if "-" in piece and piece.replace("-", "").isdigit():
            start_text, end_text = piece.split("-", 1)
            start = int(start_text)
            end = int(end_text)
            if start > end:
                start, end = end, start
            for idx in range(start, end + 1):
                if 1 <= idx <= len(sections):
                    name = sections[idx - 1]
                    if name not in seen:
                        seen.add(name)
                        selected.append(name)
            continue
        if piece.isdigit():
            idx = int(piece)
            if 1 <= idx <= len(sections):
                name = sections[idx - 1]
                if name not in seen:
                    seen.add(name)
                    selected.append(name)
            continue
        if piece in sections and piece not in seen:
            seen.add(piece)
            selected.append(piece)
    return selected


def prompt_string(label: str, current: str, *, choices: Optional[tuple[str, ...]] = None) -> str:
    suffix = f" [{'/'.join(choices)}]" if choices else ""
    while True:
        value = input(f"{label}{suffix} [{current}]: ").strip()
        if not value:
            return current
        if choices is not None and value not in choices:
            print(f"choose one of: {', '.join(choices)}")
            continue
        return value


def prompt_int(label: str, current: int) -> int:
    while True:
        value = input(f"{label} [{current}]: ").strip()
        if not value:
            return current
        try:
            return int(value)
        except ValueError:
            print("enter an integer")


def prompt_float(label: str, current: float) -> float:
    while True:
        value = input(f"{label} [{current}]: ").strip()
        if not value:
            return current
        try:
            return float(value)
        except ValueError:
            print("enter a number")


def prompt_bool(label: str, current: bool) -> bool:
    current_text = "y" if current else "n"
    while True:
        value = input(f"{label} [y/n, current={current_text}]: ").strip().lower()
        if not value:
            return current
        if value in ("y", "yes", "true", "1", "on"):
            return True
        if value in ("n", "no", "false", "0", "off"):
            return False
        print("enter y or n")


def edit_profile(name: str, profile: FilterProfile) -> FilterProfile:
    clear_screen()
    print(f"Edit profile: {name}")
    print("Press Enter to keep the current value.")
    print("")
    updated = replace(profile)
    updated.filter_type = prompt_string("filter_type", updated.filter_type, choices=FILTER_CHOICES)
    updated.window_size = prompt_int("window_size", updated.window_size)
    updated.max_speed_slew_rate = prompt_float("max_speed_slew_rate", updated.max_speed_slew_rate)
    updated.max_steer_slew_rate = prompt_float("max_steer_slew_rate", updated.max_steer_slew_rate)
    updated.use_scale_filter = prompt_bool("use_scale_filter", updated.use_scale_filter)
    updated.straight_steer_threshold = prompt_float("straight_steer_threshold", updated.straight_steer_threshold)
    updated.straight_speed_scale_ratio = prompt_float("straight_speed_scale_ratio", updated.straight_speed_scale_ratio)
    updated.cornering_speed_scale_ratio = prompt_float("cornering_speed_scale_ratio", updated.cornering_speed_scale_ratio)
    updated.steer_scale_ratio = prompt_float("steer_scale_ratio", updated.steer_scale_ratio)
    return updated


def assign_sections_interactive(state: EditorState) -> None:
    clear_screen()
    print("Assign class to sections")
    print("")
    for index, section_name in enumerate(state.sections, start=1):
        class_name = state.assignments.get(section_name, "") or "default"
        print(f"  {index:>2}) {section_name:<24} -> {class_name}")
    print("")
    print("Examples: 1-4,7   or   section_01,section_05   or   all")
    selected = parse_section_selection(input("sections: "), state.sections)
    if not selected:
        input("No sections selected. Press Enter to continue.")
        return

    current_classes = assigned_classes_in_order(state)
    if current_classes:
        print("")
        print("Existing classes:", ", ".join(current_classes))
    class_input = input("class name (empty or 'default' to clear mapping): ").strip()
    if not class_input or class_input == "default":
        for section_name in selected:
            state.assignments.pop(section_name, None)
        input(f"Cleared mapping for {len(selected)} section(s). Press Enter to continue.")
        return

    class_name = sanitize_class_name(class_input)
    if not class_name:
        input("Invalid class name. Press Enter to continue.")
        return

    if class_name not in state.class_profiles:
        state.class_profiles[class_name] = replace(state.default_profile)

    for section_name in selected:
        state.assignments[section_name] = class_name

    input(f"Assigned class '{class_name}' to {len(selected)} section(s). Press Enter to continue.")


def choose_class_name(state: EditorState) -> Optional[str]:
    used_classes = assigned_classes_in_order(state)
    clear_screen()
    print("Edit class parameters")
    print("")
    if used_classes:
        for index, class_name in enumerate(used_classes, start=1):
            print(f"  {index:>2}) {class_name}")
    else:
        print("No classes are currently assigned to sections.")
    print("")
    print("Type a class name to create/edit it, or Enter to cancel.")
    raw = input("class: ").strip()
    if not raw:
        return None
    if raw.isdigit():
        idx = int(raw)
        if 1 <= idx <= len(used_classes):
            return used_classes[idx - 1]
        return None
    return sanitize_class_name(raw)


def edit_class_interactive(state: EditorState) -> None:
    class_name = choose_class_name(state)
    if class_name is None:
        return
    if not class_name:
        input("Invalid class name. Press Enter to continue.")
        return
    if class_name not in state.class_profiles:
        state.class_profiles[class_name] = replace(state.default_profile)
    state.class_profiles[class_name] = edit_profile(class_name, state.class_profiles[class_name])
    if class_name not in state.assignments.values():
        clear_screen()
        print(f"Class '{class_name}' was edited but is not assigned to any section yet.")
        print("control_filter loads class blocks only when section_classes references them.")
        print("")
        assign = input("Assign this class to some sections now? [y/N]: ").strip().lower()
        if assign in ("y", "yes"):
            assign_sections_interactive(state)


def preview_yaml(state: EditorState) -> None:
    clear_screen()
    print(f"Preview: {state.output_path}")
    print("")
    print(build_yaml_text(state), end="")
    print("")
    input("Press Enter to continue.")


def save_yaml(state: EditorState) -> None:
    text = build_yaml_text(state)
    state.output_path.parent.mkdir(parents=True, exist_ok=True)
    state.output_path.write_text(text, encoding="utf-8")
    clear_screen()
    print(f"Saved: {state.output_path}")
    print("")
    print("Use with control_filter:")
    print(f"  ros2 launch control_filter control_filter.launch.xml control_filter_param:={state.output_path}")
    print("")
    input("Press Enter to continue.")


def run_editor(state: EditorState) -> int:
    while True:
        print_summary(state)
        command = input("> ").strip().lower()
        if command == "a":
            assign_sections_interactive(state)
        elif command == "c":
            edit_class_interactive(state)
        elif command == "d":
            state.default_profile = edit_profile("default", state.default_profile)
        elif command == "p":
            preview_yaml(state)
        elif command == "s":
            save_yaml(state)
        elif command == "q":
            return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sections-csv", required=True, help="path to sections_pixels.csv")
    parser.add_argument("--output", required=True, help="output control_filter.param.yaml path")
    parser.add_argument(
        "--base-config",
        default="",
        help="existing control_filter yaml used as the starting point; defaults to output file when it exists",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sections_csv = Path(args.sections_csv).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()
    base_config = Path(args.base_config).expanduser().resolve() if args.base_config else None
    if output_path.is_file():
        source_config_path = output_path
    else:
        source_config_path = base_config

    sections = load_sections_csv(sections_csv)
    default_profile, assignments, class_profiles = load_existing_config(source_config_path)
    for class_name in assignments.values():
        class_profiles.setdefault(class_name, replace(default_profile))

    state = EditorState(
        sections=sections,
        assignments={section_name: assignments.get(section_name, "") for section_name in sections},
        default_profile=default_profile,
        class_profiles=class_profiles,
        output_path=output_path,
        source_config_path=source_config_path,
    )
    try:
        return run_editor(state)
    except KeyboardInterrupt:
        print("")
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
