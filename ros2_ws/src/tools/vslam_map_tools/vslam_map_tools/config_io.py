from __future__ import annotations

from pathlib import Path
from typing import Any


ALIGNMENT_KEYS = (
    "parent_frame",
    "child_frame",
    "x",
    "y",
    "z",
    "roll_rad",
    "pitch_rad",
    "yaw_rad",
)


def _parse_scalar(text: str) -> Any:
    stripped = text.strip()
    if not stripped:
        return stripped

    if stripped[0] in ("'", '"') and stripped[-1] == stripped[0]:
        return stripped[1:-1]

    lowered = stripped.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        if any(ch in stripped for ch in (".", "e", "E")):
            return float(stripped)
        return int(stripped)
    except ValueError:
        return stripped


def load_alignment_config(path: str | Path) -> dict[str, Any]:
    config_path = Path(path).expanduser().resolve()
    data: dict[str, Any] = {}
    if not config_path.is_file():
        return data

    for raw_line in config_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key not in ALIGNMENT_KEYS:
            continue
        data[key] = _parse_scalar(value)

    return data


def _yaml_scalar(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, str):
        return f'"{value}"'
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(value)


def save_alignment_config(path: str | Path, values: dict[str, Any]) -> Path:
    config_path = Path(path).expanduser().resolve()
    config_path.parent.mkdir(parents=True, exist_ok=True)

    lines = ["/**:", "  ros__parameters:"]
    for key in ALIGNMENT_KEYS:
        if key in values:
            lines.append(f"    {key}: {_yaml_scalar(values[key])}")

    config_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return config_path
