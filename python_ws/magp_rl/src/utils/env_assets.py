from pathlib import Path

from omegaconf import DictConfig


def resolve_existing_path(path_str: str, base_dir: Path) -> Path:
    candidate = Path(path_str)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    cwd_path = (Path.cwd() / candidate).resolve()
    if cwd_path.exists():
        return cwd_path

    base_path = (base_dir / candidate).resolve()
    if base_path.exists():
        return base_path

    raise FileNotFoundError(f"Path not found: {path_str}")


def resolve_env_assets(env_cfg: DictConfig, base_dir: Path):
    map_ext = env_cfg.map_ext
    map_path = env_cfg.get("map_path", None)
    waypoints_path = env_cfg.get("waypoints_path", None)

    track_cfg = env_cfg.get("track", None)
    if track_cfg is not None and track_cfg.get("name", None):
        track_root = resolve_existing_path(track_cfg.root, base_dir)
        track_dir = track_root / track_cfg.name
        if not track_dir.exists():
            raise FileNotFoundError(f"Track directory not found: {track_dir}")

        if not map_path:
            map_candidates = sorted(track_dir.glob("*_map.yaml"))
            if not map_candidates:
                raise FileNotFoundError(f"No *_map.yaml found in: {track_dir}")
            map_path = str(map_candidates[0])

        if not waypoints_path:
            line_type = track_cfg.get("line_type", "centerline")
            line_candidates = sorted(track_dir.glob(f"*_{line_type}.csv"))
            if not line_candidates:
                raise FileNotFoundError(
                    f"No *_{line_type}.csv found in: {track_dir}. "
                    "Use env.track.line_type=centerline|raceline or set env.waypoints_path."
                )
            waypoints_path = str(line_candidates[0])

    if not map_path or not waypoints_path:
        raise ValueError(
            "Either set env.map_path + env.waypoints_path, "
            "or set env.track.{root,name,line_type}."
        )

    map_path = str(resolve_existing_path(map_path, base_dir))
    waypoints_path = str(resolve_existing_path(waypoints_path, base_dir))
    return map_path, map_ext, waypoints_path
