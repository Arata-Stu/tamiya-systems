from pathlib import Path
from typing import Dict, Optional, Tuple

import yaml
from omegaconf import DictConfig, OmegaConf


def _to_plain_dict(cfg_obj) -> Dict:
    if cfg_obj is None:
        return {}
    if isinstance(cfg_obj, DictConfig):
        return OmegaConf.to_container(cfg_obj, resolve=True) or {}
    if isinstance(cfg_obj, dict):
        return dict(cfg_obj)
    return {}


def _load_yaml_dict(path: Path) -> Dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(f"Vehicle file must contain a mapping: {path}")
    return data


def _is_legacy_vehicle_selector(cfg_dict: Dict) -> bool:
    legacy_keys = {"enabled", "name", "path", "dir", "params"}
    return any(k in cfg_dict for k in legacy_keys)


def _resolve_vehicle_file(name: str, project_root: Path, cfg_dir: Optional[str] = None) -> Path:
    candidates = []

    # Optional explicit directory from config.
    if cfg_dir:
        base = Path(cfg_dir)
        if not base.is_absolute():
            base = (project_root / base).resolve()
        candidates.append(base)

    # Built-in fallback locations (support both spellings).
    candidates.append((project_root / "config" / "vehicle").resolve())
    candidates.append((project_root / "config" / "vechile").resolve())

    raw = Path(name)
    has_ext = raw.suffix.lower() in {".yaml", ".yml"}

    for base in candidates:
        if has_ext:
            p = (base / raw).resolve()
            if p.exists():
                return p
        else:
            p_yaml = (base / f"{name}.yaml").resolve()
            if p_yaml.exists():
                return p_yaml
            p_yml = (base / f"{name}.yml").resolve()
            if p_yml.exists():
                return p_yml

    searched = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Vehicle preset '{name}' not found. Searched: {searched}")


def resolve_vehicle_params(cfg: DictConfig, project_root: Path) -> Tuple[Optional[Dict], str]:
    """Resolve vehicle params for F110JaxSimulator.

    Priority:
    1) Hydra vehicle group mapping (e.g. `vehicle=tamiya`)
    2) vehicle.path (explicit yaml path)
    3) vehicle.name (preset in config/vehicle or config/vechile)
    4) vehicle.params (inline override mapping)

    Returns:
      (params_dict_or_none, source_label)
    """
    vehicle_cfg = cfg.get("vehicle", None)
    if vehicle_cfg is None:
        # Backward compatibility for misspelled key.
        vehicle_cfg = cfg.get("vechile", None)

    if vehicle_cfg is None:
        return None, "default"

    cfg_dict = _to_plain_dict(vehicle_cfg)
    if not cfg_dict:
        return None, "default"

    # Hydra group mode: cfg.vehicle is directly a dynamics parameter mapping.
    # Example: defaults: [vehicle: traxxas], then override with `vehicle=tamiya`.
    if not _is_legacy_vehicle_selector(cfg_dict):
        return cfg_dict, "hydra.vehicle"

    enabled = cfg_dict.get("enabled", True)
    if enabled is False:
        return None, "default"

    source = "inline"
    params: Dict = {}
    legacy_keys = {"enabled", "name", "path", "dir", "params"}
    direct_params = {k: v for k, v in cfg_dict.items() if k not in legacy_keys}
    if direct_params:
        params.update(direct_params)
        source = "hydra.vehicle"

    path_value = cfg_dict.get("path", None)
    name_value = cfg_dict.get("name", None)
    dir_value = cfg_dict.get("dir", None)

    if path_value:
        path = Path(path_value)
        if not path.is_absolute():
            path = (project_root / path).resolve()
        if not path.exists():
            raise FileNotFoundError(f"vehicle.path not found: {path}")
        params.update(_load_yaml_dict(path))
        source = str(path)
    elif name_value:
        preset_path = _resolve_vehicle_file(str(name_value), project_root, cfg_dir=dir_value)
        params.update(_load_yaml_dict(preset_path))
        source = str(preset_path)

    inline_params = _to_plain_dict(cfg_dict.get("params", {}))
    if inline_params:
        params.update(inline_params)
        if source == "inline":
            source = "inline.params"
        else:
            source = f"{source} + inline.params"

    if not params:
        return None, "default"

    return params, source
