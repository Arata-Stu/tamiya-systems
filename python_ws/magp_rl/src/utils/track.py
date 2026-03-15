from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np


def resolve_waypoints_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.is_absolute() and path.exists():
        return path

    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path

    project_root = Path(__file__).resolve().parents[3]
    root_path = (project_root / path).resolve()
    if root_path.exists():
        return root_path

    raise FileNotFoundError(f"Waypoints file not found: {path_str}")


def load_waypoints(path_str: str, default_speed_mps: float = 2.0):
    waypoint_path = resolve_waypoints_path(path_str)
    lines = waypoint_path.read_text(encoding="utf-8").splitlines()
    comment_lines = [ln.strip().lstrip("#").strip() for ln in lines if ln.strip().startswith("#")]
    header_line = next((ln for ln in comment_lines if "x_m" in ln and "y_m" in ln), "")

    first_data_line = next((ln.strip() for ln in lines if ln.strip() and not ln.strip().startswith("#")), "")
    delimiter = ";" if (";" in header_line or ";" in first_data_line) else ","

    data = np.genfromtxt(
        str(waypoint_path),
        delimiter=delimiter,
        comments="#",
        autostrip=True,
        dtype=np.float32,
    )

    if data.ndim == 1:
        data = data[None, :]
    data = data[~np.isnan(data).any(axis=1)]
    if data.shape[1] < 2:
        raise ValueError(f"Invalid waypoint format in: {waypoint_path}")

    columns = [c.strip() for c in header_line.split(delimiter)] if header_line else []
    has_s = "s_m" in columns

    speed = None

    if has_s:
        s_idx = columns.index("s_m")
        x_idx = columns.index("x_m")
        y_idx = columns.index("y_m")
        s = data[:, s_idx]
        xy = data[:, [x_idx, y_idx]]
        if "vx_mps" in columns:
            speed = data[:, columns.index("vx_mps")]
        elif "v_mps" in columns:
            speed = data[:, columns.index("v_mps")]
    else:
        xy = data[:, :2]
        diffs = np.diff(xy, axis=0)
        ds = np.linalg.norm(diffs, axis=1)
        s = np.concatenate([[0.0], np.cumsum(ds)]).astype(np.float32)

    if speed is None:
        if data.shape[1] >= 6:
            # Common format: x, y, ..., vx, ...
            speed = data[:, 5]
        else:
            speed = np.full((xy.shape[0],), float(default_speed_mps), dtype=np.float32)

    speed = np.asarray(speed, dtype=np.float32)
    speed = np.nan_to_num(speed, nan=float(default_speed_mps), posinf=float(default_speed_mps), neginf=0.0)
    speed = np.clip(speed, 0.0, None)

    return xy, s, speed


def project_to_centerline_s(points_xy, waypoints_xy, waypoints_s, track_length):
    seg_a = waypoints_xy
    seg_b = jnp.roll(waypoints_xy, shift=-1, axis=0)
    seg_ab = seg_b - seg_a
    seg_len_sq = jnp.sum(seg_ab * seg_ab, axis=1) + 1e-8

    s_a = waypoints_s
    s_b = jnp.roll(waypoints_s, shift=-1, axis=0)
    seg_ds = jnp.where(s_b >= s_a, s_b - s_a, (s_b + track_length) - s_a)

    def project_point(p):
        ap = p[None, :] - seg_a
        t = jnp.clip(jnp.sum(ap * seg_ab, axis=1) / seg_len_sq, 0.0, 1.0)
        proj = seg_a + t[:, None] * seg_ab
        dist_sq = jnp.sum((p[None, :] - proj) ** 2, axis=1)
        idx = jnp.argmin(dist_sq)
        return s_a[idx] + t[idx] * seg_ds[idx]

    return jax.vmap(project_point)(points_xy)


def compute_progress_delta(current_s, prev_s, track_length, progress_clip):
    raw_progress = current_s - prev_s
    half_track = track_length * 0.5
    progress = jnp.where(raw_progress > half_track, raw_progress - track_length, raw_progress)
    progress = jnp.where(progress < -half_track, progress + track_length, progress)
    progress = jnp.clip(progress, -progress_clip, progress_clip)
    return progress
