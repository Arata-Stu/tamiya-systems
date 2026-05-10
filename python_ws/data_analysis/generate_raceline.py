#!/usr/bin/env python3
"""
Lightweight raceline generation from a centerline CSV.

The output follows the common F1TENTH raceline layout:
  s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2

This is intentionally dependency-light. It does not try to reproduce the full
race_stack/global_racetrajectory_optimization pipeline; instead it creates a
bounded minimum-curvature-style line that can be used as a practical first
raceline and as a stable CLI boundary for a future optimizer backend.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Tuple

import numpy as np


def circular_gaussian_smooth(values: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0 or len(values) < 5:
        return values

    radius = max(1, int(round(3.0 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()

    padded = np.concatenate([values[-radius:], values, values[:radius]])
    return np.convolve(padded, kernel, mode="same")[radius:-radius]


def smooth_points(points: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0.0 or len(points) < 5:
        return points
    xs = circular_gaussian_smooth(points[:, 0], sigma=sigma)
    ys = circular_gaussian_smooth(points[:, 1], sigma=sigma)
    return np.column_stack([xs, ys])


def remove_duplicate_points(points: np.ndarray, *extra_columns: np.ndarray) -> Tuple[np.ndarray, ...]:
    if len(points) < 2:
        return (points, *extra_columns)

    keep = np.ones(len(points), dtype=bool)
    keep[1:] = np.linalg.norm(np.diff(points, axis=0), axis=1) > 1e-9
    if np.linalg.norm(points[0] - points[-1]) <= 1e-9:
        keep[-1] = False

    return (points[keep], *(col[keep] for col in extra_columns))


def cumulative_s(points: np.ndarray, closed: bool = True) -> Tuple[np.ndarray, float]:
    pts = points
    if closed:
        pts = np.vstack([points, points[0]])

    seg_len = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg_len)])
    return s[:-1] if closed else s, float(s[-1])


def resample_closed_columns(values: np.ndarray, source_s: np.ndarray, target_s: np.ndarray, total: float) -> np.ndarray:
    source = np.concatenate([source_s, [total]])
    if values.ndim == 1:
        data = np.concatenate([values, [values[0]]])
        return np.interp(target_s, source, data)

    cols = []
    for i in range(values.shape[1]):
        data = np.concatenate([values[:, i], [values[0, i]]])
        cols.append(np.interp(target_s, source, data))
    return np.column_stack(cols)


def resample_centerline(centerline: np.ndarray, widths: np.ndarray, spacing: float) -> Tuple[np.ndarray, np.ndarray]:
    centerline, widths = remove_duplicate_points(centerline, widths)
    source_s, total = cumulative_s(centerline, closed=True)
    if total <= 1e-9:
        raise RuntimeError("Centerline length is too small.")

    n_samples = max(int(round(total / spacing)), 16)
    target_s = np.linspace(0.0, total, n_samples, endpoint=False)
    points = resample_closed_columns(centerline, source_s, target_s, total)
    width_samples = resample_closed_columns(widths, source_s, target_s, total)
    return points, width_samples


def heading_and_curvature(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    delta = next_pts - prev_pts
    psi = np.mod(np.arctan2(delta[:, 1], delta[:, 0]), 2.0 * np.pi)

    a = points - prev_pts
    b = next_pts - points
    c = next_pts - prev_pts
    la = np.linalg.norm(a, axis=1)
    lb = np.linalg.norm(b, axis=1)
    lc = np.linalg.norm(c, axis=1)
    cross = a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]
    denom = la * lb * lc

    kappa = np.zeros(len(points), dtype=np.float64)
    valid = denom > 1e-9
    kappa[valid] = 2.0 * cross[valid] / denom[valid]
    return psi, kappa


def centerline_normals(points: np.ndarray) -> np.ndarray:
    prev_pts = np.roll(points, 1, axis=0)
    next_pts = np.roll(points, -1, axis=0)
    tangent = next_pts - prev_pts
    norm = np.linalg.norm(tangent, axis=1)
    norm[norm < 1e-9] = 1.0
    tangent = tangent / norm[:, None]
    # Positive offset is to the left of the ordered centerline.
    return np.column_stack([-tangent[:, 1], tangent[:, 0]])


def load_centerline(path: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.genfromtxt(path, delimiter=",", comments="#", dtype=np.float64)
    if data.ndim == 1:
        data = data.reshape(1, -1)
    if data.shape[1] < 4:
        raise RuntimeError("Centerline CSV must contain x, y, w_tr_right, w_tr_left columns.")

    points = data[:, :2].astype(np.float64)
    widths = data[:, 2:4].astype(np.float64)
    if len(points) < 8:
        raise RuntimeError("Centerline needs at least 8 points to create a raceline.")
    return points, widths


def default_optimizer_root() -> str:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.abspath(os.path.join(script_dir, "..", "global_racetrajectory_optimization")),
        os.path.abspath(os.path.join(os.getcwd(), "tmp", "global_racetrajectory_optimization")),
        "/workspaces/tamiya-systems/python_ws/global_racetrajectory_optimization",
    ]
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def generate_raceline(
    centerline: np.ndarray,
    widths: np.ndarray,
    spacing: float,
    margin: float,
    lateral_aggression: float,
    offset_smooth_sigma: float,
    point_smooth_sigma: float,
    max_speed: float,
    min_speed: float,
    lateral_accel_limit: float,
    accel_limit: float,
    decel_limit: float,
) -> np.ndarray:
    centerline, widths = resample_centerline(centerline, widths, spacing=spacing)
    centerline = smooth_points(centerline, sigma=point_smooth_sigma)

    _, center_kappa = heading_and_curvature(centerline)
    abs_max_kappa = max(float(np.max(np.abs(center_kappa))), 1e-9)
    normals = centerline_normals(centerline)

    right_limit = np.maximum(widths[:, 0] - margin, 0.0)
    left_limit = np.maximum(widths[:, 1] - margin, 0.0)

    target_offset = np.zeros(len(centerline), dtype=np.float64)
    curve_strength = np.abs(center_kappa) / abs_max_kappa
    curve_strength = circular_gaussian_smooth(curve_strength, sigma=offset_smooth_sigma)
    turn_sign = np.sign(circular_gaussian_smooth(center_kappa, sigma=offset_smooth_sigma))

    for i, sign in enumerate(turn_sign):
        if sign > 0.0:
            # Left turn: move right to increase radius for a minimum-curvature baseline.
            target_offset[i] = -lateral_aggression * curve_strength[i] * right_limit[i]
        elif sign < 0.0:
            target_offset[i] = lateral_aggression * curve_strength[i] * left_limit[i]

    offset = circular_gaussian_smooth(target_offset, sigma=offset_smooth_sigma)
    offset = np.clip(offset, -right_limit, left_limit)

    points = centerline + offset[:, None] * normals
    points = smooth_points(points, sigma=point_smooth_sigma)
    points = centerline + np.clip(
        np.sum((points - centerline) * normals, axis=1),
        -right_limit,
        left_limit,
    )[:, None] * normals

    s, total = cumulative_s(points, closed=True)
    psi, kappa = heading_and_curvature(points)

    vx_curve = np.sqrt(np.maximum(lateral_accel_limit / np.maximum(np.abs(kappa), 1e-6), 0.0))
    vx = np.clip(vx_curve, min_speed, max_speed)

    # Backward/forward acceleration passes keep the speed profile physically plausible.
    ds = np.diff(np.concatenate([s, [total]]))
    for i in range(len(vx) - 2, -1, -1):
        vx[i] = min(vx[i], np.sqrt(max(vx[i + 1] ** 2 + 2.0 * decel_limit * ds[i], 0.0)))
    for i in range(1, len(vx)):
        vx[i] = min(vx[i], np.sqrt(max(vx[i - 1] ** 2 + 2.0 * accel_limit * ds[i - 1], 0.0)))

    ax = np.zeros(len(vx), dtype=np.float64)
    valid_ds = ds > 1e-9
    ax[valid_ds] = (np.roll(vx, -1)[valid_ds] ** 2 - vx[valid_ds] ** 2) / (2.0 * ds[valid_ds])
    ax = np.clip(ax, -decel_limit, accel_limit)

    return np.column_stack([s, points[:, 0], points[:, 1], psi, kappa, vx, ax])


def velocity_profile_from_curvature(
    points: np.ndarray,
    kappa: np.ndarray,
    max_speed: float,
    min_speed: float,
    lateral_accel_limit: float,
    accel_limit: float,
    decel_limit: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    s, total = cumulative_s(points, closed=True)
    vx_curve = np.sqrt(np.maximum(lateral_accel_limit / np.maximum(np.abs(kappa), 1e-6), 0.0))
    vx = np.clip(vx_curve, min_speed, max_speed)

    ds = np.diff(np.concatenate([s, [total]]))
    for i in range(len(vx) - 2, -1, -1):
        vx[i] = min(vx[i], np.sqrt(max(vx[i + 1] ** 2 + 2.0 * decel_limit * ds[i], 0.0)))
    for i in range(1, len(vx)):
        vx[i] = min(vx[i], np.sqrt(max(vx[i - 1] ** 2 + 2.0 * accel_limit * ds[i - 1], 0.0)))

    ax = np.zeros(len(vx), dtype=np.float64)
    valid_ds = ds > 1e-9
    ax[valid_ds] = (np.roll(vx, -1)[valid_ds] ** 2 - vx[valid_ds] ** 2) / (2.0 * ds[valid_ds])
    ax = np.clip(ax, -decel_limit, accel_limit)
    return s, vx, ax


def generate_global_opt_raceline(
    centerline: np.ndarray,
    widths: np.ndarray,
    optimizer_root: str,
    opt_type: str,
    vehicle_width: float,
    safety_margin: float,
    curvature_limit: float,
    stepsize_prep: float,
    stepsize_reg: float,
    stepsize_after_opt: float,
    spline_smoothing: float,
    max_speed: float,
    min_speed: float,
    lateral_accel_limit: float,
    accel_limit: float,
    decel_limit: float,
    debug: bool,
) -> np.ndarray:
    if opt_type not in {"shortest_path", "mincurv", "mincurv_iqp"}:
        raise RuntimeError("global-opt backend supports shortest_path, mincurv, and mincurv_iqp.")

    optimizer_root_abs = os.path.abspath(optimizer_root)
    if not os.path.isdir(optimizer_root_abs):
        raise RuntimeError(f"optimizer root not found: {optimizer_root_abs}")
    if optimizer_root_abs not in sys.path:
        sys.path.insert(0, optimizer_root_abs)

    try:
        import trajectory_planning_helpers as tph
    except ImportError as exc:
        raise RuntimeError(
            "global-opt backend requires trajectory-planning-helpers and quadprog. "
            "Install the Python 3.10 compatible optimizer dependencies first."
        ) from exc

    reftrack = np.column_stack([centerline, widths]).astype(np.float64)
    reftrack, = remove_duplicate_points(reftrack)

    min_width = vehicle_width + 2.0 * safety_margin
    total_width = reftrack[:, 2] + reftrack[:, 3]
    too_narrow = total_width < min_width
    if np.any(too_narrow):
        inflate = 0.5 * (min_width - total_width[too_narrow])
        reftrack[too_narrow, 2] += inflate
        reftrack[too_narrow, 3] += inflate

    reftrack_interp = tph.spline_approximation.spline_approximation(
        track=reftrack,
        k_reg=3,
        s_reg=spline_smoothing,
        stepsize_prep=stepsize_prep,
        stepsize_reg=stepsize_reg,
        debug=debug,
    )

    refpath_interp_cl = np.vstack((reftrack_interp[:, :2], reftrack_interp[0, :2]))
    coeffs_x, coeffs_y, a_interp, normvec = tph.calc_splines.calc_splines(path=refpath_interp_cl)

    if tph.check_normals_crossing.check_normals_crossing(
        track=reftrack_interp,
        normvec_normalized=normvec,
        horizon=10,
    ):
        raise RuntimeError("Spline normals cross. Increase --global-opt-spline-smoothing.")

    width_opt = vehicle_width + 2.0 * safety_margin
    if opt_type == "shortest_path":
        alpha = tph.opt_shortest_path.opt_shortest_path(
            reftrack=reftrack_interp,
            normvectors=normvec,
            w_veh=width_opt,
            print_debug=debug,
        )
    elif opt_type == "mincurv":
        alpha = tph.opt_min_curv.opt_min_curv(
            reftrack=reftrack_interp,
            normvectors=normvec,
            A=a_interp,
            kappa_bound=curvature_limit,
            w_veh=width_opt,
            print_debug=debug,
            plot_debug=False,
        )[0]
    else:
        alpha, reftrack_interp, normvec = tph.iqp_handler.iqp_handler(
            reftrack=reftrack_interp,
            normvectors=normvec,
            A=a_interp,
            kappa_bound=curvature_limit,
            w_veh=width_opt,
            print_debug=debug,
            plot_debug=False,
            stepsize_interp=stepsize_reg,
            iters_min=3,
            curv_error_allowed=0.01,
        )

    (
        raceline_xy,
        _,
        coeffs_x_opt,
        coeffs_y_opt,
        spline_inds,
        t_vals,
        s_points,
        _,
        el_lengths,
    ) = tph.create_raceline.create_raceline(
        refline=reftrack_interp[:, :2],
        normvectors=normvec,
        alpha=alpha,
        stepsize_interp=stepsize_after_opt,
    )

    psi, kappa = tph.calc_head_curv_an.calc_head_curv_an(
        coeffs_x=coeffs_x_opt,
        coeffs_y=coeffs_y_opt,
        ind_spls=spline_inds,
        t_spls=t_vals,
    )
    psi = np.mod(psi, 2.0 * np.pi)

    s = np.asarray(s_points, dtype=np.float64)
    if len(s) != len(raceline_xy):
        s, _, _ = velocity_profile_from_curvature(
            raceline_xy,
            kappa,
            max_speed=max_speed,
            min_speed=min_speed,
            lateral_accel_limit=lateral_accel_limit,
            accel_limit=accel_limit,
            decel_limit=decel_limit,
        )
    _, vx, ax = velocity_profile_from_curvature(
        raceline_xy,
        kappa,
        max_speed=max_speed,
        min_speed=min_speed,
        lateral_accel_limit=lateral_accel_limit,
        accel_limit=accel_limit,
        decel_limit=decel_limit,
    )

    return np.column_stack([s, raceline_xy[:, 0], raceline_xy[:, 1], psi, kappa, vx, ax])


def run(args: argparse.Namespace) -> None:
    centerline, widths = load_centerline(args.centerline)
    backend_used = args.backend

    if args.backend in {"global-opt", "auto"}:
        try:
            raceline = generate_global_opt_raceline(
                centerline=centerline,
                widths=widths,
                optimizer_root=args.optimizer_root or default_optimizer_root(),
                opt_type=args.opt_type,
                vehicle_width=args.vehicle_width,
                safety_margin=args.safety_margin,
                curvature_limit=args.curvature_limit,
                stepsize_prep=args.global_opt_stepsize_prep,
                stepsize_reg=args.global_opt_stepsize_reg,
                stepsize_after_opt=args.global_opt_stepsize_after_opt,
                spline_smoothing=args.global_opt_spline_smoothing,
                max_speed=args.max_speed,
                min_speed=args.min_speed,
                lateral_accel_limit=args.lateral_accel_limit,
                accel_limit=args.accel_limit,
                decel_limit=args.decel_limit,
                debug=args.global_opt_debug,
            )
            backend_used = "global-opt"
        except Exception as exc:
            if args.backend == "global-opt":
                raise
            print(f"Warning: global-opt backend unavailable ({exc}). Falling back to heuristic.")
            backend_used = "heuristic"
            raceline = generate_raceline(
                centerline=centerline,
                widths=widths,
                spacing=args.spacing_m,
                margin=args.track_width_margin_m,
                lateral_aggression=args.lateral_aggression,
                offset_smooth_sigma=args.offset_smooth_sigma,
                point_smooth_sigma=args.point_smooth_sigma,
                max_speed=args.max_speed,
                min_speed=args.min_speed,
                lateral_accel_limit=args.lateral_accel_limit,
                accel_limit=args.accel_limit,
                decel_limit=args.decel_limit,
            )
    else:
        raceline = generate_raceline(
            centerline=centerline,
            widths=widths,
            spacing=args.spacing_m,
            margin=args.track_width_margin_m,
            lateral_aggression=args.lateral_aggression,
            offset_smooth_sigma=args.offset_smooth_sigma,
            point_smooth_sigma=args.point_smooth_sigma,
            max_speed=args.max_speed,
            min_speed=args.min_speed,
            lateral_accel_limit=args.lateral_accel_limit,
            accel_limit=args.accel_limit,
            decel_limit=args.decel_limit,
        )

    out_path = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    header = "s_m; x_m; y_m; psi_rad; kappa_radpm; vx_mps; ax_mps2"
    np.savetxt(out_path, raceline, fmt="%.7f", delimiter=";", header=header)

    print(f"Input centerline: {args.centerline}")
    print(f"Backend: {backend_used}")
    print(f"Output points: {len(raceline)}")
    closed_length = raceline[-1, 0] + float(np.linalg.norm(raceline[0, 1:3] - raceline[-1, 1:3]))
    print(f"Track length: {closed_length:.3f} m")
    print(f"Wrote raceline CSV: {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate a lightweight raceline from a centerline CSV.")
    p.add_argument("--centerline", required=True, help="Path to centerline CSV: x,y,w_tr_right,w_tr_left")
    p.add_argument("--output", required=True, help="Path to output raceline CSV")
    p.add_argument("--backend", choices=["heuristic", "global-opt", "auto"], default="heuristic")

    p.add_argument(
        "--optimizer-root",
        default=None,
        help="Path to global_racetrajectory_optimization checkout for --backend global-opt",
    )
    p.add_argument("--opt-type", choices=["shortest_path", "mincurv", "mincurv_iqp"], default="mincurv_iqp")
    p.add_argument("--vehicle-width", type=float, default=0.18)
    p.add_argument("--safety-margin", type=float, default=0.05)
    p.add_argument("--curvature-limit", type=float, default=3.0)
    p.add_argument("--global-opt-stepsize-prep", type=float, default=0.10)
    p.add_argument("--global-opt-stepsize-reg", type=float, default=0.30)
    p.add_argument("--global-opt-stepsize-after-opt", type=float, default=0.10)
    p.add_argument("--global-opt-spline-smoothing", type=float, default=10.0)
    p.add_argument("--global-opt-debug", action="store_true")

    p.add_argument("--spacing-m", type=float, default=0.10, help="Approximate output waypoint spacing")
    p.add_argument("--track-width-margin-m", type=float, default=0.05)
    p.add_argument("--lateral-aggression", type=float, default=0.70)
    p.add_argument("--offset-smooth-sigma", type=float, default=12.0)
    p.add_argument("--point-smooth-sigma", type=float, default=2.0)
    p.add_argument("--max-speed", type=float, default=3.0)
    p.add_argument("--min-speed", type=float, default=0.8)
    p.add_argument("--lateral-accel-limit", type=float, default=2.5)
    p.add_argument("--accel-limit", type=float, default=1.5)
    p.add_argument("--decel-limit", type=float, default=2.5)
    return p


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
