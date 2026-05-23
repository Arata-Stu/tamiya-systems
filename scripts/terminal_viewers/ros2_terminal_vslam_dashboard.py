#!/usr/bin/env python3
"""VSLAM/HD-map preset wrapper for ros2_terminal_dashboard.py."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    script_dir = Path(__file__).resolve().parent
    sys.path.insert(0, str(script_dir))
    import ros2_terminal_dashboard  # type: ignore

    defaults = [
        "--map-topic",
        "",
        "--localization-topic",
        "",
        "--amcl-pose-topic",
        "",
        "--initial-pose-topic",
        "",
        "--particles-topic",
        "",
        "--path-topic",
        "/visual_slam/tracking/slam_path",
        "--vo-path-topic",
        "/visual_slam/tracking/vo_path",
        "--global-path-topic",
        "/planning/global_raceline",
        "--local-path-topic",
        "/autonomous/trajectory",
        "--section-markers-topic",
        "",
        "--hd-lane-markers-topic",
        "/hd_map/lane_markers",
        "--hd-section-markers-topic",
        "/hd_map/section_markers",
        "--current-section-marker-topic",
        "/localization/current_section_marker",
        "--current-section-topic",
        "/localization/current_section",
        "--odom-topic",
        "/visual_slam/tracking/odometry",
        "--image-topic",
        "/camera/left/image_raw",
        "--camera-info-topic",
        "/camera/left/camera_info",
        "--crop-image-topic",
        "/perception/crop/image",
        "--allow-latest-tf-fallback",
        "--assume-same-frame",
        "--best-effort",
    ]
    sys.argv = [sys.argv[0], *defaults, *sys.argv[1:]]
    return ros2_terminal_dashboard.main()


if __name__ == "__main__":
    raise SystemExit(main())
