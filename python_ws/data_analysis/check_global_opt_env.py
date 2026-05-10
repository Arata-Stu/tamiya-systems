#!/usr/bin/env python3
"""Check whether the optional global raceline optimizer dependencies are usable."""

from __future__ import annotations

import argparse
import importlib
import os
import sys


REQUIRED_MODULES = (
    "numpy",
    "scipy",
    "matplotlib",
    "quadprog",
    "trajectory_planning_helpers",
)


def version_of(module_name: str) -> str:
    module = importlib.import_module(module_name)
    return str(getattr(module, "__version__", "ok"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check global optimizer Python dependencies.")
    parser.add_argument(
        "--optimizer-root",
        default=None,
        help="Optional path to global_racetrajectory_optimization checkout.",
    )
    args = parser.parse_args()

    if args.optimizer_root:
        root = os.path.abspath(args.optimizer_root)
        if not os.path.isdir(root):
            raise SystemExit(f"NG optimizer root not found: {root}")
        sys.path.insert(0, root)
        print(f"optimizer_root: {root}")

    ok = True
    for module_name in REQUIRED_MODULES:
        try:
            print(f"OK {module_name}: {version_of(module_name)}")
        except Exception as exc:
            ok = False
            print(f"NG {module_name}: {type(exc).__name__}: {exc}")

    if args.optimizer_root:
        for module_name in ("helper_funcs_glob",):
            try:
                importlib.import_module(module_name)
                print(f"OK {module_name}: importable")
            except Exception as exc:
                ok = False
                print(f"NG {module_name}: {type(exc).__name__}: {exc}")

    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
