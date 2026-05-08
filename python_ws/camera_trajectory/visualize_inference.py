from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Optional, Tuple


SCRIPT_DIR = Path(__file__).resolve().parent
cv2 = None
np = None
torch = None
OmegaConf = None
MultiImageTrajectoryDataset = None
create_trajectory_model = None
infer_trajectory_architecture = None
Compose = None
CropImage = None
NormalizeImage = None
ResizeImage = None


def load_runtime_dependencies():
    global cv2
    global np
    global torch
    global OmegaConf
    global MultiImageTrajectoryDataset
    global create_trajectory_model
    global infer_trajectory_architecture
    global Compose
    global CropImage
    global NormalizeImage
    global ResizeImage

    import cv2 as cv2_module
    import numpy as np_module
    import torch as torch_module
    from omegaconf import OmegaConf as OmegaConf_module

    from src.dataset import MultiImageTrajectoryDataset as MultiImageTrajectoryDataset_class
    from src.model import create_trajectory_model as create_trajectory_model_fn
    from src.model import infer_trajectory_architecture as infer_trajectory_architecture_fn
    from src.transform import Compose as Compose_class
    from src.transform import CropImage as CropImage_class
    from src.transform import NormalizeImage as NormalizeImage_class
    from src.transform import ResizeImage as ResizeImage_class

    cv2 = cv2_module
    np = np_module
    torch = torch_module
    OmegaConf = OmegaConf_module
    MultiImageTrajectoryDataset = MultiImageTrajectoryDataset_class
    create_trajectory_model = create_trajectory_model_fn
    infer_trajectory_architecture = infer_trajectory_architecture_fn
    Compose = Compose_class
    CropImage = CropImage_class
    NormalizeImage = NormalizeImage_class
    ResizeImage = ResizeImage_class


def resolve_path(path: str) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def select_checkpoint_interactive(base_dir: Path) -> Path:
    options = sorted(base_dir.rglob("best_model.pth"))
    if not options:
        raise FileNotFoundError(f"No best_model.pth found under {base_dir}")

    print("--- Select checkpoint file ---")
    for i, path in enumerate(options, start=1):
        print(f"{i}) {path}")
    print(f"{len(options) + 1}) Quit")

    while True:
        choice = input("Select number: ").strip()
        if choice == str(len(options) + 1):
            raise SystemExit(1)
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            return options[int(choice) - 1]
        print("Invalid selection.")


def default_data_dir() -> Path:
    test_dir = SCRIPT_DIR / "datasets" / "test"
    train_dir = SCRIPT_DIR / "datasets" / "train"
    if test_dir.exists():
        return test_dir
    return train_dir


def load_checkpoint(path: Path, device: torch.device):
    checkpoint = torch.load(path, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"], checkpoint
    return checkpoint, {}


def denormalize_last_frame(image_seq: torch.Tensor) -> np.ndarray:
    image = image_seq[-1].detach().cpu().numpy()
    if image.ndim != 3:
        raise ValueError(f"Expected image frame shape (H,W,C) or (C,H,W), got {image.shape}")
    if image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.shape[-1] == 4:
        image = image[:, :, :3]
    if image.dtype != np.uint8:
        if image.max() <= 1.5:
            image = image * 255.0
        image = np.clip(image, 0, 255).astype(np.uint8)
    return image


def draw_polyline(
    canvas: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
    x_range: Tuple[float, float],
    y_range: Tuple[float, float],
    radius: int,
) -> None:
    if points.size == 0:
        return

    h, w = canvas.shape[:2]
    margin = 36
    x_min, x_max = x_range
    y_min, y_max = y_range

    def to_pixel(point):
        x, y = float(point[0]), float(point[1])
        u = margin + (y_max - y) / max(1e-6, y_max - y_min) * (w - 2 * margin)
        v = margin + (x_max - x) / max(1e-6, x_max - x_min) * (h - 2 * margin)
        return int(round(u)), int(round(v))

    pixels = [to_pixel(p) for p in points]
    for a, b in zip(pixels[:-1], pixels[1:]):
        cv2.line(canvas, a, b, color, 2, cv2.LINE_AA)
    for p in pixels:
        cv2.circle(canvas, p, radius, color, -1, cv2.LINE_AA)


def draw_topdown(pred: np.ndarray, gt: Optional[np.ndarray], size: int = 520) -> np.ndarray:
    canvas = np.full((size, size, 3), 248, dtype=np.uint8)
    margin = 36

    all_points = pred if gt is None else np.concatenate([pred, gt], axis=0)
    x_max = max(4.0, float(np.nanmax(all_points[:, 0])) + 1.0)
    x_min = min(-1.0, float(np.nanmin(all_points[:, 0])) - 1.0)
    y_abs = max(2.0, float(np.nanmax(np.abs(all_points[:, 1]))) + 1.0)
    x_range = (x_min, x_max)
    y_range = (-y_abs, y_abs)

    for t in np.linspace(0, 1, 7):
        u = int(round(margin + t * (size - 2 * margin)))
        v = int(round(margin + t * (size - 2 * margin)))
        cv2.line(canvas, (u, margin), (u, size - margin), (226, 226, 226), 1)
        cv2.line(canvas, (margin, v), (size - margin, v), (226, 226, 226), 1)

    cv2.rectangle(canvas, (margin, margin), (size - margin, size - margin), (150, 150, 150), 1)
    cv2.putText(canvas, "top-down trajectory", (margin, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (35, 35, 35), 2)
    cv2.putText(canvas, "x forward", (size - 150, margin - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    cv2.putText(canvas, "y left/right", (margin, size - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)

    origin_u = size // 2
    origin_v = size - margin
    cv2.circle(canvas, (origin_u, origin_v), 6, (45, 45, 45), -1, cv2.LINE_AA)
    cv2.putText(canvas, "base_link", (origin_u + 8, origin_v - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (45, 45, 45), 1)

    if gt is not None:
        draw_polyline(canvas, gt, (230, 160, 20), x_range, y_range, 4)
        cv2.putText(canvas, "GT", (margin, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 160, 20), 2)
    draw_polyline(canvas, pred, (30, 170, 80), x_range, y_range, 4)
    cv2.putText(canvas, "Pred", (margin + 64, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 170, 80), 2)

    return canvas


def project_base_points_to_image(
    points: np.ndarray,
    image_shape: Tuple[int, int],
    intrinsics: Tuple[float, float, float, float],
    calib_size: Tuple[int, int],
    camera_xyz: Tuple[float, float, float],
    pitch_down_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    fx, fy, cx, cy = intrinsics
    calib_width, calib_height = calib_size
    image_h, image_w = image_shape
    sx = image_w / float(calib_width)
    sy = image_h / float(calib_height)
    fx *= sx
    cx *= sx
    fy *= sy
    cy *= sy

    cam_x, cam_y, cam_z = camera_xyz
    pitch = np.deg2rad(pitch_down_deg)
    yaw = np.deg2rad(yaw_deg)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cyaw = np.cos(-yaw)
    syaw = np.sin(-yaw)

    pixels = []
    for point in points:
        xb = float(point[0]) - cam_x
        yb = float(point[1]) - cam_y
        zb = -cam_z

        # Optional yaw correction in base_link before converting to optical frame.
        x_rot = cyaw * xb - syaw * yb
        y_rot = syaw * xb + cyaw * yb

        # ROS optical frame: x right, y down, z forward.
        xo = -y_rot
        yo = -zb
        zo = x_rot

        # Positive pitch_down rotates the camera optical axis toward the ground.
        yc = cp * yo + sp * zo
        zc = -sp * yo + cp * zo
        xc = xo

        if zc <= 0.05:
            pixels.append((np.nan, np.nan))
            continue

        u = fx * xc / zc + cx
        v = fy * yc / zc + cy
        if u < -image_w or u > image_w * 2 or v < -image_h or v > image_h * 2:
            pixels.append((np.nan, np.nan))
        else:
            pixels.append((u, v))
    return np.array(pixels, dtype=np.float32)


def draw_projected_points(
    image_rgb: np.ndarray,
    points: np.ndarray,
    color: Tuple[int, int, int],
) -> None:
    valid = np.isfinite(points).all(axis=1)
    pixels = points[valid]
    if len(pixels) == 0:
        return
    pixels_i = [(int(round(u)), int(round(v))) for u, v in pixels]
    h, w = image_rgb.shape[:2]
    pixels_i = [(u, v) for u, v in pixels_i if 0 <= u < w and 0 <= v < h]
    for a, b in zip(pixels_i[:-1], pixels_i[1:]):
        cv2.line(image_rgb, a, b, color, 3, cv2.LINE_AA)
    for p in pixels_i:
        cv2.circle(image_rgb, p, 5, color, -1, cv2.LINE_AA)


def make_visualization(
    image_rgb: np.ndarray,
    pred: np.ndarray,
    gt: Optional[np.ndarray],
    title: str,
    project_image: bool,
    intrinsics: Tuple[float, float, float, float],
    calib_size: Tuple[int, int],
    camera_xyz: Tuple[float, float, float],
    pitch_down_deg: float,
    yaw_deg: float,
) -> np.ndarray:
    image_for_panel = image_rgb.copy()
    if project_image:
        pred_px = project_base_points_to_image(
            pred,
            image_for_panel.shape[:2],
            intrinsics,
            calib_size,
            camera_xyz,
            pitch_down_deg,
            yaw_deg,
        )
        draw_projected_points(image_for_panel, pred_px, (30, 220, 90))
        if gt is not None:
            gt_px = project_base_points_to_image(
                gt,
                image_for_panel.shape[:2],
                intrinsics,
                calib_size,
                camera_xyz,
                pitch_down_deg,
                yaw_deg,
            )
            draw_projected_points(image_for_panel, gt_px, (245, 185, 25))

    image_panel = cv2.resize(image_for_panel, (520, 390), interpolation=cv2.INTER_AREA)
    header = np.full((70, 520, 3), 248, dtype=np.uint8)
    title_text = "camera image + projection" if project_image else "camera image"
    cv2.putText(header, title_text, (16, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (35, 35, 35), 2)
    cv2.putText(header, title[:55], (16, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 80, 80), 1)
    left = np.vstack([header, image_panel, np.full((60, 520, 3), 248, dtype=np.uint8)])

    topdown = draw_topdown(pred, gt, size=520)
    return np.hstack([left, topdown])


def parse_indices(indices: str, total: int, num_samples: int, stride: int) -> List[int]:
    if indices:
        result = []
        for chunk in indices.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            result.append(int(chunk))
        return [i for i in result if 0 <= i < total]
    return list(range(0, total, max(1, stride)))[:num_samples]


def main():
    parser = argparse.ArgumentParser(description="Visualize camera trajectory inference.")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--checkpoint-base", type=str, default="./ckpts/train")
    parser.add_argument("--data-dir", type=str, default=None)
    parser.add_argument("--config", type=str, default="./config/train.yaml")
    parser.add_argument("--output-dir", type=str, default="./outputs/trajectory_vis")
    parser.add_argument("--indices", type=str, default="", help="Comma-separated dataset indices, e.g. 0,10,20")
    parser.add_argument("--num-samples", type=int, default=100)
    parser.add_argument("--stride", type=int, default=20)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--no-image-projection", action="store_true")
    parser.add_argument("--fx", type=float, default=615.9685668945312)
    parser.add_argument("--fy", type=float, default=616.263916015625)
    parser.add_argument("--cx", type=float, default=320.44207763671875)
    parser.add_argument("--cy", type=float, default=246.1153564453125)
    parser.add_argument("--calib-width", type=int, default=640)
    parser.add_argument("--calib-height", type=int, default=480)
    parser.add_argument("--camera-x", type=float, default=0.0)
    parser.add_argument("--camera-y", type=float, default=0.0)
    parser.add_argument("--camera-height", type=float, default=0.06)
    parser.add_argument("--camera-pitch-down-deg", type=float, default=0.0)
    parser.add_argument("--camera-yaw-deg", type=float, default=0.0)
    args = parser.parse_args()

    load_runtime_dependencies()
    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = resolve_path(args.checkpoint) if args.checkpoint else select_checkpoint_interactive(resolve_path(args.checkpoint_base))
    cfg = OmegaConf.load(resolve_path(args.config))
    data_dir = resolve_path(args.data_dir) if args.data_dir else default_data_dir()
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    state_dict, checkpoint_meta = load_checkpoint(checkpoint_path, device)
    num_points = int(checkpoint_meta.get("num_points", cfg.model.num_points))
    output_scale = float(checkpoint_meta.get("output_scale", cfg.model.output_scale))
    architecture = checkpoint_meta.get("model_architecture") or infer_trajectory_architecture(state_dict, num_points)

    transform = Compose(
        [
            CropImage(top_ratio=cfg.dataset.crop_top_ratio, bottom_ratio=cfg.dataset.crop_bottom_ratio),
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            NormalizeImage(mean=cfg.dataset.pixel_mean, std=cfg.dataset.pixel_std),
        ]
    )

    dataset = MultiImageTrajectoryDataset(
        base_dir=str(data_dir),
        transform=None,
        seq_len=int(cfg.dataset.sequence_length),
    )

    model = create_trajectory_model(
        architecture=architecture,
        num_points=num_points,
        input_channels=int(cfg.dataset.input_channels),
        input_height=int(cfg.dataset.image_height),
        input_width=int(cfg.dataset.image_width),
        output_scale=output_scale,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()

    selected_indices = parse_indices(args.indices, len(dataset), args.num_samples, args.stride)
    print("--- Visualization Configuration ---")
    print(f"Checkpoint : {checkpoint_path}")
    print(f"Data dir   : {data_dir}")
    print(f"Output dir : {output_dir}")
    print(f"Samples    : {selected_indices}")
    print(f"Model      : architecture={architecture}, num_points={num_points}, output_scale={output_scale}")
    print(
        "Projection : "
        f"enabled={not args.no_image_projection}, "
        f"fx={args.fx:.3f}, fy={args.fy:.3f}, cx={args.cx:.3f}, cy={args.cy:.3f}, "
        f"camera_xyz=({args.camera_x:.3f}, {args.camera_y:.3f}, {args.camera_height:.3f}), "
        f"pitch_down={args.camera_pitch_down_deg:.2f}, yaw={args.camera_yaw_deg:.2f}"
    )
    print("------------------------------------")

    with torch.no_grad():
        for idx in selected_indices:
            raw_sample = dataset[idx]
            image_rgb = denormalize_last_frame(raw_sample["image"])
            model_sample = transform({"image": raw_sample["image"].clone(), "trajectory": raw_sample["trajectory"].clone()})
            images = model_sample["image"].unsqueeze(0).to(device).contiguous()
            pred = model(images)[0].detach().cpu().numpy()
            gt = raw_sample["trajectory"].detach().cpu().numpy()

            panel = make_visualization(
                image_rgb,
                pred,
                gt,
                title=f"dataset index {idx}",
                project_image=not args.no_image_projection,
                intrinsics=(args.fx, args.fy, args.cx, args.cy),
                calib_size=(args.calib_width, args.calib_height),
                camera_xyz=(args.camera_x, args.camera_y, args.camera_height),
                pitch_down_deg=args.camera_pitch_down_deg,
                yaw_deg=args.camera_yaw_deg,
            )
            out_path = output_dir / f"trajectory_{idx:06d}.png"
            cv2.imwrite(str(out_path), cv2.cvtColor(panel, cv2.COLOR_RGB2BGR))
            print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
