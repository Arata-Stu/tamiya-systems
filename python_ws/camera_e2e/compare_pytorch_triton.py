#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from rosbags.highlevel import AnyReader

from src.dataset import MultiImageSequenceDataset
from src.model import PilotNetControl
from src.transform import Compose, ConvertToGray3Channel, CropImage, ResizeImage


SCRIPT_DIR = Path(__file__).resolve().parent


@dataclass
class SampleData:
    raw_image: np.ndarray
    description: str
    encoding: str
    ground_truth: Optional[np.ndarray] = None


def resolve_path(path: str) -> Path:
    candidate = Path(path).expanduser()
    if candidate.is_absolute():
        return candidate.resolve()
    return (SCRIPT_DIR / candidate).resolve()


def ensure_uint8_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 3 and image.shape[-1] == 1:
        image = image[:, :, 0]
    if image.dtype == np.uint8:
        return np.ascontiguousarray(image)

    image = image.astype(np.float32)
    if image.max() <= 1.0:
        image = image * 255.0
    image = np.clip(image, 0.0, 255.0).astype(np.uint8)
    return np.ascontiguousarray(image)


def to_rgb3(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.shape[2] >= 3:
        return image[:, :, :3].copy()
    raise ValueError(f"Unsupported image shape: {image.shape}")


def to_mono8(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return image.copy()
    if image.ndim != 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")
    if image.shape[2] == 1:
        return image[:, :, 0].copy()

    rgb = image[:, :, :3].astype(np.float32)
    gray = 0.2989 * rgb[:, :, 0] + 0.5870 * rgb[:, :, 1] + 0.1140 * rgb[:, :, 2]
    return np.clip(np.round(gray), 0.0, 255.0).astype(np.uint8)


def mono_to_gray3(image: np.ndarray) -> np.ndarray:
    gray = to_mono8(image)
    return np.repeat(gray[:, :, None], 3, axis=2)


def make_visual_image(image: np.ndarray) -> np.ndarray:
    image = ensure_uint8_image(image)
    if image.ndim == 2:
        return np.repeat(image[:, :, None], 3, axis=2)
    if image.ndim == 3 and image.shape[2] == 1:
        return np.repeat(image, 3, axis=2)
    if image.ndim == 3 and image.shape[2] >= 3:
        return image[:, :, :3].copy()
    raise ValueError(f"Unsupported image shape: {image.shape}")


def make_visual_image_from_float(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4):
        image = np.transpose(image, (1, 2, 0))
    if image.ndim == 2:
        image = image[:, :, None]
    if image.ndim == 3 and image.shape[-1] == 1:
        image = np.repeat(image, 3, axis=2)
    if image.ndim != 3 or image.shape[-1] < 3:
        raise ValueError(f"Unsupported image shape: {image.shape}")

    image = image[:, :, :3].astype(np.float32)
    return np.clip(np.round(image), 0.0, 255.0).astype(np.uint8)


def save_rgb_image(path: Path, image_rgb: np.ndarray) -> None:
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), image_bgr)


def select_checkpoint_interactive(base_dir: Path) -> Path:
    candidates = sorted(base_dir.rglob("best_model.pth"))
    if not candidates:
        raise FileNotFoundError(f"No best_model.pth found under {base_dir}")

    print("--- Select checkpoint file ---")
    for idx, path in enumerate(candidates, start=1):
        print(f"{idx}) {path}")
    print(f"{len(candidates) + 1}) Quit")

    while True:
        choice = input("Select number: ").strip()
        if choice == str(len(candidates) + 1):
            raise SystemExit(1)
        if choice.isdigit():
            index = int(choice)
            if 1 <= index <= len(candidates):
                return candidates[index - 1]
        print("Invalid selection.")


def resolve_checkpoint_path(path: Optional[str]) -> Path:
    if path:
        checkpoint = resolve_path(path)
        if not checkpoint.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
        return checkpoint

    ckpt_root = SCRIPT_DIR / "ckpts"
    return select_checkpoint_interactive(ckpt_root)


def find_latest_triton_onnx(model_repo_root: Path, model_name: str) -> Optional[Path]:
    model_root = model_repo_root / model_name
    if not model_root.exists():
        return None

    best_version = None
    best_path = None
    for child in model_root.iterdir():
        if not child.is_dir() or not child.name.isdigit():
            continue
        onnx_path = child / "model.onnx"
        if not onnx_path.exists():
            continue
        version = int(child.name)
        if best_version is None or version > best_version:
            best_version = version
            best_path = onnx_path
    return best_path


def resolve_onnx_path(
    checkpoint_path: Path,
    explicit_onnx: Optional[str],
    model_repo_root: Path,
    model_name: str,
) -> Optional[Path]:
    if explicit_onnx:
        onnx_path = resolve_path(explicit_onnx)
        if not onnx_path.exists():
            raise FileNotFoundError(f"ONNX not found: {onnx_path}")
        return onnx_path

    sibling = checkpoint_path.with_suffix(".onnx")
    if sibling.exists():
        return sibling

    return find_latest_triton_onnx(model_repo_root, model_name)


def decode_raw_image_message(msg: Any) -> Tuple[np.ndarray, str]:
    height = int(msg.height)
    width = int(msg.width)
    encoding = msg.encoding.lower()
    step = int(msg.step)

    if height <= 0 or width <= 0 or step <= 0:
        raise ValueError(f"Invalid image size: height={height}, width={width}, step={step}")

    raw = np.frombuffer(msg.data, dtype=np.uint8)
    expected = height * step
    if raw.size < expected:
        raise ValueError(f"Image buffer too small: {raw.size} < {expected}")

    raw = raw[:expected].reshape(height, step)

    if encoding == "rgb8":
        return raw[:, : width * 3].reshape(height, width, 3).copy(), encoding
    if encoding in ("bgr8", "8uc3"):
        bgr = raw[:, : width * 3].reshape(height, width, 3)
        return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), encoding
    if encoding == "rgba8":
        rgba = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2RGB), encoding
    if encoding == "bgra8":
        bgra = raw[:, : width * 4].reshape(height, width, 4)
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2RGB), encoding
    if encoding in ("mono8", "8uc1"):
        return raw[:, :width].copy(), encoding
    if encoding in ("mono16", "16uc1", "16sc1"):
        raw16 = np.frombuffer(msg.data, dtype=np.uint16)
        if raw16.size < height * width:
            raise ValueError(f"Image buffer too small for {encoding}: {raw16.size}")
        gray16 = raw16[: height * width].reshape(height, width)
        gray8 = np.clip(gray16 / 256.0, 0.0, 255.0).astype(np.uint8)
        return gray8, encoding
    if encoding in ("yuyv", "yuyv422", "yuv422", "uyvy"):
        yuv = raw[:, : width * 2].reshape(height, width, 2)
        code = cv2.COLOR_YUV2RGB_YUY2 if encoding != "uyvy" else cv2.COLOR_YUV2RGB_UYVY
        return cv2.cvtColor(yuv, code), encoding

    raise ValueError(f"Unsupported raw image encoding: {encoding}")


def decode_compressed_image_message(msg: Any) -> Tuple[np.ndarray, str]:
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    bgr = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Failed to decode compressed image")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), "compressed_rgb8"


def load_sample_from_bag(bag_dir: Path, image_topic: str, frame_index: int) -> SampleData:
    if frame_index < 0:
        raise ValueError("frame_index must be >= 0")

    with AnyReader([bag_dir]) as reader:
        connections = [c for c in reader.connections if c.topic == image_topic]
        if not connections:
            raise RuntimeError(f"Topic not found in bag: {image_topic}")

        seen = 0
        for conn, timestamp, raw in reader.messages(connections=connections):
            msg = reader.deserialize(raw, conn.msgtype)
            if seen != frame_index:
                seen += 1
                continue

            if conn.msgtype == "sensor_msgs/msg/Image":
                image, encoding = decode_raw_image_message(msg)
            elif conn.msgtype == "sensor_msgs/msg/CompressedImage":
                image, encoding = decode_compressed_image_message(msg)
            else:
                raise RuntimeError(f"Unsupported msg type: {conn.msgtype}")

            description = f"bag={bag_dir.name}, topic={image_topic}, frame_index={frame_index}, timestamp={timestamp}"
            return SampleData(raw_image=image, description=description, encoding=encoding)

    raise IndexError(f"frame_index {frame_index} out of range for topic {image_topic}")


def load_sample_from_dataset(dataset_root: Path, sample_index: int) -> SampleData:
    dataset = MultiImageSequenceDataset(base_dir=str(dataset_root), transform=None, seq_len=1)
    if sample_index < 0 or sample_index >= len(dataset):
        raise IndexError(f"sample_index {sample_index} out of range: dataset size={len(dataset)}")

    sample = dataset[sample_index]
    raw_image = sample["image"][-1].cpu().numpy().astype(np.uint8)

    steer = float(sample["steer"][-1].item())
    speed = float(sample["speed"][-1].item())
    description = f"dataset={dataset_root}, sample_index={sample_index}"
    return SampleData(
        raw_image=raw_image,
        description=description,
        encoding="dataset_rgb8",
        ground_truth=np.array([steer, speed], dtype=np.float32),
    )


def load_sample_from_image(image_path: Path) -> SampleData:
    if image_path.suffix.lower() == ".npy":
        image = ensure_uint8_image(np.load(image_path))
        encoding = "npy"
    else:
        image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
        if image is None:
            raise RuntimeError(f"Failed to read image: {image_path}")
        if image.ndim == 2:
            encoding = "mono8"
        elif image.ndim == 3 and image.shape[2] == 1:
            image = image[:, :, 0]
            encoding = "mono8"
        elif image.ndim == 3 and image.shape[2] == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            encoding = "rgb8"
        elif image.ndim == 3 and image.shape[2] == 4:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            encoding = "rgba8"
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")
        image = ensure_uint8_image(image)

    description = f"image={image_path}"
    return SampleData(raw_image=image, description=description, encoding=encoding)


def auto_dataset_root() -> Path:
    test_root = SCRIPT_DIR / "datasets" / "test"
    train_root = SCRIPT_DIR / "datasets" / "train"
    if test_root.exists():
        return test_root
    if train_root.exists():
        return train_root
    raise FileNotFoundError("No default dataset root found under ./datasets/test or ./datasets/train")


def load_sample(args: argparse.Namespace) -> SampleData:
    if args.bag_dir:
        return load_sample_from_bag(resolve_path(args.bag_dir), args.image_topic, args.frame_index)
    if args.dataset_root:
        return load_sample_from_dataset(resolve_path(args.dataset_root), args.sample_index)
    if args.image:
        return load_sample_from_image(resolve_path(args.image))
    return load_sample_from_dataset(auto_dataset_root(), 0)


def apply_preview_transform(image: np.ndarray, transform: Compose) -> np.ndarray:
    hwc = ensure_uint8_image(image)
    if hwc.ndim == 2:
        hwc = hwc[:, :, None]
    sample = {"image": torch.from_numpy(np.ascontiguousarray(hwc[None, ...]))}
    sample = transform(sample)
    output = sample["image"][0].detach().cpu().numpy()

    if output.ndim != 3:
        raise ValueError(f"Unexpected transformed shape: {output.shape}")
    if output.shape[0] in (1, 3, 4) and output.shape[-1] not in (1, 3, 4):
        output = np.transpose(output, (1, 2, 0))

    output = output.astype(np.float32, copy=False)
    if output.ndim == 2:
        output = output[:, :, None]
    if output.shape[-1] == 1:
        output = np.repeat(output, 3, axis=2)
    if output.shape[-1] > 3:
        output = output[:, :, :3]
    return np.ascontiguousarray(output)


def normalize_preview_image(image_rgb: np.ndarray, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    image_rgb = np.asarray(image_rgb, dtype=np.float32)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError(f"Expected HWC RGB image, got {image_rgb.shape}")

    tensor = torch.from_numpy(np.ascontiguousarray(np.transpose(image_rgb, (2, 0, 1)))).to(torch.float32)
    if tensor.max() > 1.0:
        tensor = tensor / 255.0

    mean_t = torch.tensor(list(mean), dtype=torch.float32).view(3, 1, 1)
    std_t = torch.tensor(list(std), dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean_t) / std_t


def emulate_dataset_extractor_output(raw_image: np.ndarray) -> np.ndarray:
    return to_rgb3(raw_image)


def emulate_ros2_encoder_input(
    raw_image: np.ndarray,
    *,
    force_grayscale_3ch: bool,
    use_rgb_format_converter: bool,
) -> np.ndarray:
    if force_grayscale_3ch:
        return mono_to_gray3(raw_image)
    if use_rgb_format_converter:
        return to_rgb3(raw_image)

    # Best-effort fallback. In practice camera_e2e currently expects 3 channels.
    return to_rgb3(raw_image)


def build_train_preview_transform(cfg: Any) -> Compose:
    return Compose(
        [
            CropImage(top_ratio=cfg.dataset.crop_top_ratio, bottom_ratio=cfg.dataset.crop_bottom_ratio),
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            ConvertToGray3Channel(enabled=cfg.dataset.force_grayscale_3ch),
        ]
    )


def build_ros2_preview_transform(cfg: Any) -> Compose:
    return Compose([ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width)])


def load_model_from_checkpoint(checkpoint_path: Path, cfg: Any) -> PilotNetControl:
    model = PilotNetControl(
        num_outputs=cfg.model.num_outputs,
        input_channels=cfg.dataset.input_channels,
        input_height=cfg.dataset.image_height,
        input_width=cfg.dataset.image_width,
    )

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    return model


def run_pytorch(model: PilotNetControl, input_tensor: torch.Tensor) -> np.ndarray:
    with torch.no_grad():
        output = model(input_tensor.unsqueeze(0))
    return output[0].detach().cpu().numpy().astype(np.float32)


def run_onnx(onnx_path: Path, input_tensor: torch.Tensor) -> np.ndarray:
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    output = session.run([output_name], {input_name: input_tensor.unsqueeze(0).cpu().numpy()})[0]
    return output[0].astype(np.float32)


def preview_to_uint8(image_rgb: np.ndarray) -> np.ndarray:
    return make_visual_image_from_float(image_rgb)


def tensor_stats(a: torch.Tensor, b: torch.Tensor) -> Dict[str, float]:
    diff = (a - b).abs()
    return {
        "max_abs": float(diff.max().item()),
        "mean_abs": float(diff.mean().item()),
        "l2": float(torch.linalg.vector_norm((a - b).reshape(-1)).item()),
    }


def control_dict(values: Optional[np.ndarray]) -> Optional[Dict[str, float]]:
    if values is None:
        return None
    return {"steer": float(values[0]), "speed": float(values[1])}


def save_outputs(
    output_dir: Path,
    *,
    source_preview: np.ndarray,
    extractor_preview: np.ndarray,
    train_preview: np.ndarray,
    ros2_preview: np.ndarray,
    summary: Dict[str, Any],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    save_rgb_image(output_dir / "source_raw.png", source_preview)
    save_rgb_image(output_dir / "extractor_output.png", extractor_preview)
    train_preview_u8 = preview_to_uint8(train_preview)
    ros2_preview_u8 = preview_to_uint8(ros2_preview)
    save_rgb_image(output_dir / "train_preview.png", train_preview_u8)
    save_rgb_image(output_dir / "ros2_preview.png", ros2_preview_u8)

    abs_diff = cv2.absdiff(train_preview_u8, ros2_preview_u8)
    save_rgb_image(output_dir / "preview_absdiff.png", abs_diff)

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare camera_e2e outputs for the same image across the training pipeline and the "
            "ROS2/Triton deployment pipeline."
        )
    )
    parser.add_argument("--checkpoint", type=str, default=None, help="Path to best_model.pth")
    parser.add_argument("--config", type=str, default="config/train.yaml", help="Training config path")
    parser.add_argument("--onnx", type=str, default=None, help="Path to model.onnx for Triton artifact comparison")
    parser.add_argument(
        "--model-repository-root",
        type=str,
        default="/workspaces/isaac_ros_assets/models",
        help="Triton model repository root used when --onnx is omitted",
    )
    parser.add_argument("--model-name", type=str, default="pilotnet", help="Triton model name")

    parser.add_argument("--image", type=str, default=None, help="Standalone image path (.png/.jpg/.npy)")
    parser.add_argument("--dataset-root", type=str, default=None, help="Dataset root like ./datasets/test")
    parser.add_argument("--sample-index", type=int, default=0, help="Global sample index for --dataset-root")
    parser.add_argument("--bag-dir", type=str, default=None, help="Rosbag directory containing metadata.yaml")
    parser.add_argument("--image-topic", type=str, default="/camera/left/image_raw", help="Image topic for --bag-dir")
    parser.add_argument("--frame-index", type=int, default=0, help="Frame index for --bag-dir")

    parser.add_argument(
        "--force-grayscale-3ch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emulate current ROS2 launch behavior that enforces gray->3ch before encoder",
    )
    parser.add_argument(
        "--use-rgb-format-converter",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Emulate the launch path that converts images to rgb8 before encoder",
    )
    parser.add_argument("--output-dir", type=str, default=None, help="Optional directory to save preview images/json")
    args = parser.parse_args()

    source_count = sum(1 for value in (args.image, args.dataset_root, args.bag_dir) if value)
    if source_count > 1:
        raise ValueError("Specify only one of --image, --dataset-root, or --bag-dir")

    checkpoint_path = resolve_checkpoint_path(args.checkpoint)
    cfg = OmegaConf.load(resolve_path(args.config))
    sample = load_sample(args)

    model_repo_root = Path(args.model_repository_root).expanduser().resolve()
    onnx_path = resolve_onnx_path(checkpoint_path, args.onnx, model_repo_root, args.model_name)

    train_preview_transform = build_train_preview_transform(cfg)
    ros2_preview_transform = build_ros2_preview_transform(cfg)

    extractor_image = emulate_dataset_extractor_output(sample.raw_image)
    ros2_encoder_image = emulate_ros2_encoder_input(
        sample.raw_image,
        force_grayscale_3ch=args.force_grayscale_3ch,
        use_rgb_format_converter=args.use_rgb_format_converter,
    )

    train_preview = apply_preview_transform(extractor_image, train_preview_transform)
    ros2_preview = apply_preview_transform(ros2_encoder_image, ros2_preview_transform)

    train_tensor = normalize_preview_image(train_preview, cfg.dataset.pixel_mean, cfg.dataset.pixel_std)
    ros2_tensor = normalize_preview_image(ros2_preview, cfg.dataset.pixel_mean, cfg.dataset.pixel_std)

    model = load_model_from_checkpoint(checkpoint_path, cfg)
    pytorch_train = run_pytorch(model, train_tensor)
    pytorch_ros2 = run_pytorch(model, ros2_tensor)
    onnx_ros2 = run_onnx(onnx_path, ros2_tensor) if onnx_path is not None else None

    tensor_diff = tensor_stats(train_tensor, ros2_tensor)

    summary: Dict[str, Any] = {
        "source": {
            "description": sample.description,
            "encoding": sample.encoding,
            "raw_shape": list(sample.raw_image.shape),
        },
        "artifacts": {
            "checkpoint": str(checkpoint_path),
            "onnx": str(onnx_path) if onnx_path is not None else None,
        },
        "ground_truth": control_dict(sample.ground_truth),
        "tensor_diff_train_vs_ros2": tensor_diff,
        "outputs": {
            "pytorch_train_pipeline": control_dict(pytorch_train),
            "pytorch_ros2_pipeline": control_dict(pytorch_ros2),
            "onnx_ros2_pipeline": control_dict(onnx_ros2),
        },
        "output_diff": {
            "pytorch_train_vs_ros2_abs": control_dict(np.abs(pytorch_train - pytorch_ros2)),
            "pytorch_ros2_vs_onnx_abs": (
                control_dict(np.abs(pytorch_ros2 - onnx_ros2)) if onnx_ros2 is not None else None
            ),
        },
    }

    print("=== Sample ===")
    print(f"source: {sample.description}")
    print(f"encoding: {sample.encoding}")
    print(f"raw shape: {sample.raw_image.shape}")

    if sample.ground_truth is not None:
        print("ground truth:")
        print(f"  steer={sample.ground_truth[0]:+.6f}, speed={sample.ground_truth[1]:+.6f}")

    print("\n=== Tensor Diff (train vs ROS2) ===")
    print(f"max_abs : {tensor_diff['max_abs']:.8f}")
    print(f"mean_abs: {tensor_diff['mean_abs']:.8f}")
    print(f"l2      : {tensor_diff['l2']:.8f}")

    print("\n=== Output Compare ===")
    print(f"PyTorch(train): steer={pytorch_train[0]:+.6f}, speed={pytorch_train[1]:+.6f}")
    print(f"PyTorch(ros2) : steer={pytorch_ros2[0]:+.6f}, speed={pytorch_ros2[1]:+.6f}")
    if onnx_ros2 is not None:
        print(f"ONNX(ros2)    : steer={onnx_ros2[0]:+.6f}, speed={onnx_ros2[1]:+.6f}")
        print(
            "abs(PyTorch(ros2)-ONNX): "
            f"steer={abs(pytorch_ros2[0] - onnx_ros2[0]):.8f}, "
            f"speed={abs(pytorch_ros2[1] - onnx_ros2[1]):.8f}"
        )
    else:
        print("ONNX(ros2)    : skipped (model.onnx not found)")

    print(
        "abs(PyTorch(train)-PyTorch(ros2)): "
        f"steer={abs(pytorch_train[0] - pytorch_ros2[0]):.8f}, "
        f"speed={abs(pytorch_train[1] - pytorch_ros2[1]):.8f}"
    )

    if args.output_dir:
        output_dir = resolve_path(args.output_dir)
        save_outputs(
            output_dir,
            source_preview=preview_to_uint8(sample.raw_image),
            extractor_preview=preview_to_uint8(extractor_image),
            train_preview=train_preview,
            ros2_preview=ros2_preview,
            summary=summary,
        )
        print(f"\nSaved previews and summary to: {output_dir}")


if __name__ == "__main__":
    main()
