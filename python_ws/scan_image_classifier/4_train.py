from __future__ import annotations

import os
import random
from typing import Dict, List, Optional

import hydra
import numpy as np
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.constants import LABEL_TO_ID
from src.dataset import ScanImageClassificationDataset
from src.model import create_classifier_model, set_backbone_trainable
from src.transforms import (
    Compose,
    ConvertToGray3Channel,
    RandomBrightnessContrast,
    RandomCutout,
    RandomGaussianNoise,
    RandomGaussianBlur,
    RandomHorizontalFlip,
    RandomRotate,
    RandomTranslateScale,
    ResizeImage,
    ToTensorNormalize,
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_train_transform(cfg: DictConfig):
    return Compose(
        [
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            ConvertToGray3Channel(enabled=cfg.dataset.force_grayscale_3ch),
            RandomHorizontalFlip(cfg.dataset.random_horizontal_flip_p),
            RandomRotate(
                probability=cfg.dataset.rotation_prob,
                max_degrees=cfg.dataset.rotation_deg,
                border_value=cfg.dataset.augment_fill_value,
            ),
            RandomTranslateScale(
                probability=cfg.dataset.affine_prob,
                max_translate_ratio=cfg.dataset.affine_translate_ratio,
                min_scale=cfg.dataset.affine_scale_min,
                max_scale=cfg.dataset.affine_scale_max,
                border_value=cfg.dataset.augment_fill_value,
            ),
            RandomBrightnessContrast(
                brightness=cfg.dataset.brightness_jitter,
                contrast=cfg.dataset.contrast_jitter,
            ),
            RandomGaussianBlur(
                probability=cfg.dataset.blur_prob,
                kernel_size=cfg.dataset.blur_kernel_size,
            ),
            RandomGaussianNoise(
                probability=cfg.dataset.noise_prob,
                sigma=cfg.dataset.noise_sigma,
            ),
            RandomCutout(
                probability=cfg.dataset.cutout_prob,
                min_ratio=cfg.dataset.cutout_min_ratio,
                max_ratio=cfg.dataset.cutout_max_ratio,
                fill_value=cfg.dataset.augment_fill_value,
            ),
            ToTensorNormalize(mean=cfg.dataset.pixel_mean, std=cfg.dataset.pixel_std),
        ]
    )


def build_eval_transform(cfg: DictConfig):
    return Compose(
        [
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            ConvertToGray3Channel(enabled=cfg.dataset.force_grayscale_3ch),
            ToTensorNormalize(mean=cfg.dataset.pixel_mean, std=cfg.dataset.pixel_std),
        ]
    )


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += int((logits.argmax(dim=1) == labels).sum().item())
        total_samples += labels.size(0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
    }


def evaluate(model, dataloader, criterion, device, num_classes: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            logits = model(images)
            loss = criterion(logits, labels)

            preds = logits.argmax(dim=1)
            total_loss += loss.item() * labels.size(0)
            total_correct += int((preds == labels).sum().item())
            total_samples += labels.size(0)

            for gt, pred in zip(labels.cpu().numpy(), preds.cpu().numpy()):
                confusion[int(gt), int(pred)] += 1

    recall = []
    for class_index in range(num_classes):
        denominator = confusion[class_index].sum()
        recall.append(float(confusion[class_index, class_index] / denominator) if denominator > 0 else 0.0)

    return {
        "loss": total_loss / max(total_samples, 1),
        "accuracy": total_correct / max(total_samples, 1),
        "confusion": confusion,
        "recall": recall,
    }


def compute_class_weights(dataset: ScanImageClassificationDataset, labels: List[str]) -> Optional[torch.Tensor]:
    counts = dataset.class_counts()
    if any(counts[label] <= 0 for label in labels):
        return None

    frequencies = np.asarray([counts[label] for label in labels], dtype=np.float32)
    weights = frequencies.sum() / (len(frequencies) * frequencies)
    return torch.tensor(weights, dtype=torch.float32)


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    set_seed(int(cfg.training.seed))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    labels = list(cfg.data.labels)
    num_classes = int(cfg.model.num_classes)
    if len(labels) != num_classes:
        raise ValueError("model.num_classes must match data.labels length")

    log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
    ckpt_dir = hydra.utils.to_absolute_path(cfg.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_dataset = ScanImageClassificationDataset(
        dataset_root=hydra.utils.to_absolute_path(cfg.data.dataset_root),
        annotations_path=hydra.utils.to_absolute_path(cfg.data.annotations_file),
        split=cfg.data.train_split,
        labels=labels,
        transform=build_train_transform(cfg),
        require_reviewed=cfg.data.require_reviewed,
        import_ids=cfg.data.import_ids,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = None
    try:
        val_dataset = ScanImageClassificationDataset(
            dataset_root=hydra.utils.to_absolute_path(cfg.data.dataset_root),
            annotations_path=hydra.utils.to_absolute_path(cfg.data.annotations_file),
            split=cfg.data.val_split,
            labels=labels,
            transform=build_eval_transform(cfg),
            require_reviewed=cfg.data.require_reviewed,
            import_ids=cfg.data.import_ids,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    except RuntimeError:
        val_dataset = None

    try:
        test_dataset = ScanImageClassificationDataset(
            dataset_root=hydra.utils.to_absolute_path(cfg.data.dataset_root),
            annotations_path=hydra.utils.to_absolute_path(cfg.data.annotations_file),
            split=cfg.data.test_split,
            labels=labels,
            transform=build_eval_transform(cfg),
            require_reviewed=cfg.data.require_reviewed,
            import_ids=cfg.data.import_ids,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
            pin_memory=torch.cuda.is_available(),
        )
    except RuntimeError:
        test_loader = None

    model = create_classifier_model(
        architecture=cfg.model.architecture,
        num_classes=num_classes,
        pretrained=cfg.model.pretrained,
        dropout=cfg.model.dropout,
    ).to(device)

    if cfg.model.pretrained and int(cfg.training.freeze_backbone_epochs) > 0:
        set_backbone_trainable(model, cfg.model.architecture, trainable=False)
        backbone_frozen = True
    else:
        backbone_frozen = False

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.training.learning_rate,
        weight_decay=cfg.training.weight_decay,
    )

    class_weights = None
    if cfg.training.use_class_weights:
        class_weights = compute_class_weights(train_dataset, labels)
        if class_weights is not None:
            class_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        label_smoothing=float(cfg.training.label_smoothing),
    )

    start_epoch = 0
    best_metric = -1.0
    resume_path = cfg.get("resume_ckpt_path", None)
    if resume_path:
        resume_path_abs = hydra.utils.to_absolute_path(resume_path)
        if os.path.exists(resume_path_abs):
            print(f"Resuming from {resume_path_abs}")
            checkpoint = torch.load(resume_path_abs, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            start_epoch = checkpoint["epoch"] + 1
            best_metric = float(checkpoint.get("best_metric", -1.0))

    train_counts = train_dataset.class_counts()
    print(f"Train class counts: {train_counts}")
    print(f"Train import counts: {train_dataset.import_counts()}")

    for epoch in range(start_epoch, cfg.training.epochs):
        if backbone_frozen and epoch >= int(cfg.training.freeze_backbone_epochs):
            print("Unfreezing pretrained backbone.")
            set_backbone_trainable(model, cfg.model.architecture, trainable=True)
            backbone_frozen = False

        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)

        current_metric = train_metrics["accuracy"]
        val_log = ""
        if val_loader is not None:
            val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
            current_metric = val_metrics["accuracy"]
            writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
            for class_index, recall_value in enumerate(val_metrics["recall"]):
                writer.add_scalar(f"Recall/val/{labels[class_index]}", recall_value, epoch)
            val_log = f", Val Loss: {val_metrics['loss']:.4f}, Val Acc: {val_metrics['accuracy']:.4f}"

        print(
            f"Epoch [{epoch}/{cfg.training.epochs - 1}] "
            f"Train Loss: {train_metrics['loss']:.4f}, Train Acc: {train_metrics['accuracy']:.4f}{val_log}"
        )

        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "model_architecture": cfg.model.architecture,
            "labels": labels,
            "label_to_id": LABEL_TO_ID,
            "image_height": int(cfg.dataset.image_height),
            "image_width": int(cfg.dataset.image_width),
            "input_channels": int(cfg.dataset.input_channels),
            "force_grayscale_3ch": bool(cfg.dataset.force_grayscale_3ch),
            "pixel_mean": list(cfg.dataset.pixel_mean),
            "pixel_std": list(cfg.dataset.pixel_std),
        }

        if current_metric > best_metric:
            best_metric = current_metric
            checkpoint_data["best_metric"] = best_metric
            torch.save(checkpoint_data, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"New Best Model (Acc: {best_metric:.4f})")

        torch.save(checkpoint_data, os.path.join(ckpt_dir, "last_model.pth"))

    if test_loader is not None:
        best_path = os.path.join(ckpt_dir, "best_model.pth")
        if os.path.exists(best_path):
            checkpoint = torch.load(best_path, map_location=device)
            model.load_state_dict(checkpoint["model_state_dict"])
        test_metrics = evaluate(model, test_loader, criterion, device, num_classes)
        print(
            f"Test Loss: {test_metrics['loss']:.4f}, "
            f"Test Acc: {test_metrics['accuracy']:.4f}, "
            f"Recall: {dict(zip(labels, test_metrics['recall']))}"
        )

    writer.close()
    print("Training Complete.")


if __name__ == "__main__":
    main()
