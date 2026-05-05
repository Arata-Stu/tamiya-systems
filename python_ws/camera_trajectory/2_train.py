import os

import hydra
import torch
import torch.nn as nn
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from src.dataset import MultiImageTrajectoryDataset
from src.loss import TrajectoryLoss
from src.model import PilotNetTrajectory
from src.transform import AddImageNoise, Compose, CropImage, NormalizeImage, ResizeImage


def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["trajectory"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)


def validate_one_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation", leave=False):
            images = batch["image"].to(device)
            labels = batch["trajectory"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
    return total_loss / len(dataloader)


@hydra.main(config_path="config", config_name="train", version_base="1.2")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    log_dir = hydra.utils.to_absolute_path(cfg.log_dir)
    ckpt_dir = hydra.utils.to_absolute_path(cfg.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=log_dir)

    train_transform = Compose(
        [
            CropImage(top_ratio=cfg.dataset.crop_top_ratio, bottom_ratio=cfg.dataset.crop_bottom_ratio),
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            NormalizeImage(mean=cfg.dataset.pixel_mean, std=cfg.dataset.pixel_std),
            AddImageNoise(std=cfg.dataset.image_noise_std),
        ]
    )
    val_transform = Compose(
        [
            CropImage(top_ratio=cfg.dataset.crop_top_ratio, bottom_ratio=cfg.dataset.crop_bottom_ratio),
            ResizeImage(height=cfg.dataset.image_height, width=cfg.dataset.image_width),
            NormalizeImage(mean=cfg.dataset.pixel_mean, std=cfg.dataset.pixel_std),
        ]
    )

    train_dataset = MultiImageTrajectoryDataset(
        base_dir=os.path.join(hydra.utils.to_absolute_path(cfg.data_path), "train"),
        transform=train_transform,
        seq_len=cfg.dataset.sequence_length,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=cfg.training.num_workers,
    )

    val_path = os.path.join(hydra.utils.to_absolute_path(cfg.data_path), "test")
    val_loader = None
    if os.path.exists(val_path):
        val_dataset = MultiImageTrajectoryDataset(
            base_dir=val_path, transform=val_transform, seq_len=cfg.dataset.sequence_length
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=False,
            num_workers=cfg.training.num_workers,
        )

    model = PilotNetTrajectory(
        num_points=cfg.model.num_points,
        input_channels=cfg.dataset.input_channels,
        input_height=cfg.dataset.image_height,
        input_width=cfg.dataset.image_width,
        output_scale=cfg.model.output_scale,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.learning_rate)
    criterion = TrajectoryLoss(
        nn.SmoothL1Loss(),
        heading_weight=cfg.training.heading_loss_weight,
        curvature_weight=cfg.training.curvature_loss_weight,
        smoothness_weight=cfg.training.smoothness_loss_weight,
        progress_weight=cfg.training.progress_loss_weight,
    )

    start_epoch = 0
    best_metric = float("inf")
    resume_path = cfg.get("resume_ckpt_path", None)
    if resume_path:
        resume_path_abs = hydra.utils.to_absolute_path(resume_path)
        if os.path.exists(resume_path_abs):
            print(f"Resuming from {resume_path_abs}")
            ckpt = torch.load(resume_path_abs, map_location=device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            best_metric = ckpt.get("best_metric", float("inf"))

    for epoch in range(start_epoch, cfg.training.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        writer.add_scalar("Loss/train", train_loss, epoch)

        current_metric = train_loss
        val_log = ""
        if val_loader:
            val_loss = validate_one_epoch(model, val_loader, criterion, device)
            writer.add_scalar("Loss/validation", val_loss, epoch)
            current_metric = val_loss
            val_log = f", Val Loss: {val_loss:.4f}"

        print(f"Epoch [{epoch}/{cfg.training.epochs - 1}] Train Loss: {train_loss:.4f}{val_log}")

        ckpt_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_metric": best_metric,
            "num_points": cfg.model.num_points,
            "output_scale": cfg.model.output_scale,
        }
        if current_metric < best_metric:
            best_metric = current_metric
            ckpt_data["best_metric"] = best_metric
            torch.save(ckpt_data, os.path.join(ckpt_dir, "best_model.pth"))
            print(f"New Best Model (Loss: {best_metric:.4f})")

        torch.save(ckpt_data, os.path.join(ckpt_dir, "last_model.pth"))

    writer.close()
    print("Training Complete.")


if __name__ == "__main__":
    main()
