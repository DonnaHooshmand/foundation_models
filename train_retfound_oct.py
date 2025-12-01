"""
main.py

Fine-tune RETFound (OCT MAE) on the Kermany2018 OCT2017 dataset
using a local GPU (if available).

Usage:
    python main.py

Before running:
1. Install dependencies:
   pip install "transformers>=4.40.0" timm torchvision accelerate huggingface_hub

2. Log in to HuggingFace from this machine (once):
   huggingface-cli login

3. Make sure you have access to the gated model:
   - Visit https://huggingface.co/iszt/RETFound_mae_natureOCT
   - Log in and click "Agree and access" / request access
"""

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import ImageFolder
from PIL import Image

from transformers import (
    AutoImageProcessor,
    AutoConfig,
    AutoModelForImageClassification,
)
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.nn.functional as F
from tqdm.auto import tqdm


# ============================================================
#                 CONFIGURATION – EDIT THIS
# ============================================================

class Config:
    # Path to OCT2017 directory (containing train/ val/ test/)
    DATA_ROOT = Path("/home/donna/Desktop/datasets/OCT2017")  # <-- EDIT THIS

    # HuggingFace model id for RETFound (OCT MAE)
    MODEL_ID = "iszt/RETFound_mae_natureOCT"

    # Training hyperparameters
    OUTPUT_DIR = Path("./retfound_oct_finetuned")
    BATCH_SIZE = 16
    EPOCHS = 5
    LR = 5e-5
    WEIGHT_DECAY = 0.05
    NUM_WORKERS = 4


# ============================================================
#                    DATASET & DATALOADERS
# ============================================================

class RETFoundOCTDataset(Dataset):
    """
    Wraps an ImageFolder dataset and applies the RETFound image processor.
    """

    def __init__(self, base_dataset: ImageFolder, processor: AutoImageProcessor):
        self.base = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]  # img is PIL.Image if transform is None

        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)

        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)  # [C, H, W]
        return pixel_values, label


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])      # [B, C, H, W]
    labels = torch.tensor([b[1] for b in batch]).long()    # [B]
    return pixel_values, labels


def build_dataloaders(cfg: Config, processor):
    data_root = cfg.DATA_ROOT

    train_base = ImageFolder(data_root / "train")
    val_base   = ImageFolder(data_root / "val")
    test_base  = ImageFolder(data_root / "test")

    classes = train_base.classes
    print("Classes:", classes)

    train_ds = RETFoundOCTDataset(train_base, processor)
    val_ds   = RETFoundOCTDataset(val_base,   processor)
    test_ds  = RETFoundOCTDataset(test_base,  processor)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=True,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.NUM_WORKERS,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, classes


# ============================================================
#                    TRAIN / EVAL HELPERS
# ============================================================

def evaluate(model, loader, device) -> Tuple[float, float]:
    """
    Returns: (avg_loss, accuracy)
    """
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.no_grad():
        for pixel_values, labels in tqdm(loader, leave=False):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_correct += correct
            total_samples += bs

    avg_loss = total_loss / total_samples
    acc = total_correct / total_samples
    return avg_loss, acc


def train(cfg: Config):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Sanity checks on data root
    assert (cfg.DATA_ROOT / "train").exists(), f"{cfg.DATA_ROOT}/train not found"
    assert (cfg.DATA_ROOT / "val").exists(),   f"{cfg.DATA_ROOT}/val not found"
    assert (cfg.DATA_ROOT / "test").exists(),  f"{cfg.DATA_ROOT}/test not found"

    # 1) Load image processor
    print(f"Loading image processor from '{cfg.MODEL_ID}'...")
    processor = AutoImageProcessor.from_pretrained(cfg.MODEL_ID)

    # 2) Build dataloaders
    train_loader, val_loader, test_loader, classes = build_dataloaders(cfg, processor)

    num_classes = len(classes)
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in id2label.items()}

    # 3) Load RETFound model as classifier
    print(f"Loading RETFound model from '{cfg.MODEL_ID}'...")
    config = AutoConfig.from_pretrained(cfg.MODEL_ID)
    config.num_labels = num_classes
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForImageClassification.from_pretrained(
        cfg.MODEL_ID,
        config=config,
        ignore_mismatched_sizes=True,  # replace classifier head
    )
    model.to(device)

    print("Model loaded with", config.num_labels, "labels.")
    print(model.config)

    # 4) Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=cfg.LR, weight_decay=cfg.WEIGHT_DECAY)
    scheduler = CosineAnnealingLR(optimizer, T_max=cfg.EPOCHS)

    # 5) Training loop
    cfg.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, cfg.EPOCHS + 1):
        print(f"\nEpoch {epoch}/{cfg.EPOCHS}")
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_samples = 0

        for pixel_values, labels in tqdm(train_loader):
            pixel_values = pixel_values.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            outputs = model(pixel_values, labels=labels)
            loss = outputs.loss
            logits = outputs.logits

            loss.backward()
            optimizer.step()

            preds = logits.argmax(dim=1)
            correct = (preds == labels).sum().item()
            bs = labels.size(0)

            total_loss += loss.item() * bs
            total_correct += correct
            total_samples += bs

        train_loss = total_loss / total_samples
        train_acc = total_correct / total_samples

        val_loss, val_acc = evaluate(model, val_loader, device)
        scheduler.step()

        print(f"  Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"  Val   loss: {val_loss:.4f} | Val   acc: {val_acc:.4f}")

        # Save best checkpoint
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            ckpt_path = cfg.OUTPUT_DIR / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  New best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

    # 6) Final test evaluation
    print("\nLoading best model for final test evaluation...")
    best_ckpt = cfg.OUTPUT_DIR / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    else:
        print("Warning: best_model.pth not found, using last epoch weights.")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")


# ============================================================
#                           MAIN
# ============================================================

if __name__ == "__main__":
    cfg = Config()
    print("Config:")
    print(f"  DATA_ROOT   = {cfg.DATA_ROOT}")
    print(f"  MODEL_ID    = {cfg.MODEL_ID}")
    print(f"  OUTPUT_DIR  = {cfg.OUTPUT_DIR}")
    print(f"  BATCH_SIZE  = {cfg.BATCH_SIZE}")
    print(f"  EPOCHS      = {cfg.EPOCHS}")
    print(f"  LR          = {cfg.LR}")
    print(f"  WEIGHT_DECAY= {cfg.WEIGHT_DECAY}")
    print(f"  NUM_WORKERS = {cfg.NUM_WORKERS}")

    train(cfg)
