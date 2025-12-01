import argparse
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


class RETFoundOCTDataset(Dataset):
    """
    Wraps an ImageFolder dataset and applies the RETFound image processor.
    """

    def __init__(self, base_dataset: ImageFolder, processor: AutoImageProcessor):
        self.base = base_dataset
        self.processor = processor

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img, label = self.base[idx]  # img is a PIL Image if ImageFolder.transform is None
        if not isinstance(img, Image.Image):
            # Just in case transform was set unexpectedly
            img = Image.fromarray(img)

        enc = self.processor(images=img, return_tensors="pt")
        pixel_values = enc["pixel_values"].squeeze(0)  # [C, H, W]
        return pixel_values, label


def collate_fn(batch):
    pixel_values = torch.stack([b[0] for b in batch])      # [B, C, H, W]
    labels = torch.tensor([b[1] for b in batch]).long()    # [B]
    return pixel_values, labels


def build_dataloaders(data_root: Path, processor, batch_size: int, num_workers: int = 4):
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
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn,
    )

    return train_loader, val_loader, test_loader, classes


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


def train(
    data_root: str,
    model_id: str,
    output_dir: str,
    batch_size: int = 16,
    epochs: int = 5,
    lr: float = 5e-5,
    weight_decay: float = 0.05,
    num_workers: int = 4,
):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    data_root = Path(data_root)
    assert (data_root / "train").exists(), f"{data_root}/train not found"
    assert (data_root / "val").exists(),   f"{data_root}/val not found"
    assert (data_root / "test").exists(),  f"{data_root}/test not found"

    # 1) Load processor
    print(f"Loading image processor from '{model_id}'...")
    processor = AutoImageProcessor.from_pretrained(model_id)

    # 2) Build dataloaders
    train_loader, val_loader, test_loader, classes = build_dataloaders(
        data_root=data_root,
        processor=processor,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    num_classes = len(classes)
    id2label = {i: c for i, c in enumerate(classes)}
    label2id = {c: i for i, c in id2label.items()}

    # 3) Load RETFound model as a classifier
    print(f"Loading RETFound model from '{model_id}'...")
    config = AutoConfig.from_pretrained(model_id)
    config.num_labels = num_classes
    config.id2label = id2label
    config.label2id = label2id

    model = AutoModelForImageClassification.from_pretrained(
        model_id,
        config=config,
        ignore_mismatched_sizes=True,  # replace classification head
    )
    model.to(device)

    print("Model loaded with", config.num_labels, "labels.")
    print(model.config)

    # 4) Optimizer & scheduler
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    # 5) Training loop
    os.makedirs(output_dir, exist_ok=True)
    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        print(f"\nEpoch {epoch}/{epochs}")
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
            ckpt_path = Path(output_dir) / "best_model.pth"
            torch.save(model.state_dict(), ckpt_path)
            print(f"  New best model saved to {ckpt_path} (val_acc={val_acc:.4f})")

    # 6) Final test evaluation (using best model)
    print("\nLoading best model for final test evaluation...")
    best_ckpt = Path(output_dir) / "best_model.pth"
    if best_ckpt.exists():
        model.load_state_dict(torch.load(best_ckpt, map_location=device))
    else:
        print("Warning: best_model.pth not found, using last epoch weights.")

    test_loss, test_acc = evaluate(model, test_loader, device)
    print(f"\nTest loss: {test_loss:.4f} | Test accuracy: {test_acc:.4f}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Fine-tune RETFound (MAE OCT) on Kermany2018 OCT dataset."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Path to OCT2017 directory (containing train/ val/ test/).",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="iszt/RETFound_mae_natureOCT",
        help="HuggingFace model id for RETFound (OCT MAE).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./retfound_oct_finetuned",
        help="Directory to save best_model.pth and logs.",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size for training."
    )
    parser.add_argument(
        "--epochs", type=int, default=5, help="Number of training epochs."
    )
    parser.add_argument(
        "--lr", type=float, default=5e-5, help="Learning rate."
    )
    parser.add_argument(
        "--weight-decay", type=float, default=0.05, help="Weight decay."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    train(
        data_root=args.data_root,
        model_id=args.model_id,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.weight_decay,
        num_workers=args.num_workers,
    )
