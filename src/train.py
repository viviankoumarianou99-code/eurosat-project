import argparse
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR

from src.data import get_dataloaders
from src.models import build_resnet50, build_efficientnet_b0, set_trainable_layers, get_device
from src.utils import set_seed, ensure_dir, save_json, timestamp


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        loss = criterion(logits, y)

        running_loss += loss.item() * x.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == y).sum().item()
        total += x.size(0)

    return running_loss / total, correct / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=r"data\raw\EuroSAT_RGB")
    parser.add_argument("--image_size", type=int, default=224)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--model", type=str, default="resnet50", choices=["resnet50", "efficientnet_b0"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_mode", type=str, default="head", choices=["head", "last_block", "all"])
    parser.add_argument("--out_dir", type=str, default=r"outputs")
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device()

    # Prepare output folders
    run_id = f"{args.model}_{args.train_mode}_{timestamp()}"
    run_dir = Path(args.out_dir) / run_id
    ensure_dir(str(run_dir))
    ensure_dir(str(run_dir / "checkpoints"))

    # Data
    train_loader, val_loader, test_loader, classes = get_dataloaders(
        args.data_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        seed=args.seed
    )
    num_classes = len(classes)

    # Model
    if args.model == "resnet50":
       model = build_resnet50(num_classes=num_classes, pretrained=True)
       print("Selected model:", args.model)
       print("Model class:", model.__class__.__name__)
    elif args.model == "efficientnet_b0":
        model = build_efficientnet_b0(num_classes=num_classes, pretrained=True)
    else:
        raise ValueError("Unknown model")

    set_trainable_layers(model, train_mode=args.train_mode)
    model = model.to(device)

    # Only optimize trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = Adam(trainable_params, lr=args.lr)
    scheduler = StepLR(optimizer, step_size=2, gamma=0.5)
    criterion = nn.CrossEntropyLoss()

    # Save run config
    save_json(
        {
            "run_id": run_id,
            "model": "resnet50",
            "train_mode": args.train_mode,
            "data_dir": args.data_dir,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "seed": args.seed,
            "classes": classes,
        },
        str(run_dir / "config.json")
    )

    history = []
    best_val_acc = -1.0
    best_ckpt_path = run_dir / "checkpoints" / "best.pth"

    print(f"Run: {run_id}")
    print(f"Classes: {num_classes}")
    print(f"Device: {device}")

    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        va_loss, va_acc = evaluate(model, val_loader, criterion, device)

        scheduler.step()

        epoch_log = {
            "epoch": epoch,
            "train_loss": tr_loss,
            "train_acc": tr_acc,
            "val_loss": va_loss,
            "val_acc": va_acc
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={tr_loss:.4f} train_acc={tr_acc:.4f} | "
            f"val_loss={va_loss:.4f} val_acc={va_acc:.4f}"
        )

        # Save best checkpoint
        if va_acc > best_val_acc:
            best_val_acc = va_acc
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "classes": classes,
                    "epoch": epoch,
                    "val_acc": va_acc,
                    "train_mode": args.train_mode
                },
                str(best_ckpt_path)
            )

    total_time = time.time() - start_time
    save_json(history, str(run_dir / "history.json"))
    save_json({"best_val_acc": best_val_acc, "train_seconds": total_time}, str(run_dir / "summary.json"))

    print(f"✅ Finished. Best val acc = {best_val_acc:.4f}")
    print(f"Logs saved to: {run_dir}")
    print(f"Best checkpoint: {best_ckpt_path}")


if __name__ == "__main__":
    main()
