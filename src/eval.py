import argparse
from pathlib import Path
import json

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score

from src.data import get_dataloaders
from src.models import build_resnet50, build_efficientnet_b0, get_device


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0"],
        help="Model architecture to evaluate"
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        required=True,
        help="Path to training run folder (e.g., outputs/resnet50_head_...)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=r"data\raw\EuroSAT_RGB",
        help="Dataset folder path"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16
    )
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    ckpt_path = run_dir / "checkpoints" / "best.pth"

    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = get_device()

    # Data (TEST set)
    _, _, test_loader, class_names = get_dataloaders(
        args.data_dir,
        batch_size=args.batch_size
    )
    num_classes = len(class_names)

    # Build model (match architecture)
    if args.model == "resnet50":
        model = build_resnet50(num_classes=num_classes, pretrained=False)
    else:
        model = build_efficientnet_b0(num_classes=num_classes, pretrained=False)

    # Load checkpoint
    checkpoint = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    all_preds = []
    all_targets = []

    for x, y in test_loader:
        x = x.to(device)
        logits = model(x)
        preds = torch.argmax(F.softmax(logits, dim=1), dim=1)

        all_preds.extend(preds.cpu().tolist())
        all_targets.extend(y.tolist())

    # Metrics
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)

    print(f"✅ Model: {args.model}")
    print(f"✅ Test Accuracy: {acc:.4f}")

    # Save metrics json (so you never lose the number)
    metrics_path = run_dir / "test_metrics.json"
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"model": args.model, "test_accuracy": float(acc)}, f, indent=2)
    print(f"🧾 Metrics saved to: {metrics_path}")

    # Save confusion matrix plot
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix – {args.model} (Test Set) | Acc={acc:.4f}")
    plt.tight_layout()

    out_path = run_dir / "confusion_matrix.png"
    plt.savefig(out_path, dpi=150)
    plt.close()

    print(f"📊 Confusion matrix saved to: {out_path}")


if __name__ == "__main__":
    main()