import argparse
import csv
import json
import os
import random
from typing import Dict, Iterable, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from fiqa_dataset import FIQADataset
from module.fiqa_multibranch import MultiBranchFIQANet


def parse_size(value: str) -> Tuple[int, int]:
    if isinstance(value, int):
        return value, value
    parts = str(value).lower().replace(",", "x").split("x")
    if len(parts) == 1:
        size = int(parts[0])
        return size, size
    if len(parts) == 2:
        return int(parts[0]), int(parts[1])
    raise argparse.ArgumentTypeError("size must be like 224 or 224x224")


def fix_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch: Dict[str, torch.Tensor], device: torch.device, use_mouth: bool) -> Dict[str, torch.Tensor]:
    keys = ["image", "left_eye", "right_eye", "score"]
    if use_mouth:
        keys.append("mouth")
    moved = {key: batch[key].to(device, non_blocking=True) for key in keys}
    return moved


def compute_metrics(preds: Iterable[float], targets: Iterable[float]) -> Dict[str, float]:
    pred = np.asarray(list(preds), dtype=np.float64)
    target = np.asarray(list(targets), dtype=np.float64)
    mse = float(np.mean((pred - target) ** 2))
    mae = float(np.mean(np.abs(pred - target)))
    if len(pred) > 1 and np.std(pred) > 0 and np.std(target) > 0:
        plcc = float(np.corrcoef(pred, target)[0, 1])
    else:
        plcc = 0.0
    return {"mse": mse, "mae": mae, "plcc": plcc}


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    use_mouth: bool,
    optimizer: torch.optim.Optimizer = None,
    desc: str = "train",
) -> Dict[str, float]:
    is_train = optimizer is not None
    model.train(is_train)
    losses = []
    preds = []
    targets = []

    context = torch.enable_grad() if is_train else torch.no_grad()
    with context:
        for batch in tqdm(loader, desc=desc):
            data = to_device(batch, device, use_mouth)
            pred = model(
                data["image"],
                data["left_eye"],
                data["right_eye"],
                data.get("mouth"),
            )
            target = data["score"].float().view_as(pred)
            loss = criterion(pred, target)

            if is_train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            losses.append(float(loss.item()))
            preds.extend(pred.detach().cpu().numpy().tolist())
            targets.extend(target.detach().cpu().numpy().tolist())

    metrics = compute_metrics(preds, targets)
    metrics["loss"] = float(np.mean(losses))
    return metrics


def save_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_mouth: bool,
    csv_path: str,
) -> None:
    model.eval()
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "MOS", "Pred"])
        with torch.no_grad():
            for batch in tqdm(loader, desc="predict"):
                data = to_device(batch, device, use_mouth)
                pred = model(data["image"], data["left_eye"], data["right_eye"], data.get("mouth"))
                for name, mos, score in zip(batch["name"], batch["score"], pred.cpu()):
                    writer.writerow([name, float(mos), float(score)])


def main() -> None:
    parser = argparse.ArgumentParser(description="Train multi-branch FIQA MOS regressor.")
    parser.add_argument("--dataset_root", default="../FIQA", type=str)
    parser.add_argument("--output_dir", default="./result/FIQA", type=str)
    parser.add_argument("--backbone", default="tiny", choices=["tiny", "resnet18"])
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision pretrained weights.")
    parser.add_argument("--no_left_eye", action="store_true", help="Disable left-eye ROI branch for ablation.")
    parser.add_argument("--no_right_eye", action="store_true", help="Disable right-eye ROI branch for ablation.")
    parser.add_argument("--use_mouth", action="store_true")
    parser.add_argument("--no_attention", action="store_true", help="Disable RINet-style attention branch for ablation.")
    parser.add_argument("--strict_roi", action="store_true")
    parser.add_argument("--image_size", default="224", type=parse_size)
    parser.add_argument("--roi_size", default="112", type=parse_size)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--lr", default=1e-4, type=float)
    parser.add_argument("--weight_decay", default=1e-4, type=float)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_val_samples", default=None, type=int)
    parser.add_argument("--save_test_predictions", action="store_true")
    args = parser.parse_args()

    fix_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    train_set = FIQADataset(
        args.dataset_root,
        "train",
        image_size=args.image_size,
        roi_size=args.roi_size,
        use_mouth=args.use_mouth,
        strict_roi=args.strict_roi,
        max_samples=args.max_train_samples,
    )
    val_set = FIQADataset(
        args.dataset_root,
        "val",
        image_size=args.image_size,
        roi_size=args.roi_size,
        use_mouth=args.use_mouth,
        strict_roi=args.strict_roi,
        max_samples=args.max_val_samples,
    )

    print(f"train samples: {len(train_set)}, ROI matched: {train_set.roi_report()}")
    print(f"val samples: {len(val_set)}, ROI matched: {val_set.roi_report()}")
    if any(count == 0 for count in val_set.roi_report().values()):
        print("warning: val split has missing ROI folders/files; zero ROI tensors will be used.")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MultiBranchFIQANet(
        backbone=args.backbone,
        use_left_eye=not args.no_left_eye,
        use_right_eye=not args.no_right_eye,
        use_mouth=args.use_mouth,
        use_attention=not args.no_attention,
        pretrained=args.pretrained,
    ).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))

    best_mse = float("inf")
    best_path = os.path.join(args.output_dir, "best_fiqa_model.pt")
    last_path = os.path.join(args.output_dir, "last_fiqa_model.pt")

    for epoch in range(1, args.epochs + 1):
        train_metrics = run_epoch(model, train_loader, criterion, device, args.use_mouth, optimizer, f"train {epoch}")
        val_metrics = run_epoch(model, val_loader, criterion, device, args.use_mouth, desc=f"val {epoch}")
        scheduler.step()

        print(
            f"epoch {epoch:03d} | "
            f"train loss {train_metrics['loss']:.5f} mse {train_metrics['mse']:.5f} mae {train_metrics['mae']:.5f} | "
            f"val loss {val_metrics['loss']:.5f} mse {val_metrics['mse']:.5f} mae {val_metrics['mae']:.5f} "
            f"plcc {val_metrics['plcc']:.4f}"
        )

        checkpoint = {
            "model": model.state_dict(),
            "args": vars(args),
            "epoch": epoch,
            "val_metrics": val_metrics,
        }
        torch.save(checkpoint, last_path)
        if val_metrics["mse"] < best_mse:
            best_mse = val_metrics["mse"]
            torch.save(checkpoint, best_path)
            print(f"saved best checkpoint: {best_path}")

        with open(os.path.join(args.output_dir, "metrics.json"), "w", encoding="utf-8") as f:
            json.dump(
                {
                    "best_mse": best_mse,
                    "last_epoch": epoch,
                    "last_val_metrics": val_metrics,
                    "last_train_metrics": train_metrics,
                    "args": vars(args),
                },
                f,
                indent=2,
            )

    if args.save_test_predictions:
        test_set = FIQADataset(
            args.dataset_root,
            "test",
            image_size=args.image_size,
            roi_size=args.roi_size,
            use_mouth=args.use_mouth,
            strict_roi=False,
        )
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        save_predictions(model, test_loader, device, args.use_mouth, os.path.join(args.output_dir, "test_predictions.csv"))


if __name__ == "__main__":
    main()
