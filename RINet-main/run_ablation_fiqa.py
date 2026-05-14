import argparse
import json
import os
import subprocess
import sys
from typing import Dict, List


ABLATIONS: List[Dict[str, object]] = [
    {
        "name": "A_global",
        "label": "Global",
        "extra_args": ["--no_left_eye", "--no_right_eye", "--no_attention"],
    },
    {
        "name": "B_global_attention",
        "label": "Global + Attention",
        "extra_args": ["--no_left_eye", "--no_right_eye"],
    },
    {
        "name": "C_global_eyes",
        "label": "Global + Eyes",
        "extra_args": ["--no_attention"],
    },
    {
        "name": "D_global_eyes_attention",
        "label": "Global + Eyes + Attention",
        "extra_args": [],
    },
    {
        "name": "E_global_eyes_mouth_attention",
        "label": "Global + Eyes + Mouth + Attention",
        "extra_args": ["--use_mouth"],
    },
]


def run_one(args: argparse.Namespace, ablation: Dict[str, object]) -> Dict[str, object]:
    output_dir = os.path.join(args.output_root, ablation["name"])
    command = [
        sys.executable,
        "train_fiqa.py",
        "--dataset_root",
        args.dataset_root,
        "--output_dir",
        output_dir,
        "--epochs",
        str(args.epochs),
        "--batch_size",
        str(args.batch_size),
        "--num_workers",
        str(args.num_workers),
        "--backbone",
        args.backbone,
        "--strict_roi",
    ]
    if args.max_train_samples is not None:
        command.extend(["--max_train_samples", str(args.max_train_samples)])
    if args.max_val_samples is not None:
        command.extend(["--max_val_samples", str(args.max_val_samples)])
    command.extend(ablation["extra_args"])

    print("\n" + "=" * 88)
    print(f"Running {ablation['name']}: {ablation['label']}")
    print(" ".join(command))
    print("=" * 88)
    subprocess.run(command, check=True)

    metrics_path = os.path.join(output_dir, "metrics.json")
    with open(metrics_path, "r", encoding="utf-8") as f:
        metrics = json.load(f)

    best_ckpt = os.path.join(output_dir, "best_fiqa_model.pt")
    if args.predict:
        pred_csv = os.path.join(output_dir, "test_predictions.csv")
        pred_cmd = [
            sys.executable,
            "predict_fiqa.py",
            "--dataset_root",
            args.dataset_root,
            "--checkpoint",
            best_ckpt,
            "--output_csv",
            pred_csv,
            "--split",
            "test",
            "--batch_size",
            str(args.predict_batch_size),
            "--num_workers",
            str(args.num_workers),
        ]
        subprocess.run(pred_cmd, check=True)

    row = {
        "name": ablation["name"],
        "label": ablation["label"],
        "output_dir": output_dir,
        "best_checkpoint": best_ckpt,
        "best_mse": metrics.get("best_mse"),
        "last_epoch": metrics.get("last_epoch"),
        "val_mse": metrics.get("last_val_metrics", {}).get("mse"),
        "val_mae": metrics.get("last_val_metrics", {}).get("mae"),
        "val_plcc": metrics.get("last_val_metrics", {}).get("plcc"),
    }
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FIQA ablation experiments.")
    parser.add_argument("--dataset_root", default="../FIQA")
    parser.add_argument("--output_root", default="./result/FIQA_ablation")
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--backbone", default="tiny", choices=["tiny", "resnet18"])
    parser.add_argument("--max_train_samples", default=None, type=int)
    parser.add_argument("--max_val_samples", default=None, type=int)
    parser.add_argument("--only", nargs="*", default=None, help="Optional ablation names to run.")
    parser.add_argument("--predict", action="store_true", help="Also export test predictions for each best checkpoint.")
    parser.add_argument("--predict_batch_size", default=64, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_root, exist_ok=True)
    selected = ABLATIONS
    if args.only:
        wanted = set(args.only)
        selected = [item for item in ABLATIONS if item["name"] in wanted]
        missing = wanted - {item["name"] for item in selected}
        if missing:
            raise ValueError(f"Unknown ablation names: {sorted(missing)}")

    summary = []
    for ablation in selected:
        summary.append(run_one(args, ablation))

    summary_path = os.path.join(args.output_root, "ablation_summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved summary: {summary_path}")


if __name__ == "__main__":
    main()
