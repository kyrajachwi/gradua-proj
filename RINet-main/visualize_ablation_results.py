import argparse
import csv
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_summary(path: str) -> List[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        rows = json.load(f)
    if not rows:
        raise RuntimeError(f"No ablation rows found in {path}")
    return rows


def save_csv(rows: List[Dict[str, object]], path: str) -> None:
    fields = ["name", "label", "last_epoch", "val_mse", "val_mae", "val_plcc", "best_mse", "output_dir"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def save_bar_chart(rows: List[Dict[str, object]], metric: str, ylabel: str, output_path: str) -> None:
    labels = [str(row["label"]) for row in rows]
    values = np.array([float(row[metric]) for row in rows], dtype=np.float32)

    plt.figure(figsize=(8.2, 4.8), dpi=300)
    colors = ["#64748b", "#2563eb", "#0f766e", "#7c3aed", "#dc2626"][: len(rows)]
    bars = plt.bar(np.arange(len(rows)), values, color=colors, width=0.62)
    plt.xticks(np.arange(len(rows)), labels, rotation=22, ha="right")
    plt.ylabel(ylabel)
    plt.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)

    for bar, value in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    best_idx = int(np.argmax(values)) if metric == "val_plcc" else int(np.argmin(values))
    plt.title(f"Ablation Study: {ylabel} (best: {labels[best_idx]})")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize FIQA ablation results.")
    parser.add_argument("--summary_json", default="./result/FIQA_ablation/ablation_summary.json")
    parser.add_argument("--output_dir", default="./result/FIQA_ablation/figures")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = load_summary(args.summary_json)
    save_csv(rows, os.path.join(args.output_dir, "ablation_table.csv"))
    save_bar_chart(rows, "val_mse", "Validation MSE", os.path.join(args.output_dir, "ablation_mse.png"))
    save_bar_chart(rows, "val_mae", "Validation MAE", os.path.join(args.output_dir, "ablation_mae.png"))
    save_bar_chart(rows, "val_plcc", "Validation PLCC", os.path.join(args.output_dir, "ablation_plcc.png"))

    print(f"saved ablation figures to: {args.output_dir}")
    print(" - ablation_table.csv")
    print(" - ablation_mse.png")
    print(" - ablation_mae.png")
    print(" - ablation_plcc.png")


if __name__ == "__main__":
    main()
