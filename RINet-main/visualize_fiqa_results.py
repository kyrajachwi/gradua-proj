import argparse
import csv
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def read_predictions(csv_path: str) -> List[Dict[str, float]]:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"Image", "MOS", "Pred"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"{csv_path} must contain columns: Image, MOS, Pred")
        for row in reader:
            mos = float(row["MOS"])
            pred = float(row["Pred"])
            rows.append(
                {
                    "Image": row["Image"],
                    "MOS": mos,
                    "Pred": pred,
                    "Error": abs(pred - mos),
                }
            )
    if not rows:
        raise RuntimeError(f"No rows found in {csv_path}")
    return rows


def resolve_image_path(folder: str, image_name: str) -> str:
    path = os.path.join(folder, image_name)
    if os.path.isfile(path):
        return path
    stem, _ = os.path.splitext(image_name)
    for ext in IMAGE_EXTENSIONS:
        candidate = os.path.join(folder, stem + ext)
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"Image not found: {path}")


def load_rgb(path: str, size: Tuple[int, int]) -> Image.Image:
    return Image.open(path).convert("RGB").resize(size, Image.BICUBIC)


def get_font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "arial.ttf",
        "DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "C:/Windows/Fonts/times.ttf",
    ]
    for path in candidates:
        try:
            return ImageFont.truetype(path, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def draw_label(draw: ImageDraw.ImageDraw, xy: Tuple[int, int], text: str, font: ImageFont.ImageFont) -> None:
    draw.text(xy, text, fill=(20, 24, 31), font=font)


def add_caption(image: Image.Image, caption: str, height: int = 42) -> Image.Image:
    font = get_font(18)
    canvas = Image.new("RGB", (image.width, image.height + height), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), caption, font=font)
    x = max(0, (image.width - (bbox[2] - bbox[0])) // 2)
    y = image.height + max(0, (height - (bbox[3] - bbox[1])) // 2 - 2)
    draw.text((x, y), caption, fill=(20, 24, 31), font=font)
    return canvas


def hstack(images: List[Image.Image], gap: int = 14, bg=(255, 255, 255)) -> Image.Image:
    width = sum(img.width for img in images) + gap * (len(images) - 1)
    height = max(img.height for img in images)
    canvas = Image.new("RGB", (width, height), bg)
    x = 0
    for img in images:
        canvas.paste(img, (x, 0))
        x += img.width + gap
    return canvas


def vstack(images: List[Image.Image], gap: int = 18, bg=(255, 255, 255)) -> Image.Image:
    width = max(img.width for img in images)
    height = sum(img.height for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGB", (width, height), bg)
    y = 0
    for img in images:
        x = (width - img.width) // 2
        canvas.paste(img, (x, y))
        y += img.height + gap
    return canvas


def save_scatter(rows: List[Dict[str, float]], output_path: str) -> None:
    mos = np.array([row["MOS"] for row in rows], dtype=np.float32)
    pred = np.array([row["Pred"] for row in rows], dtype=np.float32)
    plcc = np.corrcoef(mos, pred)[0, 1] if len(rows) > 1 else 0.0
    mse = np.mean((mos - pred) ** 2)
    mae = np.mean(np.abs(mos - pred))

    plt.figure(figsize=(6.2, 5.4), dpi=300)
    plt.scatter(mos, pred, s=8, alpha=0.45, c="#2563eb", edgecolors="none")
    plt.plot([0, 1], [0, 1], color="#dc2626", linewidth=1.4, linestyle="--")
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Ground-truth MOS")
    plt.ylabel("Predicted MOS")
    plt.title("Predicted MOS vs Ground-truth MOS")
    plt.grid(True, color="#e5e7eb", linewidth=0.8)
    plt.text(
        0.04,
        0.93,
        f"PLCC={plcc:.4f}\nMSE={mse:.4f}\nMAE={mae:.4f}",
        transform=plt.gca().transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", edgecolor="#d1d5db"),
    )
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def save_error_histogram(rows: List[Dict[str, float]], output_path: str) -> None:
    errors = np.array([row["Error"] for row in rows], dtype=np.float32)
    plt.figure(figsize=(6.2, 4.6), dpi=300)
    plt.hist(errors, bins=32, color="#0f766e", edgecolor="white", linewidth=0.6)
    plt.xlabel("Absolute Error |Pred - MOS|")
    plt.ylabel("Number of Images")
    plt.title("Prediction Error Distribution")
    plt.grid(True, axis="y", color="#e5e7eb", linewidth=0.8)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def make_roi_row(row: Dict[str, float], dataset_root: str, split: str, image_size: int, roi_size: int) -> Image.Image:
    split_dir = os.path.join(dataset_root, split)
    image_name = row["Image"]
    global_img = load_rgb(resolve_image_path(os.path.join(split_dir, "img"), image_name), (image_size, image_size))
    left = load_rgb(resolve_image_path(os.path.join(split_dir, "local_patches", "left_eye"), image_name), (roi_size, roi_size))
    right = load_rgb(resolve_image_path(os.path.join(split_dir, "local_patches", "right_eye"), image_name), (roi_size, roi_size))
    mouth = load_rgb(resolve_image_path(os.path.join(split_dir, "local_patches", "mouth"), image_name), (roi_size, roi_size))

    images = [
        add_caption(global_img, "Global"),
        add_caption(left, "Left Eye"),
        add_caption(right, "Right Eye"),
        add_caption(mouth, "Mouth"),
    ]
    strip = hstack(images)
    font = get_font(18)
    top = Image.new("RGB", (strip.width, 40), (255, 255, 255))
    draw = ImageDraw.Draw(top)
    text = f"{image_name}   MOS={row['MOS']:.3f}   Pred={row['Pred']:.3f}   Error={row['Error']:.3f}"
    draw_label(draw, (8, 8), text, font)
    return vstack([top, strip], gap=0)


def make_qualitative_row(row: Dict[str, float], dataset_root: str, split: str, image_size: int) -> Image.Image:
    split_dir = os.path.join(dataset_root, split)
    image_name = row["Image"]
    image = load_rgb(resolve_image_path(os.path.join(split_dir, "img"), image_name), (image_size, image_size))
    canvas = Image.new("RGB", (image.width, image.height + 76), (255, 255, 255))
    canvas.paste(image, (0, 0))
    draw = ImageDraw.Draw(canvas)
    title_font = get_font(17)
    value_font = get_font(16)
    draw_label(draw, (6, image.height + 8), image_name, title_font)
    draw_label(draw, (6, image.height + 33), f"GT {row['MOS']:.3f} | Pred {row['Pred']:.3f} | Err {row['Error']:.3f}", value_font)
    return canvas


def save_roi_examples(rows: List[Dict[str, float]], args: argparse.Namespace) -> None:
    selected = select_balanced_examples(rows, args.num_examples)
    panels = [make_roi_row(row, args.dataset_root, args.split, args.image_size, args.roi_size) for row in selected]
    vstack(panels).save(os.path.join(args.output_dir, "roi_examples.png"))


def save_qualitative_grid(rows: List[Dict[str, float]], args: argparse.Namespace, filename: str) -> None:
    panels = [make_qualitative_row(row, args.dataset_root, args.split, args.qual_image_size) for row in rows]
    lines = []
    for i in range(0, len(panels), args.grid_cols):
        lines.append(hstack(panels[i : i + args.grid_cols], gap=18))
    vstack(lines, gap=22).save(os.path.join(args.output_dir, filename))


def select_balanced_examples(rows: List[Dict[str, float]], count: int) -> List[Dict[str, float]]:
    ordered = sorted(rows, key=lambda item: item["MOS"])
    if count >= len(ordered):
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count).round().astype(int)
    return [ordered[i] for i in indices]


def main() -> None:
    parser = argparse.ArgumentParser(description="Create paper-ready FIQA visualizations.")
    parser.add_argument("--pred_csv", default="./result/FIQA/test_predictions.csv")
    parser.add_argument("--dataset_root", default="../FIQA")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--output_dir", default="./result/FIQA/figures")
    parser.add_argument("--num_examples", default=8, type=int)
    parser.add_argument("--grid_cols", default=4, type=int)
    parser.add_argument("--image_size", default=180, type=int)
    parser.add_argument("--roi_size", default=112, type=int)
    parser.add_argument("--qual_image_size", default=180, type=int)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    rows = read_predictions(args.pred_csv)
    rows_by_error = sorted(rows, key=lambda item: item["Error"])

    save_scatter(rows, os.path.join(args.output_dir, "scatter_gt_pred.png"))
    save_error_histogram(rows, os.path.join(args.output_dir, "error_histogram.png"))
    save_roi_examples(rows, args)
    save_qualitative_grid(rows_by_error[: args.num_examples], args, "qualitative_best_cases.png")
    save_qualitative_grid(rows_by_error[-args.num_examples :][::-1], args, "qualitative_worst_cases.png")

    print(f"saved figures to: {args.output_dir}")
    print(" - scatter_gt_pred.png")
    print(" - error_histogram.png")
    print(" - roi_examples.png")
    print(" - qualitative_best_cases.png")
    print(" - qualitative_worst_cases.png")


if __name__ == "__main__":
    main()
