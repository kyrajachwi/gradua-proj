import argparse
import csv
import os

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from fiqa_dataset import FIQADataset
from module.fiqa_multibranch import MultiBranchFIQANet
from train_fiqa import parse_size


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict FIQA MOS scores with a trained checkpoint.")
    parser.add_argument("--dataset_root", default="../FIQA", type=str)
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default="./result/FIQA/best_fiqa_model.pt", type=str)
    parser.add_argument("--output_csv", default="./result/FIQA/predictions.csv", type=str)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--num_workers", default=0, type=int)
    parser.add_argument("--image_size", default=None, type=parse_size)
    parser.add_argument("--roi_size", default=None, type=parse_size)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    train_args = checkpoint.get("args", {})

    image_size = args.image_size or tuple(train_args.get("image_size", (224, 224)))
    roi_size = args.roi_size or tuple(train_args.get("roi_size", (112, 112)))
    use_mouth = bool(train_args.get("use_mouth", False))
    use_left_eye = not bool(train_args.get("no_left_eye", False))
    use_right_eye = not bool(train_args.get("no_right_eye", False))
    use_attention = not bool(train_args.get("no_attention", False))
    backbone = train_args.get("backbone", "tiny")

    dataset = FIQADataset(
        args.dataset_root,
        args.split,
        image_size=image_size,
        roi_size=roi_size,
        use_mouth=use_mouth,
        strict_roi=False,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)

    model = MultiBranchFIQANet(
        backbone=backbone,
        use_left_eye=use_left_eye,
        use_right_eye=use_right_eye,
        use_mouth=use_mouth,
        use_attention=use_attention,
        pretrained=False,
    ).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Image", "MOS", "Pred"])
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"predict {args.split}"):
                image = batch["image"].to(device)
                left_eye = batch["left_eye"].to(device)
                right_eye = batch["right_eye"].to(device)
                mouth = batch["mouth"].to(device) if use_mouth else None
                pred = model(image, left_eye, right_eye, mouth)
                for name, mos, score in zip(batch["name"], batch["score"], pred.cpu()):
                    writer.writerow([name, float(mos), float(score)])
    print(f"saved predictions: {args.output_csv}")


if __name__ == "__main__":
    main()
