import argparse
import json
import os
import sys
from typing import Iterable, Tuple

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR in sys.path:
    sys.path.remove(SCRIPT_DIR)

import cv2
import numpy as np
import torch
from tqdm import tqdm

sys.path.append(SCRIPT_DIR)
from pfld import PFLDInference


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def crop_patch(img: np.ndarray, center: Tuple[float, float], size: int) -> np.ndarray:
    h, w, _ = img.shape
    cx, cy = int(center[0]), int(center[1])
    half = size // 2

    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size

    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    cropped = img[y1:y2, x1:x2]

    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        cropped = cv2.copyMakeBorder(
            cropped,
            pad_t,
            pad_b,
            pad_l,
            pad_r,
            cv2.BORDER_CONSTANT,
            value=[0, 0, 0],
        )

    return cv2.resize(cropped, (size, size))


def load_image_names(split_dir: str) -> Iterable[str]:
    label_path = os.path.join(split_dir, "label.json")
    if os.path.isfile(label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)
        return labels["Image"]
    img_dir = os.path.join(split_dir, "img")
    return sorted(name for name in os.listdir(img_dir) if name.lower().endswith(IMAGE_EXTENSIONS))


def load_pfld(checkpoint_path: str, device: torch.device) -> PFLDInference:
    model = PFLDInference().to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    state_dict = ckpt["pfld_backbone"] if isinstance(ckpt, dict) and "pfld_backbone" in ckpt else ckpt
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def infer_landmarks(model: PFLDInference, img: np.ndarray, device: torch.device, input_size: int) -> np.ndarray:
    h, w, _ = img.shape
    img_resized = cv2.resize(img, (input_size, input_size))
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img_tensor)
        landmarks = output[1] if isinstance(output, tuple) else output
        landmarks = landmarks[0].cpu().numpy().reshape(-1, 2)

    landmarks[:, 0] *= w
    landmarks[:, 1] *= h
    return landmarks


def process_split(args: argparse.Namespace, split: str, model: PFLDInference, device: torch.device) -> None:
    split_dir = os.path.join(args.dataset_root, split)
    img_dir = os.path.join(split_dir, "img")
    landmark_dir = os.path.join(split_dir, "pfld_landmarks")
    output_root = os.path.join(split_dir, "local_patches")

    if not os.path.isdir(img_dir):
        raise FileNotFoundError(f"Missing image directory: {img_dir}")

    for sub in ["left_eye", "right_eye", "mouth"]:
        os.makedirs(os.path.join(output_root, sub), exist_ok=True)
    os.makedirs(landmark_dir, exist_ok=True)

    image_names = list(load_image_names(split_dir))
    print(f"{split}: processing {len(image_names)} images")

    skipped = 0
    failed = 0
    for img_name in tqdm(image_names, desc=f"{split} ROI"):
        if not img_name.lower().endswith(IMAGE_EXTENSIONS):
            continue

        left_path = os.path.join(output_root, "left_eye", img_name)
        right_path = os.path.join(output_root, "right_eye", img_name)
        mouth_path = os.path.join(output_root, "mouth", img_name)
        if args.skip_existing and all(os.path.isfile(p) for p in [left_path, right_path, mouth_path]):
            skipped += 1
            continue

        img_path = os.path.join(img_dir, img_name)
        img = cv2.imread(img_path)
        if img is None:
            failed += 1
            continue

        landmark_path = os.path.join(landmark_dir, os.path.splitext(img_name)[0] + ".npy")
        if args.skip_existing and os.path.isfile(landmark_path):
            landmarks = np.load(landmark_path)
        else:
            landmarks = infer_landmarks(model, img, device, args.pfld_input_size)
            np.save(landmark_path, landmarks)

        left_center = np.mean(landmarks[60:68], axis=0)
        right_center = np.mean(landmarks[68:76], axis=0)
        mouth_center = np.mean(landmarks[76:96], axis=0)

        cv2.imwrite(left_path, crop_patch(img, left_center, args.patch_size))
        cv2.imwrite(right_path, crop_patch(img, right_center, args.patch_size))
        cv2.imwrite(mouth_path, crop_patch(img, mouth_center, args.patch_size))

    print(f"{split}: done, skipped={skipped}, failed={failed}, output={output_root}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate FIQA PFLD landmarks and eye/mouth ROI patches.")
    parser.add_argument("--dataset_root", default=r"D:\Desktop\study_life\gradua-proj\FIQA")
    parser.add_argument("--splits", nargs="+", default=["val", "test"], choices=["train", "val", "test"])
    parser.add_argument("--checkpoint", default=r"D:\Desktop\study_life\gradua-proj\ADMNet-main\checkpoint.pth.tar")
    parser.add_argument("--patch_size", default=112, type=int)
    parser.add_argument("--pfld_input_size", default=112, type=int)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--skip_existing", action="store_true")
    args = parser.parse_args()

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    model = load_pfld(args.checkpoint, device)
    print(f"PFLD loaded on {device}")

    for split in args.splits:
        process_split(args, split, model, device)


if __name__ == "__main__":
    main()
