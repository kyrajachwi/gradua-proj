import json
import os
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def _build_transform(size: Tuple[int, int]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def _resolve_roi_dir(split_dir: str, roi_name: str) -> Optional[str]:
    candidates = [
        os.path.join(split_dir, roi_name),
        os.path.join(split_dir, "local_patches", roi_name),
    ]
    for path in candidates:
        if os.path.isdir(path):
            return path
    return None


class FIQADataset(Dataset):
    """Dataset for MOS regression with global and optional local ROI images."""

    def __init__(
        self,
        dataset_root: str,
        split: str,
        image_size: Tuple[int, int] = (224, 224),
        roi_size: Tuple[int, int] = (112, 112),
        use_mouth: bool = False,
        strict_roi: bool = False,
        max_samples: Optional[int] = None,
    ) -> None:
        self.split_dir = os.path.join(dataset_root, split)
        self.img_dir = os.path.join(self.split_dir, "img")
        self.label_path = os.path.join(self.split_dir, "label.json")
        self.use_mouth = use_mouth
        self.strict_roi = strict_roi
        self.image_transform = _build_transform(image_size)
        self.roi_transform = _build_transform(roi_size)
        self.zero_roi = torch.zeros(3, roi_size[0], roi_size[1])

        if not os.path.isdir(self.img_dir):
            raise FileNotFoundError(f"Image directory not found: {self.img_dir}")
        if not os.path.isfile(self.label_path):
            raise FileNotFoundError(f"Label file not found: {self.label_path}")

        with open(self.label_path, "r", encoding="utf-8") as f:
            labels = json.load(f)

        images = labels.get("Image")
        mos = labels.get("MOS")
        if not isinstance(images, list) or not isinstance(mos, list):
            raise ValueError(f"{self.label_path} must contain list fields: Image and MOS")
        if len(images) != len(mos):
            raise ValueError(f"Image and MOS length mismatch in {self.label_path}")

        self.samples: List[Tuple[str, float]] = []
        for image_name, score in zip(images, mos):
            image_path = os.path.join(self.img_dir, image_name)
            if os.path.isfile(image_path):
                self.samples.append((image_name, float(score)))
        if max_samples is not None:
            self.samples = self.samples[:max_samples]
        if not self.samples:
            raise RuntimeError(f"No valid samples found for split '{split}'")

        self.roi_dirs: Dict[str, Optional[str]] = {
            "left_eye": _resolve_roi_dir(self.split_dir, "left_eye"),
            "right_eye": _resolve_roi_dir(self.split_dir, "right_eye"),
            "mouth": _resolve_roi_dir(self.split_dir, "mouth"),
        }
        self.roi_names = ["left_eye", "right_eye"]
        if self.use_mouth:
            self.roi_names.append("mouth")

        missing_dirs = [name for name in self.roi_names if self.roi_dirs[name] is None]
        if self.strict_roi and missing_dirs:
            raise FileNotFoundError(f"Missing ROI directories for {split}: {missing_dirs}")

    def _load_image(self, path: str, transform: transforms.Compose) -> torch.Tensor:
        image = Image.open(path).convert("RGB")
        return transform(image)

    def _load_roi(self, image_name: str, roi_name: str) -> Tuple[torch.Tensor, torch.Tensor]:
        roi_dir = self.roi_dirs.get(roi_name)
        if roi_dir is None:
            if self.strict_roi:
                raise FileNotFoundError(f"Missing ROI directory for {roi_name}")
            return self.zero_roi.clone(), torch.tensor(0.0)

        roi_path = os.path.join(roi_dir, image_name)
        if not os.path.isfile(roi_path):
            stem, _ = os.path.splitext(image_name)
            for ext in IMAGE_EXTENSIONS:
                candidate = os.path.join(roi_dir, stem + ext)
                if os.path.isfile(candidate):
                    roi_path = candidate
                    break
        if not os.path.isfile(roi_path):
            if self.strict_roi:
                raise FileNotFoundError(f"Missing {roi_name} ROI for {image_name}")
            return self.zero_roi.clone(), torch.tensor(0.0)

        return self._load_image(roi_path, self.roi_transform), torch.tensor(1.0)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        image_name, score = self.samples[index]
        image_path = os.path.join(self.img_dir, image_name)

        item: Dict[str, torch.Tensor] = {
            "image": self._load_image(image_path, self.image_transform),
            "score": torch.tensor(score, dtype=torch.float32),
            "name": image_name,
        }
        roi_mask = []
        for roi_name in self.roi_names:
            roi_tensor, present = self._load_roi(image_name, roi_name)
            item[roi_name] = roi_tensor
            roi_mask.append(present)
        item["roi_mask"] = torch.stack(roi_mask).float()
        return item

    def __len__(self) -> int:
        return len(self.samples)

    def roi_report(self) -> Dict[str, int]:
        report = {}
        sample_names = {name for name, _ in self.samples}
        for roi_name in self.roi_names:
            roi_dir = self.roi_dirs.get(roi_name)
            if roi_dir is None:
                report[roi_name] = 0
                continue
            report[roi_name] = len(sample_names.intersection(set(os.listdir(roi_dir))))
        return report
