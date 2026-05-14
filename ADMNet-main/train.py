# =========================================
# 文件名: train.py
# 多分支显著区域人脸质量评价训练脚本
# =========================================

import os
import json
import torch
import tornn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from tqdm import tqdm

# =========================
# 1. Dataset 定义
# =========================
class FaceQualityDataset(Dataset):
    def __init__(self, global_dir, local_root, label_file, transform=None):
        self.global_dir = global_dir
        self.local_root = local_root
        self.transform = transform

        with open(label_file, 'r', encoding='utf-8') as f:
            data = json.load(f)

        self.samples = list(zip(data["Image"], data["MOS"]))
        print(f"Dataset loaded: {len(self.samples)} samples")

    # 修改 train.py 第 33-38 行
    def _load_image(self, path):
        # 调试阶段建议直接通过，不要 try-except，确保路径绝对正确
        # try:
        img = Image.open(path).convert("RGB")
        # except:
        #     print(f"Warning: Failed to load {path}") # 至少打印一下路径
        #     img = Image.new("RGB", (224, 224))

        return self.transform(img) if self.transform else img

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        name, score = self.samples[idx]

        global_img = self._load_image(os.path.join(self.global_dir, name))
        left = self._load_image(os.path.join(self.local_root, "left_eye", name))
        right = self._load_image(os.path.join(self.local_root, "right_eye", name))
        mouth = self._load_image(os.path.join(self.local_root, "mouth", name))

        score = torch.tensor(score, dtype=torch.float32)
        return global_img, left, right, mouth, score


# =========================
# 2. PLCC Loss
# =========================
class PLCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        pred = pred.view(-1)
        target = target.view(-1)

        pred_mean = torch.mean(pred)
        target_mean = torch.mean(target)

        pred_c = pred - pred_mean
        target_c = target - target_mean

        cov = torch.sum(pred_c * target_c)
        pred_std = torch.sqrt(torch.sum(pred_c ** 2) + 1e-8)
        target_std = torch.sqrt(torch.sum(target_c ** 2) + 1e-8)

        plcc = cov / (pred_std * target_std + 1e-8)
        return 1.0 - plcc


# =========================
# 3. 多分支模型定义
# =========================
class MultiRegionIQA(nn.Module):
    def __init__(self):
        super().__init__()

        base = models.resnet18(pretrained=True)
        self.backbone = nn.Sequential(*list(base.children())[:-1])  # 去掉 fc
        self.feat_dim = 512

        self.regressor = nn.Sequential(
            nn.Linear(self.feat_dim * 4, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1)
        )

    def _extract_feat(self, x):
        return self.backbone(x).view(x.size(0), -1)

    def forward(self, g, l, r, m):
        fg = self._extract_feat(g)
        fl = self._extract_feat(l)
        fr = self._extract_feat(r)
        fm = self._extract_feat(m)

        feat = torch.cat([fg, fl, fr, fm], dim=1)
        return self.regressor(feat)


# =========================
# 4. 训练配置
# =========================
GLOBAL_IMG_DIR = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img"
LOCAL_ROOT = r"D:\Desktop\study_life\gradua-proj\FIQA\train\local_patches"
LABEL_FILE = r"D:\Desktop\study_life\gradua-proj\FIQA\train\label.json"

SAVE_DIR = "./quality_model_save"
os.makedirs(SAVE_DIR, exist_ok=True)

EPOCHS = 50
BATCH_SIZE = 16
LR = 1e-4

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# =========================
# 5. 主程序
# =========================
if __name__ == "__main__":

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    dataset = FaceQualityDataset(
        global_dir=GLOBAL_IMG_DIR,
        local_root=LOCAL_ROOT,
        label_file=LABEL_FILE,
        transform=transform
    )

    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = MultiRegionIQA().to(device)

    criterion_mse = nn.MSELoss()
    criterion_plcc = PLCCLoss()

    optimizer = optim.Adam(model.parameters(), lr=LR)

    print("🚀 Start Training...")

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0.0

        for data in tqdm(dataloader, desc=f"Epoch [{epoch+1}/{EPOCHS}]"):
            g, l, r, m, score = data
            g, l, r, m = g.to(device), l.to(device), r.to(device), m.to(device)
            score = score.to(device).unsqueeze(1)

            optimizer.zero_grad()

            pred = model(g, l, r, m)

            loss_mse = criterion_mse(pred, score)
            loss_plcc = criterion_plcc(pred, score)
            loss = loss_mse + loss_plcc

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}: Loss = {epoch_loss / len(dataloader):.4f}")

        if (epoch + 1) % 5 == 0:
            save_path = os.path.join(SAVE_DIR, f"MultiRegionIQA_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"✅ Model saved: {save_path}")

    print("🎉 Training Finished")
