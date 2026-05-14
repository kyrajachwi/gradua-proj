import os
import sys
import torch
import cv2
import numpy as np
from tqdm import tqdm

# 确保能 import models
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from pfld import PFLDInference

# ================= 配置 =================
image_dir = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img"
save_dir = r"D:\Desktop\study_life\gradua-proj\FIQA\train\pfld_landmarks"
checkpoint_path = r"./checkpoint.pth.tar"

IMG_SIZE = 112
os.makedirs(save_dir, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ================= 加载模型 =================
model = PFLDInference().to(device)
model.eval()

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['pfld_backbone'], strict=True)

print("✓ 模型加载成功")

# ================= 推理 =================
for name in tqdm(os.listdir(image_dir)):
    if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join(image_dir, name)
    img = cv2.imread(img_path)
    if img is None:
        continue

    h, w, _ = img.shape

    # FIQA数据集是纯人脸，直接resize
    img_resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # 预处理：BGR格式，归一化到[0,1]
    img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        # ⭐ 关键修复：PFLDInference返回 (features, landmarks)
        # features 用于训练时的辅助网络
        # landmarks 才是我们需要的关键点坐标
        _, landmarks = model(img_tensor)  # 解包，取第二个
        landmarks = landmarks[0].cpu().numpy()

    # 映射回原图坐标
    landmarks = landmarks.reshape(-1, 2)  # (98, 2)
    landmarks[:, 0] *= w
    landmarks[:, 1] *= h

    np.save(
        os.path.join(save_dir, name.rsplit('.', 1)[0] + '.npy'),
        landmarks
    )

print("✅ PFLD 关键点提取完成")