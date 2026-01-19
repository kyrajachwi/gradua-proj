import os
import cv2
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ================= 配置 =================
img_dir = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img"
landmark_dir = r"D:\Desktop\study_life\gradua-proj\FIQA\train\pfld_landmarks"

save_root = r"D:\Desktop\study_life\gradua-proj\FIQA\train\local_patches"
LEFT_EYE_DIR = os.path.join(save_root, "left_eye")
RIGHT_EYE_DIR = os.path.join(save_root, "right_eye")
MOUTH_DIR = os.path.join(save_root, "mouth")

for d in [LEFT_EYE_DIR, RIGHT_EYE_DIR, MOUTH_DIR]:
    os.makedirs(d, exist_ok=True)

IMG_SIZE = 224          # 假设人脸已 resize 到 224×224
HEATMAP_SIZE = 224
PATCH_SIZE = 112        # 局部 patch 大小
SIGMA = 5               # 高斯平滑参数

# ================= Step 1：构建关键点热图 =================
heatmap = np.zeros((HEATMAP_SIZE, HEATMAP_SIZE), dtype=np.float32)
count = 0

for name in tqdm(os.listdir(landmark_dir), desc="Building heatmap"):
    if not name.endswith(".npy"):
        continue

    lm = np.load(os.path.join(landmark_dir, name))  # (98, 2)
    lm = lm.astype(np.int32)

    for (x, y) in lm:
        if 0 <= x < HEATMAP_SIZE and 0 <= y < HEATMAP_SIZE:
            heatmap[y, x] += 1

    count += 1

heatmap /= (count + 1e-6)
heatmap = gaussian_filter(heatmap, sigma=SIGMA)

# ================= Step 2：定位显著区域中心 =================
# 上半脸：眼睛，下半脸：嘴巴
upper = heatmap[:HEATMAP_SIZE // 2, :]
lower = heatmap[HEATMAP_SIZE // 2:, :]

# 左右眼
left_eye_region = upper[:, :HEATMAP_SIZE // 2]
right_eye_region = upper[:, HEATMAP_SIZE // 2:]

le_y, le_x = np.unravel_index(np.argmax(left_eye_region), left_eye_region.shape)
re_y, re_x = np.unravel_index(np.argmax(right_eye_region), right_eye_region.shape)
re_x += HEATMAP_SIZE // 2

# 嘴巴
mo_y, mo_x = np.unravel_index(np.argmax(lower), lower.shape)
mo_y += HEATMAP_SIZE // 2

centers = {
    "left_eye": (le_x, le_y),
    "right_eye": (re_x, re_y),
    "mouth": (mo_x, mo_y)
}

print("显著区域中心点：", centers)

# ================= Step 3：裁剪局部 Patch =================
def crop_patch(img, cx, cy, size):
    half = size // 2
    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(img.shape[1], cx + half)
    y2 = min(img.shape[0], cy + half)
    patch = img[y1:y2, x1:x2]
    return cv2.resize(patch, (size, size))

for name in tqdm(os.listdir(img_dir), desc="Cropping patches"):
    if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img = cv2.imread(os.path.join(img_dir, name))
    if img is None:
        continue

    le = crop_patch(img, *centers["left_eye"], PATCH_SIZE)
    re = crop_patch(img, *centers["right_eye"], PATCH_SIZE)
    mo = crop_patch(img, *centers["mouth"], PATCH_SIZE)

    cv2.imwrite(os.path.join(LEFT_EYE_DIR, name), le)
    cv2.imwrite(os.path.join(RIGHT_EYE_DIR, name), re)
    cv2.imwrite(os.path.join(MOUTH_DIR, name), mo)

print("✅ 显著区域裁剪完成")
