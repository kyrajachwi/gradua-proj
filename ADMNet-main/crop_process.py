import os
import cv2
import numpy as np
from tqdm import tqdm

# ================= 配置区域 =================
# 原始图片路径
IMG_DIR = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img"
# PFLD生成的关键点路径
LANDMARK_DIR = r"D:\Desktop\study_life\gradua-proj\FIQA\train\pfld_landmarks"
# 输出路径 (对应 train.py 中的 LOCAL_ROOT)
OUTPUT_ROOT = r"D:\Desktop\study_life\gradua-proj\FIQA\train\local_patches"

# 裁剪参数
PATCH_SIZE = 112  # 裁剪出的小图大小 (112x112)


# ===========================================

def crop_patch(img, center, size):
    """以 center 为中心，裁剪 size x size 的图片，边缘填充 0"""
    h, w, _ = img.shape
    cx, cy = int(center[0]), int(center[1])
    half = size // 2

    # 计算裁剪区域坐标
    x1, y1 = cx - half, cy - half
    x2, y2 = x1 + size, y1 + size

    # 处理边界（如果裁剪超出图片范围，进行 Padding）
    pad_l = max(0, -x1)
    pad_t = max(0, -y1)
    pad_r = max(0, x2 - w)
    pad_b = max(0, y2 - h)

    # 修正原图截取坐标
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)

    cropped = img[y1:y2, x1:x2]

    # 填充边界
    if pad_l > 0 or pad_t > 0 or pad_r > 0 or pad_b > 0:
        cropped = cv2.copyMakeBorder(cropped, pad_t, pad_b, pad_l, pad_r, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # 确保尺寸严格正确
    return cv2.resize(cropped, (size, size))


def main():
    # 创建输出目录
    for sub in ["left_eye", "right_eye", "mouth"]:
        os.makedirs(os.path.join(OUTPUT_ROOT, sub), exist_ok=True)

    img_list = os.listdir(IMG_DIR)
    print(f"🔄 开始处理 ROI 裁剪，共 {len(img_list)} 张图片...")

    for img_name in tqdm(img_list):
        if not img_name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue

        # 1. 读取图片
        img_path = os.path.join(IMG_DIR, img_name)
        img = cv2.imread(img_path)
        if img is None:
            continue

        # 2. 读取对应的 .npy 关键点
        npy_name = os.path.splitext(img_name)[0] + ".npy"
        npy_path = os.path.join(LANDMARK_DIR, npy_name)

        if not os.path.exists(npy_path):
            # print(f"⚠️ 警告: 找不到关键点文件 {npy_name}，跳过")
            continue

        landmarks = np.load(npy_path)  # shape (98, 2)

        # 3. 计算 ROI 中心点 (基于 WFLW 98点索引)
        # 左眼: 60-67
        left_eye_pts = landmarks[60:68]
        left_center = np.mean(left_eye_pts, axis=0)

        # 右眼: 68-75
        right_eye_pts = landmarks[68:76]
        right_center = np.mean(right_eye_pts, axis=0)

        # 嘴巴: 76-95 (包含外轮廓和内轮廓)
        mouth_pts = landmarks[76:96]
        mouth_center = np.mean(mouth_pts, axis=0)

        # 4. 执行裁剪
        patch_left = crop_patch(img, left_center, PATCH_SIZE)
        patch_right = crop_patch(img, right_center, PATCH_SIZE)
        patch_mouth = crop_patch(img, mouth_center, PATCH_SIZE)

        # 5. 保存图片
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "left_eye", img_name), patch_left)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "right_eye", img_name), patch_right)
        cv2.imwrite(os.path.join(OUTPUT_ROOT, "mouth", img_name), patch_mouth)

    print(f"✅ ROI 裁剪完成！保存路径: {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()