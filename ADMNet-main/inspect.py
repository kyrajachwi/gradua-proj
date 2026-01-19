import os
import torch
import cv2
import numpy as np
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pfld import PFLDInference

# ================= 配置 =================
test_image = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img\000031.png" # 选择一张测试图片
checkpoint_path = r"./checkpoint.pth.tar"
IMG_SIZE = 112

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = PFLDInference().to(device)
model.eval()
ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['pfld_backbone'], strict=True)

# 初始化人脸检测器
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# ================= 读取图像 =================
img = cv2.imread(test_image)
if img is None:
    print("❌ 无法读取测试图片，请修改 test_image 路径")
    exit()

h, w, _ = img.shape
print(f"原始图像尺寸: {w}x{h}")

# ================= 人脸检测 =================
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

if len(faces) == 0:
    print("⚠️ 未检测到人脸，将使用整张图片")
    face_img = img
    x1, y1 = 0, 0
    x2, y2 = w, h
else:
    print(f"✅ 检测到 {len(faces)} 个人脸")
    face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, fw, fh = face

    # 扩展边界
    margin = 0.2
    x1 = max(0, int(x - fw * margin))
    y1 = max(0, int(y - fh * margin))
    x2 = min(w, int(x + fw * (1 + margin)))
    y2 = min(h, int(y + fh * (1 + margin)))

    face_img = img[y1:y2, x1:x2]
    print(f"人脸区域: ({x1}, {y1}) -> ({x2}, {y2})")

# Resize
face_resized = cv2.resize(face_img, (IMG_SIZE, IMG_SIZE))

# ================= 测试不同预处理方法 =================
methods = {
    "方法1: 仅归一化 (BGR)": lambda img: torch.from_numpy(img).permute(2, 0, 1).float() / 255.0,

    "方法2: 归一化 (RGB)": lambda img: torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0,
                                                                                                      1).float() / 255.0,

    "方法3: ImageNet标准化 (RGB)": lambda img: (
            (torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float() / 255.0 -
             torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)) /
            torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    ),

    "方法4: 归一化到[-1,1] (RGB)": lambda img: torch.from_numpy(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).permute(2, 0,
                                                                                                              1).float() / 127.5 - 1.0,
}

results = {}

for method_name, preprocess_fn in methods.items():
    print(f"\n测试 {method_name}...")

    img_tensor = preprocess_fn(face_resized).unsqueeze(0).to(device)

    with torch.no_grad():
        landmarks = model(img_tensor)[0].cpu().numpy()

    landmarks = landmarks.reshape(-1, 2)

    # 映射回原图
    landmarks_original = landmarks.copy()
    landmarks_original[:, 0] = landmarks[:, 0] * (x2 - x1) + x1
    landmarks_original[:, 1] = landmarks[:, 1] * (y2 - y1) + y1

    # 检查关键点是否合理
    valid_points = np.sum((landmarks_original[:, 0] >= 0) &
                          (landmarks_original[:, 0] < w) &
                          (landmarks_original[:, 1] >= 0) &
                          (landmarks_original[:, 1] < h))

    # 计算关键点的分散程度
    std_x = np.std(landmarks_original[:, 0])
    std_y = np.std(landmarks_original[:, 1])

    print(f"  有效点数: {valid_points}/98")
    print(f"  X坐标标准差: {std_x:.2f}")
    print(f"  Y坐标标准差: {std_y:.2f}")
    print(
        f"  关键点范围: X[{landmarks_original[:, 0].min():.1f}, {landmarks_original[:, 0].max():.1f}], Y[{landmarks_original[:, 1].min():.1f}, {landmarks_original[:, 1].max():.1f}]")

    results[method_name] = {
        'landmarks': landmarks_original,
        'valid': valid_points,
        'std': (std_x, std_y)
    }

# ================= 可视化对比 =================
print("\n" + "=" * 50)
print("生成对比可视化...")

for method_name, result in results.items():
    vis_img = img.copy()
    landmarks = result['landmarks']

    # 绘制人脸框
    cv2.rectangle(vis_img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # 绘制关键点
    for point in landmarks:
        x, y = int(point[0]), int(point[1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)

    # 保存
    safe_name = method_name.replace(":", "").replace(" ", "_").replace("(", "").replace(")", "")
    save_path = f"test_{safe_name}.jpg"
    cv2.imwrite(save_path, vis_img)
    print(f"已保存: {save_path}")

print("\n✅ 诊断完成！请查看生成的图片，找出效果最好的方法")
print("\n💡 提示:")
print("- 如果关键点集中在人脸区域 → 预处理正确")
print("- 如果关键点分散在全图 → 预处理不匹配")
print("- 标准差应该在 10-50 像素范围内（取决于图片尺寸）")