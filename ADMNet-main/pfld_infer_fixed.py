"""
测试修复后的代码 - 验证 landmarks 输出
"""
import torch
import cv2
import numpy as np
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from pfld import PFLDInference

# ================= 配置 =================
test_img = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img\000125.png"
checkpoint_path = r"./checkpoint.pth.tar"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("=" * 60)
print("测试修复后的 PFLD 推理")
print("=" * 60)

# ================= 加载模型 =================
print("\n1. 加载模型...")
model = PFLDInference().to(device)
model.eval()

ckpt = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(ckpt['pfld_backbone'], strict=True)
print("   ✓ 模型加载成功")

# ================= 读取图片 =================
print("\n2. 读取图片...")
img = cv2.imread(test_img)
if img is None:
    print(f"   ✗ 无法读取图片: {test_img}")
    print("   请修改脚本中的 test_img 路径")
    exit()

h, w, _ = img.shape
print(f"   ✓ 图片尺寸: {w}x{h}")

# ================= 预处理和推理 =================
print("\n3. 预处理和推理...")
img_resized = cv2.resize(img, (112, 112))
img_tensor = torch.from_numpy(img_resized).permute(2, 0, 1).float() / 255.0
img_tensor = img_tensor.unsqueeze(0).to(device)

with torch.no_grad():
    # PFLDInference 返回 (features, landmarks)
    features, landmarks = model(img_tensor)

print("   ✓ 推理完成")

print(f"\n   模型输出:")
print(f"   - features (用于训练): {features.shape}")
print(f"   - landmarks (关键点): {landmarks.shape}")

landmarks = landmarks[0].cpu().numpy()

print(f"\n   【关键】Landmarks 输出:")
print(f"   - 形状: {landmarks.shape}")
print(f"   - 范围: [{landmarks.min():.6f}, {landmarks.max():.6f}]")
print(f"   - 前10个值: {landmarks[:10]}")

# 判断
if landmarks.shape[0] == 196:
    print(f"\n   ✓✓✓ 输出正确！196个值 = 98个关键点的(x,y)坐标")
else:
    print(f"\n   ✗ 输出形状异常: {landmarks.shape}")
    exit()

# ================= 映射回原图 =================
print("\n4. 映射回原图...")
landmarks = landmarks.reshape(98, 2)
landmarks[:, 0] *= w
landmarks[:, 1] *= h

print(f"   映射后坐标:")
print(f"   - X范围: [{landmarks[:, 0].min():.1f}, {landmarks[:, 0].max():.1f}] (图片宽: {w})")
print(f"   - Y范围: [{landmarks[:, 1].min():.1f}, {landmarks[:, 1].max():.1f}] (图片高: {h})")
print(f"   - X标准差: {np.std(landmarks[:, 0]):.2f}")
print(f"   - Y标准差: {np.std(landmarks[:, 1]):.2f}")

# 有效点
valid = np.sum((landmarks[:, 0] >= 0) & (landmarks[:, 0] < w) &
               (landmarks[:, 1] >= 0) & (landmarks[:, 1] < h))
print(f"   - 有效点: {valid}/98")

# ================= 可视化 =================
print("\n5. 生成可视化...")
vis_img = img.copy()

for i, (x, y) in enumerate(landmarks):
    x, y = int(x), int(y)
    if 0 <= x < w and 0 <= y < h:
        cv2.circle(vis_img, (x, y), 2, (0, 255, 0), -1)

cv2.imwrite("test_result_FIXED.jpg", vis_img)
cv2.imwrite("test_original.jpg", img)

print(f"   ✓ 已保存: test_result_FIXED.jpg")
print(f"   ✓ 已保存: test_original.jpg")

# ================= 结果评估 =================
print("\n" + "=" * 60)
print("结果评估")
print("=" * 60)

if valid >= 90:
    print("\n✓✓✓ 太好了！几乎所有关键点都在图片内")
    if 10 < np.std(landmarks[:, 0]) < 150 and 10 < np.std(landmarks[:, 1]) < 150:
        print("✓✓✓ 关键点分散程度合理")
        print("\n恭喜！修复成功！可以开始批量处理了。")
    else:
        print("⚠️ 关键点分散程度异常")
        print("请查看可视化图片确认")
else:
    print(f"\n⚠️ 有效点较少 ({valid}/98)")
    print("可能的原因:")
    print("  - 模型输出的归一化坐标超出[0,1]范围")
    print("  - 坐标映射方式不对")

print("\n请打开 test_result_FIXED.jpg 查看结果")
print("如果关键点正确标注在眼睛、鼻子、嘴巴位置，就可以批量处理了！")

print("\n" + "=" * 60)