import torch
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw
import os

# ================= 配置路径 =================
# 指向一张你觉得“没裁剪成功”的原图路径
img_path = r"D:\Desktop\study_life\gradua-proj\FIQA\train\img_cropped_njupt\000245.png" # <--- 请修改这里，随便挑一张

# ================= 诊断代码 =================
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(keep_all=False, select_largest=True, device=device)

print(f"正在读取图片: {img_path}")
try:
    img = Image.open(img_path).convert('RGB')

    # 检测
    boxes, probs = mtcnn.detect(img)

    draw = ImageDraw.Draw(img)

    if boxes is None:
        print("❌ 结果：MTCNN 没有检测到任何人脸！")
        print("原因分析：图片太模糊、光照太暗、侧脸角度过大，或者人脸太小。")
        print("代码行为：触发了兜底逻辑，直接保存了原图。")
    else:
        print(f"✅ 结果：检测成功！找到人脸框: {boxes[0]}")
        print(f"置信度: {probs[0]}")

        # 画出检测框 (红色)
        box = boxes[0]
        draw.rectangle(box.tolist(), outline="red", width=5)

        # 模拟裁剪 (蓝色)
        x1, y1, x2, y2 = box
        w, h = x2 - x1, y2 - y1
        padding = 1.5
        new_size = max(w, h) * padding
        cx, cy = x1 + w / 2, y1 + h / 2
        nx1, ny1 = cx - new_size / 2, cy - new_size / 2
        nx2, ny2 = cx + new_size / 2, cy + new_size / 2
        draw.rectangle([nx1, ny1, nx2, ny2], outline="blue", width=5)
        print("红框 = 原始人脸检测, 蓝框 = 裁剪范围")

    # 弹窗显示图片
    img.show()

except Exception as e:
    print(f"出错: {e}")