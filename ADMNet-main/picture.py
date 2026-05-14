import os
import random
import cv2
import numpy as np


def main():
    # --- 1. 配置路径 ---
    base_path = r"D:\Desktop\study_life\gradua-proj\FIQA"
    img_dir = os.path.join(base_path, r"train\img")
    lm_dir = os.path.join(base_path, r"train\pfld_landmarks")
    save_dir = os.path.join(base_path, r"vis_pfld")

    os.makedirs(save_dir, exist_ok=True)

    # --- 2. 获取图片列表 (修改为 .png) ---
    if not os.path.exists(img_dir):
        print(f"错误: 找不到图片目录 {img_dir}")
        return

    # 筛选以 .png 结尾的文件 (忽略大小写，兼容 .PNG)
    all_names = [f for f in os.listdir(img_dir) if f.lower().endswith(".png")]

    if len(all_names) == 0:
        print("未在目录中找到 .png 图片。")
        return

    # 随机抽取
    sample_count = 5
    sample_names = random.sample(all_names, min(sample_count, len(all_names)))

    print(f"开始处理，共抽取 {len(sample_names)} 张 PNG 图片...")

    # --- 3. 循环处理 ---
    for name in sample_names:
        img_path = os.path.join(img_dir, name)

        # --- 修改点：更智能的后缀替换 ---
        # os.path.splitext(name)[0] 获取文件名（不带后缀），然后拼接 .npy
        # 例如: "image_01.png" -> "image_01" -> "image_01.npy"
        lm_name = os.path.splitext(name)[0] + ".npy"
        lm_path = os.path.join(lm_dir, lm_name)

        save_path = os.path.join(save_dir, name)

        img = cv2.imread(img_path)

        if img is None:
            print(f"警告: 无法读取图片 {name}，可能文件损坏。")
            continue

        if not os.path.exists(lm_path):
            print(f"警告: 找不到对应的关键点文件 {lm_name}，跳过。")
            continue

        try:
            landmarks = np.load(lm_path)  # (98, 2)
        except Exception as e:
            print(f"读取 npy 失败: {lm_name}, 错误: {e}")
            continue

        # --- 4. 绘制关键点 ---
        for point in landmarks:
            x, y = int(point[0]), int(point[1])
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        # --- 5. 保存结果 ---
        cv2.imwrite(save_path, img)
        print(f"已保存: {name}")

    print("\n✅ 所有 PNG 图片可视化完成！")


if __name__ == "__main__":
    main()