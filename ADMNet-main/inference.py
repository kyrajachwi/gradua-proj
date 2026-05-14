import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import os
import numpy as np

# 1. 引入你的模型结构 (必须和训练时一样)
from model.ADMNet import ADMNet

# ================= 配置区域 =================
# 训练好的参数文件路径 (找一个你目录下生成的 model_save/MYNet_epoch_xx.pth)
model_path = './model_save/MYNet_epoch_33.pth'

# 你想测试的图片文件夹
input_img_folder = r'D:\Desktop\study_life\gradua-proj\ADMNet-main\input'
# 结果保存的文件夹
output_save_folder =  r'D:\Desktop\study_life\gradua-proj\ADMNet-main\output'

# 确保输出文件夹存在
os.makedirs(output_save_folder, exist_ok=True)


# ===========================================

def inference():
    print(f"Loading model from: {model_path}")

    # --- 第一步：准备模型 ---
    net = ADMNet()

    # 加载参数 (map_location='cuda' 如果你有显卡，否则 'cpu')
    if torch.cuda.is_available():
        net.load_state_dict(torch.load(model_path))
        net.cuda()
    else:
        net.load_state_dict(torch.load(model_path, map_location='cpu'))

    net.eval()  # 切换到评估模式 (非常重要！这会冻结 Dropout 和 BN 层)

    # --- 第二步：准备图片预处理 ---
    # 这里的参数要和你训练时保持一致 (通常 inference 时只需要 resize 和 归一化)
    img_transform = transforms.Compose([
        transforms.Resize((352, 352)),  # ADMNet 通常输入是 352x352 或 336x336，看你训练设置
        transforms.ToTensor(),  # 变成 Tensor 格式，数值归一化到 [0, 1]
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # 标准 ImageNet 均值方差，如果训练没用这个，这行可以去掉
    ])

    # --- 第三步：循环处理文件夹里的每张图 ---
    img_list = os.listdir(input_img_folder)

    for img_name in img_list:
        if not img_name.endswith(('.jpg', '.png', '.jpeg')):
            continue

        print(f"Processing: {img_name} ...")

        # 3.1 读取图片
        full_path = os.path.join(input_img_folder, img_name)
        image = Image.open(full_path).convert('RGB')

        # 3.2 预处理
        # unsqueeze(0) 是为了增加一个 batch 维度
        # 变成 [1, 3, 352, 352] -> 意思是：1张图，3通道，高352，宽352
        image_tensor = img_transform(image).unsqueeze(0)

        if torch.cuda.is_available():
            image_tensor = image_tensor.cuda()

        # 3.3 模型预测 (不计算梯度，节省内存)
        with torch.no_grad():
            # ADMNet 返回 5 个结果 (d1...d5)，我们通常只取 d1 或 d2 作为最终结果
            # d1, d2, d3, d4, d5 = net(image_tensor)
            # 假设 d2 是主要输出 (具体看论文结构，通常取分辨率最高的那个)
            # 这里我们为了保险，把 d1 到 d5 打印一下 shape 看看，通常用第一个
            outputs = net(image_tensor)
            pred_mask = outputs[0]  # 取 d1

            # 3.4 后处理
            # 此时 pred_mask 里的值是任意实数，需要用 Sigmoid 压缩到 (0, 1) 代表概率
            pred_mask = torch.sigmoid(pred_mask)

            # 把 tensor 转回 numpy 图片格式
            # squeeze() 去掉 batch 和 channel 维度 -> [352, 352]
            pred_np = pred_mask.squeeze().cpu().numpy()

            # 3.5 保存图片
            # 把它变回 0-255 的整数
            pred_np = (pred_np * 255).astype(np.uint8)

            # 变回原始图片的大小 (可选，如果不做这一步，输出就是 352x352)
            final_img = Image.fromarray(pred_np)
            final_img = final_img.resize(image.size, resample=Image.BILINEAR)

            save_path = os.path.join(output_save_folder, os.path.splitext(img_name)[0] + '.png')
            final_img.save(save_path)

    print("Inference Finished! Go check the output folder.")


if __name__ == '__main__':
    inference()