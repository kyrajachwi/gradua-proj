# RINet-FAM: Face Image Quality Assessment

本项目是一个面向毕业设计的 **人脸图像质量评估（Face Image Quality Assessment, FIQA）** 实验工程。项目在原始 RINet 代码基础上扩展了多分支 MOS 回归模型，用于根据全局人脸图像、局部 ROI 和 RINet 风格注意力先验预测人脸图像质量分数。

> 原始 RINet 用于 fixation prediction。本项目中，RINet 相关模块作为注意力先验生成分支使用，不作为最终 FIQA 回归器。

## 项目特点

- 支持 `label.json` 标注文件，字段为 `Image` 和 `MOS`
- MOS 已归一化到 `[0, 1]`，训练时不再除以 100
- 多分支 FIQA 模型：
  - 全局人脸图像分支
  - 左眼 ROI 分支
  - 右眼 ROI 分支
  - 嘴部 ROI 分支
  - RINet 风格注意力分支
  - MOS 回归头
- 支持消融实验：
  - `global`
  - `global_attention`
  - `global_eyes`
  - `global_eyes_attention`
  - `global_eyes_mouth_attention`
- 评估指标：PLCC、MSE、MAE

## 目录结构

```text
.
├── RINet-main/
│   ├── train_fiqa.py              # FIQA 训练入口
│   ├── predict_fiqa.py            # FIQA 预测入口
│   ├── run_ablation_fiqa.py       # 消融实验入口
│   ├── fiqa_dataset.py            # label.json 数据加载器
│   ├── visualize_fiqa_results.py  # 结果可视化
│   ├── visualize_ablation_results.py
│   └── module/
│       └── fiqa_multibranch.py    # RINet-FAM 多分支模型
├── ADMNet-main/                   # ROI 生成相关代码
├── scripts/                       # 海报/辅助脚本
└── README.md
```

数据集和训练结果不建议上传到 GitHub。推荐通过 `.gitignore` 排除 `FIQA/`、`DUTS/`、`RINet-main/result/`、模型权重和压缩包等大文件。

## 数据集格式

默认数据集根目录为 `FIQA/`，并划分为 `train`、`val`、`test` 三个子集：

```text
FIQA/
├── train/
│   ├── label.json
│   ├── img/
│   ├── left_eye/
│   ├── right_eye/
│   └── mouth/
├── val/
│   ├── label.json
│   ├── img/
│   ├── left_eye/
│   ├── right_eye/
│   └── mouth/
└── test/
    ├── label.json
    ├── img/
    ├── left_eye/
    ├── right_eye/
    └── mouth/
```

`label.json` 示例：

```json
{
  "Image": ["000001.jpg", "000002.jpg"],
  "MOS": [0.82, 0.37]
}
```

注意事项：

- `Image` 中的文件名应能在对应 split 的 `img/` 目录中找到
- ROI 图像文件名应与原图文件名对应
- `left_eye/`、`right_eye/`、`mouth/` 也支持放在 `local_patches/` 下
- 如果 ROI 缺失，默认使用零张量；开启 `--strict_roi` 后会严格检查 ROI 文件

## 环境配置

推荐使用 Conda：

```bash
cd RINet-main
conda env create -f environment.yaml
conda activate RINet
```

如果手动安装，核心依赖包括：

```bash
pip install torch torchvision numpy pillow tqdm scipy scikit-image matplotlib opencv-python-headless
```

## 训练

进入 `RINet-main` 目录后运行：

```bash
python train_fiqa.py \
  --dataset_root ../FIQA \
  --output_dir ./result/FIQA \
  --epochs 20 \
  --batch_size 32 \
  --backbone tiny \
  --strict_roi
```

常用参数：

```text
--dataset_root        数据集根目录，默认 ../FIQA
--output_dir          输出目录，默认 ./result/FIQA
--backbone            tiny 或 resnet18
--pretrained          使用 torchvision 预训练权重
--use_mouth           启用嘴部 ROI 分支
--no_attention        关闭注意力分支
--no_left_eye         关闭左眼分支
--no_right_eye        关闭右眼分支
--image_size          全局图像尺寸，默认 224
--roi_size            ROI 图像尺寸，默认 112
--epochs              训练轮数
--batch_size          批大小
--lr                  学习率
--num_workers         DataLoader worker 数
```

训练完成后会在输出目录保存：

```text
best_fiqa_model.pt
last_fiqa_model.pt
metrics.json
```

## 预测

使用训练得到的 checkpoint 在指定 split 上导出预测结果：

```bash
python predict_fiqa.py \
  --dataset_root ../FIQA \
  --split test \
  --checkpoint ./result/FIQA/best_fiqa_model.pt \
  --output_csv ./result/FIQA/predictions.csv \
  --batch_size 64
```

输出 CSV 字段：

```text
Image,MOS,Pred
```

## 消融实验

一键运行五组消融实验：

```bash
python run_ablation_fiqa.py \
  --dataset_root ../FIQA \
  --output_root ./result/FIQA_ablation \
  --epochs 20 \
  --batch_size 32 \
  --backbone tiny \
  --predict
```

只运行部分消融实验：

```bash
python run_ablation_fiqa.py \
  --dataset_root ../FIQA \
  --output_root ./result/FIQA_ablation \
  --only A_global B_global_attention
```

消融实验名称：

| 名称 | 模型配置 |
| --- | --- |
| `A_global` | 全局人脸分支 |
| `B_global_attention` | 全局人脸分支 + 注意力分支 |
| `C_global_eyes` | 全局人脸分支 + 左右眼 ROI 分支 |
| `D_global_eyes_attention` | 全局人脸分支 + 左右眼 ROI 分支 + 注意力分支 |
| `E_global_eyes_mouth_attention` | 全局人脸分支 + 左右眼 ROI 分支 + 嘴部 ROI 分支 + 注意力分支 |

实验汇总会保存到：

```text
RINet-main/result/FIQA_ablation/ablation_summary.json
```

## 评价指标

训练和验证阶段统计：

- **MSE**：预测 MOS 与真实 MOS 的均方误差
- **MAE**：预测 MOS 与真实 MOS 的平均绝对误差
- **PLCC**：预测 MOS 与真实 MOS 的 Pearson 线性相关系数

其中 MSE 越低越好，MAE 越低越好，PLCC 越高越好。

## GitHub 上传建议

上传代码时建议保留：

```text
RINet-main/
ADMNet-main/
scripts/
README.md
.gitignore
```

不建议上传：

```text
FIQA/
DUTS/
*.zip
*.pt
*.pth
*.ckpt
RINet-main/result/
*.pdf
*.docx
```

如果需要共享数据集、权重或实验结果，建议使用网盘、Release 附件或单独的数据仓库。

## 致谢

本项目基于 RINet 相关代码进行扩展：

```bibtex
@ARTICLE{10054110,
  author={Song, Yingjie and Liu, Zhi and Li, Gongyang and Zeng, Dan and Zhang, Tianhong and Xu, Lihua and Wang, Jijun},
  journal={IEEE Transactions on Multimedia},
  title={RINet: Relative Importance-Aware Network for Fixation Prediction},
  year={2023},
  volume={25},
  pages={9263-9277},
  doi={10.1109/TMM.2023.3249481}
}
```
