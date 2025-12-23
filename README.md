# DQNimb 项目说明

简要：基于 DQN 思路训练分类器以应对不平衡分类问题。支持多种 backbone（1D 卷积、ResNet1D、LSTM、Transformer、以及用于图像的卷积网络 Q_Net_image）。训练与评估逻辑位于 `main.py`，模型定义位于 `Model.py`。

## 文件结构（关键）
- main.py — 主入口，训练/测试流程、超参设置与模型保存
- Model.py — 各类模型实现（Q_Net_image、TBM_conv1d 系列、ResNet32_1D、LSTM、BiLSTM、Transformer 等）
- datasets.py — 自定义数据集接口（提供 ImbalancedDataset）
- evaluate.py — 评估与结果写入逻辑
- README.md — 本文件
- 保存结果目录：在 `main.py` 中由 `save_dir` 指定（当前示例为 `/workspace/RL/DQNimb/patent`）

## 依赖（最低）
- Python 3.8+
- torch
- numpy
- pandas
- matplotlib
- tqdm
- openpyxl（可选，用于 Excel 单元格合并）
- 其它：项目中引用的自定义模块 `datasets`、`evaluate`

安装示例（Windows PowerShell / CMD）：
pip install torch numpy pandas matplotlib tqdm openpyxl

## 快速开始（训练）
1. 准备好 `datasets.py` 中的 `ImbalancedDataset` 所需数据（TBM 或图像数据）。
2. 在 `main.py` 中根据需要修改：
   - `TEST_ONLY`（False 表示训练 + 评估，True 表示仅加载已有模型进行评估）
   - `tbm_configs`（数据集名称和 rho）
   - `model_variant`（选择模型，例如 'Transformer'、'TBM_conv1d_1layer' 等）
   - `save_dir`（模型和结果保存目录）
3. 运行：
   python d:\SWJT-Leeds\科创\RL\DQNimb\main.py

## 快速开始（评估已有模型）
- 将 `TEST_ONLY = True`，并确保 `save_dir` 中存在对应命名的 .pth 文件（main.py 中的文件名格式见下）。
- 运行同上命令，脚本会循环加载并调用 `evaluate_model`。

## 模型选择与输入形状
在 `main.py` 中函数 `get_model_config(dataset_name, model_variant=None)` 定义了数据集到模型与输入形状的映射（常见示例）：
- 图像：mnist/fashion_mnist -> Q_Net_image (1,28,28)；cifar10/cifar100 -> Q_Net_image (3,32,32)
- TBM（时序）：如 'TBM_K' 等 -> TBM_conv1d，默认输入形状 (1024, 3) （表示 len_window=1024, feature_dim=3）
- 可通过传入 model_variant 覆盖默认模型类型（例如使用 Transformer 处理 TBM 数据）。

注意：TBM 模型在前向和训练中对输入维度有特定处理（[batch, len, channels] 与 [batch, channels, len] 之间的转换），请保持数据格式一致。

## 保存文件命名规则（示例）
{dataset_name}_rho{rho}_{model_type}_reward{reward_multiplier}_gamma{discount_factor}_训练完成比{training_ratio}_第{run}次.pth

此外会保存 loss 曲线 PNG 和 loss_history 的 .npy 文件，路径同 `save_dir`。

## 随机性与可复现
脚本在开始时调用 `set_random_seed(42)`，并在训练中设置了 CUDA 的确定性选项（torch.backends.cudnn.deterministic=True 等）。如需更改种子，请修改 `main.py` 中的调用。

## 自定义与调试要点
- 调整训练总步数：`MyRL.t_max`
- 经验回放大小、batch、学习率在 `MyRL.__init__` 中配置
- 奖励设计：`MyRL.compute_reward` 可按需修改少数/多数类奖励逻辑和终止条件
- 若出现维度错误，请检查数据在进入模型前的形状转换（main.py 中对不同 model_type 有不同预处理）

## 常见命令（Windows）
- 训练并保存模型：
  python d:\DQNimb\main.py
- 仅评估（加载已有模型）：
  编辑 main.py：将 TEST_ONLY = True，设置 save_dir 和相关文件存在，运行同上命令

## 输出与评估
- 模型权重：save_dir 下 .pth
- 损失曲线：PNG
- loss 历史：.npy
- 评估结果：evaluate.py 会将结果写入 save_dir 下的 Excel（evaluation_results.xlsx），`main.py` 包含对最近多次运行 G-mean 标准差的计算与写回逻辑（使用 pandas + openpyxl 可选合并单元格）

---
