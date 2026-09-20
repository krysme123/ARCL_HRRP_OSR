# ARCL_HRRP_OSR - 高分辨率距离像开集识别系统

> 基于 PyTorch 实现的雷达高分辨率距离像（HRRP）开集识别（Open-Set Recognition, OSR）算法框架。本项目使用自适应环形损失（AdapRingLoss）解决真实场景下未知目标识别难题。

## 📖 项目背景

在雷达自动目标识别（ATR）中，传统模型通常假设测试样本属于已知类别。然而，在实际应用中，雷达不可避免地会遇到未参与训练的未知目标。本项目聚焦于**开集识别（OSR）**挑战，旨在让模型在准确分类已知目标的同时，能够有效拒绝未知目标。

## 💡 核心贡献

- **算法改进**：提出/实现了自适应环形损失（AdapRingLoss），优化了特征空间分布，提升了开集识别性能。
- **多网络支持**：支持 VGG32ABN 处理 2D 特征，以及 CNN1D 处理 1D 原始 HRRP 序列数据。
- **完整框架**：提供了从数据加载、模型训练、到实验结果分析与可视化的全流程工程实现。

## 📊 框架图与实验结果

![Framework](/images/framework.png)
*图1：AdapRingLoss 网络框架图*

![Loss Curve](/images/loss_curve.png)
*图2：AdapRingLoss 训练收敛曲线*

## 📝 论文链接

- **论文标题**：[Adaptive Ring-constrained Loss for Radar HRRP Open Set Target Recognition]
- **论文链接**：[https://ieeexplore.ieee.org/document/11348010]

## 📁 项目结构

```text
ARCL_HRRP_OSR/
├── .idea/                           # IDE配置文件（可忽略）
├── Analysis/                        # 实验结果分析与可视化模块
├── Auxiliary/                       # 辅助工具和工具函数
├── Dataset/                         # 数据集加载与预处理模块
├── Loss/                            # 损失函数实现（包括环形损失等）
├── Network/                         # 神经网络模型定义
├── Train_Test/                      # 训练与测试流程控制
├── utils/                           # 通用工具函数库
├── universal_analysis_code.py       # 通用分析入口
├── universal_train_code.py          # 通用训练入口
└── README.md                        # 项目说明文档
```
## 🚀 快速开始

### 环境配置

- **Python**: 3.8 或更高版本（推荐 3.9+）
- **PyTorch**: 1.9 或更高版本（本代码在 2.5.1+cu121 上测试通过）
- **CUDA**: 11.0 或更高版本（推荐 12.1，用于 GPU 加速）
- **GPU**: 支持 CUDA 的 NVIDIA 显卡（推荐 RTX 4090 或更高性能显卡）

### 基本使用

#### 训练模型

使用 VGG32 网络训练 2D 数据，CNN1D 训练 1D 数据，使用 AdapRingLoss 损失：

```bash
python universal_train_code.py --network VGG32ABN --loss AdapRingLoss
python universal_train_code.py --network CNN1D --loss AdapRingLoss
```
#### 分析 AdapRingLoss 损失结果
```bash
python universal_analysis_code.py --loss AdapRingLoss
```
## 👤 我的工作

- 独立完成了 AdapRingLoss 的理论推导与 PyTorch 实现
- 设计并实现了多网络（VGG32ABN, CNN1D）训练与评估框架
- 完成实验对比与分析，验证了算法在开集识别任务上的有效性
