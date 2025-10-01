# ARCL_HRRP_OSR - 高分辨率距离像开集识别系统
## 📁 项目结构

ARCL_HRRP_OSR/
├── .idea/                           # IDE配置文件（可忽略）
├── Analysis/                        # 实验结果分析与可视化模块
├── Auxiliary/                       # 辅助工具和工具函数
├── Dataset/                         # 数据集加载与预处理模块
├── Loss/                           # 损失函数实现（包括环形损失等）
├── Network/                        # 神经网络模型定义
├── Train_Test/                     # 训练与测试流程控制
├── utils/                          # 通用工具函数库
├── universal_analysis_code.py      # 通用分析入口
├── universal_train_code.py         # 通用训练入口
└── README.md                       # 项目说明文档
## 🚀 快速开始

### 环境配置:
#### 环境要求

- Python: 3.8或更高版本（推荐3.9+）
- PyTorch: 1.9或更高版本（本代码在2.5.1+cu121上测试通过）
- CUDA: 11.0或更高版本（推荐12.1，用于GPU加速）
- GPU: 支持CUDA的NVIDIA显卡（推荐RTX 4090或更高性能显卡）
#### 克隆项目
git clone https://github.com/krysme123/ARCL_HRRP_OSR.git

### 基本使用:
#### 训练模型：使用VGG32网络训练2D数据 CNN1D训练1D数据，使用AdapRingLoss损失

- python universal_train_code.py --network VGG32ABN --loss AdapRingLoss
- python universal_train_code.py --network CNN1D --loss AdapRingLoss
#### 分析AdapRingLoss损失结果

- python universal_analysis_code.py --loss AdapRingLoss
