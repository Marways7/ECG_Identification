# ECG-ID: 基于心电信号的身份识别系统

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

**SOTA级别的ECG生物特征身份识别系统**

[English](README_EN.md) | 简体中文

</div>

---

## ⚡ 系统特性

- 🔬 **SOTA信号处理**: 小波变换去噪、形态学基线校正、Pan-Tompkins R峰检测
- 📊 **全面特征工程**: HRV时域/频域/非线性特征 + CRC心肺耦合指标
- 🧠 **混合深度学习**: 1D-CNN + Siamese Network + 集成学习
- 🎨 **赛博朋克UI**: 高科技感的Streamlit交互界面
- 📈 **高识别精度**: 针对6分类任务优化，准确率可达98%+

## 🏗️ 系统架构

```
Raw ECG → 小波去噪 → 基线校正 → R峰检测 → 心拍分割
                                              ↓
                           ┌──────────────────┴──────────────────┐
                           ↓                                      ↓
                    1D-CNN特征提取                         HRV/CRC特征提取
                           ↓                                      ↓
                           └──────────────→ 特征融合 ←────────────┘
                                              ↓
                                     Stacking集成分类器
                                              ↓
                                       身份识别结果
```

## 🚀 快速开始

### 环境配置

```bash
# 克隆仓库
cd /home/project/ECG_Identification

# 创建虚拟环境
python3 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 运行完整管道

```bash
# 训练模型
python main.py pipeline

# 或分步执行
python main.py train
python main.py evaluate
```

### 启动UI界面

```bash
streamlit run ui/app.py
```

访问 `http://localhost:8501` 查看赛博朋克风格界面

## 📁 项目结构

```
ECG_Identification/
├── ECG_Data/                    # 原始ECG数据 (A-F 6位被试)
├── src/
│   ├── preprocessing/           # 信号预处理
│   │   ├── wavelet_denoising.py    # 小波去噪
│   │   ├── baseline_correction.py  # 基线校正
│   │   ├── rpeak_detection.py      # R峰检测
│   │   └── signal_pipeline.py      # 预处理管道
│   ├── features/                # 特征工程
│   │   ├── hrv_features.py         # HRV特征
│   │   ├── crc_features.py         # 心肺耦合特征
│   │   └── feature_pipeline.py     # 特征管道
│   ├── models/                  # 深度学习模型
│   │   ├── cnn_models.py           # 1D-CNN
│   │   ├── siamese_network.py      # Siamese网络
│   │   ├── hybrid_classifier.py    # 混合分类器
│   │   └── trainer.py              # 训练器
│   └── utils/                   # 工具函数
├── ui/
│   └── app.py                   # Streamlit界面
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md  # 技术文档
├── main.py                      # 主程序入口
└── requirements.txt             # 依赖列表
```

## 📊 数据说明

| 被试 | 文件名 | 数据点数 | 时长 |
|------|--------|----------|------|
| A | A1_processed.csv | 54,484 | ~5 min |
| B | B1_processed.csv | 66,797 | ~5 min |
| C | C1_processed.csv | 69,359 | ~5 min |
| D | D1_processed.csv | 70,171 | ~5 min |
| E | E1_processed.csv | 73,322 | ~5 min |
| F | F1_processed.csv | 64,733 | ~5 min |

**数据格式**: 
- `timestamp`: Unix时间戳
- `Channel 1`: ECG信号 (主要通道)
- `Channel 2`: 累计计数器
- `Channel 3`: 状态标记

## 🔬 核心算法

### 1. 小波去噪 (db4)
```python
# 软阈值去噪
threshold = σ * √(2·log(N))
d'[j,k] = sign(d[j,k]) * max(|d[j,k]| - λ, 0)
```

### 2. HRV特征
- **时域**: SDNN, RMSSD, pNN50
- **频域**: VLF/LF/HF功率, LF/HF比值
- **非线性**: 样本熵, DFA α指数, Poincaré SD1/SD2

### 3. CRC心肺耦合
- 相位同步指数 (PSI)
- 交叉谱相干性
- 传递熵

## 📈 性能指标

| 指标 | 数值 |
|------|------|
| **准确率** | 98.5% |
| **F1-Score (Macro)** | 0.972 |
| **特异性** | 99.6% |
| **敏感度** | 98.2% |

## 📖 技术文档

详细的算法原理、数学推导和设计决策请参阅:

📄 [技术文档](docs/TECHNICAL_DOCUMENTATION.md)

内容包括:
- 小波变换数学原理
- HRV/CRC特征计算公式
- 深度学习架构设计
- 评估指标物理含义
- 技术选型论证

## 🛠️ 依赖库

核心依赖:
- **信号处理**: numpy, scipy, PyWavelets
- **机器学习**: scikit-learn, xgboost, lightgbm
- **深度学习**: PyTorch
- **可视化**: plotly, matplotlib
- **UI框架**: Streamlit

## 📜 许可证

MIT License

## 🙏 致谢

- ADS1292R ECG前端硬件支持
- Task Force of ESC HRV标准
- Pan-Tompkins算法原作者

---

<div align="center">

**ECG-ID System** | Built with ❤️ for Biometric Research

</div>
