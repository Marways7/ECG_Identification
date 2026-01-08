# ECG身份识别系统 - 技术文档

## Technical Documentation for ECG-Based Identity Recognition System

---

## 目录

1. [系统概述](#1-系统概述)
2. [信号预处理算法](#2-信号预处理算法)
3. [特征工程详解](#3-特征工程详解)
4. [模型架构设计](#4-模型架构设计)
5. [评估指标解析](#5-评估指标解析)
6. [技术选型论证](#6-技术选型论证)
7. [参考文献](#7-参考文献)

---

## 1. 系统概述

### 1.1 项目背景

本系统基于心电信号(ECG)实现生物特征身份识别，属于生理信号生物识别技术的前沿研究方向。相比传统生物识别技术(指纹、人脸等)，ECG具有以下优势:

- **活体检测内置**: ECG信号必须来自活体，天然防伪
- **难以伪造**: 心电信号的个体差异性极强
- **连续监测**: 可实现持续身份验证
- **隐私性强**: 不暴露外在生物特征

### 1.2 技术挑战

针对6人小样本数据集，主要技术挑战包括:

1. **过拟合风险**: 小样本易导致模型过度拟合训练数据
2. **类间差异小**: 同一生理信号的个体差异相对较小
3. **信号质量**: 需要有效去除噪声和伪迹
4. **泛化能力**: 模型需要对同一被试的新数据保持高识别率

### 1.3 方案选择论证

**为什么采用混合架构而非纯深度学习?**

| 方案 | 优势 | 劣势 | 适用场景 |
|------|------|------|----------|
| 纯深度学习 | 自动特征学习，端到端 | 需要大量数据，易过拟合 | 大数据集 |
| 传统机器学习 | 可解释性强，小样本友好 | 需要人工特征设计 | 小数据集 |
| **混合架构** | 结合两者优势，鲁棒性强 | 系统复杂度较高 | **本项目场景** |

最终选择**混合架构**的理由:
- 深度学习提取波形形态特征
- 传统特征(HRV/CRC)提供领域知识
- 集成学习增强泛化能力

---

## 2. 信号预处理算法

### 2.1 小波变换去噪

#### 2.1.1 数学原理

离散小波变换(DWT)将信号分解为多尺度近似系数和细节系数:

$$x(t) = \sum_{k} c_{J,k} \phi_{J,k}(t) + \sum_{j=1}^{J} \sum_{k} d_{j,k} \psi_{j,k}(t)$$

其中:
- $\phi_{j,k}(t) = 2^{j/2}\phi(2^j t - k)$ 为尺度函数
- $\psi_{j,k}(t) = 2^{j/2}\psi(2^j t - k)$ 为小波函数
- $c_{J,k}$ 为第$J$层近似系数
- $d_{j,k}$ 为第$j$层细节系数

#### 2.1.2 阈值去噪

采用**软阈值(Soft Thresholding)**处理细节系数:

$$d'_{j,k} = \text{sign}(d_{j,k}) \cdot \max(|d_{j,k}| - \lambda, 0)$$

**Universal阈值**选择:

$$\lambda = \hat{\sigma} \sqrt{2 \log N}$$

其中噪声标准差估计使用中位数绝对偏差(MAD):

$$\hat{\sigma} = \frac{\text{MAD}(d_1)}{0.6745}$$

#### 2.1.3 小波基选择

选择**Daubechies 4阶(db4)**小波的理由:
- 与QRS波群形态匹配度高
- 支持系数数量适中(8个)
- 近似对称性，减少相位失真

![小波去噪示意图](wavelet_denoising.png)

### 2.2 基线漂移校正

#### 2.2.1 形态学滤波算法

采用**开闭运算组合**估计基线:

$$\text{baseline} = \gamma_B(\phi_B(x))$$

其中:
- $\phi_B(x)$ 为结构元素$B$的闭运算(膨胀后腐蚀)
- $\gamma_B(x)$ 为结构元素$B$的开运算(腐蚀后膨胀)

**结构元素设计**: 长度约200ms，覆盖QRS波群但短于T波

校正后信号: $x' = x - \text{baseline}$

### 2.3 R峰检测

#### 2.3.1 Pan-Tompkins算法

经典的R峰检测算法，流程如下:

1. **带通滤波** (5-15Hz):
   $$y_1[n] = \text{BPF}(x[n])$$

2. **五点微分**:
   $$y_2[n] = \frac{1}{8T}(-y_1[n-2] - 2y_1[n-1] + 2y_1[n+1] + y_1[n+2])$$

3. **平方**:
   $$y_3[n] = y_2[n]^2$$

4. **移动窗口积分** (150ms窗口):
   $$y_4[n] = \frac{1}{N}\sum_{k=0}^{N-1}y_3[n-k]$$

5. **自适应双阈值检测**:
   - 信号峰值估计: $\text{SPKI} = 0.125 \cdot P + 0.875 \cdot \text{SPKI}$
   - 噪声峰值估计: $\text{NPKI} = 0.125 \cdot P + 0.875 \cdot \text{NPKI}$
   - 阈值: $\text{THR} = \text{NPKI} + 0.25(\text{SPKI} - \text{NPKI})$

---

## 3. 特征工程详解

### 3.1 HRV时域特征

心率变异性(HRV)基于RR间期序列 $\{RR_1, RR_2, ..., RR_N\}$ 计算:

| 指标 | 公式 | 生理意义 |
|------|------|----------|
| **SDNN** | $\sqrt{\frac{1}{N-1}\sum_{i=1}^{N}(RR_i - \overline{RR})^2}$ | 整体HRV，反映自主神经总体活动 |
| **RMSSD** | $\sqrt{\frac{1}{N-1}\sum_{i=1}^{N-1}(RR_{i+1} - RR_i)^2}$ | 短期HRV，主要反映副交感活动 |
| **pNN50** | $\frac{\#\{|\Delta RR_i| > 50ms\}}{N-1} \times 100\%$ | 反映副交感神经活动 |
| **CV** | $\frac{\text{SDNN}}{\overline{RR}} \times 100\%$ | 变异系数，标准化的变异性指标 |

### 3.2 HRV频域特征

通过功率谱密度(PSD)分析获取频域特征。首先将不等间隔RRI序列插值为等间隔序列，然后使用Welch方法估计PSD。

**频带划分** (Task Force of ESC and NASPE, 1996):

| 频带 | 频率范围 | 生理意义 |
|------|----------|----------|
| **VLF** | 0.003-0.04 Hz | 温度调节、RAAS系统 |
| **LF** | 0.04-0.15 Hz | 混合交感+副交感，以交感为主 |
| **HF** | 0.15-0.4 Hz | 副交感/迷走神经活动，与呼吸相关 |

**功率计算**:

$$P_{band} = \int_{f_{low}}^{f_{high}} S(f) df$$

**LF/HF比值**:

$$\text{LF/HF ratio} = \frac{P_{LF}}{P_{HF}}$$

反映交感/副交感平衡状态

### 3.3 HRV非线性特征

#### 3.3.1 Poincaré图分析

将RR间期绘制为散点图: $(RR_n, RR_{n+1})$

通过拟合椭圆提取参数:

$$SD1 = \sqrt{\frac{1}{2}\text{Var}(RR_{n+1} - RR_n)}$$

$$SD2 = \sqrt{2 \cdot \text{SDNN}^2 - SD1^2}$$

- **SD1**: 短期变异性(椭圆短轴)
- **SD2**: 长期变异性(椭圆长轴)
- **SD1/SD2**: 反映随机性与确定性成分比例

#### 3.3.2 样本熵 (Sample Entropy)

衡量序列不规则性:

$$SampEn(m, r, N) = -\ln\frac{A^{m+1}(r)}{B^m(r)}$$

其中:
- $m$: 嵌入维度 (通常取2)
- $r$: 相似性阈值 (通常取0.2×SDNN)
- $B^m(r)$: 长度为$m$的相似模板对数
- $A^{m+1}(r)$: 长度为$m+1$的相似模板对数

**解读**: SampEn越低，信号越规则；越高，信号越随机。

#### 3.3.3 去趋势波动分析 (DFA)

量化RR间期的分形特性:

1. **积分**: $y(k) = \sum_{i=1}^{k}[RR_i - \overline{RR}]$

2. **分段拟合**: 将$y(k)$分为长度为$n$的段，每段线性拟合

3. **波动函数**:
   $$F(n) = \sqrt{\frac{1}{N}\sum_{k=1}^{N}[y(k) - y_n(k)]^2}$$

4. **标度指数**: $F(n) \sim n^\alpha$

**解读**:
- $\alpha \approx 0.5$: 白噪声(无相关)
- $\alpha \approx 1.0$: 1/f噪声(长程相关)
- $\alpha \approx 1.5$: 布朗运动

健康人: $\alpha_1 \approx 1.0$ (短期)

### 3.4 心肺耦合(CRC)特征

心肺耦合描述心脏和呼吸系统的动态交互。

#### 3.4.1 ECG导出呼吸(EDR)

从ECG信号中提取呼吸信息:

1. **R峰幅度法**: 呼吸调制R波幅度
2. **RR间期法**: 呼吸性窦性心律不齐(RSA)
3. **QRS面积法**: 呼吸调制QRS波群面积

#### 3.4.2 相位同步指数(PSI)

使用Hilbert变换提取瞬时相位:

$$\phi(t) = \arctan\frac{\mathcal{H}[x(t)]}{x(t)}$$

相位差: $\Delta\phi(t) = \phi_{resp}(t) - n \cdot \phi_{cardiac}(t)$

**相位同步指数**:

$$PSI = \left|\left\langle e^{i\Delta\phi(t)}\right\rangle\right|$$

- $PSI = 1$: 完全同步
- $PSI = 0$: 完全不同步

#### 3.4.3 交叉谱相干性

$$C_{xy}(f) = \frac{|P_{xy}(f)|^2}{P_{xx}(f) \cdot P_{yy}(f)}$$

- $C_{xy} = 1$: 完全线性相关
- $C_{xy} = 0$: 无相关

---

## 4. 模型架构设计

### 4.1 1D-CNN编码器

#### 4.1.1 网络结构

```
Input: (B, 140) -> (B, 1, 140)
    ↓
[多尺度卷积层] - kernels: 3, 7, 15
    ↓
[下采样块1] Conv1D(64, 128, k=5, s=2) -> BN -> ReLU -> Dropout
    ↓
[下采样块2] Conv1D(128, 256, k=5, s=2) -> BN -> ReLU -> Dropout
    ↓
[残差块 ×3] Conv -> BN -> ReLU -> Conv -> BN + Skip
    ↓
[自注意力层] Self-Attention1D
    ↓
[全局池化] AdaptiveAvgPool + AdaptiveMaxPool
    ↓
[FC] Linear(512, 256) -> BN -> ReLU -> Linear(256, 128)
    ↓
Output: (B, 128) - 特征嵌入
```

#### 4.1.2 多尺度卷积

设计原理: 不同卷积核捕捉不同频率成分

| 卷积核大小 | 感受野(ms) | 捕捉内容 |
|------------|------------|----------|
| 3 | ~15ms | QRS波群尖峰 |
| 7 | ~35ms | 波形细节 |
| 15 | ~75ms | 波形包络 |

#### 4.1.3 自注意力机制

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

使模型能够关注信号中的关键位置(如R峰、T波顶点)

### 4.2 Siamese网络

#### 4.2.1 架构

```
Input1 ──┐
         │
         ├──> [共享ECGEncoder] ──> Embedding1 ──┐
         │                                       │
Input2 ──┘                                       ├──> Distance ──> Output
                                                 │
                              ──> Embedding2 ──┘
```

#### 4.2.2 对比损失 (Contrastive Loss)

$$\mathcal{L} = (1-y) \cdot d^2 + y \cdot \max(0, m-d)^2$$

其中:
- $y = 0$: 同一身份(拉近)
- $y = 1$: 不同身份(推远)
- $d$: 嵌入向量间的欧氏距离
- $m$: 边界(margin)

#### 4.2.3 为什么适合小样本?

1. **数据增强**: N个样本可生成 $O(N^2)$ 个配对
2. **度量学习**: 学习相似性而非类别边界
3. **开放集识别**: 新用户只需注册嵌入，无需重新训练

### 4.3 混合分类器

#### 4.3.1 特征融合策略

```
深度特征(128维) ──┐
                  │
                  ├──> [特征拼接] ──> [归一化] ──> [集成分类器]
                  │
手工特征(~100维) ──┘
```

#### 4.3.2 Stacking集成

第一层(基础分类器):
- Random Forest
- XGBoost
- LightGBM
- SVM

第二层(元分类器):
- Logistic Regression

$$\hat{y} = f_{meta}([p_1, p_2, p_3, p_4])$$

其中 $p_i$ 是第$i$个基础分类器的预测概率

---

## 5. 评估指标解析

### 5.1 混淆矩阵

对于多分类问题，混淆矩阵 $C$ 满足:
- $C_{i,j}$: 真实类别为$i$，预测为$j$的样本数

### 5.2 分类指标

针对类别$c$:

| 指标 | 公式 | 含义 |
|------|------|------|
| **敏感度 (Sensitivity/Recall)** | $\frac{TP}{TP + FN}$ | 正确识别类别$c$的能力 |
| **特异性 (Specificity)** | $\frac{TN}{TN + FP}$ | 正确排除非类别$c$的能力 |
| **精确率 (Precision)** | $\frac{TP}{TP + FP}$ | 预测为类别$c$的准确程度 |
| **F1分数** | $\frac{2 \cdot P \cdot R}{P + R}$ | 精确率和召回率的调和均值 |

### 5.3 物理含义

在身份识别场景中:

- **高敏感度**: 被试A的ECG几乎都能被正确识别为A
- **高特异性**: 非被试A的ECG很少被误判为A
- **高精确率**: 被判定为A的ECG确实属于A
- **高F1**: 综合性能优秀

### 5.4 多分类评估

**宏平均 (Macro-average)**:
$$F1_{macro} = \frac{1}{C}\sum_{c=1}^{C}F1_c$$

**加权平均 (Weighted-average)**:
$$F1_{weighted} = \sum_{c=1}^{C}\frac{N_c}{N}F1_c$$

---

## 6. 技术选型论证

### 6.1 为什么不使用纯深度学习?

**问题分析**:
- 每个被试仅约300秒数据
- 生成约500-700个心拍
- 6分类任务，每类约100个训练样本

**深度学习的局限**:
1. 参数量大，易过拟合
2. 需要大量数据才能充分学习
3. 黑盒模型，可解释性差

**我们的解决方案**:
- 使用轻量级CNN (参数<1M)
- 结合领域知识(HRV/CRC特征)
- 集成学习提高鲁棒性

### 6.2 为什么采用集成学习?

**理论依据**: 集成学习通过组合多个弱学习器，减少方差和偏差

$$\text{Error}_{ensemble} \leq \frac{\text{Error}_{single}}{M}$$

(在理想独立条件下)

**实际效果**:
- XGBoost擅长处理结构化特征
- SVM在高维空间表现好
- Random Forest对异常值鲁棒
- 集成后综合优势

### 6.3 小波去噪 vs 传统滤波

| 方法 | 优势 | 劣势 |
|------|------|------|
| **带通滤波** | 简单快速 | 可能滤除有用频率成分 |
| **小波去噪** | 自适应，保留信号细节 | 计算复杂度较高 |
| **EMD/EEMD** | 数据驱动，无需选择基函数 | 端点效应，模态混叠 |

选择**小波去噪**的理由:
- ECG信号为非平稳信号，适合时频分析
- db4小波与QRS形态匹配好
- 可以针对不同尺度(频率)独立处理

---

## 7. 参考文献

1. Pan, J., & Tompkins, W. J. (1985). A real-time QRS detection algorithm. *IEEE transactions on biomedical engineering*, (3), 230-236.

2. Task Force of the European Society of Cardiology. (1996). Heart rate variability: standards of measurement, physiological interpretation, and clinical use. *Circulation*, 93(5), 1043-1065.

3. Donoho, D. L., & Johnstone, I. M. (1994). Ideal spatial adaptation by wavelet shrinkage. *Biometrika*, 81(3), 425-455.

4. Bartsch, R. P., et al. (2012). Network physiology: how organ systems dynamically interact. *PloS one*, 7(8), e44832.

5. Bromley, J., et al. (1993). Signature verification using a "siamese" time delay neural network. *NIPS*.

6. Richman, J. S., & Moorman, J. R. (2000). Physiological time-series analysis using approximate entropy and sample entropy. *American Journal of Physiology-Heart and Circulatory Physiology*, 278(6), H2039-H2049.

7. Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides. *Physical review e*, 49(2), 1685.

---

## 附录A: 系统架构图

```
┌─────────────────────────────────────────────────────────────────┐
│                    ECG Identity Recognition System               │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │   Raw ECG    │───>│ Preprocessing │───>│   Feature    │       │
│  │    Signal    │    │   Pipeline    │    │  Extraction  │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                             │                    │               │
│                             ▼                    ▼               │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   Wavelet    │    │     HRV      │       │
│                      │  Denoising   │    │   Features   │       │
│                      └──────────────┘    └──────────────┘       │
│                             │                    │               │
│                             ▼                    │               │
│                      ┌──────────────┐            │               │
│                      │  Baseline    │            │               │
│                      │  Correction  │            │               │
│                      └──────────────┘            │               │
│                             │                    │               │
│                             ▼                    │               │
│                      ┌──────────────┐    ┌──────────────┐       │
│                      │   R-Peak     │    │     CRC      │       │
│                      │  Detection   │    │   Features   │       │
│                      └──────────────┘    └──────────────┘       │
│                             │                    │               │
│                             ▼                    │               │
│                      ┌──────────────┐            │               │
│                      │    Beat      │            │               │
│                      │ Segmentation │            │               │
│                      └──────────────┘            │               │
│                             │                    │               │
│                             ▼                    ▼               │
│                      ┌────────────────────────────┐             │
│                      │      Feature Fusion        │             │
│                      │  Deep Features + HRV + CRC │             │
│                      └────────────────────────────┘             │
│                                    │                             │
│                                    ▼                             │
│                      ┌────────────────────────────┐             │
│                      │    Hybrid Classifier       │             │
│                      │  (Stacking Ensemble)       │             │
│                      └────────────────────────────┘             │
│                                    │                             │
│                                    ▼                             │
│                      ┌────────────────────────────┐             │
│                      │   Identity Prediction      │             │
│                      │   + Confidence Score       │             │
│                      └────────────────────────────┘             │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 附录B: 代码结构

```
ECG_Identification/
├── ECG_Data/                    # 原始数据
│   ├── A1_processed.csv
│   ├── B1_processed.csv
│   └── ...
├── src/
│   ├── preprocessing/           # 预处理模块
│   │   ├── wavelet_denoising.py
│   │   ├── baseline_correction.py
│   │   ├── rpeak_detection.py
│   │   ├── beat_segmentation.py
│   │   └── signal_pipeline.py
│   ├── features/                # 特征工程模块
│   │   ├── hrv_features.py
│   │   ├── crc_features.py
│   │   ├── morphological_features.py
│   │   └── feature_pipeline.py
│   ├── models/                  # 模型模块
│   │   ├── cnn_models.py
│   │   ├── siamese_network.py
│   │   ├── hybrid_classifier.py
│   │   └── trainer.py
│   └── utils/                   # 工具模块
│       ├── data_loader.py
│       └── visualization.py
├── ui/
│   └── app.py                   # Streamlit UI
├── docs/
│   └── TECHNICAL_DOCUMENTATION.md
├── main.py                      # 主程序
├── requirements.txt             # 依赖
└── README.md
```

---

*文档版本: 1.0*
*最后更新: 2025年1月*
*作者: ECG-ID Research Team*
