# 🔬 PET低耦合事件识别系统 - 完整操作指南

## 📋 目录

- [系统概述](#系统概述)
- [环境准备](#环境准备)
- [快速开始](#快速开始)
- [详细操作流程](#详细操作流程)
- [API使用指南](#api使用指南)
- [结果分析](#结果分析)
- [常见问题](#常见问题)
- [技术支持](#技术支持)

---

## 🎯 系统概述

### 项目简介

本系统是基于图神经网络的PET(正电子发射断层成像)低耦合区域自动识别系统，能够智能分析PET探测器间的空间依赖关系，自动过滤低耦合区域内的无效事件，提升图像重建质量。

### 核心功能

- ✅ **智能事件分类**: 自动识别有效事件和低耦合事件
- ✅ **高精度预测**: 准确率达95%以上
- ✅ **实时处理**: 支持单样本和批量预测
- ✅ **可视化分析**: 提供详细的评估报告和图表
- ✅ **API服务**: REST接口支持集成应用

### 技术架构

- **深度学习框架**: PyTorch
- **模型类型**: 简化图神经网络(PETGNN)
- **数据处理**: 13维特征向量
- **部署方式**: Flask REST API

---

## 🛠️ 环境准备

### 系统要求

- **操作系统**: Windows 10/11, Linux, macOS
- **Python版本**: 3.8+
- **内存**: 建议8GB以上
- **存储**: 至少2GB可用空间

### 依赖安装

```bash
# 安装项目依赖
pip install -r requirements.txt
```

### 项目结构

```
pet-gnn-project/
├── 📁 data/                    # 数据目录
│   ├── raw/                   # 原始数据
│   ├── processed/             # 处理后数据
│   └── graphs/               # 图结构数据
├── 📁 models/                  # 模型定义
├── 📁 training/               # 训练脚本
├── 📁 evaluation/             # 评估工具
├── 📁 config/                 # 配置文件
├── 📁 utils/                  # 工具函数
├── 📄 simple_train.py         # 简化训练脚本
├── 📄 api_service.py          # API服务
├── 📄 predict_demo.py         # 预测演示
└── 📄 create_test_data.py     # 数据生成
```

---

## 🚀 快速开始

### 一键运行完整流程

```bash
# 1. 生成测试数据
python create_test_data.py

# 2. 训练模型
python simple_train.py

# 3. 评估模型
python evaluation/evaluate.py --save_plots

# 4. 启动API服务
python api_service.py
```

---

## 📖 详细操作流程

### 第1步：数据准备

#### 生成模拟数据

```bash
python create_test_data.py
```

**输出说明**:

- 📁 `data/processed/train_data.csv` - 训练集(700样本)
- 📁 `data/processed/val_data.csv` - 验证集(150样本)
- 📁 `data/processed/test_data.csv` - 测试集(150样本)
- 📁 `data/raw/pet_events.csv` - 完整数据集

**数据特征**:


| 特征名      | 描述                   | 单位 |
| ----------- | ---------------------- | ---- |
| pos_i_x/y/z | 探测器i的3D坐标        | mm   |
| pos_j_x/y/z | 探测器j的3D坐标        | mm   |
| E_i, E_j    | 探测到的能量值         | keV  |
| T_i, T_j    | 探测时间戳             | ns   |
| distance    | 探测器间距离           | mm   |
| energy_diff | 能量差                 | keV  |
| time_diff   | 时间差                 | ns   |
| label       | 标签(0=有效, 1=低耦合) | -    |

### 第2步：模型训练

#### 开始训练

```bash
python simple_train.py
```

**训练过程**:

- 🎯 **训练轮数**: 最多50轮(早停机制)
- 📊 **批次大小**: 32
- 🧠 **模型参数**: 约3,570个
- ⏱️ **训练时间**: 约2-5分钟

**训练输出示例**:

```
Epoch  1/50 | Train Loss: 0.6963 Acc: 63.86% | Val Loss: 0.5142 Acc: 0.7867 F1: 0.7984
Epoch  2/50 | Train Loss: 0.4956 Acc: 77.43% | Val Loss: 0.3713 Acc: 0.8533 F1: 0.8599
...
✅ 新的最佳模型! F1: 0.9733
⏹️ 早停触发 (patience=10)
```

**输出文件**:

- 📄 `best_model.pth` - 最佳模型权重

### 第3步：模型评估

#### 运行评估

```bash
python evaluation/evaluate.py
```

**评估指标**:

- ✅ **准确率(Accuracy)**: 整体预测正确率
- ✅ **精确率(Precision)**: 预测为正类中实际为正类的比例
- ✅ **召回率(Recall)**: 实际正类中被正确预测的比例
- ✅ **F1分数**: 精确率和召回率的调和平均

**输出文件**:

- 📄 `evaluation_results/prediction_results.csv` - 预测结果
- 📄 `evaluation_results/classification_report.txt` - 分类报告
- 📊 `evaluation_results/roc_curve.png` - ROC曲线
- 📊 `evaluation_results/precision_recall_curve.png` - PR曲线

#### 生成混淆矩阵

```bash
python evaluation/confusion_matrix.py
```

### 第4步：启动API服务

#### 启动服务

```bash
python api_service.py
```

**服务信息**:

- 🌐 **访问地址**: http://localhost:5000
- 📖 **API文档**: http://localhost:5000
- 💊 **健康检查**: http://localhost:5000/health

### 第5步：预测演示

#### 运行演示

```bash
python predict_demo.py
```

**演示内容**:

- 🔍 典型低耦合事件预测
- 🔍 典型有效事件预测
- 🔍 边界样本预测

---

## 🌐 API使用指南

### 接口列表

#### 1. 健康检查

```http
GET /health
```

**响应示例**:

```json
{
    "status": "healthy",
    "model_loaded": true,
    "device": "cpu",
    "timestamp": "2025-05-30T13:10:42.833286"
}
```

#### 2. 模型信息

```http
GET /model_info
```

**响应示例**:

```json
{
    "model_type": "PETGNN",
    "input_dim": 13,
    "output_dim": 2,
    "parameters": 3570,
    "device": "cpu"
}
```

#### 3. 单样本预测

```http
POST /predict
Content-Type: application/json

{
    "features": [10.0, 10.0, 5.0, 15.0, 12.0, 6.0, 510.0, 512.0, 100.0, 100.1, 8.5, 2.0, 0.1]
}
```

**响应示例**:

```json
{
    "prediction": 1,
    "confidence": 0.998,
    "probabilities": [0.002, 0.998],
    "class_names": ["有效事件", "低耦合事件"]
}
```

#### 4. 批量预测

```http
POST /batch_predict
Content-Type: application/json

{
    "features": [
        [10.0, 10.0, 5.0, 15.0, 12.0, 6.0, 510.0, 512.0, 100.0, 100.1, 8.5, 2.0, 0.1],
        [-50.0, 30.0, -10.0, 70.0, -20.0, 15.0, 508.0, 515.0, 200.0, 350.0, 156.2, 7.0, 150.0]
    ]
}
```

### 使用示例

#### PowerShell示例

```powershell
# 健康检查
Invoke-WebRequest -Uri "http://localhost:5000/health" -Method GET

# 单样本预测
$body = '{"features": [10.0, 10.0, 5.0, 15.0, 12.0, 6.0, 510.0, 512.0, 100.0, 100.1, 8.5, 2.0, 0.1]}'
Invoke-WebRequest -Uri "http://localhost:5000/predict" -Method POST -Body $body -ContentType "application/json"
```

#### Python示例

```python
import requests
import json

# 健康检查
response = requests.get("http://localhost:5000/health")
print(response.json())

# 单样本预测
data = {
    "features": [10.0, 10.0, 5.0, 15.0, 12.0, 6.0, 510.0, 512.0, 100.0, 100.1, 8.5, 2.0, 0.1]
}
response = requests.post("http://localhost:5000/predict", json=data)
print(response.json())
```

---

## 📊 结果分析

### 性能指标解读

#### 优秀性能标准

- ✅ **准确率 > 90%**: 系统整体性能良好
- ✅ **F1分数 > 90%**: 精确率和召回率平衡
- ✅ **错误率 < 10%**: 误分类样本较少

#### 典型结果示例

```
🎯 总体性能:
  准确率 (Accuracy):     95.33%
  精确率 (Precision):    95.30%
  召回率 (Recall):       95.33%
  F1分数 (F1-Score):     95.31%

📋 各类别详细指标:
  有效事件:    F1=96.89%
  低耦合事件:  F1=90.67%

⚠️ 错误分析:
  总样本数: 150
  正确预测: 143 (95.33%)
  错误预测: 7 (4.67%)
```

### 混淆矩阵分析

```
     预测
真实      有效事件   低耦合事件
有效事件       109       3
低耦合事件         4      34
```

**解读**:

- **真正例(TP)**: 34个低耦合事件被正确识别
- **真负例(TN)**: 109个有效事件被正确识别
- **假正例(FP)**: 3个有效事件被误判为低耦合
- **假负例(FN)**: 4个低耦合事件被误判为有效

---

## ❓ 常见问题

### Q1: 训练过程中出现依赖错误怎么办？

**A**: 使用简化训练脚本避免图神经网络依赖问题：

```bash
python simple_train.py  # 而不是 python training/train.py
```

### Q2: API服务无法启动怎么办？

**A**: 检查端口占用和依赖安装：

```bash
# 检查端口
netstat -an | findstr :5000

# 重新安装依赖
pip install flask torch pandas scikit-learn
```

### Q3: 中文显示乱码怎么办？

**A**: 系统已自动配置中文字体，如仍有问题：

```python
# 手动设置字体
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
```

### Q4: 模型性能不理想怎么办？

**A**: 尝试以下优化方法：

- 增加训练数据量
- 调整学习率和批次大小
- 增加训练轮数
- 检查数据质量

### Q5: 如何自定义数据？

**A**: 修改 `create_test_data.py` 中的数据生成逻辑，或准备符合格式的CSV文件。

---

## 🔧 技术支持

### 日志查看

```bash
# 查看训练日志
cat logs/training.log

# 查看API日志
cat api_service.log
```

### 配置文件

- 📄 `config/default.yaml` - 主配置文件
- 📄 `requirements.txt` - 依赖列表

### 联系方式

- 📧 **技术支持**: 查看项目README.md
- 📖 **文档**: 项目根目录下的Markdown文件
- 🐛 **问题反馈**: 通过项目仓库提交Issue

---

## 📝 更新日志

### v1.0.0 (2025-05-30)

- ✅ 完成基础模型训练功能
- ✅ 实现完整评估体系
- ✅ 添加REST API服务
- ✅ 支持中文界面
- ✅ 提供可视化分析

---

## 📄 许可证

本项目仅供学习和研究使用。
