"""
简化版PET-GNN训练脚本
使用基本PyTorch功能，避免复杂依赖

主要功能：
1. 定义SimplePETNet神经网络模型
2. 创建PET数据集加载器
3. 实现训练和验证函数
4. 执行完整的模型训练流程
5. 保存最佳模型并进行测试集评估
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from pathlib import Path
import logging

# 设置日志格式，方便调试和监控训练过程
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class SimplePETNet(nn.Module):
    """
    简化的PET神经网络模型
    
    功能：使用全连接层处理PET事件特征，判断事件是否为低耦合事件
    
    网络结构：
    - 输入层：接收PET事件的多维特征
    - 多个隐藏层：每层包含线性变换、ReLU激活、批归一化、Dropout
    - 输出层：输出2个类别的概率（有效事件/低耦合事件）
    
    Args:
        input_dim: 输入特征维度（默认10个特征）
        hidden_dims: 隐藏层维度列表（默认[64, 32, 16]逐层递减）
        num_classes: 分类类别数（默认2：好事件/坏事件）
        dropout: Dropout概率，防止过拟合（默认0.1）
    """
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32, 16], num_classes=2, dropout=0.1):
        super(SimplePETNet, self).__init__()
        
        # 保存模型配置参数
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 动态构建网络层列表
        layers = []
        prev_dim = input_dim  # 前一层的输出维度
        
        # 循环创建隐藏层
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),    # 线性变换层
                nn.ReLU(),                          # ReLU激活函数
                nn.BatchNorm1d(hidden_dim),         # 批归一化，加速训练
                nn.Dropout(dropout)                 # Dropout，防止过拟合
            ])
            prev_dim = hidden_dim  # 更新维度为当前层输出
        
        # 添加输出层（不需要激活函数，后面会用softmax）
        layers.append(nn.Linear(prev_dim, num_classes))
        
        # 将所有层组合成一个顺序网络
        self.network = nn.Sequential(*layers)
        
        # 初始化网络权重
        self._init_weights()
    
    def _init_weights(self):
        """
        初始化网络权重
        使用Xavier均匀分布初始化，有助于训练稳定性
        """
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 对线性层使用Xavier初始化
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    # 偏置项初始化为0
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        """
        前向传播函数
        
        Args:
            x: 输入特征张量 [batch_size, input_dim]
            
        Returns:
            输出张量 [batch_size, num_classes]
        """
        return self.network(x)


class PETDataset(torch.utils.data.Dataset):
    """
    简化的PET数据集类
    
    功能：
    1. 从CSV文件加载PET事件数据
    2. 提取指定的特征列
    3. 对特征进行标准化处理
    4. 提供PyTorch DataLoader兼容的接口
    
    数据格式：
    - 特征：探测器位置、能量、距离等10个特征
    - 标签：0=有效事件，1=低耦合事件
    """
    
    def __init__(self, csv_file, feature_cols=None):
        """
        初始化数据集
        
        Args:
            csv_file: CSV数据文件路径
            feature_cols: 特征列名列表，如果为None则使用默认特征
        """
        # 读取CSV文件
        self.data = pd.read_csv(csv_file)
        
        if feature_cols is None:
            # 使用默认的主要特征（排除标签列）
            feature_cols = [
                'pos_i_x', 'pos_i_y', 'pos_i_z',  # 探测器i的3D坐标
                'pos_j_x', 'pos_j_y', 'pos_j_z',  # 探测器j的3D坐标  
                'E_i', 'E_j',                      # 两个探测器的能量
                'distance', 'energy_diff'          # 计算得出的距离和能量差
            ]
        
        # 提取特征和标签
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data['label'].values.astype(np.int64)
        
        # 对特征进行标准化处理（均值0，方差1）
        # 这样可以让不同尺度的特征在同一水平，有助于训练
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
    def __len__(self):
        """返回数据集大小"""
        return len(self.features)
    
    def __getitem__(self, idx):
        """
        获取指定索引的数据项
        
        Args:
            idx: 数据索引
            
        Returns:
            (features, label): 特征张量和标签张量
        """
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def train_epoch(model, train_loader, optimizer, criterion, device):
    """
    训练一个epoch（完整遍历一遍训练数据）
    
    Args:
        model: 神经网络模型
        train_loader: 训练数据加载器
        optimizer: 优化器（如Adam）
        criterion: 损失函数（如交叉熵）
        device: 计算设备（CPU或GPU）
        
    Returns:
        avg_loss: 平均损失
        accuracy: 训练准确率
    """
    model.train()  # 设置模型为训练模式（启用dropout和batch norm）
    total_loss = 0  # 累计损失
    correct = 0     # 正确预测数量
    total = 0       # 总样本数量
    
    # 遍历所有训练批次
    for batch_idx, (data, target) in enumerate(train_loader):
        # 将数据移动到指定设备（GPU或CPU）
        data, target = data.to(device), target.to(device)
        
        # 梯度清零（PyTorch会累积梯度）
        optimizer.zero_grad()
        
        # 前向传播：计算模型输出
        output = model(data)
        
        # 计算损失
        loss = criterion(output, target)
        
        # 反向传播：计算梯度
        loss.backward()
        
        # 更新模型参数
        optimizer.step()
        
        # 累计统计信息
        total_loss += loss.item()
        pred = output.argmax(dim=1)  # 获取预测类别
        correct += pred.eq(target).sum().item()  # 统计正确预测数量
        total += target.size(0)  # 累计样本数量
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device):
    """
    验证一个epoch（在验证集上评估模型性能）
    
    Args:
        model: 神经网络模型
        val_loader: 验证数据加载器
        criterion: 损失函数
        device: 计算设备
        
    Returns:
        avg_loss: 平均损失
        accuracy: 准确率
        precision: 精确率
        recall: 召回率
        f1: F1分数
    """
    model.eval()  # 设置模型为评估模式（关闭dropout和batch norm）
    total_loss = 0
    all_preds = []    # 存储所有预测结果
    all_targets = []  # 存储所有真实标签
    
    # 关闭梯度计算，节省内存和计算时间
    with torch.no_grad():
        for data, target in val_loader:
            # 将数据移动到指定设备
            data, target = data.to(device), target.to(device)
            
            # 前向传播
            output = model(data)
            
            # 计算损失
            loss = criterion(output, target)
            
            # 累计损失
            total_loss += loss.item()
            
            # 获取预测结果
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(val_loader)
    
    # 使用sklearn计算详细的分类指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, accuracy, precision, recall, f1


def main():
    """
    主训练函数
    
    执行完整的模型训练流程：
    1. 数据准备和加载
    2. 模型创建和配置
    3. 训练循环执行
    4. 最佳模型保存
    5. 测试集评估
    """
    print("🚀 开始PET-GNN模型训练")
    print("=" * 50)
    
    # 1. 设备配置
    # 优先使用GPU（如果可用），否则使用CPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"📱 使用设备: {device}")
    
    # 2. 数据文件检查
    train_file = 'data/processed/train_data.csv'
    val_file = 'data/processed/val_data.csv'
    test_file = 'data/processed/test_data.csv'
    
    # 如果训练数据不存在，自动创建测试数据
    if not Path(train_file).exists():
        print("❌ 训练数据不存在，重新创建...")
        from create_test_data import create_test_pet_data
        create_test_pet_data()
    
    # 3. 数据集创建
    print("📊 加载数据...")
    train_dataset = PETDataset(train_file)  # 训练集
    val_dataset = PETDataset(val_file)      # 验证集
    test_dataset = PETDataset(test_file)    # 测试集
    
    print(f"  训练集: {len(train_dataset)} 样本")
    print(f"  验证集: {len(val_dataset)} 样本")
    print(f"  测试集: {len(test_dataset)} 样本")
    
    # 4. 数据加载器创建
    batch_size = 32  # 批次大小：每次处理32个样本
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True    # 训练时打乱数据顺序
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False   # 验证时不打乱
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False   # 测试时不打乱
    )
    
    # 5. 模型创建
    input_dim = train_dataset.features.shape[1]  # 自动获取特征维度
    model = SimplePETNet(
        input_dim=input_dim,           # 输入维度
        hidden_dims=[64, 32, 16],      # 隐藏层维度（逐层递减）
        num_classes=2                  # 2分类：有效事件/低耦合事件
    )
    model = model.to(device)  # 将模型移动到指定设备
    
    # 计算并显示模型参数数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🧠 模型参数数量: {total_params:,}")
    
    # 6. 优化器和损失函数配置
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=0.001,           # 学习率
        weight_decay=1e-4   # L2正则化，防止过拟合
    )
    criterion = nn.CrossEntropyLoss()  # 交叉熵损失，适合多分类任务
    
    # 7. 训练参数设置
    epochs = 50           # 最大训练轮数
    best_f1 = 0          # 记录最佳F1分数
    patience = 10        # 早停耐心值：连续10轮无改进就停止
    patience_counter = 0 # 早停计数器
    
    print(f"🎯 开始训练 ({epochs} epochs)")
    print("-" * 50)
    
    # 8. 训练循环
    for epoch in range(epochs):
        # 训练阶段：模型学习训练数据
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证阶段：在验证集上评估模型性能
        val_loss, val_acc, val_precision, val_recall, val_f1 = validate_epoch(model, val_loader, criterion, device)
        
        # 打印当前轮次的训练结果
        print(f"Epoch {epoch+1:2d}/{epochs} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} F1: {val_f1:.4f}")
        
        # 保存最佳模型（基于F1分数）
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(model.state_dict(), 'best_model.pth')  # 保存模型权重
            patience_counter = 0  # 重置早停计数器
            print(f"  ✅ 新的最佳模型! F1: {val_f1:.4f}")
        else:
            patience_counter += 1  # 增加早停计数器
        
        # 早停检查：如果连续patience轮无改进，停止训练
        if patience_counter >= patience:
            print(f"  ⏹️ 早停触发 (patience={patience})")
            break
    
    # 9. 测试集评估
    print("\n" + "=" * 50)
    print("📋 测试集评估")
    
    # 加载最佳模型进行最终评估
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc, test_precision, test_recall, test_f1 = validate_epoch(model, test_loader, criterion, device)
    
    # 打印最终测试结果
    print(f"测试结果:")
    print(f"  准确率: {test_acc:.4f}")    # 整体正确率
    print(f"  精确率: {test_precision:.4f}")  # 预测为正类中真正为正类的比例
    print(f"  召回率: {test_recall:.4f}")     # 所有正类中被正确识别的比例
    print(f"  F1分数: {test_f1:.4f}")        # 精确率和召回率的调和平均数
    
    print("\n🎉 训练完成!")
    print(f"最佳模型已保存为: best_model.pth")


# 程序入口点
if __name__ == "__main__":
    main() 