"""
简化的PET神经网络模型
不依赖PyTorch Geometric和torch_scatter，用于兼容性
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class PETGNN(nn.Module):
    """
    简化的PET神经网络模型
    
    使用传统的全连接网络，不依赖图神经网络库
    """
    
    def __init__(self, input_dim=13, hidden_dim=64, output_dim=2, num_layers=3, dropout=0.1):
        super(PETGNN, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout
        
        # 构建网络层
        layers = []
        
        # 输入层
        layers.append(nn.Linear(input_dim, hidden_dim))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        
        # 隐藏层
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        
        # 输出层
        layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            out: 输出logits [batch_size, output_dim]
        """
        return self.model(x)
    
    def predict(self, x):
        """
        预测方法
        
        Args:
            x: 输入特征
            
        Returns:
            predictions: 预测类别
            probabilities: 预测概率
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            
        return predictions, probabilities 