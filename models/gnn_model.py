"""
PET图神经网络模型
基于PyTorch Geometric实现的PET低耦合区域识别模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 条件导入PyTorch Geometric模块 (Windows兼容性)
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    from torch_geometric.nn import BatchNorm, LayerNorm
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Geometric不可用，将使用简化的MLP模型")
    TORCH_GEOMETRIC_AVAILABLE = False

# 条件导入自定义层
try:
    from .layers import ResidualGCNLayer, SparseAttentionLayer
    CUSTOM_LAYERS_AVAILABLE = True
except ImportError:
    print("⚠️ 自定义层不可用，将使用基础实现")


class PETGraphNet(nn.Module):
    """
    PET图神经网络模型
    
    基于图神经网络的PET低耦合区域识别模型，具有以下特性：
    - 残差连接: 缓解深层网络梯度消失问题
    - 稀疏感知: 处理PET数据的稀疏性
    - 多层图卷积: 捕获不同尺度的空间依赖关系
    - 注意力机制: 自适应学习重要特征
    """
    
    def __init__(self, config):
        super(PETGraphNet, self).__init__()
        
        self.config = config
        self.hidden_dim = config['model']['hidden_dim']
        self.num_layers = config['model']['num_layers']
        self.dropout = config['model']['dropout']
        self.conv_type = config['model']['conv_type']
        self.use_residual = config['model']['residual']
        self.use_batch_norm = config['model']['batch_norm']
        
        # 计算输入特征维度
        self.input_dim = self._calculate_input_dim(config)
        
        # 输入投影层
        self.input_projection = nn.Linear(self.input_dim, self.hidden_dim)
        
        # 构建图卷积层
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        
        for i in range(self.num_layers):
            # 选择图卷积类型
            if self.conv_type == "GCN":
                if self.use_residual:
                    conv = ResidualGCNLayer(self.hidden_dim, self.hidden_dim)
                else:
                    conv = GCNConv(self.hidden_dim, self.hidden_dim)
                    
            elif self.conv_type == "GAT":
                heads = config['model']['attention']['heads']
                concat = config['model']['attention']['concat']
                conv = GATConv(
                    self.hidden_dim, 
                    self.hidden_dim // heads if concat else self.hidden_dim,
                    heads=heads,
                    concat=concat,
                    dropout=self.dropout
                )
                
            elif self.conv_type == "GraphSAGE":
                conv = SAGEConv(self.hidden_dim, self.hidden_dim)
                
            else:
                raise ValueError(f"Unsupported conv_type: {self.conv_type}")
            
            self.conv_layers.append(conv)
            
            # 批归一化层
            if self.use_batch_norm:
                self.norm_layers.append(BatchNorm(self.hidden_dim))
            else:
                self.norm_layers.append(nn.Identity())
        
        # 稀疏注意力层
        if CUSTOM_LAYERS_AVAILABLE:
            self.sparse_attention = SparseAttentionLayer(self.hidden_dim)
        else:
            # 使用简单的恒等映射作为替代
            self.sparse_attention = nn.Identity()
        
        # 图池化层 (用于子图级别预测)
        if TORCH_GEOMETRIC_AVAILABLE:
            self.global_pool = global_mean_pool
        else:
            self.global_pool = None
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.hidden_dim // 2, config['model']['output_dim'])
        )
        
        # 初始化权重
        self._initialize_weights()
    
    def _calculate_input_dim(self, config):
        """计算输入特征维度"""
        # PET事件特征: [pos_i, pos_j, E_i, E_j, T_i, T_j]
        event_dim = len(config['data']['event_features'])
        
        # 节点特征维度 (这里需要根据实际数据调整)
        node_dim = 4  # coordinates(3) + other_features(1)
        
        return event_dim + node_dim
    
    def _initialize_weights(self):
        """初始化模型权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, input_dim]
            edge_index: 边索引 [2, num_edges]
            batch: 批次索引 [num_nodes] (用于批处理)
            edge_attr: 边特征 [num_edges, edge_dim] (可选)
            
        Returns:
            logits: 分类结果 [batch_size, num_classes]
        """
        # 输入投影
        h = self.input_projection(x)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 图卷积层
        for i, (conv, norm) in enumerate(zip(self.conv_layers, self.norm_layers)):
            h_prev = h
            
            # 图卷积
            if isinstance(conv, (GCNConv, SAGEConv)):
                h = conv(h, edge_index)
            elif isinstance(conv, GATConv):
                h = conv(h, edge_index)
            elif isinstance(conv, ResidualGCNLayer):
                h = conv(h, edge_index)
            
            # 批归一化
            h = norm(h)
            
            # 激活函数
            h = F.relu(h)
            
            # Dropout
            h = F.dropout(h, p=self.dropout, training=self.training)
        
        # 稀疏注意力
        if CUSTOM_LAYERS_AVAILABLE:
            h = self.sparse_attention(h, edge_index)
        else:
            h = self.sparse_attention(h)  # 恒等映射，只需要h参数
        
        # 图池化 (将节点特征聚合为图级特征)
        if batch is None:
            # 单图情况
            graph_repr = torch.mean(h, dim=0, keepdim=True)
        else:
            # 批处理情况
            graph_repr = self.global_pool(h, batch)
        
        # 分类
        logits = self.classifier(graph_repr)
        
        return logits
    
    def predict(self, x, edge_index, batch=None, edge_attr=None):
        """
        预测函数
        
        Returns:
            predictions: 预测标签
            probabilities: 预测概率
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x, edge_index, batch, edge_attr)
            probabilities = F.softmax(logits, dim=1)
            predictions = torch.argmax(logits, dim=1)
        
        return predictions, probabilities
    
    def get_embeddings(self, x, edge_index, batch=None):
        """
        获取图嵌入表示 (用于可视化和分析)
        """
        self.eval()
        with torch.no_grad():
            # 输入投影
            h = self.input_projection(x)
            h = F.relu(h)
            
            # 通过图卷积层
            for conv, norm in zip(self.conv_layers, self.norm_layers):
                if isinstance(conv, (GCNConv, SAGEConv, ResidualGCNLayer)):
                    h = conv(h, edge_index)
                elif isinstance(conv, GATConv):
                    h = conv(h, edge_index)
                
                h = norm(h)
                h = F.relu(h)
            
            # 图池化
            if batch is None:
                graph_repr = torch.mean(h, dim=0, keepdim=True)
            else:
                graph_repr = self.global_pool(h, batch)
        
        return graph_repr


class PETGNN(nn.Module):
    """
    简化的PET神经网络模型 (用于兼容性)
    
    当PyTorch Geometric不可用时，使用传统的全连接网络
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