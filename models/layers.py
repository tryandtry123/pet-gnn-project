"""
自定义神经网络层
为PET-GNN模型提供专用的图神经网络层
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# 条件导入PyTorch Geometric相关模块
try:
    from torch_geometric.nn import GCNConv, MessagePassing
    from torch_geometric.utils import add_self_loops, degree
    TORCH_GEOMETRIC_AVAILABLE = True
except ImportError:
    print("⚠️ PyTorch Geometric不可用")
    TORCH_GEOMETRIC_AVAILABLE = False

# 条件导入torch_scatter
try:
    from torch_scatter import scatter_add
    TORCH_SCATTER_AVAILABLE = True
except ImportError:
    print("⚠️ torch_scatter不可用，使用PyTorch原生实现")
    TORCH_SCATTER_AVAILABLE = False
    
    # 提供torch_scatter的简单替代实现
    def scatter_add(src, index, dim=0, dim_size=None):
        """scatter_add的简单替代实现"""
        if dim_size is None:
            dim_size = index.max().item() + 1
        
        shape = list(src.shape)
        shape[dim] = dim_size
        out = torch.zeros(shape, dtype=src.dtype, device=src.device)
        out.scatter_add_(dim, index.expand_as(src), src)
        return out


class ResidualGCNLayer(nn.Module):
    """
    残差图卷积层
    
    结合了残差连接的GCN层，有助于训练更深的网络
    """
    
    def __init__(self, in_channels, out_channels, bias=True, **kwargs):
        super(ResidualGCNLayer, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        # 主要的图卷积层
        self.gcn = GCNConv(in_channels, out_channels, bias=bias, **kwargs)
        
        # 残差连接的投影层 (当输入输出维度不同时)
        if in_channels != out_channels:
            self.residual_projection = nn.Linear(in_channels, out_channels, bias=False)
        else:
            self.residual_projection = nn.Identity()
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, edge_index, edge_weight=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, in_channels]
            edge_index: 边索引 [2, num_edges]
            edge_weight: 边权重 [num_edges] (可选)
            
        Returns:
            output: 输出特征 [num_nodes, out_channels]
        """
        # 残差连接
        residual = self.residual_projection(x)
        
        # 图卷积
        out = self.gcn(x, edge_index, edge_weight)
        
        # 残差连接
        out = out + residual
        
        # 层归一化
        out = self.layer_norm(out)
        
        return out


class SparseAttentionLayer(nn.Module):
    """
    稀疏注意力层
    
    专门为处理稀疏图数据设计的注意力机制
    能够自适应地关注重要的节点和边
    """
    
    def __init__(self, hidden_dim, heads=4, dropout=0.1):
        super(SparseAttentionLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.heads = heads
        self.head_dim = hidden_dim // heads
        self.dropout = dropout
        
        assert hidden_dim % heads == 0, "hidden_dim must be divisible by heads"
        
        # 查询、键、值投影层
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
        # 输出投影层
        self.W_o = nn.Linear(hidden_dim, hidden_dim)
        
        # 稀疏性门控机制
        self.sparsity_gate = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid()
        )
        
        # Dropout
        self.dropout_layer = nn.Dropout(dropout)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in [self.W_q, self.W_k, self.W_v, self.W_o]:
            nn.init.xavier_uniform_(module.weight)
    
    def forward(self, x, edge_index):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            edge_index: 边索引 [2, num_edges]
            
        Returns:
            output: 输出特征 [num_nodes, hidden_dim]
        """
        num_nodes = x.size(0)
        
        # 计算查询、键、值
        Q = self.W_q(x).view(num_nodes, self.heads, self.head_dim)  # [N, H, D]
        K = self.W_k(x).view(num_nodes, self.heads, self.head_dim)  # [N, H, D]
        V = self.W_v(x).view(num_nodes, self.heads, self.head_dim)  # [N, H, D]
        
        # 稀疏注意力计算 (只在连接的节点间计算注意力)
        row, col = edge_index
        
        # 计算注意力分数
        q_i = Q[row]  # [E, H, D] 源节点查询
        k_j = K[col]  # [E, H, D] 目标节点键
        
        # 注意力分数
        attn_scores = (q_i * k_j).sum(dim=-1) / (self.head_dim ** 0.5)  # [E, H]
        
        # 稀疏性门控
        sparsity_scores = self.sparsity_gate(x[row]).squeeze(-1)  # [E]
        attn_scores = attn_scores * sparsity_scores.unsqueeze(-1)  # [E, H]
        
        # Softmax归一化 (对每个目标节点的所有源节点)
        attn_weights = self._sparse_softmax(attn_scores, col, num_nodes)  # [E, H]
        
        # Dropout
        attn_weights = self.dropout_layer(attn_weights)
        
        # 聚合值
        v_j = V[col]  # [E, H, D]
        aggregated = attn_weights.unsqueeze(-1) * v_j  # [E, H, D]
        
        # 按目标节点聚合
        output = scatter_add(aggregated, col, dim=0, dim_size=num_nodes)  # [N, H, D]
        output = output.view(num_nodes, self.hidden_dim)  # [N, H*D]
        
        # 输出投影
        output = self.W_o(output)
        
        # 残差连接
        output = output + x
        
        return output
    
    def _sparse_softmax(self, scores, index, num_nodes):
        """
        稀疏softmax操作
        
        Args:
            scores: 注意力分数 [num_edges, heads]
            index: 目标节点索引 [num_edges]
            num_nodes: 节点总数
            
        Returns:
            weights: 归一化后的注意力权重 [num_edges, heads]
        """
        # 计算每个目标节点的最大分数 (数值稳定性)
        max_scores = scatter_add(scores, index, dim=0, dim_size=num_nodes)  # [N, H]
        max_scores = max_scores[index]  # [E, H]
        
        # 减去最大值
        scores = scores - max_scores
        
        # 计算exp
        exp_scores = torch.exp(scores)
        
        # 计算每个目标节点的分母
        sum_exp = scatter_add(exp_scores, index, dim=0, dim_size=num_nodes)  # [N, H]
        sum_exp = sum_exp[index]  # [E, H]
        
        # 归一化
        weights = exp_scores / (sum_exp + 1e-8)
        
        return weights


class GraphPoolingLayer(nn.Module):
    """
    图池化层
    
    将节点级特征聚合为图级特征
    支持多种池化策略
    """
    
    def __init__(self, hidden_dim, pool_type="mean"):
        super(GraphPoolingLayer, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.pool_type = pool_type
        
        if pool_type == "attention":
            # 注意力池化
            self.attention_weights = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1)
            )
        elif pool_type == "set2set":
            # Set2Set池化 (更复杂但效果更好)
            from torch_geometric.nn import Set2Set
            self.set2set = Set2Set(hidden_dim, processing_steps=3)
        
    def forward(self, x, batch=None):
        """
        前向传播
        
        Args:
            x: 节点特征 [num_nodes, hidden_dim]
            batch: 批次索引 [num_nodes]
            
        Returns:
            graph_repr: 图级表示
        """
        if batch is None:
            # 单图情况
            if self.pool_type == "mean":
                return torch.mean(x, dim=0, keepdim=True)
            elif self.pool_type == "max":
                return torch.max(x, dim=0, keepdim=True)[0]
            elif self.pool_type == "attention":
                weights = F.softmax(self.attention_weights(x), dim=0)
                return torch.sum(weights * x, dim=0, keepdim=True)
        else:
            # 批处理情况
            from torch_geometric.nn import global_mean_pool, global_max_pool
            
            if self.pool_type == "mean":
                return global_mean_pool(x, batch)
            elif self.pool_type == "max":
                return global_max_pool(x, batch)
            elif self.pool_type == "set2set":
                return self.set2set(x, batch)
            elif self.pool_type == "attention":
                # 实现批量注意力池化
                weights = self.attention_weights(x)
                weights = scatter_add(F.softmax(weights, dim=0), batch, dim=0)
                weighted_features = scatter_add(weights[batch] * x, batch, dim=0)
                return weighted_features 