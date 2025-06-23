"""
图数据集 - 专门处理图结构的PET数据
支持批处理和动态图构建
"""

import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import logging
from .graph_builder import GraphBuilder

class PETGraphDataset(Dataset):
    """
    PET图数据集
    
    将PET事件数据转换为图结构，支持：
    1. 动态图构建
    2. 批处理
    3. 数据增强
    4. 缓存机制
    """
    
    def __init__(self, data_path, config, mode='train', use_cache=True):
        """
        Args:
            data_path: 数据文件路径
            config: 配置字典
            mode: 数据模式 ('train', 'val', 'test')
            use_cache: 是否使用缓存
        """
        self.data_path = Path(data_path)
        self.config = config
        self.mode = mode
        self.use_cache = use_cache
        
        # 初始化图构建器
        self.graph_builder = GraphBuilder(config)
        
        # 加载数据
        self.events_data = self._load_events_data()
        
        # 构建或加载图数据
        self.graph_data = self._prepare_graph_data()
        
        # 准备样本索引
        self.sample_indices = self._prepare_samples()
        
        logging.info(f"图数据集初始化完成: {len(self.sample_indices)} 个样本")
    
    def _load_events_data(self):
        """加载事件数据"""
        if self.data_path.suffix == '.csv':
            data = pd.read_csv(self.data_path)
        else:
            raise ValueError(f"不支持的文件格式: {self.data_path.suffix}")
        
        logging.info(f"加载了 {len(data)} 个事件")
        return data
    
    def _prepare_graph_data(self):
        """准备图数据"""
        # 检查是否有缓存的图数据
        cache_path = self.data_path.parent / f'graph_cache_{self.mode}.pkl'
        
        if self.use_cache and cache_path.exists():
            logging.info("从缓存加载图数据...")
            graph_data = self.graph_builder.load_graph_data(cache_path)
        else:
            logging.info("构建新的图数据...")
            graph_data = self.graph_builder.build_detector_graph(self.events_data)
            
            if self.use_cache:
                self.graph_builder.save_graph_data(graph_data, cache_path)
        
        return graph_data
    
    def _prepare_samples(self):
        """准备样本索引"""
        # 每个事件作为一个样本
        # 实际应用中可能需要更复杂的样本策略
        return list(range(len(self.events_data)))
    
    def __len__(self):
        return len(self.sample_indices)
    
    def __getitem__(self, idx):
        """
        获取单个图样本
        
        Returns:
            sample: dict, 包含图数据和标签
        """
        event_idx = self.sample_indices[idx]
        event = self.events_data.iloc[event_idx]
        
        # 构建子图（以当前事件为中心）
        subgraph_data = self._build_event_subgraph(event, event_idx)
        
        return subgraph_data
    
    def _build_event_subgraph(self, event, event_idx):
        """
        为特定事件构建子图
        
        Args:
            event: 事件数据
            event_idx: 事件索引
            
        Returns:
            subgraph_data: 子图数据
        """
        # 获取事件涉及的探测器位置
        pos_i = np.array([event['pos_i_x'], event['pos_i_y'], event['pos_i_z']])
        pos_j = np.array([event['pos_j_x'], event['pos_j_y'], event['pos_j_z']])
        
        # 在全局图中找到对应的探测器节点
        detector_positions = self.graph_data['detector_positions']
        
        # 找到最近的探测器节点（处理浮点误差）
        node_i = self._find_nearest_detector(pos_i, detector_positions)
        node_j = self._find_nearest_detector(pos_j, detector_positions)
        
        # 构建以这两个节点为中心的子图
        subgraph_nodes = self._get_subgraph_nodes([node_i, node_j], k_hop=2)
        
        # 提取子图
        subgraph_data = self._extract_subgraph(subgraph_nodes, event, event_idx)
        
        return subgraph_data
    
    def _find_nearest_detector(self, target_pos, detector_positions, tolerance=1e-3):
        """找到最近的探测器节点"""
        distances = np.linalg.norm(detector_positions - target_pos, axis=1)
        nearest_idx = np.argmin(distances)
        
        # 检查是否在容差范围内
        if distances[nearest_idx] > tolerance:
            logging.warning(f"未找到精确匹配的探测器位置，最近距离: {distances[nearest_idx]:.6f}")
        
        return nearest_idx
    
    def _get_subgraph_nodes(self, center_nodes, k_hop=2):
        """获取k跳子图的节点"""
        adjacency_matrix = self.graph_data['adjacency_matrix']
        
        # 从中心节点开始的k跳邻居
        subgraph_nodes = set(center_nodes)
        current_nodes = set(center_nodes)
        
        for hop in range(k_hop):
            next_nodes = set()
            for node in current_nodes:
                # 找到该节点的邻居
                neighbors = np.where(adjacency_matrix[node] > 0)[0]
                next_nodes.update(neighbors)
            
            subgraph_nodes.update(next_nodes)
            current_nodes = next_nodes
        
        return sorted(list(subgraph_nodes))
    
    def _extract_subgraph(self, subgraph_nodes, event, event_idx):
        """提取子图数据"""
        # 节点映射：全局索引 -> 子图索引
        node_mapping = {global_idx: local_idx for local_idx, global_idx in enumerate(subgraph_nodes)}
        
        # 提取节点特征
        node_features = self.graph_data['node_features'][subgraph_nodes]
        
        # 提取子图的邻接关系
        adjacency_matrix = self.graph_data['adjacency_matrix']
        subgraph_adj = adjacency_matrix[np.ix_(subgraph_nodes, subgraph_nodes)]
        
        # 转换为边索引格式
        edge_indices = np.where(subgraph_adj > 0)
        edge_index = np.vstack([edge_indices[0], edge_indices[1]])
        
        # 提取边特征
        edge_features = []
        edge_weights = []
        
        for i, j in zip(edge_indices[0], edge_indices[1]):
            global_i, global_j = subgraph_nodes[i], subgraph_nodes[j]
            
            # 从全局图中找到对应的边特征
            global_edge_idx = self._find_edge_feature_index(global_i, global_j)
            if global_edge_idx is not None:
                edge_features.append(self.graph_data['edge_features'][global_edge_idx])
                edge_weights.append(subgraph_adj[i, j])
            else:
                # 默认边特征
                edge_features.append([0.0, subgraph_adj[i, j], 0.0])
                edge_weights.append(subgraph_adj[i, j])
        
        # 事件特征（可以作为图级特征）
        event_features = np.array([
            event['distance'],
            event['energy_diff'], 
            event['time_diff'],
            event['E_i'],
            event['E_j']
        ], dtype=np.float32)
        
        # 标签
        label = event['label']
        
        # 构建图数据字典
        subgraph_data = {
            'x': torch.tensor(node_features, dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(edge_features, dtype=torch.float) if edge_features else torch.empty((0, 3)),
            'edge_weight': torch.tensor(edge_weights, dtype=torch.float),
            'event_features': torch.tensor(event_features, dtype=torch.float),
            'y': torch.tensor(label, dtype=torch.long),
            'event_id': event_idx,
            'num_nodes': len(subgraph_nodes)
        }
        
        return subgraph_data
    
    def _find_edge_feature_index(self, node_i, node_j):
        """在边特征数组中找到对应的索引"""
        adjacency_matrix = self.graph_data['adjacency_matrix']
        
        # 计算边的全局索引
        edge_count = 0
        for i in range(adjacency_matrix.shape[0]):
            for j in range(adjacency_matrix.shape[1]):
                if adjacency_matrix[i, j] > 0:
                    if (i == node_i and j == node_j) or (i == node_j and j == node_i):
                        return edge_count
                    edge_count += 1
        
        return None
    
    def get_graph_statistics(self):
        """获取图数据统计信息"""
        stats = {
            'num_events': len(self.events_data),
            'num_detectors': self.graph_data['num_nodes'],
            'num_edges': self.graph_data['num_edges'],
            'graph_density': self.graph_data['num_edges'] / (self.graph_data['num_nodes'] * (self.graph_data['num_nodes'] - 1)),
            'node_feature_dim': self.graph_data['node_features'].shape[1],
            'edge_feature_dim': self.graph_data['edge_features'].shape[1],
            'label_distribution': self.events_data['label'].value_counts().to_dict()
        }
        
        return stats


def collate_graph_batch(batch):
    """
    图数据的批处理函数
    
    Args:
        batch: 图样本列表
        
    Returns:
        batched_data: 批处理后的图数据
    """
    # 分离不同类型的数据
    node_features = []
    edge_indices = []
    edge_attrs = []
    edge_weights = []
    event_features = []
    labels = []
    event_ids = []
    
    node_offset = 0
    
    for sample in batch:
        # 节点特征
        node_features.append(sample['x'])
        
        # 边索引（需要偏移）
        edge_index = sample['edge_index'] + node_offset
        edge_indices.append(edge_index)
        
        # 边特征
        edge_attrs.append(sample['edge_attr'])
        edge_weights.append(sample['edge_weight'])
        
        # 事件级特征
        event_features.append(sample['event_features'])
        
        # 标签
        labels.append(sample['y'])
        event_ids.append(sample['event_id'])
        
        # 更新节点偏移
        node_offset += sample['num_nodes']
    
    # 合并批次
    batched_data = {
        'x': torch.cat(node_features, dim=0),
        'edge_index': torch.cat(edge_indices, dim=1),
        'edge_attr': torch.cat(edge_attrs, dim=0) if edge_attrs[0].numel() > 0 else torch.empty((0, 3)),
        'edge_weight': torch.cat(edge_weights, dim=0),
        'event_features': torch.stack(event_features, dim=0),
        'y': torch.stack(labels, dim=0),
        'event_ids': event_ids,
        'batch_size': len(batch),
        'batch': torch.cat([torch.full((len(node_features[i]),), i, dtype=torch.long) 
                           for i in range(len(node_features))])
    }
    
    return batched_data 