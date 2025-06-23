"""
图构建器 - 将PET数据转换为图结构
实现探测器间的图拓扑构建和特征提取
"""

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
import networkx as nx
from pathlib import Path
import pickle
import logging

class GraphBuilder:
    """
    PET图构建器
    
    将PET探测器事件数据转换为图结构，包括：
    1. 探测器节点建模
    2. 邻接关系构建  
    3. 节点和边特征提取
    4. 图数据格式转换
    """
    
    def __init__(self, config):
        self.config = config
        self.graph_config = config['graph']
        
        # 图构建参数
        self.k_neighbors = self.graph_config['k_neighbors']
        self.distance_threshold = self.graph_config['distance_threshold']
        self.coupling_threshold = self.graph_config['coupling_threshold']
        
        # 特征标准化器
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
        # 缓存
        self.detector_positions = None
        self.adjacency_matrix = None
        
        logging.info(f"图构建器初始化: k={self.k_neighbors}, "
                    f"distance_threshold={self.distance_threshold}")
    
    def build_detector_graph(self, events_data):
        """
        从PET事件数据构建探测器图
        
        Args:
            events_data: DataFrame, 包含PET事件数据
            
        Returns:
            graph_data: dict, 包含图的所有信息
        """
        logging.info("开始构建PET探测器图...")
        
        # 1. 提取探测器位置
        detector_positions = self._extract_detector_positions(events_data)
        
        # 2. 构建邻接关系
        adjacency_matrix, edge_weights = self._build_adjacency_matrix(detector_positions)
        
        # 3. 提取节点特征
        node_features = self._extract_node_features(detector_positions, events_data)
        
        # 4. 提取边特征
        edge_features = self._extract_edge_features(adjacency_matrix, detector_positions, events_data)
        
        # 5. 构建图数据
        graph_data = {
            'detector_positions': detector_positions,
            'adjacency_matrix': adjacency_matrix,
            'edge_weights': edge_weights,
            'node_features': node_features,
            'edge_features': edge_features,
            'num_nodes': len(detector_positions),
            'num_edges': np.sum(adjacency_matrix)
        }
        
        # 6. 转换为PyTorch格式
        torch_data = self._convert_to_torch_format(graph_data)
        graph_data.update(torch_data)
        
        logging.info(f"图构建完成: {graph_data['num_nodes']} 节点, "
                    f"{graph_data['num_edges']} 边")
        
        return graph_data
    
    def _extract_detector_positions(self, events_data):
        """提取所有探测器的位置信息"""
        # 提取探测器i和j的位置
        pos_i = events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values
        pos_j = events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values
        
        # 合并所有位置并去重
        all_positions = np.vstack([pos_i, pos_j])
        unique_positions = np.unique(all_positions, axis=0)
        
        logging.info(f"发现 {len(unique_positions)} 个唯一探测器位置")
        
        return unique_positions
    
    def _build_adjacency_matrix(self, detector_positions):
        """
        构建邻接矩阵
        
        基于两种连接策略：
        1. k-近邻连接
        2. 距离阈值连接
        """
        num_detectors = len(detector_positions)
        adjacency_matrix = np.zeros((num_detectors, num_detectors), dtype=np.float32)
        
        # 方法1: k-近邻连接
        if self.k_neighbors > 0:
            knn = NearestNeighbors(n_neighbors=self.k_neighbors + 1, metric='euclidean')
            knn.fit(detector_positions)
            distances, indices = knn.kneighbors(detector_positions)
            
            for i in range(num_detectors):
                for j in range(1, self.k_neighbors + 1):  # 跳过自己（索引0）
                    neighbor_idx = indices[i, j]
                    distance = distances[i, j]
                    
                    # 基于距离的权重
                    weight = np.exp(-distance / 10.0)  # 距离越近权重越大
                    adjacency_matrix[i, neighbor_idx] = weight
                    adjacency_matrix[neighbor_idx, i] = weight  # 无向图
        
        # 方法2: 距离阈值连接
        if self.distance_threshold > 0:
            distance_matrix = cdist(detector_positions, detector_positions, metric='euclidean')
            
            # 在阈值内的探测器连接
            threshold_mask = (distance_matrix <= self.distance_threshold) & (distance_matrix > 0)
            
            # 更新邻接矩阵
            for i in range(num_detectors):
                for j in range(num_detectors):
                    if threshold_mask[i, j]:
                        weight = 1.0 / (1.0 + distance_matrix[i, j])  # 距离倒数权重
                        adjacency_matrix[i, j] = max(adjacency_matrix[i, j], weight)
        
        # 计算边权重（非零元素）
        edge_weights = adjacency_matrix[adjacency_matrix > 0]
        
        logging.info(f"邻接矩阵构建完成: 密度 {np.mean(adjacency_matrix > 0):.3f}")
        
        return adjacency_matrix, edge_weights
    
    def _extract_node_features(self, detector_positions, events_data):
        """
        提取节点（探测器）特征
        
        特征包括：
        1. 空间坐标 (x, y, z)
        2. 晶体尺寸（模拟）
        3. 检测统计量
        4. 能量统计量
        """
        num_detectors = len(detector_positions)
        
        # 基础特征：空间坐标
        node_features = detector_positions.copy()  # [x, y, z]
        
        # 为每个探测器计算统计特征
        detector_stats = []
        
        for detector_pos in detector_positions:
            # 找到涉及此探测器的所有事件
            pos_i_match = np.all(
                np.abs(events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values - detector_pos) < 1e-6, 
                axis=1
            )
            pos_j_match = np.all(
                np.abs(events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values - detector_pos) < 1e-6, 
                axis=1
            )
            
            # 统计包含此探测器的事件
            detector_events_i = events_data[pos_i_match]
            detector_events_j = events_data[pos_j_match]
            
            # 计算统计特征
            if len(detector_events_i) > 0 or len(detector_events_j) > 0:
                # 合并能量数据
                energies = []
                if len(detector_events_i) > 0:
                    energies.extend(detector_events_i['E_i'].values)
                if len(detector_events_j) > 0:
                    energies.extend(detector_events_j['E_j'].values)
                
                avg_energy = np.mean(energies) if energies else 511.0
                event_count = len(detector_events_i) + len(detector_events_j)
                
                # 低耦合事件数
                coupling_events = 0
                if len(detector_events_i) > 0:
                    coupling_events += np.sum(detector_events_i['label'] == 1)
                if len(detector_events_j) > 0:
                    coupling_events += np.sum(detector_events_j['label'] == 1)
            else:
                avg_energy = 511.0  # 默认能量
                event_count = 0
                coupling_events = 0
            
            # 模拟晶体尺寸（基于位置的函数）
            crystal_size = 2.0 + 0.1 * np.sin(detector_pos[0] * 0.1)
            
            detector_stats.append([
                crystal_size,      # 晶体尺寸
                avg_energy,        # 平均能量
                event_count,       # 事件计数
                coupling_events    # 低耦合事件数
            ])
        
        detector_stats = np.array(detector_stats)
        
        # 合并所有特征
        node_features = np.column_stack([node_features, detector_stats])
        
        # 特征标准化
        node_features = self.node_scaler.fit_transform(node_features)
        
        logging.info(f"节点特征提取完成: 形状 {node_features.shape}")
        
        return node_features.astype(np.float32)
    
    def _extract_edge_features(self, adjacency_matrix, detector_positions, events_data):
        """
        提取边特征
        
        特征包括：
        1. 欧几里得距离
        2. 功能耦合强度
        3. 事件共现频率
        """
        # 找到所有边
        edge_indices = np.where(adjacency_matrix > 0)
        num_edges = len(edge_indices[0])
        
        edge_features = []
        
        for idx in range(num_edges):
            i, j = edge_indices[0][idx], edge_indices[1][idx]
            
            pos_i = detector_positions[i]
            pos_j = detector_positions[j]
            
            # 1. 欧几里得距离
            distance = np.linalg.norm(pos_i - pos_j)
            
            # 2. 邻接矩阵中的权重（已计算的连接强度）
            connection_strength = adjacency_matrix[i, j]
            
            # 3. 事件共现频率（两个探测器同时出现在事件中的频率）
            cooccurrence = self._calculate_cooccurrence(pos_i, pos_j, events_data)
            
            edge_features.append([distance, connection_strength, cooccurrence])
        
        edge_features = np.array(edge_features)
        
        # 特征标准化
        if len(edge_features) > 0:
            edge_features = self.edge_scaler.fit_transform(edge_features)
        
        logging.info(f"边特征提取完成: {num_edges} 条边")
        
        return edge_features.astype(np.float32)
    
    def _calculate_cooccurrence(self, pos_i, pos_j, events_data, tolerance=1e-6):
        """计算两个探测器的事件共现频率"""
        # 找到涉及这两个探测器的事件
        i_events = np.all(
            np.abs(events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values - pos_i) < tolerance, 
            axis=1
        ) | np.all(
            np.abs(events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values - pos_i) < tolerance, 
            axis=1
        )
        
        j_events = np.all(
            np.abs(events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values - pos_j) < tolerance, 
            axis=1
        ) | np.all(
            np.abs(events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values - pos_j) < tolerance, 
            axis=1
        )
        
        # 直接共现的事件（i和j在同一个事件中）
        direct_cooccurrence = np.sum(
            (np.all(np.abs(events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values - pos_i) < tolerance, axis=1) &
             np.all(np.abs(events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values - pos_j) < tolerance, axis=1)) |
            (np.all(np.abs(events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values - pos_j) < tolerance, axis=1) &
             np.all(np.abs(events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values - pos_i) < tolerance, axis=1))
        )
        
        total_events = len(events_data)
        return direct_cooccurrence / max(total_events, 1)
    
    def _convert_to_torch_format(self, graph_data):
        """转换为PyTorch Geometric格式"""
        # 转换邻接矩阵为边索引格式
        adjacency_matrix = graph_data['adjacency_matrix']
        edge_indices = np.where(adjacency_matrix > 0)
        edge_index = np.vstack([edge_indices[0], edge_indices[1]])
        
        # 转换为PyTorch张量
        torch_data = {
            'x': torch.tensor(graph_data['node_features'], dtype=torch.float),
            'edge_index': torch.tensor(edge_index, dtype=torch.long),
            'edge_attr': torch.tensor(graph_data['edge_features'], dtype=torch.float),
            'edge_weight': torch.tensor(graph_data['edge_weights'], dtype=torch.float)
        }
        
        return torch_data
    
    def save_graph_data(self, graph_data, save_path):
        """保存图数据"""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存图数据
        with open(save_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        # 保存标准化器
        scaler_path = save_path.parent / 'scalers.pkl'
        with open(scaler_path, 'wb') as f:
            pickle.dump({
                'node_scaler': self.node_scaler,
                'edge_scaler': self.edge_scaler
            }, f)
        
        logging.info(f"图数据已保存到: {save_path}")
    
    def load_graph_data(self, load_path):
        """加载图数据"""
        load_path = Path(load_path)
        
        # 加载图数据
        with open(load_path, 'rb') as f:
            graph_data = pickle.load(f)
        
        # 加载标准化器
        scaler_path = load_path.parent / 'scalers.pkl'
        if scaler_path.exists():
            with open(scaler_path, 'rb') as f:
                scalers = pickle.load(f)
                self.node_scaler = scalers['node_scaler']
                self.edge_scaler = scalers['edge_scaler']
        
        logging.info(f"图数据已从 {load_path} 加载")
        return graph_data
    
    def visualize_graph(self, graph_data, save_path=None):
        """可视化图结构"""
        try:
            import matplotlib.pyplot as plt
            
            # 创建NetworkX图
            G = nx.Graph()
            
            # 添加节点
            detector_positions = graph_data['detector_positions']
            for i, pos in enumerate(detector_positions):
                G.add_node(i, pos=(pos[0], pos[1]))  # 使用x,y坐标作为位置
            
            # 添加边
            adjacency_matrix = graph_data['adjacency_matrix']
            for i in range(len(detector_positions)):
                for j in range(i+1, len(detector_positions)):
                    if adjacency_matrix[i, j] > 0:
                        G.add_edge(i, j, weight=adjacency_matrix[i, j])
            
            # 绘制图
            plt.figure(figsize=(12, 8))
            pos = nx.get_node_attributes(G, 'pos')
            
            # 绘制节点
            nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                                 node_size=50, alpha=0.7)
            
            # 绘制边
            edges = G.edges()
            weights = [G[u][v]['weight'] for u, v in edges]
            nx.draw_networkx_edges(G, pos, alpha=0.5, width=weights)
            
            plt.title("PET探测器图结构")
            plt.xlabel("X坐标")
            plt.ylabel("Y坐标")
            plt.axis('equal')
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"图可视化已保存到: {save_path}")
            
            plt.show()
            
        except ImportError:
            logging.warning("matplotlib未安装，跳过图可视化") 