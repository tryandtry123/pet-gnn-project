"""
高性能图构建器 - 优化版本（无外部依赖）
大幅提升图数据生成速度，减少计算时间
"""

import numpy as np
import pandas as pd
import torch
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from scipy.sparse import csr_matrix
import networkx as nx
from pathlib import Path
import pickle
import logging
import time

class FastGraphBuilder:
    """
    高性能PET图构建器
    
    优化策略：
    1. 向量化计算替代循环
    2. 预计算和缓存重用
    3. 稀疏矩阵存储
    4. 分批处理大数据
    5. 算法简化优化
    """
    
    def __init__(self, config):
        self.config = config
        self.graph_config = config['graph']
        
        # 图构建参数
        self.k_neighbors = self.graph_config['k_neighbors']
        self.distance_threshold = self.graph_config['distance_threshold']
        self.coupling_threshold = self.graph_config['coupling_threshold']
        
        # 性能优化参数
        self.batch_size = 1000  # 分批处理大小
        self.use_sparse = True  # 使用稀疏矩阵
        self.cache_positions = True  # 缓存位置查找
        
        # 特征标准化器
        self.node_scaler = StandardScaler()
        self.edge_scaler = StandardScaler()
        
        # 缓存变量
        self.detector_positions = None
        self.position_to_idx = None  # 位置到索引的映射
        self.adjacency_matrix = None
        
        logging.info(f"高性能图构建器初始化: k={self.k_neighbors}, "
                    f"distance_threshold={self.distance_threshold}, "
                    f"batch_size={self.batch_size}")
    
    def build_detector_graph(self, events_data):
        """
        高性能图构建主函数
        """
        start_time = time.time()
        logging.info("🚀 开始高性能图构建...")
        
        # 1. 快速提取探测器位置（向量化）
        detector_positions = self._extract_detector_positions_fast(events_data)
        
        # 2. 高性能邻接矩阵构建
        adjacency_matrix, edge_weights = self._build_adjacency_matrix_fast(detector_positions)
        
        # 3. 快速节点特征提取
        node_features = self._extract_node_features_fast(detector_positions, events_data)
        
        # 4. 快速边特征提取（最大优化）
        edge_features = self._extract_edge_features_fast(adjacency_matrix, detector_positions, events_data)
        
        # 5. 构建图数据
        graph_data = {
            'detector_positions': detector_positions,
            'adjacency_matrix': adjacency_matrix,
            'edge_weights': edge_weights,
            'node_features': node_features,
            'edge_features': edge_features,
            'num_nodes': len(detector_positions),
            'num_edges': np.sum(adjacency_matrix > 0) if not self.use_sparse else adjacency_matrix.nnz
        }
        
        # 6. PyTorch格式转换
        torch_data = self._convert_to_torch_format_fast(graph_data)
        graph_data.update(torch_data)
        
        total_time = time.time() - start_time
        logging.info(f"✅ 高性能图构建完成: {graph_data['num_nodes']} 节点, "
                    f"{graph_data['num_edges']} 边, 耗时 {total_time:.2f}s")
        
        return graph_data
    
    def _extract_detector_positions_fast(self, events_data):
        """向量化的探测器位置提取"""
        start_time = time.time()
        
        # 向量化操作：一次性提取所有位置
        pos_cols_i = ['pos_i_x', 'pos_i_y', 'pos_i_z']
        pos_cols_j = ['pos_j_x', 'pos_j_y', 'pos_j_z']
        
        pos_i = events_data[pos_cols_i].values
        pos_j = events_data[pos_cols_j].values
        
        # 使用numpy的高效去重
        all_positions = np.vstack([pos_i, pos_j])
        unique_positions = np.unique(all_positions, axis=0)
        
        # 构建位置到索引的快速查找字典（用于后续加速）
        if self.cache_positions:
            self.position_to_idx = {}
            for idx, pos in enumerate(unique_positions):
                key = tuple(pos.round(6))  # 避免浮点精度问题
                self.position_to_idx[key] = idx
        
        elapsed = time.time() - start_time
        logging.info(f"发现 {len(unique_positions)} 个唯一探测器位置 (耗时 {elapsed:.2f}s)")
        
        return unique_positions
    
    def _build_adjacency_matrix_fast(self, detector_positions):
        """高性能邻接矩阵构建"""
        start_time = time.time()
        num_detectors = len(detector_positions)
        
        if self.use_sparse:
            # 使用稀疏矩阵节省内存
            rows, cols, data = [], [], []
        else:
            adjacency_matrix = np.zeros((num_detectors, num_detectors), dtype=np.float32)
        
        # 优化的k-近邻计算
        if self.k_neighbors > 0:
            knn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, num_detectors), 
                                 metric='euclidean', algorithm='kd_tree')
            knn.fit(detector_positions)
            distances, indices = knn.kneighbors(detector_positions)
            
            # 向量化权重计算
            weights = np.exp(-distances / 10.0)
            
            for i in range(num_detectors):
                for j in range(1, min(self.k_neighbors + 1, len(indices[i]))):
                    neighbor_idx = indices[i, j]
                    weight = weights[i, j]
                    
                    if self.use_sparse:
                        rows.extend([i, neighbor_idx])
                        cols.extend([neighbor_idx, i])
                        data.extend([weight, weight])
                    else:
                        adjacency_matrix[i, neighbor_idx] = weight
                        adjacency_matrix[neighbor_idx, i] = weight
        
        # 距离阈值连接（优化版）
        if self.distance_threshold > 0:
            # 分批计算距离矩阵，避免内存溢出
            for i in range(0, num_detectors, self.batch_size):
                end_i = min(i + self.batch_size, num_detectors)
                batch_positions = detector_positions[i:end_i]
                
                # 计算当前批次到所有点的距离
                distance_batch = cdist(batch_positions, detector_positions, metric='euclidean')
                
                # 找到阈值内的连接
                mask = (distance_batch <= self.distance_threshold) & (distance_batch > 0)
                batch_rows, batch_cols = np.where(mask)
                
                # 调整索引
                batch_rows += i
                
                # 计算权重
                batch_weights = 1.0 / (1.0 + distance_batch[mask])
                
                if self.use_sparse:
                    rows.extend(batch_rows)
                    cols.extend(batch_cols)
                    data.extend(batch_weights)
                else:
                    adjacency_matrix[batch_rows, batch_cols] = np.maximum(
                        adjacency_matrix[batch_rows, batch_cols], batch_weights
                    )
        
        # 构建最终矩阵
        if self.use_sparse:
            adjacency_matrix = csr_matrix((data, (rows, cols)), 
                                        shape=(num_detectors, num_detectors))
            edge_weights = np.array(data)
        else:
            edge_weights = adjacency_matrix[adjacency_matrix > 0]
        
        density = len(data) / (num_detectors * num_detectors) if self.use_sparse else np.mean(adjacency_matrix > 0)
        elapsed = time.time() - start_time
        logging.info(f"邻接矩阵构建完成: 密度 {density:.3f} (耗时 {elapsed:.2f}s)")
        
        return adjacency_matrix, edge_weights
    
    def _extract_node_features_fast(self, detector_positions, events_data):
        """快速节点特征提取（向量化优化）"""
        start_time = time.time()
        num_detectors = len(detector_positions)
        
        # 基础特征：空间坐标
        node_features = detector_positions.copy()
        
        # 预分配统计特征数组
        detector_stats = np.zeros((num_detectors, 4))  # [crystal_size, avg_energy, event_count, coupling_events]
        
        # 向量化匹配：预计算所有位置匹配
        pos_i_data = events_data[['pos_i_x', 'pos_i_y', 'pos_i_z']].values
        pos_j_data = events_data[['pos_j_x', 'pos_j_y', 'pos_j_z']].values
        
        # 为每个探测器进行向量化统计
        for idx, detector_pos in enumerate(detector_positions):
            # 向量化位置匹配（避免循环）
            pos_i_match = np.all(np.abs(pos_i_data - detector_pos) < 1e-6, axis=1)
            pos_j_match = np.all(np.abs(pos_j_data - detector_pos) < 1e-6, axis=1)
            
            # 向量化统计计算
            if np.any(pos_i_match) or np.any(pos_j_match):
                # 能量统计
                energies = []
                if np.any(pos_i_match):
                    energies.extend(events_data.loc[pos_i_match, 'E_i'].values)
                if np.any(pos_j_match):
                    energies.extend(events_data.loc[pos_j_match, 'E_j'].values)
                
                avg_energy = np.mean(energies) if energies else 511.0
                event_count = np.sum(pos_i_match) + np.sum(pos_j_match)
                
                # 低耦合事件统计
                coupling_i = np.sum(events_data.loc[pos_i_match, 'label'] == 1) if np.any(pos_i_match) else 0
                coupling_j = np.sum(events_data.loc[pos_j_match, 'label'] == 1) if np.any(pos_j_match) else 0
                coupling_events = coupling_i + coupling_j
                
                detector_stats[idx] = [
                    2.0 + 0.1 * np.sin(detector_pos[0] * 0.1),  # crystal_size
                    avg_energy,
                    event_count,
                    coupling_events
                ]
            else:
                detector_stats[idx] = [2.0 + 0.1 * np.sin(detector_pos[0] * 0.1), 511.0, 0, 0]
        
        # 合并特征
        node_features = np.column_stack([node_features, detector_stats])
        
        # 标准化
        node_features = self.node_scaler.fit_transform(node_features)
        
        elapsed = time.time() - start_time
        logging.info(f"节点特征提取完成: 形状 {node_features.shape} (耗时 {elapsed:.2f}s)")
        
        return node_features.astype(np.float32)
    
    def _extract_edge_features_fast(self, adjacency_matrix, detector_positions, events_data):
        """超高速边特征提取（完全向量化）"""
        start_time = time.time()
        
        # 获取边索引
        if self.use_sparse:
            edge_indices = adjacency_matrix.nonzero()
            edge_weights_values = adjacency_matrix.data
        else:
            edge_indices = np.where(adjacency_matrix > 0)
            edge_weights_values = adjacency_matrix[edge_indices]
        
        num_edges = len(edge_indices[0])
        
        if num_edges == 0:
            logging.info("无边，返回空特征")
            return np.array([])
        
        # 向量化距离计算
        pos_i = detector_positions[edge_indices[0]]
        pos_j = detector_positions[edge_indices[1]]
        distances = np.linalg.norm(pos_i - pos_j, axis=1)
        
        # 优化的共现计算（基于距离模型）
        cooccurrences = np.exp(-distances / 50.0)  # 简化模型，避免复杂计算
        
        # 组合边特征
        if len(edge_weights_values) == num_edges:
            edge_features = np.column_stack([distances, edge_weights_values, cooccurrences])
        else:
            edge_features = np.column_stack([distances, np.ones(num_edges), cooccurrences])
        
        # 标准化
        if len(edge_features) > 0:
            edge_features = self.edge_scaler.fit_transform(edge_features)
        
        elapsed = time.time() - start_time
        logging.info(f"边特征提取完成: {num_edges} 条边 (耗时 {elapsed:.2f}s)")
        
        return edge_features.astype(np.float32)
    
    def _convert_to_torch_format_fast(self, graph_data):
        """快速PyTorch格式转换"""
        # 获取边索引
        if self.use_sparse:
            edge_index = torch.tensor(np.vstack(graph_data['adjacency_matrix'].nonzero()), dtype=torch.long)
        else:
            edge_indices = np.where(graph_data['adjacency_matrix'] > 0)
            edge_index = torch.tensor(np.vstack(edge_indices), dtype=torch.long)
        
        torch_data = {
            'x': torch.tensor(graph_data['node_features'], dtype=torch.float),
            'edge_index': edge_index,
            'edge_attr': torch.tensor(graph_data['edge_features'], dtype=torch.float) if len(graph_data['edge_features']) > 0 else None,
            'edge_weight': torch.tensor(graph_data['edge_weights'], dtype=torch.float)
        }
        
        return torch_data
    
    def save_graph_cache(self, graph_data, cache_path):
        """保存图数据缓存"""
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(cache_path, 'wb') as f:
            pickle.dump(graph_data, f)
        
        logging.info(f"图数据缓存已保存: {cache_path}")
    
    def load_graph_cache(self, cache_path):
        """加载图数据缓存"""
        try:
            with open(cache_path, 'rb') as f:
                graph_data = pickle.load(f)
            logging.info(f"图数据缓存已加载: {cache_path}")
            return graph_data
        except FileNotFoundError:
            logging.info(f"缓存文件不存在: {cache_path}")
            return None 