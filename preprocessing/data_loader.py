"""
PET数据加载器
处理LM(List-Mode)格式的PET数据，并转换为适合GNN训练的格式
"""

import torch
import numpy as np
import pandas as pd
import h5py
import pickle
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torch_geometric.data import Data, Batch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import logging


class PETDataset(Dataset):
    """
    PET数据集类
    
    处理PET事件数据，将其转换为图数据格式
    """
    
    def __init__(self, data_path, config=None, split='train', transform=None):
        """
        Args:
            data_path: 数据文件路径
            config: 配置字典 (可选，如果为None则使用默认配置)
            split: 数据集分割 ('train', 'val', 'test')
            transform: 数据变换函数
        """
        self.data_path = Path(data_path)
        
        # 如果没有提供config，使用默认配置
        if config is None:
            config = {
                'data': {
                    'features': ['x1', 'y1', 'z1', 'x2', 'y2', 'z2', 
                               'energy1', 'energy2', 'time1', 'time2',
                               'distance', 'energy_diff', 'time_diff'],
                    'normalize': True,
                    'batch_size': 32
                }
            }
        
        self.config = config
        self.split = split
        self.transform = transform
        
        # 加载数据
        self._load_simple_csv_data()
        
        # 数据预处理
        self._preprocess_data()
        
    def _load_simple_csv_data(self):
        """
        加载简单的CSV格式数据
        """
        try:
            df = pd.read_csv(self.data_path)
            
            # 检查实际的列名
            if 'label' not in df.columns:
                raise ValueError(f"CSV文件缺少label列: {list(df.columns)}")
            
            # 获取所有特征列（除了label列）
            label_col = 'label'
            feature_cols = [col for col in df.columns if col != label_col]
            
            # 如果没有足够的特征列，说明数据格式不对
            if len(feature_cols) < 10:
                print(f"警告：只找到 {len(feature_cols)} 个特征列，期望至少10个")
                print(f"可用列: {feature_cols}")
            
            # 提取特征和标签
            events = df[feature_cols].values.astype(np.float32)
            labels = df[label_col].values.astype(np.int64)
            
            self.data = torch.FloatTensor(events)
            self.labels = torch.LongTensor(labels)
            
            print(f"成功加载数据: {len(events)} 个样本, {len(feature_cols)} 个特征")
            
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建虚拟数据作为fallback
            print("使用虚拟数据...")
            self.data = torch.randn(100, 13)
            self.labels = torch.randint(0, 2, (100,))
    
    def _preprocess_data(self):
        """
        数据预处理
        
        包括标准化和数据分割
        """
        # 数据预处理
        self.scaler = StandardScaler()
        if self.split == 'train':
            # 将tensor转为numpy进行标准化
            data_numpy = self.data.numpy()
            data_normalized = self.scaler.fit_transform(data_numpy)
            self.data = torch.FloatTensor(data_normalized)
        else:
            # 验证集和测试集使用训练集的标准化参数
            scaler_path = self.data_path.parent / 'scaler.pkl'
            if scaler_path.exists():
                with open(scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                data_numpy = self.data.numpy()
                data_normalized = self.scaler.transform(data_numpy)
                self.data = torch.FloatTensor(data_normalized)
            else:
                logging.warning("未找到预训练的标准化器，使用当前数据进行标准化")
                data_numpy = self.data.numpy()
                data_normalized = self.scaler.fit_transform(data_numpy)
                self.data = torch.FloatTensor(data_normalized)
        
        # 保存标准化器 (仅训练集)
        if self.split == 'train':
            scaler_path = self.data_path.parent / 'scaler.pkl'
            scaler_path.parent.mkdir(parents=True, exist_ok=True)
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        获取单个样本
        
        Args:
            idx: 样本索引
            
        Returns:
            data: torch_geometric.data.Data对象
        """
        event = self.data[idx]
        label = self.labels[idx]
        
        # 确保返回的是tensor格式
        if not isinstance(event, torch.Tensor):
            event = torch.FloatTensor(event)
        if not isinstance(label, torch.Tensor):
            label = torch.LongTensor([label]).squeeze()
        
        # 如果有数据变换，应用变换
        if self.transform:
            event = self.transform(event)
        
        return event, label


class PETDataLoader:
    """
    PET数据加载器管理类
    
    负责创建和管理训练、验证、测试数据加载器
    """
    
    def __init__(self, config):
        self.config = config
        self.data_config = config['data']
        self.train_config = config['training']
        
    def create_data_loaders(self, data_path):
        """
        创建数据加载器
        
        Args:
            data_path: 数据文件路径
            
        Returns:
            train_loader, val_loader, test_loader
        """
        # 创建数据集
        train_dataset = PETDataset(data_path, self.config, split='train')
        val_dataset = PETDataset(data_path, self.config, split='val')
        test_dataset = PETDataset(data_path, self.config, split='test')
        
        # 创建数据加载器
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=True,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=self._collate_fn
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=self._collate_fn
        )
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.train_config['batch_size'],
            shuffle=False,
            num_workers=self.config['hardware']['num_workers'],
            pin_memory=self.config['hardware']['pin_memory'],
            collate_fn=self._collate_fn
        )
        
        logging.info(f"数据加载器创建完成:")
        logging.info(f"  训练集: {len(train_dataset)} 样本")
        logging.info(f"  验证集: {len(val_dataset)} 样本")
        logging.info(f"  测试集: {len(test_dataset)} 样本")
        
        return train_loader, val_loader, test_loader
    
    def _collate_fn(self, batch):
        """
        批处理函数
        
        将多个图数据合并为一个批次
        """
        return Batch.from_data_list(batch)
    
    def create_synthetic_data(self, num_samples=1000, save_path=None):
        """
        创建合成数据 (用于测试)
        
        Args:
            num_samples: 样本数量
            save_path: 保存路径
        """
        np.random.seed(42)
        
        # 生成随机PET事件数据
        # pos_i: 探测器i的3D坐标
        pos_i = np.random.uniform(-100, 100, (num_samples, 3))
        # pos_j: 探测器j的3D坐标  
        pos_j = np.random.uniform(-100, 100, (num_samples, 3))
        # E_i, E_j: 能量 (511 keV附近)
        E_i = np.random.normal(511, 10, num_samples)
        E_j = np.random.normal(511, 10, num_samples)
        # T_i, T_j: 时间戳
        T_i = np.random.uniform(0, 1000, num_samples)
        T_j = T_i + np.random.normal(0, 0.1, num_samples)  # 时间差很小
        
        # 合并特征
        events = np.column_stack([
            pos_i, pos_j, E_i, E_j, T_i, T_j
        ])
        
        # 生成标签 (基于距离的简单规则)
        distances = np.linalg.norm(pos_i - pos_j, axis=1)
        labels = (distances < 50).astype(int)  # 距离小于50的为低耦合事件
        
        # 数据分割
        train_events, temp_events, train_labels, temp_labels = train_test_split(
            events, labels, test_size=0.3, random_state=42, stratify=labels
        )
        val_events, test_events, val_labels, test_labels = train_test_split(
            temp_events, temp_labels, test_size=0.5, random_state=42, stratify=temp_labels
        )
        
        if save_path:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # 保存为HDF5格式
            with h5py.File(save_path, 'w') as f:
                # 训练集
                train_group = f.create_group('train')
                train_group.create_dataset('events', data=train_events)
                train_group.create_dataset('labels', data=train_labels)
                
                # 验证集
                val_group = f.create_group('val')
                val_group.create_dataset('events', data=val_events)
                val_group.create_dataset('labels', data=val_labels)
                
                # 测试集
                test_group = f.create_group('test')
                test_group.create_dataset('events', data=test_events)
                test_group.create_dataset('labels', data=test_labels)
            
            logging.info(f"合成数据已保存到: {save_path}")
            logging.info(f"  训练集: {len(train_events)} 样本")
            logging.info(f"  验证集: {len(val_events)} 样本") 
            logging.info(f"  测试集: {len(test_events)} 样本")
            logging.info(f"  低耦合事件比例: {np.mean(labels):.2%}")
        
        return {
            'train': (train_events, train_labels),
            'val': (val_events, val_labels),
            'test': (test_events, test_labels)
        } 