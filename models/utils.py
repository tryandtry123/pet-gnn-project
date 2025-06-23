"""
模型工具函数
提供模型保存、加载、初始化等实用功能
"""

import torch
import os
import logging
from pathlib import Path


def save_model(model, optimizer, scheduler, epoch, metrics, save_path):
    """
    保存模型检查点
    
    Args:
        model: 模型实例
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        metrics: 性能指标字典
        save_path: 保存路径
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'model_config': model.config if hasattr(model, 'config') else None
    }
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    torch.save(checkpoint, save_path)
    logging.info(f"模型已保存到: {save_path}")


def load_model(model, save_path, device='cpu', load_optimizer=False, 
               optimizer=None, scheduler=None):
    """
    加载模型检查点
    
    Args:
        model: 模型实例
        save_path: 检查点路径
        device: 设备
        load_optimizer: 是否加载优化器状态
        optimizer: 优化器实例 (当load_optimizer=True时必需)
        scheduler: 学习率调度器实例
        
    Returns:
        epoch: 加载的轮次
        metrics: 性能指标
    """
    if not os.path.exists(save_path):
        raise FileNotFoundError(f"检查点文件不存在: {save_path}")
    
    checkpoint = torch.load(save_path, map_location=device)
    
    # 加载模型状态
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # 加载优化器状态
    if load_optimizer and optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # 加载调度器状态
    if scheduler is not None and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    logging.info(f"模型已从 {save_path} 加载 (Epoch: {epoch})")
    
    return epoch, metrics


def count_parameters(model):
    """
    计算模型参数数量
    
    Args:
        model: 模型实例
        
    Returns:
        total_params: 总参数数
        trainable_params: 可训练参数数
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return total_params, trainable_params


def initialize_weights(model, init_type='xavier'):
    """
    初始化模型权重
    
    Args:
        model: 模型实例
        init_type: 初始化方法 ('xavier', 'kaiming', 'normal')
    """
    for m in model.modules():
        if isinstance(m, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d)):
            if init_type == 'xavier':
                torch.nn.init.xavier_uniform_(m.weight)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            elif init_type == 'normal':
                torch.nn.init.normal_(m.weight, mean=0, std=0.01)
            
            if m.bias is not None:
                torch.nn.init.constant_(m.bias, 0)
        
        elif isinstance(m, (torch.nn.BatchNorm1d, torch.nn.BatchNorm2d)):
            torch.nn.init.constant_(m.weight, 1)
            torch.nn.init.constant_(m.bias, 0)


def get_device(prefer_gpu=True):
    """
    获取可用设备
    
    Args:
        prefer_gpu: 是否优先使用GPU
        
    Returns:
        device: torch.device对象
    """
    if prefer_gpu and torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    
    return device


def move_batch_to_device(batch, device):
    """
    将批次数据移动到指定设备
    
    Args:
        batch: 批次数据 (字典或元组)
        device: 目标设备
        
    Returns:
        batch: 移动后的批次数据
    """
    if isinstance(batch, dict):
        return {key: value.to(device) if isinstance(value, torch.Tensor) else value 
                for key, value in batch.items()}
    elif isinstance(batch, (list, tuple)):
        return [item.to(device) if isinstance(item, torch.Tensor) else item 
                for item in batch]
    elif isinstance(batch, torch.Tensor):
        return batch.to(device)
    else:
        return batch


def create_model_from_config(config, device='cpu'):
    """
    从配置文件创建模型
    
    Args:
        config: 配置字典
        device: 设备
        
    Returns:
        model: 创建的模型实例
    """
    from .gnn_model import PETGraphNet
    
    model = PETGraphNet(config)
    model = model.to(device)
    
    # 初始化权重
    initialize_weights(model, config.get('model', {}).get('init_type', 'xavier'))
    
    # 打印模型信息
    total_params, trainable_params = count_parameters(model)
    logging.info(f"模型参数: 总计 {total_params:,}, 可训练 {trainable_params:,}")
    
    return model


def save_best_model(model, optimizer, scheduler, epoch, metrics, 
                   save_dir, metric_name='val_f1', higher_better=True):
    """
    保存最佳模型 (基于指定指标)
    
    Args:
        model: 模型实例
        optimizer: 优化器
        scheduler: 学习率调度器
        epoch: 当前轮次
        metrics: 性能指标字典
        save_dir: 保存目录
        metric_name: 用于判断的指标名称
        higher_better: 指标是否越高越好
        
    Returns:
        is_best: 是否是最佳模型
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    best_metric_file = save_dir / 'best_metric.txt'
    
    # 读取历史最佳指标
    if best_metric_file.exists():
        with open(best_metric_file, 'r') as f:
            try:
                best_metric = float(f.read().strip())
            except:
                best_metric = float('-inf') if higher_better else float('inf')
    else:
        best_metric = float('-inf') if higher_better else float('inf')
    
    current_metric = metrics.get(metric_name, 0)
    
    # 判断是否是最佳模型
    is_best = (current_metric > best_metric) if higher_better else (current_metric < best_metric)
    
    if is_best:
        # 保存最佳模型
        best_model_path = save_dir / 'best_model.pth'
        save_model(model, optimizer, scheduler, epoch, metrics, best_model_path)
        
        # 更新最佳指标
        with open(best_metric_file, 'w') as f:
            f.write(str(current_metric))
        
        logging.info(f"发现更好的模型! {metric_name}: {current_metric:.4f}")
    
    # 保存最新模型
    latest_model_path = save_dir / 'latest_model.pth'
    save_model(model, optimizer, scheduler, epoch, metrics, latest_model_path)
    
    return is_best


class EarlyStopping:
    """
    早停机制
    """
    
    def __init__(self, patience=10, min_delta=0.001, metric_name='val_loss', 
                 higher_better=False):
        self.patience = patience
        self.min_delta = min_delta
        self.metric_name = metric_name
        self.higher_better = higher_better
        
        self.best_metric = float('-inf') if higher_better else float('inf')
        self.counter = 0
        self.early_stop = False
    
    def __call__(self, metrics):
        """
        检查是否应该早停
        
        Args:
            metrics: 当前的性能指标字典
            
        Returns:
            should_stop: 是否应该停止训练
        """
        current_metric = metrics.get(self.metric_name, 0)
        
        if self.higher_better:
            is_improvement = current_metric > self.best_metric + self.min_delta
        else:
            is_improvement = current_metric < self.best_metric - self.min_delta
        
        if is_improvement:
            self.best_metric = current_metric
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            self.early_stop = True
            logging.info(f"早停触发! 连续 {self.patience} 轮未改善")
        
        return self.early_stop 