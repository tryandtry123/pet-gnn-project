"""
PET-GNN模型训练脚本
"""

import argparse
import yaml
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from models.gnn_model import PETGraphNet
from preprocessing.graph_dataset import PETGraphDataset, collate_graph_batch
from utils.logger import setup_logger


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PET-GNN模型训练')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logging.info(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logging.info("使用CPU")
    return device


def create_model(config):
    """创建模型"""
    try:
        model = PETGraphNet(config)
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logging.info(f"模型创建成功")
        logging.info(f"总参数数量: {total_params:,}")
        logging.info(f"可训练参数: {trainable_params:,}")
        
        return model
    except Exception as e:
        logging.error(f"模型创建失败: {e}")
        # 如果GNN模型创建失败，使用简单的MLP作为备选
        logging.info("使用简单MLP模型作为备选...")
        return create_simple_model(config)


def create_simple_model(config):
    """创建简单的MLP模型作为备选"""
    class SimpleMLP(nn.Module):
        def __init__(self, input_dim=10, hidden_dims=[64, 32], num_classes=2, dropout=0.1):
            super(SimpleMLP, self).__init__()
            layers = []
            prev_dim = input_dim
            
            for hidden_dim in hidden_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.BatchNorm1d(hidden_dim),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            
            layers.append(nn.Linear(prev_dim, num_classes))
            self.network = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.network(x)
    
    return SimpleMLP(input_dim=10, hidden_dims=[64, 32], num_classes=2)


def create_data_loaders(config):
    """创建数据加载器"""
    try:
        # 尝试使用图数据集
        train_dataset = PETGraphDataset(
            config, 
            data_file='data/processed/train_data.csv',
            split='train'
        )
        val_dataset = PETGraphDataset(
            config,
            data_file='data/processed/val_data.csv', 
            split='val'
        )
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            collate_fn=collate_graph_batch,
            num_workers=0  # Windows兼容性
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            collate_fn=collate_graph_batch,
            num_workers=0
        )
        
        logging.info(f"图数据加载成功: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}")
        return train_loader, val_loader, 'graph'
        
    except Exception as e:
        logging.warning(f"图数据加载失败: {e}")
        logging.info("使用简单表格数据...")
        return create_simple_data_loaders(config)


def create_simple_data_loaders(config):
    """创建简单的表格数据加载器"""
    from simple_train import PETDataset
    
    try:
        train_dataset = PETDataset('data/processed/train_data.csv')
        val_dataset = PETDataset('data/processed/val_data.csv')
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=True,
            num_workers=0
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=config['training']['batch_size'],
            shuffle=False,
            num_workers=0
        )
        
        logging.info(f"表格数据加载成功: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}")
        return train_loader, val_loader, 'tabular'
        
    except Exception as e:
        logging.error(f"数据加载失败: {e}")
        raise


def create_optimizer(model, config):
    """创建优化器和调度器"""
    optimizer_name = config['training']['optimizer']
    lr = config['training']['learning_rate']
    weight_decay = config['training']['weight_decay']
    
    if optimizer_name == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'SGD':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    # 学习率调度器
    scheduler_name = config['training'].get('scheduler', 'StepLR')
    if scheduler_name == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, 
            step_size=config['training'].get('step_size', 30),
            gamma=config['training'].get('gamma', 0.1)
        )
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    
    return optimizer, scheduler


def create_loss_function(config):
    """创建损失函数"""
    loss_name = config['training'].get('loss_function', 'CrossEntropyLoss')
    
    if loss_name == 'CrossEntropyLoss':
        # 处理类别不平衡
        class_weights = config['training'].get('class_weights', None)
        if class_weights:
            weight = torch.FloatTensor(class_weights)
            criterion = nn.CrossEntropyLoss(weight=weight)
        else:
            criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    return criterion


def train_epoch(model, train_loader, optimizer, criterion, device, data_type='graph'):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='训练')
    for batch_idx, batch in enumerate(pbar):
        
        if data_type == 'graph':
            # 图数据
            x = batch['x'].to(device)
            edge_index = batch['edge_index'].to(device)
            edge_attr = batch.get('edge_attr', None)
            if edge_attr is not None:
                edge_attr = edge_attr.to(device)
            batch_tensor = batch['batch'].to(device)
            targets = batch['y'].to(device)
            
            optimizer.zero_grad()
            try:
                outputs = model(x, edge_index, batch_tensor, edge_attr)
            except:
                # 如果图模型失败，使用事件特征
                event_features = batch['event_features'].to(device)
                outputs = model(event_features)
                
        else:
            # 表格数据
            data, targets = batch
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
        
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        pred = outputs.argmax(dim=1)
        correct += pred.eq(targets).sum().item()
        total += targets.size(0)
        
        # 更新进度条
        accuracy = 100. * correct / total
        pbar.set_postfix({
            'Loss': f'{loss.item():.4f}',
            'Acc': f'{accuracy:.2f}%'
        })
    
    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total
    
    return avg_loss, accuracy


def validate_epoch(model, val_loader, criterion, device, data_type='graph'):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc='验证'):
            
            if data_type == 'graph':
                # 图数据
                x = batch['x'].to(device)
                edge_index = batch['edge_index'].to(device)
                edge_attr = batch.get('edge_attr', None)
                if edge_attr is not None:
                    edge_attr = edge_attr.to(device)
                batch_tensor = batch['batch'].to(device)
                targets = batch['y'].to(device)
                
                try:
                    outputs = model(x, edge_index, batch_tensor, edge_attr)
                except:
                    # 如果图模型失败，使用事件特征
                    event_features = batch['event_features'].to(device)
                    outputs = model(event_features)
                    
            else:
                # 表格数据
                data, targets = batch
                data, targets = data.to(device), targets.to(device)
                outputs = model(data)
            
            loss = criterion(outputs, targets)
            total_loss += loss.item()
            
            pred = outputs.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(val_loader)
    
    # 计算指标
    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    
    return avg_loss, {
        'accuracy': accuracy,
        'precision': precision, 
        'recall': recall,
        'f1': f1
    }


def save_checkpoint(model, optimizer, epoch, metrics, config, is_best=False):
    """保存检查点"""
    save_dir = Path(config['experiment']['save_dir'])
    save_dir.mkdir(exist_ok=True)
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics,
        'config': config
    }
    
    # 保存最新检查点
    checkpoint_path = save_dir / 'latest_checkpoint.pth'
    torch.save(checkpoint, checkpoint_path)
    
    # 保存最佳模型
    if is_best:
        best_path = save_dir / 'best_model.pth'
        torch.save(checkpoint, best_path)
        logging.info(f"保存最佳模型到: {best_path}")


def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    logger = setup_logger(
        level=logging.DEBUG if args.debug else logging.INFO,
        log_file=f"{config['experiment']['save_dir']}/training.log"
    )
    
    logging.info("=" * 60)
    logging.info("🚀 开始PET-GNN模型训练")
    logging.info("=" * 60)
    
    # 设置设备
    device = setup_device()
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 创建数据加载器
    train_loader, val_loader, data_type = create_data_loaders(config)
    
    # 创建优化器和损失函数
    optimizer, scheduler = create_optimizer(model, config)
    criterion = create_loss_function(config)
    criterion = criterion.to(device)
    
    # 训练设置
    epochs = config['training']['epochs']
    best_f1 = 0
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    
    logging.info(f"训练设置:")
    logging.info(f"  - 数据类型: {data_type}")
    logging.info(f"  - 训练轮数: {epochs}")
    logging.info(f"  - 批大小: {config['training']['batch_size']}")
    logging.info(f"  - 学习率: {config['training']['learning_rate']}")
    logging.info(f"  - 早停耐心: {patience}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['metrics']['f1']
            logging.info(f"从epoch {start_epoch}恢复训练")
        except Exception as e:
            logging.warning(f"检查点加载失败: {e}")
    
    # 训练循环
    logging.info("\n🎯 开始训练循环")
    for epoch in range(start_epoch, epochs):
        logging.info(f"\nEpoch {epoch+1}/{epochs}")
        logging.info("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, data_type
        )
        
        # 验证
        val_loss, val_metrics = validate_epoch(
            model, val_loader, criterion, device, data_type
        )
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录结果
        logging.info(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        logging.info(f"验证 - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
        logging.info(f"      - Precision: {val_metrics['precision']:.4f}")
        logging.info(f"      - Recall: {val_metrics['recall']:.4f}")
        logging.info(f"      - F1: {val_metrics['f1']:.4f}")
        logging.info(f"学习率: {current_lr:.6f}")
        
        # 保存检查点
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            logging.info(f"🎉 新的最佳F1分数: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, epoch, val_metrics, config, is_best)
        
        # 早停检查
        if patience_counter >= patience:
            logging.info(f"早停触发，已连续{patience}轮无改善")
            break
    
    logging.info("\n" + "=" * 60)
    logging.info("🎉 训练完成!")
    logging.info(f"最佳F1分数: {best_f1:.4f}")
    logging.info("=" * 60)


if __name__ == '__main__':
    main() 