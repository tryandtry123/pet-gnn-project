"""
PET-GNN简化训练脚本 (避免复杂依赖)
使用简单神经网络进行训练演示
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
from sklearn.preprocessing import StandardScaler

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))


class SimplePETNet(nn.Module):
    """简化的PET神经网络模型"""
    
    def __init__(self, input_dim=10, hidden_dims=[64, 32, 16], num_classes=2, dropout=0.1):
        super(SimplePETNet, self).__init__()
        
        self.input_dim = input_dim
        self.num_classes = num_classes
        
        # 构建网络层
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
        
        # 输出层
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        return self.network(x)


class PETDataset(torch.utils.data.Dataset):
    """简化的PET数据集"""
    
    def __init__(self, csv_file, feature_cols=None):
        self.data = pd.read_csv(csv_file)
        
        if feature_cols is None:
            # 选择主要特征（排除标签）
            feature_cols = ['pos_i_x', 'pos_i_y', 'pos_i_z', 'pos_j_x', 'pos_j_y', 'pos_j_z', 
                          'E_i', 'E_j', 'distance', 'energy_diff']
        
        self.features = self.data[feature_cols].values.astype(np.float32)
        self.labels = self.data['label'].values.astype(np.int64)
        
        # 标准化特征
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.features)
        
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        return torch.tensor(self.features[idx]), torch.tensor(self.labels[idx])


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PET-GNN简化训练')
    parser.add_argument('--config', type=str, default='config/default.yaml',
                       help='配置文件路径')
    parser.add_argument('--resume', type=str, default=None,
                       help='恢复训练的检查点路径')
    parser.add_argument('--debug', action='store_true',
                       help='调试模式')
    
    return parser.parse_args()


def load_config(config_path):
    """加载配置文件"""
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件 {config_path} 不存在，使用默认配置")
        return create_default_config()


def create_default_config():
    """创建默认配置"""
    return {
        'training': {
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-4,
            'epochs': 50,
            'optimizer': 'Adam',
            'scheduler': 'StepLR',
            'step_size': 30,
            'gamma': 0.1,
            'early_stopping': {'patience': 10}
        },
        'model': {
            'hidden_dims': [64, 32, 16],
            'dropout': 0.1
        },
        'experiment': {
            'save_dir': 'experiments'
        }
    }


def setup_device():
    """设置计算设备"""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"使用GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        print("使用CPU")
    return device


def create_data_loaders(config):
    """创建数据加载器"""
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
        
        print(f"数据加载成功: 训练集{len(train_dataset)}, 验证集{len(val_dataset)}")
        return train_loader, val_loader
        
    except Exception as e:
        print(f"数据加载失败: {e}")
        print("请先运行 create_test_data.py 生成数据")
        raise


def create_model(config):
    """创建模型"""
    input_dim = 10  # 特征维度
    hidden_dims = config.get('model', {}).get('hidden_dims', [64, 32, 16])
    dropout = config.get('model', {}).get('dropout', 0.1)
    
    model = SimplePETNet(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=2,
        dropout=dropout
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型创建成功，参数数量: {total_params:,}")
    
    return model


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
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['training'].get('step_size', 30),
        gamma=config['training'].get('gamma', 0.1)
    )
    
    return optimizer, scheduler


def train_epoch(model, train_loader, optimizer, criterion, device):
    """训练一个epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    pbar = tqdm(train_loader, desc='训练')
    for batch_idx, (data, targets) in enumerate(pbar):
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


def validate_epoch(model, val_loader, criterion, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for data, targets in tqdm(val_loader, desc='验证'):
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


def save_checkpoint(model, optimizer, epoch, metrics, config, save_dir, is_best=False):
    """保存检查点"""
    save_dir = Path(save_dir)
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
        print(f"保存最佳模型到: {best_path}")


def main():
    # 解析参数
    args = parse_args()
    
    # 加载配置
    config = load_config(args.config)
    
    # 设置日志
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    
    print("=" * 60)
    print("🚀 开始PET-GNN简化训练")
    print("=" * 60)
    
    # 设置设备
    device = setup_device()
    
    # 创建数据加载器
    train_loader, val_loader = create_data_loaders(config)
    
    # 创建模型
    model = create_model(config)
    model = model.to(device)
    
    # 创建优化器和损失函数
    optimizer, scheduler = create_optimizer(model, config)
    criterion = nn.CrossEntropyLoss()
    criterion = criterion.to(device)
    
    # 训练设置
    epochs = config['training']['epochs']
    best_f1 = 0
    patience = config['training']['early_stopping']['patience']
    patience_counter = 0
    save_dir = config['experiment']['save_dir']
    
    print(f"训练设置:")
    print(f"  - 训练轮数: {epochs}")
    print(f"  - 批大小: {config['training']['batch_size']}")
    print(f"  - 学习率: {config['training']['learning_rate']}")
    print(f"  - 早停耐心: {patience}")
    
    # 恢复训练
    start_epoch = 0
    if args.resume:
        try:
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_f1 = checkpoint['metrics']['f1']
            print(f"从epoch {start_epoch}恢复训练")
        except Exception as e:
            print(f"检查点加载失败: {e}")
    
    # 训练循环
    print("\n🎯 开始训练循环")
    for epoch in range(start_epoch, epochs):
        print(f"\nEpoch {epoch+1}/{epochs}")
        print("-" * 40)
        
        # 训练
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss, val_metrics = validate_epoch(model, val_loader, criterion, device)
        
        # 更新学习率
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # 记录结果
        print(f"训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_metrics['accuracy']*100:.2f}%")
        print(f"      - Precision: {val_metrics['precision']:.4f}")
        print(f"      - Recall: {val_metrics['recall']:.4f}")
        print(f"      - F1: {val_metrics['f1']:.4f}")
        print(f"学习率: {current_lr:.6f}")
        
        # 保存检查点
        is_best = val_metrics['f1'] > best_f1
        if is_best:
            best_f1 = val_metrics['f1']
            patience_counter = 0
            print(f"🎉 新的最佳F1分数: {best_f1:.4f}")
        else:
            patience_counter += 1
        
        save_checkpoint(model, optimizer, epoch, val_metrics, config, save_dir, is_best)
        
        # 早停检查
        if patience_counter >= patience:
            print(f"早停触发，已连续{patience}轮无改善")
            break
    
    print("\n" + "=" * 60)
    print("🎉 训练完成!")
    print(f"最佳F1分数: {best_f1:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main() 