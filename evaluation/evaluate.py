"""
主评估脚本
"""

import argparse
import torch
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import yaml

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from training.train_simple import SimplePETNet, PETDataset
from evaluation.metrics import print_classification_report, plot_roc_curve, plot_precision_recall_curve, analyze_prediction_errors


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='PET-GNN模型评估')
    parser.add_argument('--model_path', type=str, default='experiments/best_model.pth',
                       help='模型检查点路径')
    parser.add_argument('--test_data', type=str, default='data/processed/test_data.csv',
                       help='测试数据路径')
    parser.add_argument('--output_dir', type=str, default='evaluation_results',
                       help='评估结果输出目录')
    parser.add_argument('--save_plots', action='store_true',
                       help='是否保存可视化图表')
    
    return parser.parse_args()


def load_model_for_evaluation(model_path, device='cpu'):
    """
    加载训练好的模型用于评估
    
    Args:
        model_path: 模型检查点路径
        device: 计算设备
    
    Returns:
        model: 加载的模型
        config: 模型配置
    """
    try:
        checkpoint = torch.load(model_path, map_location=device)
        
        # 获取配置信息
        config = checkpoint.get('config', {})
        
        # 创建模型
        model = SimplePETNet(
            input_dim=10,
            hidden_dims=config.get('model', {}).get('hidden_dims', [64, 32, 16]),
            num_classes=2,
            dropout=config.get('model', {}).get('dropout', 0.1)
        )
        
        # 加载模型状态
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(device)
        model.eval()
        
        print(f"✅ 模型加载成功: {model_path}")
        print(f"   训练轮数: {checkpoint.get('epoch', 'Unknown')}")
        print(f"   最佳指标: {checkpoint.get('metrics', {})}")
        
        return model, config
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        raise


def evaluate_model(model, test_loader, device='cpu'):
    """
    评估模型性能
    
    Args:
        model: 训练好的模型
        test_loader: 测试数据加载器
        device: 计算设备
    
    Returns:
        results: 评估结果字典
    """
    model.eval()
    all_preds = []
    all_targets = []
    all_probs = []
    
    print("🔄 正在进行模型评估...")
    
    with torch.no_grad():
        for batch_idx, (data, targets) in enumerate(test_loader):
            data, targets = data.to(device), targets.to(device)
            
            # 前向传播
            outputs = model(data)
            probs = torch.softmax(outputs, dim=1)
            preds = outputs.argmax(dim=1)
            
            # 收集结果
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
    
    # 转换为numpy数组
    y_true = np.array(all_targets)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    
    print(f"✅ 评估完成，共处理 {len(y_true)} 个样本")
    
    return {
        'y_true': y_true,
        'y_pred': y_pred, 
        'y_prob': y_prob
    }


def create_test_data_loader(test_data_path, batch_size=32):
    """创建测试数据加载器"""
    try:
        test_dataset = PETDataset(test_data_path)
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0
        )
        print(f"✅ 测试数据加载成功，共 {len(test_dataset)} 个样本")
        return test_loader
        
    except Exception as e:
        print(f"❌ 测试数据加载失败: {e}")
        # 如果测试数据不存在，使用验证数据代替
        print("尝试使用验证数据进行评估...")
        try:
            val_dataset = PETDataset('data/processed/val_data.csv')
            val_loader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0
            )
            print(f"✅ 使用验证数据评估，共 {len(val_dataset)} 个样本")
            return val_loader
        except Exception as e2:
            print(f"❌ 验证数据也加载失败: {e2}")
            raise


def main():
    # 解析参数
    args = parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("=" * 60)
    print("🚀 开始PET-GNN模型评估")
    print("=" * 60)
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    model, config = load_model_for_evaluation(args.model_path, device)
    
    # 创建测试数据加载器
    test_loader = create_test_data_loader(args.test_data, batch_size=32)
    
    # 模型评估
    results = evaluate_model(model, test_loader, device)
    
    # 提取结果
    y_true = results['y_true']
    y_pred = results['y_pred'] 
    y_prob = results['y_prob']
    
    # 打印详细评估报告
    class_names = ['有效事件', '低耦合事件']
    report_path = output_dir / 'classification_report.txt' if args.save_plots else None
    metrics = print_classification_report(y_true, y_pred, class_names, report_path)
    
    # 错误样本分析
    feature_names = ['pos_i_x', 'pos_i_y', 'pos_i_z', 'pos_j_x', 'pos_j_y', 'pos_j_z', 
                    'E_i', 'E_j', 'distance', 'energy_diff']
    analyze_prediction_errors(y_true, y_pred, max_errors=5)
    
    # 绘制可视化图表
    if args.save_plots:
        print(f"\n📊 生成可视化图表...")
        
        # ROC曲线
        roc_path = output_dir / 'roc_curve.png'
        plot_roc_curve(y_true, y_prob, save_path=roc_path)
        
        # PR曲线
        pr_path = output_dir / 'precision_recall_curve.png'
        plot_precision_recall_curve(y_true, y_prob, save_path=pr_path)
        
        print(f"📁 评估结果已保存到: {output_dir}")
    
    # 保存预测结果
    results_df = pd.DataFrame({
        'true_label': y_true,
        'predicted_label': y_pred,
        'prob_class_0': y_prob[:, 0],
        'prob_class_1': y_prob[:, 1],
        'correct': y_true == y_pred
    })
    
    results_path = output_dir / 'prediction_results.csv'
    results_df.to_csv(results_path, index=False)
    print(f"💾 预测结果已保存到: {results_path}")
    
    # 输出总结
    print("\n" + "=" * 60)
    print("📈 评估总结:")
    print(f"  准确率: {metrics['accuracy']:.4f}")
    print(f"  F1分数: {metrics['f1_weighted']:.4f}")
    if 'roc_auc' in metrics:
        print(f"  AUC:   {metrics['roc_auc']:.4f}")
    print("=" * 60)


if __name__ == '__main__':
    main() 