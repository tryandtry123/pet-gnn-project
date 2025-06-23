"""
混淆矩阵分析模块
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from pathlib import Path

# 导入字体工具
import sys
sys.path.append('..')
try:
    from utils.plot_utils import setup_chinese_font, get_chinese_labels, save_figure_with_chinese_support
except ImportError:
    # 如果导入失败，使用简单的设置
    def setup_chinese_font():
        try:
            plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            return False
    
    def get_chinese_labels():
        return {
            'title_confusion': 'PET Low-Coupling Event Recognition - Confusion Matrix',
            'xlabel_confusion': 'Predicted Label',
            'ylabel_confusion': 'True Label',
            'class_names': ['Valid Event', 'Low-Coupling Event']
        }
    
    def save_figure_with_chinese_support(fig, filepath, dpi=300, bbox_inches='tight'):
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')

# 设置中文字体
setup_chinese_font()


def plot_confusion_matrix(y_true, y_pred, class_names=None, normalize=None, 
                         title=None, save_path=None, figsize=(8, 6)):
    """
    绘制混淆矩阵
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称列表
        normalize: 标准化方式 ('true', 'pred', 'all', None)
        title: 图标题
        save_path: 保存路径
        figsize: 图形大小
    """
    # 获取标签配置
    labels = get_chinese_labels()
    
    if class_names is None:
        class_names = labels['class_names']
    
    if title is None:
        title = labels['title_confusion']
    
    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 标准化
    if normalize == 'true':
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        fmt = '.2%'
    elif normalize == 'pred':
        cm = cm.astype('float') / cm.sum(axis=0)
        fmt = '.2%'
    elif normalize == 'all':
        cm = cm.astype('float') / cm.sum()
        fmt = '.2%'
    else:
        fmt = 'd'
    
    # 创建图形
    plt.figure(figsize=figsize)
    
    # 使用seaborn绘制热力图
    ax = sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                     xticklabels=class_names, yticklabels=class_names,
                     cbar_kws={'label': 'Count' if normalize is None else 'Proportion'})
    
    # 设置标题和标签
    plt.title(title, fontsize=14, fontweight='bold', pad=20)
    plt.xlabel(labels['xlabel_confusion'], fontsize=12)
    plt.ylabel(labels['ylabel_confusion'], fontsize=12)
    
    # 旋转标签以更好显示
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    # 添加统计信息
    accuracy = np.trace(cm) / np.sum(cm) if normalize is None else np.trace(cm) / len(class_names)
    plt.figtext(0.02, 0.02, f'Accuracy: {accuracy:.3f}', fontsize=10,
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.7))
    
    plt.tight_layout()
    
    if save_path:
        save_figure_with_chinese_support(plt.gcf(), save_path)
    
    return cm


def analyze_confusion_matrix(y_true, y_pred, class_names=None):
    """
    分析混淆矩阵，提供详细的错误分析
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        class_names: 类别名称列表
        
    Returns:
        dict: 分析结果
    """
    # 获取标签配置
    labels = get_chinese_labels()
    
    if class_names is None:
        class_names = labels['class_names']
    
    cm = confusion_matrix(y_true, y_pred)
    n_classes = len(class_names)
    
    print("=" * 60)
    print("🔍 混淆矩阵详细分析")
    print("=" * 60)
    
    # 总体统计
    total_samples = np.sum(cm)
    correct_predictions = np.trace(cm)
    accuracy = correct_predictions / total_samples
    
    print(f"\n📊 总体统计:")
    print(f"  总样本数: {total_samples}")
    print(f"  正确预测: {correct_predictions}")
    print(f"  准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # 各类别统计
    print(f"\n📋 各类别详细分析:")
    for i in range(n_classes):
        true_positives = cm[i, i]
        false_positives = np.sum(cm[:, i]) - true_positives
        false_negatives = np.sum(cm[i, :]) - true_positives
        true_negatives = total_samples - true_positives - false_positives - false_negatives
        
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"\n  {class_names[i]}:")
        print(f"    真正例 (TP): {true_positives}")
        print(f"    假正例 (FP): {false_positives}")
        print(f"    假负例 (FN): {false_negatives}")
        print(f"    真负例 (TN): {true_negatives}")
        print(f"    精确率: {precision:.4f}")
        print(f"    召回率: {recall:.4f}")
        print(f"    F1分数: {f1_score:.4f}")
    
    # 错误模式分析
    print(f"\n⚠️ 错误模式分析:")
    for i in range(n_classes):
        for j in range(n_classes):
            if i != j and cm[i, j] > 0:
                error_rate = cm[i, j] / np.sum(cm[i, :]) * 100
                print(f"  {class_names[i]} → {class_names[j]}: {cm[i, j]} 个样本 ({error_rate:.2f}%)")
    
    # 类别平衡分析
    print(f"\n⚖️ 类别平衡分析:")
    for i, class_name in enumerate(class_names):
        class_total = np.sum(cm[i, :])
        class_ratio = class_total / total_samples * 100
        print(f"  {class_name}: {class_total} 个样本 ({class_ratio:.2f}%)")
    
    print("=" * 60)
    
    # 返回分析结果
    analysis = {
        'confusion_matrix': cm,
        'accuracy': accuracy,
        'total_samples': total_samples,
        'correct_predictions': correct_predictions,
        'class_statistics': {}
    }
    
    for i in range(n_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = total_samples - tp - fp - fn
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        analysis['class_statistics'][class_names[i]] = {
            'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1_score': f1
        }
    
    return analysis


def plot_normalized_confusion_matrices(y_true, y_pred, class_names=None, save_dir=None, figsize=(15, 5)):
    """
    绘制多种标准化方式的混淆矩阵对比
    """
    # 获取标签配置
    labels = get_chinese_labels()
    
    if class_names is None:
        class_names = labels['class_names']
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    fig.suptitle('Confusion Matrix Analysis', fontsize=16, fontweight='bold')
    
    normalizations = [None, 'true', 'pred']
    titles = ['Raw Counts', 'Normalized by True', 'Normalized by Prediction']
    
    for i, (norm, title) in enumerate(zip(normalizations, titles)):
        cm = confusion_matrix(y_true, y_pred)
        
        if norm == 'true':
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            fmt = '.2%'
        elif norm == 'pred':
            cm = cm.astype('float') / cm.sum(axis=0)
            fmt = '.2%'
        else:
            fmt = 'd'
        
        plt.subplot(1, 3, i+1)
        sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title(title)
        plt.xlabel(labels['xlabel_confusion'])
        if i == 0:
            plt.ylabel(labels['ylabel_confusion'])
    
    plt.tight_layout()
    
    if save_dir:
        save_path = Path(save_dir) / 'confusion_matrix_comparison.png'
        save_figure_with_chinese_support(fig, save_path)
    
    return fig 