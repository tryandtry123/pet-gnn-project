"""
评估指标计算模块
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, roc_curve, average_precision_score
)
import matplotlib.pyplot as plt
import seaborn as sns
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
            'title_roc': 'PET Low-Coupling Event Recognition - ROC Curve', 
            'title_pr': 'PET Low-Coupling Event Recognition - PR Curve',
            'xlabel_confusion': 'Predicted Label',
            'ylabel_confusion': 'True Label',
            'xlabel_roc': 'False Positive Rate (FPR)',
            'ylabel_roc': 'True Positive Rate (TPR)',
            'xlabel_pr': 'Recall',
            'ylabel_pr': 'Precision',
            'class_names': ['Valid Event', 'Low-Coupling Event'],
            'legend_roc': 'ROC Curve (AUC = {:.3f})',
            'legend_pr': 'PR Curve (AP = {:.3f})',
            'text_accuracy': 'Accuracy: {:.3f}',
            'text_precision': 'Precision: {:.3f}',
            'text_recall': 'Recall: {:.3f}',
            'text_f1': 'F1-Score: {:.3f}'
        }
    
    def save_figure_with_chinese_support(fig, filepath, dpi=300, bbox_inches='tight'):
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, facecolor='white')

# 设置中文字体
setup_chinese_font()

def calculate_metrics(y_true, y_pred, y_prob=None, class_names=None):
    """
    计算完整的分类指标
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签  
        y_prob: 预测概率 (可选)
        class_names: 类别名称 (可选)
    
    Returns:
        dict: 包含所有指标的字典
    """
    # 获取标签配置
    labels = get_chinese_labels()
    if class_names is None:
        class_names = labels['class_names']
    
    # 基础分类指标
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision_macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'precision_weighted': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall_weighted': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_weighted': f1_score(y_true, y_pred, average='weighted', zero_division=0),
    }
    
    # 各类别的详细指标
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    for i, class_name in enumerate(class_names):
        metrics[f'precision_{class_name}'] = precision_per_class[i] if i < len(precision_per_class) else 0
        metrics[f'recall_{class_name}'] = recall_per_class[i] if i < len(recall_per_class) else 0
        metrics[f'f1_{class_name}'] = f1_per_class[i] if i < len(f1_per_class) else 0
    
    # 如果有概率预测，计算AUC等指标
    if y_prob is not None:
        try:
            if len(np.unique(y_true)) == 2:  # 二分类
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
                metrics['ap_score'] = average_precision_score(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)
            else:  # 多分类
                metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')
                metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo', average='macro')
        except Exception as e:
            print(f"AUC计算失败: {e}")
    
    # 混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    metrics['confusion_matrix'] = cm
    
    return metrics


def print_classification_report(y_true, y_pred, class_names=None, save_path=None):
    """
    打印详细的分类报告
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        class_names: 类别名称
        save_path: 保存路径 (可选)
    """
    if class_names is None:
        class_names = ['有效事件', '低耦合事件']
    
    print("=" * 60)
    print("📊 PET-GNN模型评估报告")
    print("=" * 60)
    
    # 计算指标
    metrics = calculate_metrics(y_true, y_pred, class_names=class_names)
    
    # 基础指标
    print(f"\n🎯 总体性能:")
    print(f"  准确率 (Accuracy):     {metrics['accuracy']:.4f}")
    print(f"  精确率 (Precision):    {metrics['precision_weighted']:.4f}")
    print(f"  召回率 (Recall):       {metrics['recall_weighted']:.4f}")
    print(f"  F1分数 (F1-Score):     {metrics['f1_weighted']:.4f}")
    
    # 各类别详细指标
    print(f"\n📋 各类别详细指标:")
    for class_name in class_names:
        print(f"\n  {class_name}:")
        print(f"    精确率: {metrics.get(f'precision_{class_name}', 0):.4f}")
        print(f"    召回率: {metrics.get(f'recall_{class_name}', 0):.4f}")
        print(f"    F1分数: {metrics.get(f'f1_{class_name}', 0):.4f}")
    
    # 混淆矩阵
    cm = metrics['confusion_matrix']
    print(f"\n🔍 混淆矩阵:")
    print("     预测")
    print("真实", end="  ")
    for name in class_names:
        print(f"{name:>8}", end="")
    print()
    
    for i, true_name in enumerate(class_names):
        print(f"{true_name:>4}", end="  ")
        for j in range(len(class_names)):
            print(f"{cm[i,j]:>8}", end="")
        print()
    
    # 错误分析
    print(f"\n⚠️ 错误分析:")
    total_samples = len(y_true)
    correct_samples = np.sum(y_true == y_pred)
    wrong_samples = total_samples - correct_samples
    
    print(f"  总样本数: {total_samples}")
    print(f"  正确预测: {correct_samples} ({correct_samples/total_samples*100:.2f}%)")
    print(f"  错误预测: {wrong_samples} ({wrong_samples/total_samples*100:.2f}%)")
    
    # 类别分布
    print(f"\n📈 真实标签分布:")
    unique, counts = np.unique(y_true, return_counts=True)
    for i, (label, count) in enumerate(zip(unique, counts)):
        class_name = class_names[label] if label < len(class_names) else f"类别{label}"
        print(f"  {class_name}: {count} ({count/total_samples*100:.2f}%)")
    
    # 保存报告
    if save_path:
        try:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(classification_report(y_true, y_pred, target_names=class_names))
            print(f"\n💾 详细报告已保存到: {save_path}")
        except Exception as e:
            print(f"保存报告失败: {e}")
    
    print("=" * 60)
    
    return metrics


def plot_roc_curve(y_true, y_prob, save_path=None, figsize=(8, 6)):
    """绘制ROC曲线"""
    try:
        # 获取标签配置
        labels = get_chinese_labels()
        
        plt.figure(figsize=figsize)
        
        if y_prob.ndim > 1:
            # 多分类情况，只处理第二列（低耦合事件概率）
            prob_positive = y_prob[:, 1]
        else:
            prob_positive = y_prob
            
        # 计算ROC曲线
        fpr, tpr, _ = roc_curve(y_true, prob_positive)
        auc = roc_auc_score(y_true, prob_positive)
        
        # 绘制曲线
        plt.plot(fpr, tpr, linewidth=2, label=labels['legend_roc'].format(auc))
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        # 设置标签和标题
        plt.xlabel(labels['xlabel_roc'], fontsize=12)
        plt.ylabel(labels['ylabel_roc'], fontsize=12)
        plt.title(labels['title_roc'], fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加性能文本
        plt.text(0.6, 0.2, f"AUC = {auc:.3f}", fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            save_figure_with_chinese_support(plt.gcf(), save_path)
        
        return auc
        
    except Exception as e:
        print(f"ROC曲线绘制失败: {e}")
        return None


def plot_precision_recall_curve(y_true, y_prob, save_path=None, figsize=(8, 6)):
    """绘制Precision-Recall曲线"""
    try:
        # 获取标签配置
        labels = get_chinese_labels()
        
        plt.figure(figsize=figsize)
        
        if y_prob.ndim > 1:
            prob_positive = y_prob[:, 1]
        else:
            prob_positive = y_prob
            
        # 计算PR曲线
        precision, recall, _ = precision_recall_curve(y_true, prob_positive)
        ap = average_precision_score(y_true, prob_positive)
        
        # 绘制曲线
        plt.plot(recall, precision, linewidth=2, label=labels['legend_pr'].format(ap))
        
        # 基准线（随机分类器）
        no_skill = len(y_true[y_true==1]) / len(y_true)
        plt.axhline(y=no_skill, color='k', linestyle='--', alpha=0.5, 
                   label=f'Random Classifier (AP = {no_skill:.3f})')
        
        # 设置标签和标题
        plt.xlabel(labels['xlabel_pr'], fontsize=12)
        plt.ylabel(labels['ylabel_pr'], fontsize=12)
        plt.title(labels['title_pr'], fontsize=14, fontweight='bold')
        plt.legend(fontsize=10)
        plt.grid(True, alpha=0.3)
        
        # 添加性能文本
        plt.text(0.2, 0.8, f"AP = {ap:.3f}", fontsize=12,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen", alpha=0.7))
        
        plt.tight_layout()
        
        if save_path:
            save_figure_with_chinese_support(plt.gcf(), save_path)
        
        return ap
        
    except Exception as e:
        print(f"PR曲线绘制失败: {e}")
        return None


def analyze_prediction_errors(y_true, y_pred, X=None, feature_names=None, max_errors=10):
    """
    分析预测错误的样本
    
    Args:
        y_true: 真实标签
        y_pred: 预测标签
        X: 特征矩阵 (可选)
        feature_names: 特征名称 (可选)
        max_errors: 显示的最大错误数量
    """
    print("\n🔍 错误样本分析:")
    print("-" * 40)
    
    # 找出错误预测的样本
    error_mask = y_true != y_pred
    error_indices = np.where(error_mask)[0]
    
    if len(error_indices) == 0:
        print("🎉 没有预测错误的样本!")
        return
    
    print(f"总错误数量: {len(error_indices)}")
    
    # 显示前几个错误样本
    num_show = min(max_errors, len(error_indices))
    print(f"显示前 {num_show} 个错误样本:")
    
    for i, idx in enumerate(error_indices[:num_show]):
        print(f"\n错误样本 {i+1} (索引 {idx}):")
        print(f"  真实标签: {y_true[idx]}")
        print(f"  预测标签: {y_pred[idx]}")
        
        if X is not None and feature_names is not None:
            print(f"  特征值:")
            for j, feature_name in enumerate(feature_names):
                if j < X.shape[1]:
                    print(f"    {feature_name}: {X[idx, j]:.4f}")
    
    # 错误类型统计
    print(f"\n📊 错误类型统计:")
    for true_label in np.unique(y_true):
        for pred_label in np.unique(y_pred):
            if true_label != pred_label:
                count = np.sum((y_true == true_label) & (y_pred == pred_label))
                if count > 0:
                    print(f"  真实:{true_label} -> 预测:{pred_label}: {count} 个样本")


def plot_metrics_comparison(metrics_dict, save_path=None, figsize=(12, 8)):
    """绘制多个模型的指标对比图"""
    try:
        # 获取标签配置
        labels = get_chinese_labels()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
        
        # 提取指标数据
        models = list(metrics_dict.keys())
        accuracy = [metrics_dict[model]['accuracy'] for model in models]
        precision = [metrics_dict[model]['precision_weighted'] for model in models]
        recall = [metrics_dict[model]['recall_weighted'] for model in models]
        f1 = [metrics_dict[model]['f1_weighted'] for model in models]
        
        # 绘制各个指标
        axes[0, 0].bar(models, accuracy, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('Accuracy')
        axes[0, 0].set_ylim(0, 1)
        
        axes[0, 1].bar(models, precision, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('Precision')
        axes[0, 1].set_ylim(0, 1)
        
        axes[1, 0].bar(models, recall, alpha=0.7, color='salmon')
        axes[1, 0].set_title('Recall')
        axes[1, 0].set_ylim(0, 1)
        
        axes[1, 1].bar(models, f1, alpha=0.7, color='gold')
        axes[1, 1].set_title('F1-Score')
        axes[1, 1].set_ylim(0, 1)
        
        # 添加数值标签
        for ax in axes.flat:
            for i, v in enumerate(ax.containers[0]):
                height = v.get_height()
                ax.text(v.get_x() + v.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            save_figure_with_chinese_support(fig, save_path)
            
    except Exception as e:
        print(f"指标对比图绘制失败: {e}") 