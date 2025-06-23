"""
图形绘制工具模块
解决matplotlib中文字体显示问题
"""

import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import warnings

def setup_chinese_font():
    """
    设置matplotlib支持中文字体显示
    自动检测系统并配置合适的中文字体
    """
    system = platform.system()
    
    # 抑制字体警告
    warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
    
    try:
        if system == "Windows":
            # Windows系统常见中文字体
            fonts = ['SimHei', 'Microsoft YaHei', 'SimSun', 'KaiTi']
            for font in fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    # 测试字体是否可用
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=12)
                    plt.close(fig)
                    print(f"✅ 中文字体设置成功: {font}")
                    return True
                except Exception:
                    continue
                    
        elif system == "Darwin":  # macOS
            fonts = ['Helvetica', 'Arial Unicode MS', 'STHeiti', 'PingFang SC']
            for font in fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=12)
                    plt.close(fig)
                    print(f"✅ 中文字体设置成功: {font}")
                    return True
                except Exception:
                    continue
                    
        elif system == "Linux":
            fonts = ['DejaVu Sans', 'WenQuanYi Micro Hei', 'Droid Sans Fallback']
            for font in fonts:
                try:
                    plt.rcParams['font.sans-serif'] = [font]
                    plt.rcParams['axes.unicode_minus'] = False
                    fig, ax = plt.subplots(figsize=(1, 1))
                    ax.text(0.5, 0.5, '测试', fontsize=12)
                    plt.close(fig)
                    print(f"✅ 中文字体设置成功: {font}")
                    return True
                except Exception:
                    continue
        
        # 如果以上字体都不可用，使用fallback方案
        print("⚠️ 未找到合适的中文字体，使用英文替代")
        return setup_fallback_font()
        
    except Exception as e:
        print(f"❌ 字体设置失败: {e}")
        return setup_fallback_font()

def setup_fallback_font():
    """
    字体设置失败时的fallback方案
    使用英文替代中文显示
    """
    try:
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        print("📝 使用英文字体作为fallback")
        return False
    except Exception:
        print("❌ 连基础字体都设置失败")
        return False

def get_chinese_labels():
    """
    获取中文标签映射
    如果中文字体不可用，返回英文标签
    """
    chinese_available = setup_chinese_font()
    
    if chinese_available:
        return {
            'title_confusion': 'PET低耦合事件识别 - 混淆矩阵',
            'title_roc': 'PET低耦合事件识别 - ROC曲线',
            'title_pr': 'PET低耦合事件识别 - PR曲线',
            'xlabel_confusion': '预测标签',
            'ylabel_confusion': '真实标签',
            'xlabel_roc': '假正率 (FPR)',
            'ylabel_roc': '真正率 (TPR)',
            'xlabel_pr': '召回率 (Recall)',
            'ylabel_pr': '精确率 (Precision)',
            'class_names': ['有效事件', '低耦合事件'],
            'legend_roc': 'ROC曲线 (AUC = {:.3f})',
            'legend_pr': 'PR曲线 (AP = {:.3f})',
            'text_accuracy': '准确率: {:.3f}',
            'text_precision': '精确率: {:.3f}',
            'text_recall': '召回率: {:.3f}',
            'text_f1': 'F1分数: {:.3f}'
        }
    else:
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

def create_figure_with_chinese_support(figsize=(10, 8)):
    """
    创建支持中文显示的图形
    
    Args:
        figsize: 图形大小
        
    Returns:
        fig, ax: matplotlib图形对象
    """
    setup_chinese_font()
    
    fig, ax = plt.subplots(figsize=figsize)
    return fig, ax

def save_figure_with_chinese_support(fig, filepath, dpi=300, bbox_inches='tight'):
    """
    保存支持中文的图形
    
    Args:
        fig: matplotlib图形对象
        filepath: 保存路径
        dpi: 分辨率
        bbox_inches: 边界设置
    """
    try:
        fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                   facecolor='white', edgecolor='none')
        print(f"✅ 图形已保存: {filepath}")
    except Exception as e:
        print(f"❌ 图形保存失败: {e}")
        # 尝试保存为PNG格式
        try:
            png_path = str(filepath).replace('.jpg', '.png').replace('.jpeg', '.png')
            fig.savefig(png_path, dpi=dpi, bbox_inches=bbox_inches,
                       facecolor='white', edgecolor='none')
            print(f"✅ 图形已保存为PNG: {png_path}")
        except Exception as e2:
            print(f"❌ PNG保存也失败: {e2}")

# 全局设置，导入时自动配置
print("🔧 配置matplotlib中文字体...")
chinese_supported = setup_chinese_font()
if chinese_supported:
    print("✅ 中文字体配置完成")
else:
    print("⚠️ 中文字体不可用，将使用英文替代") 