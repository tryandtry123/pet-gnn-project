"""
模型模块初始化
"""

# 优先导入简化模型（无依赖问题）
try:
    from .simple_model import PETGNN
    __all__ = ['PETGNN']
    print("✅ 简化PETGNN模型加载成功")
except ImportError as e:
    print(f"❌ 无法导入简化PETGNN模型: {e}")
    __all__ = []

# 注释掉有问题的完整模型导入
# 可选：尝试导入完整的图神经网络模型（如果依赖可用）
# try:
#     # 检查是否可以安全导入完整模型
#     import torch_geometric
#     from torch_scatter import scatter_add
#     # 如果都可用，导入完整模型
#     from .gnn_model import PETGraphNet
#     __all__.append('PETGraphNet')
#     print("✅ 完整图神经网络模型也可用")
# except ImportError:
#     print("⚠️ 图神经网络依赖不可用，只使用简化模型") 