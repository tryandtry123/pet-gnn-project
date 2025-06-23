"""
创建测试用的PET数据
"""

import numpy as np
import h5py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

def create_test_pet_data():
    """创建测试用的PET数据"""
    print("🔬 创建测试PET数据...")
    
    # 设置随机种子
    np.random.seed(42)
    
    # 生成1000个事件
    num_samples = 1000
    
    # 模拟PET探测器事件数据
    # pos_i: 探测器i的3D位置 (x, y, z)
    pos_i_x = np.random.uniform(-100, 100, num_samples)
    pos_i_y = np.random.uniform(-100, 100, num_samples)  
    pos_i_z = np.random.uniform(-50, 50, num_samples)
    
    # pos_j: 探测器j的3D位置
    pos_j_x = np.random.uniform(-100, 100, num_samples)
    pos_j_y = np.random.uniform(-100, 100, num_samples)
    pos_j_z = np.random.uniform(-50, 50, num_samples)
    
    # 能量 (keV) - 正电子湮灭产生511keV光子
    E_i = np.random.normal(511, 15, num_samples)  # 探测器i检测到的能量
    E_j = np.random.normal(511, 15, num_samples)  # 探测器j检测到的能量
    
    # 时间戳 (ns)
    T_i = np.random.uniform(0, 1000, num_samples)
    T_j = T_i + np.random.normal(0, 0.5, num_samples)  # 几乎同时检测
    
    # 计算距离来生成标签
    distances = np.sqrt((pos_i_x - pos_j_x)**2 + 
                       (pos_i_y - pos_j_y)**2 + 
                       (pos_i_z - pos_j_z)**2)
    
    # 能量差
    energy_diff = np.abs(E_i - E_j)
    
    # 时间差
    time_diff = np.abs(T_i - T_j)
    
    # 生成标签：低耦合事件的判断规则
    # 1 = 低耦合事件, 0 = 有效事件
    labels = ((distances > 150) | (energy_diff > 50) | (time_diff > 2)).astype(int)
    
    print(f"📊 生成了 {num_samples} 个事件")
    print(f"📈 低耦合事件比例: {np.mean(labels):.2%}")
    
    # 创建DataFrame
    data = pd.DataFrame({
        'pos_i_x': pos_i_x,
        'pos_i_y': pos_i_y, 
        'pos_i_z': pos_i_z,
        'pos_j_x': pos_j_x,
        'pos_j_y': pos_j_y,
        'pos_j_z': pos_j_z,
        'E_i': E_i,
        'E_j': E_j,
        'T_i': T_i,
        'T_j': T_j,
        'distance': distances,
        'energy_diff': energy_diff,
        'time_diff': time_diff,
        'label': labels
    })
    
    # 数据分割
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42, stratify=data['label'])
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42, stratify=temp_data['label'])
    
    # 创建目录
    Path('data/processed').mkdir(parents=True, exist_ok=True)
    Path('data/raw').mkdir(parents=True, exist_ok=True)
    
    # 保存CSV格式 (最简单)
    train_data.to_csv('data/processed/train_data.csv', index=False)
    val_data.to_csv('data/processed/val_data.csv', index=False)
    test_data.to_csv('data/processed/test_data.csv', index=False)
    
    # 保存完整数据
    data.to_csv('data/raw/pet_events.csv', index=False)
    
    print("✅ 数据已保存:")
    print(f"  📁 训练集: {len(train_data)} 样本 -> data/processed/train_data.csv")
    print(f"  📁 验证集: {len(val_data)} 样本 -> data/processed/val_data.csv") 
    print(f"  📁 测试集: {len(test_data)} 样本 -> data/processed/test_data.csv")
    print(f"  📁 完整数据: data/raw/pet_events.csv")
    
    # 显示数据示例
    print("\n📋 数据样例:")
    print(data.head())
    
    return data

if __name__ == "__main__":
    create_test_pet_data() 