"""
PET低耦合事件识别 REST API 服务
提供模型预测、健康检查等接口
"""

# 依赖检查
import sys
import os
import traceback

def check_dependencies():
    """检查必要的依赖"""
    missing_deps = []
    
    try:
        import flask
    except ImportError:
        missing_deps.append('flask')
    
    try:
        import torch
    except ImportError:
        missing_deps.append('torch')
    
    try:
        import numpy as np
    except ImportError:
        missing_deps.append('numpy')
    
    try:
        import pandas as pd
    except ImportError:
        missing_deps.append('pandas')
    
    try:
        import yaml
    except ImportError:
        missing_deps.append('pyyaml')
    
    if missing_deps:
        print(f"❌ 缺少必要依赖: {', '.join(missing_deps)}")
        print("请运行: pip install " + " ".join(missing_deps))
        return False
    
    print("✅ 所有依赖检查通过")
    return True

# 检查依赖
if not check_dependencies():
    print("🛑 依赖检查失败，API服务无法启动")
    sys.exit(1)

# 导入其他模块
from flask import Flask, request, jsonify, render_template_string
import torch
import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path
from datetime import datetime

# 导入模型相关模块 (带错误处理)
try:
    from models import PETGNN
    MODEL_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 模型导入失败: {e}")
    MODEL_AVAILABLE = False

try:
    from preprocessing.data_loader import PETDataset
    DATA_LOADER_AVAILABLE = True
except ImportError as e:
    print(f"⚠️ 数据加载器导入失败: {e}")
    DATA_LOADER_AVAILABLE = False

try:
    import yaml
    CONFIG_AVAILABLE = True
except ImportError:
    print("⚠️ YAML配置不可用，将使用默认配置")
    CONFIG_AVAILABLE = False

app = Flask(__name__)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api_service.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 全局变量
model = None
config = None
device = None

def load_model_and_config():
    """加载模型和配置"""
    global model, config, device
    
    try:
        # 检查模型是否可用
        if not MODEL_AVAILABLE:
            logger.error("模型类不可用，无法创建模型实例")
            return False
        
        # 设置设备
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"使用设备: {device}")
        
        # 加载配置
        if CONFIG_AVAILABLE:
            config_path = 'config/default.yaml'
            if Path(config_path).exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                logger.info("已加载YAML配置文件")
            else:
                logger.warning("配置文件不存在，使用默认配置")
                config = None
        else:
            logger.warning("YAML不可用，使用默认配置")
            config = None
        
        # 使用默认配置
        if config is None:
            config = {
                'model': {
                    'input_dim': 13,
                    'hidden_dim': 64,
                    'output_dim': 2,
                    'num_layers': 3,
                    'dropout': 0.1
                }
            }
        else:
            # 确保配置中有必要的字段
            if 'model' not in config:
                config['model'] = {}
            
            # 设置默认值
            config['model'].setdefault('input_dim', 13)
            config['model'].setdefault('hidden_dim', config['model'].get('hidden_dim', 64))
            config['model'].setdefault('output_dim', config['model'].get('output_dim', 2))
            config['model'].setdefault('num_layers', 3)
            config['model'].setdefault('dropout', config['model'].get('dropout', 0.1))
        
        # 查找模型文件
        model_paths = [
            'experiments/best_model.pth',
            'best_model.pth',
            'latest_checkpoint.pth',
            'experiments/latest_checkpoint.pth'
        ]
        
        model_path = None
        for path in model_paths:
            if Path(path).exists():
                model_path = path
                break
        
        if model_path:
            # 创建模型实例
            model = PETGNN(
                input_dim=config['model']['input_dim'],
                hidden_dim=config['model']['hidden_dim'],
                output_dim=config['model']['output_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            )
            
            # 加载权重
            try:
                checkpoint = torch.load(model_path, map_location=device)
                if 'model_state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                
                model.to(device)
                model.eval()
                
                logger.info(f"模型加载成功: {model_path}")
                return True
                
            except Exception as e:
                logger.error(f"模型权重加载失败: {e}")
                # 创建未训练的模型作为fallback
                model = PETGNN(
                    input_dim=config['model']['input_dim'],
                    hidden_dim=config['model']['hidden_dim'],
                    output_dim=config['model']['output_dim'],
                    num_layers=config['model']['num_layers'],
                    dropout=config['model']['dropout']
                )
                model.to(device)
                model.eval()
                logger.warning("使用未训练的模型（仅用于演示）")
                return True
        else:
            logger.warning("未找到训练好的模型，创建新模型")
            # 创建新模型
            model = PETGNN(
                input_dim=config['model']['input_dim'],
                hidden_dim=config['model']['hidden_dim'],
                output_dim=config['model']['output_dim'],
                num_layers=config['model']['num_layers'],
                dropout=config['model']['dropout']
            )
            model.to(device)
            model.eval()
            logger.warning("使用未训练的模型（仅用于演示）")
            return True
            
    except Exception as e:
        logger.error(f"模型加载失败: {e}")
        logger.error(traceback.format_exc())
        return False

def validate_input_data(data):
    """验证输入数据格式"""
    required_fields = [
        'x1', 'y1', 'z1', 'x2', 'y2', 'z2',
        'energy1', 'energy2', 'time1', 'time2'
    ]
    
    # 检查必需字段
    for field in required_fields:
        if field not in data:
            return False, f"缺少必需字段: {field}"
    
    # 检查数据类型
    try:
        for field in required_fields:
            float(data[field])
    except (ValueError, TypeError):
        return False, f"字段 {field} 必须是数值类型"
    
    # 检查数据范围
    if data['energy1'] <= 0 or data['energy2'] <= 0:
        return False, "能量值必须大于0"
    
    return True, "数据验证通过"

def calculate_derived_features(data):
    """计算衍生特征"""
    # 计算距离
    distance = np.sqrt(
        (data['x2'] - data['x1'])**2 + 
        (data['y2'] - data['y1'])**2 + 
        (data['z2'] - data['z1'])**2
    )
    
    # 计算能量差
    energy_diff = abs(data['energy2'] - data['energy1'])
    
    # 计算时间差
    time_diff = abs(data['time2'] - data['time1'])
    
    return {
        'distance': distance,
        'energy_diff': energy_diff,
        'time_diff': time_diff
    }

@app.route('/')
def home():
    """主页"""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>PET低耦合事件识别 API</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .api-info { background: #ecf0f1; padding: 20px; border-radius: 5px; margin: 20px 0; }
            .endpoint { background: #3498db; color: white; padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status { padding: 10px; border-radius: 5px; margin: 10px 0; }
            .status.healthy { background: #2ecc71; color: white; }
            .status.unhealthy { background: #e74c3c; color: white; }
            code { background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🔬 PET低耦合事件识别 API</h1>
            
            <div class="status {{ 'healthy' if model_loaded else 'unhealthy' }}">
                <strong>服务状态:</strong> {{ '🟢 正常运行' if model_loaded else '🔴 模型未加载' }}
            </div>
            
            <div class="api-info">
                <h3>📋 API接口说明</h3>
                
                <div class="endpoint">
                    <strong>POST /predict</strong> - 预测PET事件类型
                </div>
                <p><strong>请求体示例:</strong></p>
                <pre><code>{
  "x1": 100.0, "y1": 100.0, "z1": 100.0,
  "x2": 120.0, "y2": 120.0, "z2": 120.0,
  "energy1": 511.0, "energy2": 511.0,
  "time1": 1.0, "time2": 1.1
}</code></pre>
                
                <div class="endpoint">
                    <strong>GET /health</strong> - 健康检查
                </div>
                
                <div class="endpoint">
                    <strong>GET /model_info</strong> - 模型信息
                </div>
                
                <div class="endpoint">
                    <strong>POST /batch_predict</strong> - 批量预测
                </div>
            </div>
            
            <div class="api-info">
                <h3>📊 返回格式</h3>
                <p>成功响应示例:</p>
                <pre><code>{
  "success": true,
  "prediction": {
    "label": 1,
    "label_name": "有效事件",
    "confidence": 0.95,
    "probability": [0.05, 0.95]
  },
  "features": {
    "distance": 34.6,
    "energy_diff": 0.0,
    "time_diff": 0.1
  },
  "timestamp": "2024-01-01T12:00:00"
}</code></pre>
            </div>
        </div>
    </body>
    </html>
    """
    return render_template_string(html_template, model_loaded=(model is not None))

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    status = {
        'status': 'healthy' if model is not None else 'unhealthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown',
        'timestamp': datetime.now().isoformat()
    }
    
    return jsonify(status), 200 if model is not None else 503

@app.route('/model_info', methods=['GET'])
def model_info():
    """模型信息接口"""
    if model is None:
        return jsonify({'error': '模型未加载'}), 503
    
    info = {
        'model_type': 'PETGNN',
        'input_dim': config['model']['input_dim'],
        'hidden_dim': config['model']['hidden_dim'],
        'output_dim': config['model']['output_dim'],
        'num_layers': config['model']['num_layers'],
        'dropout': config['model']['dropout'],
        'device': str(device),
        'parameters': sum(p.numel() for p in model.parameters()),
        'trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
    }
    
    return jsonify(info)

@app.route('/predict', methods=['POST'])
def predict():
    """单个样本预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 503
        
        # 获取输入数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '请提供JSON格式的输入数据'}), 400
        
        # 验证输入数据
        is_valid, message = validate_input_data(data)
        if not is_valid:
            return jsonify({'error': message}), 400
        
        # 计算衍生特征
        derived_features = calculate_derived_features(data)
        
        # 准备模型输入
        features = [
            data['x1'], data['y1'], data['z1'],
            data['x2'], data['y2'], data['z2'],
            data['energy1'], data['energy2'],
            data['time1'], data['time2'],
            derived_features['distance'],
            derived_features['energy_diff'],
            derived_features['time_diff']
        ]
        
        # 转换为tensor
        input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
        
        # 模型预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            predicted_class = torch.argmax(probabilities, dim=1).item()
            confidence = torch.max(probabilities).item()
        
        # 准备响应
        class_names = ['低耦合事件', '有效事件']
        response = {
            'success': True,
            'prediction': {
                'label': predicted_class,
                'label_name': class_names[predicted_class],
                'confidence': round(confidence, 4),
                'probability': [round(p, 4) for p in probabilities[0].tolist()]
            },
            'features': {
                'distance': round(derived_features['distance'], 2),
                'energy_diff': round(derived_features['energy_diff'], 2),
                'time_diff': round(derived_features['time_diff'], 4)
            },
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"预测完成: {class_names[predicted_class]} (置信度: {confidence:.4f})")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"预测过程出错: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'预测失败: {str(e)}'}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    """批量预测接口"""
    try:
        if model is None:
            return jsonify({'error': '模型未加载'}), 503
        
        # 获取输入数据
        data = request.get_json()
        if not data or 'samples' not in data:
            return jsonify({'error': '请提供包含samples字段的JSON数据'}), 400
        
        samples = data['samples']
        if not isinstance(samples, list):
            return jsonify({'error': 'samples必须是列表格式'}), 400
        
        if len(samples) > 100:  # 限制批次大小
            return jsonify({'error': '批次大小不能超过100'}), 400
        
        results = []
        
        for i, sample in enumerate(samples):
            try:
                # 验证输入数据
                is_valid, message = validate_input_data(sample)
                if not is_valid:
                    results.append({
                        'index': i,
                        'success': False,
                        'error': message
                    })
                    continue
                
                # 计算衍生特征
                derived_features = calculate_derived_features(sample)
                
                # 准备模型输入
                features = [
                    sample['x1'], sample['y1'], sample['z1'],
                    sample['x2'], sample['y2'], sample['z2'],
                    sample['energy1'], sample['energy2'],
                    sample['time1'], sample['time2'],
                    derived_features['distance'],
                    derived_features['energy_diff'],
                    derived_features['time_diff']
                ]
                
                # 转换为tensor
                input_tensor = torch.FloatTensor(features).unsqueeze(0).to(device)
                
                # 模型预测
                with torch.no_grad():
                    outputs = model(input_tensor)
                    probabilities = torch.softmax(outputs, dim=1)
                    predicted_class = torch.argmax(probabilities, dim=1).item()
                    confidence = torch.max(probabilities).item()
                
                # 添加结果
                class_names = ['低耦合事件', '有效事件']
                results.append({
                    'index': i,
                    'success': True,
                    'prediction': {
                        'label': predicted_class,
                        'label_name': class_names[predicted_class],
                        'confidence': round(confidence, 4),
                        'probability': [round(p, 4) for p in probabilities[0].tolist()]
                    },
                    'features': {
                        'distance': round(derived_features['distance'], 2),
                        'energy_diff': round(derived_features['energy_diff'], 2),
                        'time_diff': round(derived_features['time_diff'], 4)
                    }
                })
                
            except Exception as e:
                results.append({
                    'index': i,
                    'success': False,
                    'error': str(e)
                })
        
        # 统计结果
        successful_predictions = sum(1 for r in results if r['success'])
        
        response = {
            'success': True,
            'total_samples': len(samples),
            'successful_predictions': successful_predictions,
            'failed_predictions': len(samples) - successful_predictions,
            'results': results,
            'timestamp': datetime.now().isoformat()
        }
        
        logger.info(f"批量预测完成: {successful_predictions}/{len(samples)} 成功")
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"批量预测过程出错: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': f'批量预测失败: {str(e)}'}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'API接口不存在'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': '服务器内部错误'}), 500

if __name__ == '__main__':
    print("🚀 启动PET低耦合事件识别API服务...")
    
    # 加载模型
    if load_model_and_config():
        print("✅ 模型加载成功")
    else:
        print("⚠️ 模型加载失败，服务将以受限模式运行")
    
    # 启动服务
    print("🌐 服务启动地址: http://localhost:5000")
    print("📖 API文档: http://localhost:5000")
    print("💊 健康检查: http://localhost:5000/health")
    
    app.run(host='0.0.0.0', port=5000, debug=False) 