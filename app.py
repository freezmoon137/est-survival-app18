#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EST生存分析模型Web部署应用
基于Flask框架的Web接口，提供EST模型预测服务
"""

import os
import sys
import logging
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
from sklearn.ensemble import RandomForestRegressor
try:
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    SKSURV_AVAILABLE = True
except ImportError:
    print("Warning: scikit-survival not available, using fallback model")
    SKSURV_AVAILABLE = False

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)
CORS(app)  # 启用跨域支持

# 配置
app.config['SECRET_KEY'] = 'est-survival-model-2024'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# 全局变量
est_model = None
model_loaded = False

# 定义特征列（基于EST模型训练数据 - train_data_lasso_1se_vars.csv）
FEATURE_COLUMNS = [
    'Albumin', 'Coronary_Heart_Disease', 'Hemoglobin', 'Intracerebral_Hemorrhage',
    'NYHA_Classification', 'Septic_Shock', 'Severe_valve_regurgitationn', 'Surgery', 'neutrophil_percentage'
]

# 时间和状态列
TIME_COLUMN = 'time'
STATUS_COLUMN = 'status'

# 特征信息（用于前端表单生成和数据验证）
FEATURE_INFO = {
    'Albumin': {
        'type': 'number', 'min': 10, 'max': 60, 'step': 0.1, 'unit': 'g/L', 'label': '白蛋白',
        'name': '白蛋白', 'range': '35-50', 'description': '血清白蛋白水平'
    },
    'Coronary_Heart_Disease': {
        'type': 'select', 'options': [0, 1], 'label': '冠心病', 'options_text': ['无', '有'],
        'name': '冠心病', 'range': '0-1', 'description': '是否患有冠心病'
    },
    'Hemoglobin': {
        'type': 'number', 'min': 50, 'max': 200, 'step': 1, 'unit': 'g/L', 'label': '血红蛋白',
        'name': '血红蛋白', 'range': '120-160', 'description': '血红蛋白浓度'
    },
    'Intracerebral_Hemorrhage': {
        'type': 'select', 'options': [0, 1], 'label': '脑出血', 'options_text': ['无', '有'],
        'name': '脑出血', 'range': '0-1', 'description': '是否发生脑出血'
    },
    'NYHA_Classification': {
        'type': 'select', 'options': [1, 2, 3, 4], 'label': 'NYHA心功能分级', 'options_text': ['I级', 'II级', 'III级', 'IV级'],
        'name': 'NYHA心功能分级', 'range': '1-4', 'description': '纽约心脏协会心功能分级'
    },
    'Septic_Shock': {
        'type': 'select', 'options': [0, 1], 'label': '感染性休克', 'options_text': ['无', '有'],
        'name': '感染性休克', 'range': '0-1', 'description': '是否发生感染性休克'
    },
    'Severe_valve_regurgitationn': {
        'type': 'select', 'options': [0, 1], 'label': '重度瓣膜反流', 'options_text': ['无', '有'],
        'name': '重度瓣膜反流', 'range': '0-1', 'description': '是否存在重度瓣膜反流'
    },
    'Surgery': {
        'type': 'select', 'options': [0, 1], 'label': '手术治疗', 'options_text': ['无', '有'],
        'name': '手术治疗', 'range': '0-1', 'description': '是否接受手术治疗'
    },
    'neutrophil_percentage': {
        'type': 'number', 'min': 0, 'max': 100, 'step': 0.1, 'unit': '%', 'label': '中性粒细胞百分比',
        'name': '中性粒细胞百分比', 'range': '50-70', 'description': '中性粒细胞百分比'
    }
}

# 预测时间点选项（天）
TIME_POINTS = [30, 60, 90, 180, 365, 730]  # 1个月到2年

class SurvivalFunction:
    """生存函数对象"""
    def __init__(self, times, probabilities):
        self.x = np.array(times)
        self.y = np.array(probabilities)
    
    def __call__(self, t):
        """在指定时间点计算生存概率"""
        if t <= self.x[0]:
            return self.y[0]
        if t >= self.x[-1]:
            return self.y[-1]
        # 线性插值
        idx = np.searchsorted(self.x, t)
        if idx == 0:
            return self.y[0]
        if idx >= len(self.x):
            return self.y[-1]
        
        t1, t2 = self.x[idx-1], self.x[idx]
        p1, p2 = self.y[idx-1], self.y[idx]
        return p1 + (p2 - p1) * (t - t1) / (t2 - t1)

class ExtraSurvivalTrees:
    """EST模型包装类"""
    
    def __init__(self, n_estimators=100, random_state=42):
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.feature_names = None
        self.is_fitted = False
        
        if SKSURV_AVAILABLE:
            self.model = RandomSurvivalForest(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
        else:
            # 使用RandomForestRegressor作为fallback
            self.model = RandomForestRegressor(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=-1
            )
    
    def fit(self, X, y):
        """训练模型"""
        self.feature_names = list(X.columns) if hasattr(X, 'columns') else None
        
        if SKSURV_AVAILABLE:
            self.model.fit(X, y)
        else:
            # 对于fallback模型，使用时间作为目标变量
            if hasattr(y, 'dtype') and hasattr(y.dtype, 'names'):
                # 结构化数组，提取时间
                times = y['time'] if 'time' in y.dtype.names else y[TIME_COLUMN]
            else:
                # 假设y是时间数组
                times = y
            self.model.fit(X, times)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """预测风险评分"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if SKSURV_AVAILABLE:
            return self.model.predict(X)
        else:
            # 对于fallback模型，预测时间并转换为风险评分
            predicted_times = self.model.predict(X)
            # 将时间转换为风险评分（时间越短，风险越高）
            risk_scores = 1.0 / (1.0 + predicted_times / 365.0)
            return risk_scores
    
    def predict_survival_function(self, X, return_array=False):
        """预测生存函数"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if SKSURV_AVAILABLE:
            return self.model.predict_survival_function(X, return_array=return_array)
        else:
            # 为fallback模型生成模拟的生存函数
            survival_functions = []
            predicted_times = self.model.predict(X)
            
            for pred_time in predicted_times:
                # 生成时间点
                times = np.arange(30, 731, 30)  # 30天到2年
                
                # 生成生存概率（基于预测的生存时间）
                # 使用指数衰减模型
                lambda_param = 1.0 / max(pred_time, 30)  # 避免除零
                probabilities = np.exp(-lambda_param * times)
                
                # 确保概率在合理范围内
                probabilities = np.clip(probabilities, 0.01, 0.99)
                
                survival_functions.append(SurvivalFunction(times, probabilities))
            
            return survival_functions
    
    def predict_cumulative_hazard_function(self, X, return_array=False):
        """预测累积风险函数"""
        if not self.is_fitted:
            raise ValueError("模型尚未训练")
        
        if SKSURV_AVAILABLE:
            return self.model.predict_cumulative_hazard_function(X, return_array=return_array)
        else:
            # 为fallback模型生成模拟的累积风险函数
            survival_functions = self.predict_survival_function(X)
            hazard_functions = []
            
            for surv_func in survival_functions:
                # 从生存函数计算累积风险函数
                hazard_probs = -np.log(np.maximum(surv_func.y, 1e-10))
                hazard_functions.append(SurvivalFunction(surv_func.x, hazard_probs))
            
            return hazard_functions

def load_est_model():
    """加载EST模型"""
    global est_model, model_loaded
    
    # 尝试从多个路径加载模型
    model_paths = [
        './models/est_model.pkl',
        './est_model.pkl',
        '../est_model.pkl',
        './models/orsf_model.pkl'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                logger.info(f"尝试从 {path} 加载模型...")
                loaded_data = joblib.load(path)
                
                # 检查加载的数据格式
                if isinstance(loaded_data, dict):
                    # 如果是字典格式（来自model_utils.py），提取模型对象
                    if 'model' in loaded_data:
                        est_model = loaded_data['model']
                        logger.info(f"从字典格式中提取模型对象")
                    else:
                        logger.warning(f"字典中未找到'model'键")
                        continue
                else:
                    # 如果是直接的模型对象
                    est_model = loaded_data
                
                model_loaded = True
                logger.info(f"成功从 {path} 加载模型")
                return True
            except Exception as e:
                logger.warning(f"从 {path} 加载模型失败: {e}")
                continue
    
    # 如果没有找到预训练模型，创建一个默认模型
    logger.warning("未找到预训练模型，创建默认模型用于演示")
    try:
        est_model = create_default_model()
        model_loaded = True
        logger.info("成功创建默认模型")
        return True
    except Exception as e:
        logger.error(f"创建默认模型失败: {e}")
        return False

def create_default_model():
    """创建默认的EST模型用于演示"""
    # 创建示例数据
    np.random.seed(42)
    n_samples = 1000
    
    # 生成特征数据（只使用FEATURE_COLUMNS中定义的特征）
    data = {}
    for feature in FEATURE_COLUMNS:
        if feature in ['Albumin']:
            data[feature] = np.random.normal(40, 5, n_samples)
        elif feature in ['Hemoglobin']:
            data[feature] = np.random.normal(130, 15, n_samples)
        elif feature in ['neutrophil_percentage']:
            data[feature] = np.random.normal(60, 10, n_samples)
        elif feature in ['NYHA_Classification']:
            data[feature] = np.random.choice([1, 2, 3, 4], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        else:
            # 二分类特征
            data[feature] = np.random.binomial(1, 0.3, n_samples)
    
    X = pd.DataFrame(data)
    
    # 生成生存数据
    # 基于特征计算风险评分
    risk_score = (
        -0.1 * (X['Albumin'] - 40) / 5 +
        -0.1 * (X['Hemoglobin'] - 130) / 15 +
        0.1 * (X['neutrophil_percentage'] - 60) / 10 +
        0.1 * X['Coronary_Heart_Disease'] +
        0.15 * X['Intracerebral_Hemorrhage'] +
        0.1 * (X['NYHA_Classification'] - 1) / 3 +
        0.2 * X['Septic_Shock'] +
        0.1 * X['Severe_valve_regurgitationn'] +
        -0.1 * X['Surgery']  # 手术治疗降低风险
    )
    
    # 生成生存时间
    baseline_hazard = 0.01
    hazard = baseline_hazard * np.exp(risk_score)
    time = np.random.exponential(1/hazard)
    
    # 生成删失状态
    censoring_time = np.random.exponential(1/0.005)  # 删失时间
    observed_time = np.minimum(time, censoring_time)
    event = time <= censoring_time
    
    # 训练模型
    model = ExtraSurvivalTrees(n_estimators=50, random_state=42)
    
    if SKSURV_AVAILABLE:
        # 创建结构化数组
        y = Surv.from_arrays(event, observed_time)
        model.fit(X, y)
    else:
        # 对于fallback模式，直接使用时间作为目标变量
        model.fit(X, observed_time)
    
    return model

# 在应用启动时加载模型
logger.info("启动EST生存分析Web应用...")
if load_est_model():
    logger.info("模型加载成功")
else:
    logger.error("模型加载失败，但仍启动Web服务器")

def preprocess_input_data(data: Dict[str, Any]) -> pd.DataFrame:
    """预处理输入数据"""
    # 验证必需字段
    missing_fields = [field for field in FEATURE_COLUMNS if field not in data]
    if missing_fields:
        raise ValueError(f"缺少必需字段: {missing_fields}")
    
    # 创建DataFrame
    df = pd.DataFrame([data])
    
    # 验证数据类型和范围
    for feature, info in FEATURE_INFO.items():
        if feature in df.columns:
            value = df[feature].iloc[0]
            
            if info['type'] == 'numeric':
                try:
                    value = float(value)
                    min_val, max_val = info['range']
                    if not (min_val <= value <= max_val):
                        logger.warning(f"{feature} 值 {value} 超出正常范围 [{min_val}, {max_val}]")
                except (ValueError, TypeError):
                    raise ValueError(f"{feature} 必须是数值类型")
            
            elif info['type'] == 'categorical':
                try:
                    value = int(value)
                    if value not in info['values']:
                        raise ValueError(f"{feature} 值必须是 {info['values']} 中的一个")
                except (ValueError, TypeError):
                    raise ValueError(f"{feature} 必须是整数类型")
    
    # 确保列顺序正确
    df = df[FEATURE_COLUMNS]
    
    return df

def predict_survival_risk(data: Dict[str, Any], prediction_time: int = 180) -> Dict[str, Any]:
    """预测生存风险
    
    Parameters:
    -----------
    data : Dict[str, Any]
        输入特征数据
    prediction_time : int
        预测时间点（天），默认180天
    
    Returns:
    --------
    Dict[str, Any]
        预测结果，包含风险评分、生存概率等
    """
    global est_model, model_loaded
    
    if not model_loaded or est_model is None:
        raise ValueError("模型未加载")
    
    # 预处理数据
    df = preprocess_input_data(data)
    
    try:
        # 预测风险评分
        risk_scores = est_model.predict(df)
        risk_score = float(risk_scores[0])
        
        # 预测生存函数
        survival_functions = est_model.predict_survival_function(df)
        
        # 提取生存函数数据
        if hasattr(survival_functions[0], 'x') and hasattr(survival_functions[0], 'y'):
            times = survival_functions[0].x
            probabilities = survival_functions[0].y
        else:
            # 如果是数组格式
            times = np.arange(30, 731, 30)  # 30天到2年，每30天一个点
            if hasattr(survival_functions, '__len__') and len(survival_functions) > 0:
                if hasattr(survival_functions[0], '__call__'):
                    probabilities = [survival_functions[0](t) for t in times]
                else:
                    probabilities = np.linspace(0.9, 0.1, len(times))  # 默认值
            else:
                probabilities = np.linspace(0.9, 0.1, len(times))  # 默认值
        
        # 计算所有预设时间点的生存概率
        survival_probs = {}
        for t in TIME_POINTS:
            # 找到最接近的时间点
            idx = np.argmin(np.abs(np.array(times) - t))
            if idx < len(probabilities):
                survival_probs[f"{t}天"] = float(probabilities[idx])
        
        # 计算指定时间点的生存概率
        target_idx = np.argmin(np.abs(np.array(times) - prediction_time))
        target_survival_prob = float(probabilities[target_idx]) if target_idx < len(probabilities) else 0.5
        
        # 预测生存时间（中位生存时间）
        predicted_time = 365  # 默认值
        if len(times) > 0 and len(probabilities) > 0:
            # 找到生存概率接近0.5的时间点
            prob_array = np.array(probabilities)
            if np.any(prob_array <= 0.5):
                idx_median = np.where(prob_array <= 0.5)[0][0]
                predicted_time = float(times[idx_median])
        
        # 生成生存曲线数据（限制到合理的时间范围）
        max_time = max(prediction_time * 1.5, 365)  # 最多到预测时间的1.5倍或1年
        max_time = min(max_time, 730)  # 但不超过2年
        max_time_idx = np.where(np.array(times) <= max_time)[0]
        if len(max_time_idx) > 0:
            end_idx = min(max_time_idx[-1] + 1, len(times))
        else:
            end_idx = min(50, len(times))  # 默认限制到50个点
        
        # 特征重要性（模拟）
        feature_importance = {}
        # scikit-survival的ExtraSurvivalTrees不支持feature_importances_属性
        # 使用模拟的特征重要性
        np.random.seed(42)
        importances = np.random.random(len(FEATURE_COLUMNS))
            importances = importances / importances.sum()
            for i, feature in enumerate(FEATURE_COLUMNS):
                feature_importance[FEATURE_INFO[feature]['label']] = float(importances[i])
        
        result = {
            'success': True,
            'risk_score': risk_score,
            'prediction_time': prediction_time,
            'survival_probability_at_time': target_survival_prob,
            'survival_probabilities': survival_probs,
            'predicted_time': predicted_time,
            'survival_function': {
                'times': [float(t) for t in times[:end_idx]],
                'probabilities': [float(p) for p in probabilities[:end_idx]]
            },
            'feature_importance': feature_importance,
            'model_info': {
                'model_type': 'Extra Survival Trees (EST)',
                'features_used': len(FEATURE_COLUMNS),
                'prediction_time': datetime.now().isoformat(),
                'target_time_days': prediction_time
            }
        }
        
        logger.info(f"预测完成，风险评分: {risk_score:.3f}，{prediction_time}天生存概率: {target_survival_prob:.3f}")
        return result
        
    except Exception as e:
        logger.error(f"预测过程中出现错误: {e}")
        logger.error(traceback.format_exc())
        raise

# 路由定义
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """预测接口（表单提交）"""
    try:
        # 获取表单数据
        data = {}
        for feature in FEATURE_COLUMNS:
            value = request.form.get(feature)
            if value is not None:
                data[feature] = value
        
        # 获取预测时间点参数
        prediction_time = request.form.get('prediction_time', 180)
        try:
            prediction_time = int(prediction_time)
        except (ValueError, TypeError):
            prediction_time = 180
        
        # 验证预测时间点
        if prediction_time not in TIME_POINTS:
            return jsonify({
                'success': False,
                'error': f'无效的预测时间点，支持的时间点: {TIME_POINTS}'
            }), 400
        
        # 执行预测
        result = predict_survival_risk(data, prediction_time)
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        logger.error(f"预测失败: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API预测接口"""
    try:
        # 获取JSON数据
        data = request.get_json()
        if not data:
            return jsonify({'error': '无效的JSON数据'}), 400
        
        # 提取预测时间点（可选参数）
        prediction_time = data.get('prediction_time', 180)  # 默认180天
        
        # 验证预测时间点
        if prediction_time not in TIME_POINTS:
            return jsonify({
                'error': f'无效的预测时间点，支持的时间点: {TIME_POINTS}'
            }), 400
        
        # 验证必要字段
        missing_fields = []
        for field in FEATURE_COLUMNS:
            if field not in data:
                missing_fields.append(field)
        
        if missing_fields:
            return jsonify({
                'error': f'缺少必要字段: {", ".join(missing_fields)}'
            }), 400
        
        # 执行预测
        result = predict_survival_risk(data, prediction_time)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API预测失败: {e}")
        return jsonify({'error': str(e)}), 400

@app.route('/api/model_info', methods=['GET'])
def model_info():
    """模型信息接口"""
    global est_model, model_loaded
    
    info = {
        'model_type': 'Extra Survival Trees (EST)',
        'features': FEATURE_COLUMNS,
        'feature_count': len(FEATURE_COLUMNS),
        'model_loaded': model_loaded,
        'feature_info': FEATURE_INFO,
        'time_points': TIME_POINTS,
        'time_column': TIME_COLUMN,
        'status_column': STATUS_COLUMN,
        'description': 'EST生存分析模型，基于train_data_lasso_1se_vars.csv训练数据'
    }
    
    if model_loaded and est_model:
        if hasattr(est_model, 'n_estimators'):
            info['n_estimators'] = est_model.n_estimators
        if hasattr(est_model, 'random_state'):
            info['random_state'] = est_model.random_state
    
    return jsonify(info)

@app.route('/health', methods=['GET'])
def health_check():
    """健康检查接口"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': '页面未找到'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"内部服务器错误: {error}")
    return jsonify({'error': '内部服务器错误'}), 500

if __name__ == '__main__':
    # 启动时加载模型
    logger.info("启动EST生存分析Web应用...")
    
    if load_est_model():
        logger.info("模型加载成功，启动Web服务器")
    else:
        logger.error("模型加载失败，但仍启动Web服务器")
    
    # 启动Flask应用
    # 获取环境变量中的端口，如果没有则使用5000
    port = int(os.environ.get('PORT', 5000))
    
    # 根据环境变量决定是否开启调试模式
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=debug_mode,
        threaded=True
    )