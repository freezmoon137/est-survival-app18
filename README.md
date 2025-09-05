# EST生存分析模型Web部署系统

基于Extra Survival Trees (EST)模型的生存分析Web应用，提供用户友好的界面进行生存风险预测。

## 功能特性

- 🔬 **EST模型预测**: 基于Extra Survival Trees算法的生存分析
- 🌐 **Web界面**: 直观的数据输入和结果展示界面
- 📊 **可视化结果**: 生存曲线和特征重要性图表
- 🔒 **数据验证**: 完整的输入数据验证和错误处理
- 📱 **响应式设计**: 支持桌面和移动设备访问
- ⚡ **实时预测**: 快速的模型推理和结果返回

## 系统架构

```
est-survival-web/
├── app.py                 # Flask主应用文件
├── model_utils.py         # 模型管理工具
├── requirements.txt       # Python依赖
├── README.md             # 项目说明
├── templates/            # HTML模板
│   └── index.html
├── static/               # 静态资源
│   ├── css/
│   │   └── style.css
│   └── js/
│       └── main.js
└── models/               # 模型文件目录
    └── est_model.pkl     # 训练好的EST模型
```

## 环境要求

- Python 3.8+
- 内存: 至少2GB
- 磁盘空间: 至少1GB

## 快速开始

### 1. 克隆项目

```bash
git clone <repository-url>
cd est-survival-web
```

### 2. 创建虚拟环境

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 准备模型文件

将训练好的EST模型文件放置在以下位置之一：
- `./models/est_model.pkl`
- `./est_model.pkl`
- `../est_model.pkl`

如果没有预训练模型，系统将自动创建一个默认模型用于演示。

### 5. 启动应用

#### 开发模式

```bash
python app.py
```

应用将在 http://localhost:5000 启动

#### 生产模式

```bash
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

## 使用说明

### 数据输入

系统支持以下特征输入：

#### 生化指标
- **白蛋白 (Albumin)**: 35-50 g/L
- **血红蛋白 (Hemoglobin)**: 120-160 g/L
- **肌酐 (Creatinine)**: 44-133 μmol/L
- **尿素 (Urea)**: 2.9-8.2 mmol/L
- **钠 (Sodium)**: 136-145 mmol/L
- **钾 (Potassium)**: 3.5-5.1 mmol/L

#### 疾病史 (0=无, 1=有)
- 冠心病史
- 糖尿病史
- 高血压史
- 脑卒中史
- 恶性肿瘤史
- 慢性阻塞性肺病
- 慢性肾病
- 肝病史
- 痴呆史
- 抑郁症史

### 预测结果

系统提供以下预测结果：

1. **风险评分**: 0-1之间的数值，越高表示风险越大
2. **风险等级**: 低风险/中等风险/高风险
3. **生存概率**: 在特定时间点的生存概率
4. **生存曲线**: 个体化的生存函数图表
5. **特征重要性**: 各特征对预测结果的贡献度

## API接口

### 预测接口

**POST** `/api/predict`

请求体:
```json
{
    "Albumin": 35.2,
    "Hemoglobin": 120.5,
    "Creatinine": 88.3,
    "Urea": 6.8,
    "Sodium": 140.2,
    "Potassium": 4.1,
    "Coronary_Heart_Disease": 1,
    "Diabetes": 0,
    "Hypertension": 1,
    "Stroke": 0,
    "Malignancy": 0,
    "COPD": 0,
    "CKD": 1,
    "Liver_Disease": 0,
    "Dementia": 0,
    "Depression": 0
}
```

响应:
```json
{
    "risk_score": 0.456,
    "survival_probability": 0.723,
    "predicted_time": 365,
    "survival_function": {
        "times": [30, 60, 90, 180, 365],
        "probabilities": [0.95, 0.89, 0.82, 0.75, 0.68]
    },
    "feature_importance": {
        "Albumin": 0.15,
        "Creatinine": 0.12,
        "CKD": 0.10
    }
}
```

### 模型信息接口

**GET** `/api/model_info`

响应:
```json
{
    "model_type": "ExtraSurvivalTrees",
    "features": ["Albumin", "Hemoglobin", ...],
    "feature_count": 16,
    "model_loaded": true
}
```

### 健康检查接口

**GET** `/health`

响应:
```json
{
    "status": "healthy",
    "model_loaded": true,
    "timestamp": "2024-01-15T10:30:00Z"
}
```

## 部署配置

### Docker部署

创建 `Dockerfile`:

```dockerfile
FROM python:3.9-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "app:app"]
```

构建和运行:

```bash
docker build -t est-survival-web .
docker run -p 5000:5000 est-survival-web
```

### Nginx配置

```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
    }
    
    location /static {
        alias /path/to/your/app/static;
        expires 1y;
        add_header Cache-Control "public, immutable";
    }
}
```

## 故障排除

### 常见问题

1. **模型加载失败**
   - 检查模型文件路径是否正确
   - 确认模型文件格式兼容
   - 查看日志获取详细错误信息

2. **预测结果异常**
   - 验证输入数据格式和范围
   - 检查特征名称是否匹配
   - 确认模型训练时的特征顺序

3. **性能问题**
   - 增加Gunicorn worker数量
   - 使用Redis缓存预测结果
   - 优化模型加载策略

### 日志配置

应用日志保存在 `app.log` 文件中，包含：
- 请求处理日志
- 模型预测日志
- 错误和异常信息

## 安全注意事项

- 🔒 不要在生产环境中使用DEBUG模式
- 🛡️ 配置适当的CORS策略
- 🔐 对敏感医疗数据进行加密传输
- 📝 定期备份模型和配置文件
- 🚫 限制API访问频率防止滥用

## 许可证

本项目仅供医学研究和教育用途，不得用于临床诊断决策。

## 联系方式

如有问题或建议，请联系开发团队。

---

**免责声明**: 本系统提供的预测结果仅供参考，不能替代专业医疗诊断和治疗建议。