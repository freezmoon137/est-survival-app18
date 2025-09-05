# EST生存分析Web应用 - 免费平台部署指南

本指南将帮助您将EST生存分析Web应用部署到免费的云平台上。

## 📋 部署前准备

### 必需文件清单
确保您的项目文件夹包含以下文件：
- `app.py` - 主应用文件
- `est_model.pkl` - 训练好的EST模型
- `requirements.txt` - Python依赖包
- `templates/index.html` - 前端模板
- `static/` - 静态资源文件夹
- `Procfile` - 应用启动配置
- `render.yaml` - Render平台配置
- `runtime.txt` - Python版本指定

## 🚀 部署选项

### 1. Render平台部署（推荐）

**优势：**
- 免费额度充足
- 支持自动部署
- 配置简单

**部署步骤：**

1. **创建GitHub仓库**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/yourusername/est-survival-app.git
   git push -u origin main
   ```

2. **登录Render**
   - 访问 [render.com](https://render.com)
   - 使用GitHub账号登录

3. **创建Web Service**
   - 点击 "New" → "Web Service"
   - 连接您的GitHub仓库
   - 选择项目文件夹

4. **配置设置**
   - Name: `est-survival-app`
   - Environment: `Python 3`
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
   - Plan: `Free`

5. **部署**
   - 点击 "Create Web Service"
   - 等待部署完成（约5-10分钟）

### 2. Railway平台部署

**优势：**
- 部署速度快
- 界面友好
- 支持多种数据库

**部署步骤：**

1. **准备代码**
   - 确保代码已推送到GitHub

2. **登录Railway**
   - 访问 [railway.app](https://railway.app)
   - 使用GitHub账号登录

3. **创建项目**
   - 点击 "New Project"
   - 选择 "Deploy from GitHub repo"
   - 选择您的仓库

4. **配置环境变量**
   - 在项目设置中添加：
     - `PORT`: `8080`
     - `FLASK_ENV`: `production`

5. **部署**
   - Railway会自动检测Python项目
   - 等待部署完成

### 3. Heroku平台部署

**注意：** Heroku已取消免费计划，但仍可用于学习目的

**部署步骤：**

1. **安装Heroku CLI**
   - 下载并安装 [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli)

2. **登录Heroku**
   ```bash
   heroku login
   ```

3. **创建应用**
   ```bash
   heroku create est-survival-app
   ```

4. **部署代码**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **打开应用**
   ```bash
   heroku open
   ```

## 🔧 常见问题解决

### 1. 构建错误：setuptools.build_meta 无法导入
**错误信息：** `Cannot import 'setuptools.build_meta'`

**根本原因：** Python 3.13版本与某些包的兼容性问题

**解决方案（按优先级排序）：**

**方案A：使用固定Python版本（推荐）**
```yaml
# render.yaml中已配置
runtime: python-3.9.18
buildCommand: |
  python -m pip install --upgrade pip==23.3.1
  pip install setuptools==68.2.2 wheel==0.41.2
  pip install -r requirements-minimal.txt
```

**方案B：手动切换依赖**
```bash
# 运行部署准备脚本
bash deploy-setup.sh
# 或手动执行
cp requirements.txt requirements-full.txt
cp requirements-minimal.txt requirements.txt
```

**方案C：在Render控制台设置环境变量**
- `PYTHON_VERSION: 3.9.18`
- `PYTHONUNBUFFERED: 1`

### 2. scikit-survival 安装失败
**解决方案：**
- 使用 `requirements-minimal.txt`（不包含scikit-survival）
- 或者在代码中添加条件导入：
  ```python
  try:
      from sksurv.ensemble import RandomSurvivalForest
  except ImportError:
      print("scikit-survival not available, using alternative")
  ```

### 3. 模型文件过大
如果`est_model.pkl`文件超过100MB：
- 使用Git LFS存储大文件
- 或考虑模型压缩

### 4. 内存不足
免费计划内存有限，如果遇到内存问题：
- 优化模型加载逻辑
- 减少同时处理的请求数

### 5. 启动超时
如果应用启动超时：
- 检查依赖包是否过多
- 优化模型加载时间

### 6. 端口配置错误
确保应用使用环境变量中的端口：
```python
port = int(os.environ.get('PORT', 5000))
app.run(host='0.0.0.0', port=port)
```

## 📊 性能优化建议

1. **启用Gzip压缩**
2. **使用CDN加速静态资源**
3. **实现请求缓存**
4. **监控应用性能**

## 🔒 安全注意事项

1. **不要在代码中硬编码敏感信息**
2. **使用环境变量存储配置**
3. **启用HTTPS（平台通常自动提供）**
4. **定期更新依赖包**

## 📞 技术支持

如果在部署过程中遇到问题：
1. 检查平台的部署日志
2. 确认所有必需文件都已包含
3. 验证requirements.txt中的依赖版本
4. 查看平台的官方文档

---

**祝您部署成功！** 🎉

部署完成后，您的EST生存分析应用将可以通过公网访问，为全球用户提供服务。