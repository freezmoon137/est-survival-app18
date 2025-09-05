// 全局变量
let survivalChart = null;
let importanceChart = null;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    initializeForm();
    setupEventListeners();
});

// 初始化表单
function initializeForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        // 添加表单验证
        addFormValidation();
    }
}

// 设置事件监听器
function setupEventListeners() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.addEventListener('submit', handleFormSubmit);
    }
}

// 处理表单提交
async function handleFormSubmit(event) {
    console.log('handleFormSubmit called'); // 调试日志
    event.preventDefault();
    
    if (!validateForm()) {
        showAlert('请填写所有必填字段并确保数据格式正确', 'error');
        return;
    }
    
    // 先隐藏结果区域，重置状态
    const resultSection = document.getElementById('result-section');
    if (resultSection) {
        resultSection.style.display = 'none';
    }
    
    // 清空时间概率容器
    const timeProbContainer = document.getElementById('time-probabilities');
    if (timeProbContainer) {
        timeProbContainer.style.display = 'none';
        timeProbContainer.innerHTML = '';
    }
    
    // 销毁现有图表
    if (survivalChart) {
        survivalChart.destroy();
        survivalChart = null;
    }
    if (importanceChart) {
        importanceChart.destroy();
        importanceChart = null;
    }
    
    const formData = collectFormData();
    
    try {
        showLoading(true);
        const result = await submitPrediction(formData);
        displayResults(result);
        showResultSection();
    } catch (error) {
        console.error('预测失败:', error);
        showAlert('预测失败，请检查输入数据或稍后重试', 'error');
    } finally {
        showLoading(false);
    }
}

// 收集表单数据
function collectFormData() {
    const form = document.getElementById('prediction-form');
    const formData = new FormData(form);
    const data = {};
    
    for (let [key, value] of formData.entries()) {
        // 预测时间点保持为整数
        if (key === 'prediction_time') {
            data[key] = parseInt(value);
        }
        // 其他数值类型转换为浮点数
        else if (value !== '' && !isNaN(value)) {
            data[key] = parseFloat(value);
        } else {
            data[key] = value;
        }
    }
    
    return data;
}

// 提交预测请求
async function submitPrediction(data) {
    const response = await fetch('/api/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });
    
    if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || '预测请求失败');
    }
    
    return await response.json();
}

// 显示预测结果
function displayResults(result) {
    console.log('displayResults called'); // 调试日志
    
    // 先隐藏时间概率容器，避免残留内容
    const timeProbContainer = document.getElementById('time-probabilities');
    if (timeProbContainer) {
        timeProbContainer.style.display = 'none';
        timeProbContainer.innerHTML = '';
    }
    
    // 清空所有结果容器，防止重复内容
    const resultCards = document.querySelectorAll('.result-card');
    resultCards.forEach(card => {
        const content = card.querySelector('.card-content');
        if (content) {
            content.innerHTML = '';
        }
    });
    
    // 显示风险评分
    const riskScore = document.getElementById('risk-score');
    const riskLevel = document.getElementById('risk-level');
    
    if (riskScore && result.risk_score !== undefined) {
        riskScore.textContent = result.risk_score.toFixed(3);
        
        // 设置风险等级
        const level = getRiskLevel(result.risk_score);
        riskLevel.textContent = level.text;
        riskLevel.className = `risk-level ${level.class}`;
    }
    
    // 显示生存概率
    const survivalProb = document.getElementById('survival-prob');
    const survivalTime = document.getElementById('survival-time');
    
    if (survivalProb && result.survival_probability_at_time !== undefined) {
        survivalProb.textContent = `${(result.survival_probability_at_time * 100).toFixed(1)}%`;
    }
    
    if (survivalTime && result.prediction_time !== undefined) {
        survivalTime.textContent = `预测时间点: ${result.prediction_time} 天`;
    }
    
    // 显示各时间点生存概率
    if (result.survival_probabilities) {
        displayTimeProbabilities(result.survival_probabilities, result.prediction_time);
    }
    
    // 绘制生存曲线
    if (result.survival_function) {
        drawSurvivalChart(result.survival_function);
    }
    
    // 绘制特征重要性图
    if (result.feature_importance) {
        drawImportanceChart(result.feature_importance);
    }
}

// 显示各时间点生存概率
function displayTimeProbabilities(probabilities, selectedTime) {
    const container = document.getElementById('time-probabilities');
    if (!container) return;
    
    // 显示容器
    container.style.display = 'block';
    
    let html = `
        <h4>各时间点生存概率</h4>
        <div class="probabilities-grid">
    `;
    
    Object.entries(probabilities)
        .filter(([time]) => parseInt(time) !== 365 && parseInt(time) !== 730) // 过滤掉365天和730天
        .sort(([a], [b]) => parseInt(a) - parseInt(b))
        .forEach(([time, prob]) => {
            const isSelected = parseInt(time) === selectedTime;
            const className = isSelected ? 'probability-item selected' : 'probability-item';
            html += `
                <div class="${className}">
                    <span class="time">${time}${isSelected ? ' (选中)' : ''}</span>
                    <span class="probability">${(prob * 100).toFixed(2)}%</span>
                </div>
            `;
        });
    
    html += `</div>`;
    container.innerHTML = html;
}

// 获取风险等级
function getRiskLevel(score) {
    if (score < 0.3) {
        return { text: '低风险', class: 'risk-low' };
    } else if (score < 0.7) {
        return { text: '中等风险', class: 'risk-medium' };
    } else {
        return { text: '高风险', class: 'risk-high' };
    }
}

// 绘制生存曲线
function drawSurvivalChart(survivalData) {
    const ctx = document.getElementById('survival-chart');
    if (!ctx) return;
    
    // 销毁现有图表
    if (survivalChart) {
        survivalChart.destroy();
    }
    
    const times = survivalData.times || [];
    const probabilities = survivalData.probabilities || [];
    
    survivalChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: times,
            datasets: [{
                label: '生存概率',
                data: probabilities,
                borderColor: '#3498db',
                backgroundColor: 'rgba(52, 152, 219, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '时间 (天)'
                    }
                },
                y: {
                    title: {
                        display: true,
                        text: '生存概率'
                    },
                    min: 0,
                    max: 1
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '个体生存曲线预测'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

// 绘制特征重要性图
function drawImportanceChart(importanceData) {
    const ctx = document.getElementById('importance-chart');
    if (!ctx) return;
    
    // 销毁现有图表
    if (importanceChart) {
        importanceChart.destroy();
    }
    
    const features = Object.keys(importanceData);
    const importance = Object.values(importanceData);
    
    // 按重要性排序
    const sortedData = features.map((feature, index) => ({
        feature: getFeatureDisplayName(feature),
        importance: importance[index]
    })).sort((a, b) => b.importance - a.importance).slice(0, 10); // 只显示前10个
    
    importanceChart = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedData.map(item => item.feature),
            datasets: [{
                label: '重要性',
                data: sortedData.map(item => item.importance),
                backgroundColor: [
                    '#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                    '#1abc9c', '#34495e', '#e67e22', '#95a5a6', '#f1c40f'
                ],
                borderWidth: 1
            }]
        },
        options: {
            indexAxis: 'y',
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    title: {
                        display: true,
                        text: '重要性得分'
                    }
                }
            },
            plugins: {
                title: {
                    display: true,
                    text: '特征重要性分析'
                },
                legend: {
                    display: false
                }
            }
        }
    });
}

// 获取特征显示名称
function getFeatureDisplayName(feature) {
    const nameMap = {
        'Albumin': '白蛋白',
        'Hemoglobin': '血红蛋白',
        'Creatinine': '肌酐',
        'Urea': '尿素',
        'Sodium': '钠',
        'Potassium': '钾',
        'Coronary_Heart_Disease': '冠心病',
        'Diabetes': '糖尿病',
        'Hypertension': '高血压',
        'Stroke': '脑卒中',
        'Malignancy': '恶性肿瘤',
        'COPD': '慢性阻塞性肺病',
        'CKD': '慢性肾病',
        'Liver_Disease': '肝病',
        'Dementia': '痴呆',
        'Depression': '抑郁症'
    };
    return nameMap[feature] || feature;
}

// 表单验证
function validateForm() {
    const form = document.getElementById('prediction-form');
    const inputs = form.querySelectorAll('input[required], select[required]');
    let isValid = true;
    
    inputs.forEach(input => {
        if (!input.value.trim()) {
            isValid = false;
            input.style.borderColor = '#e74c3c';
        } else {
            input.style.borderColor = '#27ae60';
        }
    });
    
    return isValid;
}

// 添加表单验证
function addFormValidation() {
    const inputs = document.querySelectorAll('input, select');
    
    inputs.forEach(input => {
        input.addEventListener('blur', function() {
            if (this.hasAttribute('required')) {
                if (!this.value.trim()) {
                    this.style.borderColor = '#e74c3c';
                } else {
                    this.style.borderColor = '#27ae60';
                }
            }
        });
        
        input.addEventListener('input', function() {
            if (this.style.borderColor === 'rgb(231, 76, 60)') {
                this.style.borderColor = '#e9ecef';
            }
        });
    });
}

// 显示/隐藏加载动画
function showLoading(show) {
    const loading = document.getElementById('loading');
    if (loading) {
        loading.style.display = show ? 'flex' : 'none';
    }
}

// 显示结果区域
function showResultSection() {
    console.log('showResultSection called'); // 调试日志
    const resultSection = document.getElementById('result-section');
    if (resultSection) {
        resultSection.style.display = 'block';
        resultSection.scrollIntoView({ behavior: 'smooth' });
    }
}

// 重置表单
function resetForm() {
    const form = document.getElementById('prediction-form');
    if (form) {
        form.reset();
        
        // 重置边框颜色
        const inputs = form.querySelectorAll('input, select');
        inputs.forEach(input => {
            input.style.borderColor = '#e9ecef';
        });
        
        // 隐藏结果区域
        const resultSection = document.getElementById('result-section');
        if (resultSection) {
            resultSection.style.display = 'none';
        }
        
        // 销毁图表
        if (survivalChart) {
            survivalChart.destroy();
            survivalChart = null;
        }
        if (importanceChart) {
            importanceChart.destroy();
            importanceChart = null;
        }
    }
}

// 加载示例数据
function loadSampleData() {
    const sampleData = {
        'Albumin': 35.2,
        'Hemoglobin': 120.5,
        'Creatinine': 88.3,
        'Urea': 6.8,
        'Sodium': 140.2,
        'Potassium': 4.1,
        'Coronary_Heart_Disease': '1',
        'Diabetes': '0',
        'Hypertension': '1',
        'Stroke': '0',
        'Malignancy': '0',
        'COPD': '0',
        'CKD': '1',
        'Liver_Disease': '0',
        'Dementia': '0',
        'Depression': '0'
    };
    
    // 填充表单
    Object.keys(sampleData).forEach(key => {
        const element = document.querySelector(`[name="${key}"]`);
        if (element) {
            element.value = sampleData[key];
            element.style.borderColor = '#27ae60';
        }
    });
    
    showAlert('示例数据已加载', 'success');
}

// 显示提示信息
function showAlert(message, type = 'info') {
    // 创建提示框
    const alert = document.createElement('div');
    alert.className = `alert alert-${type}`;
    alert.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 8px;
        color: white;
        font-weight: 500;
        z-index: 10000;
        animation: slideIn 0.3s ease-out;
    `;
    
    // 设置背景色
    switch (type) {
        case 'success':
            alert.style.backgroundColor = '#27ae60';
            break;
        case 'error':
            alert.style.backgroundColor = '#e74c3c';
            break;
        case 'warning':
            alert.style.backgroundColor = '#f39c12';
            break;
        default:
            alert.style.backgroundColor = '#3498db';
    }
    
    alert.textContent = message;
    document.body.appendChild(alert);
    
    // 3秒后自动移除
    setTimeout(() => {
        alert.style.animation = 'slideOut 0.3s ease-in';
        setTimeout(() => {
            if (alert.parentNode) {
                alert.parentNode.removeChild(alert);
            }
        }, 300);
    }, 3000);
}

// 添加动画样式
const style = document.createElement('style');
style.textContent = `
    @keyframes slideIn {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOut {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
`;
document.head.appendChild(style);