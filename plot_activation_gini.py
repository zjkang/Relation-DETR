"""
绘制不同模型的 Query Activation Gini 系数对比图

这个脚本用于分析和可视化不同 DETR 模型的 query activation 分布，
计算 Gini 系数和 Pareto 系数，并生成对比图。
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import torch

# 设置字体以支持公式显示
rcParams['font.family'] = 'serif'
rcParams['font.serif'] = ['Times New Roman']
rcParams['mathtext.fontset'] = 'stix'

def calculate_gini_coefficient(values):
    """
    计算 Gini 系数
    
    Args:
        values: numpy array or list, activation values
        
    Returns:
        float: Gini coefficient (0-1)
            0 表示完全均匀分布
            1 表示完全不均匀分布（所有激活集中在一个 query）
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    
    values = np.array(values).flatten()
    values = values[values > 0]  # 只考虑大于0的值
    
    if len(values) == 0:
        return 0.0
    
    # 排序
    sorted_values = np.sort(values)
    n = len(sorted_values)
    
    # 计算 Gini 系数
    # Gini = (2 * sum(i * x_i)) / (n * sum(x_i)) - (n + 1) / n
    cumsum = np.cumsum(sorted_values)
    gini = (2 * np.sum((np.arange(1, n + 1) * sorted_values))) / (n * cumsum[-1]) - (n + 1) / n
    
    return gini


def calculate_pareto_coefficient(values, threshold_percentile=20):
    """
    计算 Pareto 系数 (80-20 rule)
    
    Args:
        values: numpy array, activation values
        threshold_percentile: int, 用于计算的百分位数（默认20%）
        
    Returns:
        float: Pareto coefficient
            接近1表示符合80-20规则（20%的query产生80%的activation）
            小于0.5表示分布更均匀
    """
    if isinstance(values, torch.Tensor):
        values = values.cpu().numpy()
    
    values = np.array(values).flatten()
    values = values[values > 0]
    
    if len(values) == 0:
        return 0.0
    
    # 排序（降序）
    sorted_values = np.sort(values)[::-1]
    
    # 计算累积和
    cumsum = np.cumsum(sorted_values)
    total = cumsum[-1]
    
    # 找到前threshold_percentile%的queries产生的activation占比
    n_top = max(1, int(len(sorted_values) * threshold_percentile / 100))
    top_contribution = cumsum[n_top - 1] / total
    
    return top_contribution


def plot_activation_gini_comparison(models_data, save_path='activation_gini_comparison.pdf', figsize=(15, 10)):
    """
    绘制多个模型的 activation 分布对比图
    
    Args:
        models_data: dict, 格式为 {
            'model_name': {
                'activations': numpy array or list, query activation counts
                'title': str, 子图标题 (可选)
            }
        }
        save_path: str, 保存路径
        figsize: tuple, 图片大小
    """
    n_models = len(models_data)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, data) in enumerate(models_data.items()):
        row = idx // n_cols
        col = idx % n_cols
        ax = axes[row, col]
        
        # 获取 activation 数据
        activations = data['activations']
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        activations = np.array(activations).flatten()
        
        # 过滤掉0值并排序（降序）
        activations = activations[activations > 0]
        sorted_activations = np.sort(activations)[::-1]
        
        # 计算 Gini 和 Pareto 系数
        gini = calculate_gini_coefficient(activations)
        pareto = calculate_pareto_coefficient(activations)
        
        # 绘制曲线
        x = np.arange(len(sorted_activations))
        ax.plot(x, sorted_activations, linewidth=2, color=data.get('color', 'blue'))
        
        # 设置对数坐标
        ax.set_yscale('log')
        
        # 添加20%的垂直线（Pareto线）
        pareto_line_x = int(len(sorted_activations) * 0.2)
        ax.axvline(x=pareto_line_x, color='red', linestyle='--', linewidth=1.5, alpha=0.7)
        
        # 设置标题和标签
        title = data.get('title', model_name)
        ax.set_title(f'$\\it{{{title}}}$\nGini={gini:.3f}, Pareto={pareto:.3f}', 
                    fontsize=12, pad=10)
        ax.set_xlabel('Query Index (Ranked)', fontsize=10)
        ax.set_ylabel('Activation Count (log)', fontsize=10)
        
        # 设置网格
        ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
        
        # 设置刻度
        ax.tick_params(labelsize=9)
    
    # 隐藏多余的子图
    for idx in range(n_models, n_rows * n_cols):
        row = idx // n_cols
        col = idx % n_cols
        axes[row, col].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"图片已保存到: {save_path}")
    plt.show()
    
    # 打印统计信息
    print("\n" + "="*60)
    print("模型 Gini 和 Pareto 系数对比:")
    print("="*60)
    for model_name, data in models_data.items():
        activations = data['activations']
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        gini = calculate_gini_coefficient(activations)
        pareto = calculate_pareto_coefficient(activations)
        print(f"{model_name:20s} | Gini: {gini:.3f} | Pareto: {pareto:.3f}")
    print("="*60)


def analyze_query_activations(model, images, category_id, device='cuda'):
    """
    分析模型在给定图片上的 query activation 模式
    
    Args:
        model: 模型实例
        images: list of images or tensor
        category_id: int, 目标类别ID
        device: str, 设备
        
    Returns:
        numpy array: query activation counts
    """
    model.eval()
    activation_counts = []
    
    with torch.no_grad():
        for image in images:
            if isinstance(image, np.ndarray):
                # 转换 numpy array 到 tensor
                image = torch.from_numpy(image.transpose(2, 0, 1)).float()
            
            image = image.to(device)
            
            # 前向传播
            outputs = model([image])
            
            # 获取预测结果
            pred_boxes = outputs[0]['boxes']
            pred_scores = outputs[0]['scores']
            pred_labels = outputs[0]['labels']
            
            # 统计该类别的激活query数量
            mask = (pred_labels == category_id) & (pred_scores > 0.5)
            activation_counts.append(mask.sum().item())
    
    return np.array(activation_counts)


# 示例：生成模拟数据并绘制
if __name__ == "__main__":
    # 生成模拟数据（实际使用时应该从模型推理中获取）
    np.random.seed(42)
    
    # 模拟不同模型的 activation 分布
    models_data = {
        'Deformable-DETR': {
            'activations': np.concatenate([
                np.random.exponential(1000, 50),  # 前50个query高激活
                np.random.exponential(100, 150),  # 中间150个中等激活
                np.random.exponential(10, 700)    # 后700个低激活
            ]),
            'color': 'blue',
            'title': 'Deformable-DETR'
        },
        'DN-DETR': {
            'activations': np.concatenate([
                np.random.exponential(800, 100),  # 更均匀的分布
                np.random.exponential(200, 400),
                np.random.exponential(50, 400)
            ]),
            'color': 'orange',
            'title': 'DN-DETR'
        },
        'DINO': {
            'activations': np.concatenate([
                np.random.exponential(1200, 30),  # 非常集中的分布
                np.random.exponential(80, 100),
                np.random.exponential(5, 770)
            ]),
            'color': 'green',
            'title': 'DINO'
        },
        'PaQ-Deformable-DETR': {
            'activations': np.concatenate([
                np.random.exponential(900, 70),
                np.random.exponential(120, 200),
                np.random.exponential(15, 630)
            ]),
            'color': 'blue',
            'title': 'PaQ-Deformable-DETR'
        },
        'PaQ-DN-DETR': {
            'activations': np.concatenate([
                np.random.exponential(600, 150),  # 最均匀的分布
                np.random.exponential(250, 350),
                np.random.exponential(80, 400)
            ]),
            'color': 'orange',
            'title': 'PaQ-DN-DETR'
        },
        'PaQ-DINO': {
            'activations': np.concatenate([
                np.random.exponential(1100, 40),
                np.random.exponential(90, 120),
                np.random.exponential(8, 740)
            ]),
            'color': 'green',
            'title': 'PaQ-DINO'
        }
    }
    
    # 绘制对比图
    plot_activation_gini_comparison(
        models_data,
        save_path='activation_gini_comparison.pdf',
        figsize=(15, 10)
    )
