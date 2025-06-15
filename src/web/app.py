#!/usr/bin/env python3
"""
ALQuery3D Web应用 - 基于Flask的深色主题embedding生成器
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))

import json
import numpy as np
import h5py
import atexit
from flask import Flask, render_template, request, jsonify
import plotly.graph_objs as go
import plotly.utils

from src.data.embedding_generator import EmbeddingGenerator

app = Flask(__name__)
app.secret_key = 'alquery3d_secret_key'

# 全局变量存储当前生成器和数据
current_generator = None
current_embeddings = None
current_labels = None

# 数据文件路径
DATA_DIR = os.path.join(os.path.dirname(__file__), '../../data')
DATA_FILE = os.path.join(DATA_DIR, 'tmp_data.h5')

# 确保数据目录存在
os.makedirs(DATA_DIR, exist_ok=True)


def cleanup_data_file():
    """清理临时数据文件"""
    if os.path.exists(DATA_FILE):
        try:
            os.remove(DATA_FILE)
            print(f"已清理临时数据文件: {DATA_FILE}")
        except Exception as e:
            print(f"清理数据文件时出错: {e}")


# 注册退出时的清理函数
atexit.register(cleanup_data_file)


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html')


@app.route('/api/generate', methods=['POST'])
def generate_embeddings():
    """生成embeddings的API端点"""
    global current_generator, current_embeddings, current_labels

    try:
        # 清理之前的数据文件
        cleanup_data_file()

        data = request.json

        # 解析参数
        num_classes = data['num_classes']
        class_params = data['class_params']
        inter_class_distance = data['inter_class_distance']
        inter_hyperplane_parallelism = data.get('inter_hyperplane_parallelism', 0.0)
        embedding_dim = data.get('embedding_dim', 128)

        # 构建生成器参数
        n_samples_per_class = []
        dispersion_list = []
        curvature_list = []
        flatness_list = []
        correlation_list = []
        manifold_complexity_list = []
        feature_sparsity_list = []
        noise_level_list = []
        boundary_sharpness_list = []
        dimensional_anisotropy_list = []

        for i in range(num_classes):
            params = class_params[str(i)]
            n_samples_per_class.append(int(params['samples']))
            dispersion_list.append(float(params['dispersion']))
            curvature_list.append(float(params['curvature']))
            flatness_list.append(float(params['flatness']))
            correlation_list.append(float(params['correlation']))
            manifold_complexity_list.append(float(params.get('manifold', 0.2)))
            feature_sparsity_list.append(float(params.get('sparsity', 0.1)))
            noise_level_list.append(float(params.get('noise', 0.05)))
            boundary_sharpness_list.append(float(params.get('sharpness', 0.5)))
            dimensional_anisotropy_list.append(float(params.get('anisotropy', 0.3)))

        generator_params = {
            'n_samples_per_class': n_samples_per_class,
            'dispersion': dispersion_list,
            'curvature': curvature_list,
            'flatness': flatness_list,
            'inter_class_distance': float(inter_class_distance),
            'intra_class_correlation': correlation_list,
            'inter_hyperplane_parallelism': float(inter_hyperplane_parallelism),
            'manifold_complexity': manifold_complexity_list,
            'feature_sparsity': feature_sparsity_list,
            'noise_level': noise_level_list,
            'boundary_sharpness': boundary_sharpness_list,
            'dimensional_anisotropy': dimensional_anisotropy_list
        }

        # 生成embeddings
        current_generator = EmbeddingGenerator(embedding_dim=embedding_dim, random_state=42)
        current_embeddings, current_labels = current_generator.generate_clustered_embeddings(**generator_params)

        # 保存原始embeddings到HDF5文件
        save_embeddings_to_h5(current_embeddings, current_labels)

        # 降维到3D (PCA)
        reduced_3d = current_generator.reduce_dimensions(n_components=3, method='pca')

        # 保存PCA降维结果
        save_reduced_data_to_h5('pca', reduced_3d)

        # 创建3D散点图
        fig = create_3d_plot(reduced_3d, current_labels)

        # 统计信息
        stats = {
            'total_samples': len(current_labels),
            'num_classes': len(np.unique(current_labels)),
            'class_distribution': np.bincount(current_labels).tolist(),
            'embedding_dim': current_embeddings.shape[1],
            'embedding_range': f"[{current_embeddings.min():.3f}, {current_embeddings.max():.3f}]"
        }

        return jsonify({
            'success': True,
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'stats': stats
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


def save_embeddings_to_h5(embeddings, labels):
    """保存embeddings到HDF5文件"""
    with h5py.File(DATA_FILE, 'w') as f:
        f.create_dataset('embeddings', data=embeddings)
        f.create_dataset('labels', data=labels)
        print(f"已保存embeddings到: {DATA_FILE}")


def save_reduced_data_to_h5(method, reduced_data):
    """保存降维结果到HDF5文件"""
    with h5py.File(DATA_FILE, 'a') as f:
        if method in f:
            del f[method]  # 删除已存在的数据
        f.create_dataset(method, data=reduced_data)
        print(f"已保存{method.upper()}降维结果到: {DATA_FILE}")


def load_reduced_data_from_h5(method):
    """从HDF5文件加载降维结果"""
    if not os.path.exists(DATA_FILE):
        return None

    try:
        with h5py.File(DATA_FILE, 'r') as f:
            if method in f:
                return f[method][:]
    except Exception as e:
        print(f"加载{method}数据时出错: {e}")

    return None


def create_3d_plot(embeddings_3d, labels):
    """创建3D散点图"""
    traces = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    for i, class_id in enumerate(np.unique(labels)):
        mask = labels == class_id
        trace = go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            marker=dict(
                size=5,
                color=colors[i % len(colors)],
                opacity=0.7
            ),
            name=f'Class {class_id}',
            text=[f'Class {class_id}'] * np.sum(mask),
            hovertemplate='<b>%{text}</b><br>' +
            'X: %{x:.2f}<br>' +
            'Y: %{y:.2f}<br>' +
            'Z: %{z:.2f}<extra></extra>'
        )
        traces.append(trace)

    layout = go.Layout(
        title={
            'text': '3D Embeddings Visualization',
            'x': 0.5,
            'font': {'size': 20, 'color': 'white'}
        },
        scene=dict(
            xaxis=dict(title='Component 1', color='white'),
            yaxis=dict(title='Component 2', color='white'),
            zaxis=dict(title='Component 3', color='white'),
            bgcolor='rgba(0,0,0,0)',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)
            )
        ),
        paper_bgcolor='#1E1E1E',
        plot_bgcolor='#1E1E1E',
        font=dict(color='white'),
        legend=dict(
            bgcolor='rgba(0,0,0,0.5)',
            bordercolor='white',
            borderwidth=1
        ),
        margin=dict(l=0, r=0, t=50, b=0)
    )

    return {'data': traces, 'layout': layout}


@app.route('/api/methods')
def get_methods():
    """获取可用的降维方法"""
    return jsonify({
        'methods': ['pca', 'tsne', 'umap']
    })


@app.route('/api/reduce', methods=['POST'])
def reduce_dimensions():
    """降维API"""
    global current_generator, current_embeddings, current_labels

    if current_generator is None or current_embeddings is None:
        return jsonify({
            'success': False,
            'error': 'No embeddings generated yet'
        })

    try:
        data = request.json
        method = data.get('method', 'pca')
        n_components = data.get('n_components', 3)

        # 首先尝试从缓存加载
        reduced = load_reduced_data_from_h5(method)

        if reduced is None:
            # 缓存中没有，执行降维计算
            print(f"计算{method.upper()}降维...")
            reduced = current_generator.reduce_dimensions(n_components=n_components, method=method)

            # 保存降维结果到缓存
            save_reduced_data_to_h5(method, reduced)
        else:
            print(f"从缓存加载{method.upper()}降维结果")

        # 创建新的3D图
        fig = create_3d_plot(reduced, current_labels)

        return jsonify({
            'success': True,
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
            'method': method
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
