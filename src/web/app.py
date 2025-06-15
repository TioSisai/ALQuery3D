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
from src.algorithms.fps import create_fps_sampler

app = Flask(__name__)
app.secret_key = 'alquery3d_secret_key'

# 全局变量存储当前生成器和数据
current_generator = None
current_embeddings = None
current_labels = None
current_reduction_method = 'pca'  # 当前降维方法

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

        # 重置全局变量
        current_generator = None
        current_embeddings = None
        current_labels = None
        current_reduction_method = 'pca'  # 重置为默认降维方法

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
        indices = np.where(mask)[0]  # 获取原始索引
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
            text=[f'Class {class_id} (Index: {idx})' for idx in indices],
            customdata=indices,  # 添加原始索引作为自定义数据
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


def create_3d_plot_with_fps(embeddings_3d, labels, fps_data=None):
    """创建带有FPS路径的3D散点图"""
    if fps_data is None:
        # 尝试从H5文件加载FPS数据
        fps_data = load_fps_results_from_h5()

    if fps_data is None:
        # 没有FPS数据，返回普通的3D图
        return create_3d_plot(embeddings_3d, labels)

    # 有FPS数据，创建带路径的可视化
    selected_indices = fps_data['selected_indices']
    return create_fps_visualization(embeddings_3d, labels, selected_indices)


@app.route('/api/methods')
def get_methods():
    """获取可用的降维方法"""
    return jsonify({
        'methods': ['pca', 'tsne', 'umap']
    })


@app.route('/api/reduce', methods=['POST'])
def reduce_dimensions():
    """降维API"""
    global current_generator, current_embeddings, current_labels, current_reduction_method

    if current_generator is None or current_embeddings is None:
        return jsonify({
            'success': False,
            'error': 'No embeddings generated yet'
        })

    try:
        data = request.json
        method = data.get('method', 'pca')
        n_components = data.get('n_components', 3)

        # 获取前端传递的当前范围状态
        current_range_start = data.get('current_range_start')
        current_range_end = data.get('current_range_end')
        fps_range_visible = data.get('fps_range_visible', False)

        # 更新当前降维方法
        current_reduction_method = method

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

        # 检查是否有FPS数据并保持范围状态
        fps_data = load_fps_results_from_h5()
        if fps_data is not None:
            selected_indices = fps_data['selected_indices']

            # 如果有有效的范围设置，直接创建范围可视化
            if (fps_range_visible and current_range_start is not None and current_range_end is not None and
                    1 <= current_range_start <= current_range_end <= len(selected_indices)):

                # 转换为0-based索引
                start_range_0based = current_range_start - 1
                end_range_0based = current_range_end - 1
                range_indices = selected_indices[start_range_0based:end_range_0based + 1]

                # 创建范围可视化
                fig = create_fps_range_visualization(
                    reduced, current_labels, selected_indices, range_indices, start_range_0based
                )

                # 计算范围统计信息
                fps_sampler = create_fps_sampler()
                range_stats = fps_sampler.get_path_statistics(
                    current_embeddings, range_indices, current_labels, fps_data['distance_metric']
                )

                return jsonify({
                    'success': True,
                    'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                    'method': method,
                    'has_fps': True,
                    'fps_data': {
                        'selected_indices': selected_indices,
                        'total_count': len(selected_indices)
                    },
                    'range_view': True,
                    'range_stats': range_stats
                })
            else:
                # 没有范围设置或范围无效，创建完整的FPS可视化
                fig = create_fps_visualization(reduced, current_labels, selected_indices)

                return jsonify({
                    'success': True,
                    'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                    'method': method,
                    'has_fps': True,
                    'fps_data': {
                        'selected_indices': selected_indices,
                        'total_count': len(selected_indices)
                    },
                    'range_view': False
                })
        else:
            # 没有FPS数据，创建普通的3D图
            fig = create_3d_plot_with_fps(reduced, current_labels)

            return jsonify({
                'success': True,
                'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder),
                'method': method,
                'has_fps': False
            })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/fps/start', methods=['POST'])
def start_fps():
    """开始FPS采样"""
    global current_embeddings, current_labels

    if current_embeddings is None:
        return jsonify({
            'success': False,
            'error': 'No embeddings generated yet'
        })

    try:
        data = request.json
        start_idx = int(data['start_idx'])
        num_samples = int(data['num_samples'])
        distance_metric = data.get('distance_metric', 'euclidean')

        # 验证参数
        if start_idx >= len(current_embeddings):
            return jsonify({
                'success': False,
                'error': f'Start index {start_idx} out of range [0, {len(current_embeddings) - 1}]'
            })

        if num_samples > len(current_embeddings):
            return jsonify({
                'success': False,
                'error': f'Number of samples {num_samples} exceeds total points {len(current_embeddings)}'
            })

        # 执行FPS采样
        fps_sampler = create_fps_sampler()
        selected_indices = fps_sampler.sample(
            current_embeddings,
            start_idx,
            num_samples,
            distance_metric
        )

        # 获取统计信息
        stats = fps_sampler.get_path_statistics(
            current_embeddings,
            selected_indices,
            current_labels,
            distance_metric
        )

        # 保存FPS结果到HDF5文件
        save_fps_results_to_h5(selected_indices, distance_metric, stats)

        # 获取降维后的坐标用于可视化
        reduced_data = load_reduced_data_from_h5(current_reduction_method)
        if reduced_data is None:
            return jsonify({
                'success': False,
                'error': 'No reduced data available for visualization'
            })

        # 创建包含FPS路径的可视化
        fig = create_fps_visualization(reduced_data, current_labels, selected_indices)

        return jsonify({
            'success': True,
            'selected_indices': selected_indices,
            'stats': stats,
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/fps/range', methods=['POST'])
def fps_range_view():
    """FPS范围查看"""
    try:
        data = request.json
        start_range = int(data['start_range'])
        end_range = int(data['end_range'])

        # 从HDF5文件加载FPS结果
        fps_data = load_fps_results_from_h5()
        if fps_data is None:
            return jsonify({
                'success': False,
                'error': 'No FPS results found'
            })

        selected_indices = fps_data['selected_indices']
        distance_metric = fps_data['distance_metric']

        # 验证范围
        if start_range < 0 or end_range >= len(selected_indices) or start_range > end_range:
            return jsonify({
                'success': False,
                'error': f'Invalid range [{start_range}, {end_range}]'
            })

        # 获取范围内的索引
        range_indices = selected_indices[start_range:end_range + 1]

        # 计算范围内的统计信息
        fps_sampler = create_fps_sampler()
        range_stats = fps_sampler.get_path_statistics(
            current_embeddings,
            range_indices,
            current_labels,
            distance_metric
        )

        # 获取降维后的坐标
        reduced_data = load_reduced_data_from_h5(current_reduction_method)

        # 创建范围可视化
        fig = create_fps_range_visualization(
            reduced_data,
            current_labels,
            selected_indices,
            range_indices,
            start_range
        )

        return jsonify({
            'success': True,
            'range_indices': range_indices,
            'range_stats': range_stats,
            'plot': json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
        })

    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })


@app.route('/api/fps/metrics')
def get_fps_metrics():
    """获取可用的距离度量方式"""
    fps_sampler = create_fps_sampler()
    return jsonify({
        'metrics': fps_sampler.get_available_metrics()
    })


def save_fps_results_to_h5(selected_indices, distance_metric, stats):
    """保存FPS结果到HDF5文件"""
    with h5py.File(DATA_FILE, 'a') as f:
        # 删除已存在的FPS数据
        if 'fps_results' in f:
            del f['fps_results']

        # 创建FPS结果组
        fps_group = f.create_group('fps_results')
        fps_group.create_dataset('selected_indices', data=selected_indices)
        fps_group.attrs['distance_metric'] = distance_metric
        fps_group.attrs['total_points'] = stats['total_points']
        fps_group.attrs['total_distance'] = stats['total_distance']

        # 保存类别分布
        for class_id, count in stats['class_distribution'].items():
            fps_group.attrs[f'class_{class_id}_count'] = count

        print(f"已保存FPS结果到: {DATA_FILE}")


def load_fps_results_from_h5():
    """从HDF5文件加载FPS结果"""
    if not os.path.exists(DATA_FILE):
        return None

    try:
        with h5py.File(DATA_FILE, 'r') as f:
            if 'fps_results' not in f:
                return None

            fps_group = f['fps_results']
            selected_indices = fps_group['selected_indices'][:].tolist()
            distance_metric = fps_group.attrs['distance_metric']

            return {
                'selected_indices': selected_indices,
                'distance_metric': distance_metric
            }
    except Exception as e:
        print(f"加载FPS结果时出错: {e}")
        return None


def create_fps_visualization(embeddings_3d, labels, selected_indices):
    """创建包含FPS路径的3D可视化"""
    traces = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    # 添加原始点云（保持正常透明度）
    for i, class_id in enumerate(np.unique(labels)):
        mask = labels == class_id
        indices = np.where(mask)[0]  # 获取原始索引
        trace = go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            marker=dict(
                size=4,
                color=colors[i % len(colors)],
                opacity=0.6  # 提高透明度，避免过暗
            ),
            name=f'Class {class_id}',
            text=[f'Class {class_id} (Index: {idx})' for idx in indices],
            customdata=indices,  # 添加原始索引作为自定义数据
            hovertemplate='<b>%{text}</b><br>' +
            'X: %{x:.2f}<br>' +
            'Y: %{y:.2f}<br>' +
            'Z: %{z:.2f}<extra></extra>'
        )
        traces.append(trace)

    # 添加FPS选中的点（亮青色，渐变亮度）
    if selected_indices:
        selected_coords = embeddings_3d[selected_indices]

        # 计算亮度渐变（从100%到10%）
        num_points = len(selected_indices)
        opacities = np.linspace(1.0, 0.1, num_points)

        # 为每个选中的点创建单独的trace以实现渐变效果
        for i, (idx, opacity) in enumerate(zip(selected_indices, opacities)):
            coord = embeddings_3d[idx]
            # 获取该点的类别
            point_class = labels[idx]

            # 使用RGB格式确保颜色正确显示
            rgb_color = f'rgb(0, 255, 255)'

            trace = go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(
                    size=8,
                    color=rgb_color,
                    opacity=opacity,  # 单独设置透明度
                    line=dict(width=2, color='white')
                ),
                name=f'FPS Point {i + 1}' if i == 0 else '',
                showlegend=(i == 0),
                text=[f'Class {point_class} (FPS Rank {i + 1}, Index: {idx})'],
                customdata=[idx],  # 添加原始索引作为数组
                hovertemplate='<b>Class ' + str(point_class) + ' (FPS Rank ' + str(i + 1) + ')</b><br>' +
                'Index: %{customdata}<br>' +
                'X: %{x:.2f}<br>' +
                'Y: %{y:.2f}<br>' +
                'Z: %{z:.2f}<extra></extra>'
            )
            traces.append(trace)

        # 添加路径连线（亮青色，渐变亮度）
        for i in range(len(selected_indices) - 1):
            start_coord = embeddings_3d[selected_indices[i]]
            end_coord = embeddings_3d[selected_indices[i + 1]]

            # 路径亮度也渐变
            path_opacity = opacities[i]

            path_trace = go.Scatter3d(
                x=[start_coord[0], end_coord[0]],
                y=[start_coord[1], end_coord[1]],
                z=[start_coord[2], end_coord[2]],
                mode='lines',
                line=dict(
                    width=4,
                    color='rgb(0, 255, 255)'
                ),
                opacity=path_opacity,  # 单独设置透明度
                name='FPS Path' if i == 0 else '',
                showlegend=(i == 0),
                hoverinfo='skip'
            )
            traces.append(path_trace)

    layout = go.Layout(
        title={
            'text': f'FPS Visualization ({len(selected_indices)} points selected)',
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


def create_fps_range_visualization(embeddings_3d, labels, all_selected_indices, range_indices, start_range):
    """创建FPS范围可视化"""
    traces = []
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7',
              '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

    # 添加原始点云（半透明）
    for i, class_id in enumerate(np.unique(labels)):
        mask = labels == class_id
        indices = np.where(mask)[0]  # 获取原始索引
        trace = go.Scatter3d(
            x=embeddings_3d[mask, 0],
            y=embeddings_3d[mask, 1],
            z=embeddings_3d[mask, 2],
            mode='markers',
            marker=dict(
                size=3,
                color=colors[i % len(colors)],
                opacity=0.2
            ),
            name=f'Class {class_id}',
            text=[f'Class {class_id} (Index: {idx})' for idx in indices],
            customdata=indices,  # 添加原始索引作为自定义数据
            hovertemplate='<b>%{text}</b><br>' +
            'X: %{x:.2f}<br>' +
            'Y: %{y:.2f}<br>' +
            'Z: %{z:.2f}<extra></extra>'
        )
        traces.append(trace)

    # 添加范围左侧的已选择点（橙色）
    if start_range > 0:
        left_indices = all_selected_indices[:start_range]
        for i, idx in enumerate(left_indices):
            coord = embeddings_3d[idx]
            # 获取该点的类别
            point_class = labels[idx]
            fps_rank = i + 1  # 在FPS路径中的排名

            trace = go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(
                    size=6,
                    color='orange',
                    opacity=0.7
                ),
                name='Previous Points' if i == 0 else '',
                showlegend=(i == 0),
                text=[f'Class {point_class} (FPS Rank {fps_rank}, Index: {idx})'],
                customdata=[idx],  # 添加原始索引作为数组
                hovertemplate='<b>Class ' + str(point_class) + ' (FPS Rank ' + str(fps_rank) + ')</b><br>' +
                'Index: %{customdata}<br>' +
                'X: %{x:.2f}<br>' +
                'Y: %{y:.2f}<br>' +
                'Z: %{z:.2f}<extra></extra>'
            )
            traces.append(trace)

    # 添加范围内的点（亮青色，渐变，更大更不透明）
    if range_indices:
        num_range_points = len(range_indices)
        opacities = np.linspace(1.0, 0.1, num_range_points)

        for i, (idx, opacity) in enumerate(zip(range_indices, opacities)):
            coord = embeddings_3d[idx]
            # 获取该点的类别
            point_class = labels[idx]
            # 计算在整个FPS路径中的排名
            fps_rank = start_range + i + 1

            trace = go.Scatter3d(
                x=[coord[0]],
                y=[coord[1]],
                z=[coord[2]],
                mode='markers',
                marker=dict(
                    size=10,  # 更大的点
                    color='rgb(0, 255, 255)',
                    opacity=max(opacity, 0.5),  # 单独设置透明度，确保可见性
                    line=dict(width=2, color='white')
                ),
                name=f'Range Points' if i == 0 else '',
                showlegend=(i == 0),
                text=[f'Class {point_class} (FPS Rank {fps_rank}, Index: {idx})'],
                customdata=[idx],  # 添加原始索引作为数组
                hovertemplate='<b>Class ' + str(point_class) + ' (FPS Rank ' + str(fps_rank) + ')</b><br>' +
                'Index: %{customdata}<br>' +
                'X: %{x:.2f}<br>' +
                'Y: %{y:.2f}<br>' +
                'Z: %{z:.2f}<extra></extra>'
            )
            traces.append(trace)

        # 添加范围内的路径连线
        for i in range(len(range_indices) - 1):
            start_coord = embeddings_3d[range_indices[i]]
            end_coord = embeddings_3d[range_indices[i + 1]]

            path_opacity = max(opacities[i], 0.5)

            path_trace = go.Scatter3d(
                x=[start_coord[0], end_coord[0]],
                y=[start_coord[1], end_coord[1]],
                z=[start_coord[2], end_coord[2]],
                mode='lines',
                line=dict(
                    width=6,  # 更粗的线
                    color='rgb(0, 255, 255)'
                ),
                opacity=path_opacity,  # 单独设置透明度
                name='Range Path' if i == 0 else '',
                showlegend=(i == 0),
                hoverinfo='skip'
            )
            traces.append(path_trace)

    layout = go.Layout(
        title={
            'text': f'FPS Range View ({len(range_indices)} points in range)',
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


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
