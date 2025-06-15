#!/usr/bin/env python3
"""
ALQuery3D 新特性使用示例
展示如何使用新的神经网络encoder特性
"""

import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from data.embedding_generator import EmbeddingGenerator


def example_realistic_neural_network():
    """示例1: 模拟真实神经网络encoder"""
    print("=== 示例1: 模拟真实神经网络encoder ===")

    generator = EmbeddingGenerator(embedding_dim=256, random_state=42)

    # 模拟真实神经网络的参数设置
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[200, 180, 220, 160, 190],  # 5个类别，不同样本数

        # 基础几何属性
        dispersion=[0.4, 0.5, 0.6, 0.3, 0.7],          # 不同类别的分散度
        curvature=[0.2, 0.3, 0.1, 0.4, 0.2],           # 轻微的圆锥体效应
        flatness=[0.3, 0.4, 0.5, 0.2, 0.6],            # 中等扁平度
        inter_class_distance=0.6,                        # 适中的类间距离
        intra_class_correlation=[0.3, 0.4, 0.5, 0.2, 0.6],  # 不同的类内相关性
        inter_hyperplane_parallelism=0.2,               # 轻微的超平面平行度

        # 神经网络encoder特性
        manifold_complexity=[0.3, 0.2, 0.4, 0.3, 0.2],  # 适中的流形复杂度
        feature_sparsity=[0.15, 0.1, 0.2, 0.12, 0.18],  # ReLU导致的稀疏性
        noise_level=[0.08, 0.06, 0.1, 0.05, 0.09],     # 训练噪声
        boundary_sharpness=[0.6, 0.7, 0.5, 0.8, 0.6],  # 分类边界锐度
        dimensional_anisotropy=[0.3, 0.4, 0.2, 0.5, 0.3]  # 特征重要性差异
    )

    print(f"生成embeddings: {embeddings.shape}")
    print(f"类别分布: {np.bincount(labels)}")
    print(f"数据范围: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    # 降维可视化
    reduced = generator.reduce_dimensions(n_components=3, method='pca')
    print(f"PCA降维结果: {reduced.shape}")

    return embeddings, labels


def example_challenging_data():
    """示例2: 生成挑战性数据"""
    print("\n=== 示例2: 生成挑战性数据 ===")

    generator = EmbeddingGenerator(embedding_dim=128, random_state=123)

    # 挑战性参数设置
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[150, 150, 150],

        # 高难度几何属性
        dispersion=[0.8, 0.9, 0.7],                     # 高分散度
        curvature=[0.6, 0.7, 0.8],                      # 强圆锥体效应
        flatness=[0.8, 0.9, 0.7],                       # 高扁平度
        inter_class_distance=0.3,                        # 较小类间距离
        intra_class_correlation=[0.7, 0.8, 0.6],        # 高相关性
        inter_hyperplane_parallelism=0.8,               # 高平行度

        # 高难度神经网络特性
        manifold_complexity=[0.8, 0.7, 0.9],           # 高流形复杂度
        feature_sparsity=[0.6, 0.7, 0.5],              # 高稀疏性
        noise_level=[0.3, 0.4, 0.2],                   # 高噪声
        boundary_sharpness=[0.2, 0.1, 0.3],            # 模糊边界
        dimensional_anisotropy=[0.7, 0.8, 0.6]         # 高各向异性
    )

    print(f"生成embeddings: {embeddings.shape}")
    print(f"类别分布: {np.bincount(labels)}")
    print(f"数据范围: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    return embeddings, labels


def example_ideal_data():
    """示例3: 生成理想化数据"""
    print("\n=== 示例3: 生成理想化数据 ===")

    generator = EmbeddingGenerator(embedding_dim=64, random_state=456)

    # 理想化参数设置
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[100, 100, 100, 100],

        # 理想几何属性
        dispersion=[0.3, 0.3, 0.3, 0.3],               # 低分散度
        curvature=[0.1, 0.1, 0.1, 0.1],                # 轻微曲度
        flatness=[0.2, 0.2, 0.2, 0.2],                 # 低扁平度
        inter_class_distance=0.8,                       # 大类间距离
        intra_class_correlation=[0.2, 0.2, 0.2, 0.2],  # 低相关性
        inter_hyperplane_parallelism=0.0,              # 无平行度

        # 理想神经网络特性
        manifold_complexity=[0.1, 0.1, 0.1, 0.1],     # 低流形复杂度
        feature_sparsity=[0.05, 0.05, 0.05, 0.05],    # 低稀疏性
        noise_level=[0.02, 0.02, 0.02, 0.02],         # 低噪声
        boundary_sharpness=[0.8, 0.8, 0.8, 0.8],      # 锐利边界
        dimensional_anisotropy=[0.1, 0.1, 0.1, 0.1]   # 低各向异性
    )

    print(f"生成embeddings: {embeddings.shape}")
    print(f"类别分布: {np.bincount(labels)}")
    print(f"数据范围: [{embeddings.min():.3f}, {embeddings.max():.3f}]")

    return embeddings, labels


def example_3d_special_case():
    """示例4: 3D特殊情况"""
    print("\n=== 示例4: 3D特殊情况 ===")

    generator = EmbeddingGenerator(embedding_dim=3, random_state=789)

    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[80, 80, 80],
        dispersion=[0.5, 0.6, 0.4],
        curvature=[0.3, 0.4, 0.2],
        flatness=[0.6, 0.7, 0.5],
        inter_class_distance=0.7,
        intra_class_correlation=[0.4, 0.5, 0.3],
        inter_hyperplane_parallelism=0.5,
        manifold_complexity=[0.2, 0.3, 0.1],
        feature_sparsity=[0.1, 0.15, 0.05],
        noise_level=[0.05, 0.08, 0.03],
        boundary_sharpness=[0.6, 0.7, 0.5],
        dimensional_anisotropy=[0.3, 0.4, 0.2]
    )

    print(f"生成3D embeddings: {embeddings.shape}")
    print(f"类别分布: {np.bincount(labels)}")

    # 3D情况下的降维
    reduced = generator.reduce_dimensions(n_components=3, method='pca')
    print(f"3D降维结果: {reduced.shape}")
    print("注意: 3D数据直接返回原始坐标，无需降维")

    return embeddings, labels


def example_parameter_effects():
    """示例5: 参数效果对比"""
    print("\n=== 示例5: 参数效果对比 ===")

    generator = EmbeddingGenerator(embedding_dim=32, random_state=999)

    # 基准参数
    base_params = {
        'n_samples_per_class': [100, 100],
        'dispersion': [0.5, 0.5],
        'curvature': [0.3, 0.3],
        'flatness': [0.4, 0.4],
        'inter_class_distance': 0.5,
        'intra_class_correlation': [0.3, 0.3],
        'inter_hyperplane_parallelism': 0.2,
        'manifold_complexity': [0.2, 0.2],
        'feature_sparsity': [0.1, 0.1],
        'noise_level': [0.05, 0.05],
        'boundary_sharpness': [0.5, 0.5],
        'dimensional_anisotropy': [0.3, 0.3]
    }

    # 测试不同参数的效果
    effects = {}

    # 高分散度效果
    high_dispersion_params = base_params.copy()
    high_dispersion_params['dispersion'] = [0.9, 0.9]
    emb, lab = generator.generate_clustered_embeddings(**high_dispersion_params)
    effects['high_dispersion'] = np.std(emb)

    # 高曲度效果
    high_curvature_params = base_params.copy()
    high_curvature_params['curvature'] = [0.8, 0.8]
    emb, lab = generator.generate_clustered_embeddings(**high_curvature_params)
    effects['high_curvature'] = np.std(emb)

    # 高扁平度效果
    high_flatness_params = base_params.copy()
    high_flatness_params['flatness'] = [0.9, 0.9]
    emb, lab = generator.generate_clustered_embeddings(**high_flatness_params)
    effects['high_flatness'] = np.std(emb)

    # 高噪声效果
    high_noise_params = base_params.copy()
    high_noise_params['noise_level'] = [0.4, 0.4]
    emb, lab = generator.generate_clustered_embeddings(**high_noise_params)
    effects['high_noise'] = np.std(emb)

    print("参数效果对比 (标准差):")
    for effect, std_val in effects.items():
        print(f"  {effect}: {std_val:.4f}")

    return effects


def main():
    """主函数"""
    print("ALQuery3D 神经网络Encoder特性使用示例\n")

    # 运行所有示例
    example_realistic_neural_network()
    example_challenging_data()
    example_ideal_data()
    example_3d_special_case()
    example_parameter_effects()

    print("\n=== 使用建议 ===")
    print("1. 真实神经网络模拟: 使用适中的参数值，添加适量噪声和稀疏性")
    print("2. 挑战性数据生成: 使用高噪声、高复杂度、模糊边界")
    print("3. 理想化数据生成: 使用低噪声、锐利边界、简单几何")
    print("4. 3D可视化: 直接使用3D embedding，无需降维")
    print("5. 参数调优: 通过Web界面实时调整参数观察效果")

    print("\n=== Web界面访问 ===")
    print("启动Web服务器: cd src/web && python app.py")
    print("访问地址: http://localhost:5000")
    print("在Web界面中可以实时调整所有参数并观察可视化效果")


if __name__ == "__main__":
    main()
