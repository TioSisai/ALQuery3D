#!/usr/bin/env python3
"""
高维Embeddings生成器演示脚本
展示如何使用EmbeddingGenerator生成模拟神经网络encoder的高维embeddings
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from src.data.embedding_generator import EmbeddingGenerator, create_sample_dataset


def demo_basic_generation():
    """演示基本的embedding生成功能"""
    print("=== 基本Embedding生成演示 ===")

    # 创建生成器
    generator = EmbeddingGenerator(embedding_dim=64, random_state=42)

    # 生成5个类别的数据，每个类别有不同的样本数量
    n_samples_per_class = [80, 120, 100, 90, 110]

    # 生成embeddings（使用标准化的0-1参数）
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=n_samples_per_class,
        dispersion=0.6,           # 中等分散度
        curvature=0.15,           # 低曲度
        flatness=0.6,             # 中等扁平度
        inter_class_distance=0.6,  # 中等类间距离
        intra_class_correlation=0.35  # 中等类内相关性
    )

    print(f"生成的embeddings形状: {embeddings.shape}")
    print(f"类别数量: {len(np.unique(labels))}")
    print(f"每个类别的样本数: {[np.sum(labels == i) for i in np.unique(labels)]}")

    # 获取统计信息
    stats = generator.get_class_statistics()
    print("\n类间距离统计:")
    print(f"  平均距离: {stats['inter_class_distances']['mean']:.2f}")
    print(f"  最小距离: {stats['inter_class_distances']['min']:.2f}")
    print(f"  最大距离: {stats['inter_class_distances']['max']:.2f}")

    return generator, embeddings, labels


def demo_per_class_parameters():
    """演示每个类别单独设置参数的功能"""
    print("\n=== 每类别参数控制演示 ===")

    # 创建生成器
    generator = EmbeddingGenerator(embedding_dim=32, random_state=42)

    # 为每个类别设置不同的参数
    n_samples_per_class = [60, 80, 70, 90]

    # 不同类别有不同的特性
    dispersion_per_class = [0.3, 0.7, 0.5, 0.9]      # 紧密 -> 疏松
    curvature_per_class = [0.0, 0.2, 0.5, 0.8]       # 线性 -> 高度非线性
    flatness_per_class = [0.2, 0.5, 0.8, 1.0]        # 扁平 -> 均匀
    correlation_per_class = [0.1, 0.3, 0.6, 0.9]     # 低相关 -> 高相关

    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=n_samples_per_class,
        dispersion=dispersion_per_class,
        curvature=curvature_per_class,
        flatness=flatness_per_class,
        inter_class_distance=0.7,
        intra_class_correlation=correlation_per_class
    )

    print(f"生成的embeddings形状: {embeddings.shape}")

    # 分析各类别的特性
    stats = generator.get_class_statistics()
    print("\n各类别统计信息:")
    for i in range(len(n_samples_per_class)):
        class_stats = stats[f'class_{i}']
        print(f"  类别 {i}:")
        print(f"    样本数: {class_stats['n_samples']}")
        print(f"    平均类内距离: {class_stats['intra_class_distance']:.2f}")
        print(f"    平均标准差: {class_stats['mean_std']:.2f}")
        print(f"    参数设置: dispersion={dispersion_per_class[i]}, "
              f"curvature={curvature_per_class[i]}, "
              f"flatness={flatness_per_class[i]}, "
              f"correlation={correlation_per_class[i]}")

    return embeddings, labels


def demo_parameter_effects():
    """演示不同参数对生成结果的影响"""
    print("\n=== 参数影响演示 ===")

    # 测试不同的分散度
    print("\n1. 分散度效果测试 (0.0-1.0范围):")
    dispersions = [0.2, 0.5, 0.8]

    for disp in dispersions:
        generator = EmbeddingGenerator(embedding_dim=32, random_state=42)
        embeddings, labels = generator.generate_clustered_embeddings(
            n_samples_per_class=[50, 50, 50],
            dispersion=disp,
            curvature=0.1,
            flatness=0.8,
            inter_class_distance=0.5
        )

        # 计算类内平均距离
        intra_distances = []
        for class_id in np.unique(labels):
            class_data = embeddings[labels == class_id]
            center = np.mean(class_data, axis=0)
            distances = [np.linalg.norm(sample - center) for sample in class_data]
            intra_distances.append(np.mean(distances))

        print(f"  分散度 {disp}: 平均类内距离 = {np.mean(intra_distances):.2f}")

    # 测试不同的类间距离
    print("\n2. 类间距离效果测试:")
    inter_distances = [0.2, 0.5, 0.8]

    for inter_dist in inter_distances:
        generator = EmbeddingGenerator(embedding_dim=32, random_state=42)
        embeddings, labels = generator.generate_clustered_embeddings(
            n_samples_per_class=[40, 40, 40],
            dispersion=0.5,
            curvature=0.1,
            flatness=0.8,
            inter_class_distance=inter_dist
        )

        stats = generator.get_class_statistics()
        avg_inter_distance = stats['inter_class_distances']['mean']
        print(f"  参数值 {inter_dist}: 实际平均类间距离 = {avg_inter_distance:.2f}")


def demo_dimensionality_reduction():
    """演示降维功能"""
    print("\n=== 降维演示 ===")

    # 生成高维数据，每个类别有不同的特性
    generator = EmbeddingGenerator(embedding_dim=128, random_state=42)
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[100, 100, 100, 100],
        dispersion=[0.3, 0.5, 0.7, 0.9],     # 不同的分散度
        curvature=[0.1, 0.3, 0.5, 0.7],      # 不同的曲度
        flatness=[0.4, 0.6, 0.8, 1.0],       # 不同的扁平度
        inter_class_distance=0.7,
        intra_class_correlation=[0.2, 0.4, 0.6, 0.8]  # 不同的相关性
    )

    print(f"原始维度: {embeddings.shape[1]}")
    print(f"样本数量: {embeddings.shape[0]}")

    # 测试不同的降维方法
    methods = ['pca', 'tsne', 'umap']
    results = {}

    for method in methods:
        try:
            print(f"\n--- {method.upper()}降维 ---")
            reduced_3d = generator.reduce_dimensions(n_components=3, method=method)
            results[method] = reduced_3d

            print(f"降维后维度: {reduced_3d.shape[1]}")

            if hasattr(generator, 'dimensionality_reduction_info'):
                info = generator.dimensionality_reduction_info
                if method == 'pca':
                    print(f"前3个主成分解释的方差比例: {info['explained_variance_ratio']}")
                    print(f"累积方差比例: {info['cumulative_variance_ratio']}")
                elif method == 'tsne':
                    print(f"最终KL散度: {info['kl_divergence']:.4f}")
                    print(f"迭代次数: {info['n_iter_final']}")
                elif method == 'umap':
                    print(f"n_neighbors: {info['parameters']['n_neighbors']}")
                    print(f"min_dist: {info['parameters']['min_dist']}")

        except ImportError as e:
            print(f"{method.upper()}不可用: {e}")
            continue
        except Exception as e:
            print(f"{method.upper()}降维失败: {e}")
            continue

    # 返回PCA结果用于可视化（如果可用）
    if 'pca' in results:
        return results['pca'], labels
    elif results:
        # 返回第一个可用的结果
        first_method = list(results.keys())[0]
        return results[first_method], labels
    else:
        # 如果都失败了，返回None
        return None, labels


def visualize_3d_embeddings(embeddings_3d, labels, title="Embeddings in 3D Space"):
    """可视化3D embeddings"""
    try:
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # 为每个类别使用不同颜色
        colors = plt.cm.tab10(np.linspace(0, 1, len(np.unique(labels))))

        for i, class_id in enumerate(np.unique(labels)):
            mask = labels == class_id
            ax.scatter(embeddings_3d[mask, 0],
                       embeddings_3d[mask, 1],
                       embeddings_3d[mask, 2],
                       c=[colors[i]],
                       label=f'Class {class_id}',
                       alpha=0.7,
                       s=30)

        ax.set_xlabel('Component 1')
        ax.set_ylabel('Component 2')
        ax.set_zlabel('Component 3')
        ax.set_title(title)
        ax.legend()

        plt.tight_layout()
        plt.savefig('embeddings_3d_visualization.png', dpi=150, bbox_inches='tight')
        print(f"\n3D可视化已保存为 'embeddings_3d_visualization.png'")

    except Exception as e:
        print(f"可视化时出现错误: {e}")
        print("请确保安装了matplotlib")


def demo_save_load():
    """演示保存和加载功能"""
    print("\n=== 保存和加载演示 ===")

    # 生成数据，每个类别有不同参数
    generator = EmbeddingGenerator(embedding_dim=50, random_state=42)
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=[60, 80, 70],
        dispersion=[0.3, 0.6, 0.9],
        curvature=[0.1, 0.3, 0.5],
        flatness=[0.4, 0.7, 1.0],
        inter_class_distance=0.6,
        intra_class_correlation=[0.2, 0.5, 0.8]
    )

    # 保存数据
    save_path = "sample_embeddings.npz"
    generator.save_embeddings(save_path)
    print(f"Embeddings已保存到 {save_path}")

    # 加载数据
    new_generator = EmbeddingGenerator(embedding_dim=50)
    new_generator.load_embeddings(save_path)

    print(f"加载的embeddings形状: {new_generator.embeddings.shape}")
    print(f"加载的标签形状: {new_generator.labels.shape}")
    print(f"生成参数: {new_generator.generation_params}")

    # 验证数据一致性
    assert np.allclose(embeddings, new_generator.embeddings), "加载的embeddings不一致！"
    assert np.array_equal(labels, new_generator.labels), "加载的labels不一致！"
    print("✓ 数据保存和加载验证成功")

    # 清理临时文件
    os.remove(save_path)


def main():
    """主函数"""
    print("高维Embeddings生成器完整演示")
    print("=" * 50)

    try:
        # 1. 基本生成演示
        generator, embeddings, labels = demo_basic_generation()

        # 2. 每类别参数控制演示
        demo_per_class_parameters()

        # 3. 参数影响演示
        demo_parameter_effects()

        # 4. 降维演示
        reduced_3d, labels_3d = demo_dimensionality_reduction()

        # 5. 3D可视化
        if reduced_3d is not None:
            visualize_3d_embeddings(reduced_3d, labels_3d)
        else:
            print("跳过3D可视化（降维失败）")

        # 6. 保存加载演示
        demo_save_load()

        # 7. 便捷函数演示
        print("\n=== 便捷函数演示 ===")
        simple_embeddings, simple_labels = create_sample_dataset(
            n_classes=3,
            samples_per_class=50,
            embedding_dim=64
        )
        print(f"便捷函数生成的数据形状: {simple_embeddings.shape}")

        print("\n演示完成！")
        print("主要功能:")
        print("- ✓ 高维embeddings生成")
        print("- ✓ 每类别独立参数控制")
        print("- ✓ 标准化参数范围(0-1)")
        print("- ✓ 可控制的分散度、曲度、扁平度")
        print("- ✓ 类间和类内距离控制")
        print("- ✓ 降维可视化(PCA/t-SNE/UMAP)")
        print("- ✓ 数据保存和加载")
        print("- ✓ 统计信息计算")

    except ImportError as e:
        print(f"缺少必要的依赖包: {e}")
        print("请安装: pip install numpy scikit-learn matplotlib umap-learn")
    except Exception as e:
        print(f"演示过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
