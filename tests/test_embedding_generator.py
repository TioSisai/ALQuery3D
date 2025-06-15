#!/usr/bin/env python3
"""
EmbeddingGenerator的单元测试
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import unittest
import numpy as np
from src.data.embedding_generator import EmbeddingGenerator, create_sample_dataset


class TestEmbeddingGenerator(unittest.TestCase):
    """EmbeddingGenerator测试类"""

    def setUp(self):
        """设置测试环境"""
        self.generator = EmbeddingGenerator(embedding_dim=32, random_state=42)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.generator.embedding_dim, 32)
        self.assertEqual(self.generator.random_state, 42)
        self.assertIsNone(self.generator.embeddings)
        self.assertIsNone(self.generator.labels)

    def test_basic_generation(self):
        """测试基本生成功能"""
        n_samples_per_class = [50, 60, 40]
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=n_samples_per_class,
            dispersion=0.5,
            curvature=0.2,
            flatness=0.7,
            inter_class_distance=0.6,
            intra_class_correlation=0.3
        )

        # 检查形状
        self.assertEqual(embeddings.shape, (150, 32))
        self.assertEqual(labels.shape, (150,))

        # 检查类别
        unique_labels = np.unique(labels)
        self.assertEqual(len(unique_labels), 3)

        # 检查每个类别的样本数量
        for i, expected_count in enumerate(n_samples_per_class):
            actual_count = np.sum(labels == i)
            self.assertEqual(actual_count, expected_count)

    def test_per_class_parameters(self):
        """测试每个类别单独设置参数"""
        n_samples_per_class = [30, 40, 35]
        dispersion_list = [0.2, 0.5, 0.8]
        curvature_list = [0.1, 0.3, 0.6]
        flatness_list = [0.3, 0.6, 0.9]
        correlation_list = [0.1, 0.4, 0.7]

        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=n_samples_per_class,
            dispersion=dispersion_list,
            curvature=curvature_list,
            flatness=flatness_list,
            inter_class_distance=0.5,
            intra_class_correlation=correlation_list
        )

        # 检查形状
        self.assertEqual(embeddings.shape, (105, 32))
        self.assertEqual(labels.shape, (105,))

        # 检查参数存储
        params = self.generator.generation_params
        self.assertEqual(params['dispersion'], dispersion_list)
        self.assertEqual(params['curvature'], curvature_list)
        self.assertEqual(params['flatness'], flatness_list)
        self.assertEqual(params['intra_class_correlation'], correlation_list)

    def test_parameter_validation(self):
        """测试参数验证"""
        n_samples_per_class = [30, 30]

        # 测试参数范围验证
        with self.assertRaises(ValueError):
            # dispersion超出范围
            self.generator.generate_clustered_embeddings(
                n_samples_per_class=n_samples_per_class,
                dispersion=1.5  # 超出0-1范围
            )

        with self.assertRaises(ValueError):
            # inter_class_distance超出范围
            self.generator.generate_clustered_embeddings(
                n_samples_per_class=n_samples_per_class,
                inter_class_distance=1.5  # 超出0-1范围
            )

        # 测试列表长度验证
        with self.assertRaises(ValueError):
            # 列表长度与类别数不匹配
            self.generator.generate_clustered_embeddings(
                n_samples_per_class=n_samples_per_class,  # 2个类别
                dispersion=[0.3, 0.5, 0.7]  # 3个值
            )

    def test_parameter_effects(self):
        """测试参数对生成结果的影响"""
        # 测试不同分散度
        embeddings1, labels1 = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[30, 30],
            dispersion=0.2,  # 低分散度
            curvature=0.1,
            flatness=0.8,
            inter_class_distance=0.5
        )

        generator2 = EmbeddingGenerator(embedding_dim=32, random_state=42)
        embeddings2, labels2 = generator2.generate_clustered_embeddings(
            n_samples_per_class=[30, 30],
            dispersion=0.8,  # 高分散度
            curvature=0.1,
            flatness=0.8,
            inter_class_distance=0.5
        )

        # 高分散度应该产生更大的类内方差
        var1 = np.var(embeddings1)
        var2 = np.var(embeddings2)
        self.assertLess(var1, var2)

    def test_dimensionality_reduction(self):
        """测试降维功能"""
        # 生成高维数据
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[50, 50],
            dispersion=0.5,
            curvature=0.2,
            flatness=0.7,
            inter_class_distance=0.6
        )

        # 测试PCA降维
        reduced_pca = self.generator.reduce_dimensions(n_components=3, method='pca')
        self.assertEqual(reduced_pca.shape, (100, 3))
        self.assertTrue(hasattr(self.generator, 'dimensionality_reduction_info'))

        # 测试t-SNE降维
        try:
            reduced_tsne = self.generator.reduce_dimensions(n_components=3, method='tsne')
            self.assertEqual(reduced_tsne.shape, (100, 3))
        except ImportError:
            # t-SNE在某些环境中可能不可用
            pass

        # 测试UMAP降维
        try:
            reduced_umap = self.generator.reduce_dimensions(n_components=3, method='umap')
            self.assertEqual(reduced_umap.shape, (100, 3))
        except ImportError:
            # UMAP可能未安装
            pass

        # 测试不支持的方法
        with self.assertRaises(ValueError):
            self.generator.reduce_dimensions(n_components=3, method='invalid_method')

    def test_statistics(self):
        """测试统计信息计算"""
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[40, 50, 30],
            dispersion=[0.3, 0.5, 0.7],
            curvature=0.2,
            flatness=0.8,
            inter_class_distance=0.6
        )

        stats = self.generator.get_class_statistics()

        # 检查统计信息结构
        self.assertIn('class_0', stats)
        self.assertIn('class_1', stats)
        self.assertIn('class_2', stats)
        self.assertIn('inter_class_distances', stats)

        # 检查类别统计
        self.assertEqual(stats['class_0']['n_samples'], 40)
        self.assertEqual(stats['class_1']['n_samples'], 50)
        self.assertEqual(stats['class_2']['n_samples'], 30)

    def test_save_load(self):
        """测试保存和加载功能"""
        # 生成数据
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[30, 40],
            dispersion=[0.3, 0.7],
            curvature=[0.1, 0.4],
            flatness=[0.5, 0.8],
            inter_class_distance=0.5,
            intra_class_correlation=[0.2, 0.6]
        )

        # 保存
        test_file = "test_embeddings.npz"
        self.generator.save_embeddings(test_file)

        # 加载到新的生成器
        new_generator = EmbeddingGenerator(embedding_dim=32)
        new_generator.load_embeddings(test_file)

        # 验证数据一致性
        np.testing.assert_array_equal(embeddings, new_generator.embeddings)
        np.testing.assert_array_equal(labels, new_generator.labels)

        # 验证参数一致性
        self.assertEqual(self.generator.generation_params, new_generator.generation_params)

        # 清理
        os.remove(test_file)

    def test_create_sample_dataset(self):
        """测试便捷函数"""
        embeddings, labels = create_sample_dataset(
            n_classes=4,
            samples_per_class=25,
            embedding_dim=16,
            random_state=42
        )

        self.assertEqual(embeddings.shape, (100, 16))
        self.assertEqual(labels.shape, (100,))
        self.assertEqual(len(np.unique(labels)), 4)

    def test_edge_cases(self):
        """测试边界情况"""
        # 单个类别
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[50],
            dispersion=0.5
        )
        self.assertEqual(len(np.unique(labels)), 1)

        # 零曲度
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[30, 30],
            curvature=0.0,
            dispersion=0.5
        )
        self.assertEqual(embeddings.shape, (60, 32))

        # 最大扁平度
        embeddings, labels = self.generator.generate_clustered_embeddings(
            n_samples_per_class=[30, 30],
            flatness=1.0,
            dispersion=0.5
        )
        self.assertEqual(embeddings.shape, (60, 32))

    def test_normalize_parameter_method(self):
        """测试参数标准化方法"""
        # 测试单个值
        result = self.generator._normalize_parameter(0.5, 3, 'test_param')
        self.assertEqual(result, [0.5, 0.5, 0.5])

        # 测试列表
        result = self.generator._normalize_parameter([0.2, 0.5, 0.8], 3, 'test_param')
        self.assertEqual(result, [0.2, 0.5, 0.8])

        # 测试错误情况
        with self.assertRaises(ValueError):
            # 超出范围
            self.generator._normalize_parameter(1.5, 3, 'test_param')

        with self.assertRaises(ValueError):
            # 列表长度不匹配
            self.generator._normalize_parameter([0.2, 0.5], 3, 'test_param')


def run_tests():
    """运行所有测试"""
    unittest.main()


if __name__ == "__main__":
    run_tests()
