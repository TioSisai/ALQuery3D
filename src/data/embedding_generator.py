"""
高维Embeddings生成器
模拟神经网络encoder生成的高维embeddings，具有不同类别的聚类特性
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Union
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings


class EmbeddingGenerator:
    """
    高维embeddings生成器

    该类用于生成模拟神经网络encoder输出的高维embeddings，
    具有以下特性：
    - 不同类别的样本围绕各自的中心聚集
    - 可控制的分散度（散布程度）
    - 可控制的曲度（非线性变形）
    - 可控制的扁平度（在某些维度上的压缩）
    """

    def __init__(self,
                 embedding_dim: int = 128,
                 random_state: Optional[int] = None):
        """
        初始化embeddings生成器

        Args:
            embedding_dim: 生成的embedding维度
            random_state: 随机种子，用于复现结果
        """
        self.embedding_dim = embedding_dim
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)

        # 存储生成的数据
        self.embeddings = None
        self.labels = None
        self.class_centers = None
        self.generation_params = None

    def generate_clustered_embeddings(self,
                                      n_samples_per_class: List[int],
                                      n_classes: Optional[int] = None,
                                      dispersion: Union[float, List[float]] = 0.5,
                                      curvature: Union[float, List[float]] = 0.1,
                                      flatness: Union[float, List[float]] = 0.8,
                                      inter_class_distance: float = 0.5,
                                      intra_class_correlation: Union[float, List[float]] = 0.3,
                                      inter_hyperplane_parallelism: float = 0.0,
                                      manifold_complexity: Union[float, List[float]] = 0.2,
                                      feature_sparsity: Union[float, List[float]] = 0.1,
                                      noise_level: Union[float, List[float]] = 0.05,
                                      boundary_sharpness: Union[float, List[float]] = 0.5,
                                      dimensional_anisotropy: Union[float, List[float]] = 0.3,
                                      embedding_dim: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成具有聚类特性的高维embeddings，模拟神经网络encoder的特性

        Args:
            n_samples_per_class: 每个类别的样本数量列表
            n_classes: 类别数量，如果为None则从n_samples_per_class推断

            # 基础几何属性
            dispersion: 分散度，控制到类中心距离的缩放因子 (0.0-1.0)
            curvature: 曲度，控制圆锥体分布程度 (0.0-1.0)
            flatness: 扁平度，控制靠近超平面的程度 (0.0-1.0)
            inter_class_distance: 类间距离，控制类中心间距离缩放 (0.0-1.0)
            intra_class_correlation: 类内相关性，高斯分布的sigma值 (0.0-1.0)
            inter_hyperplane_parallelism: 类间超平面平行度 (0.0-1.0)

            # 神经网络encoder特性
            manifold_complexity: 流形复杂度，控制局部非线性结构 (0.0-1.0)
            feature_sparsity: 特征稀疏性，控制激活维度的稀疏程度 (0.0-1.0)
            noise_level: 噪声水平，模拟编码过程中的信息损失 (0.0-1.0)
            boundary_sharpness: 边界锐度，控制类别边界的清晰程度 (0.0-1.0)
            dimensional_anisotropy: 维度各向异性，不同维度的重要性差异 (0.0-1.0)

            embedding_dim: embedding维度 (3-2048)

        Returns:
            embeddings: 生成的高维embeddings (n_samples, embedding_dim)
            labels: 对应的类别标签 (n_samples,)
        """
        if n_classes is None:
            n_classes = len(n_samples_per_class)

        # 设置embedding维度
        if embedding_dim is not None:
            if not (3 <= embedding_dim <= 2048):
                raise ValueError(f"embedding_dim必须在3-2048之间，当前值: {embedding_dim}")
            self.embedding_dim = embedding_dim

        # 将参数标准化为列表形式
        dispersion_list = self._normalize_parameter(dispersion, n_classes, 'dispersion')
        curvature_list = self._normalize_parameter(curvature, n_classes, 'curvature')
        flatness_list = self._normalize_parameter(flatness, n_classes, 'flatness')
        correlation_list = self._normalize_parameter(intra_class_correlation, n_classes, 'intra_class_correlation')
        manifold_complexity_list = self._normalize_parameter(manifold_complexity, n_classes, 'manifold_complexity')
        feature_sparsity_list = self._normalize_parameter(feature_sparsity, n_classes, 'feature_sparsity')
        noise_level_list = self._normalize_parameter(noise_level, n_classes, 'noise_level')
        boundary_sharpness_list = self._normalize_parameter(boundary_sharpness, n_classes, 'boundary_sharpness')
        dimensional_anisotropy_list = self._normalize_parameter(
            dimensional_anisotropy, n_classes, 'dimensional_anisotropy')

        # 验证全局参数
        if not (0.0 <= inter_class_distance <= 1.0):
            raise ValueError(f"inter_class_distance必须在0.0-1.0之间，当前值: {inter_class_distance}")
        if not (0.0 <= inter_hyperplane_parallelism <= 1.0):
            raise ValueError(f"inter_hyperplane_parallelism必须在0.0-1.0之间，当前值: {inter_hyperplane_parallelism}")

        # 映射参数到实际范围
        dispersion_mapped = self._map_parameter_to_range(dispersion_list, "dispersion")
        curvature_mapped = self._map_parameter_to_range(curvature_list, "curvature")
        flatness_mapped = self._map_parameter_to_range(flatness_list, "flatness")
        correlation_mapped = self._map_parameter_to_range(correlation_list, "intra_class_correlation")
        manifold_complexity_mapped = self._map_parameter_to_range(manifold_complexity_list, "manifold_complexity")
        feature_sparsity_mapped = self._map_parameter_to_range(feature_sparsity_list, "feature_sparsity")
        noise_level_mapped = self._map_parameter_to_range(noise_level_list, "noise_level")
        boundary_sharpness_mapped = self._map_parameter_to_range(boundary_sharpness_list, "boundary_sharpness")
        dimensional_anisotropy_mapped = self._map_parameter_to_range(
            dimensional_anisotropy_list, "dimensional_anisotropy")
        inter_distance_mapped = self._map_parameter_to_range([inter_class_distance], "inter_class_distance")[0]
        parallelism_mapped = self._map_parameter_to_range(
            [inter_hyperplane_parallelism], "inter_hyperplane_parallelism")[0]

        # 保存生成参数
        self.generation_params = {
            'n_samples_per_class': n_samples_per_class,
            'n_classes': n_classes,
            'dispersion': dispersion_list,
            'curvature': curvature_list,
            'flatness': flatness_list,
            'inter_class_distance': inter_class_distance,
            'intra_class_correlation': correlation_list,
            'inter_hyperplane_parallelism': inter_hyperplane_parallelism,
            'manifold_complexity': manifold_complexity_list,
            'feature_sparsity': feature_sparsity_list,
            'noise_level': noise_level_list,
            'boundary_sharpness': boundary_sharpness_list,
            'dimensional_anisotropy': dimensional_anisotropy_list
        }

        # 生成类别中心和超平面
        self.class_centers = self._generate_class_centers(n_classes, inter_distance_mapped)
        self.class_hyperplanes = self._generate_class_hyperplanes(n_classes, parallelism_mapped)

        # 生成每个类别的样本
        all_embeddings = []
        all_labels = []

        for class_idx, n_samples in enumerate(n_samples_per_class):
            class_embeddings = self._generate_class_embeddings(
                class_idx, n_samples,
                dispersion_mapped[class_idx],
                curvature_mapped[class_idx],
                flatness_mapped[class_idx],
                correlation_mapped[class_idx],
                manifold_complexity_mapped[class_idx],
                feature_sparsity_mapped[class_idx],
                noise_level_mapped[class_idx],
                boundary_sharpness_mapped[class_idx],
                dimensional_anisotropy_mapped[class_idx]
            )
            all_embeddings.append(class_embeddings)
            all_labels.extend([class_idx] * n_samples)

        self.embeddings = np.vstack(all_embeddings)
        self.labels = np.array(all_labels)

        # 标准化embeddings到-1~1范围
        self.embeddings = self._standardize_embeddings(self.embeddings)

        return self.embeddings, self.labels

    def _standardize_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        将embeddings标准化到-1~1范围

        Args:
            embeddings: 原始embeddings

        Returns:
            标准化后的embeddings
        """
        # 计算每个维度的最小值和最大值
        min_vals = np.min(embeddings, axis=0)
        max_vals = np.max(embeddings, axis=0)

        # 避免除零错误
        ranges = max_vals - min_vals
        ranges[ranges == 0] = 1.0

        # 标准化到-1~1范围
        standardized = 2 * (embeddings - min_vals) / ranges - 1

        return standardized

    def _normalize_parameter(self, param: Union[float, List[float]], n_classes: int, param_name: str) -> List[float]:
        """
        将参数标准化为列表形式并验证范围

        Args:
            param: 参数值，可以是单个值或列表
            n_classes: 类别数量
            param_name: 参数名称（用于错误信息）

        Returns:
            标准化后的参数列表
        """
        if isinstance(param, (int, float)):
            # 单个值，应用到所有类别
            param_list = [float(param)] * n_classes
        elif isinstance(param, list):
            # 列表形式
            if len(param) != n_classes:
                raise ValueError(f"{param_name}列表长度({len(param)})必须等于类别数量({n_classes})")
            param_list = [float(p) for p in param]
        else:
            raise ValueError(f"{param_name}必须是数值或数值列表")

        # 验证参数范围
        for i, p in enumerate(param_list):
            if not (0.0 <= p <= 1.0):
                raise ValueError(f"{param_name}[{i}]必须在0.0-1.0之间，当前值: {p}")

        return param_list

    def _map_parameter_to_range(self, normalized_values: List[float], param_name: str) -> List[float]:
        """
        将0-1范围的参数映射到实际使用范围

        Args:
            normalized_values: 0-1范围的参数值列表
            param_name: 参数名称

        Returns:
            list: 映射后的参数值列表
        """
        mapped_values = []

        for val in normalized_values:
            if param_name == 'dispersion':
                # 分散度: 0 -> 0.1 (最小缩放), 1 -> 5.0 (最大缩放)
                mapped_val = 0.1 + val * (5.0 - 0.1)
            elif param_name == 'curvature':
                # 曲度: 0 -> 0.0 (无圆锥), 1 -> 2.0 (强圆锥)
                mapped_val = val * 2.0
            elif param_name == 'flatness':
                # 扁平度: 0 -> 0.0 (不靠近超平面), 1 -> 1.0 (完全贴近超平面)
                mapped_val = val * 1.0
            elif param_name == 'intra_class_correlation':
                # 类内相关性(sigma): 0 -> 0.1 (低相关), 1 -> 3.0 (高相关)
                mapped_val = 0.1 + val * (3.0 - 0.1)
            elif param_name == 'inter_class_distance':
                # 类间距离: 0 -> 0.1, 1 -> 50.0 (非常大的距离)
                mapped_val = 0.1 + val * (50.0 - 0.1)
            elif param_name == 'inter_hyperplane_parallelism':
                # 类间超平面平行度: 0 -> 0.0 (随机方向), 1 -> 0.99 (几乎平行)
                mapped_val = val * 0.99
            elif param_name == 'manifold_complexity':
                # 流形复杂度: 0 -> 0.0 (线性), 1 -> 2.0 (高度非线性)
                mapped_val = val * 2.0
            elif param_name == 'feature_sparsity':
                # 特征稀疏性: 0 -> 0.0 (密集), 1 -> 0.9 (高度稀疏)
                mapped_val = val * 0.9
            elif param_name == 'noise_level':
                # 噪声水平: 0 -> 0.0 (无噪声), 1 -> 0.5 (高噪声)
                mapped_val = val * 0.5
            elif param_name == 'boundary_sharpness':
                # 边界锐度: 0 -> 0.0 (模糊边界), 1 -> 5.0 (锐利边界)
                mapped_val = val * 5.0
            elif param_name == 'dimensional_anisotropy':
                # 维度各向异性: 0 -> 0.0 (各向同性), 1 -> 0.8 (高度各向异性)
                mapped_val = val * 0.8
            else:
                mapped_val = val

            mapped_values.append(mapped_val)

        return mapped_values

    def _generate_class_centers(self, n_classes: int, inter_class_distance: float) -> np.ndarray:
        """生成类别中心点"""
        # inter_class_distance已经是映射后的实际距离
        actual_distance = inter_class_distance

        centers = []

        for i in range(n_classes):
            if i == 0:
                # 第一个中心设在原点附近
                center = self.rng.normal(0, 1, self.embedding_dim)
            else:
                # 后续中心要与已有中心保持一定距离
                max_attempts = 100
                for attempt in range(max_attempts):
                    candidate = self.rng.normal(0, actual_distance, self.embedding_dim)

                    # 检查与现有中心的距离
                    distances = [np.linalg.norm(candidate - existing)
                                 for existing in centers]

                    if all(d >= actual_distance * 0.8 for d in distances):
                        center = candidate
                        break
                else:
                    # 如果无法找到合适位置，使用随机位置
                    center = self.rng.normal(0, actual_distance, self.embedding_dim)

            centers.append(center)

        return np.array(centers)

    def _generate_class_hyperplanes(self, n_classes: int, inter_hyperplane_parallelism: float) -> np.ndarray:
        """生成类间超平面，支持平行度控制"""
        planes = []

        # 生成第一个随机超平面法向量
        first_plane = self.rng.normal(0, 1, self.embedding_dim)
        first_plane = first_plane / np.linalg.norm(first_plane)  # 标准化
        planes.append(first_plane)

        for i in range(1, n_classes):
            if inter_hyperplane_parallelism == 0:
                # 完全随机方向
                plane = self.rng.normal(0, 1, self.embedding_dim)
                plane = plane / np.linalg.norm(plane)
            else:
                # 根据平行度生成新的超平面
                # 选择一个已有的超平面作为参考
                reference_plane = planes[self.rng.choice(len(planes))]

                # 生成一个随机向量
                random_vector = self.rng.normal(0, 1, self.embedding_dim)
                random_vector = random_vector / np.linalg.norm(random_vector)

                # 根据平行度混合参考平面和随机向量
                plane = (inter_hyperplane_parallelism * reference_plane +
                         (1 - inter_hyperplane_parallelism) * random_vector)
                plane = plane / np.linalg.norm(plane)

            planes.append(plane)

        return np.array(planes)

    def _generate_class_embeddings(self,
                                   class_idx: int,
                                   n_samples: int,
                                   dispersion: float,
                                   curvature: float,
                                   flatness: float,
                                   intra_class_correlation: float,
                                   manifold_complexity: float,
                                   feature_sparsity: float,
                                   noise_level: float,
                                   boundary_sharpness: float,
                                   dimensional_anisotropy: float) -> np.ndarray:
        """为单个类别生成embeddings"""
        center = self.class_centers[class_idx]
        plane = self.class_hyperplanes[class_idx]

        # 1. 生成基础的高斯分布样本（使用sigma作为相关性）
        base_samples = self.rng.multivariate_normal(
            mean=np.zeros(self.embedding_dim),
            cov=self._generate_covariance_matrix(intra_class_correlation),
            size=n_samples
        )

        # 2. 应用分散度缩放（到类中心距离的缩放因子）
        distances_to_center = np.linalg.norm(base_samples, axis=1, keepdims=True)
        dispersed_samples = base_samples * dispersion

        # 3. 应用曲度变换（圆锥体分布）
        curved_samples = self._apply_curvature(dispersed_samples, curvature)

        # 4. 应用扁平度变换（靠近超平面）
        flattened_samples = self._apply_flatness(curved_samples, flatness, plane)

        # 5. 平移到类别中心
        final_samples = flattened_samples + center

        # 6. 应用流形复杂度变换
        curved_samples = self._apply_manifold_complexity(final_samples, manifold_complexity)

        # 7. 应用特征稀疏性变换
        sparse_samples = self._apply_feature_sparsity(curved_samples, feature_sparsity)

        # 8. 应用噪声水平变换
        noisy_samples = self._apply_noise_level(sparse_samples, noise_level)

        # 9. 应用边界锐度变换
        sharpened_samples = self._apply_boundary_sharpness(noisy_samples, boundary_sharpness)

        # 10. 应用维度各向异性变换
        anisotropic_samples = self._apply_dimensional_anisotropy(sharpened_samples, dimensional_anisotropy)

        # 11. 平移到超平面
        final_samples = anisotropic_samples + plane

        return final_samples

    def _generate_covariance_matrix(self, correlation: float) -> np.ndarray:
        """生成协方差矩阵"""
        # 创建具有指定相关性的协方差矩阵
        cov = np.eye(self.embedding_dim) * (1 - correlation)

        # 添加一些随机的相关性
        if correlation > 0:
            # 随机选择一些维度对添加相关性
            n_correlated_pairs = max(1, int(self.embedding_dim * correlation * 0.1))

            for _ in range(n_correlated_pairs):
                i, j = self.rng.choice(self.embedding_dim, 2, replace=False)
                corr_value = self.rng.uniform(0.3, 0.7) * correlation
                cov[i, j] = cov[j, i] = corr_value

        return cov

    def _apply_flatness(self, samples: np.ndarray, flatness: float, hyperplane: np.ndarray) -> np.ndarray:
        """应用扁平度变换，使数据靠近超平面"""
        if flatness <= 0:
            return samples

        flattened = samples.copy()

        # 标准化超平面法向量
        plane_normal = hyperplane / (np.linalg.norm(hyperplane) + 1e-8)

        # 计算每个点到超平面的距离
        distances_to_plane = np.dot(flattened, plane_normal)

        # 根据扁平度将点向超平面压缩
        compression_factor = 1.0 - flatness
        compressed_distances = distances_to_plane * compression_factor

        # 重新计算点的位置
        flattened = flattened - np.outer(distances_to_plane - compressed_distances, plane_normal)

        return flattened

    def _apply_curvature(self, samples: np.ndarray, curvature: float) -> np.ndarray:
        """应用曲度变换，形成圆锥体分布"""
        if curvature <= 0:
            return samples

        curved = samples.copy()

        # 计算每个点到原点的距离
        distances = np.linalg.norm(curved, axis=1, keepdims=True)

        # 应用圆锥体变换：距离越远，变形越大
        cone_factor = 1.0 + curvature * distances

        # 在垂直于主轴的方向上应用圆锥变形
        # 选择一个主轴方向（第一个维度）
        main_axis = np.zeros(self.embedding_dim)
        main_axis[0] = 1.0

        # 计算垂直于主轴的分量
        projection_on_axis = np.dot(curved, main_axis).reshape(-1, 1) * main_axis
        perpendicular_component = curved - projection_on_axis

        # 对垂直分量应用圆锥缩放
        curved = projection_on_axis + perpendicular_component * cone_factor

        return curved

    def _apply_manifold_complexity(self, samples: np.ndarray, manifold_complexity: float) -> np.ndarray:
        """应用流形复杂度变换"""
        if manifold_complexity <= 0:
            return samples

        curved = samples.copy()

        # 应用不同类型的非线性变换
        for i in range(self.embedding_dim):
            if self.rng.random() < manifold_complexity:
                # 随机选择变换类型
                transform_type = self.rng.choice(['quadratic', 'sine', 'tanh'])

                if transform_type == 'quadratic':
                    # 二次变换
                    curved[:, i] = samples[:, i] + manifold_complexity * 0.1 * np.square(samples[:, i])
                elif transform_type == 'sine':
                    # 正弦变换
                    curved[:, i] = samples[:, i] + manifold_complexity * 0.5 * np.sin(samples[:, i])
                elif transform_type == 'tanh':
                    # 双曲正切变换
                    curved[:, i] = samples[:, i] + manifold_complexity * 0.3 * np.tanh(samples[:, i])

        return curved

    def _apply_feature_sparsity(self, samples: np.ndarray, feature_sparsity: float) -> np.ndarray:
        """应用特征稀疏性变换"""
        if feature_sparsity <= 0:
            return samples

        sparse = samples.copy()

        # 随机选择一些维度进行稀疏化
        n_dims_to_sparse = max(1, int(self.embedding_dim * (1 - feature_sparsity)))
        dims_to_sparse = self.rng.choice(
            self.embedding_dim, n_dims_to_sparse, replace=False
        )

        sparse[:, dims_to_sparse] = 0

        return sparse

    def _apply_noise_level(self, samples: np.ndarray, noise_level: float) -> np.ndarray:
        """应用噪声水平变换"""
        if noise_level <= 0:
            return samples

        noisy = samples.copy()

        # 添加高斯噪声
        noisy += self.rng.normal(0, noise_level, samples.shape)

        return noisy

    def _apply_boundary_sharpness(self, samples: np.ndarray, boundary_sharpness: float) -> np.ndarray:
        """应用边界锐度变换，控制类别边界的清晰程度"""
        if boundary_sharpness <= 0:
            return samples

        sharpened = samples.copy()

        # 计算到类中心的距离
        center = np.mean(sharpened, axis=0)
        distances = np.linalg.norm(sharpened - center, axis=1)

        # 根据边界锐度调整距离分布
        # 高锐度 -> 更集中的分布，低锐度 -> 更分散的分布
        median_distance = np.median(distances)

        # 应用非线性变换来调整边界锐度
        adjusted_distances = distances * np.power(distances / median_distance, boundary_sharpness - 1)

        # 重新计算点的位置
        directions = (sharpened - center) / (distances.reshape(-1, 1) + 1e-8)
        sharpened = center + directions * adjusted_distances.reshape(-1, 1)

        return sharpened

    def _apply_dimensional_anisotropy(self, samples: np.ndarray, dimensional_anisotropy: float) -> np.ndarray:
        """应用维度各向异性变换"""
        if dimensional_anisotropy <= 0:
            return samples

        anisotropic = samples.copy()

        # 随机选择一些维度进行缩放
        n_dims_to_scale = max(1, int(self.embedding_dim * (1 - dimensional_anisotropy)))
        dims_to_scale = self.rng.choice(
            self.embedding_dim, n_dims_to_scale, replace=False
        )

        scale_factors = self.rng.uniform(0.5, 2.0, len(dims_to_scale))
        anisotropic[:, dims_to_scale] *= scale_factors

        return anisotropic

    def get_class_statistics(self) -> Dict[str, Any]:
        """获取各类别的统计信息"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")

        stats = {}
        unique_labels = np.unique(self.labels)

        for label in unique_labels:
            mask = self.labels == label
            class_embeddings = self.embeddings[mask]

            stats[f'class_{label}'] = {
                'n_samples': np.sum(mask),
                'center': np.mean(class_embeddings, axis=0),
                'std': np.std(class_embeddings, axis=0),
                'mean_std': np.mean(np.std(class_embeddings, axis=0)),
                'intra_class_distance': np.mean([
                    np.linalg.norm(sample - np.mean(class_embeddings, axis=0))
                    for sample in class_embeddings
                ])
            }

        # 计算类间距离
        inter_class_distances = []
        for i in range(len(unique_labels)):
            for j in range(i + 1, len(unique_labels)):
                center_i = stats[f'class_{unique_labels[i]}']['center']
                center_j = stats[f'class_{unique_labels[j]}']['center']
                distance = np.linalg.norm(center_i - center_j)
                inter_class_distances.append(distance)

        stats['inter_class_distances'] = {
            'mean': np.mean(inter_class_distances),
            'std': np.std(inter_class_distances),
            'min': np.min(inter_class_distances),
            'max': np.max(inter_class_distances)
        }

        return stats

    def reduce_dimensions(self, n_components: int = 3, method: str = 'pca', **kwargs) -> np.ndarray:
        """
        降维以用于可视化

        Args:
            n_components: 降维后的维度数
            method: 降维方法 ('pca', 'tsne', 'umap')
            **kwargs: 传递给降维算法的额外参数

        Returns:
            降维后的embeddings
        """
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")

        # 如果原始embedding维度为3且要求降维到3维，直接返回原始数据
        if self.embedding_dim == 3 and n_components == 3:
            self.dimensionality_reduction_info = {
                'method': 'original',
                'n_components': 3,
                'note': '原始数据已为3维，无需降维'
            }
            return self.embeddings.copy()

        if method == 'pca':
            reducer = PCA(n_components=n_components, random_state=self.random_state)
            reduced_embeddings = reducer.fit_transform(self.embeddings)

            # 存储降维信息
            self.dimensionality_reduction_info = {
                'method': method,
                'n_components': n_components,
                'explained_variance_ratio': reducer.explained_variance_ratio_,
                'cumulative_variance_ratio': np.cumsum(reducer.explained_variance_ratio_)
            }

        elif method == 'tsne':
            from sklearn.manifold import TSNE

            # t-SNE默认参数
            tsne_params = {
                'n_components': n_components,
                'random_state': self.random_state,
                'perplexity': min(30, max(5, self.embeddings.shape[0] - 1)),  # 防止perplexity过大或过小
                'max_iter': 1000,
                'learning_rate': 'auto'
            }
            # 更新用户传入的参数
            tsne_params.update(kwargs)

            reducer = TSNE(**tsne_params)
            reduced_embeddings = reducer.fit_transform(self.embeddings)

            self.dimensionality_reduction_info = {
                'method': method,
                'n_components': n_components,
                'kl_divergence': reducer.kl_divergence_,
                'n_iter_final': reducer.n_iter_,
                'parameters': tsne_params
            }

        elif method == 'umap':
            try:
                import umap
            except ImportError:
                raise ImportError("UMAP需要安装umap-learn包: pip install umap-learn")

            # UMAP默认参数
            umap_params = {
                'n_components': n_components,
                'random_state': self.random_state,
                'n_neighbors': min(15, self.embeddings.shape[0] - 1),  # 防止n_neighbors过大
                'min_dist': 0.1,
                'metric': 'euclidean'
            }
            # 更新用户传入的参数
            umap_params.update(kwargs)

            reducer = umap.UMAP(**umap_params)
            reduced_embeddings = reducer.fit_transform(self.embeddings)

            self.dimensionality_reduction_info = {
                'method': method,
                'n_components': n_components,
                'parameters': umap_params
            }

        else:
            raise ValueError(f"不支持的降维方法: {method}. 支持的方法: 'pca', 'tsne', 'umap'")

        return reduced_embeddings

    def save_embeddings(self, filepath: str):
        """保存生成的embeddings到文件"""
        if self.embeddings is None:
            raise ValueError("请先生成embeddings")

        np.savez(filepath,
                 embeddings=self.embeddings,
                 labels=self.labels,
                 class_centers=self.class_centers,
                 generation_params=self.generation_params)

    def load_embeddings(self, filepath: str):
        """从文件加载embeddings"""
        data = np.load(filepath, allow_pickle=True)
        self.embeddings = data['embeddings']
        self.labels = data['labels']
        self.class_centers = data['class_centers']
        self.generation_params = data['generation_params'].item()


def create_sample_dataset(n_classes: int = 5,
                          samples_per_class: int = 100,
                          embedding_dim: int = 128,
                          random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    创建示例数据集的便捷函数

    Args:
        n_classes: 类别数量
        samples_per_class: 每个类别的样本数
        embedding_dim: embedding维度
        random_state: 随机种子

    Returns:
        embeddings, labels
    """
    generator = EmbeddingGenerator(embedding_dim=embedding_dim, random_state=random_state)

    # 生成样本数量列表
    n_samples_per_class = [samples_per_class] * n_classes

    # 生成embeddings（使用标准化的0-1参数）
    embeddings, labels = generator.generate_clustered_embeddings(
        n_samples_per_class=n_samples_per_class,
        dispersion=0.6,           # 中等分散度
        curvature=0.2,            # 低曲度
        flatness=0.7,             # 较高扁平度
        inter_class_distance=0.8,  # 较大类间距离
        intra_class_correlation=0.4,  # 中等类内相关性
        inter_hyperplane_parallelism=0.0,
        manifold_complexity=0.2,
        feature_sparsity=0.1,
        noise_level=0.05,
        boundary_sharpness=0.5,
        dimensional_anisotropy=0.3
    )

    return embeddings, labels
