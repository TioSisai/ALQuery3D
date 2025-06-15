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
                                      intra_class_correlation: Union[float, List[float]] = 0.3) -> Tuple[np.ndarray, np.ndarray]:
        """
        生成具有聚类特性的高维embeddings

        Args:
            n_samples_per_class: 每个类别的样本数量列表
            n_classes: 类别数量，如果为None则从n_samples_per_class推断
            dispersion: 分散度，控制类内样本的散布程度 (0.0-1.0)
                       可以是单个值或每个类别的列表
            curvature: 曲度，控制数据的非线性变形程度 (0.0-1.0)
                      可以是单个值或每个类别的列表
            flatness: 扁平度，控制在某些维度上的压缩程度 (0.0-1.0)
                     可以是单个值或每个类别的列表
            inter_class_distance: 类间距离，控制不同类别中心之间的距离 (0.0-1.0)
            intra_class_correlation: 类内相关性，控制类内特征的相关程度 (0.0-1.0)
                                   可以是单个值或每个类别的列表

        Returns:
            embeddings: 生成的高维embeddings (n_samples, embedding_dim)
            labels: 对应的类别标签 (n_samples,)
        """
        if n_classes is None:
            n_classes = len(n_samples_per_class)

        # 将参数标准化为列表形式
        dispersion_list = self._normalize_parameter(dispersion, n_classes, 'dispersion')
        curvature_list = self._normalize_parameter(curvature, n_classes, 'curvature')
        flatness_list = self._normalize_parameter(flatness, n_classes, 'flatness')
        correlation_list = self._normalize_parameter(intra_class_correlation, n_classes, 'intra_class_correlation')

        # 验证inter_class_distance参数
        if not (0.0 <= inter_class_distance <= 1.0):
            raise ValueError(f"inter_class_distance必须在0.0-1.0之间，当前值: {inter_class_distance}")

        # 保存生成参数
        self.generation_params = {
            'n_samples_per_class': n_samples_per_class,
            'n_classes': n_classes,
            'dispersion': dispersion_list,
            'curvature': curvature_list,
            'flatness': flatness_list,
            'inter_class_distance': inter_class_distance,
            'intra_class_correlation': correlation_list
        }

        # 生成类别中心
        self.class_centers = self._generate_class_centers(n_classes, inter_class_distance)

        # 生成每个类别的样本
        all_embeddings = []
        all_labels = []

        for class_idx, n_samples in enumerate(n_samples_per_class):
            class_embeddings = self._generate_class_embeddings(
                class_idx, n_samples,
                dispersion_list[class_idx],
                curvature_list[class_idx],
                flatness_list[class_idx],
                correlation_list[class_idx]
            )
            all_embeddings.append(class_embeddings)
            all_labels.extend([class_idx] * n_samples)

        self.embeddings = np.vstack(all_embeddings)
        self.labels = np.array(all_labels)

        return self.embeddings, self.labels

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

    def _generate_class_centers(self, n_classes: int, inter_class_distance: float) -> np.ndarray:
        """生成类别中心点"""
        # 将0-1范围的inter_class_distance映射到实际距离
        actual_distance = 1.0 + inter_class_distance * 9.0  # 映射到1.0-10.0

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

    def _generate_class_embeddings(self,
                                   class_idx: int,
                                   n_samples: int,
                                   dispersion: float,
                                   curvature: float,
                                   flatness: float,
                                   intra_class_correlation: float) -> np.ndarray:
        """为单个类别生成embeddings"""
        center = self.class_centers[class_idx]

        # 1. 生成基础的高斯分布样本
        base_samples = self.rng.multivariate_normal(
            mean=np.zeros(self.embedding_dim),
            cov=self._generate_covariance_matrix(intra_class_correlation),
            size=n_samples
        )

        # 2. 应用扁平度变换
        flattened_samples = self._apply_flatness(base_samples, flatness)

        # 3. 应用曲度变换（非线性变形）
        curved_samples = self._apply_curvature(flattened_samples, curvature)

        # 4. 应用分散度缩放
        # 将0-1范围的dispersion映射到实际缩放因子
        actual_dispersion = 0.1 + dispersion * 2.9  # 映射到0.1-3.0
        dispersed_samples = curved_samples * actual_dispersion

        # 5. 平移到类别中心
        final_samples = dispersed_samples + center

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

    def _apply_flatness(self, samples: np.ndarray, flatness: float) -> np.ndarray:
        """应用扁平度变换"""
        # 随机选择一些维度进行压缩
        n_dims_to_flatten = max(1, int(self.embedding_dim * (1 - flatness)))
        dims_to_flatten = self.rng.choice(
            self.embedding_dim, n_dims_to_flatten, replace=False
        )

        flattened = samples.copy()
        # 在选定维度上应用压缩
        compression_factor = 0.1 + 0.9 * flatness
        flattened[:, dims_to_flatten] *= compression_factor

        return flattened

    def _apply_curvature(self, samples: np.ndarray, curvature: float) -> np.ndarray:
        """应用曲度变换（非线性变形）"""
        if curvature <= 0:
            return samples

        curved = samples.copy()

        # 应用不同类型的非线性变换
        for i in range(self.embedding_dim):
            if self.rng.random() < curvature:
                # 随机选择变换类型
                transform_type = self.rng.choice(['quadratic', 'sine', 'tanh'])

                if transform_type == 'quadratic':
                    # 二次变换
                    curved[:, i] = samples[:, i] + curvature * 0.1 * np.square(samples[:, i])
                elif transform_type == 'sine':
                    # 正弦变换
                    curved[:, i] = samples[:, i] + curvature * 0.5 * np.sin(samples[:, i])
                elif transform_type == 'tanh':
                    # 双曲正切变换
                    curved[:, i] = samples[:, i] + curvature * 0.3 * np.tanh(samples[:, i])

        return curved

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
        intra_class_correlation=0.4  # 中等类内相关性
    )

    return embeddings, labels
