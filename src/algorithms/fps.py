#!/usr/bin/env python3
"""
最远点采样 (Farthest Point Sampling, FPS) 算法实现
"""

import numpy as np
from typing import List, Tuple, Callable
from scipy.spatial.distance import cdist


class FarthestPointSampler:
    """最远点采样器"""
    
    def __init__(self):
        self.distance_functions = {
            'euclidean': self._euclidean_distance,
            'cosine': self._cosine_distance,
            'chebyshev': self._chebyshev_distance,
            'manhattan': self._manhattan_distance,
            'minkowski': self._minkowski_distance
        }
    
    def sample(self, 
               points: np.ndarray, 
               start_idx: int, 
               num_samples: int, 
               distance_metric: str = 'euclidean',
               **kwargs) -> List[int]:
        """
        执行最远点采样
        
        Args:
            points: 点云数据 (N, D)
            start_idx: 起始点索引
            num_samples: 采样点数量
            distance_metric: 距离度量方式
            **kwargs: 距离函数的额外参数
            
        Returns:
            采样点的索引列表
        """
        if start_idx >= len(points):
            raise ValueError(f"起始点索引 {start_idx} 超出范围 [0, {len(points)-1}]")
        
        if num_samples > len(points):
            raise ValueError(f"采样数量 {num_samples} 超过总点数 {len(points)}")
        
        if distance_metric not in self.distance_functions:
            raise ValueError(f"不支持的距离度量: {distance_metric}")
        
        selected_indices = [start_idx]
        remaining_indices = list(range(len(points)))
        remaining_indices.remove(start_idx)
        
        # 计算距离函数
        distance_func = self.distance_functions[distance_metric]
        
        for _ in range(num_samples - 1):
            if not remaining_indices:
                break
                
            max_min_distance = -1
            farthest_idx = -1
            
            # 对每个剩余点，计算到已选点的最小距离
            for candidate_idx in remaining_indices:
                min_distance = float('inf')
                
                for selected_idx in selected_indices:
                    dist = distance_func(
                        points[candidate_idx], 
                        points[selected_idx], 
                        **kwargs
                    )
                    min_distance = min(min_distance, dist)
                
                # 选择最小距离最大的点
                if min_distance > max_min_distance:
                    max_min_distance = min_distance
                    farthest_idx = candidate_idx
            
            if farthest_idx != -1:
                selected_indices.append(farthest_idx)
                remaining_indices.remove(farthest_idx)
        
        return selected_indices
    
    def _euclidean_distance(self, p1: np.ndarray, p2: np.ndarray, **kwargs) -> float:
        """欧氏距离"""
        return np.linalg.norm(p1 - p2)
    
    def _cosine_distance(self, p1: np.ndarray, p2: np.ndarray, **kwargs) -> float:
        """余弦距离"""
        dot_product = np.dot(p1, p2)
        norm_p1 = np.linalg.norm(p1)
        norm_p2 = np.linalg.norm(p2)
        
        if norm_p1 == 0 or norm_p2 == 0:
            return 1.0  # 最大余弦距离
        
        cosine_similarity = dot_product / (norm_p1 * norm_p2)
        # 余弦距离 = 1 - 余弦相似度
        return 1.0 - cosine_similarity
    
    def _chebyshev_distance(self, p1: np.ndarray, p2: np.ndarray, **kwargs) -> float:
        """切比雪夫距离（无穷范数）"""
        return np.max(np.abs(p1 - p2))
    
    def _manhattan_distance(self, p1: np.ndarray, p2: np.ndarray, **kwargs) -> float:
        """曼哈顿距离（L1范数）"""
        return np.sum(np.abs(p1 - p2))
    
    def _minkowski_distance(self, p1: np.ndarray, p2: np.ndarray, p: int = 3, **kwargs) -> float:
        """闵可夫斯基距离"""
        return np.power(np.sum(np.power(np.abs(p1 - p2), p)), 1/p)
    
    def get_available_metrics(self) -> List[str]:
        """获取可用的距离度量方式"""
        return list(self.distance_functions.keys())
    
    def compute_path_distances(self, 
                              points: np.ndarray, 
                              selected_indices: List[int], 
                              distance_metric: str = 'euclidean',
                              **kwargs) -> List[float]:
        """
        计算路径中相邻点之间的距离
        
        Args:
            points: 点云数据
            selected_indices: 选中的点索引
            distance_metric: 距离度量方式
            
        Returns:
            相邻点之间的距离列表
        """
        if len(selected_indices) < 2:
            return []
        
        distance_func = self.distance_functions[distance_metric]
        distances = []
        
        for i in range(len(selected_indices) - 1):
            idx1 = selected_indices[i]
            idx2 = selected_indices[i + 1]
            dist = distance_func(points[idx1], points[idx2], **kwargs)
            distances.append(dist)
        
        return distances
    
    def get_path_statistics(self, 
                           points: np.ndarray, 
                           selected_indices: List[int], 
                           labels: np.ndarray,
                           distance_metric: str = 'euclidean',
                           **kwargs) -> dict:
        """
        获取路径统计信息
        
        Args:
            points: 点云数据
            selected_indices: 选中的点索引
            labels: 点的标签
            distance_metric: 距离度量方式
            
        Returns:
            统计信息字典
        """
        if not selected_indices:
            return {}
        
        # 计算路径距离
        path_distances = self.compute_path_distances(
            points, selected_indices, distance_metric, **kwargs
        )
        
        # 统计各类别的点数
        selected_labels = labels[selected_indices]
        unique_labels, counts = np.unique(selected_labels, return_counts=True)
        class_distribution = dict(zip(unique_labels.tolist(), counts.tolist()))
        
        # 计算总路径长度
        total_distance = sum(path_distances) if path_distances else 0.0
        
        stats = {
            'total_points': len(selected_indices),
            'total_distance': total_distance,
            'average_step_distance': np.mean(path_distances) if path_distances else 0.0,
            'max_step_distance': np.max(path_distances) if path_distances else 0.0,
            'min_step_distance': np.min(path_distances) if path_distances else 0.0,
            'class_distribution': class_distribution,
            'path_distances': path_distances,
            'distance_metric': distance_metric
        }
        
        return stats


def create_fps_sampler() -> FarthestPointSampler:
    """创建FPS采样器实例"""
    return FarthestPointSampler() 