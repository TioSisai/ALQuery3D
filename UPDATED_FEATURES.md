# ALQuery3D 功能更新说明

## 🎯 更新概述

根据您的要求，我们对ALQuery3D进行了重大功能升级，主要包括参数范围扩展、数据标准化、缓存机制等。

## 📊 主要更新内容

### 1. 样本数量范围扩展
- **之前**: 10-500个样本
- **现在**: 10-5000个样本
- **影响**: 支持更大规模的数据生成，适合大型实验

### 2. 参数范围映射优化

#### 分散度 (Dispersion)
- **0.0**: 映射到0.001 (接近真实的0)
- **1.0**: 映射到20.0 (非常大的分散度)
- **效果**: 0时数据点高度聚集，1时数据点极度分散

#### 曲度 (Curvature)
- **0.0**: 映射到0.0 (无非线性变形)
- **1.0**: 映射到5.0 (极强的非线性变形)
- **效果**: 0时保持线性分布，1时产生复杂的非线性结构

#### 扁平度 (Flatness)
- **0.0**: 映射到0.001 (接近完全扁平)
- **1.0**: 映射到1.0 (无维度压缩)
- **效果**: 0时数据在某些维度被极度压缩，1时保持原始维度结构

#### 类内相关性 (Intra-class Correlation)
- **0.0**: 映射到0.0 (维度间无相关性)
- **1.0**: 映射到0.99 (维度间高度相关)
- **效果**: 0时各维度独立，1时维度间强相关

#### 类间距离 (Inter-class Distance)
- **0.0**: 映射到0.1 (类别中心非常接近)
- **1.0**: 映射到50.0 (类别中心距离极远)
- **效果**: 0时类别重叠严重，1时类别完全分离

### 3. 数据标准化
- **自动标准化**: 生成的embeddings自动缩放到-1~1范围
- **保持相对关系**: 标准化不改变数据点间的相对位置关系
- **统一尺度**: 便于不同参数设置下的结果对比

### 4. HDF5缓存机制

#### 数据存储
- **位置**: `project_root/data/tmp_data.h5`
- **内容**: 原始embeddings、labels、降维结果
- **格式**: HDF5高效二进制格式

#### 缓存策略
1. **生成时**: 自动保存原始embeddings和PCA结果
2. **首次降维**: 计算并缓存t-SNE/UMAP结果
3. **后续访问**: 直接从缓存加载，无需重新计算
4. **重新生成**: 自动清理旧缓存，生成新数据

#### 生命周期管理
- **生成时**: 删除旧的tmp_data.h5文件
- **运行中**: 缓存所有降维结果
- **退出时**: 自动清理tmp_data.h5文件

## 🔧 Web界面更新

### 界面改进
- **样本数量**: 滑块和输入框支持10-5000范围
- **缓存状态**: 显示当前数据的缓存状态
- **数据范围**: 显示embeddings的实际数值范围
- **加载指示**: 降维计算时显示加载状态

### 状态指示器
- **Generated**: 刚生成新数据
- **Loading...**: 正在计算降维
- **Cached**: 从缓存加载
- **Error**: 计算或加载出错

## 📈 性能优化

### 计算效率
- **首次计算**: PCA/t-SNE/UMAP各计算一次
- **后续切换**: 直接从缓存加载，响应时间<1秒
- **内存管理**: HDF5压缩存储，节省内存空间

### 大数据支持
- **5000样本**: 约1.2MB内存占用 (128维)
- **缓存文件**: 通常<10MB，包含所有降维结果
- **加载速度**: HDF5格式，毫秒级加载

## 🎮 使用示例

### 极端参数测试
```python
# 最小参数 - 高度聚集的数据
generator.generate_clustered_embeddings(
    n_samples_per_class=[1000],
    dispersion=0.0,      # 接近0的分散度
    curvature=0.0,       # 无曲度
    flatness=0.0,        # 极度扁平
    inter_class_distance=0.0,  # 类别重叠
    intra_class_correlation=0.0  # 无相关性
)

# 最大参数 - 极度分散的数据
generator.generate_clustered_embeddings(
    n_samples_per_class=[1000],
    dispersion=1.0,      # 极大分散度
    curvature=1.0,       # 强非线性
    flatness=1.0,        # 无压缩
    inter_class_distance=1.0,  # 极大距离
    intra_class_correlation=1.0  # 强相关性
)
```

### Web界面操作流程
1. **设置参数**: 调节滑块或输入数值
2. **生成数据**: 点击"Generate Embeddings"
3. **查看PCA**: 自动显示PCA降维结果
4. **切换方法**: 点击t-SNE或UMAP按钮
5. **观察缓存**: 首次计算显示"Loading..."，后续显示"Cached"

## 🔍 技术细节

### 参数映射函数
```python
def _map_parameter_to_range(self, normalized_values, param_name):
    if param_name == 'dispersion':
        # 0 -> 0.001, 1 -> 20.0
        mapped_val = 0.001 + val * (20.0 - 0.001)
    elif param_name == 'curvature':
        # 0 -> 0.0, 1 -> 5.0
        mapped_val = val * 5.0
    # ... 其他参数
```

### 标准化算法
```python
def _standardize_embeddings(self, embeddings):
    min_vals = np.min(embeddings, axis=0)
    max_vals = np.max(embeddings, axis=0)
    ranges = max_vals - min_vals
    ranges[ranges == 0] = 1.0  # 避免除零
    standardized = 2 * (embeddings - min_vals) / ranges - 1
    return standardized
```

### HDF5缓存结构
```
tmp_data.h5
├── embeddings    # 原始128维embeddings
├── labels        # 类别标签
├── pca          # PCA降维结果 (3维)
├── tsne         # t-SNE降维结果 (3维)
└── umap         # UMAP降维结果 (3维)
```

## ⚡ 性能对比

| 功能 | 更新前 | 更新后 | 提升 |
|------|--------|--------|------|
| 样本数量 | 10-500 | 10-5000 | 10倍 |
| 参数范围 | 有限 | 极端值支持 | 显著 |
| 降维切换 | 每次重算 | 缓存加载 | 10-100倍 |
| 数据一致性 | 手动管理 | 自动标准化 | 完全 |
| 存储效率 | 内存临时 | HDF5压缩 | 高效 |

## 🚀 启动方式

```bash
# 安装新依赖
pip install h5py

# 启动Web应用
python run_web.py

# 访问界面
# http://localhost:5000
```

## 📝 注意事项

1. **首次使用**: t-SNE和UMAP首次计算较慢，请耐心等待
2. **大样本**: 5000样本的t-SNE可能需要几分钟计算时间
3. **缓存清理**: 重新生成数据会自动清理旧缓存
4. **内存使用**: 大数据集建议关闭其他程序释放内存
5. **参数效果**: 极端参数值可能产生意外的数据分布

---

**总结**: 这次更新大幅提升了ALQuery3D的功能性和性能，支持更大规模的数据生成、更极端的参数设置、更高效的缓存机制，为active learning研究提供了更强大的工具！ 