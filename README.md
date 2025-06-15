# ALQuery3D
一个纯Python实现的高维embeddings生成器，专为主动学习研究设计
提供Web界面进行参数控制和3D可视化
后端使用numpy、scipy和scikit-learn等CPU-only库实现

## 项目结构

```
ALQuery3D/
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   │   ├── __init__.py
│   │   └── embedding_generator.py  # 高维embeddings生成器
│   ├── web/                      # Web界面
│   │   ├── app.py               # Flask后端
│   │   └── templates/
│   │       └── index.html       # Web前端界面
│   └── __init__.py
├── data/                         # 数据缓存目录
│   └── tmp_data.h5              # HDF5缓存文件
├── examples/                     # 示例代码
│   └── generate_embeddings_demo.py  # embeddings生成演示
├── tests/                        # 测试目录
│   └── test_embedding_generator.py  # 单元测试
├── run_web.py                    # Web应用启动脚本
├── requirements.txt              # 项目依赖
├── README.md
└── LICENSE
```

## 核心功能

### 1. 高维Embeddings生成器 (`EmbeddingGenerator`)

模拟神经网络encoder生成的高维embeddings，具有以下特性：

- **类别聚集**: 不同类别的样本围绕各自的中心聚集
- **可控分散度**: 控制类内样本的散布程度 (0.1-5.0)
- **可控曲度**: 控制数据的非线性变形程度 (0.0-1.0)  
- **可控扁平度**: 控制在某些维度上的压缩程度 (0.1-1.0)
- **类间距离控制**: 控制不同类别中心之间的距离
- **类内相关性**: 控制类内特征的相关程度

### 2. 主要功能

- ✅ **高维embeddings生成**: 支持3-2048维embeddings生成
- ✅ **Web可视化界面**: 现代化的Web界面，支持实时参数调节
- ✅ **参数化控制**: 精确控制数据的各种几何特性
- ✅ **降维可视化**: 支持PCA、t-SNE、UMAP等降维方法进行3D可视化
- ✅ **智能缓存系统**: HDF5缓存提升性能，避免重复计算
- ✅ **数据标准化**: 自动标准化到-1~1范围
- ✅ **灵活的API**: 提供便捷函数和详细的参数控制

## 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装核心依赖：

```bash
pip install numpy scikit-learn matplotlib scipy
```

## 快速开始

### Web界面使用

启动Web应用：

```bash
python run_web.py
```

然后在浏览器中访问 `http://localhost:5000`

**Web界面功能**：
- 🎛️ **参数控制**: 直观的滑动条和输入框控制所有参数
- 📊 **实时可视化**: 3D交互式图表，支持旋转、缩放
- 🔄 **降维方法切换**: 一键切换PCA、t-SNE、UMAP
- 💾 **智能缓存**: 自动缓存降维结果，提升响应速度
- 📈 **统计信息**: 实时显示数据统计和维度信息

### 编程接口使用

```python
from src.data.embedding_generator import EmbeddingGenerator

# 创建生成器
generator = EmbeddingGenerator(embedding_dim=128, random_state=42)

# 生成embeddings（所有参数标准化到0-1范围）
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[100, 150, 120],  # 每个类别的样本数
    dispersion=0.6,                       # 分散度 (0.0-1.0)
    curvature=0.2,                        # 曲度 (0.0-1.0)
    flatness=0.7,                         # 扁平度 (0.0-1.0)
    inter_class_distance=0.8,             # 类间距离 (0.0-1.0)
    intra_class_correlation=0.4           # 类内相关性 (0.0-1.0)
)

print(f"生成的embeddings形状: {embeddings.shape}")
print(f"类别数量: {len(np.unique(labels))}")
```

### 每类别独立控制

```python
# 为每个类别设置不同的参数
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[80, 100, 120],
    dispersion=[0.3, 0.6, 0.9],           # 每个类别不同的分散度
    curvature=[0.1, 0.3, 0.5],            # 每个类别不同的曲度
    flatness=[0.4, 0.7, 1.0],             # 每个类别不同的扁平度
    inter_class_distance=0.7,             # 全局类间距离
    intra_class_correlation=[0.2, 0.5, 0.8]  # 每个类别不同的相关性
)
```

### 降维可视化

```python
# PCA降维到3D
reduced_pca = generator.reduce_dimensions(n_components=3, method='pca')

# t-SNE降维到3D  
reduced_tsne = generator.reduce_dimensions(n_components=3, method='tsne')

# UMAP降维到3D
reduced_umap = generator.reduce_dimensions(n_components=3, method='umap')

# 获取降维信息
info = generator.dimensionality_reduction_info
if info['method'] == 'pca':
    print(f"前3个主成分解释的方差比例: {info['explained_variance_ratio']}")
elif info['method'] == 'tsne':
    print(f"t-SNE KL散度: {info['kl_divergence']}")
elif info['method'] == 'umap':
    print(f"UMAP参数: {info['parameters']}")
```

### 便捷函数

```python
from src.data.embedding_generator import create_sample_dataset

# 快速创建示例数据集
embeddings, labels = create_sample_dataset(
    n_classes=5,
    samples_per_class=100,
    embedding_dim=128,
    random_state=42
)
```

## 运行演示

### Web界面演示
```bash
# 启动Web应用
python run_web.py
```

### 编程接口演示
```bash
# 运行完整演示
python examples/generate_embeddings_demo.py

# 运行单元测试
python tests/test_embedding_generator.py
```

演示将展示：
- 基本embedding生成
- 参数对结果的影响
- 降维和可视化
- 数据保存和加载
- 统计信息计算

## 参数说明

### `generate_clustered_embeddings` 参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `n_samples_per_class` | List[int] | 10-5000 | 每个类别的样本数量列表 |
| `embedding_dim` | int | 3-2048 | Embedding维度 |
| `dispersion` | float/List[float] | 0.0-1.0 | 分散度，控制类内样本散布程度 |
| `curvature` | float/List[float] | 0.0-1.0 | 曲度，控制非线性变形程度 |
| `flatness` | float/List[float] | 0.0-1.0 | 扁平度，控制维度压缩程度 |
| `inter_class_distance` | float | 0.0-1.0 | 类间距离，控制类别中心间距离 |
| `intra_class_correlation` | float/List[float] | 0.0-1.0 | 类内相关性，控制特征相关程度 |

### 生成效果

- **低分散度 (0.2)**: 样本紧密聚集在类别中心周围
- **高分散度 (0.8)**: 样本在类别中心周围广泛分布
- **低曲度 (0.0)**: 数据呈现线性高斯分布特征
- **高曲度 (0.8)**: 数据具有复杂的非线性结构
- **低扁平度 (0.2)**: 数据在大部分维度上被压缩
- **高扁平度 (1.0)**: 数据在各维度上相对均匀分布
- **低类间距离 (0.2)**: 不同类别中心靠近
- **高类间距离 (0.8)**: 不同类别中心相距较远
- **低相关性 (0.1)**: 特征维度间相互独立
- **高相关性 (0.9)**: 特征维度间高度相关

### 分散度 vs 类内相关性

**分散度（dispersion）** 和 **类内相关性（intra_class_correlation）** 控制不同方面：

- **分散度**: 控制样本围绕类别中心的整体散布**范围**（标量缩放效应）
- **类内相关性**: 控制特征维度之间的**相关结构**（协方差矩阵效应）

两者相互补充，共同决定类内数据的分布特性。

## API文档

### EmbeddingGenerator类

#### 主要方法

- `generate_clustered_embeddings()`: 生成聚类embeddings
- `get_class_statistics()`: 获取类别统计信息
- `reduce_dimensions()`: 降维处理
- `save_embeddings()`: 保存数据
- `load_embeddings()`: 加载数据

#### 统计信息

生成的统计信息包括：
- 各类别样本数量
- 类别中心坐标
- 类内标准差
- 类内平均距离
- 类间距离统计（均值、最小值、最大值）

## 应用场景

1. **主动学习研究**: 为主动学习算法提供可控的测试数据
2. **聚类算法评估**: 生成具有已知结构的数据用于算法验证
3. **降维算法测试**: 提供高维数据用于降维算法性能评估
4. **可视化系统开发**: 为3D可视化系统提供数据源
5. **机器学习教学**: 帮助理解高维数据的几何特性

## 特性更新

### v1.4.0 - Embedding维度设置功能
- ✨ 新增：支持3-2048维embedding生成
- ✨ 新增：3维数据智能降维处理
- ✨ 新增：Web界面维度设置控件
- ✨ 新增：动态维度参数覆盖

### v1.3.0 - Web界面和缓存系统
- ✨ 新增：现代化Web界面
- ✨ 新增：HDF5智能缓存系统
- ✨ 新增：实时3D可视化
- ✨ 新增：参数范围扩展和标准化

### v1.2.0 - 降维和参数系统
- ✨ 新增：t-SNE、UMAP降维支持
- ✨ 新增：每类别独立参数控制
- ✨ 新增：参数范围标准化到0-1

## 下一步开发计划

- [ ] 添加更多非线性变换类型
- [ ] 添加噪声注入功能
- [ ] 添加主动学习查询算法
- [ ] 支持更多数据导出格式
- [ ] 添加批量实验功能

## 许可证

本项目采用 MIT 许可证 - 详情请见 [LICENSE](LICENSE) 文件。
