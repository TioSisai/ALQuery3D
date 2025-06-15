# ALQuery3D
一个纯Python实现的高维embeddings生成器，专为主动学习研究设计。提供Web界面进行参数控制和3D可视化，支持FPS（最远点采样）算法和多种降维方法。后端使用numpy、scipy和scikit-learn等CPU-only库实现。

## 🎯 项目特色

- 🧠 **神经网络Encoder特性模拟**: 模拟真实神经网络encoder的各种特性
- 🎛️ **精确参数控制**: 11个参数精确控制数据的几何和统计特性
- 🌐 **现代化Web界面**: 深色主题，响应式设计，专业科研工具体验
- 📊 **多种降维方法**: 支持PCA、t-SNE、UMAP三种降维算法
- 🎯 **FPS采样算法**: 完整的最远点采样实现，支持5种距离度量
- 💾 **智能缓存系统**: HDF5缓存提升性能，避免重复计算
- 🔧 **灵活维度支持**: 3-2048维embedding生成

## 📁 项目结构

```
ALQuery3D/
├── src/                          # 源代码目录
│   ├── data/                     # 数据处理模块
│   │   ├── __init__.py
│   │   └── embedding_generator.py  # 高维embeddings生成器
│   ├── algorithms/               # 算法实现
│   │   ├── __init__.py
│   │   └── fps.py               # FPS最远点采样算法
│   ├── web/                      # Web界面
│   │   ├── app.py               # Flask后端
│   │   └── templates/
│   │       └── index.html       # Web前端界面
│   └── __init__.py
├── data/                         # 数据缓存目录
│   └── tmp_data.h5              # HDF5缓存文件（运行时生成）
├── examples/                     # 示例代码
│   └── generate_embeddings_demo.py  # embeddings生成演示
├── tests/                        # 测试目录
│   └── test_embedding_generator.py  # 单元测试
├── run_web.py                    # Web应用启动脚本
├── requirements.txt              # 项目依赖
├── README.md
└── LICENSE
```

## 🚀 快速开始

### 安装依赖

```bash
pip install -r requirements.txt
```

或手动安装核心依赖：

```bash
pip install numpy scikit-learn matplotlib scipy flask plotly h5py umap-learn
```

### 启动Web界面

```bash
python run_web.py
```

然后在浏览器中访问 `http://localhost:5000`

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

## 🧠 核心功能

### 1. 高维Embeddings生成器

模拟神经网络encoder生成的高维embeddings，具有以下特性：

#### 基础几何属性
- **分散度 (Dispersion)**: 控制类内样本的散布程度
- **曲度 (Curvature)**: 控制数据的非线性变形程度，形成圆锥体分布
- **扁平度 (Flatness)**: 控制在某些维度上的压缩程度，靠近超平面
- **类间距离 (Inter-class Distance)**: 控制不同类别中心之间的距离
- **类内相关性 (Intra-class Correlation)**: 控制类内特征的相关程度
- **类间超平面平行度 (Inter-hyperplane Parallelism)**: 控制类间超平面的平行程度

#### 神经网络Encoder特性
- **流形复杂度 (Manifold Complexity)**: 模拟神经网络中的非线性激活函数效应
- **特征稀疏性 (Feature Sparsity)**: 模拟ReLU等激活函数导致的特征稀疏性
- **噪声水平 (Noise Level)**: 模拟编码过程中的信息损失
- **边界锐度 (Boundary Sharpness)**: 控制类别边界的清晰程度
- **维度各向异性 (Dimensional Anisotropy)**: 不同维度的重要性差异

### 2. 多种降维方法

- **PCA**: 主成分分析，保持最大方差
- **t-SNE**: t-分布随机邻域嵌入，保持局部结构
- **UMAP**: 统一流形逼近与投影，平衡全局和局部结构

### 3. FPS最远点采样算法

完整的FPS (Farthest Point Sampling) 实现：

- **多种距离度量**: 欧氏距离、余弦距离、切比雪夫距离、曼哈顿距离、闵可夫斯基距离
- **交互式点选择**: 在3D可视化中点击任意点设置起始位置
- **路径可视化**: 亮青色渐变显示完整的FPS遍历路径
- **范围查看功能**: 可查看FPS路径中的任意连续子序列
- **统计分析**: 路径距离、类别分布、采样质量评估

### 4. Web界面功能

- 🎛️ **参数控制**: 直观的滑动条和输入框控制所有参数
- 📊 **实时可视化**: 3D交互式图表，支持旋转、缩放
- 🔄 **降维方法切换**: 一键切换PCA、t-SNE、UMAP
- 💾 **智能缓存**: 自动缓存降维结果，提升响应速度
- 📈 **统计信息**: 实时显示数据统计和维度信息
- 🎯 **FPS采样**: 完整的FPS采样和可视化功能

## 📊 参数说明

### 基础参数

| 参数 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `n_samples_per_class` | List[int] | 10-5000 | 每个类别的样本数量列表 |
| `embedding_dim` | int | 3-2048 | Embedding维度 |

### 几何控制参数（支持每类别独立设置）

| 参数 | 范围 | 内部映射 | 说明 |
|------|------|----------|------|
| `dispersion` | 0.0-1.0 | 0.001-20.0 | 分散度，控制类内样本散布程度 |
| `curvature` | 0.0-1.0 | 0.0-5.0 | 曲度，控制非线性变形程度 |
| `flatness` | 0.0-1.0 | 0.001-1.0 | 扁平度，控制维度压缩程度 |
| `intra_class_correlation` | 0.0-1.0 | 0.0-0.99 | 类内相关性，控制特征相关程度 |

### 神经网络特性参数

| 参数 | 范围 | 内部映射 | 说明 |
|------|------|----------|------|
| `manifold_complexity` | 0.0-1.0 | 0.0-2.0 | 流形复杂度，模拟非线性激活函数 |
| `feature_sparsity` | 0.0-1.0 | 0.0-0.9 | 特征稀疏性，模拟ReLU激活 |
| `noise_level` | 0.0-1.0 | 0.0-0.5 | 噪声水平，模拟信息损失 |
| `boundary_sharpness` | 0.0-1.0 | 0.0-5.0 | 边界锐度，控制决策边界清晰度 |
| `dimensional_anisotropy` | 0.0-1.0 | 0.0-0.8 | 维度各向异性，模拟特征重要性差异 |

### 全局参数

| 参数 | 范围 | 内部映射 | 说明 |
|------|------|----------|------|
| `inter_class_distance` | 0.0-1.0 | 0.1-50.0 | 类间距离，控制类别中心间距离 |
| `inter_hyperplane_parallelism` | 0.0-1.0 | 0.0-0.99 | 类间超平面平行度 |

## 💡 使用示例

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

### 神经网络特性模拟

```python
# 模拟真实神经网络encoder
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[200, 200, 200],
    dispersion=0.5,
    curvature=0.3,
    flatness=0.6,
    manifold_complexity=0.3,              # 适中的非线性
    feature_sparsity=0.2,                 # 轻微的稀疏性
    noise_level=0.05,                     # 少量噪声
    boundary_sharpness=0.7,               # 较清晰的边界
    dimensional_anisotropy=0.4            # 中等的各向异性
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
print(f"降维方法: {info['method']}")
```

### FPS采样使用

```python
from src.algorithms.fps import create_fps_sampler

# 创建FPS采样器
fps_sampler = create_fps_sampler()

# 执行FPS采样
selected_indices = fps_sampler.sample(
    embeddings,           # 原始高维数据
    start_idx=0,         # 起始点索引
    num_samples=50,      # 采样数量
    distance_metric='euclidean'  # 距离度量
)

# 获取统计信息
stats = fps_sampler.get_path_statistics(
    embeddings, selected_indices, labels, 'euclidean'
)
print(f"采样了 {stats['total_points']} 个点")
print(f"总路径长度: {stats['total_distance']:.3f}")
```

## 🎮 Web界面使用流程

### 1. 配置参数
- 选择类别数量（1-10个）
- 设置embedding维度（3-2048）
- 为每个类别设置独立参数
- 选择降维方法（PCA/t-SNE/UMAP）

### 2. 生成数据
- 点击"Generate Embeddings"按钮
- 等待后台处理（显示加载动画）
- 查看右侧3D可视化结果

### 3. FPS采样（可选）
- 在3D图中点击任意点设置起始位置
- 配置采样参数（数量、距离度量）
- 点击"Start FPS Sampling"执行采样
- 查看FPS路径可视化和统计信息

### 4. 范围查看（可选）
- 设置查看范围（起始和结束索引）
- 点击"View Range"查看指定范围
- 观察范围内的统计信息

## 🔧 技术特点

### 性能优化
- **HDF5缓存**: 智能缓存降维结果，避免重复计算
- **数据标准化**: 自动标准化到-1~1范围，保持相对关系
- **内存管理**: 高效的数据结构设计，支持大规模数据

### 可视化效果
- **亮青色渐变**: FPS路径使用亮青色渐变显示
- **交互式3D**: Plotly高质量交互式图表
- **响应式设计**: 适配不同屏幕尺寸

### 扩展性
- **模块化设计**: 易于添加新的距离度量和降维方法
- **API友好**: 提供完整的编程接口
- **测试覆盖**: 完整的单元测试套件

## 📈 应用场景

### 1. 主动学习研究
- 生成具有特定特性的数据集
- 测试不同采样策略的效果
- 可视化采样结果和数据分布

### 2. 降维算法比较
- 在相同数据上比较PCA、t-SNE、UMAP效果
- 研究不同参数对降维结果的影响

### 3. 神经网络特性分析
- 模拟不同类型的神经网络encoder输出
- 研究高维特征的几何特性

### 4. 数据可视化教学
- 直观展示高维数据的特性
- 理解不同参数对数据分布的影响

## 🧪 运行演示

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

## 📝 注意事项

1. **首次使用**: t-SNE和UMAP首次计算较慢，请耐心等待
2. **大样本**: 5000样本的t-SNE可能需要几分钟计算时间
3. **缓存清理**: 重新生成数据会自动清理旧缓存
4. **内存使用**: 大数据集建议关闭其他程序释放内存
5. **参数效果**: 极端参数值可能产生意外的数据分布

## 🔍 故障排除

### 常见问题
1. **端口被占用**: 修改run_web.py中的端口号
2. **依赖缺失**: 运行`pip install -r requirements.txt`
3. **网络访问**: 确保防火墙允许5000端口
4. **浏览器兼容**: 推荐使用Chrome/Firefox最新版本

### 性能优化
- 大数据集建议减少样本数量
- t-SNE和UMAP计算较慢，耐心等待
- 关闭浏览器其他标签页释放内存

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情。

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个项目！

---

**ALQuery3D** - 为主动学习研究提供强大的高维数据生成和可视化工具！🚀


## 收藏历史

<a href="https://star-history.com/#TioSisai/ALQuery3D&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=TioSisai/ALQuery3D&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=TioSisai/ALQuery3D&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=TioSisai/ALQuery3D&type=Date" />
 </picture>
</a>