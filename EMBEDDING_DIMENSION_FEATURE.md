# Embedding维度设置功能

## 功能概述

ALQuery3D现在支持动态设置embedding维度，允许用户在3维到2048维之间自由选择，满足不同的研究和实验需求。

## 主要特性

### 1. 维度范围
- **最小维度**: 3维
- **最大维度**: 2048维
- **默认维度**: 128维

### 2. 智能降维处理
- **3维数据**: 当embedding维度设置为3时，所有降维方法（PCA、t-SNE、UMAP）都会直接显示原始3维坐标
- **高维数据**: 自动应用选择的降维算法将高维数据降至3维进行可视化

### 3. 界面集成
- 在Web界面的Configuration部分添加了"Embedding Dimension"设置
- 支持数值输入框和滑动条双向同步
- 实时参数验证和错误提示

## 使用方法

### Web界面使用

1. **设置维度**:
   - 在左侧控制面板的"Configuration"部分找到"Embedding Dimension"
   - 通过输入框直接输入数值（3-2048）
   - 或使用滑动条调节维度

2. **生成数据**:
   - 设置好维度后，配置其他参数（类别数量、各类参数等）
   - 点击"Generate Embeddings"按钮生成指定维度的embedding

3. **可视化**:
   - 3维数据：任何降维方法都显示原始坐标
   - 高维数据：选择PCA/t-SNE/UMAP进行降维可视化

### 编程接口使用

```python
from src.data.embedding_generator import EmbeddingGenerator

# 方法1: 初始化时设置维度
generator = EmbeddingGenerator(embedding_dim=512, random_state=42)
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[100, 100, 100]
)

# 方法2: 生成时动态设置维度
generator = EmbeddingGenerator(embedding_dim=128, random_state=42)
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[100, 100, 100],
    embedding_dim=64  # 动态覆盖为64维
)

# 3维数据特殊处理
generator_3d = EmbeddingGenerator(embedding_dim=3, random_state=42)
embeddings_3d, labels_3d = generator_3d.generate_clustered_embeddings(
    n_samples_per_class=[50, 50, 50]
)

# 降维 - 3维数据会直接返回原始数据
reduced = generator_3d.reduce_dimensions(n_components=3, method='pca')
print(generator_3d.dimensionality_reduction_info)
# 输出: {'method': 'original', 'n_components': 3, 'note': '原始数据已为3维，无需降维'}
```

## 技术实现

### 1. 后端实现
- `EmbeddingGenerator`类支持`embedding_dim`参数
- `generate_clustered_embeddings`方法支持动态维度设置
- `reduce_dimensions`方法智能检测3维情况

### 2. 前端实现
- HTML界面添加维度设置控件
- JavaScript处理维度参数传递
- 降维方法选择时的3维特殊提示

### 3. 参数验证
- 维度范围验证（3-2048）
- 输入类型验证
- 边界情况处理

## 应用场景

### 1. 低维可视化研究
- 设置为3维直接观察原始数据分布
- 无需降维，避免信息损失
- 适合简单聚类分析

### 2. 高维特征学习
- 设置为128、256、512等常见维度
- 模拟真实深度学习模型输出
- 研究高维空间中的数据分布

### 3. 极高维实验
- 设置为1024、2048等极高维度
- 研究维度诅咒现象
- 测试降维算法性能

## 性能考虑

### 1. 内存使用
- 维度越高，内存占用越大
- 2048维 × 5000样本 ≈ 80MB内存
- 建议高维度时减少样本数量

### 2. 计算时间
- 高维度生成时间较长
- 降维计算复杂度随维度增加
- HDF5缓存系统减少重复计算

### 3. 可视化效果
- 3维数据：完美保持原始结构
- 高维数据：降维可能损失部分信息
- 不同降维方法效果差异明显

## 测试验证

项目包含完整的测试套件：

```bash
# 运行维度功能测试
python test_embedding_dim.py

# 运行边界情况测试  
python test_edge_cases.py
```

测试覆盖：
- ✅ 不同维度生成（3, 16, 128, 512, 1024维）
- ✅ 动态维度设置
- ✅ 3维数据降维处理
- ✅ 边界值验证（最小3维，最大2048维）
- ✅ 无效输入异常处理

## 更新日志

### v1.4.0 - Embedding维度设置功能
- ✨ 新增：支持3-2048维embedding生成
- ✨ 新增：3维数据智能降维处理
- ✨ 新增：Web界面维度设置控件
- ✨ 新增：动态维度参数覆盖
- 🐛 修复：降维方法对3维数据的处理
- 📝 文档：完整的功能说明和使用示例

---

这个功能让ALQuery3D更加灵活，能够适应从简单3维可视化到复杂高维特征学习的各种研究需求。 