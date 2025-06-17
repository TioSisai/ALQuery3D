# ALQuery3D 聚类属性配置案例集

## 概述

本文档提供了各种聚类属性配置案例，用于模拟真实世界中不同类型的embeddings分布场景。这些配置案例基于ALQuery3D框架，可以生成具有特定聚类特性的高维数据，适用于主动学习、模型测试和算法验证等研究场景。

## 基础配置案例

### 1. 大量噪声类别配置
**场景描述**: 模拟自然图像分类中的背景噪声、随机纹理等常见但无意义的类别

```python
# 大量噪声类别 - 2000个样本，极度分散
generator = EmbeddingGenerator(embedding_dim=512, random_state=42)
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[2000],  # 大量样本
    
    # 几何属性 - 高度分散，模拟噪声
    dispersion=[0.95],                    # 极高分散度，样本广泛分布
    curvature=[0.1],                      # 低曲度，接近线性分布
    flatness=[0.2],                       # 低扁平度，避免过度压缩
    inter_class_distance=0.1,             # 极小类间距离（单类别时不影响）
    intra_class_correlation=[0.05],       # 极低相关性，特征独立
    inter_hyperplane_parallelism=0.0,    # 无平行度约束
    
    # 神经网络特性 - 模拟噪声encoder输出
    manifold_complexity=[0.8],            # 高流形复杂度，模拟复杂噪声结构
    feature_sparsity=[0.6],               # 高稀疏性，多数特征不激活
    noise_level=[0.4],                    # 高噪声水平，信息含量低
    boundary_sharpness=[0.1],             # 极低边界锐度，无明确分类边界
    dimensional_anisotropy=[0.7]          # 高各向异性，特征重要性差异大
)
```

**配置解释**:
- **高分散度(0.95)**: 模拟噪声数据在特征空间中的广泛分布，没有明确的聚集中心
- **高稀疏性(0.6)**: 模拟ReLU等激活函数在噪声输入上的表现，大部分特征不激活
- **高噪声水平(0.4)**: 模拟encoder在处理无意义输入时的高不确定性
- **低边界锐度(0.1)**: 噪声类别通常没有明确的分类边界

**实际应用场景**: 
- 自然图像数据集中的随机纹理、模糊背景
- 文本分类中的无意义字符组合
- 音频识别中的环境噪声

---

### 2. 稀有类别配置
**场景描述**: 模拟医学影像中的罕见疾病、工业检测中的稀有缺陷等少量但聚集的类别

```python
# 稀有类别 - 10个样本，高度聚集
generator = EmbeddingGenerator(embedding_dim=256, random_state=123)
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[10],  # 极少样本
    
    # 几何属性 - 高度聚集，模拟稀有特征
    dispersion=[0.15],                    # 极低分散度，样本紧密聚集
    curvature=[0.05],                     # 极低曲度，接近球形分布
    flatness=[0.1],                       # 极低扁平度，保持球形结构
    inter_class_distance=0.9,             # 高类间距离（虽然只有一类）
    intra_class_correlation=[0.8],        # 高相关性，特征高度一致
    inter_hyperplane_parallelism=0.0,    # 无平行度约束
    
    # 神经网络特性 - 模拟特异性encoder输出
    manifold_complexity=[0.1],            # 低流形复杂度，简单几何结构
    feature_sparsity=[0.05],              # 低稀疏性，重要特征都激活
    noise_level=[0.02],                   # 极低噪声，高质量特征提取
    boundary_sharpness=[0.9],             # 高边界锐度，明确的类别边界
    dimensional_anisotropy=[0.2]          # 低各向异性，特征重要性相对均匀
)
```

**配置解释**:
- **极低分散度(0.15)**: 稀有类别通常具有非常特异的特征表示，样本紧密聚集
- **高相关性(0.8)**: 稀有类别的样本往往具有相似的特征模式
- **低噪声水平(0.02)**: 稀有类别的特征通常很明确，encoder输出质量高
- **高边界锐度(0.9)**: 稀有类别与其他类别有明显区分

**实际应用场景**:
- 医学影像中的罕见疾病案例
- 制造业中的特定缺陷类型
- 金融领域中的欺诈交易模式
- 自然语言处理中的专业术语

---

### 3. 一般类别配置
**场景描述**: 模拟常见分类任务中的主要类别，如CIFAR-10中的动物类别

```python
# 一般类别 - 300个样本，中等聚集度
generator = EmbeddingGenerator(embedding_dim=384, random_state=456)
embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=[300],  # 中等样本数量
    
    # 几何属性 - 中等聚集程度
    dispersion=[0.5],                     # 中等分散度，自然的类内变化
    curvature=[0.25],                     # 中等曲度，适度的非线性结构
    flatness=[0.6],                       # 中等扁平度，部分维度压缩
    inter_class_distance=0.6,             # 中等类间距离
    intra_class_correlation=[0.4],        # 中等相关性，特征部分相关
    inter_hyperplane_parallelism=0.3,    # 中等平行度
    
    # 神经网络特性 - 模拟标准encoder输出
    manifold_complexity=[0.3],            # 中等流形复杂度，适度非线性
    feature_sparsity=[0.15],              # 中等稀疏性，部分特征选择性激活
    noise_level=[0.08],                   # 中等噪声水平，典型训练噪声
    boundary_sharpness=[0.6],             # 中等边界锐度，较清晰的类别边界
    dimensional_anisotropy=[0.35]         # 中等各向异性，特征重要性有差异
)
```

**配置解释**:
- **中等分散度(0.5)**: 反映一般类别内部的自然变化，既有聚集性又有多样性
- **中等曲度(0.25)**: 模拟真实数据的适度非线性结构
- **中等噪声水平(0.08)**: 反映标准训练条件下的encoder表现
- **中等边界锐度(0.6)**: 类别边界相对清晰但不完美

**实际应用场景**:
- 自然图像分类（CIFAR-10, ImageNet的主要类别）
- 文本分类中的主题类别
- 语音识别中的常见音素
- 推荐系统中的用户兴趣类别

## 高级配置案例

### 4. 层次聚类配置
**场景描述**: 模拟具有层次结构的数据，如生物分类学中的科属种关系

```python
# 层次聚类 - 模拟多层次分类结构
generator = EmbeddingGenerator(embedding_dim=768, random_state=789)

# 主类别（科级）
main_classes = [200, 180, 220]  # 3个主要类别

# 子类别（属级）- 每个主类别分为多个子类
sub_classes = [
    [60, 70, 70],      # 主类别1的3个子类
    [50, 60, 70],      # 主类别2的3个子类  
    [80, 70, 70]       # 主类别3的3个子类
]

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=sum(sub_classes, []),  # 展平所有子类
    
    # 层次化的分散度 - 主类别内聚，子类别间有区分
    dispersion=[0.3, 0.35, 0.25, 0.4, 0.3, 0.35, 0.45, 0.4, 0.3],
    curvature=[0.2] * 9,                      # 统一的低曲度
    flatness=[0.4] * 9,                       # 统一的中等扁平度
    inter_class_distance=0.7,                 # 较高类间距离支持层次结构
    intra_class_correlation=[0.6] * 9,        # 高相关性体现类别一致性
    
    # 神经网络特性 - 模拟层次特征学习
    manifold_complexity=[0.4] * 9,            # 中等复杂度支持层次结构
    feature_sparsity=[0.2] * 9,               # 中等稀疏性
    noise_level=[0.06] * 9,                   # 较低噪声，清晰层次边界
    boundary_sharpness=[0.7] * 9,             # 高边界锐度
    dimensional_anisotropy=[0.5] * 9          # 中等各向异性
)
```

**实际应用场景**:
- 生物分类学数据（界门纲目科属种）
- 商品分类系统（类别-子类别-品牌）
- 学科知识体系分类
- 地理位置层次（国家-省份-城市）

---

### 5. 渐变过渡配置
**场景描述**: 模拟连续变化的特征，如年龄分组、颜色渐变等

```python
# 渐变过渡类别 - 模拟连续特征的离散化
generator = EmbeddingGenerator(embedding_dim=256, random_state=101)

# 5个年龄组，呈现渐变特性
age_groups = [150, 200, 250, 200, 150]  # 中间年龄组样本更多

# 渐变的分散度 - 中间组更混合，边界组更聚集
dispersion_gradient = [0.2, 0.4, 0.6, 0.4, 0.2]
correlation_gradient = [0.7, 0.5, 0.3, 0.5, 0.7]  # 边界组相关性更高

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=age_groups,
    
    dispersion=dispersion_gradient,
    curvature=[0.3] * 5,                      # 适度曲度支持连续性
    flatness=[0.5] * 5,                       # 中等扁平度
    inter_class_distance=0.4,                 # 较小类间距离，体现连续性
    intra_class_correlation=correlation_gradient,
    inter_hyperplane_parallelism=0.6,        # 高平行度体现渐变关系
    
    # 神经网络特性 - 模拟连续特征编码
    manifold_complexity=[0.5] * 5,            # 中高复杂度支持连续变化
    feature_sparsity=[0.1] * 5,               # 低稀疏性，保持连续性
    noise_level=[0.1] * 5,                    # 中等噪声
    boundary_sharpness=[0.3] * 5,             # 低边界锐度，模糊过渡
    dimensional_anisotropy=[0.4] * 5          # 中等各向异性
)
```

**实际应用场景**:
- 年龄组分类
- 图像亮度/颜色分级
- 文本情感强度分级
- 音频音量/音调分级

---

### 6. 多模态混合配置
**场景描述**: 模拟多模态数据融合，如图像+文本的多模态分类

```python
# 多模态数据 - 视觉模态和文本模态的融合
generator = EmbeddingGenerator(embedding_dim=1024, random_state=202)

# 3个类别，每个类别包含视觉和文本信息
multimodal_classes = [120, 150, 100]

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=multimodal_classes,
    
    # 多模态特征的复杂几何结构
    dispersion=[0.6, 0.5, 0.7],              # 不同模态融合导致的变化
    curvature=[0.4, 0.5, 0.3],               # 高曲度反映模态间相互作用
    flatness=[0.7, 0.8, 0.6],                # 高扁平度，模态信息压缩
    inter_class_distance=0.5,                # 中等类间距离
    intra_class_correlation=[0.4, 0.3, 0.5], # 模态内相关性变化
    inter_hyperplane_parallelism=0.2,        # 低平行度，模态独立性
    
    # 神经网络特性 - 模拟多模态encoder
    manifold_complexity=[0.7, 0.8, 0.6],     # 高复杂度，多模态融合复杂
    feature_sparsity=[0.3, 0.25, 0.35],      # 中等稀疏性，模态选择性
    noise_level=[0.12, 0.1, 0.15],           # 多模态融合噪声
    boundary_sharpness=[0.4, 0.5, 0.3],      # 中等边界锐度
    dimensional_anisotropy=[0.6, 0.7, 0.5]   # 高各向异性，模态权重差异
)
```

**实际应用场景**:
- 图像-文本多模态分类
- 音频-视频融合识别
- 传感器数据融合
- 知识图谱与文本融合

---

### 7. 对抗样本配置
**场景描述**: 模拟对抗训练环境下的embeddings特性

```python
# 对抗样本 - 模拟对抗训练后的鲁棒表示
generator = EmbeddingGenerator(embedding_dim=512, random_state=303)

adversarial_classes = [100, 100, 100, 100]

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=adversarial_classes,
    
    # 对抗训练的几何特性
    dispersion=[0.4, 0.45, 0.35, 0.5],       # 中等分散，鲁棒性要求
    curvature=[0.2, 0.15, 0.25, 0.1],        # 低曲度，避免过拟合
    flatness=[0.3, 0.25, 0.35, 0.2],         # 低扁平度，保持信息
    inter_class_distance=0.8,                # 高类间距离，增强区分性
    intra_class_correlation=[0.6, 0.7, 0.5, 0.8],  # 高相关性，稳定表示
    inter_hyperplane_parallelism=0.1,        # 低平行度，避免攻击利用
    
    # 神经网络特性 - 对抗鲁棒性
    manifold_complexity=[0.2, 0.1, 0.3, 0.15],  # 低复杂度，简化决策面
    feature_sparsity=[0.05, 0.03, 0.08, 0.02],  # 极低稀疏性，重要特征保持
    noise_level=[0.03, 0.02, 0.04, 0.01],       # 极低噪声，高质量表示
    boundary_sharpness=[0.9, 0.85, 0.95, 0.8],  # 极高边界锐度，鲁棒决策
    dimensional_anisotropy=[0.2, 0.15, 0.25, 0.1]  # 低各向异性，均衡特征
)
```

**实际应用场景**:
- 对抗训练后的图像分类器
- 鲁棒语音识别系统
- 安全关键应用的深度学习模型
- 医疗诊断系统的鲁棒表示

---

### 8. 领域适应配置
**场景描述**: 模拟跨领域迁移学习中的源域和目标域差异

```python
# 领域适应 - 源域清晰，目标域模糊
generator = EmbeddingGenerator(embedding_dim=256, random_state=404)

# 源域类别（清晰）
source_domain = [200, 180, 220]
# 目标域类别（模糊，分布不同）
target_domain = [80, 70, 90]

# 合并配置
domain_classes = source_domain + target_domain

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=domain_classes,
    
    # 源域 vs 目标域的不同特性
    dispersion=[0.3, 0.35, 0.25,           # 源域：低分散度
                0.7, 0.8, 0.6],            # 目标域：高分散度
    curvature=[0.1, 0.15, 0.05,            # 源域：低曲度
               0.5, 0.6, 0.4],             # 目标域：高曲度
    flatness=[0.2, 0.25, 0.15,             # 源域：低扁平度
              0.8, 0.85, 0.7],             # 目标域：高扁平度
    inter_class_distance=0.6,              # 中等类间距离
    intra_class_correlation=[0.8, 0.85, 0.75,  # 源域：高相关性
                            0.3, 0.25, 0.4],   # 目标域：低相关性
    
    # 神经网络特性 - 领域差异
    manifold_complexity=[0.1, 0.05, 0.15,     # 源域：低复杂度
                        0.7, 0.8, 0.6],       # 目标域：高复杂度
    feature_sparsity=[0.05, 0.03, 0.08,       # 源域：低稀疏性
                     0.4, 0.5, 0.3],          # 目标域：高稀疏性
    noise_level=[0.02, 0.01, 0.03,            # 源域：低噪声
                0.2, 0.25, 0.15],            # 目标域：高噪声
    boundary_sharpness=[0.9, 0.95, 0.85,      # 源域：锐利边界
                       0.2, 0.15, 0.3],       # 目标域：模糊边界
    dimensional_anisotropy=[0.1, 0.05, 0.15,  # 源域：低各向异性
                          0.7, 0.8, 0.6]      # 目标域：高各向异性
)
```

**实际应用场景**:
- 跨领域图像分类（实验室→真实环境）
- 跨语言文本分类
- 医疗数据跨医院迁移
- 工业数据跨设备迁移

---

### 9. 长尾分布配置
**场景描述**: 模拟真实世界数据的长尾分布特性

```python
# 长尾分布 - 少数头部类别 + 大量尾部类别
generator = EmbeddingGenerator(embedding_dim=384, random_state=505)

# 长尾分布：2个头部类别 + 8个尾部类别
head_classes = [1000, 800]      # 头部类别：大量样本
tail_classes = [50, 45, 40, 35, 30, 25, 20, 15]  # 尾部类别：少量样本

long_tail_classes = head_classes + tail_classes

# 配置参数也呈现长尾特性
head_params = {
    'dispersion': [0.6, 0.65],     # 头部类别分散度较高
    'correlation': [0.4, 0.35],    # 头部类别相关性中等
    'noise': [0.08, 0.1],          # 头部类别噪声中等
    'boundary': [0.6, 0.55]        # 头部类别边界中等
}

tail_params = {
    'dispersion': [0.2, 0.25, 0.15, 0.3, 0.18, 0.22, 0.12, 0.28],  # 尾部类别分散度低
    'correlation': [0.8, 0.85, 0.9, 0.75, 0.88, 0.82, 0.92, 0.78], # 尾部类别相关性高
    'noise': [0.02, 0.01, 0.03, 0.015, 0.025, 0.018, 0.008, 0.035], # 尾部类别噪声低
    'boundary': [0.9, 0.95, 0.88, 0.92, 0.87, 0.93, 0.96, 0.85]     # 尾部类别边界锐利
}

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=long_tail_classes,
    
    dispersion=head_params['dispersion'] + tail_params['dispersion'],
    curvature=[0.3] * 10,                    # 统一曲度
    flatness=[0.5] * 10,                     # 统一扁平度
    inter_class_distance=0.7,                # 高类间距离，有助于区分
    intra_class_correlation=head_params['correlation'] + tail_params['correlation'],
    
    # 神经网络特性
    manifold_complexity=[0.5, 0.4] + [0.1] * 8,     # 头部复杂，尾部简单
    feature_sparsity=[0.2, 0.25] + [0.05] * 8,      # 头部稀疏，尾部密集
    noise_level=head_params['noise'] + tail_params['noise'],
    boundary_sharpness=head_params['boundary'] + tail_params['boundary'],
    dimensional_anisotropy=[0.4, 0.5] + [0.2] * 8   # 头部各向异性高
)
```

**实际应用场景**:
- 自然图像数据集（常见对象 vs 罕见对象）
- 电商推荐（热门商品 vs 长尾商品）
- 医疗诊断（常见疾病 vs 罕见疾病）
- 语言模型（高频词汇 vs 低频词汇）

---

### 10. 时序演化配置
**场景描述**: 模拟数据分布随时间变化的场景

```python
# 时序演化 - 模拟concept drift
generator = EmbeddingGenerator(embedding_dim=256, random_state=606)

# 3个时间段，每个时间段3个类别
time_periods = 3
classes_per_period = 3
samples_per_class = 100

# 时间演化的参数设置
evolution_classes = [samples_per_class] * (time_periods * classes_per_period)

# 分散度随时间增加（数据质量退化）
dispersion_evolution = [
    [0.3, 0.35, 0.25],  # T1: 低分散度，高质量数据
    [0.5, 0.55, 0.45],  # T2: 中等分散度，质量下降
    [0.8, 0.85, 0.75]   # T3: 高分散度，显著漂移
]

# 噪声随时间增加
noise_evolution = [
    [0.02, 0.015, 0.025],  # T1: 低噪声
    [0.08, 0.1, 0.06],     # T2: 中等噪声
    [0.2, 0.25, 0.18]      # T3: 高噪声
]

# 边界清晰度随时间降低
boundary_evolution = [
    [0.9, 0.95, 0.85],   # T1: 清晰边界
    [0.6, 0.65, 0.55],   # T2: 中等边界
    [0.3, 0.25, 0.35]    # T3: 模糊边界
]

embeddings, labels = generator.generate_clustered_embeddings(
    n_samples_per_class=evolution_classes,
    
    dispersion=sum(dispersion_evolution, []),
    curvature=[0.2] * 9,                     # 保持相对稳定的曲度
    flatness=[0.4] * 9,                      # 保持相对稳定的扁平度
    inter_class_distance=0.6,               # 中等类间距离
    intra_class_correlation=[0.6] * 9,      # 相对稳定的相关性
    
    # 神经网络特性的时序变化
    manifold_complexity=[0.2, 0.15, 0.25,   # T1: 低复杂度
                        0.4, 0.45, 0.35,    # T2: 中等复杂度
                        0.7, 0.8, 0.6],     # T3: 高复杂度
    feature_sparsity=[0.1] * 9,             # 保持稳定的稀疏性
    noise_level=sum(noise_evolution, []),
    boundary_sharpness=sum(boundary_evolution, []),
    dimensional_anisotropy=[0.3] * 9        # 保持稳定的各向异性
)
```

**实际应用场景**:
- 推荐系统用户偏好演化
- 金融市场数据分布变化
- 医疗数据随政策/技术变化
- 社交媒体话题演化

## 参数说明总结

### 基础几何参数
- **dispersion (0.0-1.0)**: 控制类内样本分散程度，0为极度聚集，1为极度分散
- **curvature (0.0-1.0)**: 控制分布的非线性程度，0为线性，1为强非线性
- **flatness (0.0-1.0)**: 控制维度压缩程度，0为球形，1为超平面
- **inter_class_distance (0.0-1.0)**: 控制类别间中心距离
- **intra_class_correlation (0.0-1.0)**: 控制类内特征相关性

### 神经网络特性参数
- **manifold_complexity (0.0-1.0)**: 模拟非线性激活函数效果
- **feature_sparsity (0.0-1.0)**: 模拟ReLU等激活函数的稀疏化效果
- **noise_level (0.0-1.0)**: 模拟训练过程中的信息损失
- **boundary_sharpness (0.0-1.0)**: 控制分类边界的清晰程度
- **dimensional_anisotropy (0.0-1.0)**: 模拟不同特征维度的重要性差异

## 使用建议

1. **噪声类别**: 使用高分散度、高稀疏性、低边界锐度
2. **稀有类别**: 使用低分散度、高相关性、高边界锐度
3. **标准类别**: 使用中等参数值，平衡各种特性
4. **层次数据**: 逐层调整参数，体现层次关系
5. **多模态数据**: 使用高流形复杂度和各向异性
6. **对抗场景**: 使用低复杂度、高边界锐度、低噪声
7. **领域适应**: 源域使用低噪声清晰参数，目标域使用高噪声模糊参数
8. **长尾分布**: 头部类别使用中等参数，尾部类别使用极端参数
9. **时序数据**: 随时间调整参数，模拟概念漂移

这些配置案例为研究人员提供了丰富的实验场景，可以根据具体研究需求选择或修改相应的配置参数。 