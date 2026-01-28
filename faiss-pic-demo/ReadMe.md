# Faiss 图像相似度搜索演示

基于 FAISS (Facebook AI Similarity Search) 的图像特征提取与相似度搜索演示项目。

## 项目简介

本项目展示了如何使用深度学习模型和 FAISS 库构建一个简单的图像相似度搜索系统。我们使用预训练的 ResNet50 模型提取图像特征，并使用 FAISS 构建高效的相似图像检索系统。

### 核心功能

- 使用 ResNet50 预训练模型提取图像特征向量（2048 维）
- 利用 FAISS 构建图像检索索引
- 实现基于 L2 距离的相似图像搜索
- 提供 Streamlit 前端界面进行可视化展示

## 依赖环境

### 环境准备

```bash
# 创建 Conda 环境
conda create -n faiss-env python=3.10
conda activate faiss-env

# 安装 PyTorch 和 torchvision
pip install torch torchvision

# 安装图像处理库
pip install pillow opencv-python

# 安装 FAISS
pip install faiss-cpu

# 安装 Streamlit（用于前端界面）
pip install streamlit
```

或者直接安装所有依赖：

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备图像数据

将要建立索引的图像文件放入指定目录，例如 `docs/images` 文件夹。

### 2. 运行快速入门示例

可以使用 [quickstart.ipynb](./quickstart.ipynb) 进行快速体验，该笔记本包含了：

- 图像特征提取过程
- FAISS 索引构建
- 相似图像检索演示

### 3. 启动 Streamlit 界面

```bash
cd d:\Project\easy-vectordb-practice\faiss-pic-demo
streamlit run streamlit_image_search.py
```

## 技术细节

### 特征提取

- 使用预训练的 ResNet50 模型（ImageNet 权重）
- 移除分类层，保留特征提取能力
- 每张图像输出 2048 维的特征向量
- 对特征向量进行 L2 归一化

### 索引策略

- 数据量 < 10,000：使用 IndexFlatL2（精确搜索）
- 数据量 ≥ 10,000：使用 IndexIVFFlat（近似搜索）
- 根据数据量动态调整聚类中心数量

### 搜索算法

- 基于 L2 距离计算图像相似度
- 返回最相似的 Top-K 个图像
- 支持自定义返回结果数量

## 文件结构

```
faiss-pic-demo/
├── ReadMe.md           # 项目说明文档
├── quickstart.ipynb    # 快速入门示例
├── streamlit_image_search.py  # Streamlit 界面
└── requirements.txt    # 项目依赖
```

## 性能说明

- 小规模数据集（< 10,000 张图片）：精确搜索，结果准确，内存占用较高
- 大规模数据集（≥ 10,000 张图片）：近似搜索，速度快，内存效率高

## 注意事项

1. 确保有足够的内存来加载模型和存储特征向量
2. 图像路径需要根据实际情况进行调整
3. GPU 可加快特征提取过程（如果可用）

## 扩展功能

- 可以替换为其他预训练模型（如 EfficientNet、Vision Transformer 等）
- 支持不同的 FAISS 索引类型以适应不同规模的数据集
- 可以添加更多元数据字段用于检索和展示